from typing import Annotated, Sequence, TypedDict, List, Any, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from checkpointer import MongoDBCheckpointer


class ToolCallingGraph:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

    def __init__(
        self,
        tools_list: List[Any],
        api_key: str,
        model_name: str,
        system_prompt_template: str = None,
        checkpointer: Optional[MongoDBCheckpointer] = None,
    ):
        self.tools_list = tools_list
        self.api_key = api_key
        self.model_name = model_name
        self.checkpointer = checkpointer
        self.tool_names_str = ", ".join([tool.name for tool in self.tools_list])

        self.tool_node = ToolNode(self.tools_list)

        default_system_prompt = (
            "You are a helpful AI assistant. Respond to the user's request. "
            "You have access to the following tools: {tool_names}. "
            "Only use the tools if necessary. If the question can be answered "
            "without tools, do so. If the user asks a greeting, just greet back."
        )
        final_system_prompt_template = (
            system_prompt_template or default_system_prompt
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", final_system_prompt_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.llm = ChatOpenAI(
            api_key=self.api_key, model=self.model_name, temperature=0
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools_list)

        self.app_graph = self._build_graph()

    def _agent_node(self, state: AgentState) -> dict:
        response = self.llm_with_tools.invoke(
            self.prompt.format_prompt(
                tool_names=self.tool_names_str, messages=state["messages"]
            )
        )
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("Tool calls detected, continuing to tools")
            return "continue_tool"
        return "end"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ToolCallingGraph.AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue_tool": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def run(self, user_input: str, user_id: str = None, config=None) -> AgentState:
        """
        Runs the graph with the given user input and returns the final state.
        Includes conversation history if checkpointer and user_id are provided.
        """
        # Load conversation history if checkpointer is available
        if self.checkpointer and user_id:
            conversation_history = self.checkpointer.load_checkpoint(user_id)
            initial_messages = conversation_history + [HumanMessage(content=user_input)]
        else:
            initial_messages = [HumanMessage(content=user_input)]

        final_state = self.app_graph.invoke(
            {"messages": initial_messages}, config=config
        )

        if self.checkpointer and user_id:
            self.checkpointer.save_checkpoint(user_id, final_state["messages"])

        return final_state

    def stream(self, user_input: str, user_id: str = None, config=None):
        """
        Streams the graph execution with the given user input.
        Includes conversation history if checkpointer and user_id are provided.
        """
        if self.checkpointer and user_id:
            conversation_history = self.checkpointer.load_checkpoint(user_id)
            initial_messages = conversation_history + [HumanMessage(content=user_input)]
        else:
            initial_messages = [HumanMessage(content=user_input)]

        final_messages = []
        for event in self.app_graph.stream(
            {"messages": initial_messages}, config=config
        ):
            final_messages = event.get("messages", final_messages)
            yield event

        # Save final state if checkpointer is available
        if self.checkpointer and user_id and final_messages:
            self.checkpointer.save_checkpoint(user_id, final_messages)

    def clear_history(self, user_id: str) -> bool:
        """Clear conversation history for a user"""
        if self.checkpointer and user_id:
            return self.checkpointer.clear_checkpoint(user_id)
        return False