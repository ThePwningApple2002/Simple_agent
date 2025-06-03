import os
from fastapi import FastAPI, HTTPException
from graph import ToolCallingGraph
from models import GraphResponse, UserQuery
from tools import available_tools
from config import (
    OPENAI_API_KEY, 
    OPENAI_MODEL_NAME, 
    checkpointer
)
from utils import load_system_prompt

app = FastAPI(
    title="LLM Instance Manager API",
    description="API to get and test LLM instances from various providers with conversation history.",
)




@app.post("/invoke-graph", response_model=GraphResponse)
async def invoke_graph_endpoint(user_query: UserQuery):
    if not user_query.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if not user_query.user_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty.")
    
    PROMPT_FILE_PATH = "prompt.txt"
    
    try:
        try:
            system_prompt = load_system_prompt(PROMPT_FILE_PATH)
            print(f"System prompt loaded from: {PROMPT_FILE_PATH}")
        except Exception as e:
            print(f"Failed to load system prompt from {PROMPT_FILE_PATH}: {e}")
            system_prompt = ""
        
        graph_instance = ToolCallingGraph(
            tools_list=available_tools,
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_MODEL_NAME,
            system_prompt_template=system_prompt,
            checkpointer=checkpointer,
        )
        
        final_state = graph_instance.run(
            user_query.query, 
            user_id=user_query.user_id
        )
        
        final_answer = (
            final_state["messages"][-1].content 
            if final_state.get("messages") 
            else "No answer generated."
        )
        
        return GraphResponse(answer=final_answer, user_id=user_query.user_id)
        
    except Exception as e:
        print(f"Error during graph invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-history")
async def clear_history_endpoint(thread_id: str):
    """Clear conversation history for a user"""
    if not thread_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty.")
    
    try:
        success = checkpointer.clear_checkpoint(thread_id)
        if success:
            return {"message": f"History cleared for user: {thread_id}"}
        else:
            return {"message": f"No history found for user: {thread_id}"}
    except Exception as e:
        print(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


