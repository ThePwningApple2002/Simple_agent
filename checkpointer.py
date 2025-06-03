from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
import json
from datetime import datetime


class MongoDBCheckpointer:
    def __init__(self, connection_string: str, database_name: str, collection_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        self.collection.create_index("user_id")

    def save_checkpoint(self, user_id: str, messages: List[BaseMessage]) -> None:
        """Save conversation history for a user"""
        try:
            messages_dict = [message_to_dict(msg) for msg in messages]
            
            checkpoint_data = {
                "user_id": user_id,
                "messages": messages_dict,
                "timestamp": datetime.utcnow(),
            }
            
            self.collection.replace_one(
                {"user_id": user_id}, 
                checkpoint_data, 
                upsert=True
            )
            print(f"Checkpoint saved for user: {user_id}")
            
        except Exception as e:
            print(f"Error saving checkpoint for user {user_id}: {e}")

    def load_checkpoint(self, user_id: str) -> List[BaseMessage]:
        """Load conversation history for a user"""
        try:
            result = self.collection.find_one({"user_id": user_id})
            
            if result and "messages" in result:
                # Convert dict format back to message objects
                messages = messages_from_dict(result["messages"])
                print(f"Checkpoint loaded for user: {user_id}, {len(messages)} messages")
                return messages
            else:
                print(f"No checkpoint found for user: {user_id}")
                return []
                
        except Exception as e:
            print(f"Error loading checkpoint for user {user_id}: {e}")
            return []

    def clear_checkpoint(self, user_id: str) -> bool:
        """Clear conversation history for a user"""
        try:
            result = self.collection.delete_one({"user_id": user_id})
            success = result.deleted_count > 0
            print(f"Checkpoint cleared for user: {user_id}, Success: {success}")
            return success
            
        except Exception as e:
            print(f"Error clearing checkpoint for user {user_id}: {e}")
            return False

    def get_all_user_ids(self) -> List[str]:
        """Get all user IDs that have conversation history"""
        try:
            user_ids = self.collection.distinct("user_id")
            return user_ids
        except Exception as e:
            print(f"Error getting user IDs: {e}")
            return []