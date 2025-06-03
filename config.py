import os
from dotenv import load_dotenv

from checkpointer import MongoDBCheckpointer

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = "SomniumDB"
MONGO_COLLECTION = "AIChat"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o"


checkpointer = MongoDBCheckpointer(
    connection_string=MONGO_URI,
    database_name=MONGO_DB,
    collection_name=MONGO_COLLECTION,
)