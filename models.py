from typing import List
from pydantic import BaseModel


class UserQuery(BaseModel):
    query: str
    user_id: str  


class GraphResponse(BaseModel):
    answer: str
    user_id: str  