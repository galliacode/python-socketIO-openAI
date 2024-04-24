from datetime import datetime
from pydantic import BaseModel

class ChatSessionBase(BaseModel):
    user_id: int
    chat_name: str
    created_at: datetime

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSession(ChatSessionBase):
    id: str

    class Config:
        from_attributes = True

class ChatAIMessage(BaseModel):
    user_id: int
    session_id: str
    message_type: str
    message_content: str
    content_type: str
    created_at: datetime