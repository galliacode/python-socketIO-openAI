from datetime import datetime
from pydantic import BaseModel

class ChatHistoryBase(BaseModel):
    user_id: int
    session_id: str
    message_type: str
    message_content: str
    content_type: str
    created_at: datetime

class ChatHistoryCreate(ChatHistoryBase):
    pass

class ChatHistory(ChatHistoryBase):
    id: int

    class Config:
        from_attributes = True