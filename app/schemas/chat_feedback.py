from datetime import datetime
from pydantic import BaseModel
from enum import Enum

class UserRatingEnum(str, Enum):
    good = "good"
    bad = "bad"

class ChatFeedbackBase(BaseModel):
    session_id: str
    history_id: int
    user_prompt: str
    ai_response: str
    user_rating: UserRatingEnum
    user_comment: str

class ChatFeedbackCreate(ChatFeedbackBase):
    pass

class ChatFeedback(ChatFeedbackBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True