from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session

from app.schemas.user import User, UserCreate
from app.schemas.chat_session import ChatSession, ChatSessionCreate, ChatAIMessage
from app.schemas.chat_history import ChatHistory
from app.schemas.user_db_creds import UserDBCreds, UserDBCredsCreate
from app.schemas.db_conn_types import DbConnTypes
from app.schemas.chat_feedback import ChatFeedback, ChatFeedbackCreate
from app.database import models
from app.security import pwd_context

def get_user(db: Session, username: str) -> User:
    return db.query(models.User) \
        .filter(models.User.username == username) \
        .first()

def create_user(db: Session, user: UserCreate) -> User:
    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(
        username=user.username, 
        email=user.email, 
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_chat_sessions_by_user(db: Session, user_id: int) -> ChatSession:
    return db.query(models.ChatSession) \
        .filter(models.ChatSession.user_id == user_id) \
        .all()

def create_chat_session(db: Session, session: ChatSessionCreate) -> ChatSession:
    db_chat_session = models.ChatSession(
        user_id=session.user_id,
        chat_name=session.chat_name,
        created_at=session.created_at
    )
    db.add(db_chat_session)
    db.commit()
    db.refresh(db_chat_session)
    return db_chat_session

def get_chat_session(db: Session, user_id: int, id: str) -> ChatSession:
    return db.query(models.ChatSession) \
        .filter(models.ChatSession.user_id == user_id) \
        .filter(models.ChatSession.id == id) \
        .first()

def get_chat_history(db: Session, user_id: int, session_id: str) -> List[ChatHistory]:
    return db.query(models.ChatHistory) \
        .filter(models.ChatHistory.user_id == user_id) \
        .filter(models.ChatHistory.session_id == session_id) \
        .order_by(models.ChatHistory.created_at) \
        .all()

def add_message_to_chat_session_history(db: Session, chat_history: ChatAIMessage) -> ChatHistory:
    db_chat_message = models.ChatHistory(
        user_id = chat_history.user_id,
        session_id = chat_history.session_id,
        message_type = chat_history.message_type,
        message_content = chat_history.message_content,
        content_type=chat_history.content_type,
        created_at = chat_history.created_at,
    )

    db.add(db_chat_message)
    db.commit()
    db.refresh(db_chat_message)
    return db_chat_message

def delete_chat_session(db: Session, user_id: int, id: str) -> Dict[str, Any]:
    # delete chat session
    record_to_delete = db.query(models.ChatSession) \
        .filter(models.ChatSession.user_id == user_id) \
        .filter(models.ChatSession.id == id) \
        .first()
    # delete chat session message history
    session_history_to_delete = db.query(models.ChatHistory) \
        .filter(models.ChatHistory.user_id == user_id) \
        .filter(models.ChatHistory.session_id == id)
    
    db.delete(record_to_delete)
    session_history_to_delete.delete()
    db.commit()
    
    return {"records": record_to_delete}

def get_user_db_creds(db: Session, user_id: int, all_records: Optional[bool] = False, id: Optional[int] = 0) -> Union[UserDBCreds, list[UserDBCreds]]:
    query = db.query(models.UserDBCreds).filter(models.UserDBCreds.user_id == user_id)

    if all_records:
        return query.all()
    if id:
        return query.filter(models.UserDBCreds.id == id).first()
    else:
        return query.first()

def set_user_db_creds(db: Session, creds: UserDBCredsCreate) -> UserDBCreds:
    user_creds: models.UserDBCreds = db.query(models.UserDBCreds).filter(models.UserDBCreds.user_id == creds.user_id, models.UserDBCreds.conn_uuid == creds.conn_uuid).first()
    if not user_creds:
        new_creds = models.UserDBCreds(
            **creds.model_dump(),
        )

        db.add(new_creds)
        db.commit()
        db.refresh(new_creds)

        return new_creds
    
    user_creds.dialect = creds.dialect
    user_creds.host    = creds.host
    user_creds.port    = creds.port
    user_creds.username = creds.username
    user_creds.password = creds.password
    user_creds.conn_name = creds.conn_name
    user_creds.db_name = creds.db_name
    user_creds.driver = creds.driver
    user_creds.schema_name = creds.schema_name
    #user_creds.conn_uuid = creds.conn_uuid TODO this should not behere, we should not be updating UUIDs
    user_creds.conn_type_id = creds.conn_type_id

    db.commit()
    db.refresh(user_creds)

    return user_creds

def get_db_conn_types(db: Session) -> list[DbConnTypes]:
    return db.query(models.DbConnTypes) \
        .all()

def create_feedback(db: Session, chat_feedback: ChatFeedbackCreate) -> ChatFeedback:
    db_chat_feedback = models.ChatFeedback(
        session_id = chat_feedback.session_id,
        history_id = chat_feedback.history_id,
        user_prompt = chat_feedback.user_prompt,
        ai_response = chat_feedback.ai_response,
        user_rating = chat_feedback.user_rating,
        user_comment = chat_feedback.user_comment
    )

    db.add(db_chat_feedback)
    db.commit()
    db.refresh(db_chat_feedback)
    return db_chat_feedback