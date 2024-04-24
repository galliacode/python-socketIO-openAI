from uuid import uuid4
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship
from app.database.base import Base
from app.config import SCHEMA_NAME_DEFAULT, TABLE_NAME_DEFAULT as TABLE_NAME
import datetime

class User(Base):
    __tablename__ = TABLE_NAME.DB_USER
    __table_args__ = {"schema": SCHEMA_NAME_DEFAULT}

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, index=True)
    
    chat_session = relationship("ChatSession", back_populates=TABLE_NAME.DB_USER)

class ChatSession(Base):
    __tablename__ = TABLE_NAME.DB_CHAT_SESSION
    __table_args__ = {"schema": SCHEMA_NAME_DEFAULT}

    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(Integer, ForeignKey(f"{SCHEMA_NAME_DEFAULT}.{TABLE_NAME.DB_USER}.id"))
    chat_name = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    user = relationship("User", back_populates=TABLE_NAME.DB_CHAT_SESSION)
    chat_history = relationship("ChatHistory", back_populates=TABLE_NAME.DB_CHAT_SESSION)
    chat_feedback = relationship("ChatFeedback", back_populates=TABLE_NAME.DB_CHAT_SESSION)

class ChatHistory(Base):
    __tablename__ = TABLE_NAME.DB_CHAT_HISTORY
    __table_args__ = {"schema": SCHEMA_NAME_DEFAULT}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey(f"{SCHEMA_NAME_DEFAULT}.{TABLE_NAME.DB_USER}.id"))
    session_id = Column(String(36), ForeignKey(f"{SCHEMA_NAME_DEFAULT}.{TABLE_NAME.DB_CHAT_SESSION}.id"))
    message_type = Column(String)
    message_content = Column(String)
    content_type = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    chat_session = relationship("ChatSession", back_populates=TABLE_NAME.DB_CHAT_HISTORY)
    chat_feedback= relationship("ChatFeedback", back_populates=TABLE_NAME.DB_CHAT_HISTORY)

class InvalidToken(Base):
    __tablename__ = TABLE_NAME.DB_INVALID_TOKENS
    __table_args__ = {"schema": SCHEMA_NAME_DEFAULT}

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

class UserDBCreds(Base):
    __tablename__ = TABLE_NAME.DB_USER_DB_CONNECTIONS
    __table_args__ = {"schema": SCHEMA_NAME_DEFAULT}

    id = Column(Integer, primary_key=True, index=True)
    conn_uuid = Column(String(36), index=True, default=lambda: str(uuid4()))
    user_id = Column(Integer, ForeignKey(f"{SCHEMA_NAME_DEFAULT}.{TABLE_NAME.DB_USER}.id"), index=True)
    dialect = Column(String)
    driver = Column(String)
    host = Column(String)
    port = Column(Integer)
    db_name = Column(String)
    schema_name = Column(String)
    username = Column(String)
    password = Column(String)
    conn_name = Column(String)
    conn_type_id = Column(Integer, ForeignKey(f"{SCHEMA_NAME_DEFAULT}.{TABLE_NAME.DB_CONN_TYPES}.id"), index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, index=True)
    deleted_at = Column(DateTime, index=True)

    db_conn_type= relationship("DbConnTypes", back_populates=TABLE_NAME.DB_USER_DB_CONNECTIONS)

class DbConnTypes(Base):
    __tablename__ = TABLE_NAME.DB_CONN_TYPES
    __table_args__ = {"schema": SCHEMA_NAME_DEFAULT}

    id = Column(Integer, primary_key=True, index=True)
    conn_type = Column(String, unique=True, index=True)
    conn_dialect = Column(String)
    conn_driver = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, index=True)

    user_db_conn= relationship("UserDBCreds", back_populates=TABLE_NAME.DB_CONN_TYPES)

class ChatFeedback(Base):
    __tablename__ = TABLE_NAME.DB_CHAT_FEEDBACK
    __table_args__ = {"schema": SCHEMA_NAME_DEFAULT}

    id = Column(Integer, primary_key=True, index=True)
    history_id = Column(Integer, ForeignKey(f"{SCHEMA_NAME_DEFAULT}.{TABLE_NAME.DB_CHAT_HISTORY}.id"))

    session_id = Column(String(36), ForeignKey(f"{SCHEMA_NAME_DEFAULT}.{TABLE_NAME.DB_CHAT_SESSION}.id"))

    user_prompt = Column(String)
    ai_response = Column(String)
    user_rating = Column(String)
    user_comment = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    chat_session = relationship("ChatSession", back_populates=TABLE_NAME.DB_CHAT_FEEDBACK)
    chat_history = relationship("ChatHistory", back_populates=TABLE_NAME.DB_CHAT_FEEDBACK)
