import os
import json
from typing import Annotated, Dict
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import chromadb
from app.database.base import SessionLocal
from app.database.crud import get_user
from app.schemas.user import TokenData
from app.database.models import InvalidToken
from datetime import datetime

import boto3
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint, LLMContentHandler

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Handle request and response to / from SageMaker Endpoint
class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        print("response_json:", response_json)
        result = response_json[0]["generated_text"]
        print("generated_text:", result)
        return result
    
sagemaker_content_handler = ContentHandler()


def add_invalidated_token(db: Session, token: str, expires_at: datetime):
    db.add(InvalidToken(token=token, expires_at=expires_at))
    db.commit()

def check_token_is_invalidated(db: Session, token: str) -> bool:
    # Remove expired tokens first
    db.query(InvalidToken).filter(InvalidToken.expires_at < datetime.utcnow()).delete()
    db.commit()
    # Then check if the provided token is invalidated
    return db.query(InvalidToken).filter(InvalidToken.token == token, InvalidToken.expires_at > datetime.utcnow()).first() is not None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_vector_store():
    vector_store = chromadb.HttpClient(
        host=os.getenv("CHROMA_DB_HOST"),
        port=os.getenv("CHROMA_DB_PORT")
    )
    try:
        yield vector_store
    finally:
        del vector_store

def sget_current_user(token, db):
    if check_token_is_invalidated(db, token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=[os.getenv("ALGORITHM")])
        username: str = payload.get("sub")
        if username is None:
            print("token not available")
        token_data = TokenData(username=username)
    except JWTError:
        print("token not available")
    
    user = get_user(db, username=token_data.username)
    if user is None:
        print("token not available")
    return user

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if check_token_is_invalidated(db, token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=[os.getenv("ALGORITHM")])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_sagemaker_llm(content_handler: ContentHandler = sagemaker_content_handler):
    
    if os.getenv("ENVIRONMENT") == "dev":
        client = boto3.client(
            "sagemaker-runtime",
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        )
    else:
        client = boto3.client(
            "sagemaker-runtime",
            region_name=os.getenv("AWS_REGION"),
        )

    return SagemakerEndpoint(
        endpoint_name=os.getenv("AWS_SAGEMAKER_LLM_ENDPOINT"),
        client=client,
        model_kwargs={
            "temperature": 0.1,
            "max_new_tokens": 256,
            },
        content_handler=content_handler,
        endpoint_kwargs={
            "InferenceComponentName": os.getenv("AWS_SAGEMAKER_LLM_ENDPOINT_COMP_NAME")
        }
    )
