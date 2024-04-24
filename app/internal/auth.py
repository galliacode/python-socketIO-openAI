import os
from typing import Annotated, Union
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import APIRouter, Depends, Form, HTTPException, status, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import EmailStr

from app.schemas.user import Token, User, UserCreate
from app.database import crud, models
from app.dependencies import get_db, get_user, oauth2_scheme, add_invalidated_token, check_token_is_invalidated
from app.security import pwd_context, generate_password_reset_token, verify_password_reset_token, reset_password
from app.emails.email_utils import send_password_reset_email

router = APIRouter()

def verify_password(password: str, hashed_password: str):
    return pwd_context.verify(password, hashed_password)

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("SECRET_KEY"), algorithm=os.getenv("ALGORITHM"))
    return encoded_jwt

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

@router.post("/signup", response_model=User)
def signup(username: str = Form(), email: str = Form(), password: str = Form(), db: Session = Depends(get_db)):
    user = UserCreate(username=username, email=email, password=password)
    db_user = crud.get_user(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exist")
    return crud.create_user(db=db, user=user)

@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    time_expires = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
    access_token_expires = timedelta(minutes=time_expires)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/get_current_user")
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if check_token_is_invalidated(db, token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=[os.getenv("ALGORITHM")])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return username

@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    expires_at = datetime.utcnow() + timedelta(days=30)
    add_invalidated_token(db, token, expires_at)
    return {"message": "Logged out"}


@router.post("/request-password-reset", response_model=dict)
async def request_password_reset(email: EmailStr, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    reset_token = generate_password_reset_token(email)
    send_password_reset_email(background_tasks, email, reset_token)
    return {"message": "Password reset email sent"}

@router.post("/reset-password", response_model=dict)
async def reset_password_route(
    request: Request,
    token: Annotated[str, Form()],
    password: Annotated[str, Form()],
    db: Session = Depends(get_db),
):
    if check_token_is_invalidated(db=db, token=token):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user or token")
    
    email = verify_password_reset_token(token)
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user or token")
    
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user or token")

    hashed_password = reset_password(password)
    user.hashed_password = hashed_password
    db.commit()

    # this token should be one-time use
    add_invalidated_token(db=db, token=token, expires_at=datetime.utcnow() + timedelta(days=30))
    
    return {"message": "Password reset successfully"}