from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def generate_password_reset_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=15)  # Token expiration time
    to_encode = {"exp": expire, "sub": email}
    encoded_jwt = jwt.encode(to_encode, os.getenv("SECRET_KEY"), algorithm="HS256")
    return encoded_jwt

def verify_password_reset_token(token: str) -> str:
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        return payload.get("sub")  # 'sub' is the subject (email) we encoded in the token
    except JWTError:
        return None
    
def reset_password(new_password: str) -> str:
    """
    Hash a new password using the specified hashing algorithm.
    
    Args:
        new_password (str): The new password to be hashed.

    Returns:
        str: The hashed password.
    """
    return pwd_context.hash(new_password)