from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os

ALGORITHM = os.getenv("ALGORITHM")
SECRET_KEY = os.getenv("SECRET_KEY")

def encode_data(data: str) -> str:
    encoded_jwt = jwt.encode({data: data}, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_data(encoded_jwt: str) -> dict:
    try:
        decoded_data = jwt.decode(encoded_jwt, SECRET_KEY, algorithms=[ALGORITHM])

        original_string = decoded_data.get(next(iter(decoded_data)))
        return original_string
    except JWTError:
        return None
