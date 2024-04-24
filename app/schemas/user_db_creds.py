from typing import Union
from pydantic import BaseModel

class UserDBCredsBase(BaseModel):
    """User basic information"""
    user_id: int
    dialect: str
    host: str
    port: int
    db_name: str
    conn_name: str
    driver: Union[str, None] = None
    schema_name: str
    conn_uuid: Union[str, None] = None
    conn_type_id: int

class UserDBCredsCreate(UserDBCredsBase):
    username: str
    password: str

class UserDBCreds(UserDBCredsBase):
    id: int
    username: str
    password: str

    class Config:
        from_attributes = True

class UserDBCredsOut(UserDBCredsBase):
    pass