from pydantic import BaseModel
from app.config import SCHEMA_NAME_DEFAULT, TABLE_NAME_DEFAULT as TABLE_NAME
from datetime import datetime

class DbConnTypes(BaseModel):
    __tablename__ = TABLE_NAME.DB_CONN_TYPES
    
    id: int
    conn_type: str
    conn_dialect: str
    conn_driver: str
    created_at: datetime
    updated_at: datetime

    class Config:
        schema_name = SCHEMA_NAME_DEFAULT
        from_attributes = True