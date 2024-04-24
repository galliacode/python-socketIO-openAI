from typing import Annotated, Union, Optional
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, Form, HTTPException
from app.utils import db_utils
from sqlalchemy import create_engine, exc

from app.database import crud
from app.schemas.user import User
from app.schemas.user_db_creds import UserDBCreds, UserDBCredsCreate, UserDBCredsOut
from app.schemas.db_conn_types import DbConnTypes
from app.dependencies import get_current_user, get_db
from app.utils.encryptor import encode_data

import os
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

router = APIRouter()

@router.get("/user_db_creds", response_model=Optional[UserDBCreds])
async def get_user_db_creds(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """
    Retrieve user's database connection credentials
    """
    return crud.get_user_db_creds(db=db, user_id=current_user.id)

@router.get("/user_db_creds/all", response_model=list[UserDBCreds])
async def get_all_user_db_creds(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return crud.get_user_db_creds(db=db, user_id=current_user.id, all_records=True)

@router.get("/user_db_creds/{id}", response_model=Optional[UserDBCreds])
async def get_user_db_creds_by_id(
    id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    return crud.get_user_db_creds(db=db, user_id=current_user.id, all_records=False, id=id)

@router.post("/user_db_creds", response_model=UserDBCredsOut)
async def cache_user_db_creds(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    dialect: Annotated[str, Form(examples=["redshift", "postgres"])],
    host: Annotated[str, Form(examples=["localhost", "default-workgroup.000000000000.eu-central-1.redshift-serverless.amazonaws.com"])],
    port: Annotated[int, Form(ge=1, le=65535, examples=[5432, 5439])],
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    db_name: Annotated[str, Form(examples=["dev", "sales_db", "synthetic-analytics"])],
    conn_name: Annotated[str, Form(examples=["My_Connection", "Production", "Test Database"])],
    driver: Annotated[Union[str, None], Form(examples=["psycopg2"])] = None,
    schema_name: Annotated[str, Form()] = None,
    conn_uuid: Optional[str] = Form(default=None),

    conn_type_id: Annotated[int, Form(ge=1, examples=[1])] = None,
):
    """
    Set / Update user's database connection credentials.
    
    param current_user: current logged in user (dependencies)  
    param db: Database session (dependencies)  
    param dialect: database dialect  
    param host: database host name  
    param port: database port  
    param username: username for logging into the database  
    param password: password for logging into the database  
    param db_name: the name of the database to connect to
    param conn_name: the name of the connection
    param schema_name: the schema in the database that should be used
    param driver: the driver use to connect to the database  
    param conn_type_id: db_conn_type id

    Sample response body
    {
        "user_id": 1,
        "dialect": "postgresql",
        "host": "analitiq-dev.cw1g0qte5un7.eu-central-1.rds.amazonaws.com",
        "port": 5432,
        "conn_name": "my_connection",
        "driver": null,
        "db_name": "dev",
        "schema_name": "public",
        "conn_uuid": "173c0e8f-5ca7-4aa1-a737-e8af347e714e",
        "conn_type_id": 1
    }
    """
    creds = UserDBCredsCreate(
        user_id=current_user.id,
        dialect=dialect,
        host=host,
        port=port,
        conn_name=conn_name,
        driver=driver,
        username=username,
        password=encode_data(password),
        db_name=db_name,
        schema_name=schema_name,
        conn_uuid=conn_uuid,
        conn_type_id=conn_type_id,
    )

    return crud.set_user_db_creds(db=db, creds=creds)

@router.get("/db_conn_types", response_model=list[DbConnTypes])
async def get_db_conn_types(
    db: Session = Depends(get_db)
):
    return crud.get_db_conn_types(db=db)

@router.get("/check_db_connection")
async def check_db_connection(
    dialect: str,
    host: str,
    port: int,
    username: str,
    password: str,
    db_name: str,
    driver: Optional[str] = None,
):
    try:
        engine = db_utils.create_db_engine(dialect, driver, host, port, username, password, db_name)
        schema_names = db_utils.get_schema_names(engine)

        return {"schemas": schema_names}
    except exc.SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")
