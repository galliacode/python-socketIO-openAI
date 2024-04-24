import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Annotated
from fastapi import APIRouter, Depends, Body, Form
from app.schemas.user import User
from app.schemas.chat_session import ChatSessionCreate, ChatAIMessage
from app.database import crud
from app.dependencies import get_current_user, get_db

router = APIRouter()


@router.get("/sessions")
def get_user_chat_sessions(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """
    Return all the chat sessions for a given user.

    param current_user: current logged in user (dependencies)
    param db: Database session (dependencies)  

    Sample Response Body:
    [  
        {  
            "user_id": 1,  
            "chat_name": "query-customer",
            "id": "6e2fbd79-f2b4-46b4-b34d-de29c567010f",
            "created_at": "2023-12-05T02:34:04.740581"  
        },  
        {  
            "user_id": 1,  
            "chat_name": "query-sales-stats",
            "id": "740ee8f7-649a-4db1-9c69-21034a5f5286",
            "created_at": "2023-12-05T23:21:34.564324"  
        }  
    ]  
    """
    return crud.get_chat_sessions_by_user(db=db, user_id=current_user.id)

@router.post("/create")
async def create_chat_session(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    chat_name: str,
):
    """
    Create a new chat session for a given user.

    param current_user: current logged in user (dependencies)
    param db: Database session (dependencies)  
    :param chat_name: name for the chat session

    Sample Response Body:  
    {  
        "user_id": 1,    
        "chat_name": "query-customers",
        "id": "d828b5b5-90b0-41b5-aca6-44942f471bdc",
        "created_at": "2023-12-05T23:48:33.423816"  
    }
    """
    chat_session = ChatSessionCreate(
        user_id=current_user.id,
        chat_name=chat_name,
        created_at=datetime.utcnow(),
    )
    return crud.create_chat_session(db=db, session=chat_session)

@router.get("/{id}")
async def get_chat_session(
    id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """
    Get a chat session for a user by chat session ID.

    param id: chat session ID (path parameter)
    param current_user: current logged in user (dependencies)  
    param db: Database session (dependencies)  

    Sample Response Body:  
    {  
        "user_id": 1,  
        "chat_name": "query-sales-stats",
        "id": "740ee8f7-649a-4db1-9c69-21034a5f5286",
        "created_at": "2023-12-05T23:21:34.564324"  
    }
    """
    return crud.get_chat_session(db=db, user_id=current_user.id, id=id)

@router.get("/{id}/history")
async def get_chat_session_history(
    id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """
    Get the chat history for a given chat session.

    param id: chat session ID (path parameter)
    param current_user: current logged in user (dependencies)  
    param db: Database session (dependencies)  
    
    Sample Response Body:  
    {  
        "history": [  
            {  
                "type": "user",  
                "content": "Who are my top 10 customers?"  
            },  
            {  
                "type": "ai",  
                "content": "SELECT buyerid, COUNT(*) as total_purchases \nFROM sales \nGROUP BY buyerid \nORDER BY total_purchases DESC \nLIMIT 10;"  
            },  
            {  
                "type": "user",  
                "content": "Who are my top 10 customers?"  
            },  
            {  
                "type": "ai",  
                "content": "SELECT buyerid, COUNT(*) as total_purchases \nFROM sales \nGROUP BY buyerid \nORDER BY total_purchases DESC \nLIMIT 10;"  
            },  
            {  
                "type": "ai",  
                "content": pd.DataFrame
            },  
            {  
                "type": "ai",  
                "content": "<div id=\"plot\"></div>\n<script src=\"https://cdn.plot.ly/plotly-2.20.0.min.js\" charset=\"utf-8\"></script>\n<script>\nvar queried_data = {\"schema\":{\"fields\":[{\"name\":\"buyerid\",\"type\":\"integer\"},{\"name\":\"total_purchases\",\"type\":\"integer\"}],\"pandas_version\":\"1.4.0\"},\"data\":[{\"buyerid\":3451,\"total_purchases\":32},{\"buyerid\":1325,\"total_purchases\":28},{\"buyerid\":3140,\"total_purchases\":27},{\"buyerid\":11497,\"total_purchases\":26},{\"buyerid\":4842,\"total_purchases\":26},{\"buyerid\":8933,\"total_purchases\":26},{\"buyerid\":2914,\"total_purchases\":26},{\"buyerid\":3456,\"total_purchases\":26},{\"buyerid\":11774,\"total_purchases\":26},{\"buyerid\":3501,\"total_purchases\":26}]}\n\n\nvar data = [\n  {\n    x: queried_data.data.map(function(item) { return item.buyerid; }),\n    y: queried_data.data.map(function(item) { return item.total_purchases; }),\n    type: 'bar'\n  }\n];\n\nvar layout = {\n  chat_name: 'Top 10 Buyers by Total Purchases',\n  xaxis: {\n    chat_name: 'Buyer ID',\n  },\n  yaxis: {\n    chat_name: 'Total Purchases',\n  }\n};\n\nPlotly.newPlot('plot', data, layout);\n</script>"
            }  
        ]  
    }
    """
    chat_history = crud.get_chat_history(db=db, user_id=current_user.id, session_id=id)
    
    history = [
        {
            "history_id": row.id,
            "type": row.message_type,
            "content": row.message_content,
            "content_type": row.content_type,
        }
        for row in chat_history
    ]

    return {"history": history}

@router.post("/{id}/add-message")
async def add_message_to_chat_session_history(
    id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    message_type: Annotated[str, Body()],
    message_content: Annotated[str, Body()],
    content_type: Annotated[str, Body()],
):
    """
    Add a single message to the history of a given chat session. (Not used by front-end yet)

    This logic is currently implemented in the backend.
    """

    chat_history = ChatAIMessage(
        user_id=current_user.id,
        session_id=id,
        message_type=message_type,
        message_content=message_content,
        content_type=content_type,
        created_at=datetime.utcnow(),
    )
    return crud.add_message_to_chat_session_history(
        db=db,
        chat_history=chat_history
    )

@router.patch("/{id}/")
async def update_chat_session_info(
    id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    chat_name: Annotated[str, Form()],
) -> ChatSessionCreate:
    """
    Update a chat session, such as rename the chat session name. (Not used)
    """
    raise NotImplementedError

@router.delete("/{id}/delete")
async def delete_chat_session_and_message_history(
    id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)]
):
    """
    Delete a given chat session, and all the chat history related to the given chat session.

    param id: chat session ID (path parameter)
    param current_user: current logged in user (dependencies)  
    param db: Database session (dependencies)  
    
    Sample Response Body:  
    {  
        "records": {  
            "chat_name": "query-customers",
            "user_id": 1,  
            "id": "d828b5b5-90b0-41b5-aca6-44942f471bdc",
            "created_at": "2023-12-05T23:48:33.423816",  
            "user": {},  
            "history": []  
        }  
    }  
    """
    return crud.delete_chat_session(db=db, user_id=current_user.id, id=id)