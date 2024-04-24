from datetime import datetime
from typing import Annotated
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, WebSocket, Form, HTTPException, status
import json
from app.database import crud
from app.schemas.user import User
from app.schemas.chat_session import ChatAIMessage
from app.dependencies import get_current_user, get_db, sget_current_user
from app.agent.t2s import create_query_engine, execute_query
from app.schemas.chat_feedback import ChatFeedback, ChatFeedbackCreate, UserRatingEnum
from app.utils.js_code_plot_generator import JSCodePlotGenerator
import pandas as pd

router = APIRouter()

@router.post("/{session_id}/query")
async def execute_user_query(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    user_query: Annotated[str, Form(example="How much sales we had overall?")],
):
    """
    This endpoint is responsible for interacting with the LLM. It takes the natural language
    query from the user, and construct the prompt, to perform a RAG task, generate the SQL,
    execute the SQL over the database, gather the results, and generate a plot regarding user's
    query. It implements the main logic to interact with the LLM.

    param session_id: chat session ID (path parameter)
    param current_user: current logged in user (dependencies)  
    param db: Database session (dependencies)  
    param user_query: natural language query provided by the user   
    
    Sample Response Body:  

    {
        "id": "41f4793a-bd3d-4368-9243-e883b20bc74e",
        "user_id": 1,
        "message_type": "ai",
        "sql_query": "SELECT SUM(pricepaid) FROM sales",
        "dataframe": {
            "sum": {
            "0": 110765431
            }
        },
        "plot_code": "<div id=\"plot\"></div>\n<script src=\"https://cdn.plot.ly/plotly-2.20.0.min.js\" charset=\"utf-8\"></script>\n<script>\nvar queried_data = {\"schema\":{\"fields\":[{\"name\":\"sum\",\"type\":\"string\"}],\"pandas_version\":\"1.4.0\"},\"data\":[{\"sum\":110765431.0}]}\n\n\nvar data = [{\n  type: \"indicator\",\n  mode: \"number\",\n  value: queried_data.data[0].sum,\n  chat_name: { text: \"Total Sales\" },\n}];\n\nvar layout = {\n  width: 500,\n  height: 400,\n  margin: { t: 0, b: 0 }\n};\n\nPlotly.newPlot('plot', data, layout);\n</script>"
    }
    """

    message_info = {"user_id": current_user.id, "session_id": session_id}
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="user", message_content=user_query, content_type="user_query", created_at=datetime.utcnow()
    ))

    creds = crud.get_user_db_creds(db=db, user_id=current_user.id)
    if creds is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token or credentials")

    print("create query engine")
    query_engine = await create_query_engine(
        query=user_query,
        dialect=creds.dialect, driver=creds.driver, host=creds.host, port=creds.port,
        username=creds.username, password=creds.password, db_name=creds.db_name, db_schema=creds.schema_name
    )
    print("should await for query execution")
    result = await execute_query(query_engine=query_engine, query=user_query)

    ai_text = result.get("text")
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="ai", message_content=ai_text, content_type="text", created_at=datetime.utcnow()
    ))

    sql_query = result.get("sql")
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="ai", message_content=sql_query, content_type="sql", created_at=datetime.utcnow()
    ))

    dataframe = result.get("result")
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="ai", message_content=dataframe, content_type="html", created_at=datetime.utcnow()
    ))

    plot_code = result.get("plot_code")
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="ai", message_content=plot_code, content_type="javascript", created_at=datetime.utcnow()
    ))

    return {
        "session_id": session_id,
        "user_id": current_user.id,
        "message_type": "ai",
        "text": ai_text,
        "sql_query": sql_query,
        "dataframe": dataframe,
        "plot_code": plot_code,
    }

# @router.websocket("/{session_id}/ws")
# async def websocket_endpoint(
#     websocket: WebSocket,
#     session_id: str,
#     db: Annotated[Session, Depends(get_db)]
# ):
#     """
#     A Web Socket implementation of the above `query` route.
#     """

#     await websocket.accept()

#     data = await websocket.receive_json()
#     token = data['token']
#     current_user = sget_current_user(token, db)
#     message_info = {"user_id": current_user.id, "session_id": session_id}
    
#     while True:
#         data = await websocket.receive_json()
#         user_query = data['user_query']
#         crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
#             **message_info, message_type="user", message_content=user_query, content_type="user_query", created_at=datetime.utcnow()
#         ))
        
#         creds = crud.get_user_db_creds(db=db, user_id=current_user.id)
#         if creds is None:
#             raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token or credentials")

#         print("create query engine")

#         query_engine = await create_query_engine(
#             query=user_query,
#             dialect=creds.dialect, driver=creds.driver, host=creds.host, port=creds.port,
#             username=creds.username, password=creds.password, db_name=creds.db_name, db_schema=creds.schema_name
#         )
#         print("should await for query execution")

#         response  = await query_engine.aquery(user_query)
        
#         ai_text = str(response)
#         crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
#             **message_info, message_type="ai", message_content=ai_text, content_type="text", created_at=datetime.utcnow()
#         ))
#         await websocket.send_text(json.dumps({"ai_text":ai_text}))

#         sql_query = response.metadata.get("sql_query")
#         crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
#             **message_info, message_type="ai", message_content=sql_query, content_type="sql", created_at=datetime.utcnow()
#         ))
#         await websocket.send_text(json.dumps({"sql_query": sql_query}))

#         result = response.metadata.get("result")
#         df = pd.DataFrame()
#         html_plot_js = ""

#         if result:
#             df = pd.DataFrame(result)
#             html_plot_js = await JSCodePlotGenerator(sql_query=sql_query, data=df) \
#                 .generate_plot(model_name="gpt-4")
            
#         dataframe = df.to_json(orient="split")
#         crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
#             **message_info, message_type="ai", message_content=dataframe, content_type="html", created_at=datetime.utcnow()
#         ))
#         await websocket.send_text(json.dumps({"dataframe": dataframe}))

#         plot_code = html_plot_js
#         crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
#             **message_info, message_type="ai", message_content=plot_code, content_type="javascript", created_at=datetime.utcnow()
#         ))
#         await websocket.send_text(json.dumps({"plot_code": plot_code}))

@router.post("/{history_id}/feedback", response_model=ChatFeedback)
async def create_chat_feedback(
    db: Annotated[Session, Depends(get_db)],
    history_id: int,
    session_id: Annotated[str, Form()],
    user_prompt: Annotated[str, Form()],
    ai_response: Annotated[str, Form()],
    user_rating: Annotated[UserRatingEnum, Form(examples=['good', 'bad'])],
    user_comment: Annotated[str, Form()]
):

    """
    param history_id: chat history ID (path parameter)  
    param db: Database session (dependencies)
    """

    chat_feedback = ChatFeedbackCreate(
        history_id = history_id,
        session_id = session_id,
        user_prompt=user_prompt,
        ai_response=ai_response,
        user_rating=user_rating,
        user_comment=user_comment
    )

    return crud.create_feedback(db=db, chat_feedback=chat_feedback)