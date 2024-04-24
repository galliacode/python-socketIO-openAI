from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from app.dependencies import get_current_user, get_db, sget_current_user
from typing import Annotated
from app.schemas.user import User
from app.database import crud
from app.agent.t2s import create_query_engine, execute_query
from app.schemas.chat_session import ChatAIMessage
from app.routers import chat, chat_session, vector_store, conn_params
from app.internal import auth
from fastapi.middleware.cors import CORSMiddleware
from app.utils.js_code_plot_generator import JSCodePlotGenerator
from app.database.base import Base, engine
import socketio
import json
import pandas as pd
Base.metadata.create_all(bind=engine)

async def generate_query(token, session_id, user_query, sid):
    db_generator = get_db()
    db = next(db_generator)
    current_user = sget_current_user(token, db)
    message_info = {"user_id": current_user.id, "session_id": session_id}
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
            **message_info, message_type="user", message_content=user_query, content_type="user_query", created_at=datetime.utcnow()
        ))

    creds = crud.get_user_db_creds(db=db, user_id=current_user.id)
    if creds is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token or credentials")

    query_engine = await create_query_engine(
        query=user_query,
        dialect=creds.dialect, driver=creds.driver, host=creds.host, port=creds.port,
        username=creds.username, password=creds.password, db_name=creds.db_name, db_schema=creds.schema_name
    )
    
    response  = await query_engine.aquery(user_query)
    
    ai_text = str(response)
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="ai", message_content=ai_text, content_type="text", created_at=datetime.utcnow()
    ))
    await sio.emit('message', json.dumps({"ai_text": ai_text}), room=sid)

    sql_query = response.metadata.get("sql_query")
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="ai", message_content=sql_query, content_type="sql", created_at=datetime.utcnow()
    ))
    await sio.emit('message', json.dumps({"sql_query": sql_query}), room=sid)
    
    result = response.metadata.get("result")
    df = pd.DataFrame()
    html_plot_js = ""

    if result:
        df = pd.DataFrame(result)
        html_plot_js = await JSCodePlotGenerator(sql_query=sql_query, data=df) \
            .generate_plot(model_name="gpt-4")
        
    dataframe = df.to_json(orient="split")
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="ai", message_content=dataframe, content_type="html", created_at=datetime.utcnow()
    ))
    await sio.emit('message', json.dumps({"dataframe": dataframe}), room=sid)
    
    plot_code = html_plot_js
    crud.add_message_to_chat_session_history(db=db, chat_history=ChatAIMessage(
        **message_info, message_type="ai", message_content=plot_code, content_type="javascript", created_at=datetime.utcnow()
    ))
    await sio.emit('message', json.dumps({"plot_code": plot_code}), room=sid)
    
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
socketio_app = socketio.ASGIApp(sio)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_session.router, prefix="/chat_session", tags=["chat_session"])
app.include_router(chat.router, tags=["chat"])
app.include_router(vector_store.router, prefix="/vector_store", tags=["vector_store"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(conn_params.router, prefix="/conn-params", tags=["connection parameters"])

@app.get("/", tags=["root"])
async def root():
    return {"app": "Analitiq"}

@sio.event
async def connect(sid, environ):
    print('connect ', sid)
    await sio.emit('message', 'Connected to the server!', room=sid)  # Emit a message to the connected client

@sio.event
async def disconnect(sid):
    print('disconnect ', sid)

@sio.event
async def message(sid, data):
    json_data = json.loads(data)
    user_query = json_data.get('user_query')
    session_id = json_data.get('sessionId')
    token = json_data.get('token')
    
    await generate_query(token, session_id, user_query, sid)

app.mount('/socket.io', socketio_app)