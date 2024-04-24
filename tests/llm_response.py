import os
from langchain.chat_models.openai import ChatOpenAI
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.prompts import PromptTemplate, PromptType
from fastapi import FastAPI

from sqlalchemy import create_engine, inspect
from sqlalchemy import exc
from sqlalchemy.engine import Engine
from typing import List, Optional
import sqlalchemy as sa

import socketio
import asyncio
import aiohttp

from llama_index import (
    SQLDatabase,
    VectorStoreIndex,
    LLMPredictor,
    ServiceContext,
)

from llama_index.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)

from typing import Union, Any, Dict, Tuple
from IPython.display import Markdown, display

TEXT_TO_SQL_PROMPT = (
    "Given an input question, first create a syntactically correct {dialect} "
    "query to run, then look at the results of the query and return the answer. "
    "You can order the results by a relevant column to return the most "
    "interesting examples in the database.\n\n"
    "Never query for all the columns from a specific table, only ask for a "
    "few relevant columns given the question.\n\n"
    "Pay attention to use only the column names that you can see in the schema "
    "description. "
    "Be careful to not query for columns that do not exist. "
    "Pay attention to which column is in which table. "
    "Also, qualify column names with the table name when needed. "
    "And, DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run\n"
    "SQLResult: Result of the SQLQuery\n"
    "Answer: Final answer here\n\n"
    "Only use tables listed below.\n"
    "{schema}\n\n"
    "Question: {query_str}\n"
    "SQLQuery: "
)

def create_db_engine(dialect: str, driver: str, host: str, port: int, username: str, password: str,  db_name: str, db_schema: str) -> Engine:
    return sa.create_engine(f"{dialect}+{driver}://{username}:{password}@{host}:{port}/{db_name}", connect_args={'options': '-csearch_path={}'.format(db_schema)})

def get_schema_names(engine: Engine) -> List[str]:
    try:
        insp = sa.inspect(engine)
        return insp.get_schema_names()
    except exc.SQLAlchemyError as e:
        raise e


def construct_object_index(
        dialect: str,
        driver: str,
        host: str,
        port: int,
        username: str,
        password: str,
        db_name: str,
        db_schema: str,
        include_tables: Tuple[str, None] = None
) -> ObjectIndex:

    print("constructing object index from table schema")
    sql_database = SQLDatabase(create_db_engine(dialect, driver, host, port, username, password, db_name, db_schema))
    table_node_mapping = SQLTableNodeMapping(sql_database=sql_database)

    table_schema_objs = []
    for table_name in include_tables:
        table_schema_objs.append(SQLTableSchema(table_name=table_name))

    service_context = ServiceContext.from_defaults(
        llm=None,
        embed_model=os.getenv("EMBEDDING_MODEL")
    )

    return ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
        service_context=service_context,
    )









dialect='postgresql'
host='analitiq-db.cw1g0qte5un7.eu-central-1.rds.amazonaws.com'
port='5432'
username='analitiq_pg'
password='yNgjV9mrJsCFGRAFV345e5v'
db_name='postgres'
driver='psycopg2'
db_schema = 'sample_data'

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, max_tokens=350, model_name='gpt-4'))

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=os.getenv("EMBEDDING_MODEL")
)

TEXT_TO_SQL_PROMPT_TEMPLATE = PromptTemplate(
    TEXT_TO_SQL_PROMPT,
    prompt_type=PromptType.TEXT_TO_SQL,
)


sql_database = SQLDatabase(create_db_engine(dialect, driver, host, port, username, password, db_name, db_schema))
db_eng = create_db_engine(dialect, driver, host, port, username, password, db_name, db_schema)

# Create an inspector
inspector = inspect(db_eng)

# Retrieve tables with other schema
tables_in_schema = inspector.get_table_names(schema=db_schema)



include_tables = tuple(sorted(tables_in_schema))
print(f"include tables: {include_tables}")

obj_index = construct_object_index(dialect, driver, host, port, username, password, db_name, db_schema, include_tables=include_tables)

query_engine = SQLTableRetrieverQueryEngine(
    sql_database=sql_database,
    text_to_sql_prompt=TEXT_TO_SQL_PROMPT_TEMPLATE,
    #context_str_prefix=table_context_info,
    service_context=service_context,
    table_retriever=obj_index.as_retriever(),
)


sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = socketio.ASGIApp(sio, app)

@sio.on('message')
async def message(sid, data):
    session_id = data.get('session_id')
    user_input = data.get('text')

    async with aiohttp.ClientSession() as session:
        tasks = [
            query_engine.query(f"{user_input} - Who are our top 10 clients?"),
            query_engine.query(f"{user_input} - Who are our top 5 clients?")
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            await sio.emit('response', {'session_id': session_id, 'text': result}, to=sid)


response = message('1', {"session_id": "session12345","text": "You are a data analyst."})
print(response.metadata.get("sql_query"))
print(response)

# you can also fetch the raw result from SQLAlchemy!
print(response.metadata["result"])
