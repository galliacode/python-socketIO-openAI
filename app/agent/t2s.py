import os
import logging
import sys
from functools import lru_cache

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import pandas as pd
import chromadb
from typing import Union, Any, Dict, Tuple
from app.utils import db_utils

from llama_index import (
    SQLDatabase, 
    VectorStoreIndex, 
    LLMPredictor, 
    ServiceContext,
)
from llama_index.query_engine import BaseQueryEngine
from llama_index.prompts import PromptTemplate, PromptType
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.vector_stores import ChromaVectorStore

from langchain.chat_models.openai import ChatOpenAI

from app.utils.js_code_plot_generator import JSCodePlotGenerator
#from app.dependencies import get_sagemaker_llm
from app.utils.encryptor import decode_data

CACHE_SIZE = 64


@lru_cache(maxsize=CACHE_SIZE)
def create_table_metadata_query_engine(
    collection_name: str = os.getenv("VECTOR_STORE_CONTEXT_COLLECTION"),
) -> BaseQueryEngine:
    logging.info(msg="Constructing metadata index from context file")
    vector_db = chromadb.HttpClient(
        host=os.getenv("CHROMA_DB_HOST"),
        port=os.getenv("CHROMA_DB_PORT")
    )
    collection = vector_db.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    service_context = ServiceContext.from_defaults(
        llm=None,
        embed_model=os.getenv("EMBEDDING_MODEL")
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        service_context=service_context,
    ).as_query_engine()

@lru_cache(maxsize=CACHE_SIZE)
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

    logging.info(msg="constructing object index from table schema")
    sql_database = SQLDatabase(db_utils.create_db_engine(dialect, driver, host, port, username, decode_data(password), db_name, db_schema))
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

async def create_query_engine(
    query: str,
    dialect: str,
    host: str,
    port: int,
    username: str,
    password: str,
    db_name: str,
    db_schema: str,
    model_name: str = "gpt-4",
    driver: Union[str, None] = None
) -> SQLTableRetrieverQueryEngine:
    
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

    TEXT_TO_SQL_PROMPT_TEMPLATE = PromptTemplate(
        TEXT_TO_SQL_PROMPT,
        prompt_type=PromptType.TEXT_TO_SQL,
    )

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, max_tokens=350, model_name='gpt-4'))
    #llm_predictor = LLMPredictor(llm=get_sagemaker_llm())
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, 
        embed_model=os.getenv("EMBEDDING_MODEL")
    )

    db_eng = db_utils.create_db_engine(dialect, driver, host, port, username, decode_data(password), db_name, db_schema)
    sql_database = SQLDatabase(db_eng)
    tables_in_schema = db_utils.get_tables_in_schema(db_eng, db_schema)

    include_tables = tuple(sorted(tables_in_schema))
    logging.info(f"include tables: {include_tables}")

    obj_index = construct_object_index(dialect, driver, host, port, username, password, db_name, db_schema, include_tables=include_tables)
    metadat_query_engine = create_table_metadata_query_engine()

    table_context_info = await metadat_query_engine.aquery(query)
    table_context_info = table_context_info.response

    return SQLTableRetrieverQueryEngine(
        sql_database=sql_database,
        text_to_sql_prompt=TEXT_TO_SQL_PROMPT_TEMPLATE,
        context_str_prefix=table_context_info,
        service_context=service_context,
        table_retriever=obj_index.as_retriever(),
    )

async def execute_query(query_engine: SQLTableRetrieverQueryEngine, query: str) -> Dict[str, Any]:
    
    response  = await query_engine.aquery(query)
    sql_query = response.metadata.get("sql_query")

    result    = response.metadata.get("result")
    df        = pd.DataFrame()
    html_plot_js = ""

    if result:
        df = pd.DataFrame(result)
        html_plot_js = await JSCodePlotGenerator(sql_query=sql_query, data=df) \
            .generate_plot(model_name="gpt-4")

    return {
        "sql": sql_query,
        "text": str(response),
        "result": df.to_json(orient="split"),
        "plot_code": html_plot_js,
    }
