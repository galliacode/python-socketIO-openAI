from typing import Annotated
from chromadb.api import ClientAPI
from fastapi import APIRouter, Depends, UploadFile

from app.schemas.user import User
from app.schemas.vector_store import (
    VectoreStoreCollection, 
    VectoreStoreCollectionCreate,
    FileMetadata
)
from app.vector_store import crud as vcrud
from app.dependencies import get_current_user, get_vector_store

router = APIRouter()

@router.get("/{collection_name}")
async def get_collection_by_name(
    collection_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
    client: Annotated[ClientAPI, Depends(get_vector_store)],
) -> VectoreStoreCollection:
    """
    Get a collection in the Chroma DB using the collection name.

    :param collection_name: the collection name of Chroma DB
    :param current_user: the current logged in user  
    :param client: the chroma client  
    
    Sample Response Body:  

    {  
        "name": "table_schema",  
        "metadata": null  
    }  
    """
    
    collection = vcrud.get_collection_by_name(client=client, collection_name=collection_name)
    return {
        "name": collection.name,
        "metadata": collection.metadata,
    }

@router.post("/")
async def create_collection(
    collection: VectoreStoreCollectionCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    client: Annotated[ClientAPI, Depends(get_vector_store)],
):
    """
    Create a new collection in the Chroma DB.

    :param collection_name: the collection name of Chroma DB
    :param current_user: the current logged in user  
    :param client: the chroma client  
    
    Sample Response Body:  

    {  
        "name": "test",  
        "id": "7e3fae50-357a-4e3f-b36a-b99b708d0c86",  
        "metadata": {  
            "type": "internal"  ----> for internal client  
        },  
        "tenant": null,  
        "database": null  
    }  
    """
    return vcrud.create_collection(client=client, collection=collection)

@router.patch("/")
async def update_collection_info(
    collection: VectoreStoreCollectionCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    client: Annotated[ClientAPI, Depends(get_vector_store)],
):
    """
    Update collection's information in the Chroma DB, including collection's name and its metadata.
    """
    return vcrud.update_collection_info(client=client, collection=collection)

@router.post("/{collection_name}/upload")
async def user_file_upload(
    collection_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
    client: Annotated[ClientAPI, Depends(get_vector_store)],
    file: UploadFile,
):
    """
    Upload context file into a collection. Not Implemented yet.
    """
    raise NotImplementedError

@router.delete("/{collection_name}")
async def delete_collection(
    collection_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
    client: Annotated[ClientAPI, Depends(get_vector_store)],
):
    """
    Delete a collection in the Chroma DB.

    :param collection_name: the collection name of Chroma DB
    :param current_user: the current logged in user  
    :param client: the chroma client  
    
    Sample Response Body:  
    (It returns a boolean value to indicate the success of the operation)  

    true
    
    """
    return vcrud.delete_collection(client=client, collection_name=collection_name)