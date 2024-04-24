from chromadb.api import ClientAPI
from chromadb import Collection
from app.schemas.vector_store import VectoreStoreCollectionCreate

def get_collection_by_name(client: ClientAPI, collection_name: str) -> Collection:
    return client.get_collection(collection_name)

def create_collection(
    client: ClientAPI, 
    collection: VectoreStoreCollectionCreate
) -> Collection:
    created_collection = client.create_collection(
        name=collection.name, 
        metadata=collection.metadata
    )
    return created_collection

def update_collection_info(
    client: ClientAPI, 
    collection: VectoreStoreCollectionCreate
) -> Collection:
    modified_collection = client.get_collection(name=collection.name)
    modified_collection.modify(**collection.model_dump())
    return modified_collection

def delete_collection(client: ClientAPI, collection_name: str) -> bool:
    try:
        client.delete_collection(name=collection_name)
        return True
    except:
        return False