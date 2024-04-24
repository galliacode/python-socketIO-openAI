from typing import Union, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

class VectoreStoreCollection(BaseModel):
    name: str
    metadata: Union[Dict[str, Any], None] = None

class VectoreStoreCollectionCreate(VectoreStoreCollection):
    pass

class FileMetadata(BaseModel):
    company_name: str
    document_name: str
    document_type: str
    document_categories: List[str]
    url: str