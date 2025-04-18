from gyaan.structure.schema import *
from gyaan.utils.io import *

import polars as pl
import asyncio

from datetime import datetime, timezone
from typing import Optional,Dict, List, Any
import os
from uuid import uuid4

class Memory():

    def __init__(
        self,
        memory_path:str,
        title: str,
        description: str,
        embedding: Optional[List[float]]=[],
        keywords: Optional[List[str]]=[]):

        self.id=str(uuid4())
        self.title=title
        self.description=description
        self.embedding=embedding
        self.keywords=keywords
        self.memory_storage_path=memory_path
        self.is_deleted=False
        self._lock=asyncio.Lock()

        self.metadata_path=os.path.abspath(os.path.join(self.memory_storage_path,"metadata"))
        self.nodes_path=os.path.abspath(os.path.join(self.memory_storage_path,"nodes"))
        self.edges_path=os.path.abspath(os.path.join(self.memory_storage_path,"edges"))

        os.makedirs(self.memory_storage_path,exist_ok=True)

    
    async def _initialize_tables(
        self,
        node_attributes: Optional[Dict[Any,Any]]={},
        edge_attributes: Optional[Dict[Any,Any]]={}):

        metadata_df=pl.DataFrame(data=[{
            "id":self.id,
            "title":self.title,
            "description":self.description,
            "embedding":self.embedding,
            "keywords":self.keywords,
            "is_deleted":self.is_deleted,
            "memory_storage_path":self.memory_storage_path
        }])

        self.nodes=pl.DataFrame(schema=generate_node_schema(node_attributes))
        self.edges=pl.DataFrame(schema=generate_edge_schema(edge_attributes))

        await create_table(f"file://{self.metadata_path}", metadata_df)
        await create_table(f"file://{self.nodes_path}", self.nodes)
        await create_table(f"file://{self.edges_path}", self.edges)
    
    async def _load_nodes_edges(self):
        self.nodes=await read_table(f"file://{self.nodes_path}")
        self.edges=await read_table(f"file://{self.edges_path}")

    async def update_metadata(
        self,
        title: Optional[str],
        description: Optional[str],
        embedding: Optional[List[float]],
        keywords: Optional[List[str]]):

        update_dict={
            "title":title,
            "description":description,
            "embedding":embedding,
            "keywords":keywords
        }
        filtered_update={k: v for k, v in update_dict.items() if v is not None}

    
        async with self._lock:
            self.__dict__.update(filtered_update)
            
        metadata_df=pl.DataFrame(data=[{
            "id":self.id,
            "title":self.title,
            "description":self.description,
            "embedding":self.embedding,
            "keywords":self.keywords,
            "is_deleted":self.is_deleted,
            "memory_storage_path":self.memory_storage_path
        }])

        await create_table(f"file://{self.metadata_path}", metadata_df,mode="overwrite")
    
    async def soft_delete(self):
        async with self._lock:
            self.is_deleted=True

        metadata_df=pl.DataFrame(data=[{
            "id":self.id,
            "title":self.title,
            "description":self.description,
            "embedding":self.embedding,
            "keywords":self.keywords,
            "is_deleted":self.is_deleted,
            "memory_storage_path":self.memory_storage_path
        }])

        await create_table(f"file://{self.metadata_path}", metadata_df,mode="overwrite")

    @classmethod
    async def create(
        cls, 
        memory_path: str, 
        title: str, 
        description: str, 
        embedding: Optional[List[float]] = [], 
        keywords: Optional[List[str]] = [], 
        node_attributes: Optional[Dict[Any, Any]] = {}, 
        edge_attributes: Optional[Dict[Any, Any]] = {}):
        
        memory = cls(memory_path, title, description, embedding, keywords)
        await memory._initialize_tables(node_attributes=node_attributes,edge_attributes=edge_attributes)
        return memory
    
    @classmethod 
    async def load(cls,memory_path:str):
        metadata_path=os.path.abspath(os.path.join(memory_path,"metadata"))

        try:
            metadata_df=await read_table(f"file://{metadata_path}")
            metadata_dict=metadata_df.to_dict(as_series=False)
        except:
            raise ValueError("Memory not found")
        if metadata_dict["is_deleted"][0]==False:
            memory=cls(
                memory_path,
                metadata_dict["title"][0],
                metadata_dict["description"][0],
                metadata_dict["embedding"][0],
                metadata_dict["keywords"][0]
            )

            await memory._load_nodes_edges()
            return memory
        
        else:
            raise ValueError("Memory has been soft-deleted")
    
        