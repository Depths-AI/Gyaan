from gyaan.structure.schema import *
from gyaan.utils.io import *

import polars as pl
import asyncio

from typing import Optional,Dict, List, Any
import os
from uuid import uuid4

class Memory():

    def __init__(
        self,
        memory_path:str,
        title: str,
        description: str,
        embedding: Optional[List[float]]=[0.0]*1024,
        keywords: Optional[List[str]]=["keyword"]):

        self.id=str(uuid4())
        self.title=title
        self.description=description
        self.embedding=embedding
        self.keywords=keywords
        self.memory_storage_path=memory_path
        self.deleted=False
        self._lock=asyncio.Lock()

        self.metadata_path=os.path.abspath(os.path.join(self.memory_storage_path,"metadata"))
        self.nodes_path=os.path.abspath(os.path.join(self.memory_storage_path,"nodes"))
        self.edges_path=os.path.abspath(os.path.join(self.memory_storage_path,"edges"))

        os.makedirs(self.memory_storage_path,exist_ok=True)

    
    async def _initialize_tables(
        self,
        node_attributes: Optional[Dict[Any,Any]]={},
        edge_attributes: Optional[Dict[Any,Any]]={}):

        node_schema=generate_node_schema(node_attributes)
        edge_schema=generate_edge_schema(edge_attributes)

        self.node_columns=list(node_schema.keys())
        self.edge_columns=list(edge_schema.keys())

        metadata_df=pl.DataFrame(data=[{
            "id":self.id,
            "title":self.title,
            "description":self.description,
            "embedding":self.embedding,
            "keywords":self.keywords,
            "deleted":self.deleted,
            "memory_storage_path":self.memory_storage_path,
            "node_attributes":self.node_columns,
            "edge_attributes":self.edge_columns
        }])

        

        self.nodes=pl.DataFrame(schema=node_schema)
        self.edges=pl.DataFrame(schema=edge_schema)

        await create_table(f"file://{self.metadata_path}", metadata_df)
        await create_table(f"file://{self.nodes_path}", self.nodes)
        await create_table(f"file://{self.edges_path}", self.edges)
    
    async def _load_nodes_edges(self):
        self.nodes=await read_table(f"file://{self.nodes_path}")
        self.node_columns=self.nodes.columns
        self.edges=await read_table(f"file://{self.edges_path}")
        self.edge_columns=self.edges.columns

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
            "deleted":self.deleted,
            "memory_storage_path":self.memory_storage_path,
            "node_attributes":self.node_columns,
            "edge_attributes":self.edge_columns
        }])

        await create_table(f"file://{self.metadata_path}", metadata_df,mode="overwrite")
    
    async def soft_delete(self):
        async with self._lock:
            self.deleted=True

        metadata_df=pl.DataFrame(data=[{
            "id":self.id,
            "title":self.title,
            "description":self.description,
            "embedding":self.embedding,
            "keywords":self.keywords,
            "deleted":self.deleted,
            "memory_storage_path":self.memory_storage_path,
            "node_attributes":self.node_columns,
            "edge_attributes":self.edge_columns
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
        if metadata_dict["deleted"][0]==False:
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
    
    async def add_nodes(
        self,
        labels: List[str],
        weights: List[str],
        descriptions: List[str],
        keywords: List[List[str]],
        embeddings: List[List[float]],
        **node_attributes: List[Any]):

        batch_size=len(labels)

        node_ids=[str(uuid4()) for _ in range(batch_size)]

        update_dict={
            "memory_id":[self.id]*batch_size,
            "node_id":node_ids,
            "weight":weights,
            "label":labels,
            "description":descriptions,
            "keywords":keywords,
            "embedding":embeddings,
            "deleted":[False]*batch_size,
            **node_attributes
        }
    
        assert self.node_columns==list(update_dict.keys()),f"Not all attributes have been supplied. Correct attributes are: {self.node_columns}"

        insertion_df=pl.DataFrame(data=update_dict)

        await insert_table(f"file://{self.nodes_path}", insertion_df)

        async with self._lock:
            self.nodes=await read_table(f"file://{self.nodes_path}")
            self.node_columns=self.nodes.columns

        return node_ids
    
    async def get_nodes(self):
        return self.nodes.filter(pl.col("deleted") == False)
    
    async def get_nodes_by_id(self, node_ids: List[str]):
        return self.nodes.filter((pl.col("deleted") == False) & (pl.col("node_id").is_in(node_ids)))
    
    async def update_nodes(
        self, 
        node_ids: List[str], 
        **node_attributes: List[Any]):

        update_dict={
            "node_id":node_ids,
            **node_attributes
        }

        update_df=pl.DataFrame(data=update_dict)
        await update_table(table_path=f"file://{self.nodes_path}",update_df=update_df,id_column="node_id")

        async with self._lock:
            self.nodes=await read_table(f"file://{self.nodes_path}")
            self.node_columns=self.nodes.columns

        return node_ids
    
    async def delete_nodes(
        self, 
        node_ids: List[str]):
        
        update_dict={
            "node_id":node_ids,
            "deleted":[True]*len(node_ids)
        }

        update_df=pl.DataFrame(data=update_dict)
        await update_table(table_path=f"file://{self.nodes_path}",update_df=update_df,id_column="node_id")

        async with self._lock:
            self.nodes=await read_table(f"file://{self.nodes_path}")
            self.node_columns=self.nodes.columns

        return node_ids
    
    async def add_edges(
        self,
        source_nodes: List[str],
        target_nodes: List[str],
        labels: List[str],
        weights: List[str],
        descriptions: List[str],
        keywords: List[List[str]],
        embeddings: List[List[float]],
        **edge_attributes: List[Any]):

        batch_size=len(source_nodes)

        assert len(target_nodes)==batch_size, "All edges should have source and target nodes"

        edge_ids=[str(uuid4()) for _ in range(batch_size)]

        update_dict={
            "memory_id":[self.id]*batch_size,
            "edge_id":edge_ids,
            "source_node_id":source_nodes,
            "target_node_id":target_nodes,
            "weight":weights,
            "label":labels,
            "description":descriptions,
            "keywords":keywords,
            "embedding":embeddings,
            "deleted":[False]*batch_size,
            **edge_attributes
        }

        assert self.edge_columns==list(update_dict.keys()),f"Not all attributes have been supplied. Correct attributes are: {self.edge_columns}"

        insertion_df=pl.DataFrame(data=update_dict)

        await insert_table(f"file://{self.edges_path}", insertion_df)

        async with self._lock:
            self.edges=await read_table(f"file://{self.edges_path}")
            self.edge_columns=self.edges.columns

        return edge_ids

    async def get_edges(self):
        return self.edges.filter(pl.col("deleted") == False)
    
    async def get_edges_by_id(self, edge_ids: List[str]):
        return self.edges.filter((pl.col("deleted") == False) & (pl.col("edge_id").is_in(edge_ids)))
    
    async def update_edges(
        self,
        edge_ids: List[str],
        **edge_attributes: List[str]):

        update_dict={
            "edge_id":edge_ids,
            **edge_attributes
        }

        update_df=pl.DataFrame(data=update_dict)
        await update_table(table_path=f"file://{self.edges_path}",update_df=update_df,id_column="edge_id")

        async with self._lock:
            self.edges=await read_table(f"file://{self.edges_path}")
            self.edges_columns=self.edges.columns

        return edge_ids

    async def delete_edges(
        self,
        edge_ids: List[str]):

        update_dict={
            "edge_id":edge_ids,
            "deleted":[True]*len(edge_ids)
        }

        update_df=pl.DataFrame(data=update_dict)
        await update_table(table_path=f"file://{self.edges_path}",update_df=update_df,id_column="edge_id")

        async with self._lock:
            self.edges=await read_table(f"file://{self.edges_path}")
            self.edges_columns=self.edges.columns

        return edge_ids
    