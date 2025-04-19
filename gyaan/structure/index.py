from gyaan.structure.schema import MEMORY_SCHEMA
from gyaan.structure.memory import Memory
from gyaan.utils.io import *


import polars as pl
import asyncio

from typing import Optional,Dict, List, Any
import os

class MemoryIndex():

    def __init__(
        self,
        index_path:str):

        self.index_path=os.path.abspath(index_path)
        self._lock=asyncio.Lock()
    
    async def _init_index_table(self):
        self.index=pl.DataFrame(schema=MEMORY_SCHEMA)
        await create_table(f"file://{self.index_path}", self.index)

    
    @classmethod
    async def create(
        cls,
        index_path:str):
        
        index=cls(index_path)
        await index._init_index_table()
        return index
    
    async def add(
        self,
        memory: Memory):
        
        metadata_df=await read_table(f"file://{memory.metadata_path}")
        
        await insert_table(f"file://{self.index_path}", metadata_df)
        async with self._lock:
            self.index=await read_table(f"file://{self.index_path}")

    async def remove(
        self,
        memory: Memory):
    
        metadata_df=await read_table(f"file://{memory.metadata_path}")
        
        await delete_rows(f"file://{self.index_path}", metadata_df)
        async with self._lock:
            self.index=await read_table(f"file://{self.index_path}")
        