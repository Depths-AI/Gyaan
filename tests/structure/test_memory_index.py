import sys
from shutil import rmtree
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from gyaan.structure.memory import Memory
from gyaan.structure.index import MemoryIndex

async def main():
    try:
        mem=await Memory.create(
            memory_path="test_memory",
            title="Test Memory",
            description="This is a test memory.",
            embedding=[0.0],
            keywords=["test"],
            node_attributes={"impact":int},
            edge_attributes={"type":str}
        )
        memory_index=await MemoryIndex.create(index_path="test_memory_index")
        await memory_index.add(mem)
        print(memory_index.index)
        await memory_index.remove(mem)

        print("Index after Memory removal")
        print(memory_index.index)
    
    finally:
        rmtree("test_memory_index")
        rmtree("test_memory")


asyncio.run(main())