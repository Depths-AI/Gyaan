import sys
from shutil import rmtree
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from gyaan.structure.memory import Memory

async def main():
    try:
        mem=await Memory.create(
            memory_path="test_memory",
            title="Test Memory",
            description="This is a test memory.",
            embedding=[0.0],
            keywords=["test"]
        )

        assert mem.nodes.height==0
        assert mem.edges.height==0

        retrieved_mem=await Memory.load(memory_path="test_memory")
        
        assert retrieved_mem.title==mem.title

        await retrieved_mem.update_metadata(
            title="Updated Test Memory",
            description="This is an updated test memory.",
            embedding=[0.0, 0.0],
            keywords=["test", "updated"]
        )

        assert retrieved_mem.title=="Updated Test Memory"
        
        updated_mem=await Memory.load(memory_path="test_memory")

        assert updated_mem.title==retrieved_mem.title

        await updated_mem.soft_delete()

        deleted_correctly=False
        try:
            deleted_mem=await Memory.load(memory_path="test_memory")
        except:
            deleted_correctly=True
        
        assert deleted_correctly==True
        print("Test completed successfully!")

    finally:
        rmtree("test_memory")

asyncio.run(main())