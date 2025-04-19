import sys
from shutil import rmtree
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from gyaan.structure.memory import Memory

NUM_NODES=100
async def main():
    try:
        mem=await Memory.create(
            memory_path="test_node_update",
            title="Test Memory",
            description="This is a test memory.",
            embedding=[0.0],
            keywords=["test"],
            node_attributes={"impact":int},
            edge_attributes={"type":str}
        )

        node_ids=await mem.add_nodes(
            labels=[f"Test Node {i}" for i in range(NUM_NODES)],
            weights=[0.0]*NUM_NODES,
            descriptions=["This is a test node."]*NUM_NODES,
            keywords=[["keywords"]]*NUM_NODES,
            embeddings=[[1.0,2.0]]*NUM_NODES,
            impact=[0]*NUM_NODES
        )

        node_ids=await mem.update_nodes(
            node_ids=node_ids,
            keywords=[["depths-ai"]]*NUM_NODES,
            impact=[1]*NUM_NODES)
        
        print(mem.nodes)
    
    finally:
        rmtree("test_node_update")

asyncio.run(main())