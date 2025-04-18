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
            memory_path="test_node_insertion",
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

        assert mem.nodes.height == NUM_NODES
        assert mem.nodes["node_id"].to_list()==node_ids
        assert mem.nodes["memory_id"].to_list()==[mem.id]*NUM_NODES

        print("Test completed successfully!")
    finally:
        rmtree("test_node_insertion")

asyncio.run(main())