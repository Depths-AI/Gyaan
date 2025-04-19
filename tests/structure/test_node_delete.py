import sys
from shutil import rmtree
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from gyaan.structure.memory import Memory

NUM_NODES=100
NUM_DELETE=100
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

        print(await mem.get_nodes())
        print(await mem.get_nodes_by_id(node_ids=node_ids[:10]))

        node_ids=await mem.delete_nodes(
            node_ids=node_ids[:NUM_DELETE])

        print("Nodes fetched after soft-delete")
        print(await mem.get_nodes())
        print(await mem.get_nodes_by_id(node_ids=node_ids[:10]))
        
        assert mem.nodes.height==NUM_NODES
        assert mem.nodes["deleted"].to_list()==[True]*NUM_DELETE
        print("Test completed successfully!")
    
    finally:
        rmtree("test_node_update")

asyncio.run(main())