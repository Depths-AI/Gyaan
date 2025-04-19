import sys
from shutil import rmtree
from pathlib import Path
import asyncio
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from gyaan.structure.memory import Memory

NUM_NODES=100
NUM_EDGES=10
async def main():
    try:
        mem=await Memory.create(
            memory_path="test_edge_delete",
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

        source_ids=[]
        target_ids=[]
        for i in node_ids:
            for j in node_ids:
                if i!=j and (i not in source_ids or j not in target_ids) and len(source_ids)<NUM_EDGES:
                    source_ids.append(i)
                    target_ids.append(j)

        
        edge_ids=await mem.add_edges(
            source_nodes=source_ids,
            target_nodes=target_ids,
            labels=[f"Test edge {i}" for i in range(NUM_EDGES)],
            weights=[0.0]*NUM_EDGES,
            descriptions=["This is a test node."]*NUM_EDGES,
            keywords=[["keywords"]]*NUM_EDGES,
            embeddings=[[1.0,2.0]]*NUM_EDGES,
            type=["internal"]*NUM_EDGES
        )

        print(await mem.get_edges())
        print(await mem.get_edges_by_id(edge_ids=edge_ids))

        edge_ids=await mem.delete_edges(
            edge_ids=edge_ids
        )

        print("Inspecting edges after soft-delete")
        print(await mem.get_edges())
        print(await mem.get_edges_by_id(edge_ids=edge_ids))
    
    finally:
        rmtree("test_edge_delete")

asyncio.run(main())