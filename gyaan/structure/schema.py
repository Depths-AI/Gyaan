from typing import List, Dict, Any

MEMORY_SCHEMA={
    "id": str,  
    "title": str,
    "description": str,
    "keywords": List[str],
    "embedding": List[float],
    "deleted": bool,
    "memory_storage_path":str,
    "node_attributes":List[str],
    "edge_attributes":List[str]
}

NODE_SCHEMA={
    "memory_id": str,  
    "node_id": str,
    "weight": float,
    "label": str,
    "description": str,
    "keywords": List[str],
    "embedding": List[float],
    "deleted": bool
}

EDGE_SCHEMA={
    "memory_id":str,
    "edge_id":str,
    "source_node_id": str,  
    "target_node_id": str,
    "weight": float,
    "label": str,
    "description": str,
    "keywords": List[str],
    "embedding": List[float],
    "deleted": bool
}

def generate_node_schema(custom_attributes: Dict[Any,Any]):
    node_schema = NODE_SCHEMA.copy()
    node_schema.update(custom_attributes)
    return node_schema

def generate_edge_schema(custom_attributes: Dict[Any,Any]):
    edge_schema=EDGE_SCHEMA.copy()
    edge_schema.update(custom_attributes)
    return edge_schema
    