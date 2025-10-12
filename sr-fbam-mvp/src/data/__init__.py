"""
Data loading entry points for SR-FBAM MVP.

Exposes the dataset utilities defined in ``kg_loader.py`` so training
code can import ``load_dataset`` directly from ``src.data``.

Overview references:
- Dataset construction and operators: Overview.md:101-138
- Phase 1 implementation plan: Overview.md:142-146
"""

from .kg_loader import (  # noqa: F401
    KnowledgeGraph,
    Node,
    Edge,
    Query,
    load_kg,
    load_queries,
    load_dataset,
    load_metadata,
    get_relation_info,
)

__all__ = [
    "KnowledgeGraph",
    "Node",
    "Edge",
    "Query",
    "load_kg",
    "load_queries",
    "load_dataset",
    "load_metadata",
    "get_relation_info",
]
