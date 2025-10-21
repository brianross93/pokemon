"""
Typed node/edge structures for the battle graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union


@dataclass(frozen=True)
class Node:
    """Typed entity stored in the battle graph."""

    type: str
    node_id: str
    attributes: Dict[str, object]


@dataclass(frozen=True)
class Edge:
    """Directed relation between two entities."""

    relation: str
    src: str
    dst: str
    attributes: Dict[str, object]


@dataclass(frozen=True)
class WriteOp:
    """Idempotent graph mutation request emitted by the extractor."""

    kind: Literal["node", "edge"]
    payload: Union[Node, Edge]
    confidence: float = 1.0
    fallback_required: bool = False
    turn_hint: Optional[int] = None
