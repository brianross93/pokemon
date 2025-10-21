"""
Battle-specific scaffolding for the SR-FBAM agent.

V0.1 stubs expose the interfaces for environment adapters, symbolic
extractors, and graph memory. Implementations will populate these in the
next milestones.
"""

from __future__ import annotations

from .env.core import BattleObs, EnvAdapter, LegalAction
from .extractor.base import Extractor
from .graph.memory import GraphMemory
from .graph.schema import Edge, Node, WriteOp

__all__ = [
    "BattleObs",
    "EnvAdapter",
    "LegalAction",
    "Extractor",
    "GraphMemory",
    "Node",
    "Edge",
    "WriteOp",
]

