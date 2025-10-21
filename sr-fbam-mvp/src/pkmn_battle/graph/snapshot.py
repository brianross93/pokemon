"""
Snapshot utilities for persisting battle graph state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GraphSnapshot:
    """
    Lightweight serialisable view of the battle graph.
    """

    nodes: List[Dict[str, object]]
    edges: List[Dict[str, object]]

