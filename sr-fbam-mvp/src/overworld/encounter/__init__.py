"""
Encounter bridge utilities for overworld ↔ battle hand-offs.
"""

from .types import EncounterRequest, EncounterResult, OverworldSnapshot
from .bridge import EncounterBridge

__all__ = [
    "EncounterBridge",
    "EncounterRequest",
    "EncounterResult",
    "OverworldSnapshot",
]
