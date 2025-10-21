"""
Encounter request/response dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .snapshot import OverworldSnapshot


EncounterMode = Literal["escape_first", "fast_fight", "conserve_pp", "win_if_easy"]
EncounterStatus = Literal["WIN", "ESCAPED", "FAINTED", "TIMEOUT", "ERROR"]


@dataclass
class EncounterRequest:
    mode: EncounterMode
    seed: int
    snapshot_overworld: OverworldSnapshot
    timeout_steps: int = 300


@dataclass
class EncounterResult:
    status: EncounterStatus
    turns: int
    reason: Optional[str]
    snapshot_return: Optional[OverworldSnapshot] = None
    battle_telemetry_path: Optional[Path] = None
