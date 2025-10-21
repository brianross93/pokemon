"""
Orchestrates overworld?battle encounter hand-offs.
"""

from __future__ import annotations

import random
from typing import Dict, Mapping, Optional, Tuple

from src.pkmn_battle.graph.memory import GraphMemory

from .snapshot import OverworldSnapshot, apply_snapshot, build_snapshot
from .types import EncounterRequest, EncounterResult


class EncounterBridge:
    """
    Creates deterministic encounter requests and validates post-battle invariants.
    """

    def __init__(self) -> None:
        self._active_request: Optional[EncounterRequest] = None

    def reset(self) -> None:
        self._active_request = None

    # ------------------------------------------------------------------ #
    # Request lifecycle
    # ------------------------------------------------------------------ #

    def build_request(
        self,
        *,
        observation: Mapping[str, object],
        memory: GraphMemory,
        mode: str,
        timeout_steps: int,
        rng_state: bytes,
        party_summary: Optional[Dict[str, object]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[EncounterRequest, Dict[str, object]]:
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
        snapshot = build_snapshot(
            observation,
            memory,
            rng_state=rng_state,
            party_summary=party_summary or {},
        )
        request = EncounterRequest(
            mode=mode,
            seed=seed,
            snapshot_overworld=snapshot,
            timeout_steps=max(1, int(timeout_steps)),
        )
        self._active_request = request
        telemetry = {
            "phase": "battle.entry",
            "mode": mode,
            "seed": seed,
            "overworld": {
                "map_id": snapshot.map_id,
                "tile": list(snapshot.tile_xy) if snapshot.tile_xy else None,
                "menu_open": snapshot.menu_open,
            },
        }
        return request, telemetry

    def complete(
        self,
        *,
        result: EncounterResult,
        observation: Mapping[str, object],
        memory: GraphMemory,
    ) -> Tuple[Dict[str, bool], Dict[str, object]]:
        if self._active_request is None:
            raise RuntimeError("EncounterBridge.complete called without active request.")

        entry = self._active_request.snapshot_overworld
        exit_snapshot = result.snapshot_return or build_snapshot(observation, memory)
        invariants = self._check_invariants(entry, exit_snapshot)

        if result.snapshot_return:
            apply_snapshot(memory, result.snapshot_return)

        telemetry = {
            "phase": "battle.exit",
            "result": {
                "status": result.status,
                "turns": result.turns,
                "reason": result.reason,
            },
            "overworld_invariants": invariants,
            "handoff": {
                "battle_log": str(result.battle_telemetry_path) if result.battle_telemetry_path else None,
            },
        }
        self._active_request = None
        return invariants, telemetry

    def fail(
        self,
        *,
        reason: str,
    ) -> Dict[str, object]:
        telemetry = {
            "phase": "battle.exit",
            "result": {
                "status": "ERROR",
                "turns": 0,
                "reason": reason,
            },
            "overworld_invariants": {"menu_closed": False, "tile_restored": False},
            "handoff": {"battle_log": None},
        }
        self._active_request = None
        return telemetry

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _check_invariants(
        self,
        entry: OverworldSnapshot,
        exit_snapshot: OverworldSnapshot,
    ) -> Dict[str, bool]:
        menu_closed = not exit_snapshot.menu_open
        tile_restored = bool(
            entry.tile_xy == exit_snapshot.tile_xy
            if entry.tile_xy is not None
            else True
        )
        return {"menu_closed": menu_closed, "tile_restored": tile_restored}
