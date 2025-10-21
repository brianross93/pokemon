"""
Thin wrapper around the PyBoy adapter for overworld-specific ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OverworldObservation:
    """Lightweight structured observation returned by the adapter."""

    frame: int
    overworld: Dict[str, Any]


class OverworldAdapter:
    """
    Placeholder adapter mirroring the battle adapter interface.

    The concrete implementation can wrap ``PyBoyPokemonAdapter`` and expose
    the subset of calls the overworld executor requires (observe, step, reset).
    """

    def __init__(self, pyboy_adapter: Any) -> None:
        self._adapter = pyboy_adapter

    def reset(self) -> OverworldObservation:
        raw = self._adapter.reset()
        return self._to_observation(raw)

    def observe(self) -> OverworldObservation:
        raw = self._adapter.observe()
        return self._to_observation(raw)

    def step(self, action: Dict[str, Any]) -> OverworldObservation:
        raw = self._adapter.step(action)
        return self._to_observation(raw)

    @staticmethod
    def _to_observation(raw: Dict[str, Any]) -> OverworldObservation:
        frame = int(raw.get("frame", 0))
        overworld = dict(raw.get("overworld", {}))
        return OverworldObservation(frame=frame, overworld=overworld)
