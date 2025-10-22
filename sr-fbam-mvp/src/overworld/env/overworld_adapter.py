"""
Thin wrapper around the PyBoy adapter for overworld-specific ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.middleware.pokemon_adapter import PokemonAction, PokemonTelemetry


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

    def step(self, action: PokemonAction) -> OverworldObservation:
        raw = self._adapter.step(action)
        return self._to_observation(raw)

    @staticmethod
    def _to_observation(raw: PokemonTelemetry) -> OverworldObservation:
        # Convert PokemonTelemetry to the expected format
        overworld_data = {
            "area_id": raw.area_id,
            "x": raw.x,
            "y": raw.y,
            "in_grass": raw.in_grass,
            "in_battle": raw.in_battle,
            "encounter_species_id": raw.encounter_species_id,
            "step_counter": raw.step_counter,
            "elapsed_ms": raw.elapsed_ms,
            "method": raw.method,
            "extra": raw.extra
        }
        return OverworldObservation(frame=0, overworld=overworld_data)
