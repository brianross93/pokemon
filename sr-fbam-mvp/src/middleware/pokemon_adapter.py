"""Interfaces that expose Pokemon Blue emulator telemetry to the controller."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PokemonTelemetry:
    """
    Snapshot of emulator state required by the symbolic controller.

    The adapter is responsible for filling these fields from memory or logs.
    """

    area_id: int
    x: int
    y: int
    in_grass: bool
    in_battle: bool
    encounter_species_id: Optional[int] = None
    step_counter: int = 0
    elapsed_ms: float = 0.0
    method: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> Dict[str, Any]:
        """Return a normalised context dictionary for knowledge lookups."""
        context = {
            "area_id": self.area_id,
            "method": self.method,
        }
        if "time_bucket" in self.extra:
            context["time_bucket"] = self.extra["time_bucket"]
        return context


@dataclass
class PokemonAction:
    """High-level action request sent to the emulator adapter."""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)


class PokemonBlueAdapter(ABC):
    """
    Minimal abstraction for driving Pokemon Blue via an emulator.

    A concrete implementation should bridge to PyBoy, BizHawk, or a similar
    emulator by translating `PokemonAction` into button presses or scripts and
    filling `PokemonTelemetry` from memory reads.
    """

    @abstractmethod
    def reset(self) -> PokemonTelemetry:
        """Reset to a known state and return the initial telemetry."""

    @abstractmethod
    def step(self, action: PokemonAction) -> PokemonTelemetry:
        """Apply an action and return the resulting telemetry snapshot."""

    def save_state(self, slot: int = 0) -> None:
        """Optional hook for persisting emulator state."""

    def load_state(self, slot: int = 0) -> PokemonTelemetry:
        """Optional hook for restoring emulator state."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional hook for releasing emulator resources."""
        return None
