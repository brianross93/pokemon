"""Adapter interfaces for exposing emulator frame observations to controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

try:  # pragma: no cover - imported for typing; numpy is a runtime dependency
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError:  # pragma: no cover - defensive fallback
    np = None  # type: ignore[assignment]
    NDArray = Sequence[Sequence[Any]]  # type: ignore[assignment]


@dataclass
class ObservationBundle:
    """
    Snapshot captured at the controller's decision cadence.

    Attributes
    ----------
    framebuffer:
        Downsampled RGB framebuffer, typically shape (40, 120, 3) with dtype ``uint8``.
    ram:
        Optional sparse mapping of RAM addresses to byte values captured alongside the frame.
        Intended for guard rails when the visual parse is ambiguous.
    metadata:
        Free-form metadata describing the capture (frame index, hash, latency, etc.).
    """

    framebuffer: NDArray
    raw_framebuffer: Optional[NDArray] = None
    ram: Optional[Mapping[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ObservationBundle":
        """Return a shallow copy with metadata duplicated."""
        frame = self.framebuffer.copy() if hasattr(self.framebuffer, "copy") else self.framebuffer
        raw = (
            self.raw_framebuffer.copy()
            if hasattr(self.raw_framebuffer, "copy") and self.raw_framebuffer is not None
            else self.raw_framebuffer
        )
        ram_copy = dict(self.ram) if isinstance(self.ram, Mapping) else self.ram
        return ObservationBundle(
            framebuffer=frame,
            raw_framebuffer=raw,
            ram=ram_copy,
            metadata=dict(self.metadata),
        )


@dataclass
class PokemonAction:
    """High-level action request sent to the emulator adapter."""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)


class PokemonBlueAdapter(ABC):
    """
    Minimal abstraction for driving Pokemon Blue via an emulator.

    A concrete implementation should bridge to PyBoy, BizHawk, or a similar
    emulator by translating ``PokemonAction`` into button presses or scripts and
    returning :class:`ObservationBundle` instances for downstream processing.
    """

    @abstractmethod
    def reset(self) -> ObservationBundle:
        """Reset to a known state and return the initial observation bundle."""

    @abstractmethod
    def step(self, action: PokemonAction) -> ObservationBundle:
        """Apply an action and return the resulting observation bundle."""

    def save_state(self, slot: int = 0) -> None:
        """Optional hook for persisting emulator state."""

    def load_state(self, slot: int = 0) -> ObservationBundle:
        """Optional hook for restoring emulator state."""
        raise NotImplementedError

    def observe(self) -> ObservationBundle:
        """
        Optional hook for capturing the current frame without applying an action.

        Implementations may raise :class:`NotImplementedError` when unsupported.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Optional hook for releasing emulator resources."""
        return None


# Backwards-compatible alias for legacy imports. Prefer ObservationBundle.
PokemonObservation = ObservationBundle
PokemonTelemetry = ObservationBundle  # Deprecated alias; prefer ObservationBundle
