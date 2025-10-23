"""
Thin wrapper around the PyBoy adapter for overworld-specific ticks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.middleware.pokemon_adapter import ObservationBundle, PokemonAction


@dataclass(slots=True)
class OverworldObservation:
    """Screenshot-first observation handed to SR-FBAM."""

    framebuffer: np.ndarray
    raw_framebuffer: Optional[np.ndarray] = None
    ram: Optional[Mapping[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_bundle(self) -> ObservationBundle:
        """Return a detached :class:`ObservationBundle` copy."""
        frame = self.framebuffer.copy()
        raw = self.raw_framebuffer.copy() if isinstance(self.raw_framebuffer, np.ndarray) else self.raw_framebuffer
        ram_copy = dict(self.ram) if isinstance(self.ram, Mapping) else self.ram
        return ObservationBundle(
            framebuffer=frame,
            raw_framebuffer=raw,
            ram=ram_copy,
            metadata=dict(self.metadata),
        )

    def frame_hash(self) -> str:
        """Return (or compute) a stable hash for the framebuffer."""
        existing = self.metadata.get("frame_hash")
        if isinstance(existing, str) and existing:
            return existing
        digest = hashlib.sha1(self.framebuffer.tobytes())  # nosec: non-cryptographic usage
        value = digest.hexdigest()
        self.metadata["frame_hash"] = value
        return value


class OverworldAdapter:
    """
    Adapter that surfaces frame observations from a lower-level emulator adapter.
    """

    def __init__(self, pyboy_adapter: Any) -> None:
        self._adapter = pyboy_adapter

    def reset(self) -> OverworldObservation:
        raw = self._adapter.reset()
        return self._to_observation(raw)

    def observe(self) -> OverworldObservation:
        raw_observe = getattr(self._adapter, "observe", None)
        if raw_observe is None:
            raise AttributeError("Underlying adapter does not support observe()")
        raw = raw_observe()
        return self._to_observation(raw)

    def step(self, action: PokemonAction) -> OverworldObservation:
        raw = self._adapter.step(action)
        return self._to_observation(raw)

    @staticmethod
    def _to_observation(raw: ObservationBundle) -> OverworldObservation:
        framebuffer = np.asarray(raw.framebuffer, dtype=np.uint8)
        if framebuffer.ndim == 2:
            framebuffer = np.repeat(framebuffer[:, :, None], 3, axis=2)
        raw_buffer = None
        if getattr(raw, "raw_framebuffer", None) is not None:
            raw_buffer = np.asarray(raw.raw_framebuffer, dtype=np.uint8)
        ram_copy = dict(raw.ram) if isinstance(raw.ram, Mapping) else raw.ram
        metadata = dict(raw.metadata)
        metadata.setdefault("frame_shape", tuple(int(dim) for dim in framebuffer.shape[:2]))
        return OverworldObservation(
            framebuffer=framebuffer.copy(),
            raw_framebuffer=raw_buffer.copy() if isinstance(raw_buffer, np.ndarray) else raw_buffer,
            ram=ram_copy,
            metadata=metadata,
        )


__all__ = ["OverworldAdapter", "OverworldObservation"]
