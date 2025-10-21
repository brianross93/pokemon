"""
Lightweight slot bank that stores learned entity embeddings alongside metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor


@dataclass
class Slot:
    vector: Tensor
    confidence: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)


class SlotBank:
    """
    Stores inferred entity slots for hybrid symbolic + learned memory usage.

    Slots are kept on CPU for easy inspection; callers can move tensors back to
    device as needed. Metadata is intentionally free-form.
    """

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self._slots: List[Slot] = []

    def add_slot(self, vector: Tensor, *, confidence: float = 1.0, metadata: Optional[Dict[str, object]] = None) -> None:
        slot_tensor = vector.detach().to("cpu")
        meta = dict(metadata or {})
        self._slots.append(Slot(vector=slot_tensor, confidence=float(confidence), metadata=meta))

    def add_from_summary(self, summary, *, confidence: float = 1.0, metadata: Optional[Dict[str, object]] = None) -> None:
        vector = summary.symbol_embedding
        if vector.dim() == 0:
            vector = vector.unsqueeze(0)
        self.add_slot(vector, confidence=confidence, metadata=metadata)

    def clear(self) -> None:
        self._slots.clear()

    def latest(self) -> Optional[Slot]:
        return self._slots[-1] if self._slots else None

    def slots(self) -> Sequence[Slot]:
        """Return a shallow copy of the current slot list."""

        return tuple(self._slots)

    def contains_metadata(self, key: str, value: object) -> bool:
        """Return ``True`` when any slot records ``metadata[key] == value``."""

        return any(slot.metadata.get(key) == value for slot in self._slots)

    def match_metadata(self, key: str, value: object) -> Iterable[Slot]:
        for slot in self._slots:
            if slot.metadata.get(key) == value:
                yield slot

    def tensors(self) -> Tensor:
        if not self._slots:
            return torch.empty(0)
        vectors = [slot.vector for slot in self._slots]
        return torch.stack(vectors, dim=0)

    def metadata(self) -> List[Dict[str, object]]:
        return [dict(slot.metadata) for slot in self._slots]

    def __len__(self) -> int:
        return len(self._slots)


__all__ = ["Slot", "SlotBank"]
