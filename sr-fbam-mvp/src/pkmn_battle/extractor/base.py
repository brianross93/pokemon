"""
Symbolic entity extraction interfaces for Pokemon battles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from ..env.core import BattleObs
from ..graph.schema import WriteOp


class Extractor(ABC):
    """
    Deterministic extractor that emits WRITE operations for the graph.
    """

    @abstractmethod
    def extract(self, obs: BattleObs) -> Sequence[WriteOp]:
        """Derive symbolic WRITE operations from the current observation."""

    def iter_extract(self, obs: BattleObs) -> Iterable[WriteOp]:
        """Yield WRITE operations one-by-one (default implementation)."""
        return self.extract(obs)

