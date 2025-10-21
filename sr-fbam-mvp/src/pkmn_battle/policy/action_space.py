"""
Action-space utilities for Pokemon battles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from src.pkmn_battle.env import LegalAction


@dataclass(frozen=True)
class ActionSpace:
    """
    Maps battle actions onto contiguous indices for policy heads.
    """

    max_moves: int = 4
    max_switches: int = 6

    @property
    def size(self) -> int:
        return self.max_moves + self.max_switches + 1  # +1 for forfeit/noop

    @property
    def forfeit_index(self) -> int:
        return self.size - 1

    def to_index(self, action: LegalAction) -> int:
        kind = action["kind"]
        idx = int(action["index"])
        if kind == "move":
            if idx >= self.max_moves:
                raise ValueError(f"Move index {idx} exceeds max_moves={self.max_moves}")
            return idx
        if kind == "switch":
            if idx >= self.max_switches:
                raise ValueError(f"Switch index {idx} exceeds max_switches={self.max_switches}")
            return self.max_moves + idx
        if kind == "forfeit":
            return self.forfeit_index
        raise ValueError(f"Unknown action kind '{kind}'")

    def build_mask(self, legal_actions: Sequence[LegalAction]) -> Tuple[Tuple[float, ...], Dict[int, LegalAction]]:
        mask = [float("-inf")] * self.size
        index_map: Dict[int, LegalAction] = {}
        for action in legal_actions:
            idx = self.to_index(action)
            mask[idx] = 0.0
            index_map[idx] = action
        return tuple(mask), index_map

