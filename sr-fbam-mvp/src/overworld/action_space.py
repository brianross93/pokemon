"""
Canonical action space for the overworld executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


BUTTON_ACTIONS: Tuple[Dict[str, object], ...] = (
    {"kind": "button", "label": "UP"},
    {"kind": "button", "label": "DOWN"},
    {"kind": "button", "label": "LEFT"},
    {"kind": "button", "label": "RIGHT"},
    {"kind": "button", "label": "A"},
    {"kind": "button", "label": "B"},
    {"kind": "button", "label": "START"},
    {"kind": "button", "label": "SELECT"},
)

SPECIAL_ACTIONS: Tuple[Dict[str, object], ...] = (
    {"kind": "wait"},
    {"kind": "delegate", "label": "battle_agent"},
)


@dataclass(frozen=True)
class OverworldActionSpace:
    """Enumerates the discrete actions exposed to the executor/policy."""

    button_actions: Tuple[Dict[str, object], ...] = BUTTON_ACTIONS
    special_actions: Tuple[Dict[str, object], ...] = SPECIAL_ACTIONS

    def __post_init__(self) -> None:
        object.__setattr__(self, "_actions", self.button_actions + self.special_actions)
        index_map = {}
        for idx, action in enumerate(self._actions):
            index_map[self._action_key(action)] = idx
        object.__setattr__(self, "_index_map", index_map)

    @property
    def size(self) -> int:
        return len(self._actions)

    @property
    def actions(self) -> Tuple[Dict[str, object], ...]:
        return self._actions

    def to_index(self, action: Dict[str, object]) -> int:
        key = self._action_key(action)
        if key not in self._index_map:
            raise KeyError(f"Unknown action: {action}")
        return self._index_map[key]

    def build_mask(self, legal_actions: Iterable[Dict[str, object]]) -> Tuple[float, ...]:
        mask = [float("-inf")] * self.size
        for action in legal_actions:
            try:
                idx = self.to_index(action)
            except KeyError:
                continue
            mask[idx] = 0.0
        return tuple(mask)

    @staticmethod
    def _action_key(action: Dict[str, object]) -> Tuple[str, str | None]:
        kind = action.get("kind", "")
        label = action.get("label")
        return (str(kind).lower(), str(label).lower() if label is not None else None)

