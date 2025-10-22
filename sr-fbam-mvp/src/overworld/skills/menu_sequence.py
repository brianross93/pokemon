"""
Skill that replays a scripted sequence of button presses to navigate menus.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from .base import BaseSkill, SkillProgress, SkillStatus
from .neural import BUTTONS


def _normalise_sequence(sequence: Iterable[str]) -> List[str]:
    buttons: List[str] = []
    for label in sequence:
        if not isinstance(label, str):
            continue
        label = label.upper().strip()
        if label == "":
            continue
        if label not in BUTTONS:
            if label in {"PRESS_A", "CONFIRM"}:
                label = "A"
            elif label in {"PRESS_B", "BACK"}:
                label = "B"
            elif label in {"WAIT"}:
                label = "WAIT"
            elif label in {"PRESS_START", "START_BUTTON"}:
                label = "START"
            elif label in {"PRESS_SELECT"}:
                label = "SELECT"
            elif label in {"UP", "DOWN", "LEFT", "RIGHT"}:
                pass
            else:
                continue
        buttons.append(label)
    return buttons


class MenuSequenceSkill(BaseSkill):
    """Skill that executes a fixed button sequence for menu navigation."""

    name = "MenuSequenceSkill"

    def __init__(self) -> None:
        super().__init__()
        self._sequence: List[str] = []
        self._index = 0
        self._latched_success = False

    def on_enter(self, planlet, graph) -> None:
        super().on_enter(planlet, graph)
        args: Mapping[str, object] = getattr(planlet, "args", {}) or {}
        sequence = args.get("buttons") or args.get("sequence") or ()
        if isinstance(sequence, str):
            sequence = sequence.split(",")
        self._sequence = _normalise_sequence(sequence if isinstance(sequence, Sequence) else [])
        if not self._sequence:
            self._sequence = ["A"]
        self._index = 0
        self._latched_success = False
        self.set_planner_hint({"sequence": list(self._sequence)})

    def legal_actions(self, observation: Dict[str, object], graph: object) -> tuple[Dict[str, object], ...]:
        return tuple(BUTTONS.values())

    def select_action(self, observation: Dict[str, object], graph: object) -> Dict[str, object]:
        if self._index < len(self._sequence):
            label = self._sequence[self._index]
            self._index += 1
            return dict(BUTTONS[label])
        self._latched_success = True
        return dict(BUTTONS["WAIT"])

    def progress(self, graph: object) -> SkillProgress:
        if self._latched_success:
            return SkillProgress(status=SkillStatus.SUCCEEDED)
        return SkillProgress(status=SkillStatus.IN_PROGRESS)

    def on_exit(self, graph: object) -> None:
        super().on_exit(graph)
        self._sequence = []
        self._index = 0
        self._latched_success = False


__all__ = ["MenuSequenceSkill"]
