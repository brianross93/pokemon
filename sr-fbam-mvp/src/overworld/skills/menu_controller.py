"""
Shared helpers for menu-driven overworld skills.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .base import SkillProgress, SkillStatus
from .neural import BUTTONS, NeuralButtonSkill


class MenuDrivenSkill(NeuralButtonSkill):
    """
    Mixin that coordinates scripted menu paths with simple recovery logic.
    """

    menu_failure_reason = "MENU_DESYNC"
    misdialog_reason = "MISDIALOG"
    recovery_action_label = "B"
    max_recovery_steps = 3

    def __init__(self) -> None:
        super().__init__()
        self._menu_path: List[str] = []
        self._recovering = False
        self._recovery_steps = 0
        self._menu_snapshot: List[str] = []

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _menu_states(self, graph) -> List[Dict[str, object]]:
        states = []
        assoc_fn = getattr(graph, "assoc", None)
        if assoc_fn is None:
            return states
        for node in assoc_fn(type_="MenuState"):
            states.append({"path": list(node.attributes.get("path", [])), "open": bool(node.attributes.get("open"))})
        return states

    def _current_menu_path(self, graph) -> List[str]:
        states = self._menu_states(graph)
        if not states:
            return []
        path = states[-1]["path"]
        self._menu_snapshot = path
        return path

    def _expected_path(self) -> List[str]:
        if self._menu_path:
            return self._menu_path
        if self.script:
            return list(self.script)
        return []

    def _start_recovery(self, reason: str) -> None:
        self._recovering = True
        self._recovery_steps = 0
        self.set_recovery_hint({"reason": reason, "action": "press-B"})

    def _finish_recovery(self) -> None:
        self._recovering = False
        self._recovery_steps = 0
        self.set_recovery_hint(None)

    def _recovery_action(self) -> Dict[str, object]:
        action = dict(BUTTONS.get(self.recovery_action_label, BUTTONS["B"]))
        meta = dict(action.get("meta") or {})
        meta["recovery"] = self.menu_failure_reason
        action["meta"] = meta
        return action

    def _menu_prefix_ok(self, menu_path: Sequence[str]) -> bool:
        expected = self._expected_path()
        if not expected:
            return True
        prefix = expected[: len(menu_path)]
        return list(menu_path) == list(prefix)

    # ------------------------------------------------------------------ #
    # Overrides
    # ------------------------------------------------------------------ #

    def on_enter(self, planlet, graph) -> None:
        super().on_enter(planlet, graph)
        args = getattr(planlet, "args", {}) or {}
        path = list(args.get("path") or [])
        self._menu_path = path
        self._script_index = 0
        self._recovering = False
        self._recovery_steps = 0
        hint = {"path": list(path)} if path else None
        self.set_planner_hint(hint)

    def select_action(self, observation, graph):
        if self._recovering:
            self._recovery_steps += 1
            if self._recovery_steps > self.max_recovery_steps:
                self._mark_failure(self.menu_failure_reason)
            return self._recovery_action()

        current_path = self._current_menu_path(graph)
        expected = self._expected_path()
        if current_path and not self._menu_prefix_ok(current_path):
            self._start_recovery(self.menu_failure_reason)
            return self._recovery_action()

        action = self._resolve_action(expected, ("UP", "DOWN", "LEFT", "RIGHT", "A", "B"))
        meta = dict(action.get("meta") or {})
        meta["expected_path"] = expected
        meta["menu_path"] = current_path
        action["meta"] = meta
        return dict(action)

    def progress(self, graph):
        if self._failure_reason:
            return SkillProgress(status=SkillStatus.STALLED, reason=self._failure_reason)

        if self._recovering:
            # Wait for recovery to finish.
            return SkillProgress(status=SkillStatus.IN_PROGRESS, reason=self.menu_failure_reason)

        expected = self._expected_path()
        current_path = self._current_menu_path(graph)

        if expected:
            if not current_path:
                if self._script_index >= len(expected):
                    self._completed = True
                    return SkillProgress(status=SkillStatus.SUCCEEDED)
                self._mark_failure(self.misdialog_reason)
                return SkillProgress(status=SkillStatus.STALLED, reason=self.misdialog_reason)
            if current_path == expected and self._script_index >= len(expected):
                self._completed = True
                return SkillProgress(status=SkillStatus.SUCCEEDED)

        return SkillProgress(status=SkillStatus.IN_PROGRESS)
