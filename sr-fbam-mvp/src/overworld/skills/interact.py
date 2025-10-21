"""
InteractSkill issues contextual interaction inputs (typically button `A` presses)
until a flag, menu state, or timeout condition triggers.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .base import SkillProgress, SkillStatus
from .neural import BUTTONS, NeuralButtonSkill


class InteractSkill(NeuralButtonSkill):
    """
    Lightweight interaction skill used for generic object/NPC interactions.
    """

    name = "InteractSkill"
    script: Sequence[str] = ("A", "A", "A")
    default_timeout_steps = 6

    def __init__(self) -> None:
        super().__init__()
        self._required_flags: List[str] = []
        self._menu_path: Optional[List[str]] = None
        self._min_presses = 1
        self._timeout_steps = self.default_timeout_steps
        self._timeout_reason = "INTERACT_TIMEOUT"

    # ------------------------------------------------------------------ #
    # BaseSkill overrides
    # ------------------------------------------------------------------ #

    def on_enter(self, planlet, graph) -> None:  # type: ignore[override]
        super().on_enter(planlet, graph)
        args = getattr(planlet, "args", {}) or {}

        flag_value = args.get("flag") or args.get("flags")
        if isinstance(flag_value, str):
            self._required_flags = [flag_value]
        elif isinstance(flag_value, Iterable):
            self._required_flags = [str(flag).strip() for flag in flag_value if str(flag).strip()]
        else:
            self._required_flags = []

        menu_value = args.get("menu_path") or args.get("menu")
        if isinstance(menu_value, str):
            self._menu_path = [menu_value]
        elif isinstance(menu_value, Iterable):
            self._menu_path = [str(part) for part in menu_value]
        else:
            self._menu_path = None

        self._min_presses = max(1, int(args.get("presses") or self._min_presses))

        timeout = args.get("timeout_steps")
        if timeout is None and hasattr(planlet, "timeout_steps"):
            timeout = getattr(planlet, "timeout_steps")
        if isinstance(timeout, (int, float)):
            self._timeout_steps = max(1, int(timeout))
        else:
            self._timeout_steps = self.default_timeout_steps

        self._timeout_reason = str(args.get("timeout_reason") or self._timeout_reason)
        self.set_planner_hint(
            {
                "flags": list(self._required_flags),
                "menu_path": list(self._menu_path) if self._menu_path else None,
                "presses": self._min_presses,
            }
        )
        self._completed = False
        self._failure_reason = None
        self._step_index = 0
        self._script_index = 0

    def select_action(self, observation: Dict[str, object], graph) -> Dict[str, object]:  # type: ignore[override]
        self._step_index += 1
        action = self._resolve_action(self.script or ("A",), ("A", "B", "WAIT"))
        meta = dict(action.get("meta") or {})
        if self._required_flags:
            meta.setdefault("flags", list(self._required_flags))
        if self._menu_path:
            meta.setdefault("menu_path", list(self._menu_path))
        meta.setdefault("interaction", True)
        action["meta"] = meta
        return dict(action)

    def progress(self, graph) -> SkillProgress:  # type: ignore[override]
        if self._failure_reason:
            return SkillProgress(status=SkillStatus.STALLED, reason=self._failure_reason)

        if self._interaction_resolved(graph):
            self._completed = True
            return SkillProgress(status=SkillStatus.SUCCEEDED)

        if self._step_index >= self._timeout_steps:
            self._mark_failure(self._timeout_reason)
            return SkillProgress(status=SkillStatus.STALLED, reason=self._timeout_reason)

        return SkillProgress(status=SkillStatus.IN_PROGRESS)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _interaction_resolved(self, graph) -> bool:
        if self._required_flags and not self._flags_active(graph):
            return False
        if self._menu_path and not self._menu_is_open(graph, self._menu_path):
            return False
        return self._step_index >= self._min_presses

    def _flags_active(self, graph) -> bool:
        assoc_fn = getattr(graph, "assoc", None)
        if assoc_fn is None:
            return False
        for flag in self._required_flags:
            if not flag:
                continue
            nodes = assoc_fn(type_="Flag", key=f"flag:{flag}")
            if not nodes:
                nodes = assoc_fn(type_="Flag", filters={"name": flag})
            if not nodes:
                return False
        return True

    def _menu_is_open(self, graph, expected_path: Sequence[str]) -> bool:
        assoc_fn = getattr(graph, "assoc", None)
        if assoc_fn is None:
            return False
        for node in assoc_fn(type_="MenuState"):
            path = node.attributes.get("path") or []
            if list(path) == list(expected_path) and bool(node.attributes.get("open", False)):
                return True
        return False


__all__ = ["InteractSkill"]

