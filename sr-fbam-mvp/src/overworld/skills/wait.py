"""
WaitSkill emits explicit wait actions for a fixed number of steps.
"""

from __future__ import annotations

from typing import Dict

from .base import BaseSkill, SkillProgress, SkillStatus


class WaitSkill(BaseSkill):
    """Simple skill that idles for a configurable number of steps."""

    name = "WaitSkill"

    def __init__(self) -> None:
        super().__init__()
        self._steps_required = 1
        self._steps_taken = 0

    def on_enter(self, planlet, graph) -> None:  # type: ignore[override]
        super().on_enter(planlet, graph)
        args = getattr(planlet, "args", {}) or {}
        steps = (
            args.get("steps")
            or args.get("ticks")
            or args.get("frames")
            or args.get("duration")
            or getattr(planlet, "timeout_steps", 1)
        )
        try:
            self._steps_required = max(1, int(steps))
        except (TypeError, ValueError):
            self._steps_required = 1
        self._steps_taken = 0
        self.set_planner_hint({"steps": self._steps_required})

    def legal_actions(self, observation: Dict[str, object], graph) -> tuple[Dict[str, object], ...]:  # type: ignore[override]
        return ({"kind": "wait"},)

    def select_action(self, observation: Dict[str, object], graph) -> Dict[str, object]:  # type: ignore[override]
        self._steps_taken += 1
        return {"kind": "wait"}

    def progress(self, graph) -> SkillProgress:  # type: ignore[override]
        if self._steps_taken >= self._steps_required:
            return SkillProgress(status=SkillStatus.SUCCEEDED)
        return SkillProgress(status=SkillStatus.IN_PROGRESS)


__all__ = ["WaitSkill"]

