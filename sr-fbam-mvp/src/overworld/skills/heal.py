"""
HealSkill drives Pokémon Center interactions using the SR-FBAM embedding.
"""

from __future__ import annotations

from typing import Dict

from .menu_controller import MenuDrivenSkill
from .neural import BUTTONS
from .base import SkillProgress, SkillStatus


class HealSkill(MenuDrivenSkill):
    name = "HealSkill"
    script = ("A", "A", "A", "WAIT")
    menu_failure_reason = "MENU_DESYNC"
    misdialog_reason = "MISDIALOG"

    def legal_actions(self, observation: Dict[str, object], graph: object) -> tuple[Dict[str, object], ...]:
        return tuple(BUTTONS[label] for label in ("A", "B", "UP", "DOWN", "WAIT"))

    def progress(self, graph: object) -> SkillProgress:
        result = super().progress(graph)
        if result.status is SkillStatus.IN_PROGRESS:
            # Treat repeated menu desync as failure if the gate keeps extracting.
            summary = self.summary()
            if summary is not None and summary.gate_stats.get("decision") == "EXTRACT":
                self._mark_failure("MENU_DESYNC")
                return SkillProgress(status=SkillStatus.STALLED, reason="MENU_DESYNC")
        return result


__all__ = ["HealSkill"]
