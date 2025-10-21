"""
Placeholder EncounterSkill that signals a delegation to the battle agent.
"""

from __future__ import annotations

from typing import Dict

from .base import BaseSkill, SkillProgress, SkillStatus


class EncounterSkill(BaseSkill):
    name = "EncounterSkill"

    ACTION_DELEGATE = {"kind": "delegate", "label": "battle_agent", "meta": {"handler": "battle_agent"}}

    def on_enter(self, planlet, graph) -> None:
        super().on_enter(planlet, graph)
        self.set_planner_hint({"mode": "delegate_to_battle_agent"})

    def legal_actions(self, observation: Dict[str, object], graph: object) -> tuple[Dict[str, object], ...]:
        return (self.ACTION_DELEGATE,)

    def select_action(self, observation: Dict[str, object], graph: object) -> Dict[str, object]:
        action = dict(self.ACTION_DELEGATE)
        meta = dict(action.get("meta") or {})
        meta["phase"] = "handoff"
        action["meta"] = meta
        return action

    def progress(self, graph: object) -> SkillProgress:
        return SkillProgress(status=SkillStatus.IN_PROGRESS)
