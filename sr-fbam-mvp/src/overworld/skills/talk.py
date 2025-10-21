"""
TalkSkill handles scripted NPC interactions.
"""

from __future__ import annotations

from typing import Dict, Optional

from .neural import BUTTONS, NeuralButtonSkill
from .base import SkillProgress, SkillStatus


class TalkSkill(NeuralButtonSkill):
    name = "TalkSkill"
    fallback = "A"
    npc_missing_reason = "NPC_NOT_FOUND"

    def __init__(self) -> None:
        super().__init__()
        self._npc_id: Optional[str] = None
        self.script = ("A", "A", "B")

    def on_enter(self, planlet, graph) -> None:
        super().on_enter(planlet, graph)
        args = getattr(planlet, "args", {}) or {}
        self._npc_id = args.get("npc")
        self._script_index = 0
        hint = {"npc": self._npc_id, "sequence": list(self.script)}
        self.set_planner_hint(hint)

    def legal_actions(self, observation: Dict[str, object], graph: object) -> tuple[Dict[str, object], ...]:
        return tuple(BUTTONS[label] for label in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "WAIT"))

    def select_action(self, observation: Dict[str, object], graph: object) -> Dict[str, object]:
        action = self._resolve_action(self.script, ("A", "B", "UP", "DOWN"))
        meta = dict(action.get("meta") or {})
        meta["npc"] = self._npc_id
        action["meta"] = meta
        return dict(action)

    def progress(self, graph: object) -> SkillProgress:
        if self._failure_reason:
            return SkillProgress(status=SkillStatus.STALLED, reason=self._failure_reason)

        if self._npc_id and not self._npc_present(graph):
            self._mark_failure(self.npc_missing_reason)
            return SkillProgress(status=SkillStatus.STALLED, reason=self.npc_missing_reason)

        if self._script_index >= len(self.script):
            self._completed = True
            return SkillProgress(status=SkillStatus.SUCCEEDED)

        return SkillProgress(status=SkillStatus.IN_PROGRESS)

    def _npc_present(self, graph) -> bool:
        assoc_fn = getattr(graph, "assoc", None)
        if assoc_fn is None or self._npc_id is None:
            return True
        for node in assoc_fn(type_="NPC"):
            if node.attributes.get("name") == self._npc_id or node.node_id == self._npc_id:
                return True
        return False


__all__ = ["TalkSkill"]
