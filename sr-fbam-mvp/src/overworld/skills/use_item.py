"""
UseItemSkill executes bag interactions with item validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .menu_controller import MenuDrivenSkill
from .neural import BUTTONS
from .base import SkillProgress, SkillStatus


class UseItemSkill(MenuDrivenSkill):
    name = "UseItemSkill"
    item_missing_reason = "ITEM_MISSING"

    def __init__(self) -> None:
        super().__init__()
        self._item: Optional[str] = None
        self.script = ("START", "A", "A", "B")

    def on_enter(self, planlet, graph) -> None:
        super().on_enter(planlet, graph)
        args = getattr(planlet, "args", {}) or {}
        self._item = args.get("item")
        path: List[str] = list(args.get("path") or [])
        if path:
            self.script = tuple(path)
        self._script_index = 0
        self.set_planner_hint({"item": self._item, "path": list(self.script)})

    def legal_actions(self, observation: Dict[str, object], graph: object) -> tuple[Dict[str, object], ...]:
        return tuple(BUTTONS[label] for label in ("START", "SELECT", "UP", "DOWN", "A", "B", "WAIT"))

    def progress(self, graph: object) -> SkillProgress:
        base = super().progress(graph)
        if self._script_index >= len(self.script):
            if self._item and not self._inventory_contains(graph, self._item):
                self._mark_failure(self.item_missing_reason)
                return SkillProgress(status=SkillStatus.STALLED, reason=self.item_missing_reason)
            self._completed = True
            return SkillProgress(status=SkillStatus.SUCCEEDED)
        return base

    def _inventory_contains(self, graph, item_name: str) -> bool:
        assoc_fn = getattr(graph, "assoc", None)
        if assoc_fn is None:
            return False
        for node in assoc_fn(type_="InventoryItem"):
            if node.attributes.get("name") == item_name:
                return True
        return False


__all__ = ["UseItemSkill"]
