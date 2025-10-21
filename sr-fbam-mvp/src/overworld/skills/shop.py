"""
ShopSkill purchases items with basic recovery and failure diagnostics.
"""

from __future__ import annotations

from typing import Dict, List

from .menu_controller import MenuDrivenSkill
from .neural import BUTTONS
from .base import SkillProgress, SkillStatus


class ShopSkill(MenuDrivenSkill):
    name = "ShopSkill"
    insufficient_reason = "INSUFFICIENT_FUNDS"

    def __init__(self) -> None:
        super().__init__()
        self._item: str = "unknown"
        self._qty: int = 1

    def on_enter(self, planlet, graph) -> None:
        super().on_enter(planlet, graph)
        args = getattr(planlet, "args", {}) or {}
        self._item = str(args.get("item", "unknown"))
        self._qty = max(1, int(args.get("qty", 1)))

        # Simple scripted flow: confirm dialogue, select item, choose quantity, exit.
        script: List[str] = ["A"]  # open shop dialogue
        # Simulate menu navigation by stepping down qty times.
        script.extend(["DOWN"] * max(0, self._qty - 1))
        script.extend(["A", "A"])  # confirm item, confirm quantity
        script.append("B")  # exit dialogue
        self.script = tuple(script)
        self._script_index = 0
        self.set_planner_hint({"item": self._item, "qty": self._qty, "path": list(self.script)})

    def legal_actions(self, observation: Dict[str, object], graph: object) -> tuple[Dict[str, object], ...]:
        return tuple(BUTTONS[label] for label in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "WAIT"))

    def progress(self, graph) -> SkillProgress:
        base = super().progress(graph)
        if self._script_index >= len(self.script):
            if not self._inventory_contains(graph, self._item):
                self._mark_failure(self.insufficient_reason)
                return SkillProgress(status=SkillStatus.STALLED, reason=self.insufficient_reason)
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


__all__ = ["ShopSkill"]
