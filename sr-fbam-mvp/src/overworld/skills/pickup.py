"""
PickupSkill presses the interaction button until an inventory delta is observed
or an optional flag condition is satisfied.
"""

from __future__ import annotations

from typing import Dict

from .base import SkillProgress, SkillStatus
from .interact import InteractSkill


class PickupSkill(InteractSkill):
    """Specialised interaction that validates expected inventory changes."""

    name = "PickupSkill"

    def __init__(self) -> None:
        super().__init__()
        self._target_item: str | None = None
        self._inventory_before: Dict[str, int] = {}
        self._timeout_reason = "ITEM_NOT_ACQUIRED"

    def on_enter(self, planlet, graph) -> None:  # type: ignore[override]
        super().on_enter(planlet, graph)
        args = getattr(planlet, "args", {}) or {}
        target = args.get("item") or args.get("item_id") or args.get("name")
        self._target_item = str(target).strip() if target else None
        if not self._target_item:
            self._target_item = None
        self._inventory_before = self._inventory_snapshot(graph)
        hint = {"item": self._target_item} if self._target_item else {}
        if hint:
            existing = self.planner_hint() or {}
            existing.update(hint)
            self.set_planner_hint(existing)

    def progress(self, graph) -> SkillProgress:  # type: ignore[override]
        if self._failure_reason:
            return SkillProgress(status=SkillStatus.STALLED, reason=self._failure_reason)

        if self._pickup_completed(graph):
            self._completed = True
            return SkillProgress(status=SkillStatus.SUCCEEDED)

        if self._step_index >= self._timeout_steps:
            self._mark_failure(self._timeout_reason)
            return SkillProgress(status=SkillStatus.STALLED, reason=self._timeout_reason)

        return SkillProgress(status=SkillStatus.IN_PROGRESS)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _pickup_completed(self, graph) -> bool:
        inventory_after = self._inventory_snapshot(graph)
        resolved = False
        if self._target_item:
            target_key = self._target_item.lower()
            resolved = inventory_after.get(target_key, 0) > self._inventory_before.get(target_key, 0)
        else:
            for name, qty_after in inventory_after.items():
                if qty_after > self._inventory_before.get(name, 0):
                    resolved = True
                    break

        if resolved:
            return self._flags_active(graph) if self._required_flags else True

        if self._required_flags:
            return self._flags_active(graph) and self._step_index >= self._min_presses
        return False

    @staticmethod
    def _inventory_snapshot(graph) -> Dict[str, int]:
        assoc_fn = getattr(graph, "assoc", None)
        snapshot: Dict[str, int] = {}
        if assoc_fn is None:
            return snapshot
        for node in assoc_fn(type_="InventoryItem"):
            name = str(node.attributes.get("name") or node.node_id).lower()
            qty = int(node.attributes.get("quantity", 0))
            snapshot[name] = qty
        return snapshot


__all__ = ["PickupSkill"]
