"""
Base skill interfaces for overworld execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional


class SkillStatus(Enum):
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    SUCCEEDED = auto()
    STALLED = auto()


@dataclass
class SkillProgress:
    status: SkillStatus
    reason: str | None = None


class BaseSkill:
    """Minimal interface shared across overworld skills."""

    name: str = "BaseSkill"

    def __init__(self) -> None:
        self.planlet: Optional[Any] = None
        self._summary: Optional[Any] = None
        self._memory: Optional[Any] = None
        self._slot_bank: Optional[Any] = None
        self._planner_hint: Optional[Dict[str, Any]] = None
        self._recovery_hint: Optional[Dict[str, Any]] = None

    def on_enter(self, planlet: Any, graph: Any) -> None:
        """Called once when the planlet becomes active."""
        self.planlet = planlet
        self._planner_hint = None
        self._recovery_hint = None

    def legal_actions(self, observation: Dict[str, Any], graph: Any) -> tuple[Dict[str, Any], ...]:
        """Return the set of actions the skill may issue at this step."""
        return ({"kind": "wait"},)

    def select_action(self, observation: Dict[str, Any], graph: Any) -> Dict[str, Any]:
        """Return a low-level action dictionary understood by the adapter."""
        return {"kind": "wait", "meta": {}}

    def progress(self, graph: Any) -> SkillProgress:
        return SkillProgress(status=SkillStatus.IN_PROGRESS)

    def on_exit(self, graph: Any) -> None:
        """Called when the planlet completes or is cancelled."""
        self._planner_hint = None
        self._recovery_hint = None

    # ------------------------------------------------------------------ #
    # Context helpers
    # ------------------------------------------------------------------ #

    def update_context(
        self,
        *,
        summary: Optional[Any] = None,
        memory: Optional[Any] = None,
        slot_bank: Optional[Any] = None,
    ) -> None:
        """Provide the latest SR-FBAM summary and symbolic context to the skill."""

        self._summary = summary
        self._memory = memory
        self._slot_bank = slot_bank

    def summary(self) -> Optional[Any]:
        return self._summary

    def memory(self) -> Optional[Any]:
        return self._memory

    def slot_bank(self) -> Optional[Any]:
        return self._slot_bank

    def set_planner_hint(self, hint: Optional[Dict[str, Any]]) -> None:
        self._planner_hint = hint

    def planner_hint(self) -> Optional[Dict[str, Any]]:
        return self._planner_hint

    def set_recovery_hint(self, hint: Optional[Dict[str, Any]]) -> None:
        self._recovery_hint = hint

    def recovery_hint(self) -> Optional[Dict[str, Any]]:
        return self._recovery_hint
