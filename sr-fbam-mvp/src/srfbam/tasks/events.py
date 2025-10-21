"""
Event primitives for coordination between the overworld executor and planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PlanletEvent:
    """
    Snapshot describing the outcome of a planlet execution.

    Attributes
    ----------
    plan_id:
        Identifier of the active plan (if any).
    planlet_id:
        Identifier supplied by the planner.
    planlet_kind:
        Planlet kind (e.g. ``NAVIGATE_TO``).
    status:
        Outcome string emitted by the executor (``PLANLET_COMPLETE`` or ``PLANLET_STALLED``).
    reason:
        Optional failure taxonomy (``BLOCKED_PATH``, ``MENU_DESYNC``, ``RANDOM_BATTLE``, …).
    step_index:
        Global executor step index when the event was generated.
    telemetry:
        Last-step telemetry dictionary recorded by the executor.
    trace:
        Short history (most-recent-first) of telemetry snapshots preceding the event.
    """

    plan_id: Optional[str]
    planlet_id: str
    planlet_kind: str
    status: str
    reason: Optional[str]
    step_index: int
    telemetry: Dict[str, object]
    trace: List[Dict[str, object]] = field(default_factory=list)


__all__ = ["PlanletEvent"]
