"""
Task-specific adapters built on top of the SR-FBAM core.
"""

from __future__ import annotations

from typing import Optional

from .overworld import OverworldExecutor, ExecutorStepResult

try:  # pragma: no cover - optional import, depends on battle stack
    from .battle import SRFBAMBattleAgent, BattleTelemetry  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - graceful fallback for lightweight installs
    SRFBAMBattleAgent = None  # type: ignore[assignment]
    BattleTelemetry = None  # type: ignore[assignment]

__all__ = [
    "SRFBAMBattleAgent",
    "BattleTelemetry",
    "OverworldExecutor",
    "ExecutorStepResult",
]
