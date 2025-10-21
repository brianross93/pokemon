"""Simple plan progress monitor for overworld execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.overworld.graph.overworld_memory import OverworldMemory
from src.overworld.skills.navigate import NavigateSkill


@dataclass
class MonitorRecord:
    last_distance: Optional[int] = None
    steps_without_progress: int = 0
    recovery_attempts: int = 0


class PlanMonitor:
    """Tracks planlet progress and suggests recovery actions."""

    def __init__(self, stall_threshold: int = 12) -> None:
        self._records: Dict[str, MonitorRecord] = {}
        self._stall_threshold = max(1, stall_threshold)

    def reset(self) -> None:
        self._records.clear()

    def notify_new_planlet(self, planlet_id: str) -> None:
        self._records[planlet_id] = MonitorRecord()

    def clear_planlet(self, planlet_id: str) -> None:
        self._records.pop(planlet_id, None)

    def record_step(
        self,
        planlet_id: str,
        skill: object,
        memory: OverworldMemory,
    ) -> Optional[str]:
        if planlet_id not in self._records:
            self._records[planlet_id] = MonitorRecord()
        record = self._records[planlet_id]

        if isinstance(skill, NavigateSkill) and skill.target is not None:
            player_coords = memory.player_coords()
            if player_coords is None:
                return None
            distance = memory.manhattan_distance(player_coords, tuple(skill.target))
            if record.last_distance is None or distance < record.last_distance:
                record.steps_without_progress = 0
            else:
                record.steps_without_progress += 1
            record.last_distance = distance

            if record.steps_without_progress >= self._stall_threshold:
                record.steps_without_progress = 0
                record.recovery_attempts += 1
                return "BLOCKED_PATH"
        return None

    def recovery_count(self, planlet_id: str) -> int:
        record = self._records.get(planlet_id)
        return record.recovery_attempts if record else 0
