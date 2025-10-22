from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import torch
from torch.utils.data import Dataset

from src.data.frame_dataset import frame_text_to_tensor
from src.plan.features import PLAN_FEATURE_DIM, build_plan_feature_vector

_GATE_MODE_TO_INDEX = {
    "WRITE": 0,
    "ENCODE": 0,
    "ASSOC": 1,
    "FOLLOW": 2,
    "HALT": 3,
    "PLAN_LOOKUP": 4,
    "PLAN_STEP": 5,
}


@dataclass
class BattleDecision:
    frame: torch.Tensor
    plan_features: torch.Tensor
    gate_target: torch.Tensor
    adherence_flag: torch.Tensor
    action_index: int
    action_label: Dict
    options: Dict
    metadata: Dict


class BattleDecisionDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]
):
    """
    Dataset wrapper for SR-FBAM battle imitation learning JSONL rows.

    Each record contains a 40x120 frame grid (as text), action metadata, options,
    and graph updates. We convert the frame into a uint8 tensor and map the action
    to a discrete index for cross-entropy supervision.
    """

    def __init__(
        self,
        jsonl_path: Path,
        grid_height: int = 40,
        grid_width: int = 120,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.grid_height = grid_height
        self.grid_width = grid_width

        self._items: List[BattleDecision] = []
        self._index_to_action: List[str] = []
        self._action_to_index: Dict[str, int] = {}
        self._plan_feature_dim: int = PLAN_FEATURE_DIM
        self._gate_class_count: int = len(_GATE_MODE_TO_INDEX)
        self._load()

    @property
    def action_to_index(self) -> Dict[str, int]:
        return self._action_to_index

    @property
    def index_to_action(self) -> List[str]:
        return self._index_to_action

    @property
    def num_actions(self) -> int:
        return len(self._index_to_action)

    @property
    def plan_feature_dim(self) -> int:
        return self._plan_feature_dim

    @property
    def num_gate_targets(self) -> int:
        return self._gate_class_count

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        item = self._items[idx]
        # Cast to float in [0,1] here to avoid repeated conversions in the trainer.
        return (
            item.frame.float() / 255.0,
            item.plan_features,
            item.gate_target,
            item.adherence_flag,
            item.action_index,
        )

    def iter_items(self) -> Iterable[BattleDecision]:
        return iter(self._items)

    def _load(self) -> None:
        seen_actions = set()
        raw_records: List[dict] = []
        with self.jsonl_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                record = json.loads(line)
                raw_records.append(record)
                seen_actions.add(_action_key(record["action_label"]))

        self._index_to_action = sorted(seen_actions)
        self._action_to_index = {
            action: idx for idx, action in enumerate(self._index_to_action)
        }

        for record in raw_records:
            frame_lines = record["frame"]["grid_40x120"]
            frame_text = "\n".join(frame_lines)
            frame_tensor = frame_text_to_tensor(
                frame_text,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
            )
            action_key = _action_key(record["action_label"])
            action_index = self._action_to_index[action_key]
            plan_adherence = record.get("plan_adherence")
            plan_features = build_plan_feature_vector(
                plan=record.get("plan"),
                gate=record.get("gate"),
                plan_metrics=record.get("plan_metrics"),
                plan_adherence=plan_adherence,
            )
            gate_target = torch.tensor(
                _gate_mode_to_index(record.get("gate")), dtype=torch.long
            )
            adherence_flag = torch.tensor(
                _extract_adherence_flag(
                    plan_metrics=record.get("plan_metrics"),
                    plan_adherence=plan_adherence,
                ),
                dtype=torch.float32,
            )
            decision = BattleDecision(
                frame=frame_tensor,
                plan_features=torch.tensor(plan_features, dtype=torch.float32),
                gate_target=gate_target,
                adherence_flag=adherence_flag,
                action_index=action_index,
                action_label=record["action_label"],
                options=record.get("options", {}),
                metadata={
                    "battle_id": record.get("battle_id"),
                    "turn_idx": record.get("turn_idx"),
                    "graph_updates": record.get("graph_updates", []),
                    "revealed": record.get("revealed", {}),
                },
            )
            self._items.append(decision)


def _action_key(action_label: Dict) -> str:
    action_type = action_label.get("type", "NONE")
    if action_type == "MOVE":
        move_id = action_label.get("id", "unknown")
        tera_suffix = ":TERA" if action_label.get("tera") else ""
        return f"MOVE:{move_id}{tera_suffix}"
    if action_type == "SWITCH":
        species = action_label.get("species") or action_label.get("id") or "unknown"
        return f"SWITCH:{species}"
    return action_type or "NONE"


def _gate_mode_to_index(gate: Mapping[str, object] | None) -> int:
    if not isinstance(gate, Mapping):
        return 0
    mode = str(gate.get("mode") or gate.get("decision") or "").upper()
    return _GATE_MODE_TO_INDEX.get(mode, 0)


def _extract_adherence_flag(
    *,
    plan_metrics: Mapping[str, object] | None,
    plan_adherence: Mapping[str, object] | float | int | str | None,
) -> float:
    plan_metrics = plan_metrics or {}
    if isinstance(plan_adherence, Mapping):
        status = plan_adherence.get("status") or plan_adherence.get("code")
        if isinstance(status, str):
            lowered = status.lower()
            if lowered in {"adhered", "on_plan", "on_track", "match", "aligned"}:
                return 1.0
            if lowered in {"off_plan", "deviated", "broken", "mismatch", "failed"}:
                return 0.0
        value = plan_adherence.get("score") or plan_adherence.get("fraction") or plan_adherence.get("value")
        try:
            if value is not None:
                return float(max(0.0, min(1.0, float(value))))
        except (TypeError, ValueError):
            pass
    if isinstance(plan_adherence, (int, float)):
        return float(max(0.0, min(1.0, float(plan_adherence))))
    if isinstance(plan_adherence, str):
        lowered = plan_adherence.lower()
        if lowered in {"adhered", "on_plan", "on_track", "match", "aligned"}:
            return 1.0
        if lowered in {"off_plan", "deviated", "broken", "mismatch", "failed"}:
            return 0.0
    adherence_value = plan_metrics.get("adherence") or plan_metrics.get("adherence_score")
    try:
        if adherence_value is not None:
            return float(max(0.0, min(1.0, float(adherence_value))))
    except (TypeError, ValueError):
        return 0.0
    return 0.0
