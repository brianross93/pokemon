from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from data.frame_dataset import frame_text_to_tensor


@dataclass
class BattleDecision:
    frame: torch.Tensor
    action_index: int
    action_label: Dict
    options: Dict
    metadata: Dict


class BattleDecisionDataset(Dataset[tuple[torch.Tensor, int]]):
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

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self._items[idx]
        # Cast to float in [0,1] here to avoid repeated conversions in the trainer.
        return item.frame.float() / 255.0, item.action_index

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
            decision = BattleDecision(
                frame=frame_tensor,
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
