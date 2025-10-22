from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from src.plan.features import PLAN_FEATURE_DIM, build_plan_feature_vector

VIEW_TO_INDEX = {"typed": 0, "slots": 1}


@dataclass
class OverworldDecision:
    frame_features: torch.Tensor
    plan_features: torch.Tensor
    action_index: int
    encode_flag: torch.Tensor
    view_index: torch.Tensor
    success_flag: torch.Tensor
    menu_state: torch.Tensor


class OverworldDecisionDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    """
    Dataset wrapper that consumes overworld trace JSONL logs.

    Each row corresponds to a single SR-FBAM overworld decision captured in the
    consolidated telemetry schema. The dataset exposes both low-level features
    (frame summary) and plan-aware auxiliary signals so downstream models can
    condition their behaviour on the active planlet.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        self._items: List[OverworldDecision] = []
        for path in paths:
            self._load_file(path)
        if not self._items:
            raise ValueError("Trace dataset contained no valid records.")
        self._frame_feature_dim = self._items[0].frame_features.numel()
        self._plan_feature_dim = self._items[0].plan_features.numel()

    @property
    def frame_feature_dim(self) -> int:
        return self._frame_feature_dim

    @property
    def plan_feature_dim(self) -> int:
        return self._plan_feature_dim

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self._items[index]
        return (
            item.frame_features,
            item.plan_features,
            item.action_index,
            item.encode_flag,
            item.view_index,
            item.success_flag,
            item.menu_state,
        )

    def _load_file(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                decision = _parse_overworld_record(record)
                if decision is not None:
                    self._items.append(decision)


def _parse_overworld_record(record: Mapping[str, object]) -> OverworldDecision | None:
    new_schema = isinstance(record.get("telemetry"), Mapping) and isinstance(record.get("context"), Mapping)

    menu_state = 0
    menu_flag = 0.0

    if new_schema:
        context = record.get("context", {})
        telemetry = record.get("telemetry", {})
        core = telemetry.get("core", {}) if isinstance(telemetry, Mapping) else {}
        overworld = telemetry.get("overworld", {}) if isinstance(telemetry, Mapping) else {}

        frame_features = _ensure_float_list(overworld.get("frame_features"))
        action_index = overworld.get("action_index")
        gate = core.get("gate", {}) if isinstance(core, Mapping) else {}
        if isinstance(overworld, Mapping):
            try:
                menu_state = int(overworld.get("menu_state") or 0)
            except (TypeError, ValueError):
                menu_state = 0
            menu_flag = 1.0 if overworld.get("is_menu") else 0.0
        plan = {}
        if isinstance(context, Mapping):
            plan.update(context.get("plan", {}) or {})
            mode_value = str(context.get("mode") or "").lower()
        else:
            mode_value = ""
        plan.update(core.get("plan", {}) or {})
        plan_metrics = core.get("plan_metrics") if isinstance(core, Mapping) else None
        plan_adherence = core.get("plan_adherence") if isinstance(core, Mapping) else None
        status = overworld.get("status") or (context.get("status") if isinstance(context, Mapping) else "")
    else:
        telemetry = record.get("telemetry", {})
        frame_features = _ensure_float_list(
            telemetry.get("overworld", {}).get("frame_features") if isinstance(telemetry, Mapping) else None
        )
        if not frame_features:
            frame_features = _ensure_float_list(telemetry.get("frame_features"))
        action_index = telemetry.get("overworld", {}).get("action_index") if isinstance(telemetry, Mapping) else None
        if action_index is None:
            action_index = telemetry.get("action_index")
        gate = telemetry.get("gate", {}) if isinstance(telemetry, Mapping) else {}
        plan = {
            "id": record.get("plan_id"),
            "planlet_id": record.get("planlet_id"),
            "planlet_kind": record.get("planlet_kind"),
            "status": record.get("status"),
            "step_index": record.get("step_index"),
            "steps_total": record.get("steps_total"),
        }
        plan_metrics = record.get("plan_metrics")
        plan_adherence = record.get("plan_adherence")
        mode_value = str(telemetry.get("mode") or record.get("mode") or "").lower()
        status = record.get("status")
        menu_state = int(record.get("menu_state", 0) or 0)
        menu_flag = 1.0 if record.get("is_menu") else 0.0

    if not frame_features or action_index is None:
        return None

    plan_features = build_plan_feature_vector(
        plan=plan if isinstance(plan, Mapping) else {},
        gate=gate if isinstance(gate, Mapping) else {},
        plan_metrics=plan_metrics if isinstance(plan_metrics, Mapping) else None,
        plan_adherence=plan_adherence if isinstance(plan_adherence, Mapping) else plan_adherence,
    )
    mode_bit = _derive_mode_bit(mode_value)
    plan_features_with_mode = plan_features + [mode_bit]

    encode_flag = 1.0 if isinstance(gate, Mapping) and gate.get("encode_flag") else 0.0
    view = "typed"
    if isinstance(gate, Mapping):
        view = str(gate.get("view") or "typed")

    success_flag = 1.0 if str(status or "").upper() in {"PLANLET_COMPLETE", "PLAN_COMPLETE"} else 0.0

    return OverworldDecision(
        frame_features=torch.tensor(frame_features, dtype=torch.float32),
        plan_features=torch.tensor(plan_features_with_mode, dtype=torch.float32),
        action_index=int(action_index),
        encode_flag=torch.tensor(encode_flag, dtype=torch.float32),
        view_index=torch.tensor(VIEW_TO_INDEX.get(view, 0), dtype=torch.long),
        success_flag=torch.tensor(success_flag, dtype=torch.float32),
        menu_state=torch.tensor(float(menu_state) if menu_flag else 0.0, dtype=torch.float32),
    )


def _derive_mode_bit(mode_value: str) -> float:
    mode_value = (mode_value or "").lower()
    if mode_value in {"encounter", "battle"}:
        return 1.0
    return 0.0


def _ensure_float_list(value: object) -> List[float]:
    if not isinstance(value, Iterable):
        return []
    result: List[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return []
    return result


__all__ = ["OverworldDecisionDataset", "VIEW_TO_INDEX"]
