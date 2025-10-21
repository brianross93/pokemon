from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch

from src.pkmn_battle.ingest import BattleDecisionDataset
from src.overworld.ingest import OverworldDecisionDataset


@dataclass
class ModeSlice:
    """Container for plan features originating from a single gameplay mode."""

    mode: str
    plan_features: torch.Tensor
    gate_targets: Optional[torch.Tensor] = None
    adherence: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    def count(self) -> int:
        return int(self.plan_features.size(0))

    def summary(self) -> Dict[str, float]:
        stats: Dict[str, float] = {"count": float(self.count())}
        if self.adherence is not None and self.adherence.numel() > 0:
            stats["adherence_mean"] = float(self.adherence.mean().item())
            stats["adherence_positive_frac"] = float((self.adherence > 0.5).float().mean().item())
        if self.gate_targets is not None and self.gate_targets.numel() > 0:
            unique, counts = torch.unique(self.gate_targets, return_counts=True)
            for value, count in zip(unique.tolist(), counts.tolist()):
                stats[f"gate_{int(value)}_frac"] = float(count / max(1, self.count()))
        return stats


@dataclass
class PlanFeatureStore:
    """Aggregated view over battle + overworld plan features for mixed training."""

    slices: Dict[str, ModeSlice]
    metadata: Dict[str, object]

    def sample_weights(self, priors: Mapping[str, float]) -> torch.Tensor:
        weights: List[torch.Tensor] = []
        for mode, slice_ in self.slices.items():
            prior = float(priors.get(mode, 1.0))
            if slice_.count() == 0:
                continue
            weights.append(torch.full((slice_.count(),), prior / slice_.count(), dtype=torch.float32))
        if not weights:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat(weights, dim=0)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"metadata": self.metadata, "modes": {}}
        for mode, slice_ in self.slices.items():
            mode_payload: Dict[str, object] = {
                "plan_features": slice_.plan_features,
            }
            if slice_.gate_targets is not None:
                mode_payload["gate_targets"] = slice_.gate_targets
            if slice_.adherence is not None:
                mode_payload["adherence"] = slice_.adherence
            if slice_.extras:
                mode_payload["extras"] = slice_.extras
            payload["modes"][mode] = mode_payload
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)


def build_plan_feature_store(
    *,
    battle_path: Path,
    overworld_paths: Sequence[Path],
    limit: Optional[int] = None,
) -> PlanFeatureStore:
    battle_dataset = BattleDecisionDataset(battle_path)
    overworld_dataset = OverworldDecisionDataset(overworld_paths)

    battle_slice = _collect_battle_slice(battle_dataset, limit=limit)
    overworld_slice = _collect_overworld_slice(overworld_dataset, limit=limit)

    metadata = {
        "battle_count": battle_slice.count(),
        "overworld_count": overworld_slice.count(),
        "battle_summary": battle_slice.summary(),
        "overworld_summary": overworld_slice.summary(),
        "plan_feature_dim": int(battle_slice.plan_features.size(1)),
        "overworld_plan_feature_dim": int(overworld_slice.plan_features.size(1)),
        "order": ["battle", "overworld"],
    }
    return PlanFeatureStore(
        slices={"battle": battle_slice, "overworld": overworld_slice},
        metadata=metadata,
    )


def _collect_battle_slice(dataset: BattleDecisionDataset, limit: Optional[int]) -> ModeSlice:
    plan_features: List[torch.Tensor] = []
    gate_targets: List[torch.Tensor] = []
    adherence: List[torch.Tensor] = []
    for idx, item in enumerate(dataset.iter_items()):
        if limit is not None and idx >= limit:
            break
        plan_features.append(item.plan_features.unsqueeze(0))
        gate_targets.append(item.gate_target.unsqueeze(0))
        adherence.append(item.adherence_flag.unsqueeze(0))
    if not plan_features:
        empty = torch.zeros((0, dataset.plan_feature_dim), dtype=torch.float32)
        return ModeSlice(mode="battle", plan_features=empty)
    features_tensor = torch.cat(plan_features, dim=0)
    gates_tensor = torch.cat(gate_targets, dim=0).long()
    adherence_tensor = torch.cat(adherence, dim=0).float()
    return ModeSlice(
        mode="battle",
        plan_features=features_tensor,
        gate_targets=gates_tensor,
        adherence=adherence_tensor,
    )


def _collect_overworld_slice(dataset: OverworldDecisionDataset, limit: Optional[int]) -> ModeSlice:
    plan_features: List[torch.Tensor] = []
    encode_flags: List[torch.Tensor] = []
    success_flags: List[torch.Tensor] = []
    for index in range(len(dataset)):
        if limit is not None and index >= limit:
            break
        _, plan, _, encode_flag, view_idx, success_flag = dataset[index]
        plan_features.append(plan.unsqueeze(0))
        encode_flags.append(encode_flag.view(1, -1))
        success_flags.append(success_flag.view(1, -1))
    if not plan_features:
        empty = torch.zeros((0, dataset.plan_feature_dim), dtype=torch.float32)
        return ModeSlice(mode="overworld", plan_features=empty)
    features_tensor = torch.cat(plan_features, dim=0)
    extras: Dict[str, torch.Tensor] = {}
    extras["encode_flag"] = torch.cat(encode_flags, dim=0)
    extras["success_flag"] = torch.cat(success_flags, dim=0)
    return ModeSlice(
        mode="overworld",
        plan_features=features_tensor,
        gate_targets=None,
        adherence=None,
        extras=extras,
    )


__all__ = [
    "ModeSlice",
    "PlanFeatureStore",
    "build_plan_feature_store",
]
