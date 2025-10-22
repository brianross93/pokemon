"""
CLI for materialising a plan feature store that mixes battle and overworld records.

Examples:
    # Explicit file list
    python scripts/build_plan_feature_store.py \
        --battle data/processed/il_gen9ou_train.jsonl \
        --overworld data/overworld/run_*.jsonl \
        --output data/planlets/plan_feature_store.pt \
        --weights-out data/planlets/plan_feature_weights.pt

    # Config-driven (uses configs/train_plan.yaml when --overworld omitted)
    python scripts/build_plan_feature_store.py \
        --battle data/processed/il_gen9ou_train.jsonl \
        --config configs/train_plan.yaml \
        --output data/planlets/plan_feature_store.pt \
        --weights-out data/planlets/plan_feature_weights.pt \
        --summary-out data/planlets/plan_feature_store_summary.json
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.training.plan_feature_store import build_plan_feature_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plan feature store for mixed-mode training.")
    parser.add_argument("--battle", type=Path, required=True, help="Battle JSONL file (BattleDecisionDataset source).")
    parser.add_argument(
        "--overworld",
        type=Path,
        nargs="+",
        help="One or more overworld telemetry JSONL files. If omitted, sources are read from --config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .pt file that will contain the feature store.",
    )
    parser.add_argument(
        "--weights-out",
        type=Path,
        help="Optional .pt file where per-sample sampling weights will be written.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional JSON file capturing summary statistics for dashboards.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML curriculum configuration (defaults to configs/train_plan.yaml).",
    )
    parser.add_argument(
        "--battle-prior",
        type=float,
        default=0.5,
        help="Sampling prior allocated to battle records (default: 0.5).",
    )
    parser.add_argument(
        "--overworld-prior",
        type=float,
        default=0.5,
        help="Sampling prior allocated to overworld records (default: 0.5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of examples per mode for dry runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_path = _load_config(args.config)

    overworld_paths = _resolve_overworld_paths(args.overworld, config)
    if not overworld_paths:
        raise ValueError("No overworld telemetry files found. Provide --overworld or configure telemetry sources.")

    priors, schedule_metadata = _resolve_priors(
        battle_prior=args.battle_prior,
        overworld_prior=args.overworld_prior,
        config=config,
    )

    metadata_overrides: Dict[str, object] = {}
    if config:
        metadata_overrides["curriculum_config"] = {
            "path": str(config_path) if config_path else None,
            "schedule": schedule_metadata.get("schedule"),
            "stage_used": schedule_metadata.get("stage_used"),
            "gating_heuristics": config.get("gating_heuristics"),
        }
        metadata_overrides["overworld_sources"] = schedule_metadata.get("overworld_sources")

    store = build_plan_feature_store(
        battle_path=args.battle,
        overworld_paths=overworld_paths,
        limit=args.limit,
    )
    store.metadata.update(metadata_overrides)
    store.save(args.output)

    if args.weights_out:
        weights = store.sample_weights(priors)
        args.weights_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": weights, "priors": priors, "order": store.metadata.get("order")}, args.weights_out)

    if args.summary_out:
        summary = dict(store.metadata)
        summary["priors"] = priors
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))

    return 0


def _load_config(path: Optional[Path]) -> Tuple[Mapping[str, object], Optional[Path]]:
    if path is None:
        default_path = PROJECT_ROOT / "configs" / "train_plan.yaml"
        if default_path.exists():
            path = default_path
        else:
            return {}, None
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config, path


def _resolve_overworld_paths(
    overworld_args: Optional[Sequence[Path]],
    config: Mapping[str, object],
) -> List[Path]:
    candidates: List[str] = []
    if overworld_args:
        candidates.extend(str(item) for item in overworld_args)
    elif config:
        telemetry_sources = []
        overworld_req = config.get("overworld_requirements")
        if isinstance(overworld_req, Mapping):
            telemetry_sources = overworld_req.get("telemetry_sources", []) or []
        candidates.extend(str(pattern) for pattern in telemetry_sources)

    paths: List[Path] = []
    seen = set()
    for candidate in candidates:
        expanded = glob(candidate)
        if not expanded:
            expanded = [candidate]
        for match in expanded:
            path = Path(match)
            if path.exists() and path.suffix.lower() == ".jsonl":
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(path)
    return paths


def _resolve_priors(
    *,
    battle_prior: float,
    overworld_prior: float,
    config: Mapping[str, object],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    priors = {
        "battle": max(0.0, battle_prior),
        "overworld": max(0.0, overworld_prior),
    }
    schedule_meta: Dict[str, object] = {
        "schedule": None,
        "stage_used": None,
        "overworld_sources": None,
    }

    mix_cfg = config.get("mix") if isinstance(config, Mapping) else None
    schedule = mix_cfg.get("schedule") if isinstance(mix_cfg, Mapping) else None
    if schedule:
        schedule_meta["schedule"] = schedule
        stage = schedule[-1]
        schedule_meta["stage_used"] = stage
        battle = float(stage.get("battle_fraction", priors["battle"]))
        overworld = float(stage.get("overworld_fraction", priors["overworld"]))
        total = max(battle + overworld, 1e-9)
        priors["battle"] = battle / total
        priors["overworld"] = overworld / total

    overworld_sources = []
    overworld_cfg = config.get("overworld_requirements") if isinstance(config, Mapping) else None
    if isinstance(overworld_cfg, Mapping):
        overworld_sources = overworld_cfg.get("telemetry_sources", []) or []
    if overworld_sources:
        schedule_meta["overworld_sources"] = overworld_sources

    return priors, schedule_meta


if __name__ == "__main__":
    raise SystemExit(main())
