"""
CLI for materialising a plan feature store that mixes battle and overworld records.

Example:
    python scripts/build_plan_feature_store.py \
        --battle data/processed/il_gen9ou_train.jsonl \
        --overworld data/overworld/run_*.jsonl \
        --output data/planlets/plan_feature_store.pt \
        --weights-out data/planlets/plan_feature_weights.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import torch

from src.training.plan_feature_store import build_plan_feature_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plan feature store for mixed-mode training.")
    parser.add_argument("--battle", type=Path, required=True, help="Battle JSONL file (BattleDecisionDataset source).")
    parser.add_argument(
        "--overworld",
        type=Path,
        nargs="+",
        required=True,
        help="One or more overworld telemetry JSONL files.",
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

    store = build_plan_feature_store(
        battle_path=args.battle,
        overworld_paths=args.overworld,
        limit=args.limit,
    )
    store.save(args.output)

    priors: Dict[str, float] = {
        "battle": max(0.0, args.battle_prior),
        "overworld": max(0.0, args.overworld_prior),
    }
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


if __name__ == "__main__":
    raise SystemExit(main())
