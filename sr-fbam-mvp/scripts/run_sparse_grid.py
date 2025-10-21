"""
Utility to launch FBAM sparse-memory training/evaluation sweeps.

For each (k, seed) combination the script:
  1. Trains FBAM with sparse retrieval memory and saves a checkpoint.
  2. Evaluates the checkpoint across one or more datasets.
  3. Writes aggregated metrics plus optional per-episode/profiling logs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import faiss
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.evaluate_code_model import compute_metrics  # noqa: E402
from src.training.train_fbam_sparse import (  # noqa: E402
    set_seed as set_sparse_seed,
    train_fbam_sparse,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse-memory sweeps across k values and seeds.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Training dataset directory (with train/eval JSONL).")
    parser.add_argument(
        "--eval-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Evaluation dataset directories (default: use --data-dir only).",
    )
    parser.add_argument("--k-values", type=int, nargs="+", required=True, help="k-nearest neighbours to evaluate (e.g. 5 10 20).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds to run.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per run.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--memory-slots", type=int, default=500, help="Number of memory slots.")
    parser.add_argument("--memory-dim", type=int, default=128, help="Dimensionality of memory embeddings.")
    parser.add_argument("--index-desc", type=str, default="Flat", help="FAISS index description string.")
    parser.add_argument("--min-write-gate", type=float, default=1e-3, help="Minimum gate value before writing to memory.")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/sparse_grid"), help="Root directory for outputs.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device override for training/eval.")
    parser.add_argument("--eval-split", choices=["train", "eval"], default="eval", help="Dataset split to evaluate.")
    parser.add_argument("--profile-latency", action="store_true", help="Collect profiler stats during evaluation (if supported).")
    parser.add_argument("--no-per-episode", action="store_true", help="Skip writing per-episode metrics.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip training/evaluation when outputs already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {args.data_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if args.device == "cuda" and faiss.get_num_gpus() == 0:
        print("WARNING: CUDA requested but FAISS reports zero GPUs. Using CPU FAISS with GPU PyTorch.")
        # Allow CPU FAISS with GPU PyTorch - this is a valid configuration

    eval_dirs: Iterable[Path] = args.eval_dirs or [args.data_dir]
    eval_dirs = [path if isinstance(path, Path) else Path(path) for path in eval_dirs]
    for path in eval_dirs:
        if not path.exists():
            raise FileNotFoundError(f"Evaluation data directory not found: {path}")
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    collect_episode = args.profile_latency or not args.no_per_episode
    train_device = torch.device(args.device)
    eval_device = train_device

    k_values = sorted(set(args.k_values))
    seeds = sorted(set(args.seeds))

    for k in k_values:
        for seed in seeds:
            print(f"[info] === k={k} | seed={seed} ===")

            checkpoint_path = output_root / "checkpoints" / f"k{k}" / f"seed_{seed}.pt"
            training_metrics_path = output_root / "training" / f"k{k}" / f"seed_{seed}.json"

            if checkpoint_path.exists() and args.skip_existing:
                print(f"[info] checkpoint exists, skipping training: {checkpoint_path}")
            else:
                print("[info] training sparse model...")
                set_sparse_seed(seed)
                train_metrics, eval_metrics, _ = train_fbam_sparse(
                    data_dir=args.data_dir,
                    epochs=args.epochs,
                    learning_rate=args.lr,
                    device=train_device,
                    memory_slots=args.memory_slots,
                    memory_dim=args.memory_dim,
                    k_neighbors=k,
                    index_desc=args.index_desc,
                    min_write_gate=args.min_write_gate,
                    checkpoint_out=checkpoint_path,
                )
                payload = {
                    "seed": seed,
                    "k_neighbors": k,
                    "memory_slots": args.memory_slots,
                    "memory_dim": args.memory_dim,
                    "epochs": args.epochs,
                    "learning_rate": args.lr,
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                    "checkpoint": str(checkpoint_path),
                }
                _ensure_parent(training_metrics_path)
                training_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            if not checkpoint_path.exists():
                print(f"[warn] checkpoint missing, skipping evaluation: {checkpoint_path}")
                continue

            for eval_dir in eval_dirs:
                eval_name = eval_dir.name
                metrics_path = output_root / "eval" / eval_name / f"k{k}" / f"seed_{seed}.json"
                per_episode_path = output_root / "eval" / eval_name / f"k{k}" / f"seed_{seed}_episodes.json"

                if metrics_path.exists() and args.skip_existing and (not collect_episode or per_episode_path.exists()):
                    print(f"[info] metrics already exist for {eval_name}, skipping.")
                    continue

                print(f"[info] evaluating on {eval_name}...")
                metrics, per_episode = compute_metrics(
                    "fbam_sparse",
                    eval_dir,
                    checkpoint_path,
                    eval_device,
                    memory_slots=args.memory_slots,
                    memory_dim=args.memory_dim,
                    k_neighbors=k,
                    index_desc=args.index_desc,
                    min_write_gate=args.min_write_gate,
                    eval_split=args.eval_split,
                    profile_latency=args.profile_latency,
                    collect_episode_metrics=collect_episode,
                )
                metrics["seed"] = seed
                metrics["k_neighbors"] = k
                metrics["memory_slots"] = args.memory_slots
                metrics["memory_dim"] = args.memory_dim
                metrics["index_desc"] = args.index_desc
                metrics["eval_dir"] = str(eval_dir)

                _ensure_parent(metrics_path)
                metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

                if collect_episode:
                    _ensure_parent(per_episode_path)
                    per_episode_payload = [
                        {
                            **entry,
                            "seed": seed,
                            "k_neighbors": k,
                            "eval_dir": str(eval_dir),
                        }
                        for entry in per_episode
                    ]
                    per_episode_path.write_text(json.dumps(per_episode_payload, indent=2), encoding="utf-8")

    print("[info] sweep complete.")


if __name__ == "__main__":
    main()
