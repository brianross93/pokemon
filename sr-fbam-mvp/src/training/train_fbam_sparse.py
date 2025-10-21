"""
Training harness for FBAM with sparse FAISS-based external memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import optim

from src.data.frame_dataset import FrameActionDataset, load_datasets
from src.models.fbam_sparse_memory import FBAMSparseMemoryAgent, FBAMSparseMemoryConfig


def _episode_tensors(episode) -> Tuple[list, torch.Tensor]:
    frames = [step.frame for step in episode.steps]
    actions = torch.tensor([step.action_index for step in episode.steps], dtype=torch.long)
    return frames, actions


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: FBAMSparseMemoryAgent,
    dataset: FrameActionDataset,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    wall_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for episode in dataset.episodes:
        frames, targets = _episode_tensors(episode)
        frames = [frame.to(device) for frame in frames]
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        start = perf_counter()
        logits = model.forward_episode(frames, targets)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        wall_times.append((perf_counter() - start) * 1000.0)

        total_loss += loss.item() * targets.size(0)
        total_steps += targets.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()

    avg_loss = total_loss / max(total_steps, 1)
    accuracy = correct / max(total_steps, 1)
    wall_time_ms = mean(wall_times) if wall_times else 0.0
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "wall_time_ms": wall_time_ms,
        "peak_mem_mb": peak_mem_mb,
    }


@torch.no_grad()
def evaluate(
    model: FBAMSparseMemoryAgent,
    dataset: FrameActionDataset,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    wall_times = []

    for episode in dataset.episodes:
        frames, targets = _episode_tensors(episode)
        frames = [frame.to(device) for frame in frames]
        targets = targets.to(device)

        start = perf_counter()
        preds = model.predict_actions(frames)
        wall_times.append((perf_counter() - start) * 1000.0)

        logits = model.forward_episode(frames, targets)
        loss = F.cross_entropy(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total_steps += targets.size(0)
        correct += (preds.cpu() == targets.cpu()).sum().item()

    avg_loss = total_loss / max(total_steps, 1)
    accuracy = correct / max(total_steps, 1)
    wall_time_ms = mean(wall_times) if wall_times else 0.0
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "wall_time_ms": wall_time_ms,
    }


def build_model(num_actions: int, args: argparse.Namespace) -> FBAMSparseMemoryAgent:
    config = FBAMSparseMemoryConfig(
        memory_slots=args.memory_slots,
        memory_dim=args.memory_dim,
        k_neighbors=args.k_neighbors,
        index_description=args.index_desc,
        min_write_gate=args.min_write_gate,
    )
    return FBAMSparseMemoryAgent(num_actions=num_actions, config=config)


def train_fbam_sparse(
    data_dir: Path,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    memory_slots: int = 500,
    memory_dim: int = 128,
    k_neighbors: int = 10,
    index_desc: str = "Flat",
    min_write_gate: float = 1e-3,
    checkpoint_out: Path | None = None,
) -> tuple[Dict[str, float], Dict[str, float], FBAMSparseMemoryAgent]:
    train_dataset, eval_dataset = load_datasets(data_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = argparse.Namespace(
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        k_neighbors=k_neighbors,
        index_desc=index_desc,
        min_write_gate=min_write_gate,
    )
    model = build_model(train_dataset.num_actions, args)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(
        f"FBAM+SparseMem parameters: {model.parameter_count():,} | "
        f"Actions: {train_dataset.num_actions} | Device: {device} | "
        f"slots={memory_slots} k={k_neighbors} index={index_desc}"
    )

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_dataset, optimizer, device)
        eval_metrics = evaluate(model, eval_dataset, device)
        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"train_wall={train_metrics['wall_time_ms']:.1f}ms "
            f"train_peak_mem={train_metrics['peak_mem_mb']:.1f}MB | "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.3f} "
            f"eval_wall={eval_metrics['wall_time_ms']:.1f}ms"
        )

    if checkpoint_out:
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        config_dict = dict(model.config.__dict__)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config_dict,
                "num_actions": train_dataset.num_actions,
            },
            checkpoint_out,
        )
        print(f"[info] checkpoint saved to {checkpoint_out}")

    return train_metrics, eval_metrics, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FBAM with sparse FAISS memory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/episodes_50"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (default: no seeding).")
    parser.add_argument("--memory-slots", type=int, default=500)
    parser.add_argument(
        "--slots",
        type=int,
        dest="memory_slots",
        help="Alias for --memory-slots.",
    )
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument(
        "--index-desc",
        type=str,
        default="Flat",
        help="FAISS index description (passed to faiss.index_factory, e.g., 'Flat' or 'IVF256,Flat').",
    )
    parser.add_argument(
        "--min-write-gate",
        type=float,
        default=1e-3,
        help="Minimum write gate value required before inserting into the memory.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional JSON file to store final train/eval metrics.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Optional path to save the trained model checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu else None
    train_metrics, eval_metrics, _ = train_fbam_sparse(
        args.data_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        memory_slots=args.memory_slots,
        memory_dim=args.memory_dim,
        k_neighbors=args.k_neighbors,
        index_desc=args.index_desc,
        min_write_gate=args.min_write_gate,
        checkpoint_out=args.checkpoint_out,
    )
    if args.metrics_out:
        payload = {
            "train": train_metrics,
            "eval": eval_metrics,
            "seed": args.seed,
            "model": "fbam_sparse",
            "data_dir": str(args.data_dir),
            "memory_slots": args.memory_slots,
            "memory_dim": args.memory_dim,
            "k_neighbors": args.k_neighbors,
            "index_desc": args.index_desc,
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(payload, indent=2))
        print(f"[info] metrics written to {args.metrics_out}")


if __name__ == "__main__":
    main()
