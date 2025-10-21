"""
Training harness for the Mamba-FBAM hybrid baseline.

Usage:
    python -m src.training.train_mamba_code --data-dir data/episodes_50 --epochs 3
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch import optim

from src.data.frame_dataset import FrameActionDataset, load_datasets
from src.models.mamba_fbam import MambaFBAMAgent, MambaFBAMConfig


def _episode_tensors(episode) -> Tuple[torch.Tensor, torch.Tensor]:
    frames = [step.frame for step in episode.steps]
    actions = torch.tensor([step.action_index for step in episode.steps], dtype=torch.long)
    return frames, actions


def train_epoch(
    model: MambaFBAMAgent,
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
    model: MambaFBAMAgent,
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
        preds_cpu = preds.cpu()
        targets_cpu = targets.cpu()
        correct += (preds_cpu == targets_cpu).sum().item()
        total_steps += targets.size(0)

        logits = model.forward_episode(frames, targets)
        loss = F.cross_entropy(logits, targets)

        total_loss += loss.item() * targets.size(0)

    avg_loss = total_loss / max(total_steps, 1)
    accuracy = correct / max(total_steps, 1)
    wall_time_ms = mean(wall_times) if wall_times else 0.0
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "wall_time_ms": wall_time_ms,
    }


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(dataset: FrameActionDataset, config: MambaFBAMConfig) -> MambaFBAMAgent:
    model = MambaFBAMAgent(num_actions=dataset.num_actions, config=config)
    return model


def train_mamba_code(
    data_dir: Path,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    patience: int = 0,
    min_delta: float = 0.0,
    checkpoint_out: Optional[Path] = None,
    config: Optional[MambaFBAMConfig] = None,
) -> Tuple[Dict[str, float], Dict[str, float], MambaFBAMAgent, int, List[Dict]]:
    device = _ensure_device(device)
    config = config or MambaFBAMConfig()

    train_ds, eval_ds = load_datasets(
        data_dir,
        grid_height=config.grid_height,
        grid_width=config.grid_width,
    )
    model = _build_model(train_ds, config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history: List[Dict] = []
    best_eval_metrics: Optional[Dict[str, float]] = None
    best_train_metrics: Optional[Dict[str, float]] = None
    best_model_state = None
    best_epoch = 0
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_ds, optimizer, device)
        eval_metrics = evaluate(model, eval_ds, device)

        improved = False
        if best_eval_metrics is None or eval_metrics["accuracy"] > best_eval_metrics["accuracy"] + min_delta:
            improved = True
            best_eval_metrics = deepcopy(eval_metrics)
            best_train_metrics = deepcopy(train_metrics)
            best_epoch = epoch
            epochs_without_improve = 0
            best_model_state = deepcopy(model.state_dict())
            if checkpoint_out:
                checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_state, checkpoint_out)
                print(f"[info] checkpoint updated at epoch {epoch}: {checkpoint_out}")
        else:
            epochs_without_improve += 1

        best_acc = best_eval_metrics["accuracy"] if best_eval_metrics else eval_metrics["accuracy"]

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"train_wall={train_metrics['wall_time_ms']:.1f}ms "
            f"train_peak_mem={train_metrics['peak_mem_mb']:.1f}MB | "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.3f} "
            f"eval_wall={eval_metrics['wall_time_ms']:.1f}ms "
            f"| best_acc={best_acc:.3f}{' *' if improved else ''}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_wall_ms": train_metrics["wall_time_ms"],
                "train_peak_mem_mb": train_metrics["peak_mem_mb"],
                "eval_loss": eval_metrics["loss"],
                "eval_accuracy": eval_metrics["accuracy"],
                "eval_wall_ms": eval_metrics["wall_time_ms"],
                "improved": improved,
                "epochs_without_improve": epochs_without_improve,
            }
        )

        if patience > 0 and epochs_without_improve >= patience:
            print(
                f"[info] early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, best_acc={best_acc:.3f})"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        best_train_metrics = train_metrics
        best_eval_metrics = eval_metrics
        best_epoch = epochs

    assert best_train_metrics is not None and best_eval_metrics is not None
    return best_train_metrics, best_eval_metrics, model, best_epoch, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mamba-FBAM baseline on frame episodes.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/episodes_50"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (default: no seeding).")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (in epochs). 0 disables early stopping.",
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
        help="Optional path to store the best-performing checkpoint.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement in eval accuracy required to reset patience.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu else None
    train_metrics, eval_metrics, _, best_epoch, history = train_mamba_code(
        args.data_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_out=args.checkpoint_out,
    )
    if args.metrics_out:
        payload = {
            "train": train_metrics,
            "eval": eval_metrics,
            "seed": args.seed,
            "model": "mamba_fbam",
            "data_dir": str(args.data_dir),
        }
        payload.update(
            {
                "final_eval_accuracy": eval_metrics["accuracy"],
                "final_eval_wall_ms": eval_metrics["wall_time_ms"],
                "final_train_accuracy": train_metrics["accuracy"],
                "final_train_wall_ms": train_metrics["wall_time_ms"],
                "best_epoch": best_epoch,
                "epochs_requested": args.epochs,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "history": history,
                "checkpoint": str(args.checkpoint_out) if args.checkpoint_out else None,
            }
        )
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(payload, indent=2))
        print(f"[info] metrics written to {args.metrics_out}")


if __name__ == "__main__":
    main()
