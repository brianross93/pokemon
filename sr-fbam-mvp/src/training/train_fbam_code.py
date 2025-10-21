"""
Training harness for the pure FBAM code-editing baseline.

Usage:
    python -m src.training.train_fbam_code --data-dir data/episodes_50 --epochs 3
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import optim

from src.data.frame_dataset import FrameActionDataset, load_datasets
from src.models.fbam_code_baseline import PureFBAMCodeAgent, PureFBAMCodeConfig
from src.training.functional_metrics import (
    accumulate_functional_stats,
    finalize_functional_stats,
    init_functional_stats,
)


def _episode_tensors(episode) -> Tuple[torch.Tensor, torch.Tensor]:
    frames = [step.frame for step in episode.steps]
    actions = torch.tensor([step.action_index for step in episode.steps], dtype=torch.long)
    return frames, actions


def train_epoch(
    model: PureFBAMCodeAgent,
    dataset: FrameActionDataset,
    optimizer: optim.Optimizer,
    device: torch.device,
    index_to_action: Sequence[str],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    wall_times = []
    func_stats = init_functional_stats()
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
        actual_actions = [index_to_action[idx] for idx in targets.cpu().tolist()]
        predicted_actions = [index_to_action[idx] for idx in preds.cpu().tolist()]
        terminal_outputs = [
            (episode.steps[j].metadata.get("terminal_output") or "") for j in range(len(actual_actions))
        ]
        accumulate_functional_stats(func_stats, actual_actions, predicted_actions, terminal_outputs)

    avg_loss = total_loss / max(total_steps, 1)
    accuracy = correct / max(total_steps, 1)
    wall_time_ms = mean(wall_times) if wall_times else 0.0
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    func_metrics = finalize_functional_stats(func_stats)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "wall_time_ms": wall_time_ms,
        "peak_mem_mb": peak_mem_mb,
        **func_metrics,
    }


@torch.no_grad()
def evaluate(
    model: PureFBAMCodeAgent,
    dataset: FrameActionDataset,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    wall_times = []
    func_stats = init_functional_stats()

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
        actual_actions = [dataset.index_to_action[idx] for idx in targets_cpu.tolist()]
        predicted_actions = [dataset.index_to_action[idx] for idx in preds_cpu.tolist()]
        terminal_outputs = [
            (episode.steps[j].metadata.get("terminal_output") or "") for j in range(len(actual_actions))
        ]
        accumulate_functional_stats(func_stats, actual_actions, predicted_actions, terminal_outputs)

    avg_loss = total_loss / max(total_steps, 1)
    accuracy = correct / max(total_steps, 1)
    wall_time_ms = mean(wall_times) if wall_times else 0.0
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "wall_time_ms": wall_time_ms,
        **finalize_functional_stats(func_stats),
    }


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fbam_code(
    data_dir: Path,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    patience: int = 0,
    min_delta: float = 0.0,
    checkpoint_out: Path | None = None,
    max_episodes: int | None = None,
    output_dir: Path | None = None,
    config_preset: str = "large",
) -> tuple[
    Dict[str, float],
    Dict[str, float],
    PureFBAMCodeAgent,
    int,
    List[Dict[str, float | int | bool]],
]:
    train_dataset, eval_dataset = load_datasets(data_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if max_episodes is not None and max_episodes > 0:
        train_dataset.episodes = train_dataset.episodes[:max_episodes]
        eval_dataset.episodes = eval_dataset.episodes[:max_episodes]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    config = PureFBAMCodeConfig.preset(config_preset) if config_preset else PureFBAMCodeConfig()
    model = PureFBAMCodeAgent(num_actions=train_dataset.num_actions, config=config)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(
        f"Pure FBAM parameters: {model.parameter_count():,} | "
        f"Preset: {config_preset} | Actions: {train_dataset.num_actions} | Device: {device}"
    )

    best_train_metrics: Dict[str, float] | None = None
    best_eval_metrics: Dict[str, float] | None = None
    best_epoch = 0
    best_model_state = None
    epochs_without_improve = 0
    min_delta = max(0.0, float(min_delta))
    history: List[Dict[str, float | int | bool]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_dataset, optimizer, device, train_dataset.index_to_action
        )
        eval_metrics = evaluate(model, eval_dataset, device)

        improved = False
        if best_eval_metrics is None:
            improved = True
        else:
            target = best_eval_metrics["accuracy"] + min_delta
            if eval_metrics["accuracy"] > target:
                improved = True

        if improved:
            best_eval_metrics = dict(eval_metrics)
            best_train_metrics = dict(train_metrics)
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improve = 0
            if checkpoint_out:
                checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_payload = {
                    "model_state_dict": {
                        k: v.detach().cpu() if hasattr(v, "detach") else v
                        for k, v in best_model_state.items()
                    },
                    "num_actions": train_dataset.num_actions,
                    "epoch": best_epoch,
                    "train_metrics": best_train_metrics,
                    "eval_metrics": best_eval_metrics,
                    "learning_rate": learning_rate,
                    "patience": patience,
                    "min_delta": min_delta,
                    "data_dir": str(data_dir),
                    "model_name": "fbam_code",
                }
                torch.save(checkpoint_payload, checkpoint_out)
                print(f"[info] checkpoint updated at epoch {epoch}: {checkpoint_out}")
        else:
            epochs_without_improve += 1

        best_acc = best_eval_metrics["accuracy"] if best_eval_metrics else eval_metrics["accuracy"]

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"train_wall={train_metrics['wall_time_ms']:.1f}ms "
            f"train_peak_mem={train_metrics['peak_mem_mb']:.1f}MB "
            f"train_diff_acc={train_metrics['diff_action_accuracy']:.3f} "
            f"train_test_acc={train_metrics['test_action_accuracy']:.3f} | "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.3f} "
            f"eval_wall={eval_metrics['wall_time_ms']:.1f}ms "
            f"eval_diff_acc={eval_metrics['diff_action_accuracy']:.3f} "
            f"eval_test_acc={eval_metrics['test_action_accuracy']:.3f} "
            f"eval_test_prec={eval_metrics['test_pass_precision']:.3f} "
            f"eval_test_rec={eval_metrics['test_pass_recall']:.3f} "
            f"| best_acc={best_acc:.3f}{' *' if improved else ''}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_wall_ms": train_metrics["wall_time_ms"],
                "train_peak_mem_mb": train_metrics["peak_mem_mb"],
                "train_diff_action_accuracy": train_metrics["diff_action_accuracy"],
                "train_test_action_accuracy": train_metrics["test_action_accuracy"],
                "train_test_pass_precision": train_metrics["test_pass_precision"],
                "train_test_pass_recall": train_metrics["test_pass_recall"],
                "eval_loss": eval_metrics["loss"],
                "eval_accuracy": eval_metrics["accuracy"],
                "eval_wall_ms": eval_metrics["wall_time_ms"],
                "eval_diff_action_accuracy": eval_metrics["diff_action_accuracy"],
                "eval_test_action_accuracy": eval_metrics["test_action_accuracy"],
                "eval_test_pass_precision": eval_metrics["test_pass_precision"],
                "eval_test_pass_recall": eval_metrics["test_pass_recall"],
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
    parser = argparse.ArgumentParser(description="Train Pure FBAM baseline on frame episodes.")
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
    parser.add_argument(
        "--config-preset",
        choices=["small", "large"],
        default="large",
        help="Model capacity preset to use (default: large).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional cap on train/eval episodes for quick smoke tests (0 = all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to store metrics/checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu else None
    metrics_out = args.metrics_out
    checkpoint_out = args.checkpoint_out
    output_dir = args.output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if metrics_out is None:
            metrics_out = output_dir / "metrics.json"
        if checkpoint_out is None:
            checkpoint_out = output_dir / "checkpoint.pt"

    train_metrics, eval_metrics, _, best_epoch, history = train_fbam_code(
        args.data_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_out=checkpoint_out,
        max_episodes=args.max_episodes if args.max_episodes > 0 else None,
        output_dir=output_dir,
        config_preset=args.config_preset,
    )
    target_metrics_path = args.metrics_out or metrics_out
    if target_metrics_path:
        payload = {
            "train": train_metrics,
            "eval": eval_metrics,
            "seed": args.seed,
            "model": "fbam",
            "data_dir": str(args.data_dir),
        }
        payload.update(
            {
                "final_eval_accuracy": eval_metrics["accuracy"],
                "final_eval_wall_ms": eval_metrics["wall_time_ms"],
                "final_eval_diff_action_accuracy": eval_metrics["diff_action_accuracy"],
                "final_eval_test_action_accuracy": eval_metrics["test_action_accuracy"],
                "final_eval_test_pass_precision": eval_metrics["test_pass_precision"],
                "final_eval_test_pass_recall": eval_metrics["test_pass_recall"],
                "final_train_accuracy": train_metrics["accuracy"],
                "final_train_wall_ms": train_metrics["wall_time_ms"],
                "final_train_diff_action_accuracy": train_metrics["diff_action_accuracy"],
                "final_train_test_action_accuracy": train_metrics["test_action_accuracy"],
                "final_train_test_pass_precision": train_metrics["test_pass_precision"],
                "final_train_test_pass_recall": train_metrics["test_pass_recall"],
                "best_epoch": best_epoch,
                "epochs_requested": args.epochs,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "history": history,
                "checkpoint": str(args.checkpoint_out) if args.checkpoint_out else None,
            }
        )
        target_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        target_metrics_path.write_text(json.dumps(payload, indent=2))
        print(f"[info] metrics written to {target_metrics_path}")
    elif output_dir and metrics_out:
        payload = {
            "train": train_metrics,
            "eval": eval_metrics,
            "seed": args.seed,
            "model": "fbam",
            "data_dir": str(args.data_dir),
            "best_epoch": best_epoch,
            "config_preset": args.config_preset,
            "max_episodes": args.max_episodes,
        }
        metrics_out.write_text(json.dumps(payload, indent=2))
        print(f"[info] metrics written to {metrics_out}")


if __name__ == "__main__":
    main()
