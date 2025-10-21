"""
Training harness for the SR-FBAM code-editing prototype.

Usage:
    python -m src.training.train_srfbam_code --data-dir data/episodes_50 --epochs 3
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import optim

from src.data.frame_dataset import FrameActionDataset, load_datasets
from src.models.sr_fbam_code import MemoryOp, SRFBAMCodeAgent, SRFBAMCodeConfig
from src.training.functional_metrics import (
    accumulate_functional_stats,
    finalize_functional_stats,
    init_functional_stats,
)
from src.training.gate_labeling import compute_hindsight_labels


class GateTraceWriter:
    """Simple JSONL writer for per-step gate trace records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    def close(self) -> None:
        self._fh.close()


def train_epoch(
    model: SRFBAMCodeAgent,
    dataset: FrameActionDataset,
    optimizer: optim.Optimizer,
    device: torch.device,
    index_to_action: Sequence[str],
    lambda_compute: float,
    compute_penalty: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    wall_times: List[float] = []
    encode_fracs: List[float] = []
    query_fracs: List[float] = []
    skip_fracs: List[float] = []
    predicted_speedups: List[float] = []
    observed_speedups: List[float] = []
    write_rates: List[float] = []
    salience_avgs: List[float] = []
    func_stats = init_functional_stats()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for episode in dataset.episodes:
        if not episode.steps:
            continue

        targets = torch.tensor([step.action_index for step in episode.steps], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)

        memory_labels = compute_hindsight_labels(model, episode, lambda_compute=lambda_compute, device=device)

        start = perf_counter()
        rollout = model.forward_episode(
            episode,
            memory_labels=memory_labels,
            teacher_force_actions=True,
        )
        action_logits = rollout.action_logits_env()
        if action_logits.size(0) == 0:
            continue
        memory_logits = rollout.memory_logits()
        memory_slice = memory_logits[:, : model.vocab.action_offset]
        memory_targets = torch.tensor([int(op) for op in memory_labels], dtype=torch.long, device=device)

        action_loss = F.cross_entropy(action_logits, targets)
        memory_loss = F.cross_entropy(memory_slice, memory_targets)

        gate_stats = model.last_gate_stats or {}
        encode_fraction = float(gate_stats.get("encode_fraction", 0.0))
        loss = action_loss + memory_loss + compute_penalty * encode_fraction

        loss.backward()
        optimizer.step()
        wall_times.append((perf_counter() - start) * 1000.0)

        total_loss += loss.item() * targets.size(0)
        total_steps += targets.size(0)

        preds = action_logits.argmax(dim=1)
        correct += (preds == targets).sum().item()

        encode_fracs.append(encode_fraction)
        query_fracs.append(float(gate_stats.get("query_fraction", 0.0)))
        skip_fracs.append(float(gate_stats.get("skip_fraction", 0.0)))
        predicted_speedups.append(float(gate_stats.get("predicted_speedup", 0.0)))
        observed_speedups.append(float(gate_stats.get("observed_speedup", 0.0)))
        write_rates.append(float(gate_stats.get("write_commit_rate", 0.0)))
        salience_avgs.append(float(gate_stats.get("salience_average", 0.0)))

        actual_actions = [index_to_action[idx] for idx in targets.detach().cpu().tolist()]
        predicted_actions = [index_to_action[idx] for idx in preds.detach().cpu().tolist()]
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
        "gate_encode_fraction": float(mean(encode_fracs)) if encode_fracs else 0.0,
        "gate_query_fraction": float(mean(query_fracs)) if query_fracs else 0.0,
        "gate_skip_fraction": float(mean(skip_fracs)) if skip_fracs else 0.0,
        "gate_predicted_speedup": float(mean(predicted_speedups)) if predicted_speedups else 0.0,
        "gate_observed_speedup": float(mean(observed_speedups)) if observed_speedups else 0.0,
        "gate_write_commit_rate": float(mean(write_rates)) if write_rates else 0.0,
        "gate_salience_average": float(mean(salience_avgs)) if salience_avgs else 0.0,
        **func_metrics,
    }


@torch.no_grad()
def evaluate(
    model: SRFBAMCodeAgent,
    dataset: FrameActionDataset,
    device: torch.device,
    lambda_compute: float,
    trace_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    split_name: str = "eval",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    memory_correct = 0
    memory_total = 0
    wall_times: List[float] = []
    encode_fracs: List[float] = []
    query_fracs: List[float] = []
    skip_fracs: List[float] = []
    predicted_speedups: List[float] = []
    observed_speedups: List[float] = []
    write_rates: List[float] = []
    salience_avgs: List[float] = []
    func_stats = init_functional_stats()

    for episode in dataset.episodes:
        if not episode.steps:
            continue

        targets = torch.tensor([step.action_index for step in episode.steps], dtype=torch.long, device=device)
        memory_labels = compute_hindsight_labels(model, episode, lambda_compute=lambda_compute, device=device)

        start = perf_counter()
        rollout = model.forward_episode(
            episode,
            memory_labels=None,
            teacher_force_actions=True,
        )
        wall_times.append((perf_counter() - start) * 1000.0)
        action_logits = rollout.action_logits_env()
        if action_logits.size(0) == 0:
            continue
        preds = action_logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_steps += targets.size(0)
        action_loss = F.cross_entropy(action_logits, targets)

        memory_logits = rollout.memory_logits()
        memory_slice = memory_logits[:, : model.vocab.action_offset]
        memory_targets = torch.tensor([int(op) for op in memory_labels], dtype=torch.long, device=device)
        memory_loss = F.cross_entropy(memory_slice, memory_targets)

        loss = action_loss + memory_loss
        gate_stats = model.last_gate_stats or {}
        encode_fracs.append(float(gate_stats.get("encode_fraction", 0.0)))
        query_fracs.append(float(gate_stats.get("query_fraction", 0.0)))
        skip_fracs.append(float(gate_stats.get("skip_fraction", 0.0)))
        predicted_speedups.append(float(gate_stats.get("predicted_speedup", 0.0)))
        observed_speedups.append(float(gate_stats.get("observed_speedup", 0.0)))
        write_rates.append(float(gate_stats.get("write_commit_rate", 0.0)))
        salience_avgs.append(float(gate_stats.get("salience_average", 0.0)))

        memory_predictions: List[int] = []
        for meta, token_idx in zip(rollout.metadata, rollout.executed_tokens):
            if meta.phase == "memory":
                memory_predictions.append(int(token_idx))

        memory_total += len(memory_labels)
        for pred_idx, label_op in zip(memory_predictions, memory_labels):
            if model.vocab.is_memory_index(pred_idx) and int(label_op) == pred_idx:
                memory_correct += 1

        actual_actions = [dataset.index_to_action[idx] for idx in targets.cpu().tolist()]
        predicted_actions = [dataset.index_to_action[idx] for idx in preds.cpu().tolist()]
        terminal_outputs = [
            (episode.steps[j].metadata.get("terminal_output") or "") for j in range(len(actual_actions))
        ]
        accumulate_functional_stats(func_stats, actual_actions, predicted_actions, terminal_outputs)

        if trace_writer:
            gate_trace = model.last_gate_trace or []
            target_indices = targets.cpu().tolist()
            pred_indices = preds.cpu().tolist()
            predicted_memory_ops = []
            for pred_idx in memory_predictions:
                if model.vocab.is_memory_index(pred_idx):
                    predicted_memory_ops.append(MemoryOp(pred_idx).name)
                else:
                    predicted_memory_ops.append("UNKNOWN")
            hindsight_ops = [label.name for label in memory_labels]
            for step_offset, trace_entry in enumerate(gate_trace):
                record = dict(trace_entry)
                record.update(
                    {
                        "split": split_name,
                        "target_action_index": target_indices[step_offset],
                        "predicted_action_index": pred_indices[step_offset],
                        "target_action": dataset.index_to_action[target_indices[step_offset]],
                        "predicted_action": dataset.index_to_action[pred_indices[step_offset]],
                        "action_correct": pred_indices[step_offset] == target_indices[step_offset],
                        "predicted_memory_op": (
                            predicted_memory_ops[step_offset] if step_offset < len(predicted_memory_ops) else None
                        ),
                        "hindsight_memory_op": (
                            hindsight_ops[step_offset] if step_offset < len(hindsight_ops) else None
                        ),
                    }
                )
                trace_writer(record)

        total_loss += loss.item() * targets.size(0)

    avg_loss = total_loss / max(total_steps, 1)
    accuracy = correct / max(total_steps, 1)
    memory_accuracy = memory_correct / max(memory_total, 1)
    wall_time_ms = mean(wall_times) if wall_times else 0.0
    func_metrics = finalize_functional_stats(func_stats)
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "wall_time_ms": wall_time_ms,
        "memory_accuracy": memory_accuracy,
        "gate_encode_fraction": float(mean(encode_fracs)) if encode_fracs else 0.0,
        "gate_query_fraction": float(mean(query_fracs)) if query_fracs else 0.0,
        "gate_skip_fraction": float(mean(skip_fracs)) if skip_fracs else 0.0,
        "gate_predicted_speedup": float(mean(predicted_speedups)) if predicted_speedups else 0.0,
        "gate_observed_speedup": float(mean(observed_speedups)) if observed_speedups else 0.0,
        "gate_write_commit_rate": float(mean(write_rates)) if write_rates else 0.0,
        "gate_salience_average": float(mean(salience_avgs)) if salience_avgs else 0.0,
        **func_metrics,
    }


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_srfbam_code(
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
    disable_gate: bool = False,
    disable_memory: bool = False,
    gate_mode: str = "learned",
    gate_threshold: float | None = None,
    gate_temperature: float | None = None,
    gate_lambda: float = 0.002,
    compute_penalty: float = 0.0,
    gate_trace_out: Path | None = None,
) -> tuple[
    Dict[str, float],
    Dict[str, float],
    SRFBAMCodeAgent,
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

    config = SRFBAMCodeConfig.preset(config_preset) if config_preset else SRFBAMCodeConfig()
    if disable_memory:
        config.enable_memory = False
    else:
        config.enable_memory = bool(config.enable_memory)

    if gate_mode:
        config.gate_mode = gate_mode.lower()
    else:
        config.gate_mode = (config.gate_mode or "learned").lower()

    if gate_threshold is not None:
        config.gate_threshold = max(0.0, min(1.0, gate_threshold))
    if gate_temperature is not None:
        config.gate_temperature = max(1e-6, gate_temperature)

    valid_gate_modes = {"learned", "always_extract", "always_reuse", "random"}
    if config.gate_mode not in valid_gate_modes:
        config.gate_mode = "learned"

    if disable_gate:
        config.enable_gate = False
        config.gate_mode = "always_extract"
    else:
        config.enable_gate = bool(config.enable_gate)

    if config.gate_mode != "learned":
        config.enable_gate = False

    gate_lambda = max(0.0, float(gate_lambda))
    compute_penalty = max(0.0, float(compute_penalty))

    model = SRFBAMCodeAgent(num_actions=train_dataset.num_actions, config=config)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(
        f"SR-FBAM parameters: {model.parameter_count():,} | "
        f"Preset: {config_preset} | Actions: {train_dataset.num_actions} | Device: {device} | "
        f"GateMode: {config.gate_mode} | GateEnabled: {config.enable_gate} | "
        f"MemoryEnabled: {config.enable_memory} | GateThreshold: {config.gate_threshold:.2f} | "
        f"GateTemp: {config.gate_temperature:.2f}"
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
            model,
            train_dataset,
            optimizer,
            device,
            train_dataset.index_to_action,
            lambda_compute=gate_lambda,
            compute_penalty=compute_penalty,
        )
        eval_metrics = evaluate(
            model,
            eval_dataset,
            device,
            lambda_compute=gate_lambda,
        )

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
                    "model_name": "srfbam_code",
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
            f"train_gate_encode={train_metrics['gate_encode_fraction']:.2f} "
            f"train_gate_query={train_metrics['gate_query_fraction']:.2f} "
            f"train_speedup={train_metrics['gate_predicted_speedup']:.2f} "
            f"train_speedup_obs={train_metrics['gate_observed_speedup']:.2f} "
            f"train_diff_acc={train_metrics['diff_action_accuracy']:.3f} "
            f"train_test_acc={train_metrics['test_action_accuracy']:.3f} | "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.3f} "
            f"eval_wall={eval_metrics['wall_time_ms']:.1f}ms "
            f"eval_gate_encode={eval_metrics['gate_encode_fraction']:.2f} "
            f"eval_gate_query={eval_metrics['gate_query_fraction']:.2f} "
            f"eval_speedup={eval_metrics['gate_predicted_speedup']:.2f} "
            f"eval_speedup_obs={eval_metrics['gate_observed_speedup']:.2f} "
            f"eval_memory_acc={eval_metrics.get('memory_accuracy', 0.0):.3f} "
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
                "train_gate_encode_fraction": train_metrics["gate_encode_fraction"],
                "train_gate_query_fraction": train_metrics["gate_query_fraction"],
                "train_gate_skip_fraction": train_metrics["gate_skip_fraction"],
                "train_gate_predicted_speedup": train_metrics["gate_predicted_speedup"],
                "train_gate_observed_speedup": train_metrics["gate_observed_speedup"],
                "train_gate_write_commit_rate": train_metrics["gate_write_commit_rate"],
                "train_gate_salience_average": train_metrics["gate_salience_average"],
                "train_diff_action_accuracy": train_metrics["diff_action_accuracy"],
                "train_test_action_accuracy": train_metrics["test_action_accuracy"],
                "train_test_pass_precision": train_metrics["test_pass_precision"],
                "train_test_pass_recall": train_metrics["test_pass_recall"],
                "eval_loss": eval_metrics["loss"],
                "eval_accuracy": eval_metrics["accuracy"],
                "eval_wall_ms": eval_metrics["wall_time_ms"],
                "eval_gate_encode_fraction": eval_metrics["gate_encode_fraction"],
                "eval_gate_query_fraction": eval_metrics["gate_query_fraction"],
                "eval_gate_skip_fraction": eval_metrics["gate_skip_fraction"],
                "eval_gate_predicted_speedup": eval_metrics["gate_predicted_speedup"],
                "eval_gate_observed_speedup": eval_metrics["gate_observed_speedup"],
                "eval_gate_write_commit_rate": eval_metrics["gate_write_commit_rate"],
                "eval_gate_salience_average": eval_metrics["gate_salience_average"],
                "eval_memory_accuracy": eval_metrics.get("memory_accuracy", 0.0),
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
        # Fallback in case no improvement logic triggered
        best_train_metrics = train_metrics
        best_eval_metrics = eval_metrics
        best_epoch = epochs

    if gate_trace_out:
        trace_writer = GateTraceWriter(gate_trace_out)
        try:
            evaluate(
                model,
                eval_dataset,
                device,
                lambda_compute=gate_lambda,
                trace_writer=trace_writer.write,
                split_name="eval",
            )
        finally:
            trace_writer.close()

    assert best_train_metrics is not None and best_eval_metrics is not None
    return best_train_metrics, best_eval_metrics, model, best_epoch, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SR-FBAM prototype on frame episodes.")
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
    parser.add_argument(
        "--disable-gate",
        action="store_true",
        help="Disable the learned gate (always extract new symbols).",
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable symbolic memory storage and reuse.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["learned", "always_extract", "always_reuse", "random"],
        default="learned",
        help="Override gate policy (default: learned).",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="Override gate decision threshold (0-1).",
    )
    parser.add_argument(
        "--gate-temperature",
        type=float,
        default=None,
        help="Override gate temperature during training.",
    )
    parser.add_argument(
        "--gate-lambda",
        type=float,
        default=0.002,
        help="Trade-off λ used for hindsight gate labeling (higher = penalise latency).",
    )
    parser.add_argument(
        "--compute-penalty",
        type=float,
        default=0.0,
        help="Additional loss weight applied to the encode fraction (default: 0).",
    )
    parser.add_argument(
        "--gate-trace-out",
        type=Path,
        default=None,
        help="Optional JSONL file to record per-step gate traces on final eval.",
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

    train_metrics, eval_metrics, trained_model, best_epoch, history = train_srfbam_code(
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
        disable_gate=args.disable_gate,
        disable_memory=args.disable_memory,
        gate_mode=args.gate_mode,
        gate_threshold=args.gate_threshold,
        gate_temperature=args.gate_temperature,
        gate_lambda=args.gate_lambda,
        compute_penalty=args.compute_penalty,
        gate_trace_out=args.gate_trace_out,
    )
    target_metrics_path = args.metrics_out or metrics_out
    if target_metrics_path:
        payload = {
            "train": train_metrics,
            "eval": eval_metrics,
            "seed": args.seed,
            "model": "srfbam",
            "data_dir": str(args.data_dir),
            "best_epoch": best_epoch,
        }
        payload.update(
            {
                "final_eval_accuracy": eval_metrics["accuracy"],
                "final_eval_wall_ms": eval_metrics["wall_time_ms"],
                "final_eval_gate_encode_fraction": eval_metrics["gate_encode_fraction"],
                "final_eval_gate_query_fraction": eval_metrics["gate_query_fraction"],
                "final_eval_gate_skip_fraction": eval_metrics["gate_skip_fraction"],
                "final_eval_gate_predicted_speedup": eval_metrics["gate_predicted_speedup"],
                "final_eval_gate_observed_speedup": eval_metrics["gate_observed_speedup"],
                "final_eval_gate_write_commit_rate": eval_metrics["gate_write_commit_rate"],
                "final_eval_gate_salience_average": eval_metrics["gate_salience_average"],
                "final_eval_memory_accuracy": eval_metrics.get("memory_accuracy", 0.0),
                "final_eval_diff_action_accuracy": eval_metrics["diff_action_accuracy"],
                "final_eval_test_action_accuracy": eval_metrics["test_action_accuracy"],
                "final_eval_test_pass_precision": eval_metrics["test_pass_precision"],
                "final_eval_test_pass_recall": eval_metrics["test_pass_recall"],
                "final_train_accuracy": train_metrics["accuracy"],
                "final_train_wall_ms": train_metrics["wall_time_ms"],
                "final_train_gate_encode_fraction": train_metrics["gate_encode_fraction"],
                "final_train_gate_query_fraction": train_metrics["gate_query_fraction"],
                "final_train_gate_skip_fraction": train_metrics["gate_skip_fraction"],
                "final_train_gate_predicted_speedup": train_metrics["gate_predicted_speedup"],
                "final_train_gate_observed_speedup": train_metrics["gate_observed_speedup"],
                "final_train_gate_write_commit_rate": train_metrics["gate_write_commit_rate"],
                "final_train_gate_salience_average": train_metrics["gate_salience_average"],
                "final_train_diff_action_accuracy": train_metrics["diff_action_accuracy"],
                "final_train_test_action_accuracy": train_metrics["test_action_accuracy"],
                "final_train_test_pass_precision": train_metrics["test_pass_precision"],
                "final_train_test_pass_recall": train_metrics["test_pass_recall"],
                "epochs_requested": args.epochs,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "gate_lambda": args.gate_lambda,
                "compute_penalty": args.compute_penalty,
                "history": history,
                "checkpoint": str(args.checkpoint_out) if args.checkpoint_out else None,
                "config_gate_mode": trained_model.config.gate_mode,
                "config_gate_enabled": bool(trained_model.config.enable_gate),
                "config_gate_threshold": trained_model.config.gate_threshold,
                "config_gate_temperature": trained_model.config.gate_temperature,
                "config_memory_enabled": bool(trained_model.config.enable_memory),
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
            "model": "srfbam",
            "data_dir": str(args.data_dir),
            "best_epoch": best_epoch,
            "config_preset": args.config_preset,
            "max_episodes": args.max_episodes,
            "gate_lambda": args.gate_lambda,
            "compute_penalty": args.compute_penalty,
            "config_gate_mode": trained_model.config.gate_mode,
            "config_gate_enabled": bool(trained_model.config.enable_gate),
            "config_gate_threshold": trained_model.config.gate_threshold,
            "config_gate_temperature": trained_model.config.gate_temperature,
            "config_memory_enabled": bool(trained_model.config.enable_memory),
        }
        metrics_out.write_text(json.dumps(payload, indent=2))
        print(f"[info] metrics written to {metrics_out}")


if __name__ == "__main__":
    main()
