"""
Training harness scaffolding for the SR-FBAM MVP.

This module wires the dataset loaders (`src.data`) and experiment logger
(`src.evaluation.logger_schema`) so Phase 1 models can plug into a
consistent training loop.

Design references:
- Overview.md:101-138 for data/operators
- Overview.md:142-146 for Phase 1 training requirements
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

# Ensure the package root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data import KnowledgeGraph, Query, load_dataset
from src.evaluation.logger_schema import ExperimentLogger, QueryLog, current_gpu_memory_mb
from src.models.sr_fbam import Action, MemoryContext, SRFBAM, SRFBAMConfig, create_srfbam


@dataclass
class TrainingConfig:
    """Minimal configuration for Phase 1 experiments."""

    data_dir: Path
    variant: Optional[str] = None
    split: str = "train"
    experiment_name: str = "sr_fbam_mve"

    @classmethod
    def from_args(
        cls,
        data_dir: str,
        variant: Optional[str] = None,
        split: str = "train",
        experiment_name: Optional[str] = None,
    ) -> "TrainingConfig":
        """Helper for CLI-style construction."""
        return cls(
            data_dir=Path(data_dir).expanduser(),
            variant=variant,
            split=split,
            experiment_name=experiment_name or f"sr_fbam_{split}",
        )


class TrainingHarness:
    """
    Lightweight wrapper that exposes the loaded dataset and logger.

    Model-specific trainers can inherit from this class to reuse the data
    plumbing set up here.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.kg, self.queries = self._load()
        self.logger = ExperimentLogger(config.experiment_name)

    def _load(self) -> Tuple[KnowledgeGraph, list[Query]]:
        """Load KG and query set via the shared loader."""
        kg, queries = load_dataset(
            self.config.data_dir,
            variant=self.config.variant,
            split=self.config.split,
        )
        return kg, queries

    def iter_queries(self) -> Iterator[Query]:
        """Yield queries in the loaded split (Phase 1 uses full-batch)."""
        yield from self.queries

    def dataset_summary(self) -> str:
        """Return a human-readable summary for logging/debug."""
        return (
            f"KG nodes={self.kg.num_nodes}, edges={self.kg.num_edges}, "
            f"queries={len(self.queries)}, variant={self.config.variant or 'baseline'}"
        )

    def save_logs(self, output_path: Path) -> None:
        """Persist accumulated query logs."""
        self.logger.save(output_path)

    @property
    def graph_node_count(self) -> int:
        return len(self.kg.nodes_by_id)

    @property
    def graph_edge_count(self) -> int:
        return len(self.kg.edges)


def prepare_harness(
    data_dir: str,
    variant: Optional[str] = None,
    split: str = "train",
    experiment_name: Optional[str] = None,
) -> TrainingHarness:
    """
    Convenience helper for scripts.

    Example:
        harness = prepare_harness("data", split="train")
        print(harness.dataset_summary())
    """
    config = TrainingConfig.from_args(
        data_dir=data_dir,
        variant=variant,
        split=split,
        experiment_name=experiment_name,
    )
    return TrainingHarness(config)


class QueryDataset(Dataset):
    """Dataset wrapper for batching queries."""

    def __init__(self, queries: List[Query]) -> None:
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Query:
        return self.queries[idx]


def collate_queries(batch: List[Query]) -> List[Query]:
    """Identity collation function."""
    return batch


def compute_action_loss(model: SRFBAM, query: Query, kg: KnowledgeGraph) -> torch.Tensor:
    """
    Supervised loss that teaches the integrator to choose the action sequence
    specified in the query's symbolic plan (teacher forcing).
    """
    device = model.device
    frame_embed, _ = model.frame_head(query, device=device)
    memory = MemoryContext(kg, model.embeddings)
    h, c = model.integrator.init_state(1, device)
    prev_action_idx = model.start_action_index

    losses = []
    plan = query.symbolic_plan if isinstance(query.symbolic_plan, list) else []

    for step in plan:
        action_name = (
            step.get("action")
            or step.get("operator")
            or step.get("op")
            or step.get("type")
        )
        if not action_name:
            continue

        action_name = action_name.upper()
        if action_name not in Action.__members__:
            continue

        node_embed = memory.embed_current_nodes()
        h, c, logits, _ = model.integrator.step(
            frame_embed, node_embed, prev_action_idx, (h, c)
        )

        target_action = Action[action_name]
        target_idx = model.action_to_index[target_action]
        target_tensor = torch.tensor(
            [target_idx], dtype=torch.long, device=device
        )
        loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
        losses.append(loss)

        # Execute the ground-truth action to update memory context.
        if action_name != "HALT":
            memory.execute(target_action, step)

        prev_action_idx = target_idx

        if action_name == "HALT":
            break

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(losses).mean()


def compute_action_loss_batch(
    model: SRFBAM,
    queries: List[Query],
    kg: KnowledgeGraph,
) -> Tuple[torch.Tensor, List[float]]:
    """Compute average loss for a batch of queries."""
    losses: List[torch.Tensor] = []
    for query in queries:
        losses.append(compute_action_loss(model, query, kg))

    if not losses:
        device = model.device
        return torch.tensor(0.0, device=device, requires_grad=True), []

    stacked = torch.stack(losses)
    return stacked.mean(), [float(loss.item()) for loss in losses]


def train_srfbam(
    harness: TrainingHarness,
    model: SRFBAM,
    epochs: int = 1,
    lr: float = 1e-3,
    clip_norm: float = 1.0,
    batch_size: int = 8,
    use_amp: bool = True,
    log_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    save_every: int = 1,
) -> None:
    """Supervised training loop for SR-FBAM with batching and optional AMP."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    amp_enabled = use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
    log_dir = Path(log_dir or (ROOT_DIR / "results"))
    checkpoint_dir = Path(checkpoint_dir or (ROOT_DIR / "checkpoints"))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = model.device
    print(
        f"Training config: batch_size={batch_size}, "
        f"AMP={amp_enabled}, device={device}"
    )

    best_accuracy = 0.0
    best_checkpoint_path = checkpoint_dir / f"{harness.config.experiment_name}_best.pt"

    for epoch in range(1, epochs + 1):
        random.shuffle(harness.queries)
        dataset = QueryDataset(harness.queries)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_queries,
        )

        epoch_losses: List[float] = []
        correct = 0
        total = 0

        for batch_queries in dataloader:
            model.train()
            start_time = perf_counter()

            if amp_enabled:
                with torch.cuda.amp.autocast():
                    batch_loss, individual_losses = compute_action_loss_batch(
                        model, batch_queries, harness.kg
                    )
            else:
                batch_loss, individual_losses = compute_action_loss_batch(
                    model, batch_queries, harness.kg
                )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(batch_loss).backward()
                if clip_norm is not None and clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                if clip_norm is not None and clip_norm > 0:
                    clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            training_duration_ms = (perf_counter() - start_time) * 1000.0
            epoch_losses.extend(individual_losses)

            model.eval()
            with torch.no_grad():
                for query, query_loss in zip(batch_queries, individual_losses):
                    inference_start = perf_counter()
                    output = model.reason(query, harness.kg)
                    inference_ms = (perf_counter() - inference_start) * 1000.0
                    wall_time_ms = (training_duration_ms / max(len(batch_queries), 1)) + inference_ms

                    prediction_id = output.prediction_id or ""
                    is_correct = prediction_id == query.answer_id
                    correct += int(is_correct)
                    total += 1

                    query_log = QueryLog(
                        query_id=query.query_id,
                        query_text=query.natural_language,
                        ground_truth=query.answer_id,
                        prediction=prediction_id,
                        correct=is_correct,
                        total_hops=len(output.hop_traces),
                        wall_time_ms=float(wall_time_ms),
                        final_loss=float(query_loss),
                        peak_gpu_memory_mb=current_gpu_memory_mb(),
                        graph_size_nodes=harness.graph_node_count,
                        model_type="sr_fbam",
                        timestamp_iso=datetime.utcnow().isoformat(),
                        hops=output.hop_traces,
                    )
                    harness.logger.log_query(query_log)

        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        accuracy = correct / max(total, 1)
        print(
            f"[Epoch {epoch}] "
            f"examples={total} "
            f"loss={mean_loss:.4f} "
            f"accuracy={accuracy:.3f}"
        )

        if epoch % max(save_every, 1) == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model.config.__dict__,
                "accuracy": accuracy,
                "loss": mean_loss,
                "timestamp": datetime.utcnow().isoformat(),
            }
            epoch_path = checkpoint_dir / f"{harness.config.experiment_name}_epoch{epoch}.pt"
            torch.save(checkpoint, epoch_path)
            print(f"Saved checkpoint: {epoch_path.name}")

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save(checkpoint, best_checkpoint_path)
                print(f"[BEST] accuracy {accuracy:.3f} stored at {best_checkpoint_path.name}")

    log_path = log_dir / f"{harness.config.experiment_name}.json"
    harness.save_logs(log_path)
    print(f"Saved query logs to {log_path.resolve()}")
    aggregates = harness.logger.aggregate_metrics()
    if aggregates:
        print("Aggregate metrics:", aggregates)

    if best_checkpoint_path.exists():
        print(f"Best checkpoint: {best_checkpoint_path.resolve()}")


def load_checkpoint(checkpoint_path: Path, device: Optional[str] = None) -> SRFBAM:
    """Load a saved SR-FBAM model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config_kwargs = checkpoint.get("config", {})
    config = SRFBAMConfig(**config_kwargs)
    model = SRFBAM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    if device:
        model = model.to(torch.device(device))
    print(
        f"Loaded checkpoint {checkpoint_path.name} "
        f"(epoch {checkpoint.get('epoch')}, accuracy={checkpoint.get('accuracy', 0):.3f})"
    )
    return model


def eval_srfbam(
    harness: TrainingHarness,
    model: SRFBAM,
    log_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate SR-FBAM on a dataset split without training."""
    log_dir = Path(log_dir or (ROOT_DIR / "results"))
    log_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    correct = 0
    total = 0

    print(
        f"Evaluating {len(harness.queries)} queries "
        f"(variant={harness.config.variant or 'baseline'})"
    )

    with torch.no_grad():
        for idx, query in enumerate(harness.queries, start=1):
            start_time = perf_counter()
            output = model.reason(query, harness.kg)
            wall_time_ms = (perf_counter() - start_time) * 1000.0

            prediction_id = output.prediction_id or ""
            is_correct = prediction_id == query.answer_id
            correct += int(is_correct)
            total += 1

            query_log = QueryLog(
                query_id=query.query_id,
                query_text=query.natural_language,
                ground_truth=query.answer_id,
                prediction=prediction_id,
                correct=is_correct,
                total_hops=len(output.hop_traces),
                wall_time_ms=float(wall_time_ms),
                final_loss=0.0,
                peak_gpu_memory_mb=current_gpu_memory_mb(),
                graph_size_nodes=harness.graph_node_count,
                model_type="sr_fbam",
                timestamp_iso=datetime.utcnow().isoformat(),
                hops=output.hop_traces,
            )
            harness.logger.log_query(query_log)

            if verbose and (idx % 10 == 0 or idx == len(harness.queries)):
                print(f"  [{idx}/{len(harness.queries)}] accuracy={correct/total:.3f}")

    log_path = log_dir / f"{harness.config.experiment_name}.json"
    harness.save_logs(log_path)
    accuracy = correct / max(total, 1)
    aggregates = harness.logger.aggregate_metrics()

    print(
        f"\n[OK] Evaluation complete: accuracy={accuracy:.3f}, "
        f"mean_hops={aggregates.get('mean_hops', 0):.2f}, "
        f"mean_wall_time_ms={aggregates.get('mean_wall_time_ms', 0):.2f}"
    )
    print(f"Logs saved to {log_path.resolve()}")

    return {"accuracy": accuracy, **aggregates}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train/Evaluate SR-FBAM (Phase 1 MVE)")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval mode")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--variant", default=None, help="Stress-test variant")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--experiment-name", default=None, help="Log experiment name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even if CUDA is available")
    parser.add_argument("--device", default=None, help="Torch device (e.g., cuda, cpu)")
    parser.add_argument("--log-dir", default=None, help="Directory for JSON logs")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for model checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoints every N epochs")
    args = parser.parse_args(argv)

    harness = prepare_harness(
        args.data_dir,
        variant=args.variant,
        split=args.split,
        experiment_name=args.experiment_name,
    )
    print(harness.dataset_summary())

    if args.mode == "train":
        model = create_srfbam(SRFBAMConfig(), device=args.device)
        train_srfbam(
            harness,
            model,
            epochs=args.epochs,
            lr=args.lr,
            clip_norm=args.clip_norm,
            batch_size=args.batch_size,
            use_amp=not args.no_amp,
            log_dir=Path(args.log_dir) if args.log_dir else None,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
            save_every=args.save_every,
        )
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in eval mode")
        model = load_checkpoint(Path(args.checkpoint), device=args.device)
        eval_srfbam(
            harness,
            model,
            log_dir=Path(args.log_dir) if args.log_dir else None,
        )


if __name__ == "__main__":
    main()
