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
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

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
        if action_name == "HALT":
            break

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
        memory.execute(target_action, step)
        prev_action_idx = target_idx

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(losses).mean()


def train_srfbam(
    harness: TrainingHarness,
    model: SRFBAM,
    epochs: int = 1,
    lr: float = 1e-3,
    clip_norm: float = 1.0,
    log_dir: Optional[Path] = None,
) -> None:
    """Supervised training loop for SR-FBAM using symbolic plans."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_dir = Path(log_dir or (ROOT_DIR / "results"))
    log_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        random.shuffle(harness.queries)
        epoch_losses: List[float] = []
        correct = 0
        total = 0

        for query in harness.queries:
            start_time = perf_counter()
            model.train()
            loss = compute_action_loss(model, query, harness.kg)

            optimizer.zero_grad()
            loss.backward()
            if clip_norm is not None and clip_norm > 0:
                clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            model.eval()
            with torch.no_grad():
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
                final_loss=float(loss.item()),
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

    log_path = log_dir / f"{harness.config.experiment_name}.json"
    harness.save_logs(log_path)
    print(f"Saved query logs to {log_path.resolve()}")
    aggregates = harness.logger.aggregate_metrics()
    if aggregates:
        print("Aggregate metrics:", aggregates)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train SR-FBAM (Phase 1 MVE)")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--variant", default=None, help="Stress-test variant")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--experiment-name", default=None, help="Log experiment name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--device", default=None, help="Torch device (e.g., cuda, cpu)")
    parser.add_argument("--log-dir", default=None, help="Directory for JSON logs")
    args = parser.parse_args(argv)

    harness = prepare_harness(
        args.data_dir,
        variant=args.variant,
        split=args.split,
        experiment_name=args.experiment_name,
    )
    print(harness.dataset_summary())

    model = create_srfbam(SRFBAMConfig(), device=args.device)
    train_srfbam(
        harness,
        model,
        epochs=args.epochs,
        lr=args.lr,
        clip_norm=args.clip_norm,
        log_dir=Path(args.log_dir) if args.log_dir else None,
    )


if __name__ == "__main__":
    main()
