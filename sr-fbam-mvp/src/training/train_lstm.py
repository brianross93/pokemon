"""
Training and evaluation script for the LSTM baseline.
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, rnn
from torch.utils.data import DataLoader

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.lstm_baseline import LSTMBaseline, LSTMBaselineConfig
from src.training.train import (
    TrainingHarness,
    prepare_harness,
    QueryDataset,
    collate_queries,
)
from src.evaluation.logger_schema import QueryLog, current_gpu_memory_mb


@dataclass
class LSTMTrainingConfig:
    data_dir: Path
    variant: Optional[str]
    split: str
    experiment_name: str


class LSTMBatcher:
    """Utility to encode queries and targets for the LSTM baseline."""

    def __init__(self, model: LSTMBaseline, node_to_idx: Dict[str, int], device: torch.device) -> None:
        self.model = model
        self.node_to_idx = node_to_idx
        self.device = device

    def encode_queries(self, queries: List) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            self.model.encode_query(query.natural_language, self.device)
            for query in queries
        ]
        padded = rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

        targets = []
        for query in queries:
            idx = self.node_to_idx.get(query.answer_id)
            if idx is None:
                raise ValueError(f"Answer {query.answer_id} not in node vocabulary.")
            targets.append(idx)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        return padded, target_tensor


def train_lstm(
    harness: TrainingHarness,
    model: LSTMBaseline,
    epochs: int = 10,
    lr: float = 1e-3,
    clip_norm: float = 1.0,
    batch_size: int = 8,
    use_amp: bool = True,
    log_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    save_every: int = 1,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    amp_enabled = use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    log_dir = Path(log_dir or (ROOT_DIR / "results"))
    checkpoint_dir = Path(checkpoint_dir or (ROOT_DIR / "checkpoints"))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    node_to_idx = {node_id: idx for idx, node_id in enumerate(harness.kg.nodes_by_id.keys())}
    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

    batcher = LSTMBatcher(model, node_to_idx, device)

    best_accuracy = 0.0
    best_checkpoint = checkpoint_dir / f"{harness.config.experiment_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(harness.queries)
        dataset = QueryDataset(harness.queries)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_queries,
        )

        epoch_losses: List[float] = []
        total = 0
        correct = 0

        for batch_queries in dataloader:
            inputs, targets = batcher.encode_queries(batch_queries)

            optimizer.zero_grad()
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, targets)
            else:
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_norm:
                    clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            epoch_losses.append(float(loss.item()))
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += targets.size(0)

        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        accuracy = correct / max(total, 1)
        print(f"[Epoch {epoch}] loss={mean_loss:.4f} accuracy={accuracy:.3f}")

        if epoch % max(save_every, 1) == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model.config.__dict__,
                "accuracy": accuracy,
                "loss": mean_loss,
                "timestamp": datetime.utcnow().isoformat(),
                "idx_to_node": idx_to_node,
            }
            epoch_path = checkpoint_dir / f"{harness.config.experiment_name}_epoch{epoch}.pt"
            torch.save(checkpoint, epoch_path)
            print(f"Saved checkpoint: {epoch_path.name}")

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save(checkpoint, best_checkpoint)
                print(f"[BEST] accuracy {accuracy:.3f} at {best_checkpoint.name}")

        # log summary for this epoch (no per-query logs during training)
        summary_log = log_dir / f"{harness.config.experiment_name}_train.txt"
        with summary_log.open("a", encoding="utf-8") as f:
            f.write(
                f"epoch={epoch}, loss={mean_loss:.4f}, accuracy={accuracy:.3f}\n"
            )

    if best_checkpoint.exists():
        print(f"Best checkpoint: {best_checkpoint.resolve()}")


def load_lstm_checkpoint(checkpoint_path: Path, device: Optional[str] = None) -> Tuple[LSTMBaseline, Dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config = LSTMBaselineConfig(**checkpoint.get("config", {}))
    model = LSTMBaseline(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    if device:
        model = model.to(torch.device(device))
    idx_to_node = checkpoint.get("idx_to_node")
    if idx_to_node is None:
        raise ValueError("Checkpoint missing idx_to_node mapping.")
    print(
        f"Loaded LSTM checkpoint {checkpoint_path.name} "
        f"(epoch={checkpoint.get('epoch')}, accuracy={checkpoint.get('accuracy', 0):.3f})"
    )
    return model, idx_to_node


def eval_lstm(
    harness: TrainingHarness,
    model: LSTMBaseline,
    idx_to_node: Dict[int, str],
    log_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    log_dir = Path(log_dir or (ROOT_DIR / "results"))
    log_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    node_to_idx = {node_id: idx for idx, node_id in idx_to_node.items()}
    batcher = LSTMBatcher(model, node_to_idx, device)

    correct = 0
    total = 0
    total_wall = []

    for idx, query in enumerate(harness.queries, start=1):
        query_tensor, target_tensor = batcher.encode_queries([query])
        start = perf_counter()
        with torch.no_grad():
            logits = model(query_tensor)
        wall_ms = (perf_counter() - start) * 1000.0

        pred_idx = int(logits.argmax(dim=1).item())
        pred_node = idx_to_node.get(pred_idx)
        is_correct = pred_node == query.answer_id
        correct += int(is_correct)
        total += 1
        total_wall.append(wall_ms)

        query_log = QueryLog(
            query_id=query.query_id,
            query_text=query.natural_language,
            ground_truth=query.answer_id,
            prediction=pred_node or "",
            correct=is_correct,
            total_hops=0,
            wall_time_ms=float(wall_ms),
            final_loss=0.0,
            peak_gpu_memory_mb=current_gpu_memory_mb(),
            graph_size_nodes=harness.graph_node_count,
            model_type="lstm_baseline",
            timestamp_iso=datetime.utcnow().isoformat(),
            hops=[],
        )
        harness.logger.log_query(query_log)

        if verbose and (idx % 10 == 0 or idx == len(harness.queries)):
            print(f"  [{idx}/{len(harness.queries)}] accuracy={correct/total:.3f}")

    log_path = log_dir / f"{harness.config.experiment_name}.json"
    harness.save_logs(log_path)

    accuracy = correct / max(total, 1)
    mean_wall = sum(total_wall) / max(len(total_wall), 1)
    print(f"\n[OK] LSTM evaluation complete: accuracy={accuracy:.3f}, mean_wall_time_ms={mean_wall:.2f}")
    print(f"Logs saved to {log_path.resolve()}")

    return {"accuracy": accuracy, "mean_wall_time_ms": mean_wall}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train/Eval LSTM baseline")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--variant", default=None, help="Stress-test variant")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--experiment-name", default=None, help="Experiment name prefix")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--log-dir", default=None, help="Directory for JSON logs")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="Checkpoint every N epochs")
    args = parser.parse_args(argv)

    harness = prepare_harness(
        args.data_dir,
        variant=args.variant,
        split=args.split,
        experiment_name=args.experiment_name or f"lstm_{args.split}",
    )
    print(harness.dataset_summary())

    if args.mode == "train":
        config = LSTMBaselineConfig()
        model = LSTMBaseline(config).to(torch.device(args.device))
        print(f"LSTM parameter count: {model.parameter_count():,}")
        train_lstm(
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
        model, idx_to_node = load_lstm_checkpoint(Path(args.checkpoint), device=args.device)
        model = model.to(torch.device(args.device))
        eval_lstm(
            harness,
            model,
            idx_to_node,
            log_dir=Path(args.log_dir) if args.log_dir else None,
        )


if __name__ == "__main__":
    main()
