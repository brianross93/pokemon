"""
Minimal SR-FBAM battle imitation learning trainer.

This script wires the new BattleDecisionDataset into a lightweight MLP so we
can sanity-check the end-to-end data flow on CPU. It is not intended to replace
the production trainer, but provides a fast feedback loop while the full model
stack is implemented.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pkmn_battle.ingest import BattleDecisionDataset


class BattlePolicyMLP(nn.Module):
    def __init__(self, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40 * 120, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple MLP on SR-FBAM battle decisions.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Path to training JSONL emitted by build_il_corpus.py.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=None,
        help="Optional validation JSONL file.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        help="Optional path to write JSON metrics (accuracy, loss).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    train_dataset = BattleDecisionDataset(args.train)
    val_dataset = (
        BattleDecisionDataset(args.val) if args.val is not None else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if val_dataset is not None
        else None
    )

    model = BattlePolicyMLP(
        num_actions=train_dataset.num_actions,
        hidden_dim=args.hidden_dim,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logging.info(
        "Training on %d decisions (%d actions) for %d epochs",
        len(train_dataset),
        train_dataset.num_actions,
        args.epochs,
    )

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        logging.info(
            "[epoch %d] train loss %.4f acc %.3f",
            epoch,
            train_loss,
            train_acc,
        )
        if val_loader:
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer=None, device=device, train=False
            )
            logging.info(
                "[epoch %d] val   loss %.4f acc %.3f",
                epoch,
                val_loss,
                val_acc,
            )
            best_val_acc = max(best_val_acc, val_acc)

    if args.metrics_out:
        payload = {
            "train_path": str(args.train),
            "val_path": str(args.val) if args.val else None,
            "num_train": len(train_dataset),
            "num_actions": train_dataset.num_actions,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "best_val_accuracy": best_val_acc,
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(payload, indent=2))
        logging.info("Metrics written to %s", args.metrics_out)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(frames)
            loss = criterion(logits, labels)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * frames.size(0)
        total_examples += frames.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    average_loss = total_loss / max(1, total_examples)
    accuracy = correct / max(1, total_examples)
    return average_loss, accuracy


if __name__ == "__main__":
    main()
