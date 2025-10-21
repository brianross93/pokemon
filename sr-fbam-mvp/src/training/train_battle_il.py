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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pkmn_battle.ingest import BattleDecisionDataset


class BattlePolicyMLP(nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_gate_classes: int,
        hidden_dim: int,
        plan_feature_dim: int,
    ) -> None:
        super().__init__()
        input_dim = 40 * 120 + plan_feature_dim
        self.flatten = nn.Flatten()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.gate_head = nn.Linear(hidden_dim, num_gate_classes)

    def forward(
        self, frames: torch.Tensor, plan_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_frames = self.flatten(frames)
        if plan_features.dim() == 1:
            plan_features = plan_features.unsqueeze(0)
        x = torch.cat([flat_frames, plan_features], dim=1)
        shared = self.backbone(x)
        return self.action_head(shared), self.gate_head(shared)


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
        "--gate-weight",
        type=float,
        default=0.5,
        help="Contribution of gate loss relative to action loss.",
    )
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

    def _make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
        )

    train_loader = _make_loader(train_dataset, shuffle=True)
    val_loader = (
        _make_loader(val_dataset, shuffle=False)
        if val_dataset is not None
        else None
    )

    model = BattlePolicyMLP(
        num_actions=train_dataset.num_actions,
        num_gate_classes=train_dataset.num_gate_targets,
        hidden_dim=args.hidden_dim,
        plan_feature_dim=train_dataset.plan_feature_dim,
    ).to(device)
    action_criterion = nn.CrossEntropyLoss()
    gate_criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logging.info(
        "Training on %d decisions (%d actions) for %d epochs",
        len(train_dataset),
        train_dataset.num_actions,
        args.epochs,
    )

    best_val_acc = 0.0
    last_train_metrics: Dict[str, float] = {
        "action_loss": 0.0,
        "action_acc": 0.0,
        "gate_loss": 0.0,
        "gate_acc": 0.0,
    }
    for epoch in range(1, args.epochs + 1):
        last_train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            action_criterion=action_criterion,
            gate_criterion=gate_criterion,
            optimizer=optimizer,
            device=device,
            gate_weight=args.gate_weight,
            train=True,
        )
        logging.info(
            "[epoch %d] train action_loss %.4f action_acc %.3f gate_loss %.4f gate_acc %.3f",
            epoch,
            last_train_metrics["action_loss"],
            last_train_metrics["action_acc"],
            last_train_metrics["gate_loss"],
            last_train_metrics["gate_acc"],
        )
        if val_loader:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                action_criterion=action_criterion,
                gate_criterion=gate_criterion,
                optimizer=None,
                device=device,
                gate_weight=args.gate_weight,
                train=False,
            )
            logging.info(
                "[epoch %d] val   action_loss %.4f action_acc %.3f gate_loss %.4f gate_acc %.3f",
                epoch,
                val_metrics["action_loss"],
                val_metrics["action_acc"],
                val_metrics["gate_loss"],
                val_metrics["gate_acc"],
            )
            best_val_acc = max(best_val_acc, val_metrics["action_acc"])

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
            "gate_weight": args.gate_weight,
            "plan_feature_dim": train_dataset.plan_feature_dim,
            "gate_classes": train_dataset.num_gate_targets,
            "best_val_accuracy": best_val_acc,
            "last_train_action_loss": last_train_metrics["action_loss"],
            "last_train_gate_loss": last_train_metrics["gate_loss"],
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(payload, indent=2))
        logging.info("Metrics written to %s", args.metrics_out)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    action_criterion: nn.Module,
    gate_criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    gate_weight: float,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_action_loss = 0.0
    total_gate_loss = 0.0
    total_examples = 0
    action_correct = 0
    gate_correct = 0

    for frames, plan_feats, gate_targets, adherence_flags, labels in loader:
        frames = frames.to(device)
        plan_feats = plan_feats.to(device)
        gate_targets = gate_targets.to(device)
        adherence_flags = adherence_flags.to(device)
        labels = labels.to(device)

        batch_size = frames.size(0)

        with torch.set_grad_enabled(train):
            action_logits, gate_logits = model(frames, plan_feats)
            action_loss = action_criterion(action_logits, labels)
            gate_loss_unreduced = gate_criterion(gate_logits, gate_targets)
            gate_weights = 1.0 + adherence_flags
            gate_loss = (gate_loss_unreduced * gate_weights).mean()
            combined_loss = action_loss + gate_weight * gate_loss

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            combined_loss.backward()
            optimizer.step()

        total_action_loss += action_loss.item() * batch_size
        total_gate_loss += gate_loss.item() * batch_size
        total_examples += batch_size
        action_correct += (action_logits.argmax(dim=1) == labels).sum().item()
        gate_correct += (gate_logits.argmax(dim=1) == gate_targets).sum().item()

    denom = max(1, total_examples)
    return {
        "action_loss": total_action_loss / denom,
        "action_acc": action_correct / denom,
        "gate_loss": total_gate_loss / denom,
        "gate_acc": gate_correct / denom,
    }


if __name__ == "__main__":
    main()
