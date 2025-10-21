"""
Behavioural cloning trainer for overworld trace datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.overworld.action_space import OverworldActionSpace
from src.overworld.ingest import OverworldDecisionDataset, VIEW_TO_INDEX


class OverworldBCModel(nn.Module):
    """Two-head MLP predicting action logits and gate decisions."""

    def __init__(self, feature_dim: int, plan_feature_dim: int, action_space: OverworldActionSpace) -> None:
        super().__init__()
        hidden_dim = 64
        input_dim = feature_dim + plan_feature_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_space.size)
        self.encode_head = nn.Linear(hidden_dim, 1)
        self.view_head = nn.Linear(hidden_dim, len(VIEW_TO_INDEX))

    def forward(self, features: Tensor, plan_features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if plan_features.dim() == 1:
            plan_features = plan_features.unsqueeze(0)
        combined = torch.cat([features, plan_features], dim=-1)
        embedding = self.backbone(combined)
        return self.policy_head(embedding), self.encode_head(embedding), self.view_head(embedding)


def collate_batch(
    batch: Sequence[Tuple[Tensor, Tensor, int, Tensor, Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    frame = torch.stack([item[0] for item in batch], dim=0)
    plan = torch.stack([item[1] for item in batch], dim=0)
    actions = torch.tensor([item[2] for item in batch], dtype=torch.long)
    encode = torch.stack([item[3] for item in batch], dim=0)
    view = torch.stack([item[4] for item in batch], dim=0)
    success = torch.stack([item[5] for item in batch], dim=0)
    return frame, plan, actions, encode, view, success


def run_epoch(
    model: OverworldBCModel,
    dataloader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    total_correct_actions = 0
    total_encode_matches = 0
    total_view_matches = 0
    encode_predictions: List[float] = []
    encode_targets: List[float] = []
    view_predictions: List[int] = []
    view_targets: List[int] = []
    success_flags: List[float] = []

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    for features, plan, actions, encode, view, success in dataloader:
        features = features.to(device)
        plan = plan.to(device)
        actions = actions.to(device)
        encode = encode.to(device)
        view = view.to(device)
        success_flags.extend(success.cpu().tolist())

        policy_logits, encode_logits, view_logits = model(features, plan)

        loss_policy = ce(policy_logits, actions)
        loss_encode = bce(encode_logits.squeeze(-1), encode)
        loss_view = ce(view_logits, view)
        loss = loss_policy + loss_encode + loss_view

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_examples += features.size(0)

        pred_actions = policy_logits.argmax(dim=-1)
        total_correct_actions += (pred_actions == actions).sum().item()

        pred_encode = torch.sigmoid(encode_logits.squeeze(-1)) > 0.5
        total_encode_matches += (pred_encode.float() == encode).sum().item()
        encode_predictions.extend(pred_encode.float().cpu().tolist())
        encode_targets.extend(encode.cpu().tolist())

        pred_view = view_logits.argmax(dim=-1)
        total_view_matches += (pred_view == view).sum().item()
        view_predictions.extend(pred_view.cpu().tolist())
        view_targets.extend(view.cpu().tolist())

    metrics = {
        "loss": total_loss / max(1, total_examples),
        "action_acc": total_correct_actions / max(1, total_examples),
        "encode_acc": total_encode_matches / max(1, total_examples),
        "view_acc": total_view_matches / max(1, total_examples),
        "encode_true_frac": sum(encode_targets) / max(1, len(encode_targets)),
        "encode_pred_frac": sum(encode_predictions) / max(1, len(encode_predictions)),
        "success_rate": sum(success_flags) / max(1, len(success_flags)),
        "view_true_mix": _distribution(view_targets, len(VIEW_TO_INDEX)),
        "view_pred_mix": _distribution(view_predictions, len(VIEW_TO_INDEX)),
    }
    return metrics


def _distribution(values: Iterable[int], size: int) -> List[float]:
    counts = [0 for _ in range(size)]
    total = 0
    for value in values:
        if 0 <= value < size:
            counts[value] += 1
            total += 1
    if total == 0:
        return counts
    return [count / total for count in counts]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a behavioural cloning policy on overworld traces.")
    parser.add_argument("--traces", type=Path, nargs="+", required=True, help="One or more trace JSONL files.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    dataset = OverworldDecisionDataset([Path(path) for path in args.traces])
    num_val = int(len(dataset) * max(0.0, min(args.val_split, 0.9)))
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(0))

    def _make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_batch)

    train_loader = _make_loader(train_dataset, shuffle=True)
    val_loader = _make_loader(val_dataset, shuffle=False) if num_val > 0 else None

    feature_dim = dataset.frame_feature_dim
    plan_feature_dim = dataset.plan_feature_dim
    action_space = OverworldActionSpace()
    model = OverworldBCModel(feature_dim, plan_feature_dim, action_space).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_metrics = run_epoch(model, val_loader, optimizer=None, device=device) if val_loader is not None else {}
        summary = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        print(json.dumps(summary, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
