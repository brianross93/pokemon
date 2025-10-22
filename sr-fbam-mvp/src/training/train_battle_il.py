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
from typing import Dict, Optional, Tuple, Mapping

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from torch.utils.data import DataLoader, Dataset

from src.pkmn_battle.ingest import BattleDecisionDataset

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


GATE_INDEX_ENCODE = 0
GATE_INDEX_PLAN_LOOKUP = 4
GATE_INDEX_PLAN_STEP = 5


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
    parser.add_argument(
        "--curriculum-config",
        type=Path,
        help="Optional YAML config (defaults to configs/train_plan.yaml when present).",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to load before training (expects torch.save payload with 'model_state').",
    )
    parser.add_argument(
        "--preplan-checkpoint-out",
        type=Path,
        default=None,
        help="If provided, writes model weights before training for calibration baselines.",
    )
    parser.add_argument(
        "--postplan-checkpoint-out",
        type=Path,
        default=None,
        help="If provided, writes model weights after training for calibration comparisons.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze frame encoder (flatten + MLP backbone) to isolate gate head adjustments.",
    )
    parser.add_argument(
        "--freeze-action-head",
        action="store_true",
        help="Freeze action classification head when calibrating gates only.",
    )
    parser.add_argument(
        "--freeze-gate-head",
        action="store_true",
        help="Freeze gate classification head when focusing on the action policy.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=("none", "linear", "cosine", "step"),
        help="Learning-rate scheduler to use during calibration sweeps.",
    )
    parser.add_argument(
        "--scheduler-warmup-epochs",
        type=int,
        default=0,
        help="Warmup epochs for linear scheduler (ignored by other schedulers).",
    )
    parser.add_argument(
        "--scheduler-total-epochs",
        type=int,
        default=0,
        help="Override total epochs seen by scheduler (0 defaults to --epochs).",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=1,
        help="Step size (epochs) for step scheduler.",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.5,
        help="Gamma factor for step scheduler.",
    )
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate for linear/cosine schedulers (absolute value).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name. When set, metrics are logged to W&B.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of W&B tags.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        help="Optional W&B mode override (e.g., 'offline').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    curriculum = _apply_curriculum_defaults(args)

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
    if args.load_checkpoint:
        logging.info("Loading checkpoint from %s", args.load_checkpoint)
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)

    _configure_trainability(model, args)

    if args.preplan_checkpoint_out:
        _save_checkpoint(
            args.preplan_checkpoint_out,
            model,
            {
                "stage": "preplan",
                "train_path": str(args.train),
                "val_path": str(args.val) if args.val else None,
                "num_actions": train_dataset.num_actions,
                "gate_classes": train_dataset.num_gate_targets,
                "plan_feature_dim": train_dataset.plan_feature_dim,
            },
        )
        logging.info("Wrote pre-plan checkpoint to %s", args.preplan_checkpoint_out)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_param_count = _trainable_parameter_count(model)
    if trainable_param_count == 0:
        logging.warning("No trainable parameters remain after freezing; optimizer steps will be skipped.")
    else:
        logging.info("Trainable parameters: %d", trainable_param_count)

    if not trainable_params:
        optimizer: Optional[torch.optim.Optimizer] = None
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    if args.scheduler != "none" and optimizer is None:
        logging.warning("Scheduler '%s' requested but optimizer has no parameters; disabling scheduler.", args.scheduler)
        scheduler = None
    elif optimizer is not None:
        scheduler = _build_scheduler(optimizer, args)
        if scheduler is not None:
            logging.info("Scheduler enabled: %s", args.scheduler)
        else:
            if args.scheduler != "none":
                logging.warning("Scheduler '%s' could not be constructed; proceeding without scheduling.", args.scheduler)
    else:
        scheduler = None

    wandb_logger = _init_wandb(
        args,
        {
            "train_path": str(args.train),
            "val_path": str(args.val) if args.val else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "gate_weight": args.gate_weight,
            "freeze_backbone": bool(args.freeze_backbone),
            "freeze_action_head": bool(args.freeze_action_head),
            "freeze_gate_head": bool(args.freeze_gate_head),
            "scheduler": args.scheduler,
            "scheduler_warmup_epochs": args.scheduler_warmup_epochs,
            "scheduler_total_epochs": args.scheduler_total_epochs if args.scheduler_total_epochs > 0 else args.epochs,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            "scheduler_min_lr": args.scheduler_min_lr,
            "trainable_parameters": trainable_param_count,
            "curriculum": curriculum,
        },
    )

    action_criterion = nn.CrossEntropyLoss()
    gate_criterion = nn.CrossEntropyLoss(reduction="none")

    logging.info(
        "Training on %d decisions (%d actions) for %d epochs",
        len(train_dataset),
        train_dataset.num_actions,
        args.epochs,
    )

    best_val_acc = 0.0
    last_train_metrics: Dict[str, float] = {}
    last_val_metrics: Optional[Dict[str, float]] = None
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
        current_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
        logging.info(
            "[epoch %d] train action_loss %.4f action_acc %.3f gate_loss %.4f gate_acc %.3f",
            epoch,
            last_train_metrics["action_loss"],
            last_train_metrics["action_acc"],
            last_train_metrics["gate_loss"],
            last_train_metrics["gate_acc"],
        )
        logging.info(
            "           gate_encode %.3f gate_lookup %.3f gate_step %.3f gate_skip %.3f adherence_mean %.3f lr %.6e",
            last_train_metrics["gate_encode_frac"],
            last_train_metrics["gate_lookup_frac"],
            last_train_metrics["gate_step_frac"],
            last_train_metrics["gate_skip_frac"],
            last_train_metrics["adherence_mean"],
            current_lr,
        )
        val_metrics: Optional[Dict[str, float]] = None
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
            logging.info(
                "           gate_encode %.3f gate_lookup %.3f gate_step %.3f gate_skip %.3f adherence_mean %.3f",
                val_metrics["gate_encode_frac"],
                val_metrics["gate_lookup_frac"],
                val_metrics["gate_step_frac"],
                val_metrics["gate_skip_frac"],
                val_metrics["adherence_mean"],
            )
            best_val_acc = max(best_val_acc, val_metrics["action_acc"])
            last_val_metrics = val_metrics
        if wandb_logger:
            payload = {
                "epoch": epoch,
                "lr": current_lr,
                "train/action_loss": last_train_metrics["action_loss"],
                "train/action_acc": last_train_metrics["action_acc"],
                "train/gate_loss": last_train_metrics["gate_loss"],
                "train/gate_acc": last_train_metrics["gate_acc"],
                "train/gate_encode_frac": last_train_metrics["gate_encode_frac"],
                "train/gate_lookup_frac": last_train_metrics["gate_lookup_frac"],
                "train/gate_step_frac": last_train_metrics["gate_step_frac"],
                "train/gate_skip_frac": last_train_metrics["gate_skip_frac"],
                "train/adherence_mean": last_train_metrics["adherence_mean"],
            }
            if val_metrics is not None:
                payload.update(
                    {
                        "val/action_loss": val_metrics["action_loss"],
                        "val/action_acc": val_metrics["action_acc"],
                        "val/gate_loss": val_metrics["gate_loss"],
                        "val/gate_acc": val_metrics["gate_acc"],
                        "val/gate_encode_frac": val_metrics["gate_encode_frac"],
                        "val/gate_lookup_frac": val_metrics["gate_lookup_frac"],
                        "val/gate_step_frac": val_metrics["gate_step_frac"],
                        "val/gate_skip_frac": val_metrics["gate_skip_frac"],
                        "val/adherence_mean": val_metrics["adherence_mean"],
                    }
                )
            wandb_logger.log(payload, step=epoch)
    if scheduler is not None:
        scheduler.step()

    train_stats = _extract_gate_stats(last_train_metrics)
    val_stats = _extract_gate_stats(last_val_metrics) if last_val_metrics else None
    curriculum_alerts = _evaluate_curriculum_alerts(curriculum, train_stats)
    if curriculum_alerts:
        for name, info in curriculum_alerts.items():
            logging.warning("Curriculum alert: %s %s", name, info)
    else:
        logging.info(
            "Curriculum gating metrics within targets (encode %.3f lookup %.3f step %.3f adherence %.3f)",
            train_stats["encode"],
            train_stats["lookup"],
            train_stats["step"],
            train_stats["adherence_mean"],
        )
    if curriculum.get("enabled"):
        curriculum["train_stats"] = train_stats
        if val_stats:
            curriculum["val_stats"] = val_stats
        if curriculum_alerts:
            curriculum["alerts"] = curriculum_alerts

    final_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else None

    if args.postplan_checkpoint_out:
        _save_checkpoint(
            args.postplan_checkpoint_out,
            model,
            {
                "stage": "postplan",
                "train_path": str(args.train),
                "val_path": str(args.val) if args.val else None,
                "num_actions": train_dataset.num_actions,
                "gate_classes": train_dataset.num_gate_targets,
                "plan_feature_dim": train_dataset.plan_feature_dim,
            },
        )
        logging.info("Wrote post-plan checkpoint to %s", args.postplan_checkpoint_out)

    wandb_run_id = None
    if wandb_logger:
        run = getattr(wandb_logger, "run", None)
        wandb_run_id = getattr(run, "id", None)

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
            "trainable_parameters": trainable_param_count,
            "freeze_backbone": bool(args.freeze_backbone),
            "freeze_action_head": bool(args.freeze_action_head),
            "freeze_gate_head": bool(args.freeze_gate_head),
            "scheduler": args.scheduler,
            "scheduler_warmup_epochs": args.scheduler_warmup_epochs,
            "scheduler_total_epochs": args.scheduler_total_epochs if args.scheduler_total_epochs > 0 else args.epochs,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            "scheduler_min_lr": args.scheduler_min_lr,
            "scheduler_enabled": bool(scheduler),
            "final_lr": final_lr,
            "wandb_project": args.wandb_project,
            "wandb_run_id": wandb_run_id,
            "wandb_mode": args.wandb_mode,
            "curriculum": curriculum,
            "train_gate_encode_frac": train_stats["encode"],
            "train_gate_lookup_frac": train_stats["lookup"],
            "train_gate_step_frac": train_stats["step"],
            "train_gate_skip_frac": train_stats["skip"],
            "train_adherence_mean": train_stats["adherence_mean"],
            "train_adherence_positive_frac": train_stats["adherence_positive"],
            "train_gate_hist": train_stats["hist"],
        }
        if val_stats:
            payload.update(
                {
                    "val_gate_encode_frac": val_stats["encode"],
                    "val_gate_lookup_frac": val_stats["lookup"],
                    "val_gate_step_frac": val_stats["step"],
                    "val_gate_skip_frac": val_stats["skip"],
                    "val_adherence_mean": val_stats["adherence_mean"],
                    "val_adherence_positive_frac": val_stats["adherence_positive"],
                    "val_gate_hist": val_stats["hist"],
                }
            )
        if curriculum_alerts:
            payload["curriculum_alerts"] = curriculum_alerts
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(payload, indent=2))
        logging.info("Metrics written to %s", args.metrics_out)

    if wandb_logger:
        wandb_logger.finish()


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
    gate_hist = torch.zeros(model.gate_head.out_features, dtype=torch.long)
    adherence_sum = 0.0
    adherence_positive = 0

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
        adherence_sum += adherence_flags.sum().item()
        adherence_positive += (adherence_flags > 0.5).sum().item()
        gate_hist += torch.bincount(gate_targets.detach().cpu(), minlength=gate_hist.numel())

    denom = max(1, total_examples)
    gate_total = max(1, int(gate_hist.sum().item()))
    encode_fraction = gate_hist[GATE_INDEX_ENCODE].item() / gate_total
    lookup_fraction = gate_hist[min(GATE_INDEX_PLAN_LOOKUP, gate_hist.numel() - 1)].item() / gate_total
    step_fraction = gate_hist[min(GATE_INDEX_PLAN_STEP, gate_hist.numel() - 1)].item() / gate_total
    skip_fraction = max(0.0, 1.0 - encode_fraction - lookup_fraction - step_fraction)
    adherence_mean = adherence_sum / denom
    adherence_positive_frac = adherence_positive / denom
    return {
        "action_loss": total_action_loss / denom,
        "action_acc": action_correct / denom,
        "gate_loss": total_gate_loss / denom,
        "gate_acc": gate_correct / denom,
        "gate_hist": gate_hist.tolist(),
        "gate_encode_frac": encode_fraction,
        "gate_lookup_frac": lookup_fraction,
        "gate_step_frac": step_fraction,
        "gate_skip_frac": skip_fraction,
        "adherence_mean": adherence_mean,
        "adherence_positive_frac": adherence_positive_frac,
        "examples": total_examples,
    }


def _build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    schedule = (args.scheduler or "none").lower()
    if schedule == "none":
        return None

    total_epochs = args.scheduler_total_epochs if args.scheduler_total_epochs > 0 else args.epochs
    if total_epochs <= 0:
        logging.warning("Scheduler requested but total epochs is <= 0; skipping scheduler.")
        return None

    if schedule == "linear":
        warmup = max(0, args.scheduler_warmup_epochs)
        base_lr = max(args.lr, 1e-12)
        min_lr = max(0.0, args.scheduler_min_lr)
        min_ratio = min(min_lr / base_lr, 1.0)

        if warmup >= total_epochs:
            warmup = max(total_epochs - 1, 0)

        def lr_lambda(epoch_index: int) -> float:
            step = epoch_index + 1
            if warmup > 0 and step <= warmup:
                return 1.0
            progress_steps = max(total_epochs - warmup, 1)
            progress = min(1.0, (step - warmup) / progress_steps)
            return (1.0 - progress) * (1.0 - min_ratio) + min_ratio

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    if schedule == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=max(0.0, args.scheduler_min_lr),
        )

    if schedule == "step":
        step_size = max(1, args.scheduler_step_size)
        gamma = args.scheduler_gamma
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    logging.warning("Unknown scheduler '%s'; disabling scheduler.", schedule)
    return None


def _init_wandb(args: argparse.Namespace, config: Dict[str, object]):
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        logging.error("W&B logging requested but the 'wandb' package is not installed.")
        return None

    init_kwargs = {"project": project, "config": config}
    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name
    if args.wandb_tags:
        init_kwargs["tags"] = args.wandb_tags
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode

    wandb.init(**init_kwargs)
    return wandb


def _configure_trainability(model: BattlePolicyMLP, args: argparse.Namespace) -> None:
    if getattr(args, "freeze_backbone", False):
        logging.info("Freezing backbone parameters for calibration.")
        for module in (model.flatten, model.backbone):
            for param in module.parameters():
                param.requires_grad = False
    if getattr(args, "freeze_action_head", False):
        logging.info("Freezing action head parameters.")
        for param in model.action_head.parameters():
            param.requires_grad = False
    if getattr(args, "freeze_gate_head", False):
        logging.info("Freezing gate head parameters.")
        for param in model.gate_head.parameters():
            param.requires_grad = False


def _save_checkpoint(path: Path, model: BattlePolicyMLP, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "metadata": metadata,
        "model_class": model.__class__.__name__,
    }
    torch.save(payload, path)


def _trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _extract_gate_stats(metrics: Optional[Mapping[str, object]]) -> Dict[str, float]:
    if not metrics:
        return {
            "encode": 0.0,
            "lookup": 0.0,
            "step": 0.0,
            "skip": 0.0,
            "adherence_mean": 0.0,
            "adherence_positive": 0.0,
            "hist": [],
        }
    encode = float(metrics.get("gate_encode_frac", 0.0))
    lookup = float(metrics.get("gate_lookup_frac", 0.0))
    step = float(metrics.get("gate_step_frac", 0.0))
    skip = float(metrics.get("gate_skip_frac", 0.0))
    if skip <= 0.0:
        skip = max(0.0, 1.0 - encode - lookup - step)
    adherence_mean = float(metrics.get("adherence_mean", 0.0))
    adherence_positive = float(metrics.get("adherence_positive_frac", 0.0))
    hist = metrics.get("gate_hist") or []
    if isinstance(hist, list):
        hist_values = [int(x) for x in hist]
    else:
        hist_values = []
    return {
        "encode": encode,
        "lookup": lookup,
        "step": step,
        "skip": skip,
        "adherence_mean": adherence_mean,
        "adherence_positive": adherence_positive,
        "hist": hist_values,
    }


def _evaluate_curriculum_alerts(curriculum: Mapping[str, object], stats: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    heuristics = curriculum.get("gating_heuristics") if isinstance(curriculum, Mapping) else None
    if not isinstance(heuristics, Mapping):
        return {}
    alerts: Dict[str, Dict[str, float]] = {}

    mixed = heuristics.get("mixed_curriculum") if isinstance(heuristics, Mapping) else None
    if isinstance(mixed, Mapping):
        encode_band = mixed.get("encode_band")
        lookup_band = mixed.get("plan_lookup_band")
        step_band = mixed.get("plan_step_band")
        adherence_target = mixed.get("adherence_target")

        if isinstance(encode_band, (list, tuple)) and len(encode_band) == 2:
            if not (encode_band[0] <= stats["encode"] <= encode_band[1]):
                alerts["encode_band"] = {"value": stats["encode"], "band": list(encode_band)}
        if isinstance(lookup_band, (list, tuple)) and len(lookup_band) == 2:
            if not (lookup_band[0] <= stats["lookup"] <= lookup_band[1]):
                alerts["plan_lookup_band"] = {"value": stats["lookup"], "band": list(lookup_band)}
        if isinstance(step_band, (list, tuple)) and len(step_band) == 2:
            if not (step_band[0] <= stats["step"] <= step_band[1]):
                alerts["plan_step_band"] = {"value": stats["step"], "band": list(step_band)}
        if isinstance(adherence_target, (int, float)) and stats["adherence_mean"] < float(adherence_target):
            alerts["adherence_target"] = {"value": stats["adherence_mean"], "target": float(adherence_target)}

    alert_cfg = heuristics.get("alerts") if isinstance(heuristics, Mapping) else None
    if isinstance(alert_cfg, Mapping):
        skip_threshold = alert_cfg.get("skip_spike_threshold")
        if isinstance(skip_threshold, (int, float)) and stats["skip"] > float(skip_threshold):
            alerts["skip_spike"] = {"value": stats["skip"], "threshold": float(skip_threshold)}

    return alerts


def _apply_curriculum_defaults(args: argparse.Namespace) -> Dict[str, object]:
    """
    Load curriculum config (if available) and adjust training arguments to stay in sync
    with the Phase 5 schedule (epochs, scheduler, logging metadata).
    """
    curriculum_info: Dict[str, object] = {"enabled": False}
    path = args.curriculum_config
    if path is None:
        default_path = Path("configs") / "train_plan.yaml"
        if default_path.exists():
            path = default_path
    if path is None or not path.exists():
        return curriculum_info
    if yaml is None:
        logging.warning("Curriculum config %s requested but PyYAML is not installed.", path)
        return curriculum_info

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    curriculum_info["enabled"] = True
    curriculum_info["path"] = str(path)

    mix_cfg = config.get("mix") if isinstance(config, dict) else {}
    schedule = mix_cfg.get("schedule") if isinstance(mix_cfg, dict) else None
    scheduler_cfg = mix_cfg.get("scheduler") if isinstance(mix_cfg, dict) else {}
    sampler_cfg = mix_cfg.get("sampler") if isinstance(mix_cfg, dict) else {}

    if schedule:
        curriculum_info["schedule"] = schedule
        final_stage = schedule[-1]
        end_epoch = int(final_stage.get("end_epoch", args.epochs))
        if end_epoch > args.epochs:
            logging.info("Curriculum extending epochs from %d -> %d", args.epochs, end_epoch)
            args.epochs = end_epoch
        curriculum_info["target_mix"] = {
            "battle_fraction": final_stage.get("battle_fraction"),
            "overworld_fraction": final_stage.get("overworld_fraction"),
        }

    scheduler_type = scheduler_cfg.get("type")
    if scheduler_type:
        args.scheduler = scheduler_type.lower()
    if scheduler_cfg.get("min_lr") is not None:
        args.scheduler_min_lr = float(scheduler_cfg["min_lr"])
    if scheduler_cfg.get("warmup_epochs") is not None:
        args.scheduler_warmup_epochs = int(scheduler_cfg["warmup_epochs"])
    if scheduler_cfg.get("total_epochs") is not None:
        args.scheduler_total_epochs = int(scheduler_cfg["total_epochs"])
    curriculum_info["scheduler_config"] = scheduler_cfg

    curriculum_info["sampler"] = sampler_cfg
    curriculum_info["augmentations"] = config.get("augmentations")
    curriculum_info["gating_heuristics"] = config.get("gating_heuristics")

    overworld_req = config.get("overworld_requirements") if isinstance(config, dict) else {}
    curriculum_info["telemetry_sources"] = overworld_req.get("telemetry_sources")

    return curriculum_info


if __name__ == "__main__":
    main()
