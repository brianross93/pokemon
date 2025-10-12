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

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import sys

# Ensure the package root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data import KnowledgeGraph, Query, load_dataset
from src.evaluation.logger_schema import ExperimentLogger


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


if __name__ == "__main__":
    # Manual smoke test for loader wiring
    default_data_dir = Path(__file__).parent.parent.parent / "data"
    harness = prepare_harness(default_data_dir.as_posix())
    print(harness.dataset_summary())
