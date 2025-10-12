from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import json
import time

import numpy as np
import torch


@dataclass
class HopTrace:
    """Single hop within a recall chain."""

    hop_number: int
    action: str  # ASSOC, FOLLOW, VOTE, WRITE, HALT
    query: str
    result: str
    confidence: float
    timestamp_ms: float
    gpu_memory_mb: float


@dataclass
class QueryLog:
    """Structured record for a single query execution."""

    query_id: str
    query_text: str
    ground_truth: str
    prediction: str
    correct: bool
    total_hops: int
    wall_time_ms: float
    final_loss: float
    peak_gpu_memory_mb: float
    graph_size_nodes: int
    model_type: str
    timestamp_iso: str
    hops: List[HopTrace]


class ExperimentLogger:
    """Accumulates structured logs for an experiment run."""

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self._logs: List[QueryLog] = []
        self._start_time = time.time()

    def log_query(self, query_log: QueryLog) -> None:
        self._logs.append(query_log)

    def to_json(self) -> List[Dict]:
        return [asdict(log) for log in self._logs]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    def aggregate_metrics(self) -> Dict[str, float]:
        if not self._logs:
            return {}
        losses = np.array([log.final_loss for log in self._logs], dtype=float)
        wall_times = np.array([log.wall_time_ms for log in self._logs], dtype=float)
        accuracies = np.array([1.0 if log.correct else 0.0 for log in self._logs], dtype=float)
        hops = np.array([log.total_hops for log in self._logs], dtype=float)
        return {
            "mean_loss": float(losses.mean()),
            "mean_wall_time_ms": float(wall_times.mean()),
            "accuracy": float(accuracies.mean()),
            "mean_hops": float(hops.mean()),
        }


def current_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))