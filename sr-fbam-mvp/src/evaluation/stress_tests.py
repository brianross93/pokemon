from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class StressTestSpec:
    name: str
    description: str
    setup: Dict[str, str]
    expected_failure_mode: str
    target_metrics: Dict[str, str]


STRESS_TEST_SUITE: List[StressTestSpec] = [
    StressTestSpec(
        name="baseline",
        description="Clean KG, unambiguous queries",
        setup={
            "num_queries": "100",
            "kg_nodes": "100",
            "kg_edges": "300",
            "noise_level": "0.0",
            "ambiguity": "none",
        },
        expected_failure_mode="None",
        target_metrics={
            "accuracy": "~0.95",
            "mean_hops": "<=5",
        },
    ),
    StressTestSpec(
        name="ambiguous_nodes",
        description="Multiple entities share surface forms",
        setup={
            "duplicate_entities": "5",
            "tie_breaking": "vote",
        },
        expected_failure_mode="VOTE ties cause accuracy drop",
        target_metrics={
            "accuracy": "~0.80",
        },
    ),
    StressTestSpec(
        name="long_chains",
        description="Queries require 8-15 hops",
        setup={
            "hop_range": "[8,15]",
            "graph_type": "genealogy",
        },
        expected_failure_mode="Error accumulation, HALT before resolution",
        target_metrics={
            "accuracy": "<0.60",
            "mean_hops": ">=10",
        },
    ),
    StressTestSpec(
        name="noisy_edges",
        description="Random spurious relations injected",
        setup={
            "noise_fraction": "0.2",
        },
        expected_failure_mode="FOLLOW retrieves wrong nodes",
        target_metrics={
            "accuracy": "~0.70",
        },
    ),
    StressTestSpec(
        name="missing_nodes",
        description="Queries reference removed entities",
        setup={
            "removed_fraction": "0.1",
        },
        expected_failure_mode="HALT with error or backtrack",
        target_metrics={
            "no_answer_rate": "~0.10",
        },
    ),
    StressTestSpec(
        name="graph_growth",
        description="Dynamic graph with online updates",
        setup={
            "start_nodes": "50",
            "nodes_per_100_queries": "5",
            "prune_threshold": "2x",
        },
        expected_failure_mode="Pruning evicts relevant nodes",
        target_metrics={
            "memory_bound": "maintained",
            "accuracy": "stable",
        },
    ),
]


def run_stress_test(model, spec: StressTestSpec):
    """Placeholder for executing a predefined stress scenario."""

    raise NotImplementedError("Implement during Phase 1 (Overview.md:168-174)")