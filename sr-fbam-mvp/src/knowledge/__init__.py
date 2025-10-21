"""
Lightweight knowledge-graph utilities for symbolic recurrence middleware.

The submodule exposes dataclasses and managers that keep track of evidence,
Bayesian posteriors, and high-level rules that can be queried or refined.
"""

from .knowledge_graph import (  # noqa: F401
    BetaConfidence,
    Context,
    EncounterKB,
    Evidence,
    KnowledgeGraph,
    KnowledgeRule,
    Posterior,
    RefinementResult,
)
