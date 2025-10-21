"""
Core building blocks for SR-FBAM controllers.

These modules encapsulate the reusable neural components (memory,
extractor, gating, recurrence) that power both the existing code-editing
experiments and forthcoming domains such as Pokemon battles.
"""

from __future__ import annotations

from .agent import SRFBAMCore
from .types import EncodedFrame, SrfbamStepSummary

__all__ = ["SRFBAMCore", "EncodedFrame", "SrfbamStepSummary"]

