"""
Lightweight dataclasses shared across SR-FBAM tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass
class EncodedFrame:
    """
    Dense frame representation produced by a domain-specific encoder.

    Attributes
    ----------
    grid:
        Tokenised 2D representation (typically ASCII grid flattened) that
        feeds the Transformer symbol extractor.
    features:
        Numeric features describing the frame; shape `[1, F]`.
    context_key:
        Hashable key identifying the high-level context (e.g. file path,
        encounter area) used for persistent memory lookups.
    extra:
        Arbitrary metadata carried through for downstream modules.
    """

    grid: Tensor
    features: Tensor
    context_key: str
    extra: Dict[str, int]


@dataclass
class SrfbamStepSummary:
    """
    Summary emitted after the SR-FBAM core processes a frame.

    The embedding and gate statistics are shared with downstream
    controllers and telemetry sinks.
    """

    embedding: Tensor
    symbol_embedding: Tensor
    numeric_features: Tensor
    gate_stats: Dict[str, float]
    context_key: str


def ensure_tensor_device(tensor: Tensor, device: torch.device) -> Tensor:
    """Return `tensor` placed on `device` if it is not already there."""
    if tensor.device == device:
        return tensor
    return tensor.to(device)

