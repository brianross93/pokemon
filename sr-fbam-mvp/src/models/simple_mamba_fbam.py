"""
Simplified Mamba-FBAM: placeholder implementation without mamba-ssm dependency.

The original project planned to reproduce a selective state-space variant in
pure PyTorch, but the implementation was never completed.  The stub below
keeps the module importable so compile checks succeed, while making it clear
that the model is not yet implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass
class SimpleMambaConfig:
    """Minimal configuration placeholder for the unfinished Mamba baseline."""

    hidden_dim: int = 256
    action_embedding_dim: int = 32


class SimpleMambaFBAM(nn.Module):
    """
    Placeholder module signalling the missing Mamba-FBAM implementation.

    Any attempt to instantiate or use this model raises ``NotImplementedError``.
    """

    def __init__(self, num_actions: int, config: SimpleMambaConfig | None = None) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.config = config or SimpleMambaConfig()

        raise NotImplementedError(
            "SimpleMambaFBAM is a placeholder; the Mamba baseline was not implemented."
        )

    def forward(self, frames: Sequence[Tensor]) -> Tensor:  # pragma: no cover
        raise NotImplementedError(
            "SimpleMambaFBAM is a placeholder; call sites should not reach here."
        )
