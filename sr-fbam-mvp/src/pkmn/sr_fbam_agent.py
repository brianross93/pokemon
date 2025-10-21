"""
SR-FBAM middleware for Pokemon telemetry.
"""
from __future__ import annotations

from srfbam.core import EncodedFrame, SRFBAMCore, SrfbamStepSummary


class SRFBAMPokemonAgent(SRFBAMCore):
    """
    Thin domain wrapper around the task-agnostic SR-FBAM core.

    The Pokemon-specific adapters can extend this class to add bespoke
    policy heads or telemetry hooks without duplicating the neural core.
    """

    def encode_step(self, encoded: EncodedFrame) -> SrfbamStepSummary:
        # Delegate to the shared SR-FBAM core while preserving the public signature.
        return super().encode_step(encoded)

