"""Utilities for supervising the memory gate with hindsight labels."""

from __future__ import annotations

from typing import List, Sequence

import torch

from src.data.frame_dataset import FrameActionEpisode
from src.models.sr_fbam_code import MemoryOp, SRFBAMCodeAgent


@torch.no_grad()
def compute_hindsight_labels(
    model: SRFBAMCodeAgent,
    episode: FrameActionEpisode,
    lambda_compute: float,
    device: torch.device,
) -> List[MemoryOp]:
    """
    Derive a per-step supervision signal for the memory gate by contrasting
    ``ENCODE`` and ``ASSOC`` counterfactuals.

    Args:
        model: SR-FBAM agent (will be temporarily switched to eval mode).
        episode: trajectory to analyse.
        lambda_compute: trade-off between action confidence and latency.
        device: torch device for temporary tensors.

    Returns:
        List of ``MemoryOp`` entries with length equal to the number of steps.
    """
    steps = len(episode.steps)
    if steps == 0:
        return []

    original_training_mode = model.training
    model.eval()

    labels: List[MemoryOp] = []
    encode_latency = float(model.config.encode_latency_ms)
    query_latency = float(model.config.assoc_latency_ms)

    try:
        for idx in range(steps):
            encode_plan: List[MemoryOp] = labels + [MemoryOp.ENCODE] + [MemoryOp.ENCODE] * (steps - idx - 1)
            query_plan: List[MemoryOp] = labels + [MemoryOp.ASSOC] + [MemoryOp.ENCODE] * (steps - idx - 1)

            rollout_encode = model.forward_episode(
                episode,
                memory_labels=encode_plan,
                teacher_force_actions=True,
            )
            rollout_query = model.forward_episode(
                episode,
                memory_labels=query_plan,
                teacher_force_actions=True,
            )

            action_logits_encode = rollout_encode.action_logits_env()
            action_logits_query = rollout_query.action_logits_env()
            if action_logits_encode.size(0) <= idx or action_logits_query.size(0) <= idx:
                labels.append(MemoryOp.ENCODE)
                continue

            action_index = int(episode.steps[idx].action_index)
            probs_encode = torch.softmax(action_logits_encode[idx], dim=0)
            probs_query = torch.softmax(action_logits_query[idx], dim=0)
            p0 = float(probs_encode[action_index])
            p1 = float(probs_query[action_index])

            delta = (p1 - p0) - lambda_compute * (query_latency - encode_latency)
            labels.append(MemoryOp.ASSOC if delta > 0.0 else MemoryOp.ENCODE)
    finally:
        model.train(original_training_mode)

    return labels
