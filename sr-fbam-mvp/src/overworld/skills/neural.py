"""
Lightweight neural-inspired controller helpers for overworld skills.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor

from .base import BaseSkill, SkillProgress, SkillStatus


BUTTONS: Dict[str, Dict[str, object]] = {
    "UP": {"kind": "button", "label": "UP"},
    "DOWN": {"kind": "button", "label": "DOWN"},
    "LEFT": {"kind": "button", "label": "LEFT"},
    "RIGHT": {"kind": "button", "label": "RIGHT"},
    "A": {"kind": "button", "label": "A"},
    "B": {"kind": "button", "label": "B"},
    "START": {"kind": "button", "label": "START"},
    "SELECT": {"kind": "button", "label": "SELECT"},
    "WAIT": {"kind": "wait"},
}


class NeuralButtonSkill(BaseSkill):
    """
    Base class that converts SR-FBAM embeddings into button selections.

    These controllers blend a scripted action sequence with a simple
    embedding-driven scorer. They are intentionally lightweight so the
    behaviour remains deterministic prior to fine-tuning.
    """

    script: Sequence[str] = ()
    fallback: str = "WAIT"

    def __init__(self) -> None:
        super().__init__()
        self._step_index = 0
        self._script_index = 0
        self._completed = False
        self._failure_reason: Optional[str] = None
        self.set_planner_hint(None)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _embedding(self) -> Tensor:
        summary = self.summary()
        if summary is None or getattr(summary, "embedding", None) is None:
            return torch.zeros(1)
        embedding = summary.embedding
        if embedding.ndim == 0:
            return embedding.unsqueeze(0)
        return embedding.flatten()

    def _score_candidates(self, candidates: Sequence[str]) -> Tensor:
        vector = self._embedding()
        if vector.numel() == 0:
            return torch.zeros(len(candidates))

        base = torch.tanh(vector.mean()).item()
        offsets = torch.linspace(-0.5, 0.5, len(candidates))
        head = vector[: len(candidates)]
        if head.numel() < len(candidates):
            head = torch.nn.functional.pad(head, (0, len(candidates) - head.numel()))
        scores = base + offsets + torch.tanh(head)
        return scores

    def _script_action(self) -> str:
        if not self.script:
            return self.fallback
        action = self.script[min(self._script_index, len(self.script) - 1)]
        self._script_index = min(self._script_index + 1, len(self.script))
        return action

    def _neural_action(self, candidates: Sequence[str]) -> str:
        if not candidates:
            return self.fallback
        scores = self._score_candidates(candidates)
        index = int(torch.argmax(scores).item())
        return candidates[index]

    def _resolve_action(self, primary: Sequence[str], backup: Optional[Sequence[str]] = None) -> Dict[str, object]:
        if not primary:
            primary = ()
        action_label = self._script_action()
        if action_label not in primary and backup:
            action_label = self._neural_action(backup)
        elif action_label not in primary:
            action_label = self._neural_action(primary or backup or (self.fallback,))
        return BUTTONS.get(action_label, BUTTONS[self.fallback])

    def _mark_failure(self, reason: str) -> None:
        self._failure_reason = reason

    # ------------------------------------------------------------------ #
    # BaseSkill overrides
    # ------------------------------------------------------------------ #

    def legal_actions(self, observation, graph) -> tuple[Dict[str, object], ...]:
        return tuple(BUTTONS[label] for label in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "WAIT"))

    def select_action(self, observation, graph) -> Dict[str, object]:
        self._step_index += 1
        return BUTTONS[self.fallback]

    def progress(self, graph) -> SkillProgress:
        if self._failure_reason:
            return SkillProgress(status=SkillStatus.STALLED, reason=self._failure_reason)
        if self._completed:
            return SkillProgress(status=SkillStatus.SUCCEEDED)
        return SkillProgress(status=SkillStatus.IN_PROGRESS)

    def on_exit(self, graph) -> None:
        super().on_exit(graph)
        self._completed = False
        self._failure_reason = None


__all__ = ["NeuralButtonSkill", "BUTTONS"]
