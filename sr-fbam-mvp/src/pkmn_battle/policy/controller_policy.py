"""
SR-FBAM controller-backed policy for Pokemon battles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from srfbam.core import EncodedFrame, SRFBAMCore

from src.pkmn_battle.env import BattleObs, LegalAction
from src.pkmn_battle.graph import GraphMemory

from .action_space import ActionSpace
from .heuristic_policy import BattleGateDecision, BattleGateMode


def _build_encoded_frame(obs: BattleObs) -> EncodedFrame:
    turn = int(obs.get("turn", 0))
    my_active = obs.get("my_active", {}) or {}
    opp_active = obs.get("opp_active", {}) or {}
    field = obs.get("field", {}) or {}

    def hp_ratio(mon: Dict[str, object]) -> float:
        hp = int(mon.get("hp") or 0)
        hp_max = int(mon.get("hp_max") or 0)
        if hp_max <= 0:
            return 0.0
        return float(max(0, min(hp, hp_max)) / hp_max)

    features = torch.tensor(
        [
            min(turn, 255) / 255.0,
            hp_ratio(my_active),
            hp_ratio(opp_active),
            float(bool(my_active.get("status"))),
            float(bool(opp_active.get("status"))),
            (int(field.get("weather_id") or 0) % 256) / 255.0,
            (int(field.get("player_screens") or 0) % 256) / 255.0,
            (int(field.get("enemy_screens") or 0) % 256) / 255.0,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    grid = torch.zeros((40, 120), dtype=torch.long)
    context_key = f"battle:turn:{turn}"
    return EncodedFrame(grid=grid, features=features, context_key=context_key, extra={})


class BattleActionHead(nn.Module):
    """MLP that maps SR-FBAM embeddings to action logits."""

    def __init__(self, embed_dim: int, symbol_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden = max(64, embed_dim // 2 + symbol_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + symbol_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, embedding: Tensor, symbol_embedding: Tensor) -> Tensor:
        concat = torch.cat([embedding, symbol_embedding], dim=-1)
        return self.net(concat)


@dataclass
class BattleControllerPolicy:
    """
    Uses the SR-FBAM core to generate gate decisions and action logits.
    """

    action_space: ActionSpace
    core: Optional[SRFBAMCore] = None
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        self.core = self.core or SRFBAMCore()
        self.device = self.device or self.core.device
        self.action_head = BattleActionHead(
            embed_dim=self.core.config.integrator_hidden_dim,
            symbol_dim=self.core.config.symbol_feature_dim,
            action_dim=self.action_space.size,
        ).to(self.device)

    def select(
        self,
        obs: BattleObs,
        legal_actions: Sequence[LegalAction],
        action_mask: Sequence[float],
        index_map: Dict[int, LegalAction],
        graph: GraphMemory,
        action_space: ActionSpace,
    ) -> Tuple[LegalAction, BattleGateDecision]:
        encoded = _build_encoded_frame(obs)
        summary = self.core.encode_step(encoded)

        embedding = summary.embedding.unsqueeze(0).to(self.device)
        symbol_embedding = summary.symbol_embedding.unsqueeze(0).to(self.device)
        logits = self.action_head(embedding, symbol_embedding).squeeze(0)

        mask_tensor = torch.tensor(action_mask, device=logits.device, dtype=logits.dtype)
        masked_logits = logits + mask_tensor
        action_index = int(masked_logits.argmax().item())
        self.core.set_last_action_index(action_index)

        action = index_map.get(action_index)
        if action is None:
            action = {"kind": "forfeit", "index": 0, "meta": {"reason": "no-legal-action"}}

        gate_mode = summary.gate_stats.get("decision", "WRITE")
        gate_mode_upper = str(gate_mode).upper()
        if gate_mode_upper == "EXTRACT":
            mode = BattleGateMode.HALT
            encode_flag = True
        elif gate_mode_upper == "CACHE_HIT":
            mode = BattleGateMode.ASSOC
            encode_flag = False
        elif gate_mode_upper == "REUSE":
            mode = BattleGateMode.FOLLOW
            encode_flag = False
        elif gate_mode_upper == BattleGateMode.PLAN_LOOKUP.value:
            mode = BattleGateMode.PLAN_LOOKUP
            encode_flag = False
        elif gate_mode_upper == BattleGateMode.PLAN_STEP.value:
            mode = BattleGateMode.PLAN_STEP
            encode_flag = False
        else:
            mode = BattleGateMode.WRITE
            encode_flag = False
        gate_decision = BattleGateDecision(
            mode=mode,
            reason=gate_mode.lower(),
            encode_flag=encode_flag,
            metadata={
                "cache_hits": summary.gate_stats.get("cache_hits"),
                "reuse": summary.gate_stats.get("reuse"),
                "extract": summary.gate_stats.get("extract"),
            },
        )
        return action, gate_decision
