"""
Heuristic battle policy with lightweight gate decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from src.pkmn_battle.env import BattleObs, LegalAction
from src.pkmn_battle.graph import GraphMemory

from .action_space import ActionSpace


class BattleGateMode(str, Enum):
    """Discrete gate modes mirroring SR-FBAM operations."""

    WRITE = "WRITE"
    ASSOC = "ASSOC"
    FOLLOW = "FOLLOW"
    HALT = "HALT"
    PLAN_LOOKUP = "PLAN_LOOKUP"
    PLAN_STEP = "PLAN_STEP"


@dataclass
class BattleGateDecision:
    """Telemetry describing the policy's gate choice."""

    mode: BattleGateMode
    reason: str
    encode_flag: bool
    confidence: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)


class HeuristicBattlePolicy:
    """
    Simple policy that chooses a move or switch based on current telemetry.

    This policy is intentionally lightweight; it demonstrates how a gate
    decision can be surfaced without requiring a trained controller.
    """

    def select(
        self,
        *,
        obs: BattleObs,
        legal_actions: Sequence[LegalAction],
        action_mask: Sequence[float],
        index_map: Dict[int, LegalAction],
        graph: GraphMemory,
        action_space: ActionSpace,
    ) -> Tuple[LegalAction, BattleGateDecision]:
        if not legal_actions:
            return (
                {"kind": "forfeit", "index": 0, "meta": {"reason": "no-legal"}},
                BattleGateDecision(
                    mode=BattleGateMode.HALT,
                    reason="no_legal_actions",
                    encode_flag=False,
                    metadata={"mask": list(action_mask)},
                ),
            )

        best_move = self._pick_best_move(obs, legal_actions)
        if best_move is not None:
            decision = self._gate_for_move(obs, graph, best_move)
            return best_move, decision

        switch_action = self._pick_switch(legal_actions)
        if switch_action is not None:
            return (
                switch_action,
                BattleGateDecision(
                    mode=BattleGateMode.ASSOC,
                    reason="switch_fallback",
                    encode_flag=False,
                    metadata={"index": switch_action.get("index")},
                ),
            )

        return (
            {"kind": "forfeit", "index": 0, "meta": {"reason": "no-usable-actions"}},
            BattleGateDecision(
                mode=BattleGateMode.HALT,
                reason="fallback_forfeit",
                encode_flag=False,
                metadata={"mask": list(action_mask)},
            ),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _pick_best_move(
        self,
        obs: BattleObs,
        legal_actions: Sequence[LegalAction],
    ) -> Optional[LegalAction]:
        best: Optional[LegalAction] = None
        best_score = -1
        for action in legal_actions:
            if action.get("kind") != "move":
                continue
            meta = action.get("meta", {})
            if meta.get("disabled"):
                continue
            pp = int(meta.get("pp", 0))
            if pp <= 0:
                continue
            if pp > best_score:
                best_score = pp
                best = action
        return best

    def _pick_switch(self, legal_actions: Sequence[LegalAction]) -> Optional[LegalAction]:
        for action in legal_actions:
            if action.get("kind") == "switch":
                return action
        return None

    def _gate_for_move(
        self,
        obs: BattleObs,
        graph: GraphMemory,
        action: LegalAction,
    ) -> BattleGateDecision:
        pokemon = obs.get("my_active", {})
        species = int(pokemon.get("species") or 0)
        node_id = f"pokemon:my:0:{species}" if species else None
        metadata = {
            "species": species,
            "selected_move": action.get("meta", {}).get("move_id"),
            "pp": action.get("meta", {}).get("pp"),
        }
        if node_id and graph.assoc(type_="Pokemon", key=node_id):
            return BattleGateDecision(
                mode=BattleGateMode.FOLLOW,
                reason="graph_follow_move",
                encode_flag=False,
                metadata=metadata,
            )
        return BattleGateDecision(
            mode=BattleGateMode.WRITE,
            reason="direct_observation_move",
            encode_flag=False,
            metadata=metadata,
        )
