"""
Battle plan interpreter that converts planlet steps into legal actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from src.plan.planner_llm import PlanletSpec
from src.pkmn_battle.env import BattleObs, LegalAction
from .catalog import move_id_from_name, species_id_from_name


@dataclass
class PlanDecision:
    status: str  # "action", "complete", "abort"
    action: Optional[LegalAction] = None
    reason: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class BattlePlanExecutor:
    """Lightweight interpreter for battle planlets."""

    def __init__(self, planlet: PlanletSpec, *, plan_id: Optional[str] = None) -> None:
        self.planlet = planlet
        self.plan_id = plan_id
        self.step_index = 0
        self._started = False
        self._precondition_ok = True
        self._precondition_errors: List[str] = []

    @property
    def steps_total(self) -> int:
        return len(self.planlet.script)

    def start(self, obs: BattleObs) -> None:
        if self._started:
            return
        self._precondition_ok, self._precondition_errors = self._evaluate_preconditions(obs)
        self._started = True

    @property
    def preconditions_satisfied(self) -> bool:
        return self._precondition_ok

    def precondition_errors(self) -> List[str]:
        return list(self._precondition_errors)

    def next_action(self, obs: BattleObs, legal_actions: Sequence[LegalAction]) -> PlanDecision:
        if not self._started:
            self.start(obs)
        if not self._precondition_ok:
            return PlanDecision(
                status="abort",
                reason="precondition-failed",
                metadata=self._metadata(status="rejected", info={"errors": list(self._precondition_errors)}),
            )

        while self.step_index < len(self.planlet.script):
            step = self.planlet.script[self.step_index]
            decision = self._execute_step(step, obs, legal_actions, self.step_index)
            if decision.status == "action":
                self.step_index += 1
                return decision
            if decision.status == "skip":
                self.step_index += 1
                continue
            if decision.status == "abort":
                return decision

        return PlanDecision(
            status="complete",
            metadata=self._metadata(status="completed", info={"step_index": self.step_index}),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _metadata(self, *, status: str, info: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "planlet_id": self.planlet.id,
            "plan_id": self.plan_id,
            "status": status,
            "steps_total": len(self.planlet.script),
        }
        if info:
            payload.update(dict(info))
        return payload

    def _execute_step(
        self,
        step: Mapping[str, object],
        obs: BattleObs,
        legal_actions: Sequence[LegalAction],
        index: int,
    ) -> PlanDecision:
        op = str(step.get("op", "")).upper()
        metadata = self._metadata(
            status="executing",
            info={"step_index": index, "op": op},
        )

        if op in {"ATTACK", "STATUS", "SET_HAZARD"}:
            move_id = self._resolve_move_id(step)
            if move_id is None:
                return PlanDecision(
                    status="abort",
                    reason="move-unresolved",
                    metadata={**metadata, "error": step.get("move")},
                )
            action = _find_move_action(legal_actions, move_id)
            if action is None:
                fallback = step.get("fallback")
                if isinstance(fallback, Mapping):
                    return self._execute_step(fallback, obs, legal_actions, index)
                return PlanDecision(
                    status="abort",
                    reason="move-unavailable",
                    metadata={**metadata, "move_id": move_id},
                )
            metadata["move_id"] = move_id
            return PlanDecision(status="action", action=action, metadata=metadata)

        if op in {"BRING_IN", "SWITCH"}:
            species_id = self._resolve_species_id(step, obs)
            if species_id is None:
                return PlanDecision(
                    status="abort",
                    reason="species-unresolved",
                    metadata=metadata,
                )
            action = _find_switch_action(legal_actions, species_id)
            if action is None:
                return PlanDecision(
                    status="abort",
                    reason="switch-unavailable",
                    metadata={**metadata, "species_id": species_id},
                )
            metadata["species_id"] = species_id
            return PlanDecision(status="action", action=action, metadata=metadata)

        if op == "WAIT":
            return PlanDecision(status="skip", metadata=metadata)

        return PlanDecision(
            status="abort",
            reason="unsupported-op",
            metadata={**metadata, "error": op},
        )

    def _resolve_move_id(self, step: Mapping[str, object]) -> Optional[int]:
        move_id = step.get("move_id")
        if isinstance(move_id, int):
            return move_id
        move_name = step.get("move") or step.get("hazard")
        if isinstance(move_name, str):
            return move_id_from_name(move_name)
        return None

    def _resolve_species_id(self, step: Mapping[str, object], obs: BattleObs) -> Optional[int]:
        raw_id = step.get("species_id")
        if isinstance(raw_id, int):
            return raw_id
        actor = step.get("actor") or step.get("species") or step.get("target")
        if isinstance(actor, str):
            resolved = species_id_from_name(actor)
            if resolved is not None:
                return resolved
        slot = step.get("slot")
        if isinstance(slot, int):
            party = obs.get("my_party", [])
            if isinstance(party, list) and 0 <= slot < len(party):
                candidate = party[slot]
                if isinstance(candidate, Mapping):
                    value = candidate.get("species")
                    if isinstance(value, int):
                        return value
        return None

    def _evaluate_preconditions(self, obs: BattleObs) -> Tuple[bool, List[str]]:
        failures: List[str] = []
        active = obs.get("my_active", {}) if isinstance(obs, Mapping) else {}
        moves = []
        if isinstance(active, Mapping):
            moves = active.get("moves", [])
        move_ids = {
            move.get("move_id")
            for move in moves
            if isinstance(move, Mapping) and isinstance(move.get("move_id"), int)
        }

        for cond in self.planlet.pre:
            if isinstance(cond, str):
                continue
            if not isinstance(cond, Mapping):
                continue
            op = str(cond.get("op", "")).upper()
            if op == "HAS_MOVE":
                move_name = cond.get("move") or cond.get("value")
                move_id = cond.get("move_id")
                if move_id is None and isinstance(move_name, str):
                    move_id = move_id_from_name(move_name)
                if move_id not in move_ids:
                    failures.append(f"missing-move:{move_name}")
            elif op == "ACTIVE_SPECIES":
                actor = cond.get("species") or cond.get("value") or cond.get("name")
                species_id = cond.get("species_id")
                if species_id is None and isinstance(actor, str):
                    species_id = species_id_from_name(actor)
                active_species = active.get("species") if isinstance(active, Mapping) else None
                if species_id is not None and species_id != active_species:
                    failures.append(f"unexpected-active:{actor}")
            elif op == "HAS_PARTY":
                actor = cond.get("species") or cond.get("value") or cond.get("name")
                species_id = cond.get("species_id")
                if species_id is None and isinstance(actor, str):
                    species_id = species_id_from_name(actor)
                if species_id is not None:
                    party_species = {
                        mon.get("species")
                        for mon in obs.get("my_party", [])
                        if isinstance(mon, Mapping)
                    }
                    if species_id not in party_species:
                        failures.append(f"missing-party:{actor}")
            else:
                continue
        return (not failures, failures)


def _find_move_action(legal_actions: Sequence[LegalAction], move_id: int) -> Optional[LegalAction]:
    for action in legal_actions:
        if action.get("kind") != "move":
            continue
        meta = action.get("meta", {})
        if isinstance(meta, Mapping) and int(meta.get("move_id", -1)) == int(move_id):
            return dict(action)
    return None


def _find_switch_action(legal_actions: Sequence[LegalAction], species_id: int) -> Optional[LegalAction]:
    for action in legal_actions:
        if action.get("kind") != "switch":
            continue
        meta = action.get("meta", {})
        if isinstance(meta, Mapping) and int(meta.get("species", -1)) == int(species_id):
            return dict(action)
    return None


__all__ = ["BattlePlanExecutor", "PlanDecision"]
