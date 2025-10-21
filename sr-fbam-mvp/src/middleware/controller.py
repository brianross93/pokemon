"""Controller utilities that connect emulator telemetry to knowledge stores."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from src.knowledge.knowledge_graph import (
    Context,
    EncounterKB,
    KnowledgeGraph,
    KnowledgeRule,
)

from .pokemon_adapter import PokemonTelemetry


class GateMode(str, Enum):
    ASSOC = "ASSOC"
    FOLLOW = "FOLLOW"
    HALT = "HALT"
    WRITE = "WRITE"


@dataclass
class GateDecision:
    """Result returned by the symbolic controller."""

    mode: GateMode
    reason: str
    context: Optional[Context] = None
    rule: Optional[KnowledgeRule] = None
    target_context: Optional[Context] = None


class SymbolicController:
    """
    Implements the ASSOC/FOLLOW/WRITE/HALT gate for Pokemon middleware.

    The controller does three things:
      1. Update the encounter model and knowledge graph with new telemetry (`observe`).
      2. Decide whether to stay in the current context or move to a better one (`decide`).
      3. Surface recommendations or request LLM help when knowledge is insufficient.
    """

    def __init__(
        self,
        target_species_id: int,
        target_entity_name: str,
        game: str,
        encounter_kb: Optional[EncounterKB] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        min_confidence: float = 0.75,
        min_rule_samples: int = 5,
    ) -> None:
        self.target_species_id = int(target_species_id)
        self.target_entity_name = target_entity_name
        self.game = game
        self.encounters = encounter_kb or EncounterKB()
        self.knowledge = knowledge_graph or KnowledgeGraph()
        self.min_confidence = float(min_confidence)
        self.min_rule_samples = int(min_rule_samples)
        self._last_step_counter: Optional[int] = None
        self._last_elapsed_ms: Optional[float] = None
        self._prev_in_battle = False

    def _build_context(self, telemetry: PokemonTelemetry) -> Context:
        time_bucket = telemetry.extra.get("time_bucket") if telemetry.extra else None
        return Context(
            game=self.game,
            area_id=telemetry.area_id,
            method=telemetry.method,
            time_bucket=time_bucket,
        )

    def observe(self, telemetry: PokemonTelemetry) -> GateDecision:
        """Update knowledge stores based on the latest telemetry."""

        ctx = self._build_context(telemetry)
        steps = telemetry.step_counter
        elapsed_ms = telemetry.elapsed_ms

        if self._last_step_counter is None:
            self._last_step_counter = steps
        if self._last_elapsed_ms is None:
            self._last_elapsed_ms = elapsed_ms

        time_delta = max(0.0, elapsed_ms - self._last_elapsed_ms)
        encounter_started = telemetry.in_battle and not self._prev_in_battle

        self.encounters.step(ctx, encounter_started)

        if encounter_started:
            species_id = telemetry.encounter_species_id or -1
            self.encounters.encounter(ctx, species_id, self.target_species_id)
            success = species_id == self.target_species_id
            quality = 1.0 if telemetry.in_grass else 0.7
            self.knowledge.add_evidence(
                entity=self.target_entity_name,
                context={
                    "game": self.game,
                    "area_id": telemetry.area_id,
                    "method": telemetry.method,
                },
                outcome="encounter",
                success=success,
                time_spent=time_delta,
                quality=quality,
                metadata={
                    "species_id": species_id,
                    "step_counter": steps,
                },
            )

        self._prev_in_battle = telemetry.in_battle
        self._last_step_counter = steps
        self._last_elapsed_ms = elapsed_ms

        return GateDecision(mode=GateMode.WRITE, reason="updated_knowledge", context=ctx)

    def decide(
        self,
        telemetry: PokemonTelemetry,
        candidate_contexts: Sequence[Context] | None = None,
    ) -> GateDecision:
        """
        Decide whether to stay, move, or escalate to the LLM.

        Args:
            telemetry: latest emulator snapshot.
            candidate_contexts: optional contexts reachable via scripted routes.
        """

        ctx = self._build_context(telemetry)
        current_context = {
            "game": self.game,
            "area_id": telemetry.area_id,
            "method": telemetry.method,
        }

        rule = self.knowledge.get_best_action(self.target_entity_name, current_context)

        if rule and rule.confidence >= self.min_confidence and rule.evidence_count >= self.min_rule_samples:
            return GateDecision(
                mode=GateMode.FOLLOW,
                reason="confident_rule",
                context=ctx,
                rule=rule,
            )

        if candidate_contexts:
            best_ctx = self.encounters.thompson_pick_context(candidate_contexts, self.target_species_id)
            if best_ctx and (best_ctx.area_id != ctx.area_id or best_ctx.method != ctx.method):
                return GateDecision(
                    mode=GateMode.ASSOC,
                    reason="better_context_available",
                    context=ctx,
                    target_context=best_ctx,
                )

        return GateDecision(
            mode=GateMode.HALT,
            reason="insufficient_knowledge",
            context=ctx,
            rule=rule,
        )

    def summarise(self, telemetry: PokemonTelemetry) -> dict:
        """Return a compact summary of the current context and beliefs."""
        ctx = self._build_context(telemetry)
        summary = self.encounters.summary(ctx, self.target_species_id)
        summary.update(
            {
                "confidence": None,
                "evidence_count": 0,
            }
        )
        best_rule = self.knowledge.get_best_action(
            self.target_entity_name,
            {
                "game": self.game,
                "area_id": telemetry.area_id,
                "method": telemetry.method,
            },
        )
        if best_rule:
            summary["confidence"] = best_rule.confidence
            summary["evidence_count"] = best_rule.evidence_count
        return summary
