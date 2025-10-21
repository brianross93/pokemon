from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pkmn_battle.graph.memory import GraphMemory
from pkmn_battle.summarizer import summarize_for_llm
from pkmn_overworld.summarizer import summarize_world_for_llm
from src.overworld import OverworldMemory

from ..llm_client import LLMClient, LLMConfig
from .proposer import PlanletProposal, PlanletProposer
from src.plan.storage import PlanletRecord, PlanletStore
from src.plan.cache import PlanCache, CacheHit


@dataclass
class PlanletService:
    """
    End-to-end helper connecting the battle/overworld graph summaries with the planlet proposer.
    """

    proposer: PlanletProposer
    client: LLMClient
    store: Optional[PlanletStore] = None
    cache: Optional[PlanCache] = None
    _plan_cache_index: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def request_planlet(
        self,
        memory: GraphMemory,
        *,
        side_view: str,
        allow_search: bool = True,
    ) -> PlanletProposal:
        summary = summarize_for_llm(memory, side_view=side_view)
        cache_key: Optional[str] = None
        if self.cache is not None:
            cache_key = self.cache.battle_key(summary)
            hit = self.cache.lookup(cache_key)
            if hit is not None:
                proposal = self._proposal_from_cache(summary, hit)
                self._register_planlet(proposal.planlet.get("planlet_id"), cache_key)
                self._persist_planlet(summary, proposal)
                return proposal

        proposal = self.proposer.generate_planlet(summary, self.client, allow_search=allow_search)
        proposal.cache_key = cache_key
        self._persist_planlet(summary, proposal)
        if self.cache is not None and cache_key is not None:
            self.cache.store(
                cache_key,
                proposal.planlet,
                retrieved_docs=proposal.retrieved_docs,
                token_usage=proposal.token_usage,
                raw_response=proposal.raw_response,
                metadata={
                    "domain": "battle",
                    "side": summary.side,
                    "turn": summary.turn,
                    "format": summary.format,
                },
            )
            self._register_planlet(proposal.planlet.get("planlet_id"), cache_key)
        return proposal

    def request_overworld_planlet(
        self,
        memory: OverworldMemory,
        *,
        nearby_limit: int = 5,
        allow_search: bool = True,
    ) -> PlanletProposal:
        summary = summarize_world_for_llm(memory, nearby_limit=nearby_limit)
        cache_key: Optional[str] = None
        if self.cache is not None:
            cache_key = self.cache.overworld_key(summary, nearby_limit=nearby_limit)
            hit = self.cache.lookup(cache_key)
            if hit is not None:
                proposal = self._proposal_from_cache(summary, hit)
                self._register_planlet(proposal.planlet.get("planlet_id"), cache_key)
                self._persist_planlet(summary, proposal)
                return proposal

        proposal = self.proposer.generate_planlet(summary, self.client, allow_search=allow_search)
        proposal.cache_key = cache_key
        self._persist_planlet(summary, proposal)
        if self.cache is not None and cache_key is not None:
            self.cache.store(
                cache_key,
                proposal.planlet,
                retrieved_docs=proposal.retrieved_docs,
                token_usage=proposal.token_usage,
                raw_response=proposal.raw_response,
                metadata={
                    "domain": "overworld",
                    "map_id": summary.map_id,
                    "side": summary.side,
                },
            )
            self._register_planlet(proposal.planlet.get("planlet_id"), cache_key)
        return proposal

    def record_feedback(self, planlet_id: str, *, success: bool, weight: float = 1.0) -> None:
        if not planlet_id or self.cache is None:
            return
        cache_key = self._plan_cache_index.get(planlet_id)
        if cache_key is not None:
            self.cache.record_feedback(cache_key, success, weight=weight)
        else:
            self.cache.record_feedback_by_planlet(planlet_id, success, weight=weight)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _proposal_from_cache(self, summary, hit: CacheHit) -> PlanletProposal:
        proposal = PlanletProposal(
            planlet=hit.planlet,
            summary=summary,
            search_calls=0,
            retrieved_docs=hit.retrieved_docs,
            token_usage=hit.token_usage,
            raw_response=hit.raw_response,
            source="cache",
            cache_hit=True,
            cache_key=hit.cache_key,
        )
        return proposal

    def _register_planlet(self, planlet_id: Optional[str], cache_key: Optional[str]) -> None:
        if not planlet_id or not cache_key or self.cache is None:
            return
        self._plan_cache_index[str(planlet_id)] = cache_key
        self.cache.register_planlet(str(planlet_id), cache_key)

    def _persist_planlet(self, summary, proposal: PlanletProposal) -> None:
        if self.store is None:
            return
        planlet = proposal.planlet
        planlet_id = str(planlet.get("planlet_id") or "")
        if not planlet_id:
            return

        raw_kind = str(planlet.get("kind") or "")
        raw_upper = raw_kind.upper()
        if raw_upper == "BATTLE":
            mode = "battle"
        elif raw_upper == "OVERWORLD":
            mode = "overworld"
        else:
            mode = raw_kind.lower() or "battle"
        record = PlanletRecord(
            planlet_id=planlet_id,
            mode="battle" if mode == "battle" else mode,
            planlet_kind=planlet.get("kind"),
            goal=planlet.get("goal"),
            seed_frame_id=planlet.get("seed_frame_id"),
            summary=summary.to_payload(),
            retrieved_docs=proposal.retrieved_docs,
            token_usage=proposal.token_usage,
            llm_model=self._llm_model(),
            llm_config=self._safe_llm_config(getattr(self.client, "config", None)),
            raw_planlet=planlet,
            source=proposal.source,
            cache_key=proposal.cache_key,
            cache_hit=proposal.cache_hit,
            extra={
                "summary_turn": getattr(summary, "turn", None),
                "summary_side": getattr(summary, "side", None),
                "summary_format": getattr(summary, "format", getattr(summary, "map_id", None)),
                "cache_key": proposal.cache_key,
            },
        )
        self.store.append(record)

    def _llm_model(self) -> Optional[str]:
        config = getattr(self.client, "config", None)
        return getattr(config, "model", None)

    @staticmethod
    def _safe_llm_config(config: Optional[LLMConfig]) -> Optional[Dict[str, Any]]:
        if config is None:
            return None
        return {
            key: value
            for key, value in config.__dict__.items()
            if key != "api_key"
        }
