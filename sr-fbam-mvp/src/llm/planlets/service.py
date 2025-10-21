from __future__ import annotations

from dataclasses import dataclass

from pkmn_battle.graph.memory import GraphMemory
from pkmn_battle.summarizer import summarize_for_llm

from ..llm_client import LLMClient
from .proposer import PlanletProposal, PlanletProposer


@dataclass
class PlanletService:
    """
    End-to-end helper connecting the battle graph, summariser, and planlet proposer.
    """

    proposer: PlanletProposer
    client: LLMClient

    def request_planlet(
        self,
        memory: GraphMemory,
        *,
        side_view: str,
        allow_search: bool = True,
    ) -> PlanletProposal:
        summary = summarize_for_llm(memory, side_view=side_view)
        return self.proposer.generate_planlet(summary, self.client, allow_search=allow_search)
