from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jsonschema import Draft7Validator

from pkmn_battle.summarizer import GraphSummary
from ..llm_client import LLMClient
from .schema import PLANLET_SCHEMA

PLANLET_SYSTEM_PROMPT = (
    "You are the planlet synthesis module for the SR-FBAM battle agent. "
    "Given a structured battle summary, produce a planlet JSON object that "
    "matches the documented schema. Use concise scripts (<=4 steps) and only "
    "legal Pokémon moves and actions. If you need external knowledge, you may "
    "invoke the `web_search` tool described in the OpenAI web-search guide "
    "(https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses). "
    "Do not return prose—emit strict JSON only."
)

WEB_SEARCH_TOOL = {
    "type": "web_search",
    "web_search": {
        "mode": "results",
        "retrieval": {"scope": "web"},
    },
}


@dataclass
class PlanletProposal:
    planlet: Dict[str, Any]
    summary: GraphSummary
    search_calls: int = 0
    retrieved_docs: Optional[List[Dict[str, str]]] = None


class PlanletProposer:
    def __init__(self) -> None:
        self._validator = Draft7Validator(PLANLET_SCHEMA)

    def propose_stub(
        self,
        summary: GraphSummary,
        retrieved_docs: Optional[List[Dict[str, str]]] = None,
    ) -> PlanletProposal:
        planlet = {
            "planlet_id": f"stub_{summary.turn}",
            "seed_frame_id": summary.turn,
            "format": summary.format,
            "side": summary.side,
            "goal": "Placeholder goal",
            "rationale": "Stub rationale for testing",
            "preconditions": [],
            "script": [
                {
                    "op": "ATTACK",
                    "actor": "stub",
                    "move": "stub",
                    "target": "opponent_active",
                }
            ],
        }
        if retrieved_docs:
            planlet["retrieved_docs"] = retrieved_docs
        self._validator.validate(planlet)
        return PlanletProposal(
            planlet=planlet,
            summary=summary,
            search_calls=len(retrieved_docs or []),
            retrieved_docs=retrieved_docs,
        )

    def generate_planlet(
        self,
        summary: GraphSummary,
        client: LLMClient,
        *,
        allow_search: bool = True,
    ) -> PlanletProposal:
        """Generate a planlet via the configured LLM client."""

        payload = summary.to_payload()
        user_prompt = (
            "PLANLET_REQUEST\n"
            "You are acting for side: {side}\n"
            "Battle summary JSON:\n{state}\n"
            "Respond with a single JSON object that conforms to PLANLET_SCHEMA.\n"
        ).format(side=summary.side, state=json.dumps(payload, indent=2, sort_keys=True))

        messages = [
            {"role": "system", "content": PLANLET_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        original_tools = getattr(client, "config", None)
        if original_tools and allow_search:
            if isinstance(original_tools.tools, list):
                if WEB_SEARCH_TOOL not in original_tools.tools:
                    original_tools.tools.append(WEB_SEARCH_TOOL)

        raw = client.generate_response(messages)
        planlet = self._parse_planlet(raw)
        self._validator.validate(planlet)

        return PlanletProposal(
            planlet=planlet,
            summary=summary,
            search_calls=len(planlet.get("retrieved_docs") or []),
            retrieved_docs=planlet.get("retrieved_docs"),
        )

    def _parse_planlet(self, raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON: {raw}") from exc
