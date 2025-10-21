from __future__ import annotations

import json
from typing import Dict, List

from llm.llm_client import LLMClient, LLMConfig


class FakeOverworldLLM(LLMClient):
    """Deterministic LLM stub that emits OVERWORLD planlets for CI tests."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(model="fake-overworld")

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        payload = ""
        for message in messages:
            if message["role"] == "user":
                payload = message.get("content", "")
                break
        map_id = "overworld"
        if "\"map_id\"" in payload:
            try:
                start = payload.index("{")
                end = payload.rfind("}")
                summary_json = payload[start : end + 1]
                data = json.loads(summary_json)
                map_id = data.get("map_id", map_id)
            except Exception:
                pass

        planlet = {
            "planlet_id": "fake_overworld_plan",
            "kind": "OVERWORLD",
            "seed_frame_id": 0,
            "format": map_id,
            "side": "p1",
            "goal": "Walk forward",
            "rationale": "Deterministic CI stub",
            "preconditions": [],
            "script": [
                {"op": "NAVIGATE", "actor": "player", "target": {"tile": [0, 0]}},
                {"op": "WAIT", "actor": "player", "duration": 1}
            ],
            "retrieved_docs": [],
        }
        return json.dumps(planlet)
