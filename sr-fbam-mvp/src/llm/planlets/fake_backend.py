from __future__ import annotations

import json
from typing import Dict, List

from llm.llm_client import LLMClient, LLMConfig


class FakeOverworldLLM(LLMClient):
    """Deterministic planner that emits scripted MENU_SEQUENCE planlets matching the intro flow."""

    _SCRIPT: List[Dict[str, object]] = [
        {"goal": "Leave the title screen", "buttons": ["START"]},
        {"goal": "Confirm New Game selection", "buttons": ["A"]},
        {"goal": "Advance Professor Oak's introduction", "buttons": ["A"] * 24},
        {"goal": "Choose preset player name", "buttons": ["DOWN", "A"]},
        {"goal": "Confirm player name dialogue", "buttons": ["A"] * 12},
        {"goal": "Choose preset rival name", "buttons": ["DOWN", "A"]},
        {"goal": "Finish rival dialogue", "buttons": ["A"] * 18},
        {"goal": "Skip final intro animations", "buttons": ["A"] * 20},
        {"goal": "Close any lingering text boxes", "buttons": ["B"]},
    ]

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(model="fake-overworld")
        self._stage = 0

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        if self._stage < len(self._SCRIPT):
            entry = self._SCRIPT[self._stage]
        else:
            entry = {"goal": "Advance dialogue", "buttons": ["A"]}

        buttons = [str(btn).upper() for btn in entry.get("buttons", []) or ["A"]]
        goal = str(entry.get("goal", "Advance dialogue"))
        self._stage += 1

        planlet = {
            "planlet_id": f"fake_overworld_plan_{self._stage}",
            "id": f"fake_overworld_plan_{self._stage}",
            "kind": "MENU_SEQUENCE",
            "seed_frame_id": 0,
            "format": "overworld_intro",
            "side": "p1",
            "goal": goal,
            "rationale": "Scripted heuristic for CI smoke tests",
            "args": {"buttons": buttons},
            "pre": [],
            "post": [],
            "hints": {},
            "script": [{"op": "MENU_SEQUENCE", "buttons": buttons}],
            "timeout_steps": max(60, 12 * len(buttons)),
            "retrieved_docs": [],
        }
        return json.dumps(planlet)
