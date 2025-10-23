from __future__ import annotations

import base64
import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from jsonschema import Draft7Validator

from pkmn_battle.summarizer import GraphSummary
from ..llm_client import LLMClient
from .schema import PLANLET_SCHEMA

PLANLET_SYSTEM_PROMPT = (
    "You synthesise SR-FBAM planlets for both battle and overworld contexts.\n"
    "Always respond with strict JSON that matches PLANLET_SCHEMA; never emit prose or legacy fallback formats.\n"
    "Your JSON must be entirely comment-free—do not include //, /* ... */, #, or any other annotations.\n"
    "An image of the current Game Boy screen may be attached to the user message. Use it, together with the symbolic summary, to reason about menus, dialog boxes, letter grids, cursor positions, and unexpected overlays.\n"
    "When the overworld summary or image indicates an open menu, emit a planlet of kind `MENU_SEQUENCE` with an args.buttons array describing the exact button presses (e.g. START, A, DOWN, B).\n"
    "When the naming screen appears, select the canonical defaults: name the player whatever you want :) and the rival `BLUE` by either choosing the preset option or typing the letters. Compute the directional button sequence required to highlight each letter before pressing A.\n"
    "Example MENU_SEQUENCE planlet: {\"planlet_id\": \"boot_start_a\", \"kind\": \"MENU_SEQUENCE\", \"seed_frame_id\": 0, \"format\": \"title_menu\", \"side\": \"p1\", \"goal\": \"Reach overworld from title screen\", \"args\": {\"buttons\": [\"START\", \"A\", \"A\"]}, \"script\": [{\"op\": \"MENU_SEQUENCE\", \"buttons\": [\"START\", \"A\", \"A\"]}], \"timeout_steps\": 120}.\n"
    "Ensure every planlet includes ids, kinds, args, script entries, and timeout_steps as required by PLANLET_SCHEMA."
)

WEB_SEARCH_TOOL = {
    "type": "web_search",
}


@dataclass
class PlanletProposal:
    planlet: Dict[str, Any]
    summary: GraphSummary
    search_calls: int = 0
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    token_usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    source: str = "llm"
    cache_hit: bool = False
    cache_key: Optional[str] = None


class PlanletProposer:
    def __init__(self) -> None:
        self._validator = Draft7Validator(PLANLET_SCHEMA)

    def propose_stub(
        self,
        summary: GraphSummary,
        retrieved_docs: Optional[List[Dict[str, str]]] = None,
    ) -> PlanletProposal:
        if summary.format.lower().startswith("gen"):
            planlet = {
                "planlet_id": f"stub_{summary.turn}",
                "id": f"stub_{summary.turn}",
                "kind": "BATTLE",
                "seed_frame_id": summary.turn,
                "format": summary.format,
                "side": summary.side,
                "goal": "Placeholder battle planlet",
                "rationale": "Stub rationale for testing",
                "pre": [],
                "script": [
                    {
                        "op": "ATTACK",
                        "actor": "stub",
                        "move": "stub",
                        "target": "opponent_active",
                    }
                ],
                "timeout_steps": 600,
            }
        else:
            planlet = {
                "planlet_id": f"stub_{summary.turn}",
                "id": f"stub_{summary.turn}",
                "kind": "MENU_SEQUENCE",
                "seed_frame_id": summary.turn,
                "format": summary.format,
                "side": summary.side,
                "goal": "Placeholder menu sequence",
                "rationale": "Stub rationale for testing",
                "args": {"buttons": ["A"]},
                "pre": [],
                "script": [
                    {"op": "MENU_SEQUENCE", "buttons": ["A"]},
                ],
                "timeout_steps": 120,
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
        frame_image: Optional[bytes] = None,
        mission_plan: Optional[Mapping[str, Any]] = None,
    ) -> PlanletProposal:
        """Generate a planlet via the configured LLM client."""

        payload = summary.to_payload()
        user_prompt = (
            "PLANLET_REQUEST\n"
            "You are acting for side: {side}\n"
            "Battle summary JSON:\n{state}\n"
            "Respond with a single JSON object that conforms to PLANLET_SCHEMA.\n"
        ).format(side=summary.side, state=json.dumps(payload, indent=2, sort_keys=True))

        if mission_plan is not None:
            user_prompt += (
                "\nMission scratchpad JSON:\n"
                f"{json.dumps(mission_plan, indent=2)}\n"
                "If you change mission progress, include an `updates` object that merges into this structure."
            )

        tool_descriptions: List[str] = []
        config = getattr(client, "config", None)
        if config is not None and hasattr(config, "tools"):
            current_tools = list(getattr(config, "tools", []) or [])
            if allow_search:
                if not any(tool.get("type") == "web_search" for tool in current_tools):
                    current_tools.append(copy.deepcopy(WEB_SEARCH_TOOL))
                tool_descriptions.append(
                    "- `web_search`: query the live web for context (each call hits the Retrieval API and incurs extra cost)."
                )
            else:
                current_tools = [tool for tool in current_tools if tool.get("type") != "web_search"]
            config.tools = current_tools
        elif allow_search:
            tool_descriptions.append(
                "- `web_search`: query the live web for context (each call hits the Retrieval API and incurs extra cost)."
            )

        if tool_descriptions:
            user_prompt += "\nAvailable tools:\n" + "\n".join(tool_descriptions)

        if frame_image:
            b64 = base64.b64encode(frame_image).decode("ascii")
            user_prompt += (
                "\nThe following PNG encodes the latest frame. "
                "Use it to infer menu/dialog/cursor state when useful.\n"
                f"FRAME_PNG_BASE64:\n{b64}\n"
            )

        messages: List[Dict[str, Any]]
        messages = [{"role": "system", "content": PLANLET_SYSTEM_PROMPT}]
        model_name = getattr(getattr(client, "config", None), "model", "") or ""
        supports_image = any(token in model_name.lower() for token in ("gpt", "o1"))
        if frame_image is not None and supports_image:
            image_b64 = base64.b64encode(frame_image).decode("utf-8")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}" }},
                    ],
                }
            )
        else:
            if frame_image is not None:
                snippet = base64.b64encode(frame_image).decode("utf-8")[:8000]
                user_prompt = f"{user_prompt}\n(Frame PNG base64 snippet: {snippet}...)"
            messages.append({"role": "user", "content": user_prompt})

        raw = client.generate_response(messages)
        content, token_usage = self._normalise_llm_output(raw)
        planlet = self._parse_planlet(content, summary=summary)
        self._validator.validate(planlet)

        return PlanletProposal(
            planlet=planlet,
            summary=summary,
            search_calls=len(planlet.get("retrieved_docs") or []),
            retrieved_docs=planlet.get("retrieved_docs"),
            token_usage=token_usage,
            raw_response=content,
        )

    def _parse_planlet(self, raw: str, *, summary: GraphSummary) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            decoder = json.JSONDecoder()
            try:
                data, _ = decoder.raw_decode(raw)
            except Exception:
                raise ValueError(f"LLM did not return valid JSON: {raw}") from exc

        if not isinstance(data, Mapping):
            raise ValueError(f"LLM returned non-object JSON: {data}")

        # Common wrappers: {"planlet": {...}} or {"planlets": [...]}
        if "planlet" in data and isinstance(data["planlet"], Mapping):
            data = data["planlet"]
        elif "planlets" in data and isinstance(data["planlets"], list) and data["planlets"]:
            first = data["planlets"][0]
            if isinstance(first, Mapping):
                data = first

        # Minimal menu sequence support: {"buttons": [...]} or {"commands": [...]}
        if "planlet_id" not in data and (
            "buttons" in data or "commands" in data
        ):
            buttons = data.get("buttons") or data.get("commands")
            if not isinstance(buttons, list) or not buttons:
                raise ValueError("MENU_SEQUENCE planlet must include non-empty 'buttons' array.")
            timeout_steps = data.get("timeout_steps")
            if not isinstance(timeout_steps, int) or timeout_steps <= 0:
                timeout_steps = max(60, 12 * len(buttons))
            planlet_id = str(data.get("planlet_id") or data.get("id") or "menu_sequence_planlet")
            goal = str(data.get("goal") or "Execute menu sequence")
            return {
                "planlet_id": planlet_id,
                "kind": "MENU_SEQUENCE",
                "seed_frame_id": int(data.get("seed_frame_id", summary.turn)),
                "format": str(data.get("format", summary.format)),
                "side": str(data.get("side", summary.side)),
                "goal": goal,
                "script": [
                    {"op": "MENU_SEQUENCE", "buttons": [str(btn).upper() for btn in buttons]}
                ],
                "timeout_steps": timeout_steps,
            }

        return dict(data)

    def _normalise_llm_output(self, raw: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        if isinstance(raw, str):
            return raw, None
        if isinstance(raw, Mapping):
            content = raw.get("content")
            usage = raw.get("usage")
            return self._stringify_content(content), usage  # type: ignore[arg-type]
        if hasattr(raw, "content"):
            content = getattr(raw, "content")
            usage = getattr(raw, "usage", None)
            return self._stringify_content(content), usage
        return self._stringify_content(raw), None

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (bytes, bytearray)):
            return content.decode("utf-8", errors="replace")
        try:
            return json.dumps(content)
        except TypeError:
            return str(content)
