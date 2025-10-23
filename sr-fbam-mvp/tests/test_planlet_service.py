from __future__ import annotations

import json
from pathlib import Path

from llm.llm_client import LLMClient, LLMConfig
from llm.planlets import PlanletProposer, PlanletService
from pkmn_battle.graph.memory import GraphMemory
from pkmn_battle.graph.schema import Node, WriteOp
from src.overworld import OverworldMemory
from src.plan.cache import PlanCache
from src.plan.storage import PlanletStore


class _StubLLMClient(LLMClient):
    def __init__(self, response) -> None:
        self._response = response
        self.config = LLMConfig(model="mock")

    def generate_response(self, messages):
        return self._response


class _CountingLLM(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.config = LLMConfig(model="mock")

    def generate_response(self, messages):
        self.calls += 1
        return self.response


def _make_memory() -> GraphMemory:
    memory = GraphMemory()
    memory.write(WriteOp(kind="node", payload=Node(type="turn", node_id="turn-1", attributes={"turn": 5})))
    memory.write(WriteOp(kind="node", payload=Node(type="format", node_id="fmt", attributes={"name": "gen9ou"})))
    memory.write(WriteOp(kind="node", payload=Node(type="side", node_id="side-p1", attributes={"name": "p1"})))
    return memory


def test_planlet_service_requests_planlet() -> None:
    memory = _make_memory()
    planlet_response = {
        "planlet_id": "pl_service",
        "kind": "BATTLE",
        "seed_frame_id": 5,
        "format": "gen9ou",
        "side": "p1",
        "goal": "Maintain advantage",
        "rationale": "Service test",
        "preconditions": [],
        "script": [
            {"op": "ATTACK", "actor": "stub", "move": "stub", "target": "opponent_active"}
        ],
    }

    client = _StubLLMClient(json.dumps(planlet_response))
    service = PlanletService(proposer=PlanletProposer(), client=client)

    proposal = service.request_planlet(memory, side_view="p1", allow_search=False)

    assert proposal.planlet["planlet_id"] == "pl_service"
    assert proposal.planlet["kind"] == "BATTLE"
    assert proposal.summary.side == "p1"
    assert proposal.token_usage is None
    assert proposal.raw_response == json.dumps(planlet_response)
    assert proposal.cache_hit is False
    assert proposal.cache_key is None


def test_planlet_service_persists_planlet(tmp_path: Path) -> None:
    memory = _make_memory()
    planlet_response = {
        "planlet_id": "pl_from_store",
        "kind": "BATTLE",
        "seed_frame_id": 8,
        "format": "gen9ou",
        "side": "p1",
        "goal": "Store this planlet",
        "script": [
            {"op": "ATTACK", "actor": "stub", "move": "stub", "target": "opponent_active"}
        ],
        "retrieved_docs": [{"id": "doc-1"}],
    }

    client = _StubLLMClient(
        {
            "content": json.dumps(planlet_response),
            "usage": {"prompt_tokens": 9, "completion_tokens": 3},
        }
    )
    store = PlanletStore(tmp_path)
    service = PlanletService(proposer=PlanletProposer(), client=client, store=store)

    proposal = service.request_planlet(memory, side_view="p1", allow_search=True)

    assert proposal.token_usage == {"prompt_tokens": 9, "completion_tokens": 3}
    assert proposal.cache_key is None

    jsonl_path = tmp_path / "planlets.jsonl"
    assert jsonl_path.exists()
    stored = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert stored[0]["planlet_id"] == "pl_from_store"
    assert stored[0]["token_usage"]["prompt_tokens"] == 9
    assert stored[0]["summary"]["turn"] == 5


def test_planlet_service_uses_cache(tmp_path: Path) -> None:
    memory = _make_memory()
    planlet_response = json.dumps(
        {
            "planlet_id": "pl_cache",
            "kind": "BATTLE",
            "seed_frame_id": 5,
            "format": "gen9ou",
            "side": "p1",
            "goal": "Cache me",
            "script": [{"op": "ATTACK", "actor": "stub", "move": "stub", "target": "opponent_active"}],
        }
    )
    client = _CountingLLM(planlet_response)
    cache = PlanCache(max_size=4, ttl_seconds=300.0)
    store = PlanletStore(tmp_path)
    service = PlanletService(proposer=PlanletProposer(), client=client, store=store, cache=cache)

    first = service.request_planlet(memory, side_view="p1", allow_search=False)
    assert client.calls == 1
    assert first.cache_hit is False
    assert first.cache_key is not None
    assert first.source == "llm"

    second = service.request_planlet(memory, side_view="p1", allow_search=True)
    assert client.calls == 1  # cache hit
    assert second.source == "cache"
    assert second.cache_hit is True
    assert second.cache_key == first.cache_key

    service.record_feedback("pl_cache", success=True, weight=2.0)
    stats = cache.stats()
    assert stats["successes"] >= 2.0
    assert "pl_cache" in stats["planlets"].values()


def test_planlet_service_overworld_cache() -> None:
    memory = OverworldMemory()
    planlet_response = json.dumps(
        {
            "planlet_id": "pl_overworld",
            "id": "pl_overworld",
            "kind": "MENU_SEQUENCE",
            "seed_frame_id": 0,
            "format": "map",
            "side": "p1",
            "goal": "Open menu",
            "args": {"buttons": ["START", "A"]},
            "script": [
                {"op": "MENU_SEQUENCE", "buttons": ["START", "A"]},
            ],
            "timeout_steps": 120,
        }
    )
    client = _CountingLLM(planlet_response)
    cache = PlanCache(max_size=2, ttl_seconds=300.0)
    service = PlanletService(proposer=PlanletProposer(), client=client, cache=cache)

    first = service.request_overworld_planlet(memory, allow_search=False)
    assert first.planlet["kind"] == "MENU_SEQUENCE"
    assert first.cache_hit is False
    assert first.cache_key is not None

    second = service.request_overworld_planlet(memory, allow_search=True)
    assert second.cache_hit is True
    assert second.source == "cache"
    assert client.calls == 1
