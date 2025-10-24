import json

from llm.llm_client import LLMClient, LLMConfig
from llm.planlets.proposer import PlanletProposer
from pkmn_battle.graph.memory import GraphMemory
from pkmn_battle.graph.schema import Node, WriteOp
from pkmn_battle.summarizer import summarize_for_llm


class _StubLLMClient(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None
        self.config = LLMConfig(model="mock")

    def generate_response(self, messages):
        self.messages = messages
        return self.response


def test_planlet_proposer_stub_validates_schema():
    memory = GraphMemory()
    memory.write(WriteOp(kind="node", payload=Node(type="turn", node_id="turn-1", attributes={"turn": 4})))
    memory.write(WriteOp(kind="node", payload=Node(type="format", node_id="fmt", attributes={"name": "gen9ou"})))
    summary = summarize_for_llm(memory, side_view="p1")

    proposer = PlanletProposer()
    proposal = proposer.propose_stub(summary)

    assert proposal.planlet["planlet_id"] == "stub_4"
    assert proposal.planlet["kind"] in {"BATTLE", "MENU_SEQUENCE"}
    assert proposal.planlet["side"] == "p1"
    assert proposal.planlet["script"]
    assert proposal.search_calls == 0


def test_generate_planlet_uses_llm_and_validates():
    memory = GraphMemory()
    memory.write(WriteOp(kind="node", payload=Node(type="turn", node_id="turn-1", attributes={"turn": 8})))
    memory.write(WriteOp(kind="node", payload=Node(type="format", node_id="fmt", attributes={"name": "gen9ou"})))
    summary = summarize_for_llm(memory, side_view="p2")

    planlet_response = {
        "planlet_id": "pl_001",
        "kind": "BATTLE",
        "seed_frame_id": 8,
        "format": "gen9ou",
        "side": "p2",
        "goal": "Win the tempo battle",
        "rationale": "Stub planlet for test",
        "preconditions": [],
        "script": [
            {"op": "ATTACK", "actor": "stub", "move": "stub", "target": "opponent_active"}
        ],
        "retrieved_docs": [],
    }
    client = _StubLLMClient(json.dumps(planlet_response))

    proposer = PlanletProposer()
    proposal = proposer.generate_planlet(summary, client, allow_search=False)

    assert proposal.planlet["planlet_id"] == "pl_001"
    assert proposal.planlet["side"] == "p2"
    assert proposal.search_calls == 0
    assert client.messages is not None
    assert any("PLANLET_REQUEST" in msg["content"] for msg in client.messages if msg["role"] == "user")


def test_generate_planlet_prompt_includes_naming_overlay_metadata():
    memory = GraphMemory()
    memory.write(WriteOp(kind="node", payload=Node(type="turn", node_id="turn-1", attributes={"turn": 8})))
    memory.write(WriteOp(kind="node", payload=Node(type="format", node_id="fmt", attributes={"name": "gen9ou"})))
    summary = summarize_for_llm(memory, side_view="p2")

    planlet_response = {
        "planlet_id": "pl_001",
        "kind": "MENU_SEQUENCE",
        "seed_frame_id": 8,
        "format": "gen9ou",
        "side": "p2",
        "goal": "Advance naming overlay",
        "args": {"buttons": ["A"]},
        "script": [{"op": "MENU_SEQUENCE", "buttons": ["A"]}],
        "timeout_steps": 60,
    }
    client = _StubLLMClient(json.dumps(planlet_response))

    proposer = PlanletProposer()
    mission_plan = {
        "environment": {
            "overworld_snapshot": {
                "naming_screen": {
                    "grid_letters": [["R", "E", "D"], ["B", "L", "U", "E"]],
                    "cursor": {"row": 0, "col": 1, "letter": "E"},
                    "cursor_history": [{"row": 0, "col": 1, "letter": "E"}],
                    "presets": [{"label": "RED", "row": 0, "col": 0}],
                },
                "overlay_state": {"naming_active": True},
                "tile_adjacency_stats": {"tracked_tiles": 4, "avg_degree": 2.5},
            }
        }
    }

    proposal = proposer.generate_planlet(
        summary,
        client,
        allow_search=False,
        mission_plan=mission_plan,
    )

    assert proposal.planlet["planlet_id"] == "pl_001"
    assert client.messages is not None
    user_messages = [msg["content"] for msg in client.messages if msg["role"] == "user"]
    assert user_messages
    prompt = user_messages[0]
    assert "overlay_state" in prompt
    assert "naming_screen" in prompt
    assert "tile_adjacency_stats" in prompt
    assert "While overlay_state.naming_active is true" in prompt
