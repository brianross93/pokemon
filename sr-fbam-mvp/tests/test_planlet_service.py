import json

from llm.llm_client import LLMClient, LLMConfig
from llm.planlets import PlanletProposer, PlanletService
from pkmn_battle.graph.memory import GraphMemory
from pkmn_battle.graph.schema import Node, WriteOp


class _StubLLMClient(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.config = LLMConfig(model="mock")

    def generate_response(self, messages):
        return self.response


def test_planlet_service_requests_planlet():
    memory = GraphMemory()
    memory.write(WriteOp(kind="node", payload=Node(type="turn", node_id="turn-1", attributes={"turn": 5})))
    memory.write(WriteOp(kind="node", payload=Node(type="format", node_id="fmt", attributes={"name": "gen9ou"})))

    planlet_response = {
        "planlet_id": "pl_service",
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
    assert proposal.summary.side == "p1"
