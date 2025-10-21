from __future__ import annotations

from typing import List

from src.plan.planner_llm import PlanletSpec
from src.pkmn_battle.env import BattleObs, EnvAdapter, LegalAction
from src.pkmn_battle.extractor import Extractor
from src.pkmn_battle.policy.heuristic_policy import BattleGateDecision, BattleGateMode
from src.srfbam.tasks.battle import SRFBAMBattleAgent


class _StubExtractor(Extractor):
    def extract(self, obs: BattleObs):  # type: ignore[override]
        return []


class _PlanEnv(EnvAdapter):
    def __init__(self) -> None:
        self._turn = 0
        self._moves = [
            {"slot": 0, "move_id": 71, "pp": 10, "max_pp": 10, "disabled": False},
            {"slot": 1, "move_id": 1, "pp": 10, "max_pp": 10, "disabled": False},
        ]
        self._party = [
            {"species": 4, "slot": 0, "fainted": False, "is_active": False},
        ]

    def reset(self) -> BattleObs:  # type: ignore[override]
        self._turn = 0
        return self.observe()

    def observe(self) -> BattleObs:  # type: ignore[override]
        return {
            "turn": self._turn,
            "my_active": {"species": 1, "moves": self._moves},
            "my_party": self._party,
        }

    def legal_actions(self) -> List[LegalAction]:  # type: ignore[override]
        return [
            {"kind": "move", "index": 0, "meta": {"move_id": 71, "pp": 10}},
            {"kind": "move", "index": 1, "meta": {"move_id": 1, "pp": 10}},
            {"kind": "switch", "index": 0, "meta": {"species": 4}},
        ]

    def step(self, action: LegalAction) -> BattleObs:  # type: ignore[override]
        self._turn += 1
        return self.observe()


def _policy_stub(
    obs: BattleObs,
    legal: List[LegalAction],
    mask,
    index_map,
    graph,
    action_space,
):
    return legal[0], BattleGateDecision(mode=BattleGateMode.WRITE, reason="stub", encode_flag=False)


def test_battle_agent_executes_plan_step() -> None:
    env = _PlanEnv()
    agent = SRFBAMBattleAgent(env=env, extractor=_StubExtractor(), policy_fn=_policy_stub)
    agent.reset()
    planlet = PlanletSpec(
        id="pl_attack",
        kind="BATTLE",
        script=[{"op": "ATTACK", "move": "Absorb"}],
    )
    assert agent.load_planlet(planlet, plan_id="plan-1", source="llm")
    action = agent.act()
    assert action["kind"] == "move"
    assert action["meta"]["move_id"] == 71
    payload = agent.telemetry().to_payload()
    context = payload["core"].get("plan", {})
    assert context.get("planlet_id") == "pl_attack"
    assert context.get("status") == "executing"

    # Next call should fall back to policy once plan completes
    action2 = agent.act()
    assert action2["kind"] == "move"
    payload2 = agent.telemetry().to_payload()
    context2 = payload2["core"].get("plan", {})
    assert context2.get("status") in {"completed", None, "executing"}


def test_battle_agent_rejects_plan_without_move() -> None:
    env = _PlanEnv()
    agent = SRFBAMBattleAgent(env=env, extractor=_StubExtractor(), policy_fn=_policy_stub)
    agent.reset()
    planlet = PlanletSpec(
        id="pl_bad",
        kind="BATTLE",
        pre=[{"op": "HAS_MOVE", "move": "Surf"}],
        script=[{"op": "ATTACK", "move": "Surf"}],
    )
    assert not agent.load_planlet(planlet, plan_id="plan-2", source="llm")
    payload = agent.telemetry().to_payload()
    context = payload["core"].get("plan", {})
    assert context.get("status") == "rejected"
