from __future__ import annotations

from typing import List

from src.plan.planner_llm import PlanletSpec
from src.pkmn_battle.env import LegalAction
from src.pkmn_battle.plan.interpreter import BattlePlanExecutor


def _battle_obs(move_id: int = 71, species: int = 1):
    return {
        "turn": 1,
        "my_active": {
            "species": species,
            "moves": [
                {"move_id": move_id, "pp": 10, "max_pp": 10, "disabled": False},
            ],
        },
        "my_party": [],
    }


def _legal_actions(move_id: int = 71, species: int = 4) -> List[LegalAction]:
    return [
        {"kind": "move", "index": 0, "meta": {"move_id": move_id, "pp": 10}},
        {"kind": "switch", "index": 0, "meta": {"species": species}},
    ]


def test_plan_executor_attack_maps_move() -> None:
    planlet = PlanletSpec(
        id="pl_attack",
        kind="BATTLE",
        script=[{"op": "ATTACK", "move": "Absorb"}],
    )
    executor = BattlePlanExecutor(planlet)
    obs = _battle_obs(move_id=71)
    executor.start(obs)
    decision = executor.next_action(obs, _legal_actions(move_id=71))
    assert decision.status == "action"
    assert decision.action == {"kind": "move", "index": 0, "meta": {"move_id": 71, "pp": 10}}


def test_plan_executor_switch_maps_species() -> None:
    planlet = PlanletSpec(
        id="pl_switch",
        kind="BATTLE",
        script=[{"op": "BRING_IN", "actor": "Charmander"}],
    )
    executor = BattlePlanExecutor(planlet)
    obs = _battle_obs(move_id=71)
    obs["my_party"] = [{"species": 4, "slot": 0}]
    executor.start(obs)
    decision = executor.next_action(obs, _legal_actions(move_id=71, species=4))
    assert decision.status == "action"
    assert decision.action["kind"] == "switch"
    assert decision.action["meta"]["species"] == 4


def test_plan_executor_precondition_failure() -> None:
    planlet = PlanletSpec(
        id="pl_fail",
        kind="BATTLE",
        pre=[{"op": "HAS_MOVE", "move": "Surf"}],
        script=[{"op": "ATTACK", "move": "Surf"}],
    )
    executor = BattlePlanExecutor(planlet)
    obs = _battle_obs(move_id=71)
    executor.start(obs)
    assert executor.precondition_errors()
    decision = executor.next_action(obs, _legal_actions())
    assert decision.status == "abort"
    assert decision.reason == "precondition-failed"
