from __future__ import annotations

from typing import List

from src.pkmn_battle.env import BattleObs, EnvAdapter, LegalAction
from src.pkmn_battle.extractor import Extractor
from src.srfbam.tasks.battle import SRFBAMBattleAgent


class _Extractor(Extractor):
    def extract(self, obs: BattleObs):  # type: ignore[override]
        return []


class _Env(EnvAdapter):
    def __init__(self) -> None:
        self._turn = 0

    def reset(self) -> BattleObs:  # type: ignore[override]
        self._turn = 0
        return self.observe()

    def observe(self) -> BattleObs:  # type: ignore[override]
        return {"turn": self._turn, "raw": {}}

    def legal_actions(self) -> List[LegalAction]:  # type: ignore[override]
        return [
            {"kind": "move", "index": 0},
            {"kind": "switch", "index": 0},
        ]

    def step(self, action: LegalAction) -> BattleObs:  # type: ignore[override]
        self._turn += 1
        return self.observe()


def _make_agent() -> SRFBAMBattleAgent:
    return SRFBAMBattleAgent(env=_Env(), extractor=_Extractor())


def test_plan_context_defaults_to_none() -> None:
    agent = _make_agent()
    agent.reset()
    context = agent.planlet_context()
    assert context["planlet_id"] == "NONE"
    assert context["planlet_kind"] == "NONE"


def test_plan_context_reflects_set_metadata() -> None:
    agent = _make_agent()
    agent.reset()
    agent.set_planlet_context(
        plan_id="battle-plan-1",
        planlet_id="pl-100",
        planlet_kind="BATTLE",
        source="llm",
        cache_hit=False,
        metadata={"goal": "win quickly"},
    )
    agent.step()
    telemetry = agent.telemetry()
    context = telemetry.plan_context()
    assert context["planlet_id"] == "pl-100"
    assert context["planlet_kind"] == "BATTLE"
    assert context["source"] == "llm"
    assert context["goal"] == "win quickly"

    payload = telemetry.to_payload()
    plan_payload = payload["core"]["plan"]
    assert plan_payload["planlet_id"] == "pl-100"
    assert plan_payload["source"] == "llm"
