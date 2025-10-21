from __future__ import annotations

from typing import List

from src.pkmn_battle.env import BattleObs, EnvAdapter, LegalAction
from src.pkmn_battle.extractor import Extractor
from src.pkmn_battle.policy import BattleGateDecision, BattleGateMode
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
        return [{"kind": "move", "index": 0}, {"kind": "switch", "index": 0}]

    def step(self, action: LegalAction) -> BattleObs:  # type: ignore[override]
        self._turn += 1
        return self.observe()


def _make_agent() -> SRFBAMBattleAgent:
    return SRFBAMBattleAgent(env=_Env(), extractor=_Extractor())


def test_plan_gates_count_as_query_and_not_skip() -> None:
    agent = _make_agent()
    agent.reset()

    agent._update_metrics(
        BattleGateDecision(mode=BattleGateMode.PLAN_LOOKUP, reason="test", encode_flag=False),
        latency_ms=12.0,
    )
    agent._update_metrics(
        BattleGateDecision(mode=BattleGateMode.PLAN_STEP, reason="test", encode_flag=False),
        latency_ms=15.0,
    )

    telemetry = agent.telemetry()
    fractions = telemetry.fractions
    assert fractions["query"] == 1.0
    assert fractions["skip"] == 0.0

    profile = agent.profile_stats()
    assert profile["plan_lookup"]["count"] == 1
    assert profile["plan_step"]["count"] == 1
