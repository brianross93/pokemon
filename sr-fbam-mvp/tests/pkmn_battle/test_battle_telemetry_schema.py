from __future__ import annotations

from typing import Dict, List

from src.pkmn_battle.env import BattleObs, EnvAdapter, LegalAction
from src.pkmn_battle.extractor import Extractor
from src.srfbam.tasks.battle import SRFBAMBattleAgent
from src.telemetry import make_validator


class DummyExtractor(Extractor):
    def extract(self, obs: BattleObs):  # type: ignore[override]
        return []


class DummyEnv(EnvAdapter):
    def __init__(self) -> None:
        self._turn = 0

    def reset(self) -> BattleObs:  # type: ignore[override]
        self._turn = 0
        return self.observe()

    def observe(self) -> BattleObs:  # type: ignore[override]
        return {"turn": self._turn, "raw": {}}

    def legal_actions(self) -> List[LegalAction]:  # type: ignore[override]
        return [
            {"kind": "move", "index": 0, "meta": {"pp": 10}},
            {"kind": "switch", "index": 0, "meta": {}},
        ]

    def step(self, action: LegalAction) -> BattleObs:  # type: ignore[override]
        self._turn += 1
        return self.observe()


def test_battle_telemetry_matches_schema() -> None:
    env = DummyEnv()
    agent = SRFBAMBattleAgent(env=env, extractor=DummyExtractor())

    obs = agent.reset()
    obs = agent.step()
    telemetry = agent.telemetry()

    payload = telemetry.to_payload()
    record: Dict[str, object] = {
        "source": "sr-fbam.battle.agent",
        "context": {
            "domain": "battle",
            "battle": {"turn": int(obs.get("turn", 0)), "step": 1},
        },
        "observation": obs,
        "telemetry": payload,
    }

    validator = make_validator()
    validator.validate(record)


