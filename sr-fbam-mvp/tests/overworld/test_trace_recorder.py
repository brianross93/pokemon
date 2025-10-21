import copy
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import torch

from src.overworld.graph.overworld_memory import OverworldMemory
from src.overworld.recording import OverworldTraceRecorder, TraceValidationError
from src.plan.planner_llm import PlanBundle, PlanletSpec
from src.srfbam.core import EncodedFrame, SrfbamStepSummary
from src.srfbam.tasks.overworld import OverworldExecutor
from src.overworld.skills.base import BaseSkill, SkillProgress, SkillStatus
from src.overworld.encounter import EncounterResult


class EmptyExtractor:
    def extract(self, observation):
        return []


class FakeCore:
    def __init__(self, decisions):
        self.decisions = deque(decisions)
        self.device = torch.device("cpu")
        self.config = type(
            "Config",
            (),
            {
                "encode_latency_ms": 40.0,
                "assoc_latency_ms": 5.0,
                "follow_latency_ms": 5.0,
                "write_latency_ms": 2.0,
                "skip_latency_ms": 1.5,
            },
        )()

    def encode_step(self, encoded: EncodedFrame) -> SrfbamStepSummary:
        decision = self.decisions[0] if len(self.decisions) == 1 else self.decisions.popleft()
        gate_stats = {"decision": decision, "cache_hits": 0, "reuse": 0, "extract": 0}
        return SrfbamStepSummary(
            embedding=torch.zeros(1),
            symbol_embedding=torch.zeros(1),
            numeric_features=torch.zeros(1),
            gate_stats=gate_stats,
            context_key=encoded.context_key,
        )

    def set_last_action_index(self, index: int) -> None:
        return


class PassiveSkill(BaseSkill):
    name = "PassiveSkill"

    def legal_actions(self, observation, graph):
        return ({"kind": "wait"},)

    def select_action(self, observation, graph):
        return {"kind": "wait"}

    def progress(self, graph):
        return SkillProgress(status=SkillStatus.IN_PROGRESS)


class RandomBattleSkill(BaseSkill):
    name = "RandomBattleSkill"
    _GLOBAL_TRIGGERED = False

    def legal_actions(self, observation, graph):
        return ({"kind": "wait"},)

    def select_action(self, observation, graph):
        return {"kind": "wait"}

    def progress(self, graph):
        if not RandomBattleSkill._GLOBAL_TRIGGERED:
            RandomBattleSkill._GLOBAL_TRIGGERED = True
            return SkillProgress(status=SkillStatus.STALLED, reason="RANDOM_BATTLE")
        return SkillProgress(status=SkillStatus.SUCCEEDED)


def make_plan(plan_id: str = "plan", planlet_id: str = "p1") -> PlanBundle:
    return PlanBundle(
        plan_id=plan_id,
        goal=None,
        planlets=[
            PlanletSpec(
                id=planlet_id,
                kind="NAVIGATE_TO",
                args={},
                pre=[],
                post=[],
                hints={},
                timeout_steps=10,
                recovery=[],
            )
        ],
    )


def make_executor():
    return OverworldExecutor(
        memory=OverworldMemory(),
        extractor=EmptyExtractor(),
        core=FakeCore(["REUSE", "REUSE", "WRITE"]),
        skill_registry={"NavigateSkill": PassiveSkill},
    )


def test_trace_recorder_writes_entries(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    executor = make_executor()
    executor.load_plan_bundle(make_plan())
    observation = {"overworld": {"map": {"id": "pal"}, "player": {"tile": [0, 0], "facing": 0}}}

    with OverworldTraceRecorder(trace_path) as recorder:
        executor.register_trace_recorder(recorder)
        for _ in range(3):
            executor.step(observation)

    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "Expected trace file to contain entries"
    entry = lines[0]
    assert entry["context"]["plan"]["id"] == "plan"
    assert entry["telemetry"]["core"]["action"]["kind"] == "wait"
    assert "frame_features" in entry["telemetry"]["overworld"]
    assert "memory" in entry["telemetry"]["overworld"]
    assert entry["telemetry"]["core"]["gate"]["view"] in {"typed", "slots"}


def test_trace_recorder_validation(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    recorder = OverworldTraceRecorder(trace_path)
    try:
        recorder.record({"source": "test"})  # missing required fields
    except TraceValidationError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected TraceValidationError for invalid payload")
    finally:
        recorder.close()


def test_overworld_bc_smoke(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    executor = make_executor()
    executor.load_plan_bundle(make_plan())
    observation = {"overworld": {"map": {"id": "pal"}, "player": {"tile": [0, 0], "facing": 0}}}

    with OverworldTraceRecorder(trace_path) as recorder:
        executor.register_trace_recorder(recorder)
        for _ in range(4):
            executor.step(observation)

    cmd = [
        sys.executable,
        "-m",
        "src.training.overworld_bc",
        "--traces",
        str(trace_path),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr



def test_trace_recorder_captures_encounter_phases(tmp_path: Path):
    RandomBattleSkill._GLOBAL_TRIGGERED = False
    trace_path = tmp_path / "encounter.jsonl"
    executor = OverworldExecutor(
        memory=OverworldMemory(),
        extractor=EmptyExtractor(),
        core=FakeCore(["REUSE", "REUSE", "REUSE", "WRITE"]),
        skill_registry={"NavigateSkill": RandomBattleSkill},
    )
    executor.load_plan_bundle(make_plan("base", "nav"))

    def battle_handler(request):
        return EncounterResult(
            status="WIN",
            turns=1,
            reason=None,
            snapshot_return=request.snapshot_overworld,
        )

    executor.register_battle_handler(battle_handler)

    base_obs = {
        "overworld": {
            "map": {"id": "01", "name": "Map_01"},
            "player": {"map_id": "01", "tile": [0, 0], "facing": "south"},
            "tiles": [
                {"map_id": "01", "x": 0, "y": 0, "passable": True, "terrain": "path", "special": ""},
                {"map_id": "01", "x": 0, "y": 1, "passable": True, "terrain": "path", "special": ""},
            ],
            "warps": [],
            "npcs": [],
            "menus": [],
        }
    }

    with OverworldTraceRecorder(trace_path) as recorder:
        executor.register_trace_recorder(recorder)
        executor.step(copy.deepcopy(base_obs))

    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    encounter_events = [event for entry in lines for event in entry["telemetry"]["overworld"].get("encounter", [])]
    phases = [event.get("phase") for event in encounter_events]
    assert phases == ["battle.entry", "battle.exit"]












