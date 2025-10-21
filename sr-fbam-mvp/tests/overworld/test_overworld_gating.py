import sys
from pathlib import Path
import math
from collections import deque
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.pkmn_battle.graph.schema import Node, WriteOp
from src.srfbam.core import EncodedFrame, SrfbamStepSummary
from src.overworld.graph.overworld_memory import OverworldMemory
from src.plan.planner_llm import PlanBundle, PlanletSpec
from src.srfbam.tasks.overworld import OverworldExecutor
from src.overworld.skills.base import BaseSkill, SkillProgress, SkillStatus


class EmptyExtractor:
    def extract(self, observation):
        return []


class FakeCore:
    def __init__(self, decisions):
        self.decisions = deque(decisions)
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            encode_latency_ms=40.0,
            assoc_latency_ms=5.0,
            follow_latency_ms=5.0,
            write_latency_ms=2.0,
            skip_latency_ms=1.5,
        )

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


def stub_frame_encoder(_obs):
    features = torch.zeros(1, 8)
    grid = torch.zeros((40, 120), dtype=torch.long)
    return EncodedFrame(grid=grid, features=features, context_key="ctx", extra={})


def test_gating_metrics_accumulate():
    decisions = ["EXTRACT", "CACHE_HIT", "REUSE"]
    executor = OverworldExecutor(
        memory=OverworldMemory(),
        extractor=EmptyExtractor(),
        core=FakeCore(decisions),
        frame_encoder=stub_frame_encoder,
        skill_registry={"NavigateSkill": PassiveSkill},
    )
    executor.load_plan_bundle(
        PlanBundle(
            plan_id="plan",
            goal=None,
            planlets=[
                PlanletSpec(
                    id="p1",
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
    )

    telemetry = None
    for _ in range(3):
        result = executor.step({})
        telemetry = result.telemetry

    assert telemetry is not None
    gate = telemetry["core"]["gate"]
    assert gate["mode"] == "FOLLOW"
    fractions = telemetry["core"]["fractions"]
    assert math.isclose(fractions["encode"], 1 / 3, rel_tol=1e-3)
    assert math.isclose(fractions["query"], 2 / 3, rel_tol=1e-3)
    assert fractions["skip"] == 0.0

    speedup = telemetry["core"]["speedup"]
    assert speedup["predicted"] is not None
    assert speedup["observed"] is not None


def test_gating_skip_category():
    decisions = ["WRITE"]
    executor = OverworldExecutor(
        memory=OverworldMemory(),
        extractor=EmptyExtractor(),
        core=FakeCore(decisions),
        frame_encoder=stub_frame_encoder,
        skill_registry={"NavigateSkill": PassiveSkill},
    )
    executor.load_plan_bundle(
        PlanBundle(
            plan_id="plan",
            goal=None,
            planlets=[
                PlanletSpec(
                    id="p1",
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
    )
    telemetry = executor.step({}).telemetry
    fractions = telemetry["core"]["fractions"]
    assert fractions["encode"] == 0.0
    assert fractions["query"] == 0.0
    assert fractions["skip"] == 1.0


def test_hybrid_projection_populates_graph():
    executor = OverworldExecutor(
        memory=OverworldMemory(),
        extractor=EmptyExtractor(),
        core=FakeCore(["REUSE"]),
        frame_encoder=stub_frame_encoder,
        skill_registry={"NavigateSkill": PassiveSkill},
    )
    executor.load_plan_bundle(
        PlanBundle(
            plan_id="plan-hybrid",
            goal=None,
            planlets=[
                PlanletSpec(
                    id="p1",
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
    )

    executor.step({})  # prime slot bank
    telemetry = executor.step({}).telemetry

    slot_nodes = executor.memory.assoc(type_="SlotView")
    assert slot_nodes, "Expected SlotView nodes to be written when using the slot view"
    assert slot_nodes[0].attributes.get("context") == "ctx"
    assert telemetry["core"]["gate"]["view"] == "slots"
    assert telemetry["overworld"]["hybrid"]["projected"] > 0
    assert telemetry["overworld"]["view_usage"]["slots"] >= 1


def test_hybrid_ingest_restores_slots():
    executor = OverworldExecutor(
        memory=OverworldMemory(),
        extractor=EmptyExtractor(),
        core=FakeCore(["WRITE"]),
        frame_encoder=stub_frame_encoder,
        skill_registry={"NavigateSkill": PassiveSkill},
    )
    executor.load_plan_bundle(
        PlanBundle(
            plan_id="plan-ingest",
            goal=None,
            planlets=[
                PlanletSpec(
                    id="p1",
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
    )

    slot_node = Node(
        type="SlotView",
        node_id="slot:test",
        attributes={
            "context": "ctx",
            "confidence": 0.5,
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"source": "fixture"},
        },
    )
    executor.memory.write(WriteOp(kind="node", payload=slot_node))

    telemetry = executor.step({}).telemetry

    assert executor.slot_bank.contains_metadata("slot_node_id", "slot:test")
    assert telemetry["core"]["gate"]["view"] == "typed"
    assert telemetry["overworld"]["hybrid"]["ingested"] >= 1

