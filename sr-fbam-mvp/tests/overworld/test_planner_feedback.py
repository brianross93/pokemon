import pytest

from src.overworld.graph.overworld_memory import OverworldMemory
from src.plan.planner_llm import PlanBundle, PlanletSpec
from src.srfbam.tasks.overworld import OverworldExecutor
from src.srfbam.tasks.events import PlanletEvent
from src.overworld.skills.base import BaseSkill, SkillProgress, SkillStatus


class StubExtractor:
    def extract(self, observation):
        return []


class BlockedSkill(BaseSkill):
    name = "BlockedSkill"

    def legal_actions(self, observation, graph):
        return ({"kind": "wait"},)

    def select_action(self, observation, graph):
        return {"kind": "wait"}

    def progress(self, graph):
        return SkillProgress(status=SkillStatus.STALLED, reason="BLOCKED_PATH")


class MenuDesyncSkill(BaseSkill):
    name = "MenuDesyncSkill"

    def legal_actions(self, observation, graph):
        return ({"kind": "wait"},)

    def select_action(self, observation, graph):
        return {"kind": "wait"}

    def progress(self, graph):
        return SkillProgress(status=SkillStatus.STALLED, reason="MENU_DESYNC")


class RandomBattleSkill(BaseSkill):
    name = "RandomBattleSkill"

    def legal_actions(self, observation, graph):
        return ({"kind": "wait"},)

    def select_action(self, observation, graph):
        return {"kind": "wait"}

    def progress(self, graph):
        return SkillProgress(status=SkillStatus.STALLED, reason="RANDOM_BATTLE")


def make_plan(plan_id: str, planlet_id: str, kind: str = "NAVIGATE_TO") -> PlanBundle:
    return PlanBundle(
        plan_id=plan_id,
        goal=None,
        planlets=[
            PlanletSpec(
                id=planlet_id,
                kind=kind,
                args={},
                pre=[],
                post=[],
                hints={},
                timeout_steps=10,
                recovery=[],
            )
        ],
    )


def test_replan_triggers_on_blocked_path():
    events = []
    replan_calls = []

    executor = OverworldExecutor(
        memory=OverworldMemory(),
        extractor=StubExtractor(),
        skill_registry={"NavigateSkill": BlockedSkill},
    )
    executor.register_event_sink(events.append)

    def handler(event: PlanletEvent):
        replan_calls.append(event)
        return make_plan("replan-1", "rp1")

    executor.register_replan_handler(handler)
    executor.load_plan_bundle(make_plan("plan-1", "p1"))

    result = executor.step({})

    assert result.status == "PLANLET_STALLED"
    assert events, "Expected planlet event to be emitted"
    last_event = events[-1]
    assert last_event.reason == "BLOCKED_PATH"
    assert last_event.status == "PLANLET_STALLED"
    assert last_event.trace, "Expected short trace to be populated"
    assert replan_calls, "Replan handler should have been invoked"
    assert executor.plan_id == "replan-1"
    assert len(executor.state.plan_queue) == 1


def test_replan_respects_cap_and_menu_reason():
    events = []
    replan_calls = []

    executor = OverworldExecutor(
        memory=OverworldMemory(),
        extractor=StubExtractor(),
        skill_registry={"NavigateSkill": MenuDesyncSkill},
    )
    executor.register_event_sink(events.append)
    executor.register_replan_handler(
        lambda event: replan_calls.append(event) or make_plan("cap-1", "cap-planlet")
    )
    executor.set_max_replans(1)
    executor.load_plan_bundle(make_plan("plan-cap", "pc1"))

    executor.step({})
    assert events[-1].reason == "MENU_DESYNC"
    assert len(replan_calls) == 1

    executor.step({})
    assert len(replan_calls) == 1, "Replan handler should not fire beyond cap"


def test_event_emitted_for_random_battle_without_replan():
    events = []
    executor = OverworldExecutor(
        memory=OverworldMemory(),
        extractor=StubExtractor(),
        skill_registry={"NavigateSkill": RandomBattleSkill},
    )
    executor.register_event_sink(events.append)
    executor.load_plan_bundle(make_plan("plan-rand", "pr1"))

    executor.step({})

    assert events
    last_event = events[-1]
    assert last_event.reason == "RANDOM_BATTLE"
    assert last_event.status == "PLANLET_STALLED"
    assert last_event.trace
