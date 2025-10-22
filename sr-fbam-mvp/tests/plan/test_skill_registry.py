import random
import string

import pytest

from src.plan.compiler import PlanCompilationError, PlanCompiler
from src.plan.planner_llm import PlanBundle, PlanletSpec
from src.srfbam.tasks.overworld import OverworldExecutor


EXPECTED_PLAN_REGISTRY = {
    "NAVIGATE_TO": "NavigateSkill",
    "HEAL_AT_CENTER": "HealSkill",
    "BUY_ITEM": "ShopSkill",
    "TALK_TO": "TalkSkill",
    "OPEN_MENU": "MenuSkill",
    "MENU_SEQUENCE": "MenuSequenceSkill",
    "USE_ITEM": "UseItemSkill",
    "INTERACT": "InteractSkill",
    "PICKUP_ITEM": "PickupSkill",
    "WAIT": "WaitSkill",
    "HANDLE_ENCOUNTER": "EncounterSkill",
}


def test_default_registry_matches_expected_mapping() -> None:
    compiler = PlanCompiler()
    registry = compiler.registry()
    assert registry == EXPECTED_PLAN_REGISTRY

    executor_registry = OverworldExecutor.DEFAULT_SKILL_REGISTRY
    missing = {skill for skill in registry.values() if skill not in executor_registry}
    assert not missing, f"Executor registry missing skills: {sorted(missing)}"


def test_plan_compiler_rejects_unknown_kinds_fuzz() -> None:
    compiler = PlanCompiler()
    known_kinds = set(compiler.registry())
    rng = random.Random(1337)
    attempts = 0
    while attempts < 32:
        candidate = "".join(rng.choice(string.ascii_uppercase) for _ in range(6))
        if candidate in known_kinds:
            continue
        attempts += 1
        bundle = PlanBundle(
            plan_id=f"fuzz_{attempts}",
            goal=None,
            planlets=[
                PlanletSpec(id=f"pl_{attempts}", kind=candidate, args={}, timeout_steps=10),
            ],
        )
        with pytest.raises(PlanCompilationError):
            compiler.compile(bundle)


def test_plan_compiler_menu_sequence() -> None:
    compiler = PlanCompiler()
    bundle = PlanBundle(
        plan_id="menu_seq",
        goal="Exit menu",
        planlets=[
            PlanletSpec(
                id="boot",
                kind="MENU_SEQUENCE",
                args={"buttons": ["START", "A", "A"]},
                timeout_steps=60,
            ),
        ],
    )
    compiled = compiler.compile(bundle)
    assert compiled.planlets[0].skill == "MenuSequenceSkill"
