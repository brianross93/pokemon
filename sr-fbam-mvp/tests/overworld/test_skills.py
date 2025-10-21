import torch

from src.plan.planner_llm import PlanletSpec
from src.overworld.graph.overworld_memory import OverworldMemory
from src.overworld.skills import (
    HealSkill,
    InteractSkill,
    MenuSkill,
    NavigateSkill,
    PickupSkill,
    ShopSkill,
    SkillStatus,
    TalkSkill,
    UseItemSkill,
    WaitSkill,
)
from src.overworld.skills.base import SkillProgress


class FakeSummary:
    def __init__(self, decision: str):
        self.gate_stats = {"decision": decision}
        self.embedding = torch.zeros(16)


def make_basic_memory() -> OverworldMemory:
    memory = OverworldMemory()
    memory.write(OverworldMemory.make_map_region("pallet", "Pallet Town"))
    memory.write(OverworldMemory.make_tile("pallet", 0, 0, passable=True, terrain="floor"))
    memory.write(OverworldMemory.make_tile("pallet", 1, 0, passable=True, terrain="floor"))
    memory.write(OverworldMemory.make_adjacent_edge("tile:pallet:0:0", "tile:pallet:1:0", passable=True))
    memory.write(OverworldMemory.make_adjacent_edge("tile:pallet:1:0", "tile:pallet:0:0", passable=True))
    memory.write(OverworldMemory.make_player("pallet", 0, 0, facing="DOWN"))
    return memory


def test_navigate_skill_writes_last_path_edges():
    memory = make_basic_memory()
    spec = PlanletSpec(id="nav", kind="NAVIGATE_TO", args={"target": {"map": "pallet", "tile": [1, 0]}})
    skill = NavigateSkill()
    skill.on_enter(spec, memory)

    action = skill.select_action({}, memory)
    assert action["kind"] in {"button", "wait"}

    edges = memory.follow(src="tile:pallet:0:0", relation="last_path")
    assert edges, "expected last_path edges after path computation"


def test_navigate_skill_reuses_last_path_with_gate_hint():
    memory = make_basic_memory()
    spec = PlanletSpec(id="nav", kind="NAVIGATE_TO", args={"target": {"map": "pallet", "tile": [1, 0]}})
    skill = NavigateSkill()
    skill.on_enter(spec, memory)
    skill.select_action({}, memory)  # prime path and edges

    skill.path = ()
    skill.update_context(summary=FakeSummary("CACHE_HIT"), memory=memory, slot_bank=None)
    action = skill.select_action({}, memory)
    assert action["kind"] == "button", "Expected button action when reusing last_path edges"


def test_menu_skill_triggers_recovery_on_desync():
    memory = OverworldMemory()
    memory.write(OverworldMemory.make_menu_state("main", path=["ITEMS"], open_=True))
    spec = PlanletSpec(id="menu", kind="OPEN_MENU", args={"path": ["POKEDEX", "BAG"]})
    skill = MenuSkill()
    skill.on_enter(spec, memory)

    action = skill.select_action({}, memory)
    assert action["kind"] == "button" and action["label"] == "B"
    recovery = skill.recovery_hint()
    assert recovery and recovery["reason"] == "MENU_DESYNC"


def test_shop_skill_flags_missing_inventory():
    memory = OverworldMemory()
    spec = PlanletSpec(id="shop", kind="BUY_ITEM", args={"item": "Potion", "qty": 2})
    skill = ShopSkill()
    skill.on_enter(spec, memory)
    for _ in range(len(skill.script)):
        skill.select_action({}, memory)
    progress = skill.progress(memory)
    assert isinstance(progress, SkillProgress)
    assert progress.status is SkillStatus.STALLED
    assert progress.reason == "INSUFFICIENT_FUNDS"


def test_use_item_skill_succeeds_when_item_present():
    memory = OverworldMemory()
    memory.write(OverworldMemory.make_inventory_item("potion", name="Potion", quantity=1))
    spec = PlanletSpec(id="use", kind="USE_ITEM", args={"item": "Potion"})
    skill = UseItemSkill()
    skill.on_enter(spec, memory)
    for _ in range(len(skill.script)):
        skill.select_action({}, memory)
    progress = skill.progress(memory)
    assert progress.status is SkillStatus.SUCCEEDED


def test_talk_skill_marks_failure_when_npc_missing():
    memory = OverworldMemory()
    spec = PlanletSpec(id="talk", kind="TALK_TO", args={"npc": "Prof. Oak"})
    skill = TalkSkill()
    skill.on_enter(spec, memory)
    for _ in range(len(skill.script)):
        skill.select_action({}, memory)
    progress = skill.progress(memory)
    assert progress.status is SkillStatus.STALLED
    assert progress.reason == "NPC_NOT_FOUND"


def test_interact_skill_succeeds_when_flag_set():
    memory = make_basic_memory()
    spec = PlanletSpec(id="interact", kind="INTERACT", args={"flag": "DoorOpen"})
    skill = InteractSkill()
    skill.on_enter(spec, memory)

    skill.select_action({}, memory)
    first_progress = skill.progress(memory)
    assert first_progress.status is SkillStatus.IN_PROGRESS

    memory.write(OverworldMemory.make_flag("DoorOpen", name="DoorOpen", value=True))
    final_progress = skill.progress(memory)
    assert final_progress.status is SkillStatus.SUCCEEDED


def test_pickup_skill_detects_inventory_delta():
    memory = make_basic_memory()
    spec = PlanletSpec(id="pickup", kind="PICKUP_ITEM", args={"item": "Potion"})
    skill = PickupSkill()
    skill.on_enter(spec, memory)

    skill.select_action({}, memory)
    assert skill.progress(memory).status is SkillStatus.IN_PROGRESS

    memory.write(OverworldMemory.make_inventory_item("potion", name="Potion", quantity=1))
    assert skill.progress(memory).status is SkillStatus.SUCCEEDED


def test_wait_skill_completes_after_required_steps():
    memory = make_basic_memory()
    spec = PlanletSpec(id="wait", kind="WAIT", args={"steps": 2})
    skill = WaitSkill()
    skill.on_enter(spec, memory)

    assert skill.progress(memory).status is SkillStatus.IN_PROGRESS
    skill.select_action({}, memory)
    assert skill.progress(memory).status is SkillStatus.IN_PROGRESS
    skill.select_action({}, memory)
    assert skill.progress(memory).status is SkillStatus.SUCCEEDED
