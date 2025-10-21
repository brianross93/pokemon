from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.plan import PlanCompiler, PlanCompilationError, PlanValidationError, load_plan_bundle
from src.plan.planner_llm import PlanBundle, PlanletSpec, validate_plan_json
from src.overworld.extractor import OverworldExtractor
from src.overworld.graph import OverworldMemory


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "docs" / "examples"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "overworld"


@pytest.mark.parametrize(
    "plan_file",
    [
        EXAMPLE_DIR / "plan_heal_and_shop.json",
        EXAMPLE_DIR / "plan_route1_scouting.json",
    ],
)
def test_example_plans_validate_and_compile(plan_file: Path) -> None:
    bundle = load_plan_bundle(plan_file)
    compiler = PlanCompiler()
    compiled = compiler.compile(bundle)
    assert compiled.plan_id == bundle.plan_id
    assert len(compiled.planlets) == len(bundle.planlets)
    assert all(planlet.skill for planlet in compiled.planlets)


def test_plan_compiler_enforces_limits() -> None:
    compiler = PlanCompiler()
    too_many_planlets = PlanBundle(
        plan_id="limit",
        goal=None,
        planlets=[
            PlanletSpec(
                id=f"p{i}",
                kind="NAVIGATE_TO",
                args={},
                pre=[],
                post=[],
                hints={},
                timeout_steps=10,
                recovery=[],
            )
            for i in range(9)
        ],
    )
    with pytest.raises(PlanCompilationError):
        compiler.compile(too_many_planlets)

    zero_timeout_bundle = PlanBundle(
        plan_id="timeout",
        goal=None,
        planlets=[
            PlanletSpec(
                id="p1",
                kind="NAVIGATE_TO",
                args={},
                pre=[],
                post=[],
                hints={},
                timeout_steps=0,
                recovery=[],
            )
        ],
    )
    with pytest.raises(PlanCompilationError):
        compiler.compile(zero_timeout_bundle)


def test_validate_plan_json_rejects_invalid_payload() -> None:
    invalid_json = json.dumps({"plan_id": "bad", "planlets": [{"kind": "NAVIGATE_TO"}]})
    with pytest.raises(PlanValidationError):
        validate_plan_json(invalid_json)


def test_overworld_extractor_produces_nodes_and_edges() -> None:
    snapshot_path = FIXTURE_DIR / "snapshot_basic.json"
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    extractor = OverworldExtractor()
    writes = extractor.extract(data)

    node_ids = {op.payload.node_id for op in writes if op.kind == "node"}
    edge_relations = {op.payload.relation for op in writes if op.kind == "edge"}

    assert "player:me" in node_ids
    assert any(node_id.startswith("tile:Pallet_Town") for node_id in node_ids)
    assert "adjacent" in edge_relations
    assert "owns" in edge_relations


def test_overworld_extractor_handles_ram_snapshot() -> None:
    ram = bytearray(0xE000)
    ram[0xD35E] = 0x01
    ram[0xD361] = 0x0C
    ram[0xD362] = 0x09
    ram[0xD360] = 0x0C
    ram[0xCC3C] = 0x01
    # Warp entry: raw (y, x) of (13, 16) -> tile (9, 12)
    ram[0xD31E] = 0x01
    ram[0xD31F] = 0x0D
    ram[0xD320] = 0x10
    ram[0xD321] = 0x03  # dest warp id
    ram[0xD322] = 0x02  # dest map
    # NPC entry: raw (y, x) of (14, 17) -> tile (10, 13)
    ram[0xD2F4] = 0x01
    ram[0xD2F5] = 0x0E
    ram[0xD2F6] = 0x11
    ram[0xD2F7] = 0x2A
    ram[0xD2F8] = 0x05

    extractor = OverworldExtractor()
    writes = extractor.extract({"ram": ram})

    node_ids = {op.payload.node_id for op in writes if op.kind == "node"}
    assert "map:01" in node_ids
    assert "player:me" in node_ids
    assert "warp:0" in node_ids
    assert "npc:0" in node_ids


def test_overworld_memory_summarise_nodes_counts_types() -> None:
    memory = OverworldMemory()
    writes = [
        OverworldMemory.make_map_region("01", "Map_01"),
        OverworldMemory.make_tile("01", 12, 9, passable=True, terrain="grass"),
        OverworldMemory.make_tile("01", 12, 10, passable=True, terrain="grass"),
        OverworldMemory.make_contains_edge("map:01", "tile:01:12:9"),
    ]
    for op in writes:
        memory.write(op)
    summary = memory.summarise_nodes()
    assert summary["MapRegion"] == 1
    assert summary["Tile"] == 2
