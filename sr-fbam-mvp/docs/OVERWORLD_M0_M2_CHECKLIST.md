# Overworld Planner–Executor Roadmap (Milestones M0–M2)

This document translates the high-level overworld spec into concrete, hand-off-ready pull-request checklists for the first three milestones. Each checklist is intentionally small in scope so we can iterate quickly and keep diffs reviewable.

## M0 – Scaffolding

- [x] Create `src/plan/` package with `__init__.py`.
- [x] Stub `planner_llm.py` with typed planlet container **and** hook JSON-schema validation for LLM output.
- [x] Add `src/plan/compiler.py` with placeholder compiler returning `CompiledPlan`.
- [x] Introduce `src/overworld/` package with subpackages:
  - [x] `env/__init__.py` and `env/overworld_adapter.py` (class skeleton mirroring battle adapter interface).
  - [x] `extractor/__init__.py` and `overworld_extractor.py` emitting initial nodes/edges.
  - [x] `graph/__init__.py` and `overworld_memory.py` stub extending existing `GraphMemory`.
  - [x] `skills/__init__.py` plus stub classes (`BaseSkill`, `NavigateSkill`, `HealSkill`, `ShopSkill`, `TalkSkill`, `MenuSkill`, `UseItemSkill`, `EncounterSkill`).
- [x] Wire CLI entry points:
  - [x] `scripts/run_overworld_agent.py` wrapper.
  - [x] Packaged `sr-fbam-overworld` console script (see `pyproject.toml`).
- [x] Add `docs/OVERWORLD_PLANNING.md` (skeleton referencing spec and TODOs for each milestone).
- [x] Add smoke test covering plan validation/compilation round-trip (`tests/overworld/test_plan_roundtrip.py`).

## M1 – Overworld Graph & Extractor

- [x] Implement `OverworldExtractor.extract(obs)`:
  - [x] Parse player position, facing, current map, and menu flags from structured snapshot.
  - [x] Extract tiles, warps, NPCs, inventory items, and story flags.
  - [x] Emit structured `WriteOp` objects compatible with `GraphMemory`.
- [x] Extend `GraphMemory` with overworld-specific node/edge factories:
  - [x] Node helpers (`MapRegion`, `Tile`, `Warp`, `NPC`, `Item`, `InventoryItem`, `Flag`, `MenuState`, `Player`).
  - [x] Edge helpers (`contains`, `adjacent`, `warp_to`, `located_at`, `owns`, `offers`, `requires`).
  - [x] Ensure writes are idempotent and maintain hop traces for telemetry.
- [x] Unit tests for extractor emit expected nodes/edges using recorded RAM fixture.
- [x] Document RAM offsets and assumptions in `docs/OVERWORLD_PLANNING.md`.
- [x] Update telemetry schema draft (`docs/telemetry_overworld_schema.json`) with new node/edge fields.
- [x] Verify `sr-fbam-overworld --goal-json stub` runs extractor and logs writes (no skills yet).

## M2 – NavigateSkill

- [x] Implement `NavigateSkill` with:
  - [x] Graph search over `Tile.adjacent` (current BFS; upgrade to A* for heuristics).
  - [x] Warp handling via `Warp` nodes.
  - [x] Step loop that issues button actions toward path waypoints.
- [x] Integrate `NavigateSkill` into executor skeleton (`src/srfbam/tasks/overworld.py`):
  - [x] Gate stub provides `FOLLOW`/telemetry placeholders (SR-FBAM wiring to follow).
  - [x] Skill progression statuses (`NOT_STARTED`, `IN_PROGRESS`, `SUCCEEDED`, `STALLED`).
- [x] Add regression tests:
  - [x] Pathfinding around obstacle.
  - [x] Warp traversal (e.g., door transition).
  - [x] Timeout/stall detection (blocked tile).
- [x] Extend telemetry logging to include `planlet_id`, `planlet_kind`, and navigation-specific hop traces.
- [x] Update documentation with usage instructions and expected metrics (encode fraction target, speedup).

> Each milestone builds linearly; landing M0 gives us compilable stubs, M1 populates the symbolic world, and M2 delivers the first SR-FBAM-backed overworld skill.
