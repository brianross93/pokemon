# Integrated Planlet Migration Checklist (Battle + Overworld)

Legend: [x] done, [ ] pending, [new] overworld-specific
DoD = Definition of Done (unblocks the next phase)

## Phase 0 - Foundations (battle assets ready; add overworld stubs)
- [x] Mechanics data mirrored (`data/mechanics/`)
- [x] Battle graph + SR-FBAM memory operators
- [x] Frame renderer -> 40x120 grid
- [x] Metamon trajectory ingestion + JSONL schema
- [x] [new] WorldGraph skeleton (`pkmn_overworld/world_graph.py`)
- [x] [new] Import static overworld entities from `pret/pokered` (map graph, NPCs, shops, hidden items -> `data/overworld/static_entities.json`)
- [x] [new] Screen parsers for overworld (`pkmn_overworld/screen_parse.py`)

DoD: One replay runs SR-FBAM-only in both modes (battle + corridor demo).

## Phase 1 - Schema & Telemetry Contracts (extend, don't refactor)
- [x] Publish planlet schema (`docs/PLANLET_SCHEMA.md`)
- [x] Note planlet telemetry fields
- [x] Add JSONSchema validation to telemetry ingestion (`telemetry_schema.json`)
- [x] Update dashboards to surface plan metrics
- [x] [new] Unify planlets with kind flag (`"BATTLE"` | `"OVERWORLD"`) and op enums
- [x] [new] Telemetry additions (overworld): mode, skill, path_len_planned, path_len_executed, encounters, abort_code, adhere_code

DoD: CI fails on invalid planlet/telemetry JSON; dashboards show plan metrics for both modes.

## Phase 2 - Graph Summary & LLM Harness
- [x] Deterministic battle summarizer (`pkmn_battle/summarizer.py`)
- [x] Planlet proposer stub + schema validation (`llm/planlets/`)
- [x] Integrate GPT-5 tool prompts + search stubs (`llm/planlets/proposer.py`)
- [x] End-to-end harness to request planlets from live state (`llm/planlets/service.py`)
- [x] [new] Overworld summarizer (`pkmn_overworld/summarizer.py`)
- [x] [new] Fake-LLM deterministic backend for CI (`llm/planlets/fake_backend.py`)

DoD: scripts/request_planlet.py --mode overworld emits a valid planlet; snapshot tests stay stable.

## Phase 3 - Plan Logging & Storage
- [x] Extend battle runner to attach planlet metadata per turn
- [x] Persist planlets (Parquet/JSONL) with retrieved docs & tokens
- [x] Manifest support for plan-conditioned JSONL (train/val/test)
- [x] [new] Plan cache keyed by graph neighborhood hash (battle + overworld)
- [x] [new] Success-weighted eviction for plan cache; retention policy/TTL

DoD: After a mixed run, data/planlets/planlets.parquet contains both kinds; cache hits reduce LLM calls.

## Phase 4 - Executor Updates
- [x] Add SR-FBAM gates `PLAN_LOOKUP` / `PLAN_STEP`
- [x] Map battle plan ops -> legal actions + fallbacks
- [x] Runtime precondition checks & graceful abort path
- [x] Telemetry: log gate probabilities, plan adherence, fallbacks
- [x] [new] Overworld skills (navigate, interact, talk, buy, pickup, use_item, menu, wait)
- [x] [new] Op->Skill registry with unit tests + illegal-state fuzzing
- [x] [new] Context switch logic (battle interrupts + resume or revise)
- [x] [new] Rule-based confidence gate v0 (pre-learned head)

DoD: Fake planlets execute >=2 script steps via PLAN_STEP; aborts log adhere codes; battle resumes handled.

## Phase 5 - Training & Distillation
- [x] Extend battle dataset with plan features (per-turn plan embeddings, gate targets, adherence flags)
- [ ] Materialize plan feature store + sampling weights for mixed-mode batches (requires overworld trace import)
- [x] Implement plan-conditioned controller training loop (dual-mode batches, gate supervision, teacher forcing)
- [ ] Calibrate gate heads + scheduler (freeze/unfreeze plan gates, distillation losses, telemetry)
- [ ] Set up self-distillation loop (LLM-on -> relabel -> retrain -> LLM-off) with checkpoint automation
- [x] [new] Ship OverworldDecisionDataset with mode_bit + plan features (trajectory slicing, skill labels)
- [ ] [new] Mix battle/overworld curriculum + augmentations (tile jitter, encounter injection, plan dropout) — blocked on overworld telemetry import
- [ ] [new] Generate overworld telemetry via PyBoy corridor runs (plan metadata + gate/adherence logs) — harness at `scripts/capture_overworld_telemetry.py`
- [ ] [new] Ablations: no-plan vs plan features; text masked; gate frozen vs trainable; overworld-only gate freeze

DoD: Mixed tasks maintain/improve success with >=40% fewer LLM calls; gate mix shows ENCODE down, PLAN_* up.

## Phase 6 - Evaluation & QA
- [ ] Define metrics: plan success, LLM reliance, latency, win-rate
- [ ] Regression tests for planlet schema + prompt compliance
- [ ] Benchmarks for latency with and without planlets
- [ ] [new] Overworld tasks: reach location, heal+shop, fetch+deliver, constrained routes
- [ ] [new] Overworld metrics: success rate, path stretch, encounters/100 tiles, p95 latency, LLM calls/task, tokens/km, gate mix
- [ ] [new] Value-of-computation analysis (delta success when LLM invoked)

DoD: Dashboards show battle & overworld plan metrics incl. LLM calls, tokens, adherence, p95 latency; report includes gate mix and speedups.

## Phase 7 - Cleanup & Decommission
- [ ] Retire legacy code-editing dataset + trainers
- [ ] Prune unused imitation scripts once plan flow is primary
- [ ] Document planlet mode handbook (README + telemetry guide)
- [ ] [new] Flags: `--disable-planlets`, `--disable-overworld`

DoD: System runs under all flags; README includes Overworld Runbook (failure modes, budgets, resume rules).
