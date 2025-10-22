# SR-FBAM Overworld Migration Plan

## Background
SR-FBAM already runs reliably inside battle simulations, but overworld execution never joined the same loop. The current repository trains imitation models from Gen 9 battle logs, logs overworld telemetry to JSONL, and ships curriculum updates that only touch the battle trainer. No runtime harness actually keeps PyBoy alive, calls the SR-FBAM gates, and escalates HALT events to the planner or LLM. This plan documents the migration path to a live controller that plays the overworld end-to-end while collecting telemetry online.

## Current State
- **Training pipeline**: `BattleDecisionDataset` and `train_battle_il.py` supervise battle actions and gate labels from 40×120 battle frames. Overworld telemetry is captured with `scripts/capture_overworld_telemetry.py` but never consumed by a trainer, so curriculum metrics stay pinned at `encode=100%`.
- **Runtime control loop**: No script boots PyBoy, stays in menus, consults the planner, and resumes play. The capture script presses scripted/random inputs, but it never calls `src/srfbam/tasks/overworld.py`. The real executor exists, yet no controller drives it continuously.
- **Result**: We can bootstrap the ROM and grab frames, but there is no autonomous overworld playthrough via the SR-FBAM pipeline.

## Target State
- **Online overworld data**: Skip offline training for overworld behavior. Run the game live, log telemetry continuously, and surface HALT events to SR-FBAM plus the planner/LLM. Data collection happens during execution.
- **Unified control loop**: `scripts/run_overworld_controller.py` (or successor) should:
  1. Start `PyBoyPokemonAdapter` and reach the overworld (optionally via `--policy-boot`).
  2. Extract entities every frame with `OverworldExtractor`.
  3. Hand observations to `OverworldExecutor`.
  4. On HALT, request planlets from the planner/LLM.
  5. Execute returned skill scripts, write inputs back through PyBoy, and record telemetry (menu states, gate decisions, adherence).
  6. Repeat indefinitely to build the dataset online.
- **Live-first mindset**: Keep battle imitation training for stats, but let SR-FBAM drive menus and overworld exploration in real time. Future model improvements can reuse the telemetry we log now.

## Gaps to Close
- `run_overworld_controller.py` needs a real planner integration—today it emits random `NAVIGATE_TO` planlets and never consults the LLM.
- `OverworldExecutor.register_replan_handler` is unwired. HALT should trigger `PlanletService` to propose planlets (menus, corridor exploration, etc.).
- Telemetry should stay attached during live runs (`OverworldTraceRecorder`) so we log gate/plan metadata without running offline jobs.
- Bootstrapping remains brittle: ensure PyBoy clears the title screen and lands in overworld hands-free.
- Need repeatable seeds, ROM offsets validation, and failure handling so long runs are reliable.

## Migration Plan

### Milestone 1 — Controller Foundations
- Harden `scripts/run_overworld_controller.py`: deterministic boot with fallback to `--policy-boot`, watchdog for stuck states, graceful shutdown.
- Ensure `OverworldExecutor` loads compiled planlets on start and continues after `PLANLET_COMPLETE` or `PLAN_COMPLETE`.
- Wire continuous telemetry via `OverworldTraceRecorder`; include gate mode, skill metadata, plan IDs, and menu states every step.
- DoD: Script can run for ≥2k steps without crashing, emits telemetry JSONL, and never loses the overworld state after boot.

### Milestone 2 — Planner/LLM Integration
- Connect `OverworldExecutor.register_replan_handler` to `PlanletService` so HALT events fetch planlets from the configured backend (deterministic fake or LLM).
- Extend planlet bundle compilation to cover menu navigation, interactions, and fallback scripts (not just `NAVIGATE_TO`).
- Stage prompt context to include overworld summaries, gate history, and failure affordances.
- DoD: When the executor HALTs, the controller obtains a planlet from the planner backend and resumes play; telemetry marks HALT/RESUME spans.

### Milestone 3 — Telemetry & Observability
- Keep recorder active for every domain (menu, overworld, battle). Log gate mix, adherence, tokens, and HALT causes.
- Surface telemetry dashboards for live runs (success, path stretch, encounters, LLM calls).
- Validate RAM offsets and entity extraction with `scripts/debug_overworld_addresses.py` before long captures.
- DoD: Telemetry dashboards show overworld gate metrics trending, LLM usage, and path quality from live runs.

### Milestone 4 — Runtime Reliability
- Add watchdogs for stuck planlets, ROM reloads, or adapter failures; auto-resume from save states where possible.
- Capture reproducible seeds/config snapshots alongside telemetry.
- Document failure modes and recovery steps in a runbook.
- DoD: Overnight runs complete with minimal manual intervention; operators can recover from crashes using the runbook.

### Milestone 5 — Optional Training Follow-ups
- Once telemetry accumulates, update feature store jobs and mixed-mode curriculum (`configs/train_plan.yaml`) to ingest overworld traces.
- Revisit gate head calibration and plan adherence training using live-collected data.
- Evaluate learned vs. scripted planlets with A/B toggles.
- DoD: Mixed training leverages overworld traces without blocking the live loop; experiments measure impact on gate mix and LLM calls.

## Deliverables
- Updated `scripts/run_overworld_controller.py` with live planner wiring and telemetry.
- Documented runbook covering bootstrapping, HALT events, recovery, and telemetry inspection.
- Dashboards tracking overworld plan performance, LLM reliance, and latency.
- (Optional) Updated training configs that consume live telemetry once available.
