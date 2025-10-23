# SR-FBAM Overworld System Overview

This document frames how the overworld controller, SR-FBAM pipeline, and planner collaborate during live runs. It complements `docs/OVERWORLD_RUNBOOK.md` (operator procedures) and `docs/OVERWORLD_MIGRATION_PLAN.md` (roadmap), focusing on why each subsystem exists and how information flows end to end.

---

## 1. Mission and Principles

- Deliver a live control loop that keeps PyBoy in the overworld, routes *every* HALT through the planner/LLM, and executes planlets exclusively via SR-FBAM.  
- Preserve a strict “no deterministic fallback” policy—controllers never synthesize their own planlets or replay canned sequences. If the LLM fails, the run should halt and surface telemetry for debugging.  
- Treat SR-FBAM as the single source of truth for perception, entity extraction, skill execution, and telemetry. The controller is orchestration glue; all game-facing behavior originates inside SR-FBAM gates and skills.

---Thanks—this is a super clear write‑up. If you’re still not getting past the menu, the failure almost always lives in one of four seams in this loop:

1. **Boot handoff** (adapter → SR‑FBAM)
2. **Menu extraction** (extractor → summariser)
3. **Planner contract** (PlanCoordinator → LLM → schema)
4. **Plan execution + progress accounting** (skills → gates → watchdogs)

Below is a targeted diagnosis plan and concrete fixes for each seam. I’ve kept everything aligned with your “no deterministic fallback” principle: we never synthesize planlets locally, we just make the planner + skills airtight.

---

## A. Quick isolation (10‑minute triage)

1. **Prove the handoff path:** Start from a savestate 2–3 frames *before* the title screen. Run with `--policy-boot` **only**, then halt the controller right after `_ensure_bootstrapped()` returns. If you ever see `is_menu=true` at this point, boot didn’t fully clear the title flow.

   * **Fix:** Make `_ensure_bootstrapped()` idempotent and “two‑phase”: `phase1: title->continue`, `phase2: continue->overworld`. Persist a “boot_phase” flag in adapter state; only hand control to SR‑FBAM once `boot_phase == OVERWORLD_READY`.

2. **Force a single HALT from menus:** Start from a state with the Start menu open. On the first SR‑FBAM tick, confirm a HALT with `reason=MENU_DETECTED` and a **non‑empty** `menu_snapshot` (cursor row/col, menu name/id/hash). If HALT isn’t emitted, extraction/gating is the issue. If HALT is emitted but no planner call fires, it’s the controller/PlanCoordinator wiring.

3. **Validate planner I/O in isolation:** Take that HALT packet and run it through the `fake`/`mock` planner backend offline. Ensure you get a **schema‑valid** `MENU_SEQUENCE` with uppercase buttons. If you don’t, it’s prompt or schema drift.

4. **Step-plan adherence locally:** Feed the returned planlet to the SR‑FBAM executor in a “dry‑run” (emulator muted) for N frames and assert that the planlet reports `PLANLET_COMPLETE` or raises HALT with a **specific** adherence code (not a generic stall).

If these four pass, you’ll get past the menu in live runs.

---

## B. Where menu flows typically break (and how to harden them)

### 1) Boot handoff is incomplete or races with SR‑FBAM

* **Symptom:** You see HALTs at the title screen or the Continue/New Game screen; LLM keeps returning “PRESS START” plans but nothing advances.
* **Cause:** `_ensure_bootstrapped()` presses START, but handoff occurs mid‑transition; SR‑FBAM immediately re‑detects a menu and re‑requests plans, creating a loop.
* **Fixes:**

  * **Two‑phase boot** as above. Gate handoff on *overworld‑ready* predicates (tilemap + RAM flags), not just “not title”.
  * **Boot transition settle time:** Add a fixed 10–15 frame settle window after the last boot keypress before giving SR‑FBAM the first frame.

### 2) Menu extraction is too weak for planner‑grade context

* **Symptom:** HALTs are emitted, but planner returns junk/invalid JSON or repeats the wrong plan. Telemetry shows `menu_snapshot` missing cursor or menu id.
* **Causes:**

  * State hash doesn’t include cursor position/menu id → cache/planner sees states as identical.
  * Extractor uses visual cues only; sprites/tilesheets vary between screens; RAM offsets drifted.
* **Fixes:**

  * **State hash expansion:** Include `(menu_id, cursor_index, selected_text, menu_depth, ram_menu_flag_bytes)` in the state hash. Caches become robust, and planner context becomes specific.
  * **RAM‑first menu id:** Prefer RAM offsets for menu mode/cursor index; use pixels as fallback. You already note offset drift—re‑run `scripts/debug_overworld_addresses.py` and assert offsets in CI whenever the ROM changes.
  * **Snapshot invariants:** Always attach `{menu_name, options[], cursor_index, can_scroll, confirm_button}` to `telemetry.overworld.menu_snapshot` so few‑shots stay stable.

### 3) Planner contract ambiguities for menus

* **Symptom:** Planner returns `NAVIGATE_TO` when a menu is open, or `MENU_SEQUENCE` lacks uppercase buttons or a stopping condition; schema retries get exhausted.
* **Causes:** Prompt doesn’t make `MENU_SEQUENCE` *the only legal* kind when `is_menu=true`, and it doesn’t tell the model when to stop pressing buttons.
* **Fixes:**

  * **Hard constraint in prompt:** “If `is_menu=true`, you must return exactly one planlet of kind `MENU_SEQUENCE` with `buttons: [ ... ]` (uppercase), and `metadata.target in {overworld_ready, close_menu, select_continue, open_start_menu}`.”
  * **Stop condition in schema:** Extend `MENU_SEQUENCE` schema with `metadata.target` and `metadata.max_frames`. The skill owns the assertion for the target condition; if not met, it raises HALT with `ADHERENCE_TARGET_UNMET`.
  * **Few‑shot refresh:** Keep `docs/examples/menu_boot_sequence.json` synced to the exact snapshot format you emit (options list, cursor index, id). Place the target explicitly in the example.
  * **Zero‑temp, JSON‑only mode:** Use function‑calling/JSON schema tools on the LLM side (or strict regex validators + auto‑repair) so invalid JSON never reaches the executor.

**Example (what the planner should send for boot‑to‑overworld):**

```json
{
  "plan_id": "boot_to_overworld_v1",
  "kind": "MENU_SEQUENCE",
  "buttons": ["START","A","A"],
  "metadata": {
    "target": "overworld_ready",
    "max_frames": 180,
    "notes": "Title->Continue->Overworld"
  }
}
```

### 4) Plan execution doesn’t “prove” progress to SR‑FBAM

* **Symptom:** Skills press buttons, the game advances, but SR‑FBAM never declares `PLANLET_COMPLETE`. Watchdogs fire; you loop back to planner.
* **Causes:** Plan adherence predicate is too weak (e.g., only “menu closed”) or too strict (timeouts too short), or keypress semantics are wrong (hold vs pulse).
* **Fixes:**

  * **Per‑button progress predicates:** For each button in a `MENU_SEQUENCE`, encode a progress check the extractor can verify (cursor moved, menu id changed, overworld bit true). On failure, raise HALT with a *specific* adherence code: `ADHERENCE_CURSOR_NOT_MOVED`, `ADHERENCE_MENU_NOT_CLOSED`, etc.
  * **Pulse semantics:** Ensure skills send **edge‑triggered** presses (`press` → wait N frames → `release`), not long holds, and that N covers animation frames (tune 6–10 frames on classic Pokémon menus).
  * **Watchdog budgets:** Increase `--planlet-watchdog-steps` for menus; many screens need ~40–80 frames for two ‘A’ confirms plus fades. Too‑tight budgets look like stalls.
  * **Idempotent skills:** If a plan step is re‑executed (due to retry), skills should be safe (e.g., pressing `A` again on “Continue?” should either advance or no‑op, then re‑assert the target condition).

---

## C. Telemetry you need to see on every menu HALT (and how you’ll use it)

Add these fields (if they aren’t there already) to `OverworldTraceRecorder` for steps where `is_menu=true`:

* `telemetry.overworld.menu_snapshot`: `{menu_id, menu_name, options[], cursor_index, can_scroll, visible_confirm}`
* `sr_fbam.active_plan.kind` and `adherence_code` (enumerated, not a free string)
* `state_hash.components`: the breakdown (so you can spot missing bits like cursor)
* `gate_probs`: values for `PLAN_LOOKUP`, `PLAN_STEP`, `HALT`, confidence gate
* `planner.source`: `llm` vs `cache`, and `plan_id` returned
* `button_events`: `[{"button":"START","down_frame":F1,"up_frame":F3}, ...]`

With those fields, a single JSONL slice will tell you **which seam failed**: no HALT → extractor/gating; HALT/no plan → coordinator/schema; plan/no progress → skill adherence; progress/no complete → gate threshold/watchdog.

---

## D. Caching & hashing: avoid “same plan for different menu”

* **Problem:** If the state hash ignores cursor or submenu ids, the cache will hand you a prior `MENU_SEQUENCE` that was correct for a different cursor position.
* **Fix:** **Hash on (menu_id, options[], cursor_index)** at minimum. Include an 8–16 byte RAM fingerprint for menu state bits. Caches stay enabled *without* violating the “planner mandatory” rule; you’re just giving the planner identical context → identical output.

---

## E. Planner prompt hardening (drop‑in text you can add)

> **When `is_menu` is true:**
> ‑ You MUST return exactly one planlet of `kind: "MENU_SEQUENCE"`.
> ‑ Provide `buttons` as an ordered array of uppercase strings from {A, B, START, SELECT, UP, DOWN, LEFT, RIGHT}.
> ‑ Include `metadata.target` ∈ {`overworld_ready`, `close_menu`, `select_continue`, `open_start_menu`} and `metadata.max_frames` (int).
> ‑ Do not return `NAVIGATE_TO` while a menu is open.
> ‑ Keep all explanations in `metadata.notes`; the executor ignores free text.

Having a crisp contract like this materially reduces planner variance and invalid JSON.

---

## F. Minimal harnesses that catch this before live runs

1. **`scripts/probe_menu.py`** – Loads a savestate with Start menu open, runs the loop until either `overworld_ready` or N frames. Emits a one‑page report (HALTs, plan ids, adherence codes).
2. **`scripts/replay_menu_halts.py`** – Replays recorded HALT packets against `PlanCoordinator` (fake/mocked) and validates schema + targets.
3. **`scripts/step_sim_menu.json`** – A gold file with {buttons → expected extractor deltas} so you can unit‑test the skill adherence logic without the planner.

---

## G. Why this should work (and stay robust)

Your architecture already centralizes behavior in SR‑FBAM gates/skills and pushes all decision‑making to planlets. The only reason menus still block is that one of the seams above is under‑specified. Making menus **first‑class planlets with explicit targets and adherence predicates** gives the gates real signals to decide on `PLAN_STEP` vs `HALT` and gives the planner a narrow, schema‑enforced surface area. This is consistent with SR‑FBAM’s design philosophy: discrete, symbolic operations with explicit HALT conditions generalize and recover better than soft heuristics. 

---

### TL;DR checklist to unblock you now

* [ ] Hand off only after `_ensure_bootstrapped()` asserts `overworld_ready` (two‑phase boot).
* [ ] Expand `menu_snapshot` + state hash to include `{menu_id, options[], cursor_index, RAM bits}`.
* [ ] Clamp planner: `MENU_SEQUENCE` only when `is_menu=true`, with `metadata.target` and `max_frames`.
* [ ] Implement per‑button adherence predicates + edge‑triggered presses; loosen watchdogs for menus.
* [ ] Add the three harnesses so you can catch this offline.

If you paste a single HALT packet from a stuck run (the menu snapshot + the planlet that came back), I can call out exactly which seam is failing and suggest the smallest code diff to fix it.


## 2. Subsystem Overview

**PyBoy + Adapter**  
- `PyBoyPokemonAdapter` boots the ROM, manages savestates (watchdog reloads), and yields emulator surfaces (framebuffer, RAM).  
- `_ensure_bootstrapped` (with `--policy-boot`) drives past the title/menu screens so SR-FBAM starts from an overworld-ready state.

**Perception & State Extraction**  
- `OverworldExtractor` ingests each PyBoy frame and RAM snapshot, projecting the active map, player pose, NPC entities, items, and menu state.  
- Extraction outputs feed both SR-FBAM gates and the summariser that briefs the planner.

**SR-FBAM Core**  
- Gate heads (`PLAN_LOOKUP`, `PLAN_STEP`, skill selectors, confidence gate) decide whether to request a planlet, execute the active planlet, or raise HALT.  
- Skill registry maps planlet ops (`NAVIGATE_TO`, `MENU_SEQUENCE`, `INTERACT`, etc.) to executable scripts that press buttons, wait on conditions, and update SR-FBAM’s internal entity cache.  
- SR-FBAM persists entity state, adheres to plan metadata, and logs the gate probabilities, adherence codes, and failure reasons that power downstream telemetry.

**Planlet Service & Planner**  
- `PlanCoordinator` captures HALT context, composes the planner prompt (including overworld summary + menu snapshot), and invokes the configured backend (`fake`, `mock`, or live LLM).  
- The LLM must return a JSON planlet conforming to the schema in `docs/PLANLET_SCHEMA.md`; supported kinds today include `NAVIGATE_TO`, `MENU_SEQUENCE`, `OPEN_MENU`, `USE_ITEM`, `HANDLE_ENCOUNTER`.  
- Planlets are cached (when enabled) for identical state hashes, but caches never bypass the planner requirement—the cache simply stores prior LLM outputs.

**Telemetry & Storage**  
- `OverworldTraceRecorder` streams JSONL with per-step state, gate decisions, planlet metadata, and menu snapshots.  
- Optional outputs: `--telemetry-out` for step-level JSONL, `--metadata-out` for run configuration, plus planner token/accounting logs.

**Watchdogs**  
- `--planlet-watchdog-steps` re-requests a plan when SR-FBAM consumes steps without progressing the active planlet.  
- `--stall-watchdog-steps` reloads a PyBoy savestate when overworld step counters freeze. Both watchdogs rely on the planner to recover—there is no scripted fallback.

---

## 3. Frame-to-Plan Loop

1. **Frame Capture** – PyBoy surfaces a framebuffer + RAM view each tick.  
2. **Extraction** – `OverworldExtractor` builds a structured state (map, menus, entities).  
3. **Gate Evaluation** – SR-FBAM gates score the state; when an active planlet exists, `PLAN_STEP` consumes its next instruction.  
4. **HALT Decision** – If confidence drops, the planlet completes, or SR-FBAM detects an unhandled situation (e.g., new menu, encounter), it emits a HALT event with reason codes.  
5. **Planner Invocation** – The controller forwards the HALT context through the Planlet Service to the LLM. No deterministic rewrites or local fallbacks occur.  
6. **Planlet Validation** – Returned JSON must parse and pass schema checks; invalid responses cause the controller to retry with the planner (or stop the run if retries exhaust).  
7. **Skill Execution** – Planlet ops compile into SR-FBAM skills. The executor replays button presses through PyBoy, monitors adherence, and updates telemetry.  
8. **Loop Continuation** – When SR-FBAM reports `PLANLET_COMPLETE`, control returns to step 1 with the new game state.

At all times, SR-FBAM is the actuator: the controller never injects raw button presses outside of bootstrapping helpers.

---

## 4. HALT Lifecycle & Planner Expectations

- **HALT Triggers**: menu detection, navigation dead-ends, unexpected encounters, watchdog timeouts, or missing planlets.  
- **Context Packets**: include map position, facing, nearby entities, recent gate history, and (if `is_menu` is true) a menu snapshot with cursor positions.  
- **LLM Contracts**:  
  - Return a single planlet with `plan_id`, `kind`, `ops[]`, and metadata fields defined in the schema checklist.  
  - `MENU_SEQUENCE` must provide an ordered `buttons` list (uppercase).  
  - Navigation planlets specify targets (tile, entity) and optional guardrails (max steps, avoidance).  
  - All explanatory fields remain in JSON metadata; natural-language comments are stripped before execution.
- **Failure Handling**:  
  - Invalid JSON → retry the planner (with the same context) up to the configured limit.  
  - Planner refusal or repeated validation failure → abort the run and flag the operator (see Runbook §3-4).  
  - Watchdogs simply trigger new planner calls; they do not fall back to canned button sequences.

---

## 5. Telemetry, Observability, and Artifacts

- **Per-step JSONL**: captures frame index, gate chosen, confidence, active planlet state, and menu snapshots (`telemetry.overworld.menu_snapshot`).  
- **Planlet Logs**: store planner source (`llm`, `cache`), tokens, adherence metrics, and watchdog events.  
- **Metadata Bundles**: run configuration (CLI arguments, planner backend/model, ROM checksum) saved via `--metadata-out` for reproducibility.  
- **Dashboards**: downstream jobs (`scripts/summarize_telemetry.py`) roll JSONL into summaries for gate mix, HALT taxonomy, plan success, and planner latency.  
- **Debug Artifacts**: savestates from watchdog reloads, planner prompts/responses (when logging is enabled), and PyBoy screenshots can be attached for postmortems. See `docs/OVERWORLD_RUNBOOK.md` for operator playbooks.

---

## 6. Operational Notes

- **Planner Is Mandatory**: Runs abort immediately if planner credentials are missing. The controller never synthesizes planlets locally.  
- **Menu Handling**: Keep few-shot examples (`docs/examples/menu_boot_sequence.json`) up to date so the LLM reliably emits `MENU_SEQUENCE` planlets.  
- **Prompt Hygiene**: Update `src/plan/prompt_builder.py` whenever new planlet kinds or failure modes emerge; re-run short tests to confirm the planner reacts correctly.  
- **Entity Fidelity**: SR-FBAM’s extractor must stay aligned with ROM offsets. Use `scripts/debug_overworld_addresses.py` after ROM revisions to avoid silent entity drift.  
- **Telemetry Budget**: Monitor JSONL size and plan token expenditure—planner loops can grow expensive if HALTs spike. Adjust prompts before touching controller logic.

---

## 7. Current Focus & Open Work

- Wire the live planner integration inside `scripts/run_overworld_controller.py` so HALTs always route through `PlanCoordinator` (see Migration Plan §Milestone 1-2).  
- Expand planlet coverage beyond navigation by enriching prompts with encounter/menu exemplars.  
- Harden boot automation and watchdog recovery so overnight runs require minimal manual intervention.  
- Capture overworld telemetry at scale, unblock mixed-mode curriculum ingestion, and feed measurements back into gate calibration (Migration Plan §Phase 5).  
- Continue documenting recovery procedures in `docs/OVERWORLD_RUNBOOK.md` whenever planner policies or planlet schemas change.

---

By adhering to this loop—frames into SR-FBAM, HALTs straight to the LLM, planlets executed strictly through SR-FBAM skills—we maintain a single consistent runtime story that operators, planners, and modelers can reason about together.

