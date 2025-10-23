# Overworld Operator Runbook

This runbook documents the procedures for launching long-form overworld runs, handling menu navigation, and responding to HALT/watchdog events. It reflects the latest planner-only workflow: every planlet must originate from the configured planner/LLM—no deterministic fallbacks exist.

---

## 1. Pre-Run Checklist

1. **Planner credentials**  
   - Ensure `--planner-backend` is set (`fake`, `mock`, `openai`, `anthropic`, etc.).  
   - Provide `--planner-model`, API key, and base URL (when applicable). Runs abort immediately if the planner is missing.

2. **Prompt expectations**  
   - Planner must emit planlets using the schema in `docs/PLANLET_MIGRATION_CHECKLIST.md`.  
   - Supported kinds today: `NAVIGATE_TO`, `MENU_SEQUENCE`, `OPEN_MENU`, `USE_ITEM`, `HANDLE_ENCOUNTER`.

3. **Run metadata + telemetry**  
   - Use `--metadata-out` to capture full CLI/planner configuration.  
   - Use `--telemetry-out` to stream JSONL logs for every step (menu snapshots + plan metadata).

4. **Watchdog configuration**  
   - `--planlet-watchdog-steps` re-requests a plan when the active planlet stalls.  
   - `--stall-watchdog-steps` reloads the save-state slot when the overworld step counter stops advancing.  
   - `--watchdog-save-slot` (optional) persists a baseline PyBoy snapshot after boot.

5. **ROM / RAM offsets**  
   - Validate RAM offsets for the current ROM via `scripts/debug_overworld_addresses.py` before overnight runs.

---

## 2. Launching the Controller

Example invocation:

```bash
python scripts/run_overworld_controller.py \
  --rom Pokemon\ Blue.gb \
  --planner-backend openai \
  --planner-model gpt-5 \
  --telemetry-out runs/night1.jsonl \
  --metadata-out runs/night1.meta.json \
  --planlet-watchdog-steps 900 \
  --stall-watchdog-steps 600 \
  --watchdog-save-slot 3 \
  --window null \
  --steps 50000
```

If the planner fails or is misconfigured, the controller exits and logs the reason. Do **not** restart with missing planner parameters—menu navigation will stall immediately.

---

## 3. Menu HALT Workflow

1. **Detection**  
   - Telemetry includes `telemetry.overworld.menu_snapshot` with fields `state`, `cursor`, `is_menu`, and `menus[]`.  
   - The planner receives this context in the prompt (`Menu Context` section).

2. **Planner expectations**  
   - When `is_menu` is true, the planner **must** emit a `MENU_SEQUENCE` planlet with a `buttons` array (e.g., `["START", "A", "A"]`).  
   - The controller also accepts a minimal JSON payload `{ "buttons": [...] }` and wraps it into a canonical `MENU_SEQUENCE` planlet automatically.  
   - Example template: `docs/examples/menu_boot_sequence.json`.

3. **Execution**  
   - The executor binds `MENU_SEQUENCE` to `MenuSequenceSkill`, replaying the exact button order.  
   - A planlet is considered complete when the sequence finishes; watchdogs only re-request a plan if the planner stalls.

4. **Operator action items**  
   - If HALTs repeat with reason `MENU_STALLED` or `no-path`, inspect the telemetry to confirm the planner is returning a `MENU_SEQUENCE`.  
   - If not, adjust the planner prompt (see section 4) and restart the run—do not inject manual inputs.

---

## 4. Planner Prompt & Policy Guidance

Recent prompt updates (see `src/plan/prompt_builder.py`):

- The planner is reminded of supported planlet kinds at the top of every prompt.
- Menu context now lists open menus and their paths/states.
- The prompt explicitly directs the LLM to emit `MENU_SEQUENCE` planlets while menus are open.

### Policy tuning steps

1. **Add menu exemplars**  
   - Seed the planner with few-shot examples containing `MENU_SEQUENCE` JSON (e.g., `menu_boot_sequence.json`).  
   - Always include the button array as uppercase strings; `START`, `A`, `B`, `UP`, `DOWN`, etc.

2. **Monitor failure taxonomy**  
   - Telemetry logs stalled planlets with reasons (`MENU_DESYNC`, `no-path`, etc.).  
   - Update the prompt with new failure explanations so the LLM can recover intelligently.

3. **Iterative testing**  
   - Run short sessions capturing telemetry.  
   - Use `scripts/summarize_telemetry.py --print-summary` to inspect gate mix, halt reasons, and plan sources.  
   - Refine planner prompts until `MENU_SEQUENCE` planlets appear reliably during boot/title menus.

---

## 5. Failure Recovery & Watchdogs

1. **Planlet watchdog**  
   - Re-requests a planlet after N controller steps without progress.  
   - If the planner repeatedly produces invalid planlets, telemetry will show repeated `watchdog_planlet` reasons—tune the prompt.

2. **Stall watchdog**  
   - Reloads the configured save slot when the overworld step counter stalls.  
   - After reload, the controller immediately re-requests a plan from the planner.

3. **Escalation**  
   - Persistent stalls or planner errors should be reported to the planner owners.  
   - Include telemetry snippets (last HALT, plan metadata, `menu_snapshot`) for debugging.

---

## 6. Appendix

- `docs/examples/menu_boot_sequence.json` — reference `MENU_SEQUENCE` planlet.  
- `scripts/summarize_telemetry.py` — aggregates telemetry to Parquet/dashboards.  
- `docs/PLANLET_MIGRATION_CHECKLIST.md` — end-to-end planlet migration status.  
- `scripts/debug_overworld_addresses.py` — RAM offset validation tool.

Keep this runbook pinned during overnight operations. Update it whenever planlet schemas or planner policies change.
