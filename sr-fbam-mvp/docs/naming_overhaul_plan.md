# Naming Overlay Improvements Worklog

## Objectives
- Enhance the naming-screen extractor so downstream systems receive a stable 20x18 grid, cursor history, and preset detection metadata.
- Enrich planner prompts with overlay context (naming snapshot, adjacency stats, overlay flags) so HALTs stay in `MENU_SEQUENCE` mode while the overlay is active.
- Detect and recover from naming stalls by resetting cached macros when the same planlet repeats without state changes.
- Extend planner cache/service APIs to accept feedback from the controller and evict stale entries.
- Back-fill regression tests covering the new extractor metadata, prompt payload, and watchdog behaviour.

## Task Breakdown
1. **Extractor upgrades**
   - [x] Maintain a bounded cursor history keyed by frame hash.
   - [x] Capture preset rows (e.g., RED / BLUE options) alongside the letter grid.
   - [x] Emit normalized metadata: `grid_letters`, `cursor`, `cursor_history`, `presets`, and `dialog_text`.

2. **Prompt enrichment**
   - [x] Surface naming snapshot + adjacency stats in `PlanCoordinator._overworld_snapshot_for_prompt`.
   - [x] Thread the metadata into `PlanletProposer.generate_planlet` so the planner sees overlay/cursor state.

3. **Stall detection & cache feedback**
   - [x] Track repeated menu planlets and detect unchanged naming snapshots.
   - [x] Invalidate cached macros and log feedback through `PlanletService`.

4. **Planner cache API updates**
   - [x] Provide helpers to drop cache entries by key or planlet id.
   - [x] Ensure controller hooks call these helpers on stall detection.

5. **Testing**
   - [x] Extend overworld extractor tests for new naming metadata.
   - [x] Update planlet proposer tests for prompt wiring and overlay constraints.
   - [ ] Add controller-level tests (if feasible) for stall detection logic.

## Notes
- Prior runs (`runs/live_visual_5000_nav_fix5.jsonl`) show repeated `MENU_SEQUENCE` outputs due to missing overlay state; use them for regression evidence.
- `PlanletService.request_overworld_planlet` currently returns cached planlets verbatim; we’ll need new feedback hooks to evict stale macros after stall detection.
- Keep all new metadata ASCII and document structures with inline comments sparingly.
