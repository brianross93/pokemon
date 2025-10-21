# Plan Curriculum & Augmentation Notes

This document expands on the knobs defined in `configs/train_plan.yaml` and captures the rationale behind the Phase 5 mixed-mode schedule.

## Curriculum Phases

1. **Battle Warmup (epochs 0-3)**  
   - Ratio: 85% battle / 15% overworld (corridor-only).  
   - Goal: Stabilise gate heads on battle traces while introducing corridor planlets slowly enough to avoid catastrophic gate drift.  
   - Heuristics: Encode fraction should remain below 0.35; raise alerts if plan lookup falls under 0.20.

2. **Mixing Ramp (epochs 4-6)**  
   - Ratio: 65% battle / 35% overworld.  
   - Gate targets: Encourage PLAN_LOOKUP ≥ 0.18, PLAN_STEP ≥ 0.12 while keeping adherence ≥ 0.65.  
   - Augmentations: Enable encounter injection to exercise recovery branches.

3. **Balanced Phase (epochs 7-10)**  
   - Ratio: 50% battle / 50% overworld (requires ≥50 traces per overworld skill).  
   - Plan dropout fades linearly after epoch 8 once adherence stabilises above 0.68.  
   - Monitor encode/query/skip blend; trigger alerts for skip spikes > baseline +0.10.

## Augmentation Recipes

- **Tile jitter**: Applies ±1 tile offsets to overworld frames with 0.4 probability to desynchronise memorised layouts.  
- **Encounter injection**: Inserts scripted encounters (e.g., Pidgey in corridor, hidden item fetch) to test policy robustness; each template lists minimum step to avoid disrupting early warmup.  
- **Plan dropout**: Drops plan features/gate targets with 20% / 5% probability respectively to make the controller resilient to missing planlets; adherence flag is clamped to ≥0.2 to avoid degenerate zero vectors.

## Gating Heuristics & Alerts

- **Warmup**: Encode ≤0.35, PLAN_LOOKUP ≥0.20; alert if adherence drops >0.05 absolute compared to the previous evaluation.  
- **Mixed phase**: Target encode ∈[0.28, 0.40], PLAN_LOOKUP ∈[0.18, 0.28], PLAN_STEP ∈[0.12, 0.22]; expected adherence ≥0.68.  
- **Alerts**: raise when `skip_fraction` exceeds baseline +0.10 or adherence decreases >0.05 for two consecutive evaluations.

## Telemetry Requirements

- Corridor seed set: `corridor_a`..`corridor_d`, two captures each (≥128 steps).  
- Skills covered: navigate, interact, talk, buy, pickup, use_item, menu, wait (≥50 traces per skill).  
- Each capture should record gate/adherence JSONL via `--gate-jsonl` to allow post-run alerting and curriculum QA.

## Next Steps

1. Populate real corpus paths in `configs/train_plan.yaml` once telemetry landing zones are ready.  
2. Wire the sampler to consume the YAML via Hydra or equivalent config system.  
3. Add monitoring hooks that compare live gate metrics against the thresholds above and surface alerts to the training dashboard.
