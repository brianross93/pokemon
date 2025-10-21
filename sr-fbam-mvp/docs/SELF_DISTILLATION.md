# Self-Distillation Pipeline

This document outlines the orchestration and metadata required for the Phase 5 self-distillation loop.

## Overview

The self-distillation process follows four stages:

1. **LLM-on capture** - run the teacher controller with LLM enabled to produce new planlets and telemetry.
2. **Relabel** - enrich captured planlets with generation metadata (model, prompts, cache hits) and teacher scorecards.
3. **Retrain** - fine-tune the student controller using the updated plan features, typically freezing the encoder to calibrate gate heads.
4. **LLM-off evaluation** - optional regression pass with the student model to measure reliance on planlets vs. LLM.

`scripts/self_distill.sh` coordinates these stages and writes structured metadata for dashboards/alerts.

## Usage

1. Copy `configs/self_distill/template.env` to a new file (e.g. `configs/self_distill/pyboy.env`) and fill in the commands for your environment. Each variable is executed in order by `self_distill.sh`.
2. Run the orchestrator:

   ```bash
   bash scripts/self_distill.sh \
     --config configs/self_distill/pyboy.env \
     --teacher-checkpoint checkpoints/preplan.pt \
     --teacher-scorecard results/calibration/teacher_scorecard.json
   ```

3. Outputs are written to `runs/self_distill/<run-id>/` with subdirectories for logs, checkpoints, artifacts, and metadata.

## Metadata Contracts

The orchestrator tracks metadata in `runs/self_distill/<run-id>/metadata.json` with these required fields:

- `run_id`, `started_at`, `completed_at`, `config`
- `teacher_checkpoint` (optional but recommended)
- `teacher_scorecard` (optional)
- `student_checkpoint` (auto-populated after retrain if a checkpoint is deposited in `checkpoints/`)
- `stages[]` entries with `stage`, `status`, `command`, and `log` paths

During the relabel stage the script also emits `metadata/relabel_tags.json` capturing:

- `planlet_tag_file` - where the relabel command writes planlet metadata (JSON/JSONL)
- `teacher_scorecard` - pointer carried forward for downstream jobs
- `teacher_checkpoint` - makes the provenance explicit for dashboards
- `status` - initialised to `pending`; downstream jobs can update once validation succeeds.

Ensure the relabel command populates the file referenced by `RELABEL_TAG_FILE` in the config so downstream ingestion jobs can attach the metadata without guessing paths.

## Promotion & Rollback

Supply `--promotion-target` when launching `self_distill.sh` to automatically copy the newest student checkpoint into a canonical location once the loop finishes. The helper will:

- Back up any existing target checkpoint with a timestamped `.bak.<run-id>` suffix.
- Write promotion metadata to `--promotion-record` (defaults to `metadata/promotion.json`).
- Emit a storage reminder in `--storage-alert-file` (defaults to `metadata/storage_alert.txt`) so infra can verify quota and clean up obsolete checkpoints.

Example:

```bash
bash scripts/self_distill.sh \
  --config configs/self_distill/pyboy.env \
  --promotion-target checkpoints/stable/student_latest.pt
```

The generated promotion metadata includes the original checkpoint path and the backup path, providing a straightforward manual rollback lever if regressions are observed after deployment.
