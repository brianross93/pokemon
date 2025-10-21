# Self-Distillation Command Templates

The `scripts/self_distill.sh` helper expects a shell-compatible `.env` file that defines
per-stage commands and optional metadata paths. Copy `template.env` to a new file,
fill in the commands for your environment, and pass `--config` when launching the script.

### Required variables

- `LLM_ON_CMD` – command to run the teacher policy with LLM enabled and capture telemetry / planlets.
- `RELABEL_CMD` – command that relabels collected planlets (e.g. tag adherence, teacher scores).
- `RETRAIN_CMD` – command that retrains the student controller. Should write checkpoints to `runs/self_distill/<run-id>/checkpoints/`.
- `LLM_OFF_CMD` – optional evaluation run with LLM disabled for comparison.

### Optional variables

- `RELABEL_TAG_FILE` – path to the planlet metadata/JSONL emitted during relabeling.
- `TEACHER_SCORECARD` – path to the teacher evaluation summary.

Commands can use Hydra multi-run syntax directly (for example: `python -m hydra.core.hydra_config --multirun ...`).
Placeholders such as `{RUN_ID}` or environment variables may be expanded by the shell when invoked.
