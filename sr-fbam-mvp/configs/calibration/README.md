# Calibration Runs

This folder tracks command templates for gate calibration and scheduler sweeps.
Populate per-run argument lists using the new options exposed in `src/training/train_battle_il.py`:

- `--load-checkpoint` to resume from a pre-plan baseline.
- `--preplan-checkpoint-out` / `--postplan-checkpoint-out` to snapshot weights around the calibration phase.
- `--freeze-backbone`, `--freeze-action-head`, `--freeze-gate-head` to control which components are trainable.

Add JSON or YAML files alongside this README with the exact argument sets once datasets and paths are finalized. Each file should include:

```jsonc
{
  "description": "short human-readable label",
  "train_args": [
    "--train", "data/planlets/battle_train.jsonl",
    "--val", "data/planlets/battle_val.jsonl",
    "--epochs", "3",
    "--freeze-backbone",
    "--preplan-checkpoint-out", "results/checkpoints/preplan.pt",
    "--postplan-checkpoint-out", "results/checkpoints/postplan.pt"
  ]
}
```

Use the same schema to document scheduler variants (linear, cosine, step) once those configs are available.
