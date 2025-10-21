## Gate Calibration Scheduler Templates

This folder captures reference argument sets for sweeping learning-rate schedules while calibrating gate heads. Each JSON file lists a representative `python -m src.training.train_battle_il ...` command that exercises the new scheduler flags.

- `linear.json` – linear decay with optional warmup and minimum LR.
- `cosine.json` – cosine annealing toward a minimum LR.
- `step.json` – step decay with configurable gamma and step size.

Replace placeholder paths (`<path-to-*>`) before executing the commands, or copy the entries into your Hydra launch configs.
