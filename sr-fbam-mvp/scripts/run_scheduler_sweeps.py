#!/usr/bin/env python
"""
Execute the learning-rate scheduler experiments defined under configs/schedules/*.json.

Each JSON file should expose a `command` array (argv style) with optional placeholders:
    <path-to-battle-train.jsonl>
    <path-to-battle-val.jsonl>

Usage:
    python scripts/run_scheduler_sweeps.py \
        --schedules configs/schedules/*.json \
        --battle data/planlets/battle_train.jsonl \
        --val data/planlets/battle_val.jsonl \
        --curriculum-config configs/train_plan.yaml

If --val is omitted the placeholder (and preceding --val flag) will be pruned.
Use --dry-run to preview commands without executing them.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional


PLACEHOLDER_BATTLE = "<path-to-battle-train.jsonl>"
PLACEHOLDER_VAL = "<path-to-battle-val.jsonl>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LR scheduler sweeps defined in configs/schedules/*.json.")
    parser.add_argument(
        "--schedules",
        type=str,
        default="configs/schedules/*.json",
        help="Glob pattern for scheduler JSON definitions.",
    )
    parser.add_argument(
        "--battle",
        type=Path,
        default=Path("data/planlets/battle_train.jsonl"),
        help="Battle training JSONL path.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=None,
        help="Optional validation JSONL path.",
    )
    parser.add_argument(
        "--curriculum-config",
        type=Path,
        default=None,
        help="Optional curriculum YAML to append to each command.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable to use (defaults to sys.executable).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


def load_schedule_files(pattern: str) -> List[Path]:
    paths = sorted(Path(p) for p in glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No schedule files matched pattern: {pattern}")
    return paths


def build_command(
    base_command: Iterable[str],
    *,
    battle_path: Path,
    val_path: Optional[Path],
    curriculum_config: Optional[Path],
    python_executable: Optional[str],
) -> List[str]:
    command: List[str] = []
    for token in base_command:
        if token == PLACEHOLDER_BATTLE:
            command.append(str(battle_path))
            continue
        if token == PLACEHOLDER_VAL:
            if val_path is None:
                if command and command[-1].startswith("--"):
                    command.pop()
                continue
            command.append(str(val_path))
            continue
        command.append(token)

    if python_executable and command and command[0] == "python":
        command[0] = python_executable

    if curriculum_config and "--curriculum-config" not in command:
        command.extend(["--curriculum-config", str(curriculum_config)])

    return command


def run_command(command: List[str], *, dry_run: bool) -> int:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"[sweep] {printable}")
    if dry_run:
        return 0
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"[sweep] command failed with exit code {result.returncode}")
    return result.returncode


def main() -> int:
    args = parse_args()
    schedule_files = load_schedule_files(args.schedules)
    any_failures = False
    python_exec = args.python

    for path in schedule_files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        command = payload.get("command")
        if not isinstance(command, list):
            print(f"[warn] Skipping {path}: no 'command' array found.")
            continue
        resolved = build_command(
            command,
            battle_path=args.battle,
            val_path=args.val,
            curriculum_config=args.curriculum_config,
            python_executable=python_exec,
        )
        rc = run_command(resolved, dry_run=args.dry_run)
        if rc != 0:
            any_failures = True

    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
