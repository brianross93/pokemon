#!/usr/bin/env python3
"""
Run the overworld executor on a scripted observation stream and record traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

from src.plan import PlanBundle, load_plan_bundle
from src.srfbam.tasks.overworld import OverworldExecutor
from src.overworld.recording import OverworldTraceRecorder
from src.overworld.skills import BaseSkill, SkillProgress, SkillStatus


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record overworld executor traces from scripted observations.")
    parser.add_argument("--plan-json", type=Path, required=True, help="Path to a plan bundle JSON file.")
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="JSONL file containing one observation mapping per line.",
    )
    parser.add_argument("--trace-out", type=Path, required=True, help="Destination JSONL trace path.")
    parser.add_argument(
        "--passive-skill",
        action="store_true",
        help="Use a passive stub skill for NavigateSkill to simplify scripted runs.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on executed steps (0 for unlimited).")
    return parser.parse_args(argv)


class PassiveNavigateSkill(BaseSkill):
    """Stub skill used for deterministic trace collection."""

    name = "PassiveSkill"

    def legal_actions(self, observation, graph):
        return ({"kind": "wait"},)

    def select_action(self, observation, graph):
        return {"kind": "wait"}

    def progress(self, graph):
        return SkillProgress(status=SkillStatus.IN_PROGRESS)


def load_observations(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    bundle: PlanBundle = load_plan_bundle(args.plan_json)

    skill_registry = None
    if args.passive_skill:
        skill_registry = {"NavigateSkill": PassiveNavigateSkill}

    executor = OverworldExecutor(skill_registry=skill_registry)
    executor.load_plan_bundle(bundle)

    max_steps = max(0, int(args.max_steps))
    steps_executed = 0

    with OverworldTraceRecorder(args.trace_out) as recorder:
        executor.register_trace_recorder(recorder)

        for observation in load_observations(args.observations):
            result = executor.step(observation)
            steps_executed += 1
            if result.status == "PLAN_COMPLETE":
                break
            if max_steps and steps_executed >= max_steps:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
