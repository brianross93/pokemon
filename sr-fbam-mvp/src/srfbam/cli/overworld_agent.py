"""
Entry point for running the SR-FBAM overworld executor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.plan import PlanCompiler, PlanCompilationError, PlanValidationError, load_plan_bundle, validate_plan_json
from src.overworld import DEFAULT_OVERWORLD_RAM_MAP, OverworldExtractor, OverworldMemory


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SR-FBAM overworld agent scaffolding.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--goal-json", type=Path, help="Path to a planlet bundle JSON file.")
    group.add_argument(
        "--prompt",
        type=str,
        help="Free-form goal description to feed into the planner (LLM integration pending).",
    )

    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Optional overworld snapshot JSON for extractor smoke testing.",
    )
    parser.add_argument(
        "--ram-snapshot",
        type=Path,
        help="Optional binary RAM dump to decode using DEFAULT_OVERWORLD_RAM_MAP.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Emit plan summary to stdout after validation.",
    )
    return parser.parse_args(argv)


def _load_plan_from_prompt(prompt: str) -> str:
    raise SystemExit(
        "Planner prompt mode is not yet implemented. Provide --goal-json with a plan bundle instead."
    )


def load_plan_bundle_from_args(args: argparse.Namespace):
    if args.goal_json:
        return load_plan_bundle(args.goal_json)
    prompt_payload = _load_plan_from_prompt(args.prompt)
    return validate_plan_json(prompt_payload)


def load_snapshot_data(args: argparse.Namespace) -> dict:
    payload: dict = {}
    if args.snapshot:
        payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if args.ram_snapshot:
        ram_bytes = args.ram_snapshot.read_bytes()
        payload.setdefault("ram", ram_bytes)
        payload.setdefault(
            "metadata",
            {"ram_addresses": dict(DEFAULT_OVERWORLD_RAM_MAP)},
        )
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    try:
        bundle = load_plan_bundle_from_args(args)
    except PlanValidationError as exc:
        details = "\n".join(f"- {err}" for err in exc.errors) if exc.errors else ""
        message = f"Plan validation failed: {exc}"
        if details:
            message = f"{message}\n{details}"
        raise SystemExit(message) from exc

    compiler = PlanCompiler()
    try:
        compiled = compiler.compile(bundle)
    except PlanCompilationError as exc:
        raise SystemExit(f"Unable to compile plan bundle: {exc}") from exc

    if args.print_plan:
        print(f"Plan {compiled.plan_id}")
        if compiled.goal:
            print(f"  Goal: {compiled.goal}")
        for planlet in compiled.planlets:
            print(f"  - [{planlet.skill}] {planlet.spec.id}: {planlet.spec.kind}")

    snapshot_payload = load_snapshot_data(args)
    if snapshot_payload:
        extractor = OverworldExtractor()
        memory = OverworldMemory()
        writes = extractor.extract(snapshot_payload)
        for op in writes:
            memory.write(op)

        counts = memory.summarise_nodes()
        print("Snapshot summary:")
        for node_type, count in sorted(counts.items()):
            print(f"  {node_type}: {count}")
        print(f"  hop_trace_entries: {len(memory.drain_hops())}")

    else:
        print("No snapshot provided; plan validated but executor not run.")

    return 0


__all__ = ["main", "parse_args"]
