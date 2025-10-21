#!/usr/bin/env python3
"""
Summarise SR-FBAM telemetry logs for dashboards and parity checks.

This command consumes one or more JSONL files produced by the battle agent
or overworld executor and emits aggregate metrics (averaged latency,
gate fractions, etc.) using the consolidated telemetry schema.

Examples
--------
Summarise a single battle run::

    python scripts/summarize_telemetry.py --input runs/test_cpu.jsonl --domain battle

Combine multiple files and pretty-print the result::

    python scripts/summarize_telemetry.py --input runs/*.jsonl --pretty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List

from src.telemetry import iter_entries
from src.telemetry.analytics import summarize_entries


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise SR-FBAM telemetry logs.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more telemetry JSONL files (battle or overworld).",
    )
    parser.add_argument(
        "--domain",
        choices=["battle", "overworld", "all"],
        default="all",
        help="Optional domain filter. 'all' aggregates every record.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON summary (defaults to stdout).",
    )
    return parser.parse_args(argv)


def load_all_entries(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Telemetry log not found: {path}")
        yield from iter_entries(path)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    domain_filter = None if args.domain == "all" else args.domain
    entries = list(load_all_entries(args.input))
    summary = summarize_entries(entries, domain=domain_filter)

    payload = json.dumps(summary, indent=2 if args.pretty else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

