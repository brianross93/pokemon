#!/usr/bin/env python3
"""
Ingest telemetry logs and materialise dashboard-friendly summaries.

The script discovers telemetry JSONL files (new schema or legacy) and
generates aggregated metrics for the overall corpus plus per-domain
breakdowns.  The output is written to ``results/summary/telemetry_dashboard.json``
by default, which is where the existing visualization notebooks expect to
find consolidated metrics.

Example
-------
Run with defaults (scans ``runs/*.jsonl`` and ``results/pkmn_logs/*.jsonl``)::

    python scripts/update_dashboard_metrics.py

Provide explicit locations and pretty-print the output::

    python scripts/update_dashboard_metrics.py \\
        --inputs data/logs/**/*.jsonl more_logs/*.jsonl \\
        --output results/summary/custom_dashboard.json \\
        --pretty
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from src.telemetry import iter_entries
from src.telemetry.analytics import summarize_entries

DEFAULT_PATTERNS = ("runs/*.jsonl", "results/pkmn_logs/*.jsonl")
DEFAULT_OUTPUT = Path("results/summary/telemetry_dashboard.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate aggregated telemetry summary for dashboards.")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=DEFAULT_PATTERNS,
        help="Glob patterns or file paths to telemetry JSONL logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON file (default: results/summary/telemetry_dashboard.json).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (indent=2).",
    )
    return parser.parse_args(argv)


def resolve_inputs(patterns: Iterable[str]) -> List[Path]:
    files: Set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        if path.exists():
            files.add(path.resolve())
            continue
        for candidate in Path().glob(pattern):
            if candidate.is_file():
                files.add(candidate.resolve())
    return sorted(files)


def load_entries_from_files(files: Iterable[Path]):
    for path in files:
        yield from iter_entries(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = resolve_inputs(args.inputs)
    if not files:
        raise SystemExit("No telemetry logs found. Provide paths via --inputs.")

    entries = list(load_entries_from_files(files))
    if not entries:
        raise SystemExit("Telemetry files were empty after normalization.")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [str(path) for path in files],
        "overall": summarize_entries(entries),
        "battle": summarize_entries(entries, domain="battle"),
        "overworld": summarize_entries(entries, domain="overworld"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2 if args.pretty else None),
        encoding="utf-8",
    )
    print(f"[ok] wrote dashboard summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

