#!/usr/bin/env python3
"""
Batch-normalise legacy telemetry logs into the consolidated schema.

This utility rewrites JSONL telemetry files (battle or overworld) using
``src.telemetry.normalize_entry`` so downstream consumers can drop the old
flat structures safely.

Usage
-----
Normalise a directory in place (writes mirrored structure under ``out/``)::

    python scripts/normalize_telemetry_logs.py --input legacy_logs --output out

Provide individual files and overwrite the originals::

    python scripts/normalize_telemetry_logs.py --input runs/battle.jsonl --in-place
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from src.telemetry import iter_entries, normalize_entry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalise telemetry JSONL files to the consolidated schema.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Telemetry JSONL files or directories containing them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination directory for normalised files (mirrors input layout).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".normalized",
        help="Suffix appended before .jsonl when writing (ignored with --in-place).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input files instead of writing to a separate directory.",
    )
    return parser.parse_args(argv)


def iter_files(inputs: Iterable[Path]) -> Iterator[Path]:
    for item in inputs:
        if item.is_dir():
            yield from sorted(path for path in item.rglob("*.jsonl") if path.is_file())
        elif item.is_file():
            yield item
        else:
            raise FileNotFoundError(f"Telemetry input not found: {item}")


def write_normalized(src: Path, dest: Path) -> None:
    records = list(iter_entries(src))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.in_place and args.output is not None:
        raise SystemExit("Use either --in-place or --output, not both.")

    files = list(iter_files(args.input))
    if not files:
        raise SystemExit("No telemetry files matched the provided inputs.")

    for src in files:
        if args.in_place:
            # Normalise in memory first to avoid partial overwrite on failure.
            temp_path = src.with_suffix(src.suffix + ".tmp_normalized")
            write_normalized(src, temp_path)
            temp_path.replace(src)
            print(f"[ok] normalised {src}")
            continue

        if args.output is None:
            raise SystemExit("Provide --output when not using --in-place.")

        relative = src.relative_to(src.anchor if src.is_absolute() else Path.cwd())
        dest = args.output / relative
        dest = dest.with_suffix(dest.suffix + args.suffix)
        write_normalized(src, dest)
        print(f"[ok] wrote {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

