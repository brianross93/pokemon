"""
Entry point for converting Metamon replay shards into SR-FBAM JSONL corpora.

This script wires the CLI arguments, resolves shard paths, and delegates the
per-turn conversion to `pkmn_battle.ingest.pipeline.convert_shard`. Implement
the converter before running this end-to-end.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

from pkmn_battle.ingest import ConversionConfig, convert_shard, discover_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SR-FBAM imitation-learning JSONL corpus from Metamon shards.",
    )
    parser.add_argument(
        "--mechanics",
        type=Path,
        required=True,
        help="Directory containing the Showdown mechanics files (pokedex.json, etc.).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Replay shard directory, glob, or file path. Repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--format",
        default="gen9ou",
        help="Optional format filter passed to the converter.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional guard to stop after processing N turns (per shard).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of truncating it.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional text file listing shard paths (one per line). When provided, "
            "the script only processes the listed shards. Relative entries are "
            "resolved against the manifest file's parent directory."
        ),
    )
    return parser.parse_args()


def collect_shards(sources: Iterable[Path], manifest: Path | None) -> List[Path]:
    if manifest:
        entries = []
        base_dir = manifest.parent
        with manifest.open("r", encoding="utf-8") as fin:
            for line in fin:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                path = Path(stripped)
                if not path.is_absolute():
                    path = (base_dir / path).resolve()
                if not path.exists():
                    logging.warning("Manifest entry missing: %s", path)
                    continue
                if not _is_lz4(path):
                    logging.warning("Skipping non-LZ4 entry in manifest: %s", path)
                    continue
                entries.append(path)
        return entries

    shards: List[Path] = []
    for source in sources:
        matched = list(discover_shards(source))
        if not matched:
            logging.warning("No shards found under %s", source)
            continue
        shards.extend(matched)
    return shards


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = ConversionConfig(
        mechanics_dir=args.mechanics,
        format_filter=args.format,
        max_turns=args.max_turns,
    )
    shards = collect_shards(args.source, args.manifest)
    if not shards:
        raise SystemExit("No shards matched the provided --source arguments.")

    logging.info("Found %d shards to process.", len(shards))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    total_records = 0
    with args.output.open(mode, encoding="utf-8") as writer:
        for shard_path in shards:
            logging.info("Processing %s", shard_path)
            try:
                total_records += convert_shard(shard_path, config, writer)
            except NotImplementedError as exc:
                logging.error("%s", exc)
                raise SystemExit(
                    "Conversion logic not implemented yet. "
                    "Fill in pkmn_battle.ingest.pipeline.convert_shard."
                ) from exc

    logging.info("Wrote %d decisions to %s", total_records, args.output)


if __name__ == "__main__":
    main()


def _is_lz4(path: Path) -> bool:
    return any(suffix in path.name for suffix in (".jsonl.lz4", ".json.lz4"))
