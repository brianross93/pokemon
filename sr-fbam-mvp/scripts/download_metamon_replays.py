"""
Download parsed Pokémon Showdown replays from the Metamon dataset on Hugging Face.

The dataset publishes each format as a large `*.tar.gz` archive containing
thousands of `.jsonl.lz4` shards. This helper streams the archive and only
extracts the first N shards so that local experimentation can start without
pulling the full multi-gigabyte payload.

Example:
    python scripts/download_metamon_replays.py --format gen9ou --max-files 3
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import tarfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from tqdm import tqdm
import lz4.frame


DATASET_ID = "jakegrigsby/metamon-parsed-replays"
DEFAULT_TIMEOUT = 600  # seconds


def stream_extract(
    url: str,
    dest_dir: Path,
    max_files: int | None,
    overwrite: bool,
    timeout: float | None,
) -> int:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        reader = io.BufferedReader(response.raw)

        extracted = 0
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                if max_files is not None and extracted >= max_files:
                    break
                if not member.isfile() or not member.name.endswith(".jsonl.lz4"):
                    continue

                target_path = dest_dir / Path(member.name).name
                if target_path.exists() and not overwrite:
                    logging.info("Skipping existing shard: %s", target_path)
                    continue

                dest_dir.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as member_file:
                    if member_file is None:
                        continue
                    with target_path.open("wb") as handle:
                        shutil.copyfileobj(member_file, handle)
                extracted += 1

    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Metamon replay shards by streaming the archive.",
    )
    parser.add_argument(
        "--repo-id",
        default=DATASET_ID,
        help="Hugging Face dataset repository id.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision to pin (tag/branch/commit).",
    )
    parser.add_argument(
        "--format",
        default="gen9ou",
        help="Showdown format (determines the archive name <format>.tar.gz).",
    )
    parser.add_argument(
        "--filename",
        help="Override the archive filename if it does not match <format>.tar.gz.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw/metamon"),
        help="Destination directory for extracted shards.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Number of shards to extract from the archive (None for all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite shards that already exist in the destination.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a small synthetic shard instead of downloading the archive.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout (seconds) for the streaming request; set to a large number for big archives.",
    )
    return parser.parse_args()


def write_demo_shard(dest: Path) -> Path:
    sample = {
        "battle_id": "gen9ou-demo-0001",
        "format": "gen9ou",
        "p1_elo": 1810,
        "p2_elo": 1765,
        "decisions": [
            {
                "turn_idx": 1,
                "frame": {"grid_40x120": ["Row 01", "Row 02"]},
                "options": {
                    "moves": [
                        {"id": "hydropump", "target": "foe-active"},
                        {"id": "icebeam", "target": "foe-active"},
                    ],
                    "switches": [{"species": "Dondozo"}],
                    "tera_available": True,
                },
                "action_label": {
                    "type": "MOVE",
                    "id": "icebeam",
                    "target": "foe-active",
                    "tera": False,
                },
                "graph_updates": [
                    {"op": "WRITE", "add_node": {"type": "field", "id": "snow"}}
                ],
                "revealed": {
                    "p1_active": {"species": "Greninja", "hp_pct": 63, "status": None},
                    "p2_active": {
                        "species": "Landorus-Therian",
                        "hp_pct": 41,
                        "status": "brn",
                    },
                },
                "log_events": [
                    "|move|p1a: Greninja|Ice Beam|p2a: Landorus-Therian",
                    "|-status|p2a: Landorus-Therian|brn",
                ],
            },
            {
                "turn_idx": 2,
                "frame": {"grid_40x120": ["Row 01", "Row 02"]},
                "options": {
                    "moves": [
                        {"id": "icebeam", "target": "foe-active"},
                        {"id": "darkpulse", "target": "foe-active"},
                    ],
                    "switches": [{"species": "Glowking"}],
                    "tera_available": False,
                },
                "action_label": {
                    "type": "MOVE",
                    "id": "darkpulse",
                    "target": "foe-active",
                    "tera": False,
                },
                "graph_updates": [],
                "revealed": {
                    "p1_active": {"species": "Greninja", "hp_pct": 41, "status": None},
                    "p2_active": {
                        "species": "Landorus-Therian",
                        "hp_pct": 12,
                        "status": "brn",
                    },
                },
                "log_events": [
                    "|move|p1a: Greninja|Dark Pulse|p2a: Landorus-Therian",
                    "|-damage|p2a: Landorus-Therian|0 fnt",
                ],
            },
        ],
    }
    dest.mkdir(parents=True, exist_ok=True)
    shard_path = dest / "gen9ou_demo_0001.jsonl.lz4"
    with lz4.frame.open(shard_path, "wb") as handle:
        handle.write(json.dumps(sample).encode("utf-8"))
    return shard_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.demo:
        shard_path = write_demo_shard(args.dest)
        logging.info("Wrote demo shard to %s", shard_path)
        return

    archive_name = args.filename or f"{args.format}.tar.gz"
    url = hf_hub_url(
        repo_id=args.repo_id,
        filename=archive_name,
        repo_type="dataset",
        revision=args.revision,
    )

    logging.info(
        "Streaming %s from %s (max_files=%s)",
        archive_name,
        args.repo_id,
        args.max_files,
    )
    progress = tqdm(total=args.max_files, unit="file", disable=args.max_files is None)
    extracted = stream_extract(
        url,
        args.dest,
        args.max_files,
        args.overwrite,
        timeout=args.timeout,
    )
    progress.update(extracted)
    progress.close()

    if extracted == 0:
        logging.warning("No shards were extracted. Check the archive name or flags.")
    else:
        logging.info("Extracted %d shard(s) into %s", extracted, args.dest)


if __name__ == "__main__":
    main()
