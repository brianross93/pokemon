"""
Download static mechanics tables from the official Pokémon Showdown data dump.

This script mirrors the JSON/JS files used by the simulator so we can pre-seed
the SR-FBAM entity graph (species, moves, items, abilities, types, formats).

Example:
    python scripts/download_mechanics_data.py --dest data/mechanics
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


DEFAULT_FILES = [
    "pokedex.json",
    "moves.json",
    "abilities.js",
    "items.js",
    "typechart.js",
    "formats.js",
    "formats-data.js",
]

BASE_URL = "https://play.pokemonshowdown.com/data"


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    overwrite: bool = False,
) -> None:
    if destination.exists() and not overwrite:
        logging.info("Skipping %s (already exists)", destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        bar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        )
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 15):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))
        bar.close()


def resolve_targets(files: Iterable[str], dest_dir: Path) -> list[tuple[str, Path]]:
    targets = []
    for file_name in files:
        if not file_name:
            continue
        cleaned = file_name.strip()
        url = f"{BASE_URL}/{cleaned}"
        destination = dest_dir / cleaned
        targets.append((url, destination))
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror Pokémon Showdown static mechanics files locally.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/mechanics"),
        help="Directory to store the downloaded files.",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=DEFAULT_FILES,
        help="File names relative to the Showdown data root to download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if the destination already exists.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional JSON manifest listing additional files to download. "
            "Schema: {\"files\": [\"pokedex.json\", ...]}"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Override the upstream base URL (useful for mirrors or pinning commits).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    files = list(args.files)
    if args.manifest:
        manifest_data = json.loads(args.manifest.read_text())
        files.extend(manifest_data.get("files", []))
    dest_dir: Path = args.dest
    base_url: str = args.base_url.rstrip("/")

    targets = resolve_targets(files, dest_dir)
    if not targets:
        logging.info("No files specified – exiting.")
        return

    with requests.Session() as session:
        for url, destination in targets:
            url = url.replace(BASE_URL, base_url, 1)
            download_file(session, url, destination, overwrite=args.overwrite)

    logging.info("Downloaded %d mechanics files into %s", len(targets), dest_dir)


if __name__ == "__main__":
    main()
