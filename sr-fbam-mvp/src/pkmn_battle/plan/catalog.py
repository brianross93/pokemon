"""
Lookup helpers for battle plan execution (moves, species, etc.).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _load_moves() -> Dict[str, int]:
    root = Path(__file__).resolve().parents[3]
    moves_path = root / "data" / "mechanics" / "moves.json"
    if not moves_path.exists():
        return {}
    with moves_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    lookup: Dict[str, int] = {}
    for key, data in raw.items():
        num = data.get("num")
        if not isinstance(num, int):
            continue
        for candidate in (key, data.get("name")):
            if not candidate:
                continue
            normalized = _normalize(candidate)
            if normalized and normalized not in lookup:
                lookup[normalized] = num
    return lookup


@lru_cache(maxsize=1)
def _load_species() -> Dict[str, int]:
    root = Path(__file__).resolve().parents[3]
    pokedex_path = root / "data" / "mechanics" / "pokedex.json"
    if not pokedex_path.exists():
        return {}
    with pokedex_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    lookup: Dict[str, int] = {}
    for key, data in raw.items():
        num = data.get("num")
        name = data.get("name")
        if not isinstance(num, int) or not isinstance(name, str):
            continue
        for candidate in (key, name):
            normalized = _normalize(candidate)
            if normalized and normalized not in lookup:
                lookup[normalized] = num
    return lookup


def move_id_from_name(name: str) -> Optional[int]:
    """Return the numeric move identifier for ``name`` if known."""

    if not name:
        return None
    normalized = _normalize(name)
    return _load_moves().get(normalized)


def species_id_from_name(name: str) -> Optional[int]:
    """Return the numeric species identifier for ``name`` if known."""

    if not name:
        return None
    normalized = _normalize(name)
    return _load_species().get(normalized)


__all__ = ["move_id_from_name", "species_id_from_name"]
