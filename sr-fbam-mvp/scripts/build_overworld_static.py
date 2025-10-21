"""
Extract overworld static entities from a local clone of the `pret/pokered` disassembly.

Usage:
    python scripts/build_overworld_static.py \
        --pokered-root ../pokered-master/pokered-master \
        --output data/overworld/static_entities.json

The script parses map headers and object definitions to produce a lightweight
JSON description of overworld locations, warps, background triggers, and NPCs.
This data seeds the SR-FBAM WorldGraph with canonical topology before any
telemetry is ingested.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


MAP_HEADER_RE = re.compile(
    r"^\s*map_header\s+(?P<label>\w+),\s*(?P<const>\w+),\s*(?P<class>\w+),\s*(?P<flags>.+?)\s*$"
)
CONNECTION_RE = re.compile(
    r"^\s*connection\s+(?P<direction>\w+),\s*(?P<target_label>\w+),\s*(?P<target_const>\w+),\s*(?P<offset>[^\s]+)\s*$"
)
WARP_RE = re.compile(
    r"^\s*warp_event\s+(?P<x>\d+),\s*(?P<y>\d+),\s*(?P<dest_map>\w+),\s*(?P<dest_warp>\d+)\s*$"
)
BG_EVENT_RE = re.compile(
    r"^\s*bg_event\s+(?P<x>\d+),\s*(?P<y>\d+),\s*(?P<script>\w+)\s*$"
)
OBJECT_EVENT_RE = re.compile(
    r"^\s*object_event\s+(?P<x>\d+),\s*(?P<y>\d+),\s*(?P<sprite>\w+),\s*(?P<motion>\w+),\s*(?P<movement>\w+),\s*(?P<script>\w+)\s*$"
)
ITEM_EVENT_RE = re.compile(
    r"^\s*item_event\s+(?P<x>\d+),\s*(?P<y>\d+),\s*(?P<item>\w+),\s*(?P<quantity>\d+)\s*$"
)
HIDDEN_ITEM_RE = re.compile(
    r"^\s*hidden_item\s+(?P<x>\d+),\s*(?P<y>\d+),\s*(?P<item>\w+),\s*(?P<flag>\w+)\s*$"
)


@dataclass
class Connection:
    direction: str
    target_label: str
    target_const: str
    offset: str


@dataclass
class Warp:
    x: int
    y: int
    destination_map: str
    destination_warp: int


@dataclass
class BackgroundEvent:
    x: int
    y: int
    script: str


@dataclass
class NPC:
    x: int
    y: int
    sprite: str
    motion: str
    movement: str
    script: str


@dataclass
class ItemSpot:
    x: int
    y: int
    item: str
    quantity: int = 1
    hidden_flag: Optional[str] = None


@dataclass
class MapStatic:
    label: str
    const: str
    map_class: str
    flags: str
    connections: List[Connection] = field(default_factory=list)
    warps: List[Warp] = field(default_factory=list)
    bg_events: List[BackgroundEvent] = field(default_factory=list)
    npcs: List[NPC] = field(default_factory=list)
    items: List[ItemSpot] = field(default_factory=list)

    def to_payload(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "constant": self.const,
            "map_class": self.map_class,
            "flags": self.flags,
            "connections": [connection.__dict__ for connection in self.connections],
            "warps": [warp.__dict__ for warp in self.warps],
            "background_events": [bg.__dict__ for bg in self.bg_events],
            "npcs": [npc.__dict__ for npc in self.npcs],
            "items": [item.__dict__ for item in self.items],
        }


def parse_map_headers(path: Path) -> Dict[str, MapStatic]:
    maps: Dict[str, MapStatic] = {}
    for header_file in sorted(path.glob("*.asm")):
        label = None
        lines = header_file.read_text(encoding="utf-8").splitlines()
        current: Optional[MapStatic] = None
        for line in lines:
            header_match = MAP_HEADER_RE.match(line)
            if header_match:
                data = header_match.groupdict()
                label = data["label"]
                current = MapStatic(
                    label=label,
                    const=data["const"],
                    map_class=data["class"],
                    flags=data["flags"],
                )
                maps[label] = current
                continue
            connection_match = CONNECTION_RE.match(line)
            if connection_match and current is not None:
                conn_data = connection_match.groupdict()
                current.connections.append(
                    Connection(
                        direction=conn_data["direction"],
                        target_label=conn_data["target_label"],
                        target_const=conn_data["target_const"],
                        offset=conn_data["offset"],
                    )
                )
        # Ensure we encountered the header; skip files without map_header (rare).
        if label is None:
            continue
    return maps


def parse_object_file(path: Path, target: MapStatic) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        warp_match = WARP_RE.match(line)
        if warp_match:
            data = warp_match.groupdict()
            target.warps.append(
                Warp(
                    x=int(data["x"]),
                    y=int(data["y"]),
                    destination_map=data["dest_map"],
                    destination_warp=int(data["dest_warp"]),
                )
            )
            continue
        obj_match = OBJECT_EVENT_RE.match(line)
        if obj_match:
            data = obj_match.groupdict()
            target.npcs.append(
                NPC(
                    x=int(data["x"]),
                    y=int(data["y"]),
                    sprite=data["sprite"],
                    motion=data["motion"],
                    movement=data["movement"],
                    script=data["script"],
                )
            )
            continue
        bg_match = BG_EVENT_RE.match(line)
        if bg_match:
            data = bg_match.groupdict()
            target.bg_events.append(
                BackgroundEvent(
                    x=int(data["x"]),
                    y=int(data["y"]),
                    script=data["script"],
                )
            )
            continue
        item_match = ITEM_EVENT_RE.match(line)
        if item_match:
            data = item_match.groupdict()
            target.items.append(
                ItemSpot(
                    x=int(data["x"]),
                    y=int(data["y"]),
                    item=data["item"],
                    quantity=int(data["quantity"]),
                )
            )
            continue
        hidden_match = HIDDEN_ITEM_RE.match(line)
        if hidden_match:
            data = hidden_match.groupdict()
            target.items.append(
                ItemSpot(
                    x=int(data["x"]),
                    y=int(data["y"]),
                    item=data["item"],
                    hidden_flag=data["flag"],
                )
            )


def enrich_with_objects(root: Path, maps: Dict[str, MapStatic]) -> None:
    for obj_file in sorted(root.glob("*.asm")):
        label = obj_file.stem
        if label not in maps:
            # Some map objects exist for special scenes; skip those without headers.
            continue
        parse_object_file(obj_file, maps[label])


def build_payload(maps: Mapping[str, MapStatic]) -> Dict[str, object]:
    return {
        "source": "pret/pokered",
        "maps": {label: map_static.to_payload() for label, map_static in maps.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract static overworld entities from pokered.")
    parser.add_argument(
        "--pokered-root",
        type=Path,
        required=True,
        help="Path to the root of the pret/pokered repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/overworld/static_entities.json"),
        help="Destination JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headers_dir = args.pokered_root / "data" / "maps" / "headers"
    objects_dir = args.pokered_root / "data" / "maps" / "objects"
    if not headers_dir.is_dir():
        raise SystemExit(f"Could not locate map headers at {headers_dir}")
    if not objects_dir.is_dir():
        raise SystemExit(f"Could not locate map objects at {objects_dir}")

    maps = parse_map_headers(headers_dir)
    enrich_with_objects(objects_dir, maps)
    payload = build_payload(maps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total_maps = len(maps)
    total_warps = sum(len(map_static.warps) for map_static in maps.values())
    total_npcs = sum(len(map_static.npcs) for map_static in maps.values())
    total_items = sum(len(map_static.items) for map_static in maps.values())
    print(
        f"Wrote {total_maps} maps, {total_warps} warps, {total_npcs} NPCs, "
        f"and {total_items} item spots to {args.output}"
    )


if __name__ == "__main__":
    main()
