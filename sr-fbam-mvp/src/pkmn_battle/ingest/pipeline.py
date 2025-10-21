from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO

import lz4.frame


LZ4_SUFFIXES: Sequence[str] = (".jsonl.lz4", ".json.lz4")


@dataclass
class ConversionConfig:
    """Context shared across shard conversions."""

    mechanics_dir: Path
    format_filter: str | None = "gen9ou"
    max_turns: int | None = None


def discover_shards(source: Path) -> Iterator[Path]:
    """
    Yield `.lz4` shard paths under `source`.

    `source` may be a directory, a single file, or a glob pattern.
    """
    if source.is_file():
        if _is_lz4_path(source):
            yield source
        return

    if "*" in source.name or "?" in source.name or "[" in source.name:
        for path in Path().glob(str(source)):
            if path.is_file() and _is_lz4_path(path):
                yield path
        return

    if source.is_dir():
        yield from sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and _is_lz4_path(path)
        )


def convert_shard(
    shard_path: Path,
    config: ConversionConfig,
    writer: TextIO,
) -> int:
    """Convert a single Metamon shard into SR-FBAM JSONL lines."""
    with lz4.frame.open(shard_path, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))

    format_name = (payload.get("format") or "").lower()
    states = payload.get("states") or []
    if not format_name and states:
        format_name = (states[0].get("format") or "").lower()

    if config.format_filter and format_name and format_name != config.format_filter:
        logging.info(
            "Skipping shard %s (format %s != %s)",
            shard_path,
            format_name,
            config.format_filter,
        )
        return 0

    if "decisions" in payload:
        return _convert_decisions(payload, config, writer)

    if "states" in payload and "actions" in payload:
        return _convert_universal_states(
            shard_path,
            payload,
            format_name=format_name or config.format_filter or "",
            config=config,
            writer=writer,
        )

    raise ValueError(f"Unrecognised shard schema: {shard_path}")


def _convert_decisions(
    payload: dict,
    config: ConversionConfig,
    writer: TextIO,
) -> int:
    base_record = {
        "battle_id": payload.get("battle_id"),
        "format": payload.get("format"),
        "p1_elo": payload.get("p1_elo"),
        "p2_elo": payload.get("p2_elo"),
    }

    decisions: list[dict] = payload.get("decisions", [])
    records = []
    for idx, decision in enumerate(decisions, start=1):
        if config.max_turns is not None and idx > config.max_turns:
            break

        record = dict(base_record)
        record["turn_idx"] = decision.get("turn_idx", idx)
        record["frame"] = decision.get("frame", {})
        record["options"] = decision.get("options", {})
        record["action_label"] = decision.get("action_label")
        record["graph_updates"] = decision.get("graph_updates", [])
        record["revealed"] = decision.get("revealed", {})
        record["log_events"] = decision.get("log_events", [])
        records.append(record)

    return write_jsonl(records, writer)


def _convert_universal_states(
    shard_path: Path,
    payload: dict,
    *,
    format_name: str,
    config: ConversionConfig,
    writer: TextIO,
) -> int:
    states: list[dict] = payload.get("states", [])
    actions: list[int] = payload.get("actions", [])
    if not states or not actions:
        logging.warning("Shard %s missing states/actions", shard_path)
        return 0

    battle_id = payload.get("battle_id") or shard_path.stem
    records: list[dict] = []

    for idx, (state, action_value) in enumerate(zip(states, actions)):
        if config.max_turns is not None and idx >= config.max_turns:
            break

        next_state = states[idx + 1] if idx + 1 < len(states) else state
        record = _state_to_record(
            battle_id=battle_id,
            turn_idx=idx + 1,
            format_name=format_name,
            state=state,
            next_state=next_state,
            action_value=action_value,
        )
        records.append(record)

    return write_jsonl(records, writer)


def _state_to_record(
    *,
    battle_id: str,
    turn_idx: int,
    format_name: str,
    state: dict,
    next_state: dict,
    action_value: int,
) -> dict:
    frame = _render_frame(turn_idx, format_name, state)
    options = _extract_options(state)
    action_label = _infer_action_label(state, next_state, action_value)
    graph_updates = _diff_states(state, next_state)

    return {
        "battle_id": battle_id,
        "turn_idx": turn_idx,
        "format": format_name,
        "p1_elo": state.get("player_elo"),
        "p2_elo": state.get("opponent_elo"),
        "frame": frame,
        "options": options,
        "action_label": action_label,
        "graph_updates": graph_updates,
        "revealed": {
            "player_active": _active_snapshot(state.get("player_active_pokemon", {})),
            "opponent_active": _active_snapshot(
                state.get("opponent_active_pokemon", {})
            ),
        },
        "log_events": [],
    }


def _render_frame(turn_idx: int, format_name: str, state: dict) -> dict:
    lines: list[str] = []
    player = state.get("player_active_pokemon", {})
    opponent = state.get("opponent_active_pokemon", {})
    switches = state.get("available_switches", [])

    def pad(text: str) -> str:
        trimmed = text[:120]
        return trimmed + " " * (120 - len(trimmed))

    lines.append(
        pad(
            f"Turn {turn_idx:02d} | Format {format_name.upper()} | "
            f"Forced switch={state.get('forced_switch', False)} | "
            f"Can tera={state.get('can_tera', False)}"
        )
    )
    lines.append(pad("Player Active:"))
    lines.append(
        pad(
            f"  {player.get('name','?')} HP:{_pct(player.get('hp_pct'))} "
            f"Status:{player.get('status','?')} Item:{player.get('item','?')} "
            f"Ability:{player.get('ability','?')}"
        )
    )
    lines.append(
        pad(
            f"  Types:{player.get('types','?')} Boosts:"
            f" Atk{player.get('atk_boost',0)} Spa{player.get('spa_boost',0)} "
            f"Def{player.get('def_boost',0)} SpD{player.get('spd_boost',0)} "
            f"Spe{player.get('spe_boost',0)}"
        )
    )
    lines.append(pad("  Moves:"))
    for move in player.get("moves", []):
        lines.append(
            pad(
                f"    {move.get('name','?'):16} PP {move.get('current_pp',0):>2}/{move.get('max_pp',0):<2} "
                f"{move.get('move_type','?'):>7} {move.get('category','?'):>7} BP:{move.get('base_power',0)}"
            )
        )

    lines.append(pad(""))
    lines.append(pad("Opponent Active:"))
    lines.append(
        pad(
            f"  {opponent.get('name','?')} HP:{_pct(opponent.get('hp_pct'))} "
            f"Status:{opponent.get('status','?')} Item:{opponent.get('item','?')} "
            f"Ability:{opponent.get('ability','?')}"
        )
    )
    lines.append(
        pad(
            f"  Types:{opponent.get('types','?')} Boosts:"
            f" Atk{opponent.get('atk_boost',0)} Spa{opponent.get('spa_boost',0)} "
            f"Def{opponent.get('def_boost',0)} SpD{opponent.get('spd_boost',0)} "
            f"Spe{opponent.get('spe_boost',0)}"
        )
    )

    lines.append(pad(""))
    lines.append(
        pad(
            "Field:"
            f" Weather={state.get('weather','none')} Terrain={state.get('battle_field','none')} "
            f"OppRemaining={state.get('opponents_remaining')}"
        )
    )
    lines.append(pad(f"Player conditions: {state.get('player_conditions', 'none')}"))
    lines.append(pad(f"Opponent conditions: {state.get('opponent_conditions', 'none')}"))
    lines.append(pad(""))
    lines.append(pad("Switch options:"))
    for mon in switches:
        lines.append(
            pad(
                f"  {mon.get('name','?'):16} HP:{_pct(mon.get('hp_pct'))} "
                f"Status:{mon.get('status','?')} Item:{mon.get('item','?')} Ability:{mon.get('ability','?')}"
            )
        )

    while len(lines) < 40:
        lines.append(" " * 120)
    if len(lines) > 40:
        lines = lines[:40]

    return {"grid_40x120": lines}


def _extract_options(state: dict) -> dict:
    moves = []
    for move in state.get("player_active_pokemon", {}).get("moves", []):
        moves.append(
            {
                "id": move.get("name"),
                "move_type": move.get("move_type"),
                "category": move.get("category"),
                "current_pp": move.get("current_pp"),
                "max_pp": move.get("max_pp"),
                "base_power": move.get("base_power"),
                "target": "foe-active",
            }
        )

    switches = []
    for mon in state.get("available_switches", []):
        switches.append(
            {
                "species": mon.get("name"),
                "hp_pct": mon.get("hp_pct"),
                "status": mon.get("status"),
                "item": mon.get("item"),
            }
        )

    return {
        "moves": moves,
        "switches": switches,
        "forced_switch": state.get("forced_switch", False),
        "tera_available": state.get("can_tera", False),
    }


def _infer_action_label(state: dict, next_state: dict, action_value: int) -> dict:
    player_move = (next_state.get("player_prev_move") or {}).get("name")
    if player_move and player_move != "nomove":
        tera_used = bool(state.get("can_tera", False) and not next_state.get("can_tera", False))
        return {
            "type": "MOVE",
            "id": player_move,
            "target": "foe-active",
            "tera": tera_used,
        }

    current_active = (state.get("player_active_pokemon") or {}).get("name")
    next_active = (next_state.get("player_active_pokemon") or {}).get("name")
    if next_active and next_active != current_active:
        return {
            "type": "SWITCH",
            "species": next_active,
            "from_species": current_active,
        }

    return {
        "type": "NONE",
        "raw_action": action_value,
    }


def _diff_states(state: dict, next_state: dict) -> list[dict]:
    updates: list[dict] = []

    def compare(path: str, before, after) -> None:
        if before != after:
            updates.append({"op": "WRITE", "path": path, "value": after})

    player_before = state.get("player_active_pokemon", {})
    player_after = next_state.get("player_active_pokemon", {})
    compare("player_active.hp_pct", player_before.get("hp_pct"), player_after.get("hp_pct"))
    compare("player_active.status", player_before.get("status"), player_after.get("status"))
    compare("player_active.item", player_before.get("item"), player_after.get("item"))
    compare("player_active.ability", player_before.get("ability"), player_after.get("ability"))

    opponent_before = state.get("opponent_active_pokemon", {})
    opponent_after = next_state.get("opponent_active_pokemon", {})
    compare("opponent_active.hp_pct", opponent_before.get("hp_pct"), opponent_after.get("hp_pct"))
    compare("opponent_active.status", opponent_before.get("status"), opponent_after.get("status"))
    compare("opponent_active.item", opponent_before.get("item"), opponent_after.get("item"))
    compare("opponent_active.ability", opponent_before.get("ability"), opponent_after.get("ability"))

    compare("player.conditions", state.get("player_conditions"), next_state.get("player_conditions"))
    compare("opponent.conditions", state.get("opponent_conditions"), next_state.get("opponent_conditions"))
    compare("field.weather", state.get("weather"), next_state.get("weather"))
    compare("field.terrain", state.get("battle_field"), next_state.get("battle_field"))
    compare("opponent.remaining", state.get("opponents_remaining"), next_state.get("opponents_remaining"))

    return updates


def _active_snapshot(entity: dict) -> dict:
    return {
        "species": entity.get("name"),
        "hp_pct": entity.get("hp_pct"),
        "status": entity.get("status"),
        "item": entity.get("item"),
        "ability": entity.get("ability"),
        "types": entity.get("types"),
        "boosts": {
            "atk": entity.get("atk_boost"),
            "def": entity.get("def_boost"),
            "spa": entity.get("spa_boost"),
            "spd": entity.get("spd_boost"),
            "spe": entity.get("spe_boost"),
            "accuracy": entity.get("accuracy_boost"),
            "evasion": entity.get("evasion_boost"),
        },
    }


def write_jsonl(records: Iterable[dict], writer: TextIO) -> int:
    """Append records to a JSONL writer and return how many were written."""
    count = 0
    for record in records:
        writer.write(json.dumps(record))
        writer.write("\n")
        count += 1
    return count


def _pct(value: float | None) -> str:
    if value is None:
        return "??"
    return f"{int(round(value * 100))}%"


def _is_lz4_path(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in LZ4_SUFFIXES)
