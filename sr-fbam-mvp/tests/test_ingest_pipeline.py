import io
import json
from pathlib import Path

import lz4.frame

from pkmn_battle.ingest import (
    BattleDecisionDataset,
    ConversionConfig,
    convert_shard,
    discover_shards,
)
from scripts.download_metamon_replays import write_demo_shard


def test_discover_shards(tmp_path: Path) -> None:
    shard = tmp_path / "gen9ou_high_elo.jsonl.lz4"
    shard.write_text("")

    discovered = list(discover_shards(tmp_path))
    assert discovered == [shard]


def test_convert_decision_shard(tmp_path: Path) -> None:
    shard = write_demo_shard(tmp_path)
    buf = io.StringIO()
    config = ConversionConfig(mechanics_dir=tmp_path, format_filter="gen9ou")

    written = convert_shard(shard, config, buf)

    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert written == 2
    assert len(lines) == 2
    assert lines[0]["battle_id"] == "gen9ou-demo-0001"
    assert lines[0]["turn_idx"] == 1
    assert lines[0]["options"]["moves"][1]["id"] == "icebeam"
    assert lines[1]["turn_idx"] == 2
    assert lines[1]["action_label"]["id"] == "darkpulse"


def test_convert_universal_state_shard(tmp_path: Path) -> None:
    shard = tmp_path / "sample-battle.json.lz4"
    payload = {
        "states": [
            {
                "format": "gen9ou",
                "player_active_pokemon": _pokemon(
                    name="greninja",
                    hp_pct=1.0,
                    moves=[
                        _move("hydropump", base_power=110, move_type="water"),
                        _move("icebeam", base_power=90, move_type="ice"),
                        _move("darkpulse", base_power=80, move_type="dark"),
                        _move("uturn", base_power=70, move_type="bug"),
                    ],
                ),
                "opponent_active_pokemon": _pokemon(
                    name="landorustherian",
                    hp_pct=1.0,
                    item="rockyhelmet",
                    ability="intimidate",
                ),
                "available_switches": [
                    _pokemon("dragonite", hp_pct=0.8, item="lum", ability="multiscale")
                ],
                "player_prev_move": {"name": "nomove"},
                "opponent_prev_move": {"name": "nomove"},
                "player_conditions": "noconditions",
                "opponent_conditions": "stealthrock",
                "weather": "sandstorm",
                "battle_field": "nffield",
                "forced_switch": False,
                "battle_won": False,
                "battle_lost": False,
                "can_tera": True,
                "opponents_remaining": 6,
            },
            {
                "format": "gen9ou",
                "player_active_pokemon": _pokemon(
                    name="greninja",
                    hp_pct=0.8,
                    moves=[
                        _move("hydropump", base_power=110, move_type="water", current_pp=4),
                        _move("icebeam", base_power=90, move_type="ice"),
                        _move("darkpulse", base_power=80, move_type="dark"),
                        _move("uturn", base_power=70, move_type="bug"),
                    ],
                ),
                "opponent_active_pokemon": _pokemon(
                    name="landorustherian",
                    hp_pct=0.55,
                    item="rockyhelmet",
                    ability="intimidate",
                ),
                "available_switches": [
                    _pokemon("dragonite", hp_pct=0.8, item="lum", ability="multiscale")
                ],
                "player_prev_move": {"name": "hydropump"},
                "opponent_prev_move": {"name": "earthquake"},
                "player_conditions": "noconditions",
                "opponent_conditions": "stealthrock",
                "weather": "sandstorm",
                "battle_field": "nffield",
                "forced_switch": True,
                "battle_won": False,
                "battle_lost": False,
                "can_tera": True,
                "opponents_remaining": 6,
            },
            {
                "format": "gen9ou",
                "player_active_pokemon": _pokemon(
                    name="dragonite",
                    hp_pct=0.8,
                    moves=[
                        _move("dragondance", base_power=0, move_type="dragon"),
                        _move("dualwingbeat", base_power=40, move_type="flying"),
                        _move("earthquake", base_power=100, move_type="ground"),
                        _move("firepunch", base_power=75, move_type="fire"),
                    ],
                ),
                "opponent_active_pokemon": _pokemon(
                    name="landorustherian",
                    hp_pct=0.55,
                    item="rockyhelmet",
                    ability="intimidate",
                ),
                "available_switches": [
                    _pokemon("greninja", hp_pct=0.2, item="lifeorb", ability="protean")
                ],
                "player_prev_move": {"name": "nomove"},
                "opponent_prev_move": {"name": "earthquake"},
                "player_conditions": "noconditions",
                "opponent_conditions": "stealthrock",
                "weather": "sandstorm",
                "battle_field": "nffield",
                "forced_switch": False,
                "battle_won": False,
                "battle_lost": False,
                "can_tera": True,
                "opponents_remaining": 6,
            },
        ],
        "actions": [0, 4, -1],
    }
    with lz4.frame.open(shard, "wb") as handle:
        handle.write(json.dumps(payload).encode("utf-8"))

    buf = io.StringIO()
    config = ConversionConfig(mechanics_dir=tmp_path, format_filter="gen9ou")
    written = convert_shard(shard, config, buf)

    records = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert written == 3
    assert records[0]["action_label"] == {
        "type": "MOVE",
        "id": "hydropump",
        "target": "foe-active",
        "tera": False,
    }
    assert records[1]["action_label"]["type"] == "SWITCH"
    assert records[1]["action_label"]["species"] == "dragonite"
    assert records[2]["action_label"]["type"] == "NONE"
    assert any(
        update["path"] == "opponent_active.hp_pct"
        and abs(update["value"] - 0.55) < 1e-6
        for update in records[0]["graph_updates"]
    )
    assert records[0]["frame"]["grid_40x120"]
    assert len(records[0]["frame"]["grid_40x120"]) == 40

    # Write JSONL for dataset loader validation
    jsonl_path = tmp_path / "battle.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record))
            fout.write("\n")

    dataset = BattleDecisionDataset(jsonl_path)
    assert len(dataset) == 3
    assert dataset.num_actions >= 2
    frames, label = dataset[0]
    assert frames.shape == (40, 120)
    assert frames.max() <= 1.0
    assert isinstance(label, int)


def _move(
    name: str,
    *,
    base_power: int,
    move_type: str,
    category: str = "special",
    current_pp: int = 8,
    max_pp: int = 8,
) -> dict:
    return {
        "name": name,
        "move_type": move_type,
        "category": category,
        "base_power": base_power,
        "accuracy": 1.0,
        "priority": 0,
        "current_pp": current_pp,
        "max_pp": max_pp,
    }


def _pokemon(
    name: str,
    *,
    hp_pct: float,
    item: str | None = "lifeorb",
    ability: str | None = "protean",
    status: str = "nostatus",
    types: str = "water dark",
    moves: list[dict] | None = None,
) -> dict:
    return {
        "name": name,
        "hp_pct": hp_pct,
        "types": types,
        "item": item,
        "ability": ability,
        "status": status,
        "effect": "noeffect",
        "moves": moves or [],
        "atk_boost": 0,
        "spa_boost": 0,
        "def_boost": 0,
        "spd_boost": 0,
        "spe_boost": 0,
        "accuracy_boost": 0,
        "evasion_boost": 0,
        "base_atk": 95,
        "base_spa": 95,
        "base_def": 95,
        "base_spd": 95,
        "base_spe": 95,
        "base_hp": 95,
        "tera_type": "water",
        "base_species": name,
    }
