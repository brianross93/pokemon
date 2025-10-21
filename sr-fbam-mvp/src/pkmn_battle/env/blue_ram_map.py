"""
Canonical RAM map for Pokemon Blue battle observations.

The offsets are sourced from the community reverse-engineering effort
around Red/Blue (see the `pokered` disassembly). They target the English
Pokemon Blue v1.0 ROM and may need adjustments for other regional builds.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .blue_ram_adapter import (
    BlueBattleRAMMap,
    FieldSpec,
    MoveSlotSpec,
    PokemonSlotSpec,
    PartySpec,
)


# --------------------------------------------------------------------------- #
# Default offsets (English Pokemon Blue v1.0)
# --------------------------------------------------------------------------- #

DEFAULT_MY_ACTIVE = PokemonSlotSpec(
    species=0xD163,  # wBattleMonSpecies
    level=0xD16D,  # wBattleMonLevel
    current_hp=(0xD166, 0xD165),  # hi, lo (wBattleMonHP)
    max_hp=(0xD168, 0xD167),  # hi, lo (wBattleMonMaxHP)
    status=0xD16A,  # wBattleMonStatus
    moves=(
        MoveSlotSpec(move_id=0xD16F, pp=0xD173),
        MoveSlotSpec(move_id=0xD170, pp=0xD174),
        MoveSlotSpec(move_id=0xD171, pp=0xD175),
        MoveSlotSpec(move_id=0xD172, pp=0xD176),
    ),
)

DEFAULT_OPP_ACTIVE = PokemonSlotSpec(
    species=0xD0CE,  # wEnemyMonSpecies
    level=0xD0D2,  # wEnemyMonLevel
    current_hp=(0xD0F3, 0xD0F2),  # hi, lo (wEnemyMonHP)
    max_hp=(0xD0F5, 0xD0F4),  # hi, lo (wEnemyMonMaxHP)
    status=0xD0D1,  # wEnemyMonStatus
    moves=(
        MoveSlotSpec(move_id=0xD0D8, pp=0xD0DC),
        MoveSlotSpec(move_id=0xD0D9, pp=0xD0DD),
        MoveSlotSpec(move_id=0xD0DA, pp=0xD0DE),
        MoveSlotSpec(move_id=0xD0DB, pp=0xD0DF),
    ),
)

DEFAULT_BLUE_BATTLE_RAM_MAP = BlueBattleRAMMap(
    turn_counter=0xD05A,  # wBattleTurnNumber
    my_active=DEFAULT_MY_ACTIVE,
    opp_active=DEFAULT_OPP_ACTIVE,
    my_party=None,
    opp_party=None,
    field=None,
)


def ram_map_from_dict(config: Mapping[str, object]) -> BlueBattleRAMMap:
    """
    Build a :class:`BlueBattleRAMMap` from a nested dictionary.

    The schema mirrors the dataclass structure and is intended for YAML
    or JSON overrides. Missing keys fall back to the defaults above.
    """

    def _move(spec: Mapping[str, object], default: MoveSlotSpec) -> MoveSlotSpec:
        move_id = int(spec.get("move_id", default.move_id))
        pp = int(spec.get("pp", default.pp))
        max_pp = spec.get("max_pp", default.max_pp)
        disabled_flag = spec.get("disabled_flag", default.disabled_flag)
        return MoveSlotSpec(
            move_id=move_id,
            pp=pp,
            max_pp=int(max_pp) if max_pp is not None else None,
            disabled_flag=int(disabled_flag) if disabled_flag is not None else None,
        )

    def _parse_moves(moves_cfg: Sequence[object] | None, defaults: tuple[MoveSlotSpec, ...]) -> tuple[MoveSlotSpec, ...]:
        if moves_cfg is None:
            return defaults
        moves: list[MoveSlotSpec] = []
        for idx, item in enumerate(moves_cfg):
            if not isinstance(item, Mapping):
                continue
            default = defaults[idx] if idx < len(defaults) else defaults[-1]
            moves.append(_move(item, default))
        # pad/truncate to four slots
        while len(moves) < len(defaults):
            moves.append(defaults[len(moves)])
        return tuple(moves[: len(defaults)])

    def _opt_int(spec: Mapping[str, object], key: str, fallback: int | None) -> int | None:
        value = spec.get(key)
        if value is None:
            return fallback
        return int(value)

    def _opt_pair(spec: Mapping[str, object], key: str, fallback: tuple[int, int] | None) -> tuple[int, int] | None:
        value = spec.get(key)
        if value is None:
            return fallback
        hi = int(value[0])
        if len(value) > 1 and value[1] is not None:
            lo = int(value[1])
        else:
            lo = hi + 1
        return (hi, lo)

    def _pokemon(spec: Mapping[str, object], default: PokemonSlotSpec) -> PokemonSlotSpec:
        moves_cfg = spec.get("moves")
        moves = _parse_moves(moves_cfg if isinstance(moves_cfg, Sequence) else None, default.moves)
        return PokemonSlotSpec(
            species=_opt_int(spec, "species", default.species),
            level=_opt_int(spec, "level", default.level),
            current_hp=_opt_pair(spec, "current_hp", default.current_hp),
            max_hp=_opt_pair(spec, "max_hp", default.max_hp),
            status=_opt_int(spec, "status", default.status),
            type1=_opt_int(spec, "type1", default.type1),
            type2=_opt_int(spec, "type2", default.type2),
            stat_stages=_opt_int(spec, "stat_stages", default.stat_stages),
            moves=moves,
        )

    my_active_cfg = config.get("my_active")
    my_active = _pokemon(my_active_cfg, DEFAULT_MY_ACTIVE) if isinstance(my_active_cfg, Mapping) else DEFAULT_MY_ACTIVE

    opp_active_cfg = config.get("opp_active")
    opp_active = _pokemon(opp_active_cfg, DEFAULT_OPP_ACTIVE) if isinstance(opp_active_cfg, Mapping) else DEFAULT_OPP_ACTIVE

    turn_counter = (
        int(config["turn_counter"])
        if config.get("turn_counter") is not None
        else DEFAULT_BLUE_BATTLE_RAM_MAP.turn_counter
    )

    my_party_cfg = config.get("my_party")
    opp_party_cfg = config.get("opp_party")
    field_cfg = config.get("field")

    def _party(spec: object) -> Optional[PartySpec]:
        if not isinstance(spec, Mapping):
            return None
        count = int(spec.get("count", 0))
        slots_cfg = spec.get("slots")
        if not isinstance(slots_cfg, Sequence):
            return None
        slots = []
        for idx in range(min(count, len(slots_cfg))):
            slot_cfg = slots_cfg[idx]
            if not isinstance(slot_cfg, Mapping):
                continue
            slots.append(_pokemon(slot_cfg, PokemonSlotSpec()))
        return PartySpec(count=len(slots), slots=tuple(slots))

    def _field(spec: object) -> Optional[FieldSpec]:
        if not isinstance(spec, Mapping):
            return None
        return FieldSpec(
            weather=int(spec["weather"]) if spec.get("weather") is not None else None,
            terrain=int(spec["terrain"]) if spec.get("terrain") is not None else None,
            player_screens=int(spec["player_screens"]) if spec.get("player_screens") is not None else None,
            enemy_screens=int(spec["enemy_screens"]) if spec.get("enemy_screens") is not None else None,
            player_side_effects=int(spec["player_side_effects"]) if spec.get("player_side_effects") is not None else None,
            enemy_side_effects=int(spec["enemy_side_effects"]) if spec.get("enemy_side_effects") is not None else None,
        )

    return BlueBattleRAMMap(
        turn_counter=turn_counter,
        my_active=my_active,
        opp_active=opp_active,
        my_party=_party(my_party_cfg),
        opp_party=_party(opp_party_cfg),
        field=_field(field_cfg),
    )
