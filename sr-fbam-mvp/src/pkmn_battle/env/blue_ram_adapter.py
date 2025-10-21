"""
RAM-backed environment adapter for Pokemon Blue battles.

The adapter reads memory snapshots from a callable ``read_u8`` helper
and projects them into the task-agnostic :class:`BattleObs` structure.
It intentionally errs on the side of being forgiving: if a configured
address is missing the adapter fills in sensible defaults so upstream
modules can still progress while instrumentation is refined.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

from .core import BattleObs, EnvAdapter, LegalAction


# --------------------------------------------------------------------------- #
# Address configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MoveSlotSpec:
    """Offsets describing a single move slot in RAM."""

    move_id: int
    pp: int
    max_pp: Optional[int] = None
    disabled_flag: Optional[int] = None


@dataclass(frozen=True)
class PokemonSlotSpec:
    """
    Offsets describing the state of a Pokemon slot (active or party).

    Multi-byte values such as HP are represented as ``(hi, lo)`` tuples.
    When the tuple's second item is ``None`` the adapter assumes the next
    byte in memory stores the low component.
    """

    species: Optional[int] = None
    level: Optional[int] = None
    current_hp: Optional[Tuple[int, Optional[int]]] = None
    max_hp: Optional[Tuple[int, Optional[int]]] = None
    status: Optional[int] = None
    type1: Optional[int] = None
    type2: Optional[int] = None
    stat_stages: Optional[int] = None
    moves: Tuple[MoveSlotSpec, ...] = tuple()


@dataclass(frozen=True)
class PartySpec:
    """
    Offsets describing a Pokemon party table.

    Each entry is assumed to follow the same layout as ``slot_spec``.
    """

    count: int
    slots: Tuple[PokemonSlotSpec, ...]


@dataclass(frozen=True)
class FieldSpec:
    """Offsets describing shared battle field effects."""

    weather: Optional[int] = None
    terrain: Optional[int] = None
    player_screens: Optional[int] = None
    enemy_screens: Optional[int] = None
    player_side_effects: Optional[int] = None
    enemy_side_effects: Optional[int] = None


@dataclass(frozen=True)
class BlueBattleRAMMap:
    """Collects all offsets needed to decode a battle snapshot."""

    turn_counter: Optional[int] = None
    my_active: PokemonSlotSpec = PokemonSlotSpec()
    opp_active: PokemonSlotSpec = PokemonSlotSpec()
    my_party: Optional[PartySpec] = None
    opp_party: Optional[PartySpec] = None
    field: Optional[FieldSpec] = None


# --------------------------------------------------------------------------- #
# Adapter implementation
# --------------------------------------------------------------------------- #


def _default_read_u8(memory: Mapping[int, int], address: int) -> int:
    value = memory.get(address)
    if value is None:
        raise KeyError(f"Address 0x{address:04X} missing from memory mapping.")
    return int(value) & 0xFF


class BlueRAMAdapter(EnvAdapter):
    """
    Environment adapter that sources battle state from raw RAM reads.

    Parameters
    ----------
    read_u8:
        Callable that returns the byte stored at ``address``. The adapter
        never mutates RAM; any state transitions should be driven via the
        optional ``action_executor`` callback.
    ram_map:
        Static description of where relevant battle fields live in RAM.
    snapshot_extra:
        Optional callable returning auxiliary metadata merged into the
        ``raw`` section of :class:`BattleObs`.
    action_executor:
        Optional callback that receives the chosen :class:`LegalAction`
        when :meth:`step` is invoked. It is responsible for applying the
        action to the underlying emulator or replay driver.
    reset_callback:
        Optional callable invoked before taking the first snapshot in
        :meth:`reset`.
    """

    def __init__(
        self,
        *,
        read_u8: Callable[[int], int],
        ram_map: BlueBattleRAMMap,
        snapshot_extra: Optional[Callable[[], Mapping[str, object]]] = None,
        action_executor: Optional[Callable[[LegalAction], None]] = None,
        reset_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self._read_u8 = read_u8
        self._map = ram_map
        self._snapshot_extra = snapshot_extra
        self._action_executor = action_executor
        self._reset_callback = reset_callback
        self._last_obs: Optional[BattleObs] = None

    # ------------------------------------------------------------------ #
    # EnvAdapter API
    # ------------------------------------------------------------------ #

    def reset(self) -> BattleObs:
        if self._reset_callback is not None:
            self._reset_callback()
        self._last_obs = self._build_observation()
        return self._last_obs

    def observe(self) -> BattleObs:
        self._last_obs = self._build_observation()
        return self._last_obs

    def legal_actions(self) -> List[LegalAction]:
        if self._last_obs is None:
            return []
        legal: List[LegalAction] = []

        moves = self._last_obs.get("my_active", {}).get("moves", [])
        for idx, move in enumerate(moves):
            meta = {
                "move_id": move.get("move_id"),
                "pp": move.get("pp", 0),
                "max_pp": move.get("max_pp"),
                "disabled": bool(move.get("disabled", False)),
            }
            if meta["pp"] > 0 and not meta["disabled"]:
                legal.append({"kind": "move", "index": idx, "meta": meta})  # type: ignore[arg-type]

        for idx, mon in enumerate(self._last_obs.get("my_party", [])):
            if mon.get("is_active"):
                continue
            if mon.get("fainted"):
                continue
            legal.append(  # type: ignore[arg-type]
                {
                    "kind": "switch",
                    "index": idx,
                    "meta": {
                        "species": mon.get("species"),
                        "slot": mon.get("slot"),
                    },
                }
            )

        if not legal:
            # Allow higher-level policies to explicitly forfeit or noop.
            legal.append({"kind": "forfeit", "index": 0, "meta": {}})  # type: ignore[arg-type]
        return legal

    def step(self, action: LegalAction) -> BattleObs:
        if self._action_executor is not None:
            self._action_executor(action)
        self._last_obs = self._build_observation()
        return self._last_obs

    # ------------------------------------------------------------------ #
    # Snapshot helpers
    # ------------------------------------------------------------------ #

    def _build_observation(self) -> BattleObs:
        turn = self._read_optional_u8(self._map.turn_counter) if self._map.turn_counter is not None else 0
        my_active = self._read_pokemon(self._map.my_active, side="my", slot=0)
        opp_active = self._read_pokemon(self._map.opp_active, side="opp", slot=0)

        my_party = self._read_party(self._map.my_party, side="my")
        opp_party = self._read_party(self._map.opp_party, side="opp")
        field_state = self._read_field(self._map.field)

        obs: BattleObs = {
            "turn": turn,
            "my_active": my_active,
            "opp_active": opp_active,
            "my_party": my_party,
            "opp_party": opp_party,
            "field": field_state,
            "raw": self._raw_snapshot(),
        }
        return obs

    def _read_party(self, spec: Optional[PartySpec], *, side: str) -> List[Dict[str, object]]:
        if spec is None:
            return []
        party: List[Dict[str, object]] = []
        for idx, slot_spec in enumerate(spec.slots):
            mon = self._read_pokemon(slot_spec, side=side, slot=idx)
            mon["is_active"] = False
            mon["slot"] = idx
            party.append(mon)
        return party

    def _read_pokemon(self, spec: PokemonSlotSpec, *, side: str, slot: int) -> Dict[str, object]:
        attrs: MutableMapping[str, object] = {"side": side, "slot": slot}

        if spec.species is not None:
            attrs["species"] = self._read_optional_u8(spec.species)
        if spec.level is not None:
            attrs["level"] = self._read_optional_u8(spec.level)
        if spec.current_hp is not None:
            attrs["hp"] = self._read_u16(spec.current_hp)
        if spec.max_hp is not None:
            attrs["hp_max"] = self._read_u16(spec.max_hp)
        if spec.status is not None:
            attrs["status"] = self._read_optional_u8(spec.status)
        if spec.type1 is not None:
            attrs.setdefault("types", []).append(self._read_optional_u8(spec.type1))
        if spec.type2 is not None:
            attrs.setdefault("types", []).append(self._read_optional_u8(spec.type2))
        if spec.stat_stages is not None:
            attrs["stat_stages"] = self._read_optional_u8(spec.stat_stages)

        moves: List[Dict[str, object]] = []
        for idx, move_spec in enumerate(spec.moves):
            move_id = self._read_optional_u8(move_spec.move_id)
            pp = self._read_optional_u8(move_spec.pp)
            max_pp = self._read_optional_u8(move_spec.max_pp) if move_spec.max_pp is not None else None
            disabled = bool(self._read_optional_u8(move_spec.disabled_flag)) if move_spec.disabled_flag else False
            moves.append(
                {
                    "slot": idx,
                    "move_id": move_id,
                    "pp": pp,
                    "max_pp": max_pp,
                    "disabled": disabled,
                }
            )
        attrs["moves"] = moves
        attrs["fainted"] = attrs.get("hp", 1) == 0
        attrs["is_active"] = slot == 0 and side in {"my", "opp"}
        return dict(attrs)

    def _read_u16(self, addresses: Tuple[int, Optional[int]]) -> int:
        hi_addr, lo_addr = addresses
        hi = self._read_optional_u8(hi_addr)
        lo_address = lo_addr if lo_addr is not None else hi_addr + 1
        lo = self._read_optional_u8(lo_address)
        return (hi << 8) | lo

    def _read_optional_u8(self, address: Optional[int]) -> int:
        if address is None:
            return 0
        try:
            return self._read_u8(address)
        except KeyError:
            return 0

    def _read_field(self, spec: Optional[FieldSpec]) -> Dict[str, object]:
        if spec is None:
            return {}
        field: Dict[str, object] = {}
        if spec.weather is not None:
            field["weather_id"] = self._read_optional_u8(spec.weather)
        if spec.terrain is not None:
            field["terrain_id"] = self._read_optional_u8(spec.terrain)
        if spec.player_screens is not None:
            field["player_screens"] = self._read_optional_u8(spec.player_screens)
        if spec.enemy_screens is not None:
            field["enemy_screens"] = self._read_optional_u8(spec.enemy_screens)
        if spec.player_side_effects is not None:
            field["player_side_effects"] = self._read_optional_u8(spec.player_side_effects)
        if spec.enemy_side_effects is not None:
            field["enemy_side_effects"] = self._read_optional_u8(spec.enemy_side_effects)
        return field

    def _raw_snapshot(self) -> Dict[str, object]:
        extra = dict(self._snapshot_extra() or {}) if self._snapshot_extra is not None else {}
        extra.setdefault("addresses", {})
        if isinstance(extra["addresses"], MutableMapping):
            addr_map: MutableMapping[str, object] = extra["addresses"]
        else:
            addr_map = {}
            extra["addresses"] = addr_map

        def record(label: str, value: Optional[int]) -> None:
            if value is None:
                return
            addr_map[label] = f"0x{value:04X}"

        record("turn_counter", self._map.turn_counter)

        def record_slot(prefix: str, slot_spec: PokemonSlotSpec) -> None:
            record(f"{prefix}.species", slot_spec.species)
            record(f"{prefix}.level", slot_spec.level)
            if slot_spec.current_hp is not None:
                record(f"{prefix}.hp_hi", slot_spec.current_hp[0])
                record(f"{prefix}.hp_lo", slot_spec.current_hp[1] if slot_spec.current_hp[1] is not None else slot_spec.current_hp[0] + 1)
            if slot_spec.max_hp is not None:
                record(f"{prefix}.hp_max_hi", slot_spec.max_hp[0])
                record(f"{prefix}.hp_max_lo", slot_spec.max_hp[1] if slot_spec.max_hp[1] is not None else slot_spec.max_hp[0] + 1)
            if slot_spec.status is not None:
                record(f"{prefix}.status", slot_spec.status)
            for idx, move_spec in enumerate(slot_spec.moves):
                record(f"{prefix}.move{idx}.id", move_spec.move_id)
                record(f"{prefix}.move{idx}.pp", move_spec.pp)
                if move_spec.max_pp is not None:
                    record(f"{prefix}.move{idx}.max_pp", move_spec.max_pp)
                if move_spec.disabled_flag is not None:
                    record(f"{prefix}.move{idx}.disabled", move_spec.disabled_flag)

        record_slot("my_active", self._map.my_active)
        record_slot("opp_active", self._map.opp_active)
        if self._map.field is not None:
            record("field.weather", self._map.field.weather)
            record("field.terrain", self._map.field.terrain)
            record("field.player_screens", self._map.field.player_screens)
            record("field.enemy_screens", self._map.field.enemy_screens)
            record("field.player_side_effects", self._map.field.player_side_effects)
            record("field.enemy_side_effects", self._map.field.enemy_side_effects)
        return extra


def build_blue_ram_adapter_from_mapping(
    *,
    memory: Mapping[int, int],
    ram_map: BlueBattleRAMMap,
    snapshot_extra: Optional[Callable[[], Mapping[str, object]]] = None,
    action_executor: Optional[Callable[[LegalAction], None]] = None,
    reset_callback: Optional[Callable[[], None]] = None,
) -> BlueRAMAdapter:
    """
    Convenience helper for tests where RAM snapshots are preloaded as
    dictionaries rather than read from a live emulator.
    """

    return BlueRAMAdapter(
        read_u8=lambda addr: _default_read_u8(memory, addr),
        ram_map=ram_map,
        snapshot_extra=snapshot_extra,
        action_executor=action_executor,
        reset_callback=reset_callback,
    )
