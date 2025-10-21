"""
Deterministic extractor for Pokemon Blue battle snapshots.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..env.core import BattleObs
from ..graph.schema import Edge, Node, WriteOp
from .base import Extractor


class BlueExtractor(Extractor):
    """
    Convert :class:`BattleObs` structures into typed graph write operations.

    The extractor emits nodes/edges with confidence metadata so downstream
    controllers can decide when to fall back to dense encoding.
    """

    def __init__(self, *, include_party: bool = True) -> None:
        self.include_party = include_party
        self.status_lookup = {
            1: "PAR",
            2: "SLEEP",
            3: "FREEZE",
            4: "BURN",
            5: "POISON",
            6: "BAD_POISON",
        }
        self.screen_flags = {0: "light_screen", 1: "reflect"}
        self.side_effect_flags = {
            0: "leech_seed",
            1: "toxic_counter",
            2: "substitute",
            3: "mist",
        }
        self._current_turn: int = 0

    def extract(self, obs: BattleObs) -> List[WriteOp]:
        writes: List[WriteOp] = []
        self._current_turn = int(obs.get("turn", 0))

        seen_nodes: Dict[Tuple[str, str], Tuple[Node, Dict[str, object]]] = {}
        seen_edges: Dict[Tuple[str, str, str], Tuple[Edge, Dict[str, object]]] = {}

        def add_node(
            node_type: str,
            node_id: str,
            attributes: Dict[str, object],
            *,
            confidence: float = 1.0,
            fallback_required: bool = False,
            turn_hint: Optional[int] = None,
        ) -> None:
            node = Node(node_type, node_id, dict(attributes))
            seen_nodes[(node_type, node_id)] = (
                node,
                {
                    "confidence": confidence,
                    "fallback_required": fallback_required,
                    "turn_hint": turn_hint if turn_hint is not None else self._current_turn,
                },
            )

        def add_edge(
            relation: str,
            src: str,
            dst: str,
            attributes: Dict[str, object],
            *,
            confidence: float = 1.0,
            turn_hint: Optional[int] = None,
        ) -> None:
            edge = Edge(relation, src, dst, dict(attributes))
            seen_edges[(src, relation, dst)] = (
                edge,
                {
                    "confidence": confidence,
                    "fallback_required": False,
                    "turn_hint": turn_hint if turn_hint is not None else self._current_turn,
                },
            )

        turn_id = f"turn:{self._current_turn}"
        add_node("Turn", turn_id, {"turn": self._current_turn}, confidence=1.0, turn_hint=self._current_turn)

        field = obs.get("field", {}) or {}
        add_node("Field", "field:global", dict(field), confidence=1.0)
        add_edge("field_state", turn_id, "field:global", {}, confidence=1.0)
        self._emit_field_effects(turn_id, field, add_node, add_edge)

        for side in ("my", "opp"):
            add_node("Side", f"side:{side}", {"side": side}, confidence=1.0)

        for side, pokemon in (("my", obs.get("my_active") or {}), ("opp", obs.get("opp_active") or {})):
            node_id = self._emit_pokemon(side, pokemon, add_node, add_edge)
            if node_id:
                add_edge(
                    "active_pokemon",
                    f"side:{side}",
                    node_id,
                    {"turn": self._current_turn},
                    confidence=1.0,
                )
                add_edge(
                    "turn_active",
                    turn_id,
                    node_id,
                    {"side": side},
                    confidence=1.0,
                )

        if self.include_party:
            for side, party in (("my", obs.get("my_party") or []), ("opp", obs.get("opp_party") or [])):
                for pokemon in party:
                    node_id = self._emit_pokemon(side, pokemon, add_node, add_edge)
                    if node_id:
                        add_edge(
                            "in_party",
                            f"side:{side}",
                            node_id,
                            {"slot": pokemon.get("slot", 0)},
                            confidence=1.0,
                        )

        for (_, _), (node, meta) in seen_nodes.items():
            writes.append(WriteOp(kind="node", payload=node, **meta))
        for (_, _, _), (edge, meta) in seen_edges.items():
            writes.append(WriteOp(kind="edge", payload=edge, **meta))
        return writes

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _emit_pokemon(
        self,
        side: str,
        pokemon: Dict[str, object],
        add_node: Callable[..., None],
        add_edge: Callable[..., None],
    ) -> Optional[str]:
        node_id = self._pokemon_node_id(side, pokemon)
        if not node_id:
            return None
        species = int(pokemon.get("species") or 0)
        revealed = species > 0
        confidence = 1.0 if revealed else 0.25
        fallback_required = not revealed
        attributes = {
            "side": side,
            "slot": pokemon.get("slot", 0),
            "species": species if revealed else None,
            "revealed": revealed,
            "level": pokemon.get("level"),
            "hp": pokemon.get("hp"),
            "hp_max": pokemon.get("hp_max"),
            "status": pokemon.get("status"),
            "types": tuple(pokemon.get("types", [])),
            "is_active": pokemon.get("is_active", False),
            "fainted": pokemon.get("fainted", False),
        }
        add_node(
            "Pokemon",
            node_id,
            attributes,
            confidence=confidence,
            fallback_required=fallback_required,
        )
        self._emit_status(node_id, pokemon, add_node, add_edge)
        for move in pokemon.get("moves", []):
            self._emit_move(node_id, move, add_node, add_edge)
        return node_id

    def _emit_move(
        self,
        pokemon_node_id: str,
        move: Dict[str, object],
        add_node: Callable[..., None],
        add_edge: Callable[..., None],
    ) -> None:
        move_id = move.get("move_id")
        if move_id in (None, 0):
            return
        node_id = f"move:{int(move_id)}"
        add_node(
            "Move",
            node_id,
            {
                "move_id": int(move_id),
                "max_pp": move.get("max_pp"),
            },
            confidence=1.0,
        )
        add_edge(
            "has_move",
            pokemon_node_id,
            node_id,
            {
                "slot": move.get("slot", 0),
                "pp": move.get("pp", 0),
                "max_pp": move.get("max_pp"),
                "disabled": bool(move.get("disabled", False)),
            },
            confidence=1.0,
        )

    def _emit_status(
        self,
        pokemon_node_id: str,
        pokemon: Dict[str, object],
        add_node: Callable[..., None],
        add_edge: Callable[..., None],
    ) -> None:
        status_code = int(pokemon.get("status") or 0)
        if status_code <= 0:
            return
        status_name = self.status_lookup.get(status_code, f"STATUS_{status_code}")
        status_node_id = f"status:{status_name.lower()}"
        add_node(
            "Status",
            status_node_id,
            {"code": status_code, "name": status_name},
            confidence=1.0,
        )
        add_edge("has_status", pokemon_node_id, status_node_id, {}, confidence=1.0)

    def _emit_field_effects(
        self,
        turn_id: str,
        field: Dict[str, object],
        add_node: Callable[..., None],
        add_edge: Callable[..., None],
    ) -> None:
        side_player = "side:my"
        side_enemy = "side:opp"
        screens_player = int(field.get("player_screens") or 0)
        screens_enemy = int(field.get("enemy_screens") or 0)
        for bit, name in self.screen_flags.items():
            mask = 1 << bit
            if screens_player & mask:
                self._register_screen(side_player, name, turn_id, add_node, add_edge)
            if screens_enemy & mask:
                self._register_screen(side_enemy, name, turn_id, add_node, add_edge)

        effects_player = int(field.get("player_side_effects") or 0)
        effects_enemy = int(field.get("enemy_side_effects") or 0)
        for bit, name in self.side_effect_flags.items():
            mask = 1 << bit
            if effects_player & mask:
                self._register_side_effect(side_player, name, turn_id, add_node, add_edge)
            if effects_enemy & mask:
                self._register_side_effect(side_enemy, name, turn_id, add_node, add_edge)

    def _register_screen(
        self,
        side_node: str,
        screen_name: str,
        turn_id: str,
        add_node: Callable[..., None],
        add_edge: Callable[..., None],
    ) -> None:
        screen_id = f"screen:{screen_name}"
        add_node("Screen", screen_id, {"name": screen_name}, confidence=1.0)
        add_edge("active_screen", side_node, screen_id, {"turn": turn_id}, confidence=1.0)

    def _register_side_effect(
        self,
        side_node: str,
        effect_name: str,
        turn_id: str,
        add_node: Callable[..., None],
        add_edge: Callable[..., None],
    ) -> None:
        effect_id = f"side_effect:{effect_name}"
        add_node("SideEffect", effect_id, {"name": effect_name}, confidence=1.0)
        add_edge("active_side_effect", side_node, effect_id, {"turn": turn_id}, confidence=1.0)

    @staticmethod
    def _pokemon_node_id(side: str, pokemon: Dict[str, object]) -> Optional[str]:
        slot = pokemon.get("slot")
        species = pokemon.get("species")
        if slot is None and species is None:
            return None
        slot_val = int(slot or 0)
        species_val = int(species or 0)
        return f"pokemon:{side}:{slot_val}:{species_val}"
