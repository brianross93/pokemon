"""
Persistent symbolic memory for Pokemon battles.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import Edge, Node, WriteOp
from .snapshot import GraphSnapshot


@dataclass
class _NodeRecord:
    node: Node
    version: int
    last_seen_turn: Optional[int]
    confidence: float


@dataclass
class _EdgeRecord:
    edge: Edge
    version: int
    last_seen_turn: Optional[int]


class GraphMemory:
    """
    In-memory graph store supporting SR-FBAM's ASSOC/FOLLOW operations.

    Nodes are stored per-type with LRU eviction and optional sliding window
    pruning over historical turns. Writes are idempotent and keyed by
    ``(type, node_id)``.
    """

    def __init__(
        self,
        max_nodes_per_type: int = 256,
        turn_horizon: int = 64,
        confidence_threshold: float = 0.6,
    ) -> None:
        self._max_nodes_per_type = max(1, int(max_nodes_per_type))
        self._turn_horizon = max(0, int(turn_horizon))
        self._confidence_threshold = float(confidence_threshold)

        self._nodes_by_type: Dict[str, OrderedDict[str, _NodeRecord]] = defaultdict(OrderedDict)
        self._nodes_by_id: Dict[str, _NodeRecord] = {}

        self._edges: Dict[Tuple[str, str, str], _EdgeRecord] = {}
        self._edges_by_src: Dict[Tuple[str, str], Dict[str, _EdgeRecord]] = defaultdict(dict)
        self._edge_hops: List[Dict[str, object]] = []

        self._current_turn: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    def write(self, op: WriteOp) -> None:
        confidence = getattr(op, "confidence", 1.0)
        turn_hint = getattr(op, "turn_hint", None)

        if op.kind == "node":
            self._write_node(op.payload, confidence, turn_hint)
        elif op.kind == "edge":
            self._write_edge(op.payload, turn_hint)
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown WriteOp kind '{op.kind}'")

    def _write_node(self, node: Node, confidence: float, turn_hint: Optional[int]) -> None:
        node_type = node.type
        node_id = node.node_id
        type_table = self._nodes_by_type[node_type]
        record = type_table.get(node_id)

        last_seen_turn = self._extract_turn_hint(node, turn_hint)
        if node_type.lower() == "turn" and last_seen_turn is not None:
            self._current_turn = last_seen_turn

        if record is None:
            record = _NodeRecord(node=node, version=1, last_seen_turn=last_seen_turn, confidence=confidence)
            type_table[node_id] = record
            self._nodes_by_id[node_id] = record
        else:
            record.node = node
            record.version += 1
            if last_seen_turn is not None:
                record.last_seen_turn = last_seen_turn
            record.confidence = confidence if confidence is not None else record.confidence
            type_table.move_to_end(node_id)

        if record.confidence < self._confidence_threshold:
            record.confidence = confidence

        self._prune_type(node_type)
        self._prune_by_turn()

    def _write_edge(self, edge: Edge, turn_hint: Optional[int]) -> None:
        key = (edge.src, edge.relation, edge.dst)
        record = self._edges.get(key)
        last_seen_turn = self._current_turn if turn_hint is None else turn_hint

        if record is None:
            record = _EdgeRecord(edge=edge, version=1, last_seen_turn=last_seen_turn)
            self._edges[key] = record
            self._edges_by_src[(edge.src, edge.relation)][edge.dst] = record
        else:
            record.edge = edge
            record.version += 1
            if last_seen_turn is not None:
                record.last_seen_turn = last_seen_turn

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def assoc(
        self,
        *,
        type_: Optional[str] = None,
        key: Optional[str] = None,
        filters: Optional[Dict[str, object]] = None,
    ) -> Sequence[Node]:
        if type_ is not None:
            type_table = self._nodes_by_type.get(type_, OrderedDict())
            candidates = type_table.values()
        else:
            candidates = self._nodes_by_id.values()

        results: List[Node] = []
        for record in candidates:
            node = record.node
            if key is not None and key not in {node.node_id, node.attributes.get("name")}:
                continue
            if filters and not _match_filters(node.attributes, filters):
                continue
            results.append(node)
        results.sort(key=lambda n: n.node_id)
        return results

    def follow(
        self,
        *,
        src: str,
        relation: str,
        filters: Optional[Dict[str, object]] = None,
    ) -> Sequence[Node]:
        edge_records = self._edges_by_src.get((src, relation))
        if not edge_records:
            return []

        nodes: List[Node] = []
        for dst, record in list(edge_records.items()):
            node_record = self._nodes_by_id.get(dst)
            if node_record is None:
                self._remove_edge((src, relation, dst))
                continue
            if filters and not _match_filters(record.edge.attributes, filters):
                continue
            nodes.append(node_record.node)
            self._edge_hops.append(
                {
                    "src": src,
                    "relation": relation,
                    "dst": node_record.node.node_id,
                    "edge_attributes": dict(record.edge.attributes),
                    "node_attributes": dict(node_record.node.attributes),
                }
            )
        nodes.sort(key=lambda n: n.node_id)
        return nodes

    def iter_edges(self) -> Iterable[Edge]:
        return (record.edge for record in self._edges.values())

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def snapshot(self) -> GraphSnapshot:
        node_dicts = [
            {"type": record.node.type, "node_id": record.node.node_id, "attributes": dict(record.node.attributes)}
            for record in sorted(self._nodes_by_id.values(), key=lambda r: r.node.node_id)
        ]
        edge_dicts = [
            {
                "relation": record.edge.relation,
                "src": record.edge.src,
                "dst": record.edge.dst,
                "attributes": dict(record.edge.attributes),
            }
            for record in sorted(self._edges.values(), key=lambda r: (r.edge.src, r.edge.relation, r.edge.dst))
        ]
        return GraphSnapshot(nodes=node_dicts, edges=edge_dicts)

    def drain_hops(self) -> List[Dict[str, object]]:
        hops, self._edge_hops = self._edge_hops, []
        return hops

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _extract_turn_hint(self, node: Node, explicit: Optional[int]) -> Optional[int]:
        if explicit is not None:
            return int(explicit)
        attr_turn = node.attributes.get("turn")
        if isinstance(attr_turn, int):
            return int(attr_turn)
        return self._current_turn

    def _prune_type(self, node_type: str) -> None:
        type_table = self._nodes_by_type[node_type]
        while len(type_table) > self._max_nodes_per_type:
            node_id, _ = type_table.popitem(last=False)
            self._remove_node(node_type, node_id)

    def _prune_by_turn(self) -> None:
        if self._turn_horizon <= 0 or self._current_turn is None:
            return
        threshold = self._current_turn - self._turn_horizon
        for node_type, type_table in list(self._nodes_by_type.items()):
            for node_id, record in list(type_table.items()):
                last_seen = record.last_seen_turn
                if last_seen is not None and last_seen < threshold:
                    self._remove_node(node_type, node_id)

    def _remove_node(self, node_type: str, node_id: str) -> None:
        type_table = self._nodes_by_type.get(node_type)
        if type_table and node_id in type_table:
            del type_table[node_id]
        self._nodes_by_id.pop(node_id, None)
        self._remove_edges_for_node(node_id)

    def _remove_edges_for_node(self, node_id: str) -> None:
        for key, edge_records in list(self._edges_by_src.items()):
            for dst in list(edge_records.keys()):
                if dst == node_id:
                    self._remove_edge((key[0], key[1], dst))
        for key in list(self._edges.keys()):
            if key[0] == node_id or key[2] == node_id:
                self._remove_edge(key)

    def _remove_edge(self, key: Tuple[str, str, str]) -> None:
        record = self._edges.pop(key, None)
        if record is None:
            return
        src_rel = (key[0], key[1])
        dst_map = self._edges_by_src.get(src_rel)
        if dst_map is not None:
            dst_map.pop(key[2], None)
            if len(dst_map) == 0:
                self._edges_by_src.pop(src_rel, None)


def _match_filters(attributes: Dict[str, object], filters: Dict[str, object]) -> bool:
    for key, expected in filters.items():
        if attributes.get(key) != expected:
            return False
    return True
