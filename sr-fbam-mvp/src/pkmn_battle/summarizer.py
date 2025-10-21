from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .graph.memory import GraphMemory


@dataclass
class GraphSummary:
    """Structured snapshot of the battle graph for LLM consumption."""

    turn: int
    side: str
    format: str
    data: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-serialisable payload for prompts."""

        return {
            "turn": self.turn,
            "side": self.side,
            "format": self.format,
            **self.data,
        }


def summarize_for_llm(memory: GraphMemory, side_view: str) -> GraphSummary:
    """
    Produce a deterministic summary of the graph suitable for the planlet LLM.

    The summary groups nodes by type, lists relations, and highlights key
    battle entities (active Pokémon, revealed info, hazards). The output is
    deterministic (sorted) so the prompt remains stable run-to-run.
    """

    snapshot = memory.snapshot()
    nodes = snapshot.nodes
    edges = snapshot.edges

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        node_type = str(node.get("type", "")).lower()
        bucket = by_type.setdefault(node_type, [])
        entry = {"node_id": node.get("node_id")}
        entry.update(node.get("attributes", {}))
        bucket.append(entry)

    for entries in by_type.values():
        entries.sort(key=lambda item: (str(item.get("node_id")), json_key_order(item)))

    relations = sorted(
        [
            {
                "src": edge.get("src"),
                "relation": edge.get("relation"),
                "dst": edge.get("dst"),
                **edge.get("attributes", {}),
            }
            for edge in edges
        ],
        key=lambda e: (str(e.get("src")), str(e.get("relation")), str(e.get("dst")), json_key_order(e)),
    )

    turn = _extract_turn(nodes)
    battle_format = _extract_format(nodes)
    actives = _extract_actives(edges, side_view)
    hazards = _extract_hazards(relations)

    data = {
        "entities": by_type,
        "relations": relations,
        "actives": actives,
        "hazards": hazards,
    }

    return GraphSummary(turn=turn, side=side_view, format=battle_format, data=data)


def _extract_turn(nodes: Iterable[Dict[str, Any]]) -> int:
    for node in nodes:
        if str(node.get("type", "")).lower() == "turn":
            attr = node.get("attributes", {})
            value = attr.get("turn")
            if isinstance(value, int):
                return value
    return 0


def _extract_format(nodes: Iterable[Dict[str, Any]]) -> str:
    for node in nodes:
        attributes = node.get("attributes", {})
        if "format" in attributes and isinstance(attributes["format"], str):
            return attributes["format"]
        node_type = str(node.get("type", "")).lower()
        if node_type == "format" and isinstance(attributes.get("name"), str):
            return attributes["name"]
    return "unknown"


def _extract_actives(edges: Iterable[Dict[str, Any]], side_view: str) -> Dict[str, Any]:
    actives = {}
    target_map: Dict[str, str] = {}
    for edge in edges:
        if str(edge.get("relation")) == "active":
            src = str(edge.get("src"))
            dst = str(edge.get("dst"))
            target_map[src] = dst
    our_key = f"side-{side_view}"
    opp_key = "side-p2" if side_view == "p1" else "side-p1"
    actives["ours"] = target_map.get(our_key)
    actives["opponent"] = target_map.get(opp_key)
    return actives


def _extract_hazards(relations: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    hazards: Dict[str, List[str]] = {"ours": [], "opponent": []}
    for rel in relations:
        relation = rel.get("relation")
        if relation not in {"has_hazard", "hazard"}:
            continue
        src = str(rel.get("src", ""))
        hazard_name = rel.get("hazard") or rel.get("name") or rel.get("dst")
        if isinstance(hazard_name, str):
            if src.endswith("side-p1"):
                hazards["ours"].append(hazard_name)
            elif src.endswith("side-p2"):
                hazards["opponent"].append(hazard_name)
    for key in hazards:
        hazards[key].sort()
    return hazards


def json_key_order(record: Dict[str, Any]) -> Tuple:
    """Helper to stabilise ordering when attributes are dictionaries."""

    return tuple(sorted(record.items()))
