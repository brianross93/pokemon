"""
Planlet caching utilities shared across battle and overworld domains.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from pkmn_battle.summarizer import GraphSummary
from pkmn_overworld.summarizer import WorldSummary


def _hash_payload(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA256 digest for the given payload."""

    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _trim_entities(entities: Mapping[str, Iterable[Mapping[str, Any]]], limit: int) -> Dict[str, List[Mapping[str, Any]]]:
    trimmed: Dict[str, List[Mapping[str, Any]]] = {}
    for type_name, items in sorted(entities.items()):
        bucket = []
        for item in items:
            bucket.append(dict(item))
            if len(bucket) >= limit:
                break
        if bucket:
            trimmed[type_name] = bucket
    return trimmed


def _trim_list(items: Iterable[Mapping[str, Any]], limit: int) -> List[Mapping[str, Any]]:
    result: List[Mapping[str, Any]] = []
    for item in items:
        result.append(dict(item))
        if len(result) >= limit:
            break
    return result


def make_battle_cache_key(summary: GraphSummary, *, max_entities_per_type: int = 16, max_relations: int = 64) -> str:
    """
    Compute a cache key for a battle planlet based on the local graph neighbourhood.
    """

    payload = summary.to_payload()
    entities = payload.get("entities", {})
    relations = payload.get("relations", [])
    digest_payload = {
        "domain": "battle",
        "turn": payload.get("turn"),
        "side": payload.get("side"),
        "format": payload.get("format"),
        "entities": _trim_entities(entities, max_entities_per_type) if isinstance(entities, Mapping) else {},
        "relations": _trim_list(relations, max_relations) if isinstance(relations, list) else [],
        "actives": payload.get("actives"),
        "hazards": payload.get("hazards"),
    }
    return _hash_payload(digest_payload)


def make_overworld_cache_key(summary: WorldSummary, *, nearby_limit: int = 8) -> str:
    """
    Compute a cache key for an overworld planlet using nearby tiles and actors.
    """

    payload = summary.to_payload()
    nearby = payload.get("nearby", {})
    digest_payload = {
        "domain": "overworld",
        "map_id": summary.map_id,
        "player": payload.get("player"),
        "tiles": _trim_list(payload.get("tiles", []), nearby_limit) if isinstance(payload.get("tiles"), list) else [],
        "nearby": {
            "npcs": _trim_list(nearby.get("npcs", []), nearby_limit) if isinstance(nearby, Mapping) else [],
            "shops": _trim_list(nearby.get("shops", []), nearby_limit) if isinstance(nearby, Mapping) else [],
            "hazards": _trim_list(nearby.get("hazards", []), nearby_limit) if isinstance(nearby, Mapping) else [],
        },
        "graph": payload.get("graph"),
    }
    return _hash_payload(digest_payload)


@dataclass
class CacheHit:
    planlet: Dict[str, Any]
    retrieved_docs: Optional[List[Mapping[str, Any]]]
    token_usage: Optional[Mapping[str, Any]]
    raw_response: Optional[str]
    metadata: Dict[str, Any]
    cache_key: str


@dataclass
class CacheEntry:
    planlet: Dict[str, Any]
    retrieved_docs: Optional[List[Mapping[str, Any]]]
    token_usage: Optional[Mapping[str, Any]]
    raw_response: Optional[str]
    metadata: Dict[str, Any]
    planlet_id: str
    created_at: float
    last_used: float
    hits: int = 0
    successes: float = 0.0
    attempts: float = 0.0

    def success_rate(self) -> float:
        if self.attempts <= 1e-9:
            return 0.5
        return self.successes / self.attempts if self.attempts > 1e-9 else 0.0


class PlanCache:
    """Success-weighted planlet cache with TTL-based retention."""

    def __init__(
        self,
        *,
        max_size: int = 64,
        ttl_seconds: Optional[float] = 900.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = ttl_seconds if ttl_seconds is None or ttl_seconds > 0 else None
        self._clock = clock
        self._entries: Dict[str, CacheEntry] = {}
        self._planlet_index: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Key helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def battle_key(summary: GraphSummary, *, max_entities_per_type: int = 16, max_relations: int = 64) -> str:
        return make_battle_cache_key(summary, max_entities_per_type=max_entities_per_type, max_relations=max_relations)

    @staticmethod
    def overworld_key(summary: WorldSummary, *, nearby_limit: int = 8) -> str:
        return make_overworld_cache_key(summary, nearby_limit=nearby_limit)

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def lookup(self, key: str) -> Optional[CacheHit]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            self._remove(key)
            return None

        entry.hits += 1
        entry.last_used = self._clock()

        return CacheHit(
            planlet=copy.deepcopy(entry.planlet),
            retrieved_docs=copy.deepcopy(entry.retrieved_docs),
            token_usage=copy.deepcopy(entry.token_usage),
            raw_response=entry.raw_response,
            metadata=dict(entry.metadata),
            cache_key=key,
        )

    def store(
        self,
        key: str,
        planlet: Mapping[str, Any],
        *,
        retrieved_docs: Optional[List[Mapping[str, Any]]] = None,
        token_usage: Optional[Mapping[str, Any]] = None,
        raw_response: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        now = self._clock()
        self._prune_expired()

        planlet_copy = copy.deepcopy(dict(planlet))
        docs_copy = copy.deepcopy(retrieved_docs) if retrieved_docs is not None else None
        usage_copy = copy.deepcopy(token_usage) if token_usage is not None else None
        entry = self._entries.get(key)
        planlet_id = str(planlet_copy.get("planlet_id") or "")

        if entry is not None:
            if entry.planlet_id and entry.planlet_id != planlet_id:
                self._planlet_index.pop(entry.planlet_id, None)
            entry.planlet = planlet_copy
            entry.retrieved_docs = docs_copy
            entry.token_usage = usage_copy
            entry.raw_response = raw_response
            entry.metadata = dict(metadata or {})
            entry.planlet_id = planlet_id
            entry.last_used = now
            entry.created_at = now
        else:
            entry = CacheEntry(
                planlet=planlet_copy,
                retrieved_docs=docs_copy,
                token_usage=usage_copy,
                raw_response=raw_response,
                metadata=dict(metadata or {}),
                planlet_id=planlet_id,
                created_at=now,
                last_used=now,
            )
            self._entries[key] = entry

        if planlet_id:
            self._planlet_index[planlet_id] = key

        self._evict_if_needed()

    def register_planlet(self, planlet_id: str, key: str) -> None:
        if not planlet_id or key not in self._entries:
            return
        self._entries[key].planlet_id = planlet_id
        self._planlet_index[planlet_id] = key

    def record_feedback(self, key: str, success: bool, *, weight: float = 1.0) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        weight = max(0.0, float(weight))
        entry.attempts += weight
        if success:
            entry.successes += weight
        entry.last_used = self._clock()

    def record_feedback_by_planlet(self, planlet_id: str, success: bool, *, weight: float = 1.0) -> None:
        key = self._planlet_index.get(planlet_id)
        if key is None:
            return
        self.record_feedback(key, success, weight=weight)

    def prune(self) -> None:
        """Eagerly remove expired entries."""

        self._prune_expired()

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._entries),
            "keys": list(self._entries.keys()),
            "planlets": {key: entry.planlet_id for key, entry in self._entries.items()},
            "total_hits": sum(entry.hits for entry in self._entries.values()),
            "attempts": sum(entry.attempts for entry in self._entries.values()),
            "successes": sum(entry.successes for entry in self._entries.values()),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _is_expired(self, entry: CacheEntry) -> bool:
        if self.ttl_seconds is None:
            return False
        return (self._clock() - entry.created_at) >= self.ttl_seconds

    def _prune_expired(self) -> None:
        for key in list(self._entries.keys()):
            entry = self._entries.get(key)
            if entry is None:
                continue
            if self._is_expired(entry):
                self._remove(key)

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        if entry.planlet_id:
            self._planlet_index.pop(entry.planlet_id, None)

    def _evict_if_needed(self) -> None:
        if len(self._entries) <= self.max_size:
            return
        while len(self._entries) > self.max_size:
            key, _ = min(
                self._entries.items(),
                key=lambda item: (item[1].success_rate(), item[1].last_used),
            )
            self._remove(key)


__all__ = [
    "PlanCache",
    "CacheHit",
    "make_battle_cache_key",
    "make_overworld_cache_key",
]
