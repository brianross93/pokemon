"""
Utilities for persisting planlets and planner metadata to disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd


def _iso_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PlanletRecord:
    """
    Container describing a single planlet emission for storage.

    Attributes
    ----------
    planlet_id:
        Identifier supplied by the planner (unique per planlet instance).
    mode:
        High-level mode (`"battle"` or `"overworld"`).
    planlet_kind:
        Domain-specific kind (e.g. `"BATTLE"`, `"NAVIGATE_TO"`).
    plan_id:
        Optional identifier for the enclosing plan/bundle.
    goal:
        High-level goal string provided by the planner.
    seed_frame_id:
        Frame/turn identifier associated with the planlet request.
    summary:
        Snapshot payload shown to the planner (graph summary or state digest).
    retrieved_docs:
        Any documents returned by tool retrieval and fed into the planner.
    token_usage:
        Token accounting metadata returned by the LLM backend.
    llm_model:
        Model identifier used to generate the planlet.
    llm_config:
        Additional configuration parameters (temperature, reasoning effort, etc.).
    generated_at:
        Timestamp when the planlet was emitted (UTC ISO).
    raw_planlet:
        The validated planlet payload as emitted by the planner.
    source:
        Origin of the planlet (`"llm"` vs `"cache"`).
    cache_key:
        Optional cache key used to retrieve/store the planlet.
    cache_hit:
        Whether this planlet was served from cache.
    extra:
        Domain-specific metadata (e.g. graph signatures, executor hints).
    """

    planlet_id: str
    mode: str
    planlet_kind: Optional[str]
    plan_id: Optional[str] = None
    goal: Optional[str] = None
    seed_frame_id: Optional[int] = None
    summary: Optional[Mapping[str, Any]] = None
    retrieved_docs: Optional[Sequence[Mapping[str, Any]]] = None
    token_usage: Optional[Mapping[str, Any]] = None
    llm_model: Optional[str] = None
    llm_config: Optional[Mapping[str, Any]] = None
    generated_at: datetime = field(default_factory=_iso_now)
    raw_planlet: Mapping[str, Any] = field(default_factory=dict)
    source: str = "llm"
    cache_key: Optional[str] = None
    cache_hit: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """Return a serialisable representation for storage backends."""

        payload = asdict(self)
        timestamp = self.generated_at.isoformat()
        payload["generated_at"] = timestamp
        raw_planlet = payload.get("raw_planlet") or {}
        if not raw_planlet:
            raw_planlet = {"planlet_id": self.planlet_id}
        payload["raw_planlet"] = raw_planlet
        if not payload.get("extra"):
            payload["extra"] = None
        return payload

    def manifest_entry(self) -> Dict[str, str]:
        """Create a manifest entry for downstream dataset tooling."""

        return {
            "planlet_id": self.planlet_id,
            "planlet_json": json.dumps(self.raw_planlet, ensure_ascii=False, sort_keys=True),
        }


class PlanletStore:
    """Persists planlets to JSONL/Parquet while keeping a manifest in sync."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / "planlets.jsonl"
        self.parquet_path = self.root / "planlets.parquet"
        self.manifest_path = self.root / "planlets_manifest.json"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def append(self, record: PlanletRecord) -> None:
        """Persist a planlet record to all storage formats."""

        row = record.to_row()
        self._append_jsonl(row)
        self._append_parquet(row)
        self._update_manifest(record.manifest_entry())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _append_jsonl(self, row: Mapping[str, Any]) -> None:
        if not self.jsonl_path.exists():
            self.jsonl_path.touch()
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")

    def _append_parquet(self, row: Mapping[str, Any]) -> None:
        row_df = pd.DataFrame([row])
        if self.parquet_path.exists():
            existing = pd.read_parquet(self.parquet_path)
            combined = pd.concat([existing, row_df], ignore_index=True)
        else:
            combined = row_df
        combined.to_parquet(self.parquet_path, index=False)

    def _update_manifest(self, entry: Mapping[str, str]) -> None:
        manifest: list[Dict[str, str]]
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = []

        filtered = [item for item in manifest if item.get("planlet_id") != entry["planlet_id"]]
        filtered.append(dict(entry))
        filtered.sort(key=lambda item: item["planlet_id"])
        self.manifest_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["PlanletRecord", "PlanletStore"]
