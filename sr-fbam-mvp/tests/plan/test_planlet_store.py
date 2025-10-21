from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.plan.storage import PlanletRecord, PlanletStore


def test_planlet_store_appends_to_all_backends(tmp_path: Path) -> None:
    store = PlanletStore(tmp_path)
    record = PlanletRecord(
        planlet_id="pl_test_9000",
        mode="battle",
        planlet_kind="BATTLE",
        plan_id="plan-1",
        goal="Win quickly",
        seed_frame_id=42,
        summary={"turn": 17, "side": "p1"},
        retrieved_docs=[{"title": "doc", "snippet": "content"}],
        token_usage={"prompt_tokens": 123, "completion_tokens": 45},
        llm_model="gpt-5",
        llm_config={"reasoning_effort": "medium"},
        generated_at=datetime(2025, 10, 20, 12, 0, tzinfo=timezone.utc),
        raw_planlet={"planlet_id": "pl_test_9000", "script": []},
        source="llm",
        cache_key="hash:abc",
        cache_hit=False,
        extra={"graph_signature": "hash123"},
    )

    store.append(record)

    jsonl_path = tmp_path / "planlets.jsonl"
    parquet_path = tmp_path / "planlets.parquet"
    manifest_path = tmp_path / "planlets_manifest.json"

    assert jsonl_path.exists()
    assert parquet_path.exists()
    assert manifest_path.exists()

    with jsonl_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert len(lines) == 1
    stored = lines[0]
    assert stored["planlet_id"] == "pl_test_9000"
    assert stored["token_usage"]["completion_tokens"] == 45

    df = pd.read_parquet(parquet_path)
    assert df.shape[0] == 1
    assert df.loc[0, "cache_key"] == "hash:abc"
    assert df.loc[0, "summary"]["turn"] == 17

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == [
        {
            "planlet_id": "pl_test_9000",
            "planlet_json": json.dumps({"planlet_id": "pl_test_9000", "script": []}, ensure_ascii=False, sort_keys=True),
        }
    ]


def test_planlet_store_replaces_existing_manifest_entry(tmp_path: Path) -> None:
    store = PlanletStore(tmp_path)
    first = PlanletRecord(planlet_id="dup", mode="overworld", planlet_kind="NAVIGATE_TO")
    second = PlanletRecord(planlet_id="dup", mode="overworld", planlet_kind="NAVIGATE_TO", raw_planlet={"id": "dup"})

    store.append(first)
    store.append(second)

    manifest = json.loads((tmp_path / "planlets_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert manifest[0]["planlet_id"] == "dup"
    assert manifest[0]["planlet_json"] == json.dumps({"id": "dup"}, ensure_ascii=False, sort_keys=True)
