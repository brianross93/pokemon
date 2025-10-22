#!/usr/bin/env python3
"""
Aggregate SR-FBAM telemetry JSONL logs into Parquet summaries for dashboards.

The script focuses on overworld telemetry and computes:
  * gate mix per plan source
  * HALT reasons (status + reason counts)
  * plan-source cache hit rates

Example:
    python scripts/summarize_telemetry.py \
        --input data/telemetry/overworld_run.jsonl \
        --output data/telemetry/overworld_summary.parquet \
        --run-id corridor_seed_001
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - defensive guard
    raise SystemExit(
        "pandas is required for summarize_telemetry.py. "
        "Install project dependencies (pip install -r requirements.txt)."
    ) from exc


LOGGER = logging.getLogger("summarize_telemetry")
SUMMARY_COLUMNS = [
    "run_id",
    "metric",
    "plan_source",
    "gate_mode",
    "reason",
    "status",
    "planner_origin",
    "count",
    "fraction",
    "total_plans",
    "cache_hits",
    "cache_hit_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SR-FBAM telemetry logs.")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more telemetry JSONL files or directories containing them.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination Parquet file for aggregated metrics.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run identifier applied to all records. "
        "Defaults to each file's stem when omitted.",
    )
    parser.add_argument(
        "--domain",
        default="overworld",
        help="Telemetry domain to summarize (default: overworld).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Log the aggregated summary to stdout after writing the Parquet file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on malformed JSON lines instead of skipping them.",
    )
    return parser.parse_args()


def resolve_inputs(paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for item in paths:
        path = Path(item)
        if not path.exists():
            LOGGER.warning("Input path not found: %s", path)
            continue
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            files.append(path)
    return files


def iter_entries(
    paths: Iterable[Path],
    *,
    domain: str,
    run_id_override: Optional[str],
    strict: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        run_id = run_id_override or path.stem
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Unable to open %s: %s", path, exc)
            continue
        with handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    message = f"Invalid JSON in {path} line {line_number}: {exc}"
                    if strict:
                        raise RuntimeError(message) from exc
                    LOGGER.warning("%s", message)
                    continue
                record = _extract_record(entry, run_id=run_id, source_path=str(path), domain=domain)
                if record is not None:
                    records.append(record)
    return records


def _extract_record(
    entry: Mapping[str, Any],
    *,
    run_id: str,
    source_path: str,
    domain: str,
) -> Optional[Dict[str, Any]]:
    context = entry.get("context")
    if not isinstance(context, Mapping):
        return None
    if context.get("domain") != domain:
        return None

    telemetry = entry.get("telemetry")
    if not isinstance(telemetry, Mapping):
        telemetry = {}
    core = telemetry.get("core")
    if not isinstance(core, Mapping):
        core = {}
    gate = core.get("gate")
    if not isinstance(gate, Mapping):
        gate = {}
    plan_info = _merge_plan_info(context.get("plan"), core.get("plan"))

    record: Dict[str, Any] = {
        "run_id": run_id,
        "source_path": source_path,
        "entry_source": entry.get("source"),
        "timestamp": entry.get("timestamp"),
        "status": context.get("status"),
        "reason": context.get("reason"),
        "step_index": context.get("step_index"),
        "plan_id": plan_info.get("id") or plan_info.get("plan_id"),
        "planlet_id": plan_info.get("planlet_id"),
        "planlet_kind": plan_info.get("planlet_kind"),
        "plan_source": plan_info.get("plan_source") or plan_info.get("source"),
        "planner_origin": plan_info.get("planner_origin"),
        "planner_reason": plan_info.get("planner_reason"),
        "planner_cache_hit": _to_bool(plan_info.get("planner_cache_hit") or plan_info.get("cache_hit")),
        "planner_cache_key": plan_info.get("planner_cache_key") or plan_info.get("cache_key"),
        "gate_mode": gate.get("mode"),
        "gate_raw": gate.get("raw"),
        "gate_encode": _to_bool(gate.get("encode_flag")),
        "gate_view": gate.get("view"),
        "planner_token_usage": plan_info.get("planner_token_usage"),
    }
    return record


def _merge_plan_info(*plans: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for plan in plans:
        if isinstance(plan, Mapping):
            for key, value in plan.items():
                merged.setdefault(key, value)
    return merged


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def build_summary(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    df = pd.DataFrame(records)
    df["plan_source"] = df["plan_source"].fillna("unknown")
    df["planner_origin"] = df["planner_origin"].fillna("unknown")
    df["planner_cache_hit"] = df["planner_cache_hit"].fillna(False).astype(bool)

    summary_rows: List[Dict[str, Any]] = []

    gate_df = df.dropna(subset=["gate_mode"])
    if not gate_df.empty:
        gate_counts = (
            gate_df.groupby(["run_id", "plan_source", "gate_mode"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        gate_counts["fraction"] = gate_counts.groupby(["run_id", "plan_source"])["count"].transform(
            lambda col: col / col.sum() if col.sum() else 0.0
        )
        for row in gate_counts.itertuples():
            summary_rows.append(
                {
                    "run_id": row.run_id,
                    "metric": "gate_mix",
                    "plan_source": row.plan_source,
                    "gate_mode": row.gate_mode,
                    "reason": None,
                    "status": None,
                    "planner_origin": None,
                    "count": int(row.count),
                    "fraction": float(row.fraction),
                    "total_plans": None,
                    "cache_hits": None,
                    "cache_hit_rate": None,
                }
            )

    halt_df = df[df["reason"].notna()]
    if not halt_df.empty:
        halt_counts = (
            halt_df.groupby(["run_id", "plan_source", "status", "reason"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        for row in halt_counts.itertuples():
            summary_rows.append(
                {
                    "run_id": row.run_id,
                    "metric": "halt_reason",
                    "plan_source": row.plan_source,
                    "gate_mode": None,
                    "reason": row.reason,
                    "status": row.status,
                    "planner_origin": None,
                    "count": int(row.count),
                    "fraction": None,
                    "total_plans": None,
                    "cache_hits": None,
                    "cache_hit_rate": None,
                }
            )

    plan_df = df.dropna(subset=["plan_id"])
    if not plan_df.empty:
        plan_keys = (
            plan_df[
                [
                    "run_id",
                    "plan_source",
                    "planner_origin",
                    "plan_id",
                    "planner_cache_hit",
                ]
            ]
            .drop_duplicates(subset=["run_id", "plan_id"])
            .reset_index(drop=True)
        )
        plan_summary = (
            plan_keys.groupby(["run_id", "plan_source", "planner_origin"], dropna=False)
            .agg(
                total_plans=("plan_id", "nunique"),
                cache_hits=("planner_cache_hit", lambda col: int(col.sum())),
            )
            .reset_index()
        )
        plan_summary["cache_hit_rate"] = plan_summary.apply(
            lambda row: (row["cache_hits"] / row["total_plans"]) if row["total_plans"] else 0.0,
            axis=1,
        )
        for row in plan_summary.itertuples():
            summary_rows.append(
                {
                    "run_id": row.run_id,
                    "metric": "plan_source",
                    "plan_source": row.plan_source,
                    "gate_mode": None,
                    "reason": None,
                    "status": None,
                    "planner_origin": row.planner_origin,
                    "count": None,
                    "fraction": None,
                    "total_plans": int(row.total_plans),
                    "cache_hits": int(row.cache_hits),
                    "cache_hit_rate": float(row.cache_hit_rate),
                }
            )

    if not summary_rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary_df = pd.DataFrame(summary_rows)
    for column in SUMMARY_COLUMNS:
        if column not in summary_df:
            summary_df[column] = None
    return summary_df[SUMMARY_COLUMNS]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    inputs = resolve_inputs(args.input)
    if not inputs:
        LOGGER.error("No telemetry files found for inputs: %s", args.input)
        return 1

    LOGGER.info("Processing %d telemetry file(s).", len(inputs))
    records = iter_entries(
        inputs,
        domain=args.domain,
        run_id_override=args.run_id,
        strict=args.strict,
    )
    summary = build_summary(records)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary.to_parquet(output_path, index=False)
    except Exception as exc:  # pragma: no cover - primarily IO errors
        LOGGER.error("Unable to write Parquet file %s: %s", output_path, exc)
        return 1

    LOGGER.info("Wrote %s rows to %s", len(summary), output_path)
    if args.print_summary:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
