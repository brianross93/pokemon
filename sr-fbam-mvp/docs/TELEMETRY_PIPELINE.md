# Telemetry Dashboard Pipeline

The consolidated telemetry schema (`telemetry.core`, `telemetry.overworld`,
`telemetry.battle`) feeds two downstream workflows:

1. **Dashboard ingestion** – aggregate metrics consumed by the paper notebooks
   and internal dashboards.
2. **Historical archival** – migrate legacy JSONL logs into the new schema so
   the ecosystem no longer needs to special-case flat payloads.

This guide explains how to run both steps using the helpers that ship with the
repository.

## 1. Generate dashboard summaries

`scripts/update_dashboard_metrics.py` wraps the `summarize_entries` helper and
produces a consolidated JSON payload. By default it scans runtime directories
(`runs/*.jsonl`, `results/pkmn_logs/*.jsonl`) and writes the result to
`results/summary/telemetry_dashboard.json` – the location used by the existing
visualisation notebooks. Planlet-aware runs add telemetry fields such as `planlet_id`,
`plan_confidence`, `plan_step_idx`, gate distribution including `PLAN_LOOKUP` / `PLAN_STEP`,
and detailed LLM latency buckets (`plan.llm_ms`, `plan.search_ms`, `plan.cache_hit_ms`).

Planlet-aware runs add telemetry fields: planlet_id, plan_confidence, plan_step_idx, gate distribution including PLAN_LOOKUP/PLAN_STEP, and detailed LLM latency buckets (plan.llm_ms, plan.search_ms, plan.cache_hit).

```bash
python scripts/update_dashboard_metrics.py
```

Custom locations are supported via `--inputs`/`--output` and `--pretty` enables
human-readable JSON for ad-hoc inspection.

The emitted file contains:

```json
{
  "generated_at": "...",
  "sources": ["runs/battle_agent.jsonl", "..."],
  "overall": {...},
  "battle": {...},
  "overworld": {...}
}
```

The `overall/battle/overworld` sections mirror the `summarize_entries` output
(averaged latency, encode/query/skip fractions, gate view counts, etc.).

### CI / automation hint

To keep dashboards up to date, add the following line to your ingestion job:

```bash
python scripts/update_dashboard_metrics.py --pretty
```

This command is idempotent and can be run whenever new telemetry JSONL files
are produced.

## 2. Normalise legacy JSONL archives

Legacy logs recorded before the schema consolidation can be migrated using
`scripts/normalize_telemetry_logs.py`. The tool understands both the battle
and overworld legacy shapes and rewrites them into the new structure so all
analytics run on a consistent format.

### Mirror into a new directory

```bash
python scripts/normalize_telemetry_logs.py \
    --input /path/to/legacy_battle_logs \
    --output /path/to/normalized_logs
```

Each `.jsonl` file is processed via `normalize_entry`, a suffix is appended
(`.normalized` by default), and the directory structure is preserved under
`--output`.

### In-place conversion

If you prefer to overwrite the originals:

```bash
python scripts/normalize_telemetry_logs.py \
    --input /path/to/legacy_logs/*.jsonl \
    --in-place
```

The script writes to a temporary file and then replaces the source to avoid
partial writes.

After normalising archives you can delete the flat-schema copies and rely on
`scripts/update_dashboard_metrics.py` (or `scripts/summarize_telemetry.py`) for
future aggregation.

## Related helpers

- `scripts/summarize_telemetry.py` – interactive CLI for quick summaries of
  specific logs or domains.
- `src/telemetry/parsing.py` – normalisation utilities used by all the scripts.
- `src/telemetry/analytics.py` – metric aggregation helpers.

