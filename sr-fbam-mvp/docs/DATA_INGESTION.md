# SR‑FBAM Data Ingestion Checklist

This guide captures the three staged ingestion process needed before RL fine
tuning:

1. **Mechanics tables** – mirror the static Pokémon Showdown data.
2. **Expert trajectories** – download Metamon parsed replay shards.
3. **JSONL emission** – transform shards into the SR‑FBAM frame/action schema.

All commands assume the repository root (`sr-fbam-mvp/`).

## 1. Mirror mechanics files

Use `scripts/download_mechanics_data.py` to pull the canonical simulator data.
The default list matches the minimum required to seed the symbolic entity graph.

```bash
python scripts/download_mechanics_data.py \
  --dest data/mechanics
```

The script supports additional files via `--files` or a manifest JSON, and an
alternate `--base-url` if you want to pin to a specific Showdown commit mirror.

## 2. Fetch Metamon replay shards

`scripts/download_metamon_replays.py` wraps the Hugging Face API and filters
shards by format / Elo. By default it downloads high-Elo `[Gen 9] OU` games into
`data/raw/metamon/`.

```bash
python scripts/download_metamon_replays.py \
  --format gen9ou \
  --dest data/raw/metamon \
  --max-files 4
```

For quick smoke tests without downloading the multi-gigabyte archive, append
`--demo` to generate a tiny synthetic shard.

```bash
python scripts/download_metamon_replays.py --dest data/raw/metamon --demo
```

Set `--revision` to pin an exact dataset snapshot or `--filename` if the archive
name differs from `<format>.tar.gz`.

## 3. Emit SR‑FBAM training JSONL

The processing stage converts each Metamon turn into the SR‑FBAM imitation
learning schema. The starter script `scripts/build_il_corpus.py` (see below)
wire-ups the CLI, while the actual transformation logic lives in
`src/pkmn_battle/ingest/pipeline.py`.

The target row shape (one decision per line) is:

```json
{
  "battle_id": "gen9ou-1234567890",
  "turn_idx": 37,
  "format": "gen9ou",
  "p1_elo": 1810,
  "p2_elo": 1765,
  "frame": { "grid_40x120": ["...40 rows of text..."] },
  "options": {
    "moves": [{"id":"hydropump","target":"foe-active"}, {"id":"icebeam"}],
    "switches": [{"species":"Dondozo"}, {"species":"Glowking"}],
    "tera_available": true
  },
  "action_label": {"type":"MOVE","id":"icebeam","target":"foe-active","tera":false},
  "graph_updates": [
    {"op":"WRITE","add_node":{"type":"field","id":"snow"}},
    {"op":"WRITE","add_edge":{"from":"hazards","rel":"spikes","to":"p2"}}
  ],
  "revealed": {
    "p1_active": {"species":"Greninja","hp_pct":63,"status":null,"boosts":{"spa":1}},
    "p2_active": {"species":"Landorus-Therian","hp_pct":41,"status":"brn"}
  },
  "log_events": [
    "|-status|p2a: Landorus-Therian|brn",
    "|move|p1a: Greninja|Ice Beam|p2a: Landorus-Therian"
  ]
}
```

### Skeleton CLI

```bash
python scripts/build_il_corpus.py \
  --mechanics data/mechanics \
  --source data/raw/metamon \
  --output data/processed/il_gen9ou_train.jsonl
```

`build_il_corpus.py` handles CLI orchestration (directory discovery, batching,
job metadata). You can also point it at a manifest (`--manifest data/metamon_manifest.txt`)
containing explicit shard paths; this is the safest way to avoid globbing a
20GB directory by mistake.

### Battle dataset loader + dry-run training

The emitted JSONL files integrate with `BattleDecisionDataset`:

```python
from pathlib import Path
from pkmn_battle.ingest import BattleDecisionDataset

dataset = BattleDecisionDataset(Path("data/processed/il_gen9ou_train.jsonl"))
frame_tensor, action_index = dataset[0]
print(dataset.index_to_action[action_index])
```

For a quick CPU-only smoke test that the schema flows through the model stack:

```bash
python -m training.train_battle_il \
  --train data/processed/il_gen9ou_train.jsonl \
  --val data/processed/il_gen9ou_val.jsonl \
  --epochs 2 \
  --batch-size 32 \
  --lr 1e-3 \
  --hidden-dim 256 \
  --metrics-out results/summary/battle_il_metrics.json
```

This uses a lightweight MLP to verify batching, action vocab construction, and
loss computation end-to-end.

### Test harness

`tests/test_ingest_pipeline.py` includes a placeholder test showing how to feed
a tiny fixture shard into the converter. Once the transformation logic is in
place, extend the fixture and tighten the assertions (options coverage, graph
updates, frame rendering, etc.).

---

With all three steps operational you can train the BC baseline (IL) and then
warm-start PPO/A3C fine tuning on the cleaned dataset splits.
