# SR-FBAM Code-Editing Benchmark

This repository now contains exactly the assets we rely on for two things:
1. Reproducing the SR-FBAM experiments described in the paper draft (Sections 5-7).
2. Running the LLM-assisted Pokemon Blue demo powered by the SR-FBAM gating stack.

Legacy data-pipeline automation, planning notes, and exploratory utilities have been
removed to keep the tree focused on these deliverables.

## Current status
- **Code editing**: SR-FBAM, baselines, and plotting scripts reproduce the paper experiments (see `results/summary/`).
- **Pokemon battle agent**: end-to-end inference pipeline (PyBoy -> RAM extractor -> symbolic graph -> SR-FBAM controller) with JSONL telemetry logging and per-gate profiling; offline training / replay ingestion are planned but not yet implemented.
- **Documentation**: environment setup lives in `docs/ENVIRONMENT_SETUP.md`; battle architecture and telemetry schema are described in `docs/SRFBAM_BATTLE_SYSTEM.md`, `docs/telemetry_schema.json`, and the telemetry pipeline guide (`docs/TELEMETRY_PIPELINE.md`).

## Repository layout

```
sr-fbam-mvp/
+-- data/
    +-- episodes_50/          # Synthetic 50-frame training/eval episodes (JSONL)
    +-- git_episodes/         # 65-action Git commit evaluation set (JSONL)
+-- results/summary/          # JSON summaries used in the paper tables/figures
+-- scripts/                  # Data generation, evaluation, plotting, and Pokemon helpers
+-- src/                      # Model implementations and training scripts
+-- docs/                     # Architecture notes (e.g., SR-FBAM battle system)
```

## Quick start (paper experiments)

1. **Install dependencies (conda-first)**
   ```bash
   conda create -n srfbam python=3.10
   conda activate srfbam
   # Ensure CUDA/driver versions line up with the PyTorch build you want.
   pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
   conda install -c conda-forge faiss-gpu cudatoolkit=12.4
   pip install -r sr-fbam-mvp/requirements.txt
   ```
   If you must avoid conda, build FAISS with CUDA manually or use a Linux/WSL
   environment where the official `faiss-gpu` wheels are published. The project
   assumes GPU FAISS is available for the sparse-memory baselines.

2. **Generate additional episode lengths** (optional)
   ```bash
   cd sr-fbam-mvp
   python scripts/generate_frame_action_episodes.py --steps 200 \
       --train 0 --eval 5 --output-dir data/episodes_200
   python scripts/build_long_eval.py --base-dir data/episodes_50 \
       --output-dir data/episodes_1000 --episodes 5
   ```
   The paper uses 50/100/200/338/1000-frame splits. We ship the 50-frame split to
   keep the repo light; the commands above re-create the larger ones.

3. **Evaluate trained models**
   ```bash
   python -m src.training.train_fbam_code --data-dir data/episodes_50 --epochs 3 --metrics-out results/summary/fbam_seed0.json
   python -m src.training.train_fbam_soft --data-dir data/episodes_50 --epochs 3 --metrics-out results/summary/fbam_soft_seed0.json
   python -m src.training.train_fbam_sparse --data-dir data/episodes_50 --epochs 3 --k-neighbors 10 --checkpoint-out checkpoints/fbam_sparse_k10_seed0.pt --metrics-out results/summary/fbam_sparse_k10_seed0.json
   python -m src.training.train_srfbam_code --data-dir data/episodes_50 --epochs 3 --metrics-out results/summary/srfbam_seed0.json
   ```
   (Pre-computed summaries used in the paper live in `results/summary/`).

4. **Recreate figures**
   ```bash
   python scripts/plot_scaling.py
   python scripts/plot_commit_types.py
   python scripts/entity_extraction_audit.py --samples 30 --seed 0
   ```

## Pokemon Blue integration

### Overworld / LLM controller

1. `pip install -r requirements.txt` inside `sr-fbam-mvp/` (PyBoy, torch, etc.).
2. Provide a Game Boy ROM (`pokemonblue/Pokemon Blue.gb` by default) and an OpenAI-compatible API key.
3. Launch the controller:
   ```bash
   python scripts/run_pokemon_controller.py --rom ../pokemonblue/Pokemon\ Blue.gb --steps 2000
   ```
   or call the LLM-backed driver directly with `python scripts/llm_pokemon_controller.py --rom ...`.

Supporting utilities (`debug_intro.py`, `debug_movement.py`, `inspect_pyboy_memory.py`, etc.)
remain available for targeted debugging while iterating on the demo.

### SR-FBAM battle agent

The battle pipeline mirrors the SR-FBAM architecture:

```
PyBoy -> PyBoyPokemonAdapter.read_u8
     -> BlueRAMAdapter (RAM -> BattleObs) -> BlueExtractor (BattleObs -> WriteOps)
     -> GraphMemory (assoc/follow/write + hop trace)
     -> SRFBAMBattleAgent (SRFBAMCore + BattleControllerPolicy)
     -> BattleActionHead (masked logits) -> PyBoy button scripts
```

1. Ensure the ROM path is correct and PyBoy can boot into the battle state.
2. Run the battle agent harness (log file is required):
   ```bash
   python scripts/run_battle_agent.py --rom ../pokemonblue/"Pokemon Blue.gb" \
       --steps 200 --log-file runs/battle_agent.jsonl --profile
   ```
   - Each step appends a JSONL record using the consolidated schema (`telemetry.core`, plus `telemetry.battle` for battle-specific fields). See `docs/telemetry_schema.json` for exact field names.
   - Console output mirrors the one-line summary, including running p50/p95 latency, action, gate mode, and fallback status.
   - To materialise dashboard summaries, run `python scripts/update_dashboard_metrics.py --pretty` (aggregates battle + overworld logs and writes `results/summary/telemetry_dashboard.json`).
   - For ad-hoc inspection you can also call `python scripts/summarize_telemetry.py --input runs/battle_agent.jsonl --domain battle --pretty`.
   - The PyBoy adapter executes real moves and switches via deterministic button scripts (Fight menu navigation or party selection).
3. Inspect the log file or console output to trace gate usage (`ASSOC/FOLLOW/WRITE/HALT`), latency, and symbolic hops.

The default RAM map covers active Pokemon, party slots, and basic field effects. Adjust `src/pkmn_battle/env/blue_ram_map.py`
if you ingest a different ROM variant.

## Key scripts

- `generate_frame_action_episodes.py` - builds synthetic episodes with configurable lengths.
- `build_long_eval.py` - repeats episodes to reach longer horizons (1000 frames in the paper).
- `evaluate_code_model.py` - lightweight GPU/CPU inference driver (supports --model fbam_sparse and sparse-memory flags).
- `run_sparse_grid.py` - launches sparse-memory sweeps across k values and seeds.
- `plot_scaling.py`, `plot_commit_types.py` - recreate paper figures.
- `compute_significance.py`, `entity_extraction_audit.py` - statistical tests and parser audit.
- `summarize_telemetry.py` - quick telemetry summaries for specific logs or domains.
- `update_dashboard_metrics.py` - generates consolidated telemetry summaries for dashboards (`results/summary/telemetry_dashboard.json` by default).
- `normalize_telemetry_logs.py` - batch normalises legacy telemetry JSONL files into the consolidated schema.
- `llm_pokemon_controller.py`, `run_pokemon_controller.py` - end-to-end Pokemon Blue controllers using SR-FBAM gating.
- `run_battle_agent.py` - SR-FBAM battle agent harness that emits JSONL telemetry (see `docs/telemetry_schema.json`).

Additional architectural notes live in `docs/SRFBAM_BATTLE_SYSTEM.md`.

## Models

- `src/models/sr_fbam_code.py` - SR-FBAM with symbolic memory for code editing (main model).
- `src/models/fbam_code_baseline.py` - frame-based baseline without symbolic memory.
- `src/models/fbam_soft_memory.py` - dense soft-attention external memory baseline.
- `src/models/fbam_sparse_memory.py` - FAISS-backed sparse retrieval memory baseline.

Training entry points live in `src/training/` with CPU defaults pointing at
`data/episodes_50`.

## Citation

If you find this code helpful, please cite the accompanying paper draft.

