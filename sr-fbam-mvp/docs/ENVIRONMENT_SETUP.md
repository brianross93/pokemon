# Environment Setup (Conda-First)

To run either the paper replication or the Pokémon pipelines you need a CUDA-capable
environment with FAISS GPU enabled. The recommended workflow is:

```bash
conda create -n srfbam python=3.10
conda activate srfbam

# Install a CUDA-enabled PyTorch build compatible with your drivers.
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Pull in FAISS GPU (bundles the CUDA runtime when installed from conda-forge).
conda install -c conda-forge faiss-gpu cudatoolkit=12.4

# Finish with the project requirements (PyBoy, telemetry tooling, etc.).
pip install -r requirements.txt
```

If you prefer another CUDA toolchain, swap the PyTorch wheel and matching
`cudatoolkit` version accordingly, but keep FAISS from conda-forge to avoid
building it manually. For WSL/Linux the steps are similar; native Windows pip
wheels for `faiss-gpu` are not published.

### Downloaded artifacts

Running the ingestion helpers produces:

- `data/mechanics/` &mdash; static Pokémon Showdown tables (`pokedex.json`,
  `moves.json`, `abilities.js`, `items.js`, `typechart.js`, `formats*.js`)
  fetched via `scripts/download_mechanics_data.py`.
- `data/raw/metamon/gen9ou_demo_0001.jsonl.lz4` &mdash; a tiny synthetic Metamon
  shard emitted by `scripts/download_metamon_replays.py --demo`. Replace this
  with real shards once you stream the official `gen9ou.tar.gz` archive from the
  dataset.

Use these as the starting point for the SR-FBAM imitation-learning pipeline
(`scripts/build_il_corpus.py`).
