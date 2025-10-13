# SR-FBAM Quick Start Guide

This guide shows how to train, evaluate, and interact with the Symbolic-Recurrence FBAM (SR-FBAM) minimum viable experiment.

## 1. Installation

```bash
git clone <repo-url>
cd sr-fbam-mvp
python -m venv .venv
source .venv/bin/activate      # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train SR-FBAM

```bash
python src/training/train.py \
    --mode train \
    --split train \
    --epochs 10 \
    --batch-size 8 \
    --device cpu \
    --checkpoint-dir checkpoints \
    --log-dir results
```

This command produces:
- `checkpoints/sr_fbam_train_best.pt` (best checkpoint)
- `checkpoints/sr_fbam_train_epoch*.pt` (per-epoch checkpoints)
- `results/sr_fbam_train.json` (per-query logs)

## 3. Evaluate on Held-Out Data

```bash
python src/training/train.py \
    --mode eval \
    --checkpoint checkpoints/sr_fbam_train_best.pt \
    --split eval \
    --log-dir results
```

Evaluation logs are written to `results/sr_fbam_eval.json`.

## 4. Train the LSTM Baseline

```bash
python src/training/train_lstm.py \
    --mode train \
    --split train \
    --epochs 10 \
    --batch-size 8 \
    --device cpu \
    --checkpoint-dir checkpoints \
    --log-dir results \
    --experiment-name lstm_train
```

Evaluate the baseline:

```bash
python src/training/train_lstm.py \
    --mode eval \
    --checkpoint checkpoints/lstm_train_best.pt \
    --split eval \
    --device cpu \
    --experiment-name lstm_eval \
    --log-dir results
```

## 5. Train the Transformer Baseline

```bash
python src/training/train_transformer.py \
    --mode train \
    --split train \
    --epochs 10 \
    --batch-size 8 \
    --device cpu \
    --checkpoint-dir checkpoints \
    --log-dir results \
    --experiment-name transformer_train
```

Evaluate:

```bash
python src/training/train_transformer.py \
    --mode eval \
    --checkpoint checkpoints/transformer_train_best.pt \
    --split eval \
    --device cpu \
    --experiment-name transformer_eval \
    --log-dir results
```

## 6. Train the FBAM Baseline

```bash
python src/training/train_fbam.py \
    --mode train \
    --split train \
    --epochs 10 \
    --batch-size 8 \
    --device cpu \
    --checkpoint-dir checkpoints \
    --log-dir results \
    --experiment-name fbam_train
```

Evaluate:

```bash
python src/training/train_fbam.py \
    --mode eval \
    --checkpoint checkpoints/fbam_train_best.pt \
    --split eval \
    --device cpu \
    --experiment-name fbam_eval \
    --log-dir results
```

## 7. Run Stress Tests

```bash
for variant in ambiguous_nodes long_chains noisy_edges missing_nodes graph_growth; do
    python src/training/train.py \
        --mode eval \
        --checkpoint checkpoints/sr_fbam_train_best.pt \
        --variant "$variant" \
        --split eval \
        --experiment-name "sr_fbam_${variant}" \
        --log-dir results
done
```

## 8. Analyze Hop-Depth Scaling

```bash
python scripts/analyze_scaling.py \
    --results results/sr_fbam_eval.json \
    --output-dir plots/scaling
```

This prints accuracy and loss by hop depth, fits a power law, and saves plots/summary under `plots/scaling/`.

## 9. Generate Plots

```bash
python scripts/plot_results.py \
    --results results/sr_fbam_train.json \
    --output-dir plots/train

python scripts/plot_results.py \
    --results results/sr_fbam_eval.json \
    --output-dir plots/eval
```

The script emits PNG plots (loss, accuracy, hop distribution, wall-time, action usage, confidence trajectories) and a `summary.txt`.

## 10. Interactive Demo

```bash
python scripts/demo.py \
    --checkpoint checkpoints/sr_fbam_train_best.pt \
    --data-dir data
```

Follow the prompts to inspect sample queries and view hop-by-hop traces.

## 11. Notebook Exploration

Open `notebooks/02_demo_inference.ipynb` in Jupyter Lab or VS Code to run queries and visualise reasoning graphs interactively.

## Expected Metrics (Train Split, 10 epochs)

| Metric          | Value    |
|-----------------|----------|
| Accuracy        | ~0.64    |
| Mean hops       | ~2.8     |
| Mean wall-time  | ~4.3 ms |
| HALT usage      | ~35%     |

> Note: Metrics on the evaluation split differ slightly but should be close when trained for 10 epochs.

## Citation

If you use SR-FBAM in academic work, please cite the accompanying whitepaper draft once available.
