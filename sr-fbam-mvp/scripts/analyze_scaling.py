"""
Analyze SR-FBAM scaling with respect to hop depth.

Given a results JSON (from eval mode with teacher-forcing losses), compute
accuracy and mean loss by symbolic plan length, fit a power law, and plot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_by_plan_length(entries: List[Dict]) -> Dict[int, Dict[str, float]]:
    buckets: Dict[int, Dict[str, float]] = {}
    for entry in entries:
        plan_len = entry.get("plan_length", 0)
        if plan_len <= 0:
            continue
        bucket = buckets.setdefault(plan_len, {"losses": [], "correct": 0, "total": 0})
        bucket["losses"].append(entry.get("teacher_forcing_loss", entry.get("final_loss", 0.0)))
        if entry.get("correct"):
            bucket["correct"] += 1
        bucket["total"] += 1
    summary: Dict[int, Dict[str, float]] = {}
    for plan_len, stats in sorted(buckets.items()):
        losses = np.array(stats["losses"], dtype=float)
        total = max(stats["total"], 1)
        summary[plan_len] = {
            "mean_loss": float(losses.mean()) if losses.size else 0.0,
            "std_loss": float(losses.std()) if losses.size else 0.0,
            "accuracy": stats["correct"] / total,
            "count": total,
        }
    return summary


def fit_power_law(summary: Dict[int, Dict[str, float]]) -> float:
    hop_lengths = np.array(list(summary.keys()), dtype=float)
    losses = np.array([summary[h]["mean_loss"] for h in hop_lengths], dtype=float)
    mask = (hop_lengths > 0) & (losses > 0)
    hop_lengths = hop_lengths[mask]
    losses = losses[mask]
    if hop_lengths.size < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(hop_lengths), np.log(losses), 1)
    slope = coeffs[0]
    return slope  # slope should be negative: loss  A * H^slope


def plot(summary: Dict[int, Dict[str, float]], slope: float, output_dir: Path) -> None:
    hop_lengths = np.array(list(summary.keys()), dtype=float)
    losses = np.array([summary[h]["mean_loss"] for h in hop_lengths], dtype=float)
    accuracies = np.array([summary[h]["accuracy"] for h in hop_lengths], dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.loglog(hop_lengths, losses, marker="o")
    plt.xlabel("Hop depth (H)")
    plt.ylabel("Mean teacher-forcing loss")
    if not np.isnan(slope):
        plt.title(f"Loss vs hop depth (slope={slope:.3f})")
    else:
        plt.title("Loss vs hop depth")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_vs_hop_depth.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(hop_lengths, accuracies, marker="o")
    plt.xlabel("Hop depth (H)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_hop_depth.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SR-FBAM hop-depth scaling")
    parser.add_argument("--results", required=True, help="Path to JSON results (sr_fbam_eval.json)")
    parser.add_argument("--output-dir", default="plots/scaling", help="Directory for plots/summary")
    args = parser.parse_args()

    results_path = Path(args.results)
    entries = load_results(results_path)
    summary = aggregate_by_plan_length(entries)
    slope = fit_power_law(summary)

    output_dir = Path(args.output_dir)
    plot(summary, slope, output_dir)

    lines = ["Hop Depth | Count | Accuracy | Mean Loss | Std Loss"]
    lines.append("-" * 60)
    for hop, stats in sorted(summary.items()):
        lines.append(
            f"{hop:9d} | {stats['count']:5d} | {stats['accuracy']*100:7.2f}% "
            f"| {stats['mean_loss']:.4f} | {stats['std_loss']:.4f}"
        )
    if not np.isnan(slope):
        lines.append(f"\nFitted power-law slope (log-loss vs log-hop): {slope:.4f}")
        lines.append(f"Estimated alpha (loss  A * H^-alpha): {-slope:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "scaling_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved plots and summary to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
