"""
Visualization tooling for SR-FBAM experiment results.

Generates plots and summary statistics from results JSON logs, providing
assets for Sections 6–8 of the whitepaper. Covers loss curves, accuracy,
hop distributions, wall-time, action usage, and confidence trajectories.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path: Path) -> List[Dict]:
    """Load query logs from JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curve(queries: Sequence[Dict], output_dir: Path) -> None:
    losses = [q["final_loss"] for q in queries]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, alpha=0.6, label="Per-query loss")

    window = min(20, max(len(losses) // 10, 1))
    if window > 1:
        kernel = np.ones(window) / window
        moving = np.convolve(losses, kernel, mode="valid")
        ax.plot(range(window - 1, len(losses)), moving, linewidth=2, label=f"Moving avg (window={window})")

    ax.set_xlabel("Query index")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = output_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {output_path.name}")


def plot_accuracy(queries: Sequence[Dict], output_dir: Path) -> None:
    correct = np.array([1 if q["correct"] else 0 for q in queries], dtype=float)
    cumulative = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative, linewidth=2)
    ax.axhline(y=cumulative[-1], color="red", linestyle="--", label=f"Final accuracy {cumulative[-1]:.3f}")

    ax.set_xlabel("Query index")
    ax.set_ylabel("Cumulative accuracy")
    ax.set_title("Training accuracy trajectory")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = output_dir / "accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {output_path.name}")


def plot_hop_distribution(queries: Sequence[Dict], output_dir: Path) -> None:
    hop_counts = [q["total_hops"] for q in queries]
    bins = range(1, max(hop_counts) + 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(hop_counts, bins=bins, alpha=0.75, edgecolor="black")
    ax.set_xlabel("Number of hops")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Hop distribution (mean={np.mean(hop_counts):.1f})")
    ax.grid(True, alpha=0.3, axis="y")

    output_path = output_dir / "hop_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {output_path.name}")


def plot_wall_time(queries: Sequence[Dict], output_dir: Path) -> None:
    wall_times = [q["wall_time_ms"] for q in queries]

    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))
    ax_box.boxplot(wall_times, vert=True)
    ax_box.set_ylabel("Wall time (ms)")
    ax_box.set_title("Wall time distribution")
    ax_box.grid(True, alpha=0.3, axis="y")

    ax_hist.hist(wall_times, bins=30, alpha=0.75, edgecolor="black")
    ax_hist.set_xlabel("Wall time (ms)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(f"Wall time histogram (median={np.median(wall_times):.2f} ms)")
    ax_hist.grid(True, alpha=0.3, axis="y")

    output_path = output_dir / "wall_time.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {output_path.name}")


def plot_action_distribution(queries: Sequence[Dict], output_dir: Path) -> None:
    action_counts: Counter = Counter()
    for q in queries:
        for hop in q.get("hops", []):
            action_counts[hop["action"]] += 1

    if not action_counts:
        print("[WARN] No hop data detected; skipping action distribution plot.")
        return

    actions, counts = zip(*sorted(action_counts.items(), key=lambda x: x[0]))
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(actions, counts, alpha=0.8, edgecolor="black")
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")
    ax.set_title("Action distribution across hops")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha="center", va="bottom")

    output_path = output_dir / "action_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {output_path.name}")


def plot_confidence_trajectory(
    queries: Sequence[Dict],
    output_dir: Path,
    sample_size: int = 20,
) -> None:
    correct = [q for q in queries if q["correct"]]
    incorrect = [q for q in queries if not q["correct"]]

    sample_correct = correct[:sample_size]
    sample_incorrect = incorrect[:sample_size]

    fig, (ax_correct, ax_incorrect) = plt.subplots(1, 2, figsize=(14, 6))

    for q in sample_correct:
        confidences = [hop["confidence"] for hop in q.get("hops", [])]
        ax_correct.plot(range(1, len(confidences) + 1), confidences, alpha=0.3, color="green")
    ax_correct.set_title(f"Confidence trajectory (correct, n={len(sample_correct)})")
    ax_correct.set_xlabel("Hop number")
    ax_correct.set_ylabel("Confidence")
    ax_correct.set_ylim([0, 1])
    ax_correct.grid(True, alpha=0.3)

    for q in sample_incorrect:
        confidences = [hop["confidence"] for hop in q.get("hops", [])]
        ax_incorrect.plot(range(1, len(confidences) + 1), confidences, alpha=0.3, color="red")
    ax_incorrect.set_title(f"Confidence trajectory (incorrect, n={len(sample_incorrect)})")
    ax_incorrect.set_xlabel("Hop number")
    ax_incorrect.set_ylabel("Confidence")
    ax_incorrect.set_ylim([0, 1])
    ax_incorrect.grid(True, alpha=0.3)

    output_path = output_dir / "confidence_trajectory.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {output_path.name}")


def generate_summary_stats(queries: Sequence[Dict], output_path: Path) -> None:
    losses = np.array([q["final_loss"] for q in queries], dtype=float)
    wall_times = np.array([q["wall_time_ms"] for q in queries], dtype=float)
    hops = np.array([q["total_hops"] for q in queries], dtype=float)
    accuracy = sum(1 for q in queries if q["correct"]) / max(len(queries), 1)

    action_counts: Counter = Counter()
    for q in queries:
        for hop in q.get("hops", []):
            action_counts[hop["action"]] += 1

    summary_lines = [
        "SR-FBAM Training Summary",
        "========================",
        f"Total queries: {len(queries)}",
        f"Accuracy: {accuracy:.3f}",
        "",
        "Loss:",
        f"  Mean: {losses.mean():.4f}",
        f"  Median: {np.median(losses):.4f}",
        f"  Std: {losses.std():.4f}",
        "",
        "Wall time (ms):",
        f"  Mean: {wall_times.mean():.2f}",
        f"  Median: {np.median(wall_times):.2f}",
        f"  Min: {wall_times.min():.2f}",
        f"  Max: {wall_times.max():.2f}",
        "",
        "Hops:",
        f"  Mean: {hops.mean():.1f}",
        f"  Median: {np.median(hops):.0f}",
        f"  Mode: {Counter(hops).most_common(1)[0][0]:.0f}",
        f"  Max: {hops.max():.0f}",
        "",
        "Action distribution:",
    ]

    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_actions if total_actions else 0.0
        summary_lines.append(f"  {action}: {count} ({pct:.1f}%)")

    summary_text = "\n".join(summary_lines)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\n[OK] Saved {output_path.name}")
    print("\n" + summary_text)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate plots from SR-FBAM results JSON")
    parser.add_argument(
        "--results",
        default="results/sr_fbam_train.json",
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory for generated plots and summaries",
    )
    args = parser.parse_args(argv)

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_path}")
    queries = load_results(results_path)
    print(f"Loaded {len(queries)} queries\n")

    print("Generating plots...")
    plot_loss_curve(queries, output_dir)
    plot_accuracy(queries, output_dir)
    plot_hop_distribution(queries, output_dir)
    plot_wall_time(queries, output_dir)
    plot_action_distribution(queries, output_dir)
    plot_confidence_trajectory(queries, output_dir)
    generate_summary_stats(queries, output_dir / "summary.txt")
    print(f"\n[DONE] Outputs stored in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
