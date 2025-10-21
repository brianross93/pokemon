"""
Generate scaling plots for FBAM vs SR-FBAM experiments.
Loads summary JSON files and produces wall-time and accuracy figures.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

SUMMARY_BY_LENGTH = Path("results/summary/synthetic_lengths_summary.json")
SUMMARY_SYNTHETIC = Path("results/summary/real_and_synthetic.json")


@dataclass
class ScalingPoint:
    length: int
    fbam_time_mean: float
    fbam_time_std: float
    srfbam_time_mean: float
    srfbam_time_std: float
    fbam_acc_mean: float
    fbam_acc_std: float
    srfbam_acc_mean: float
    srfbam_acc_std: float
    speedup_mean: float
    speedup_std: float


def load_scaling_points() -> List[ScalingPoint]:
    points: List[ScalingPoint] = []

    if SUMMARY_BY_LENGTH.exists():
        with SUMMARY_BY_LENGTH.open("r", encoding="utf-8") as fin:
            length_summary = json.load(fin)
        for key, metrics in length_summary.items():
            length = int(key)
            fbam = metrics["fbam"]
            srfbam = metrics["srfbam"]
            points.append(
                ScalingPoint(
                    length=length,
                    fbam_time_mean=fbam["wall_time_ms_mean"],
                    fbam_time_std=fbam["wall_time_ms_std"],
                    srfbam_time_mean=srfbam["wall_time_ms_mean"],
                    srfbam_time_std=srfbam["wall_time_ms_std"],
                    fbam_acc_mean=fbam["accuracy_mean"],
                    fbam_acc_std=fbam["accuracy_std"],
                    srfbam_acc_mean=srfbam["accuracy_mean"],
                    srfbam_acc_std=srfbam["accuracy_std"],
                    speedup_mean=metrics["speedup_mean"],
                    speedup_std=metrics["speedup_std"],
                )
            )

    lengths_present = {point.length for point in points}

    if SUMMARY_SYNTHETIC.exists() and 338 not in lengths_present:
        with SUMMARY_SYNTHETIC.open("r", encoding="utf-8") as fin:
            combined = json.load(fin)
        synthetic = combined.get("synthetic", {})
        fbam_times = np.array(synthetic.get("fbam", {}).get("wall_time_ms", []), dtype=float)
        srfbam_times = np.array(synthetic.get("srfbam", {}).get("wall_time_ms", []), dtype=float)
        fbam_acc = np.array(synthetic.get("fbam", {}).get("accuracy", []), dtype=float)
        srfbam_acc = np.array(synthetic.get("srfbam", {}).get("accuracy", []), dtype=float)

        if fbam_times.size and srfbam_times.size and fbam_acc.size and srfbam_acc.size:
            speedup_samples = fbam_times / srfbam_times
            points.append(
                ScalingPoint(
                    length=338,
                    fbam_time_mean=float(np.mean(fbam_times)),
                    fbam_time_std=float(np.std(fbam_times, ddof=0)),
                    srfbam_time_mean=float(np.mean(srfbam_times)),
                    srfbam_time_std=float(np.std(srfbam_times, ddof=0)),
                    fbam_acc_mean=float(np.mean(fbam_acc)),
                    fbam_acc_std=float(np.std(fbam_acc, ddof=0)),
                    srfbam_acc_mean=float(np.mean(srfbam_acc)),
                    srfbam_acc_std=float(np.std(srfbam_acc, ddof=0)),
                    speedup_mean=float(np.mean(speedup_samples)),
                    speedup_std=float(np.std(speedup_samples, ddof=0)),
                )
            )

    points.sort(key=lambda p: p.length)
    return points


SCALING_POINTS = load_scaling_points()
if not SCALING_POINTS:
    raise FileNotFoundError(
        "No scaling summary data found. Expected "
        f"{SUMMARY_BY_LENGTH} or {SUMMARY_SYNTHETIC} to exist."
    )

episode_lengths = [p.length for p in SCALING_POINTS]
fbam_times = [p.fbam_time_mean for p in SCALING_POINTS]
srfbam_times = [p.srfbam_time_mean for p in SCALING_POINTS]
speedups = [p.speedup_mean for p in SCALING_POINTS]


def plot_walltime_scaling(output_dir: Path):
    """Main result: wall-time scaling curves with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fbam_std = [p.fbam_time_std for p in SCALING_POINTS]
    srfbam_std = [p.srfbam_time_std for p in SCALING_POINTS]

    ax.errorbar(
        episode_lengths,
        fbam_times,
        yerr=fbam_std,
        fmt='o-',
        linewidth=2,
        markersize=8,
        label='Pure FBAM',
        color='#d62728',
        capsize=4,
    )
    ax.errorbar(
        episode_lengths,
        srfbam_times,
        yerr=srfbam_std,
        fmt='s-',
        linewidth=2,
        markersize=8,
        label='SR-FBAM (ours)',
        color='#2ca02c',
        capsize=4,
    )

    # Fit linear trends
    fbam_fit = np.polyfit(episode_lengths, fbam_times, 1)
    srfbam_fit = np.polyfit(episode_lengths, srfbam_times, 1)

    x_range = np.linspace(0, max(episode_lengths) + 50, 200)
    fbam_trend = fbam_fit[0] * x_range + fbam_fit[1]
    srfbam_trend = srfbam_fit[0] * x_range + srfbam_fit[1]

    ax.plot(
        x_range,
        fbam_trend,
        '--',
        alpha=0.5,
        color='#d62728',
        label=f'FBAM trend: {fbam_fit[0]:.2f}n + {fbam_fit[1]:.1f} ms',
    )
    ax.plot(
        x_range,
        srfbam_trend,
        '--',
        alpha=0.5,
        color='#2ca02c',
        label=f'SR-FBAM trend: {srfbam_fit[0]:.2f}n + {srfbam_fit[1]:.1f} ms',
    )

    for length, fbam_val, speedup in zip(episode_lengths, fbam_times, speedups):
        ax.annotate(
            f'{speedup:.2f}x',
            xy=(length, fbam_val),
            xytext=(length + 10, fbam_val + 20),
            fontsize=9,
            color='#1f77b4',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1),
        )

    ax.set_xlabel('Episode length (frames)', fontsize=12)
    ax.set_ylabel('Wall time (ms)', fontsize=12)
    ax.set_title('Wall-time scaling: FBAM vs SR-FBAM', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    avg_speedup = float(np.mean(speedups))
    ax.text(
        0.98,
        0.02,
        f'Average speedup: {avg_speedup:.2f}x',
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        ha='right',
        va='bottom',
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'walltime_scaling.png', dpi=150)
    plt.savefig(output_dir / 'walltime_scaling.pdf')
    plt.close()
    print("Saved: walltime_scaling.png")


def plot_accuracy_comparison(output_dir: Path):
    """Accuracy across episode lengths with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fbam_acc_pct = [p.fbam_acc_mean * 100 for p in SCALING_POINTS]
    srfbam_acc_pct = [p.srfbam_acc_mean * 100 for p in SCALING_POINTS]
    fbam_acc_std_pct = [p.fbam_acc_std * 100 for p in SCALING_POINTS]
    srfbam_acc_std_pct = [p.srfbam_acc_std * 100 for p in SCALING_POINTS]

    ax.errorbar(
        episode_lengths,
        fbam_acc_pct,
        yerr=fbam_acc_std_pct,
        fmt='o-',
        linewidth=2,
        markersize=8,
        label='Pure FBAM',
        color='#d62728',
        capsize=4,
    )
    ax.errorbar(
        episode_lengths,
        srfbam_acc_pct,
        yerr=srfbam_acc_std_pct,
        fmt='s-',
        linewidth=2,
        markersize=8,
        label='SR-FBAM (ours)',
        color='#2ca02c',
        capsize=4,
    )

    ax.set_xlabel('Episode length (frames)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Action prediction accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.axhline(float(np.mean(fbam_acc_pct)), color='#d62728', linestyle=':', alpha=0.5)
    ax.axhline(float(np.mean(srfbam_acc_pct)), color='#2ca02c', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()
    print("Saved: accuracy_comparison.png")


def write_summary(output_dir: Path) -> None:
    avg_speedup = float(np.mean(speedups))

    with open(output_dir / 'scaling_summary.txt', 'w', encoding='utf-8') as f:
        f.write("SR-FBAM Scaling Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Episode length experiments:\n")
        for point in SCALING_POINTS:
            f.write(
                f"  {point.length:>4d} frames: {point.speedup_mean:.2f}x +/- "
                f"{point.speedup_std:.2f}x\n"
            )
            f.write(
                f"    FBAM: {point.fbam_time_mean:.1f}ms +/- {point.fbam_time_std:.1f}ms, "
                f"{point.fbam_acc_mean:.1%} +/- {point.fbam_acc_std:.1%}\n"
            )
            f.write(
                f"    SR-FBAM: {point.srfbam_time_mean:.1f}ms +/- {point.srfbam_time_std:.1f}ms, "
                f"{point.srfbam_acc_mean:.1%} +/- {point.srfbam_acc_std:.1%}\n"
            )
        f.write(f"\nAverage speedup: {avg_speedup:.2f}x\n\n")

        fbam_fit = np.polyfit(episode_lengths, fbam_times, 1)
        srfbam_fit = np.polyfit(episode_lengths, srfbam_times, 1)
        f.write("Scaling coefficients (linear fit):\n")
        f.write(f"  FBAM: {fbam_fit[0]:.2f}n + {fbam_fit[1]:.1f} ms\n")
        f.write(f"  SR-FBAM: {srfbam_fit[0]:.2f}n + {srfbam_fit[1]:.1f} ms\n")


def main():
    output_dir = Path('plots/scaling')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating scaling plots...")
    plot_walltime_scaling(output_dir)
    plot_accuracy_comparison(output_dir)
    write_summary(output_dir)

    print(f"\nAll plots saved to: {output_dir.resolve()}")
    print("  - walltime_scaling.png (main result)")
    print("  - accuracy_comparison.png")
    print("  - walltime_scaling.pdf")
    print("  - scaling_summary.txt")


if __name__ == '__main__':
    main()
