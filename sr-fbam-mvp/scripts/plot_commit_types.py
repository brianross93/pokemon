"""
Generate commit type breakdown visualizations based on summary statistics.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Measured statistics from Table 3
commit_types = ['Bug Fix', 'Feature\nAdd', 'Refactor', 'Other']
episodes_count = [5, 4, 2, 29]

speedup_means = [5.1, 5.0, 5.1, 5.2]
speedup_stds = [0.4, 0.3, 0.2, 0.5]

reuse_means = [98.2, 96.8, 98.5, 98.1]
reuse_stds = [0.6, 1.2, 0.5, 0.8]

fbam_times = [72, 95, 86, 84]
fbam_stds = [8, 10, 6, 9]
srfbam_times = [14, 19, 17, 16]
srfbam_stds = [1, 2, 1, 1]

fbam_acc = [68, 58, 59, 62]
fbam_acc_std = [5, 8, 3, 6]
srfbam_acc = [89, 78, 85, 82]
srfbam_acc_std = [3, 7, 4, 6]


def plot_speedup_by_type(output_dir: Path) -> None:
    """Speedup by commit type with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(commit_types))
    width = 0.55
    colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']

    bars = ax.bar(
        x,
        speedup_means,
        width,
        yerr=speedup_stds,
        capsize=6,
        color=colors,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8,
    )

    overall_mean = float(np.mean(speedup_means))
    ax.axhline(
        overall_mean,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Overall average: {overall_mean:.2f}x',
        alpha=0.7,
    )

    for idx, (bar, count, std) in enumerate(zip(bars, episodes_count, speedup_stds)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.2,
            f'{speedup_means[idx]:.1f}x\n+/-{std:.1f}x\n(n={count})',
            ha='center',
            fontsize=9,
            fontweight='bold',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(commit_types)
    ax.set_ylabel('Speedup factor (x)', fontsize=12)
    ax.set_xlabel('Commit Type', fontsize=12)
    ax.set_title('SR-FBAM Speedup by Commit Type (Real Git)', fontsize=14, fontweight='bold')
    ax.set_ylim([3, 7])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_by_commit_type.png', dpi=150)
    plt.close()
    print('Saved: speedup_by_commit_type.png')


def plot_reuse_by_type(output_dir: Path) -> None:
    """Entity reuse rate by commit type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(commit_types))
    width = 0.6
    colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']

    bars = ax.bar(
        x,
        reuse_means,
        width,
        yerr=reuse_stds,
        capsize=5,
        color=colors,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7,
    )

    for idx, (bar, std) in enumerate(zip(bars, reuse_stds)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.3,
            f'{reuse_means[idx]:.1f}%\n+/-{std:.1f}%',
            ha='center',
            fontsize=9,
        )

    ax.set_ylabel('Entity reuse rate (%)', fontsize=12)
    ax.set_xlabel('Commit Type', fontsize=12)
    ax.set_title('Entity Reuse Rate by Commit Type (Real Git)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(commit_types)
    ax.set_ylim([94, 100])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(98.1, color='red', linestyle='--', linewidth=2, label='Overall: 98.1%', alpha=0.7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'reuse_by_commit_type.png', dpi=150)
    plt.close()
    print('Saved: reuse_by_commit_type.png')


def plot_accuracy_by_type(output_dir: Path) -> None:
    """Accuracy comparison by commit type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(commit_types))
    width = 0.35

    ax.bar(
        x - width / 2,
        fbam_acc,
        width,
        yerr=fbam_acc_std,
        capsize=4,
        label='Pure FBAM',
        color='#ff9999',
        edgecolor='black',
        linewidth=1,
    )
    ax.bar(
        x + width / 2,
        srfbam_acc,
        width,
        yerr=srfbam_acc_std,
        capsize=4,
        label='SR-FBAM',
        color='#99ff99',
        edgecolor='black',
        linewidth=1,
    )

    for i, (f_val, s_val) in enumerate(zip(fbam_acc, srfbam_acc)):
        ax.text(i - width / 2, f_val + fbam_acc_std[i] + 2, f'{f_val}%', ha='center', fontsize=9)
        ax.text(i + width / 2, s_val + srfbam_acc_std[i] + 2, f'{s_val}%', ha='center', fontsize=9)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Commit Type', fontsize=12)
    ax.set_title('Accuracy by Commit Type: FBAM vs SR-FBAM (Real Git)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(commit_types)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_commit_type.png', dpi=150)
    plt.close()
    print('Saved: accuracy_by_commit_type.png')


def write_summary(output_dir: Path) -> None:
    with open(output_dir / 'commit_type_summary.txt', 'w', encoding='utf-8') as f:
        f.write('Commit Type Analysis Summary\n')
        f.write('=' * 50 + '\n\n')
        for idx, label in enumerate(['bugfix', 'feature', 'refactor', 'other']):
            f.write(f'{commit_types[idx]} ({label}):\n')
            f.write(f'  Episodes: {episodes_count[idx]}\n')
            f.write(f'  Entity reuse: {reuse_means[idx]:.1f}% +/- {reuse_stds[idx]:.1f}%\n')
            f.write(f'  Speedup: {speedup_means[idx]:.1f}x +/- {speedup_stds[idx]:.1f}x\n')
            f.write(
                f'  FBAM: {fbam_times[idx]}ms +/- {fbam_stds[idx]}ms, '
                f'{fbam_acc[idx]}% +/- {fbam_acc_std[idx]}%\n'
            )
            f.write(
                f'  SR-FBAM: {srfbam_times[idx]}ms +/- {srfbam_stds[idx]}ms, '
                f'{srfbam_acc[idx]}% +/- {srfbam_acc_std[idx]}%\n'
            )
            f.write('\n')
        f.write('Key insight: speedup remains consistent across commit types (5.0-5.2x).\n')
        f.write('Validates generalization to diverse editing patterns.\n')


def main() -> None:
    output_dir = Path('plots/commit_types')
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Generating commit type analysis plots...')
    plot_speedup_by_type(output_dir)
    plot_reuse_by_type(output_dir)
    plot_accuracy_by_type(output_dir)
    write_summary(output_dir)

    print(f'\nAll plots saved to: {output_dir.resolve()}')
    print('  - speedup_by_commit_type.png')
    print('  - reuse_by_commit_type.png')
    print('  - accuracy_by_commit_type.png')
    print('  - commit_type_summary.txt')


if __name__ == '__main__':
    main()
