#!/usr/bin/env python3
"""
Compute paired t-test and confidence intervals for speedup between FBAM and SR-FBAM.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List


def t_test_paired(x: List[float], y: List[float]) -> tuple[float, float]:
    """Return t-statistic and p-value (if SciPy available, else NaN)."""
    diffs = [a - b for a, b in zip(x, y)]
    mean_diff = sum(diffs) / len(diffs)
    var = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
    std = math.sqrt(var)
    t_stat = mean_diff / (std / math.sqrt(len(diffs)))
    try:
        from scipy import stats  # type: ignore

        p_val = 2 * stats.t.sf(abs(t_stat), df=len(diffs) - 1)
    except Exception:
        p_val = float("nan")
    return t_stat, p_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute significance for speedup.")
    parser.add_argument("--summary", type=Path, default=Path("results/real_git/summary.json"))
    parser.add_argument("--model-a", default="fbam")
    parser.add_argument("--model-b", default="srfbam")
    parser.add_argument("--label-a", default="FBAM")
    parser.add_argument("--label-b", default="SR-FBAM")
    parser.add_argument("--cf", type=float, default=None, help="Optional frame time override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text())
    a_times = data[args.model_a]["wall_time_ms"]
    b_times = data[args.model_b]["wall_time_ms"]
    t_stat, p_val = t_test_paired(a_times, b_times)

    speedups = [a / b for a, b in zip(a_times, b_times)]
    mean_speed = sum(speedups) / len(speedups)
    std_speed = math.sqrt(sum((x - mean_speed) ** 2 for x in speedups) / (len(speedups) - 1))
    ci_low = mean_speed - 2.776 * std_speed / math.sqrt(len(speedups))  # t_{0.975,4}
    ci_high = mean_speed + 2.776 * std_speed / math.sqrt(len(speedups))

    print(f"{args.label_a} vs {args.label_b} (n={len(a_times)} seeds)")
    print(f"  Paired t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_val:.4g}")
    print(f"  Speedup mean +/- std: {mean_speed:.2f} +/- {std_speed:.2f}x")
    print(f"  95% CI: [{ci_low:.2f}x, {ci_high:.2f}x]")


if __name__ == "__main__":
    main()
