"""Utility to run the three SR-FBAM verification passes on code-editing episodes.

Pass A — Smoke tests / ablations:
    * SR-FBAM learned gate (default configuration)
    * SR-FBAM with gate forced to ENCODE
    * SR-FBAM with memory disabled (FBAM baseline)

Pass B — Entity-reuse vs. speedup:
    Reads the metrics from Pass A runs and reports e/q/s fractions together with
    predicted vs. observed speedups, checking that the analytic model matches the
    measured latency within a tolerance.

Pass C — Real commit slice:
    Re-runs the learned gate variant on a set of real code-editing episodes and
    reports the same telemetry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class RunSpec:
    name: str
    extra_args: Sequence[str]


def run_command(cmd: List[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_metrics(metrics_path: Path) -> Dict:
    with metrics_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def run_pass_a(
    data_dir: Path,
    output_root: Path,
    python_bin: str,
    config_preset: str,
    epochs: int,
    gate_lambda: float,
    compute_penalty: float,
) -> Dict[str, Path]:
    specs = [
        RunSpec("srfbam_learned", []),
        RunSpec("srfbam_always_encode", ["--gate-mode", "always_extract"]),
        RunSpec("fbam_no_memory", ["--disable-memory"]),
    ]
    metrics_paths: Dict[str, Path] = {}
    for spec in specs:
        run_dir = output_root / "pass_a" / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        cmd = [
            python_bin,
            "-m",
            "src.training.train_srfbam_code",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(run_dir),
            "--metrics-out",
            str(metrics_path),
            "--epochs",
            str(epochs),
            "--config-preset",
            config_preset,
            "--gate-lambda",
            str(gate_lambda),
            "--compute-penalty",
            str(compute_penalty),
        ]
        cmd.extend(spec.extra_args)
        run_command(cmd)
        metrics_paths[spec.name] = metrics_path
    return metrics_paths


def summarise_speedups(metrics: Dict) -> Dict[str, float]:
    eval_metrics = metrics.get("eval", {})
    return {
        "encode_fraction": float(eval_metrics.get("gate_encode_fraction", 0.0)),
        "query_fraction": float(eval_metrics.get("gate_query_fraction", 0.0)),
        "skip_fraction": float(eval_metrics.get("gate_skip_fraction", 0.0)),
        "predicted_speedup": float(eval_metrics.get("gate_predicted_speedup", 0.0)),
        "observed_speedup": float(eval_metrics.get("gate_observed_speedup", 0.0)),
    }


def run_pass_b(metrics_paths: Dict[str, Path]) -> None:
    print("\n[pass B] Entity reuse vs. speedup summary")
    for name, path in metrics_paths.items():
        metrics = load_metrics(path)
        summary = summarise_speedups(metrics)
        delta = abs(summary["predicted_speedup"] - summary["observed_speedup"])
        print(
            f"  {name:24s} e={summary['encode_fraction']:.3f} "
            f"q={summary['query_fraction']:.3f} s={summary['skip_fraction']:.3f} "
            f"pred={summary['predicted_speedup']:.2f} obs={summary['observed_speedup']:.2f} "
            f"| Δ={delta:.3f}"
        )
        if delta > 0.02:
            print(f"    [warn] predicted vs observed speedup differ by {delta:.3f} (>0.02)")


def run_pass_c(
    real_data_dir: Path | None,
    python_bin: str,
    output_root: Path,
    config_preset: str,
    epochs: int,
    gate_lambda: float,
    compute_penalty: float,
) -> None:
    if not real_data_dir:
        print("[pass C] Skipped (no --real-data-dir provided)")
        return
    run_dir = output_root / "pass_c" / "real_commits"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    cmd = [
        python_bin,
        "-m",
        "src.training.train_srfbam_code",
        "--data-dir",
        str(real_data_dir),
        "--output-dir",
        str(run_dir),
        "--metrics-out",
        str(metrics_path),
        "--epochs",
        str(epochs),
        "--config-preset",
        config_preset,
        "--gate-lambda",
        str(gate_lambda),
        "--compute-penalty",
        str(compute_penalty),
    ]
    run_command(cmd)
    metrics = load_metrics(metrics_path)
    summary = summarise_speedups(metrics)
    print(
        "\n[pass C] Real commit telemetry "
        f"e={summary['encode_fraction']:.3f} q={summary['query_fraction']:.3f} "
        f"s={summary['skip_fraction']:.3f} pred={summary['predicted_speedup']:.2f} "
        f"obs={summary['observed_speedup']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SR-FBAM verification passes.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to smoke-test dataset.")
    parser.add_argument("--real-data-dir", type=Path, default=None, help="Optional real commit dataset.")
    parser.add_argument("--output-root", type=Path, default=Path("results/verification"))
    parser.add_argument("--config-preset", choices=["small", "large"], default="small")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gate-lambda", type=float, default=0.002)
    parser.add_argument("--compute-penalty", type=float, default=0.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_paths = run_pass_a(
        data_dir=args.data_dir.resolve(),
        output_root=output_root,
        python_bin=args.python,
        config_preset=args.config_preset,
        epochs=args.epochs,
        gate_lambda=args.gate_lambda,
        compute_penalty=args.compute_penalty,
    )
    run_pass_b(metrics_paths)
    run_pass_c(
        real_data_dir=args.real_data_dir.resolve() if args.real_data_dir else None,
        python_bin=args.python,
        output_root=output_root,
        config_preset=args.config_preset,
        epochs=args.epochs,
        gate_lambda=args.gate_lambda,
        compute_penalty=args.compute_penalty,
    )


if __name__ == "__main__":
    main()
