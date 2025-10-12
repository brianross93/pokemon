"""
Interactive demo for exploring SR-FBAM reasoning traces.

Usage:
    python scripts/demo.py --checkpoint checkpoints/sr_fbam_train_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data import load_dataset
from src.training.train import load_checkpoint


def format_hop(hop: object) -> str:
    hop_dict = hop.__dict__ if hasattr(hop, "__dict__") else hop
    return (
        f"[{hop_dict['hop_number']:02d}] {hop_dict['action']:6s}  "
        f"{hop_dict['result']}  (confidence={hop_dict['confidence']:.2f})"
    )


def run_demo(checkpoint: Path, data_dir: Path, device: str) -> None:
    print("Loading checkpoint and dataset...")
    model = load_checkpoint(checkpoint, device=device)
    kg, queries = load_dataset(data_dir, split="eval")
    print(f"Loaded {len(queries)} evaluation queries.\n")

    while True:
        print("\nOptions:")
        print("  1) Show sample queries")
        print("  2) Evaluate accuracy on entire split")
        print("  q) Quit")
        choice = input("Select option: ").strip().lower()

        if choice == "q":
            print("Exiting demo.")
            break

        if choice == "1":
            sample = queries[:5]
            for idx, query in enumerate(sample, start=1):
                print(f"\n[{idx}] {query.natural_language}")
            selected = input("Pick query (1-5): ").strip()
            if not selected.isdigit() or not (1 <= int(selected) <= len(sample)):
                print("Invalid choice.")
                continue
            query = sample[int(selected) - 1]
            print(f"\nQuery: {query.natural_language}")
            print(f"Answer (ground truth): {query.answer_id}")

            output = model.reason(query, kg)
            print("\nHop trace:")
            for hop in output.hop_traces:
                print(" ", format_hop(hop))

            print(
                f"\nPrediction: {output.prediction_id} "
                f"(name={output.prediction_name})"
            )
            print(f"Correct: {output.prediction_id == query.answer_id}")
            print(f"Confidence: {output.confidence:.3f}")
            print(f"Hops: {len(output.hop_traces)}")

        elif choice == "2":
            correct = 0
            for query in queries:
                output = model.reason(query, kg)
                if output.prediction_id == query.answer_id:
                    correct += 1
            accuracy = correct / len(queries) if queries else 0.0
            print(f"\nAccuracy on {len(queries)} queries: {accuracy:.3f}")
        else:
            print("Unknown option.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SR-FBAM interactive demo")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained SR-FBAM checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Dataset directory (default: data)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run on (default: cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(Path(args.checkpoint), Path(args.data_dir), args.device)
