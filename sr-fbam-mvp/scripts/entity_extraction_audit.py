from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

#!/usr/bin/env python3
"""Audit symbolic entity extraction precision/recall and fallback."""

import argparse
import ast
import json
import random
from typing import Dict, List, Tuple

from src.data.frame_dataset import FrameActionStep, load_datasets
from src.models.sr_fbam_code import FrameSymbolExtractor

CODE_HEADER = "----- code snapshot -----"
TERMINAL_HEADER = "----- terminal -----"


def extract_code(frame_text: str) -> str | None:
    if CODE_HEADER not in frame_text:
        return None
    _, rest = frame_text.split(CODE_HEADER, 1)
    if TERMINAL_HEADER in rest:
        code_section, _ = rest.split(TERMINAL_HEADER, 1)
    else:
        code_section = rest
    if "----- instructions -----" in code_section:
        code_section = code_section.split("----- instructions -----", 1)[0]
    lines = []
    for raw in code_section.strip("\n").splitlines():
        if "|" in raw:
            _, tail = raw.split("|", 1)
            lines.append(tail)
        else:
            lines.append(raw)
    import textwrap
    code = textwrap.dedent("\n".join(lines)).strip()
    return code or None


def ast_entities(code: str) -> Tuple[set[str], set[str]]:
    tree = ast.parse(code)
    functions: set[str] = set()
    classes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
    return functions, classes


def gather_steps() -> List[FrameActionStep]:
    candidate_dirs = [
        Path("data/episodes_50"),
        Path("data/episodes_100"),
        Path("data/episodes_200"),
        Path("data/episodes_338"),
        Path("data/episodes_1000"),
        Path("data/git_episodes"),
    ]
    datasets = []
    for directory in candidate_dirs:
        if directory.exists():
            _, eval_ds = load_datasets(directory)
            datasets.append(eval_ds)
    steps: List[FrameActionStep] = []
    for dataset in datasets:
        for episode in dataset.episodes:
            if episode.steps:
                steps.append(episode.steps[-1])
    return steps


def audit(sample_size: int, seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    steps = gather_steps()
    rng.shuffle(steps)
    steps = steps[:sample_size]
    actual_count = len(steps)

    extractor = FrameSymbolExtractor(max_tokens=64)

    func_hits = func_preds = func_truth = 0
    class_hits = class_preds = class_truth = 0
    parseable = 0
    fallback = 0

    for step in steps:
        tokens = extractor.extract(step)
        if not tokens:
            fallback += 1
        code = extract_code(step.frame_text)
        if not code:
            continue
        try:
            functions, classes = ast_entities(code)
        except SyntaxError:
            continue
        parseable += 1

        token_set = set(tokens)
        func_truth += len(functions)
        class_truth += len(classes)

        func_pred = token_set & functions
        class_pred = token_set & classes

        func_preds += len(func_pred)
        class_preds += len(class_pred)
        func_hits += len(func_pred)
        class_hits += len(class_pred)

    function_precision = func_hits / func_preds if func_preds else 0.0
    function_recall = func_hits / func_truth if func_truth else 0.0
    class_precision = class_hits / class_preds if class_preds else 0.0
    class_recall = class_hits / class_truth if class_truth else 0.0

    return {
        "sample_size": actual_count,
        "parseable": parseable,
        "fallback_rate": fallback / actual_count if actual_count else 0.0,
        "function_precision": function_precision,
        "function_recall": function_recall,
        "class_precision": class_precision,
        "class_recall": class_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/entity_audit.json"))
    args = parser.parse_args()

    metrics = audit(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
