"""
Utility helpers to compute functional evaluation metrics (diff correctness and
test pass rates) from per-step action predictions.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence


DIFF_ACTIONS = {"INSERT_LINE", "DELETE_LINE"}
TEST_ACTION = "RUN_TESTS"
PASS_KEYWORDS = ("PASSED", "SUCCESS", "OK")


def init_functional_stats() -> Dict[str, float]:
    return {
        "total_steps": 0,
        "diff_total": 0,
        "diff_correct": 0,
        "test_total": 0,
        "test_correct": 0,
        "test_actual_pass": 0,
        "test_pred_positive": 0,
        "test_true_positive": 0,
    }


def accumulate_functional_stats(
    stats: Dict[str, float],
    actual_actions: Sequence[str],
    predicted_actions: Sequence[str],
    terminal_outputs: Sequence[str],
) -> None:
    for action, prediction, output in zip(actual_actions, predicted_actions, terminal_outputs):
        stats["total_steps"] += 1
        is_diff = action in DIFF_ACTIONS
        is_test_actual = action == TEST_ACTION
        is_test_pred = prediction == TEST_ACTION
        passed = _terminal_output_passed(output) if is_test_actual else False

        if is_diff:
            stats["diff_total"] += 1
            if prediction == action:
                stats["diff_correct"] += 1

        if is_test_actual:
            stats["test_total"] += 1
            if prediction == action:
                stats["test_correct"] += 1
            if passed:
                stats["test_actual_pass"] += 1
                if is_test_pred:
                    stats["test_true_positive"] += 1

        if is_test_pred:
            stats["test_pred_positive"] += 1


def finalize_functional_stats(stats: Dict[str, float]) -> Dict[str, float]:
    diff_acc = _safe_ratio(stats["diff_correct"], stats["diff_total"])
    test_acc = _safe_ratio(stats["test_correct"], stats["test_total"])
    test_pass_precision = _safe_ratio(stats["test_true_positive"], stats["test_pred_positive"])
    test_pass_recall = _safe_ratio(stats["test_true_positive"], stats["test_actual_pass"])

    return {
        "diff_action_accuracy": diff_acc,
        "test_action_accuracy": test_acc,
        "test_pass_precision": test_pass_precision,
        "test_pass_recall": test_pass_recall,
    }


def _terminal_output_passed(output: str | None) -> bool:
    if not output:
        return False
    upper = output.upper()
    return any(keyword in upper for keyword in PASS_KEYWORDS)


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0

