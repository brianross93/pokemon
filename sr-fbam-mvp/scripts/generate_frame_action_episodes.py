#!/usr/bin/env python3
"""
Generate synthetic frame/action episodes that mimic long-horizon code editing workflows.

Each episode captures agent actions while editing code. Frames serialize editor state
and terminal output so FBAM-style models can consume them.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class TaskContext:
    window_class: str
    label_name: str
    default_window: int


@dataclass
class TaskBlueprint:
    task_id: str
    instructions: str
    initial_files: Dict[str, List[str]]
    final_factory: Callable[[random.Random], Tuple[Dict[str, List[str]], TaskContext]]
    test_command: str

    def test_output(self, progress: float, ctx: TaskContext, rng: random.Random) -> str:
        header = "============================= test session starts =============================\n"
        footer = "=========================== short test summary info ===========================\n"
        command = "collected 6 items\n"

        if progress < 0.35:
            body = (
                "tests/test_metrics.py::test_window_push_and_average ERROR                      [ 16%]\n"
                "tests/test_metrics.py::test_trend_requires_history ERROR                        [ 33%]\n"
                "tests/test_metrics.py::test_normalize_handles_constant SKIPPED                 [ 50%]\n"
                "tests/test_metrics.py::test_rolling_average_basics SKIPPED                     [ 66%]\n"
                "tests/test_metrics.py::test_normalize_balances_range SKIPPED                   [ 83%]\n"
                "tests/test_metrics.py::test_summarize_metrics_basics SKIPPED                   [100%]\n"
                "==================================== ERRORS ====================================\n"
                "____ ERROR at setup of test_window_push_and_average ____\n"
                f"E   AttributeError: module 'analytics.metrics' has no attribute '{ctx.window_class}'\n"
                "________________ ERROR at setup of test_trend_requires_history ________________\n"
                "E   ImportError: cannot import name 'MetricSample' from 'analytics.metrics'\n"
            )
        elif progress < 0.8:
            body = (
                "tests/test_metrics.py::test_window_push_and_average FAILED                    [ 16%]\n"
                "tests/test_metrics.py::test_trend_requires_history FAILED                      [ 33%]\n"
                "tests/test_metrics.py::test_normalize_handles_constant PASSED                  [ 50%]\n"
                "tests/test_metrics.py::test_rolling_average_basics FAILED                      [ 66%]\n"
                "tests/test_metrics.py::test_normalize_balances_range PASSED                    [ 83%]\n"
                "tests/test_metrics.py::test_summarize_metrics_basics FAILED                    [100%]\n"
                "=================================== FAILURES ===================================\n"
                "______________________ test_window_push_and_average ___________________________\n"
                f"E   AssertionError: expected average to reflect latest {ctx.label_name} sample\n"
                "______________________ test_trend_requires_history ___________________________\n"
                "E   AssertionError: Trend should be None when fewer than 2 samples exist\n"
                "______________________ test_rolling_average_basics ___________________________\n"
                "E   AssertionError: Rolling mean did not drop oldest element\n"
                "______________________ test_summarize_metrics_basics _________________________\n"
                "E   ValueError: no metric values provided\n"
            )
        else:
            body = (
                "tests/test_metrics.py::test_window_push_and_average PASSED                    [ 16%]\n"
                "tests/test_metrics.py::test_trend_requires_history PASSED                      [ 33%]\n"
                "tests/test_metrics.py::test_normalize_handles_constant PASSED                  [ 50%]\n"
                "tests/test_metrics.py::test_rolling_average_basics PASSED                      [ 66%]\n"
                "tests/test_metrics.py::test_normalize_balances_range PASSED                    [ 83%]\n"
                "tests/test_metrics.py::test_summarize_metrics_basics PASSED                    [100%]\n"
                "============================== 6 passed in 0.72s ===============================\n"
            )

        return header + command + body + footer

    def success_output(self, ctx: TaskContext) -> str:
        return (
            "============================= test session starts =============================\n"
            "platform linux -- Python 3.11.0, pytest-7.4.0, pluggy-1.4.0\n"
            "rootdir: /workspace\n"
            "collected 6 items\n"
            "tests/test_metrics.py ......                                               [100%]\n"
            "============================== 6 passed in 0.51s ===============================\n"
        )


def make_metrics_blueprint() -> TaskBlueprint:
    def initial_files() -> Dict[str, List[str]]:
        return {
            "analytics/metrics.py": [
                '"""Placeholder analytics metrics module."""',
                "",
                "from typing import Iterable",
                "",
                "",
                "def summarize_metrics(values: Iterable[float]) -> float:",
                "    raise NotImplementedError('TODO: implement summarizer')",
            ],
            "tests/test_metrics.py": [
                "import pytest",
                "",
                "",
                "def test_placeholder() -> None:",
                "    pytest.skip('not implemented')",
            ],
        }

    metrics_template = """\"\"\"Analytics metric utilities.\"\"\"
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Iterable, Iterator, List, Optional

__all__ = [
    "MetricSample",
    "{window_class}",
    "summarize_metrics",
    "rolling_average",
    "normalize",
]


@dataclass(slots=True)
class MetricSample:
    timestamp: float
    value: float
    label: str = "{label_name}"


class {window_class}:
    \"\"\"Fixed-size window that tracks the most recent metric samples.\"\"\"

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("window size must be positive")
        self.size = int(size)
        self.samples: Deque[MetricSample] = deque(maxlen=self.size)

    def push(self, sample: MetricSample) -> None:
        self.samples.append(sample)

    def is_full(self) -> bool:
        return len(self.samples) == self.size

    def __len__(self) -> int:
        return len(self.samples)

    def clear(self) -> None:
        self.samples.clear()

    def values(self) -> List[float]:
        return [sample.value for sample in self.samples]

    def average(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.values()) / len(self.samples)

    def recent(self, limit: int = 5) -> List[MetricSample]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return list(self.samples)[-limit:]

    def trend(self) -> Optional[float]:
        if len(self.samples) < 2:
            return None
        snapshots = list(self.samples)
        diffs = [curr.value - prev.value for prev, curr in zip(snapshots, snapshots[1:])]
        if not diffs:
            return 0.0
        return sum(diffs) / len(diffs)


def summarize_metrics(values: Iterable[float]) -> float:
    data = [value for value in values if value is not None]
    if not data:
        raise ValueError("no metric values provided")
    return float(mean(data))


def rolling_average(series: Iterable[float], window: int) -> Iterator[float]:
    limit = int(window)
    if limit <= 0:
        raise ValueError("window must be positive")
    history: Deque[float] = deque(maxlen=limit)
    for value in series:
        history.append(float(value))
        if len(history) == limit:
            yield sum(history) / len(history)


def normalize(values: Iterable[float]) -> List[float]:
    numbers = [float(item) for item in values]
    if not numbers:
        return []
    minimum = min(numbers)
    maximum = max(numbers)
    if minimum == maximum:
        return [1.0 for _ in numbers]
    span = maximum - minimum
    return [(value - minimum) / span for value in numbers]
"""

    tests_template = """import math
import itertools

import pytest

from analytics.metrics import MetricSample, {window_class}, normalize, rolling_average, summarize_metrics


def test_window_push_and_average() -> None:
    window = {window_class}(size={default_window})
    samples = [MetricSample(timestamp=float(idx), value=float(idx + 1)) for idx in range(5)]
    for sample in samples:
        window.push(sample)
    assert window.is_full() is True
    assert math.isclose(window.average(), sum(range(2, 2 + {default_window})) / {default_window})


def test_trend_requires_history() -> None:
    window = {window_class}(size=4)
    assert window.trend() is None
    window.push(MetricSample(timestamp=0.0, value=1.0))
    assert window.trend() is None
    window.push(MetricSample(timestamp=1.0, value=3.0))
    assert window.trend() == pytest.approx(2.0)


def test_normalize_handles_constant() -> None:
    assert normalize([3.14, 3.14]) == [1.0, 1.0]
    assert normalize([]) == []


def test_rolling_average_basics() -> None:
    series = [1.0, 3.0, 5.0, 7.0, 9.0]
    averages = list(rolling_average(series, window=2))
    assert averages == [2.0, 4.0, 6.0, 8.0]


def test_normalize_balances_range() -> None:
    numbers = [0.0, 5.0, 10.0]
    normalized = normalize(numbers)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(1.0)


def test_summarize_metrics_basics() -> None:
    assert summarize_metrics([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        summarize_metrics([])
"""

    def final_factory(rng: random.Random) -> Tuple[Dict[str, List[str]], TaskContext]:
        window_class = rng.choice(["RollingMetricWindow", "SlidingMetricWindow"])
        label_name = rng.choice(["default", "primary", "metric"])
        default_window = rng.choice([3, 4, 5])
        ctx = TaskContext(
            window_class=window_class,
            label_name=label_name,
            default_window=default_window,
        )
        metrics_code = metrics_template.format(
            window_class=window_class,
            label_name=label_name,
        )
        tests_code = tests_template.format(
            window_class=window_class,
            default_window=default_window,
        )
        return (
            {
                "analytics/metrics.py": metrics_code.strip("\n").splitlines(),
                "tests/test_metrics.py": tests_code.strip("\n").splitlines(),
            },
            ctx,
        )

    return TaskBlueprint(
        task_id="rolling_metrics_window",
        instructions=(
            "Implement the rolling metrics utilities module with proper averaging,"
            " trend estimation, normalization helpers, and accompanying unit tests."
        ),
        initial_files=initial_files(),
        final_factory=final_factory,
        test_command="pytest tests/test_metrics.py -q",
    )


class CodeEnvironment:
    def __init__(self, instructions: str, files: Dict[str, List[str]]) -> None:
        if not files:
            raise ValueError("environment requires at least one file")
        self.instructions = instructions
        self.files = {path: list(lines) for path, lines in files.items()}
        self.active_file = next(iter(self.files))
        self.cursor_line = 1
        self.cursor_col = 1
        self.terminal_output = "shell ready\n"

    def ensure_file(self, path: str) -> None:
        if path not in self.files:
            self.files[path] = []

    def open_file(self, path: str) -> None:
        self.ensure_file(path)
        self.active_file = path
        self.cursor_line = 1
        self.cursor_col = 1

    def move_cursor(self, line: int, column: int = 1) -> None:
        self.cursor_line = max(1, line)
        self.cursor_col = max(1, column)

    def insert_line(self, line: int, content: str) -> None:
        lines = self.files[self.active_file]
        index = max(0, min(line - 1, len(lines)))
        lines.insert(index, content)
        self.cursor_line = index + 1
        self.cursor_col = len(content) + 1

    def delete_line(self, line: int) -> None:
        lines = self.files[self.active_file]
        index = line - 1
        if 0 <= index < len(lines):
            lines.pop(index)
        self.cursor_line = max(1, min(line, len(lines))) if lines else 1
        self.cursor_col = 1

    def run_tests(self, command: str, output: str) -> None:
        transcript = output.strip("\n")
        self.terminal_output = f"$ {command}\n{transcript}\n"

    def render_frame(self) -> str:
        lines = self.files.get(self.active_file, [])
        code_block = []
        for idx, line in enumerate(lines, start=1):
            marker = ">" if idx == self.cursor_line else " "
            code_block.append(f"{marker}{idx:03d}| {line}")
        code_section = "\n".join(code_block) if code_block else "(empty file)"
        terminal_section = self.terminal_output.rstrip("\n") or "(no terminal output)"
        return (
            f"Task: {self.instructions}\n"
            f"Active File: {self.active_file}\n"
            f"Cursor: line {self.cursor_line}, column {self.cursor_col}\n"
            "----- code snapshot -----\n"
            f"{code_section}\n"
            "----- terminal -----\n"
            f"{terminal_section}"
        )


def sample_duration_ms(kind: str, rng: random.Random) -> int:
    if kind == "RUN_TESTS":
        return rng.randint(1800, 3400)
    if kind == "MOVE_CURSOR":
        return rng.randint(90, 200)
    if kind == "OPEN_FILE":
        return rng.randint(120, 260)
    if kind == "DELETE_LINE":
        return rng.randint(260, 520)
    if kind == "INSERT_LINE":
        return rng.randint(380, 760)
    return rng.randint(240, 480)


class EpisodeComplete(Exception):
    pass


def simulate_episode(
    episode_index: int,
    blueprint: TaskBlueprint,
    rng: random.Random,
    max_steps: Optional[int] = None,
) -> Tuple[Dict, List[Dict]]:
    final_files, ctx = blueprint.final_factory(rng)
    env = CodeEnvironment(blueprint.instructions, blueprint.initial_files)
    cumulative_ms = 0
    steps: List[Dict] = []
    episode_id = f"episode_{episode_index:04d}"
    context_payload = asdict(ctx)

    def record(kind: str, description: str, **kwargs) -> None:
        nonlocal cumulative_ms
        env_action(kind, **kwargs)
        delta = sample_duration_ms(kind, rng)
        cumulative_ms += delta
        step_number = len(steps) + 1
        step_record = {
            "episode_id": episode_id,
            "task_id": blueprint.task_id,
            "instructions": blueprint.instructions,
            "context": context_payload,
            "step": step_number,
            "frame_id": f"ep{episode_index:04d}_step{step_number:03d}",
            "action": {
                "type": kind,
                "summary": description,
                "args": kwargs,
            },
            "state": {
                "active_file": env.active_file,
                "cursor": {"line": env.cursor_line, "column": env.cursor_col},
            },
            "frame_text": env.render_frame(),
            "terminal_output": env.terminal_output.rstrip("\n"),
            "delta_wall_time_ms": delta,
            "cumulative_wall_time_ms": cumulative_ms,
        }
        steps.append(step_record)
        if max_steps and len(steps) >= max_steps:
            raise EpisodeComplete

    def env_action(kind: str, **kwargs) -> None:
        if kind == "OPEN_FILE":
            env.open_file(kwargs["path"])
        elif kind == "MOVE_CURSOR":
            env.move_cursor(kwargs["line"], kwargs.get("column", 1))
        elif kind == "INSERT_LINE":
            env.insert_line(kwargs["line"], kwargs["content"])
        elif kind == "DELETE_LINE":
            env.delete_line(kwargs["line"])
        elif kind == "RUN_TESTS":
            env.run_tests(kwargs["command"], kwargs["output"])
        else:
            raise ValueError(f"unsupported action kind: {kind}")

    for path in final_files:
        env.ensure_file(path)

    for path, initial_lines in blueprint.initial_files.items():
        if path not in final_files:
            final_files[path] = []

    try:
        for file_path, target_lines in final_files.items():
            record("OPEN_FILE", f"Focus editor on {file_path}", path=file_path)

            starting_lines = list(blueprint.initial_files.get(file_path, []))
            for offset, existing in enumerate(reversed(starting_lines)):
                line_no = len(starting_lines) - offset
                record(
                    "MOVE_CURSOR",
                    f"Jump to line {line_no} to remove placeholder content",
                    line=line_no,
                    column=1,
                )
                record(
                    "DELETE_LINE",
                    f"Delete placeholder line {line_no}: {existing.strip() or '<blank>'}",
                    line=line_no,
                )

            inserted = 0
            for line_no, content in enumerate(target_lines, start=1):
                record(
                    "MOVE_CURSOR",
                    f"Position cursor for line {line_no}",
                    line=line_no,
                    column=1,
                )
                record(
                    "INSERT_LINE",
                    f"Insert code at line {line_no}",
                    line=line_no,
                    content=content,
                )
                inserted += 1
                if inserted % 7 == 0:
                    progress = inserted / max(1, len(target_lines))
                    record(
                        "RUN_TESTS",
                        "Run tests to assess progress",
                        command=blueprint.test_command,
                        output=blueprint.test_output(progress, ctx, rng),
                    )

            record(
                "RUN_TESTS",
                "Run tests after finishing file edits",
                command=blueprint.test_command,
                output=blueprint.success_output(ctx),
            )
    except EpisodeComplete:
        pass

    episode_summary = {
        "episode_id": episode_id,
        "task_id": blueprint.task_id,
        "instructions": blueprint.instructions,
        "num_steps": len(steps),
        "total_wall_time_ms": cumulative_ms,
        "context": context_payload,
    }
    return episode_summary, steps


def write_step_jsonl(path: Path, episodes_steps: Sequence[Sequence[Dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for step_list in episodes_steps:
            for record in step_list:
                fout.write(json.dumps(record) + "\n")


def write_episode_summaries(path: Path, summaries: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for summary in summaries:
            fout.write(json.dumps(summary) + "\n")


def build_metadata(
    train_path: Path,
    eval_path: Path,
    train_summaries: Sequence[Dict],
    eval_summaries: Sequence[Dict],
) -> Dict:
    train_total_steps = sum(ep["num_steps"] for ep in train_summaries)
    eval_total_steps = sum(ep["num_steps"] for ep in eval_summaries)
    return {
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_episodes": len(train_summaries),
        "eval_episodes": len(eval_summaries),
        "train_total_steps": train_total_steps,
        "eval_total_steps": eval_total_steps,
        "train_average_steps": train_total_steps / max(1, len(train_summaries)),
        "eval_average_steps": eval_total_steps / max(1, len(eval_summaries)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frame/action editor episodes.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/episodes"))
    parser.add_argument("--train", type=int, default=4)
    parser.add_argument("--eval", type=int, default=2)
    parser.add_argument("--steps", type=int, default=338, help="Target steps per episode (approx).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    blueprint = make_metrics_blueprint()

    train_results = [
        simulate_episode(idx, blueprint, rng, max_steps=args.steps)
        for idx in range(args.train)
    ]
    eval_offset = args.train
    eval_results = [
        simulate_episode(eval_offset + idx, blueprint, rng, max_steps=args.steps)
        for idx in range(args.eval)
    ]

    train_summaries = [summary for summary, _ in train_results]
    train_steps = [steps for _, steps in train_results]
    eval_summaries = [summary for summary, _ in eval_results]
    eval_steps = [steps for _, steps in eval_results]

    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    write_step_jsonl(train_path, train_steps)
    write_step_jsonl(eval_path, eval_steps)

    summary_path = args.output_dir / "episodes.jsonl"
    write_episode_summaries(summary_path, [*train_summaries, *eval_summaries])

    metadata = build_metadata(train_path, eval_path, train_summaries, eval_summaries)
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[ok] wrote {len(train_summaries)} training episodes to {train_path}")
    print(f"[ok] wrote {len(eval_summaries)} evaluation episodes to {eval_path}")
    print(f"[ok] wrote episode summaries to {summary_path}")
    print(f"[ok] metadata: {metadata}")


if __name__ == "__main__":
    main()
