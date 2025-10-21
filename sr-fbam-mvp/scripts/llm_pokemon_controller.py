#!/usr/bin/env python3
"""LLM-driven controller for Pokemon Blue with SR-FBAM gating."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.knowledge.knowledge_graph import Context, EncounterKB
from src.llm.llm_client import LLMConfig, create_llm_client
from src.llm.pokemon_prompts import PokemonPrompts
from src.middleware.pyboy_adapter import PyBoyConfig, PyBoyPokemonAdapter
from src.middleware.pokemon_adapter import PokemonAction, PokemonTelemetry
from srfbam.core import EncodedFrame
from src.pkmn.frame_encoder import encode_frame
from src.pkmn.sr_fbam_agent import SRFBAMPokemonAgent, SrfbamStepSummary

INTRO_SEQUENCE = [
    ("START", 60, 10),
    ("A", 120, 8),
    ("DOWN", 1, 24),
    ("A", 1, 12),
    ("A", 300, 8),
    ("DOWN", 1, 24),
    ("A", 1, 12),
    ("A", 1200, 8),
]

FREE_MENU_STATES = {0}
NUMERIC_FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "area_norm",
    "in_grass",
    "in_battle",
    "joy_ignore",
    "menu_state",
    "menu_cursor",
]
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
FACING_MAP = {0: "DOWN", 4: "UP", 8: "LEFT", 12: "RIGHT"}


@dataclass
class DebouncedFlag:
    """Debounces state transitions so we only react to sustained changes."""

    min_stable: int = 6
    key: Optional[tuple] = None
    stable_for: int = 0

    def update(self, key: tuple) -> None:
        if key == self.key:
            self.stable_for += 1
        else:
            self.key = key
            self.stable_for = 1

    def fired(self) -> bool:
        return self.stable_for == max(1, self.min_stable)


class ProgressWindow:
    """Tracks recent movement progress to detect stalls."""

    def __init__(self, size: int = 60) -> None:
        self.buffer: deque[tuple[int, int, bool, bool]] = deque(maxlen=max(1, size))

    def push(self, x: int, y: int, in_menu: bool, in_battle: bool) -> None:
        self.buffer.append((x, y, in_menu, in_battle))

    @property
    def ratio(self) -> float:
        active = [(x, y) for x, y, menu, battle in self.buffer if not menu and not battle]
        if not active:
            return 0.0
        return min(1.0, len({(x, y) for x, y in active}) / max(1, len(active)))


class FlipFlopDetector:
    """Detects oscillation between opposing actions (e.g., LEFT ? RIGHT)."""

    OPPOSITES = {("LEFT", "RIGHT"), ("RIGHT", "LEFT"), ("UP", "DOWN"), ("DOWN", "UP")}

    def __init__(self, window: int = 12, threshold: int = 3) -> None:
        self.actions: deque[str] = deque(maxlen=max(2, window))
        self.threshold = max(1, threshold)

    def update(self, action: Optional[str]) -> bool:
        if not action:
            return False
        self.actions.append(action.upper())
        seq = list(self.actions)
        flips = sum(1 for prev, cur in zip(seq, seq[1:]) if (prev, cur) in self.OPPOSITES)
        return flips >= self.threshold


@dataclass
class Plan:
    action: str
    direction: Optional[str]
    remaining_frames: int
    issued_step: int
    area_id: int
    in_battle: bool
    source: str = "LLM"


@dataclass
class Event:
    area_id: int
    pos: Tuple[int, int]
    action: str
    reward: float
    in_battle: bool
    ts: int


class EpisodeMemory:
    """Ring buffer for lightweight episodic traces and habit frequencies."""

    def __init__(self, cap: int = 4000) -> None:
        self.buf: deque[Event] = deque(maxlen=max(1, cap))
        self.habit: defaultdict[Tuple[int, str], defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def add(self, ev: Event, last_dir: Optional[str]) -> None:
        self.buf.append(ev)
        if ev.reward > 0.5 and last_dir:
            key = (ev.area_id, last_dir)
            self.habit[key][ev.action] += 1

    def propose(self, area_id: int, last_dir: Optional[str]) -> Optional[str]:
        if not last_dir:
            return None
        table = self.habit[(area_id, last_dir)]
        if not table:
            return None
        return max(table.items(), key=lambda kv: kv[1])[0]

    def top_snippets(self, area_id: int, k: int = 2) -> List[Event]:
        window = list(self.buf)[-800:]
        snippets = [ev for ev in window if ev.area_id == area_id and ev.reward > 1.0]
        snippets.sort(key=lambda ev: (ev.reward, ev.ts))
        return snippets[-k:]


@dataclass
class Goal:
    name: str
    done: Callable[[], bool]
    budget: int
    on_fail: Optional[Callable[[], None]] = None


def badness_score(over_budget: float, progress_ratio: float, menu_trap: bool, flip_flop: bool) -> float:
    capped = min(2.0, max(0.0, over_budget))
    progress_term = 1.0 - max(0.0, min(1.0, progress_ratio))
    score = 0.4 * capped + 0.4 * progress_term
    if menu_trap:
        score += 0.1
    if flip_flop:
        score += 0.1
    return score


class ControlStateTracker:
    """Classifies emulator phase (intro/menu/cutscene/battle/overworld)."""

    CONTROL_WINDOW = 12

    def __init__(self) -> None:
        self.joy_history: deque[int] = deque(maxlen=self.CONTROL_WINDOW)
        self.menu_history: deque[int] = deque(maxlen=self.CONTROL_WINDOW)
        self.control_frames = 0
        self.control_acquired = False
        self.phase = "intro"

    def observe(self, telemetry: PokemonTelemetry) -> None:
        joy_ignore = int(telemetry.extra.get("joy_ignore", 0))
        menu_state = int(telemetry.extra.get("menu_state", 0))
        self.joy_history.append(joy_ignore)
        self.menu_history.append(menu_state)

        joy_locked = (joy_ignore != 0) or self._history_joy_locked()
        menu_overlay = menu_state not in FREE_MENU_STATES
        menu_active = menu_overlay or self._history_menu_active()

        if not joy_locked and not menu_active and not telemetry.in_battle:
            self.control_frames = min(self.control_frames + 1, self.CONTROL_WINDOW)
        else:
            self.control_frames = 0

        if self.control_frames >= self.CONTROL_WINDOW:
            self.control_acquired = True

        if not self.control_acquired:
            self.phase = "intro"
        elif telemetry.in_battle:
            self.phase = "battle"
        elif joy_locked:
            self.phase = "cutscene"
        elif menu_active:
            self.phase = "menu"
        else:
            self.phase = "overworld"

    def has_control(self) -> bool:
        return self.control_acquired

    def _history_menu_active(self) -> bool:
        if not self.menu_history:
            return False
        return sum(1 for state in self.menu_history if state not in FREE_MENU_STATES) > len(self.menu_history) // 2

    def _history_joy_locked(self) -> bool:
        if not self.joy_history:
            return False
        return sum(1 for value in self.joy_history if value > 0) > len(self.joy_history) // 2


class LLMPokemonController:
    def __init__(
        self,
        rom_path: str,
        llm_config: LLMConfig,
        decision_interval: int = 10,
        visual: bool = False,
        min_decision_interval: int = 45,
        decision_cooldown: int = 120,
        state_debounce: int = 6,
        progress_window: int = 60,
        progress_floor: float = 0.10,
        escalation_threshold: float = 1.6,
        llm_rate_window_sec: float = 60.0,
        max_llm_calls_per_min: int = 3,
        default_budget: int = 36,
        stuck_base: int = 48,
        flipflop_threshold: int = 3,
        mode: str = "explore",
    ) -> None:
        self.prompts = PokemonPrompts()
        self.llm = create_llm_client(llm_config)
        self.knowledge = EncounterKB()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.srfbam = SRFBAMPokemonAgent().to(self.device).eval()
        self.decision_interval = max(1, decision_interval)
        self.min_decision_interval = max(1, min_decision_interval)
        self.decision_cooldown = max(0, decision_cooldown)
        self.escalation_threshold = float(escalation_threshold)
        self.base_threshold = float(self.escalation_threshold)
        self.base_cooldown = max(1, int(self.decision_cooldown or 1))
        self.base_ttl = 180
        self.arousal = 0.15
        self.arousal_decay = 0.985
        self.llm_rate_window_sec = max(1.0, float(llm_rate_window_sec))
        self.max_llm_calls_per_min = max(1, int(max_llm_calls_per_min))
        self.default_action_budget = max(1, int(default_budget))
        self.stuck_base = max(1, int(stuck_base))
        self.debouncer = DebouncedFlag(min_stable=max(1, int(state_debounce)))
        self.progress = ProgressWindow(size=max(1, int(progress_window)))
        self.flipflop_detector = FlipFlopDetector(threshold=max(1, int(flipflop_threshold)))
        self.stuck_progress_floor = max(0.0, min(1.0, float(progress_floor)))
        self.plan: Optional[Plan] = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending: Optional[Future] = None
        self._was_dialog_like = False
        self._dialog_refractory = 0
        self.last_llm_step = -self.decision_cooldown
        self.llm_call_timestamps: deque[float] = deque(maxlen=64)
        self.llm_calls_total = 0
        self.last_gate_metrics: Dict[str, Any] = {}
        self.last_summary: Optional[SrfbamStepSummary] = None
        self.last_encoded: Optional[EncodedFrame] = None
        self.log_dir = ROOT / "results" / "pkmn_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_logs: List[Dict[str, Any]] = []
        self.epmem = EpisodeMemory()
        self.motion: defaultdict[Tuple[int, str], defaultdict[Tuple[int, int], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._last_xy: Optional[Tuple[int, int]] = None
        self.mode = (mode or "explore").lower()
        if self.mode not in {"explore", "focus"}:
            self.mode = "explore"

        config = PyBoyConfig(
            rom_path=rom_path,
            window_type="SDL2" if visual else "null",
            addresses={
                "map_id": 0xD35E,
                "player_x": 0xD362,
                "player_y": 0xD361,
                "sprite_y": 0xC204,
                "sprite_x": 0xC205,
                "facing": 0xC109,
                "in_grass": 0xD5A5,
                "in_battle": 0xD057,
                "species_id": 0xD058,
                "joy_ignore": 0xCD6B,
                "menu_state": 0xCC29,
                "menu_cursor": 0xCC26,
            },
        )
        self.adapter = PyBoyPokemonAdapter(config)
        self.telemetry: Optional[PokemonTelemetry] = None

        self.last_decision: Optional[str] = None
        self.last_decision_step = -self.decision_interval
        self.prev_area_id: Optional[int] = None
        self.prev_in_battle: Optional[bool] = None
        self.prev_in_grass: Optional[bool] = None
        self.last_position: Optional[tuple[int, int]] = None
        self.current_context: Optional[str] = None
        self.current_action_frames = 0
        self.stuck_frames = 0
        self.current_walk_direction = "DOWN"
        self.control_tracker = ControlStateTracker()
        self.goal: Optional[Goal] = None
        self.bedroom_id: Optional[int] = None

    def close(self) -> None:
        closer = getattr(self.adapter, "close", None)
        if callable(closer):
            closer()
        self.pool.shutdown(wait=False)

    def start_game(self) -> None:
        self.telemetry = self.adapter.reset()
        self._observe_control()
        self.srfbam.reset_state()
        if self.telemetry is not None:
            self.bedroom_id = int(self.telemetry.area_id)
            self.goal = Goal(
                name="ExitBedroom",
                done=lambda: bool(
                    self.telemetry
                    and int(self.telemetry.area_id) != (self.bedroom_id if self.bedroom_id is not None else -1)
                ),
                budget=600,
                on_fail=lambda: setattr(self, "current_walk_direction", "DOWN"),
            )

        print("Starting game - deterministic intro sequence.")
        self._wait_for_control(grace_frames=1200)
        self._leave_bedroom_safe()
        self.prev_area_id = self.telemetry.area_id
        self.prev_in_battle = self.telemetry.in_battle
        self.prev_in_grass = self.telemetry.in_grass
        self.last_position = self._xy() if self.telemetry else None
        self.current_context = self._extract_context(self.telemetry)

    def _tap(self, button: str, times: int, frames: int) -> None:
        for _ in range(times):
            self.telemetry = self.adapter.step(PokemonAction(button, {"frames": frames}))
            self._observe_control()

    def _tap_button(self, button: str, down: int = 2, up: int = 2) -> None:
        self.telemetry = self.adapter.step(PokemonAction(button, {"frames": down}))
        self.telemetry = self.adapter.step(PokemonAction("WAIT", {"frames": up}))
        self._observe_control()

    def _attempt_move(self, direction: str, frames: int = 24) -> bool:
        """Move one tile; return True if position changed."""
        for hold in (16, frames, max(frames, 32)):
            start = self._xy()
            self.telemetry = self.adapter.step(PokemonAction(direction, {"frames": hold}))
            self._observe_control()
            if self._xy() != start:
                self.current_walk_direction = direction
                return True
            # brief release before retrying
            self.telemetry = self.adapter.step(PokemonAction("WAIT", {"frames": 2}))
            self._observe_control()

        order = ["UP", "RIGHT", "DOWN", "LEFT"]
        try:
            idx = order.index(direction)
        except ValueError:
            idx = 0
        turn = order[(idx + 1) % 4]
        start = self._xy()
        self.telemetry = self.adapter.step(PokemonAction(turn, {"frames": 16}))
        self._observe_control()
        moved = self._xy() != start
        if moved:
            self.current_walk_direction = turn
        return moved

    def _run_sequence(self, directions: List[str]) -> bool:
        """Execute a sequence of moves, aborting if any step fails."""
        for direction in directions:
            if not self._attempt_move(direction):
                return False
        return True

    def _observe_control(self) -> None:
        if self.telemetry is not None:
            self.control_tracker.observe(self.telemetry)

    def _xy(self) -> Tuple[int, int]:
        """Return map/world coordinates (wXCoord, wYCoord)."""
        if self.telemetry is None:
            return (0, 0)
        return int(self.telemetry.x), int(self.telemetry.y)

    def _screen_xy(self) -> Tuple[int, int]:
        """Return screen-relative sprite coordinates for diagnostics."""
        if self.telemetry is None:
            return (0, 0)
        extra = getattr(self.telemetry, "extra", {})
        sx = extra.get("sprite_x")
        sy = extra.get("sprite_y")
        if sx is None or sy is None:
            return self._xy()
        return int(sx), int(sy)

    def _facing_dir(self) -> str:
        if self.telemetry is None:
            return "DOWN"
        raw = int(self.telemetry.extra.get("facing", 0)) & 0x0C
        return FACING_MAP.get(raw, "DOWN")

    def _wait_for_control(self, grace_frames: int) -> None:
        frames = 0
        while frames < grace_frames and not self.control_tracker.has_control():
            if frames < 300:
                self.telemetry = self.adapter.step(PokemonAction("A", {"frames": 4}))
                frames += 4
            elif frames < 600:
                if frames % 20 == 0:
                    self.telemetry = self.adapter.step(PokemonAction("DOWN", {"frames": 4}))
                    frames += 4
                else:
                    self.telemetry = self.adapter.step(PokemonAction("A", {"frames": 4}))
                    frames += 4
            else:
                sequence = ["DOWN", "UP", "LEFT", "RIGHT"]
                direction = sequence[(frames // 30) % len(sequence)]
                self.telemetry = self.adapter.step(PokemonAction(direction, {"frames": 8}))
                frames += 8
            self._observe_control()
            if self.control_tracker.has_control():
                break
            self.telemetry = self.adapter.step(PokemonAction("WAIT", {"frames": 2}))
            frames += 2
            self._observe_control()

    def _leave_bedroom_safe(self) -> None:
        start_area = int(self.telemetry.area_id)
        start_pos = self._xy()

        candidate_paths = [
            ["DOWN", "DOWN", "RIGHT", "DOWN"],
            ["RIGHT", "DOWN", "DOWN", "LEFT", "DOWN"],
            ["LEFT", "DOWN", "DOWN", "RIGHT", "DOWN"],
            ["UP", "RIGHT", "DOWN", "DOWN", "DOWN"],
        ]

        for path in candidate_paths:
            if self._run_sequence(path):
                curr_y = self._xy()[1]
                if int(self.telemetry.area_id) != start_area or curr_y >= start_pos[1] + 2:
                    return

        # Fallback: try a few exploratory steps
        for _ in range(8):
            moved = False
            for direction in ["DOWN", "RIGHT", "LEFT", "UP"]:
                if self._attempt_move(direction):
                    moved = True
                    break
            curr_y = self._xy()[1]
            if int(self.telemetry.area_id) != start_area or curr_y >= start_pos[1] + 2:
                return
            if not moved:
                self.telemetry = self.adapter.step(PokemonAction("WAIT", {"frames": 8}))
                self._observe_control()

        print("[warn] failed to leave bedroom safely; continuing anyway.")

    def _extract_context(self, telemetry: PokemonTelemetry) -> str:
        if telemetry.in_battle:
            return "battle"
        if telemetry.extra.get("joy_ignore", 0):
            return "cutscene"
        if telemetry.in_grass:
            return "grass"
        return "overworld"

    def _action_budget(self, action: Optional[str]) -> int:
        budgets = {
            "A": 18,
            "B": 18,
            "FIGHT": 18,
            "WALK": 24,
            "UP": 24,
            "DOWN": 24,
            "LEFT": 24,
            "RIGHT": 24,
            "WAIT": 30,
            "START": 30,
            "SELECT": 30,
        }
        if not action:
            return self.default_action_budget
        return budgets.get(action.upper(), self.default_action_budget)

    def _rate_limited(self) -> bool:
        now = time.time()
        while self.llm_call_timestamps and now - self.llm_call_timestamps[0] > self.llm_rate_window_sec:
            self.llm_call_timestamps.popleft()
        return len(self.llm_call_timestamps) >= self.max_llm_calls_per_min

    def _schedule_llm(self) -> None:
        if self.pending is not None:
            return
        context_prompt = self.prompts.get_decision_prompt(self._context_dict())
        memory_hints = ""
        if self.telemetry is not None:
            snippets = self.epmem.top_snippets(int(self.telemetry.area_id), k=2)
            if snippets:
                lines = [
                    f"- Area {ev.area_id} near {ev.pos} did {ev.action} → reward {ev.reward:.1f}"
                    for ev in snippets
                ]
                memory_hints = "\n\nMemory hints:\n" + "\n".join(lines)
        messages = [
            {"role": "system", "content": self.prompts.get_system_prompt()},
            {"role": "user", "content": context_prompt + memory_hints},
        ]
        self.pending = self.pool.submit(self.llm.generate_response, messages)
        self.last_llm_step = self.step_count
        self.llm_call_timestamps.append(time.time())
        self.llm_calls_total += 1

    def _maybe_adopt_plan(self) -> None:
        if self.pending is None or not self.pending.done():
            return
        try:
            response = self.pending.result()
        except Exception as exc:
            print(f"[warn] LLM call failed: {exc}")
            self.pending = None
            return
        self.pending = None
        if not response:
            return

        text = response.upper()
        tokens = [token.strip(',.;:') for token in text.split()]
        action = "WALK"
        direction: Optional[str] = None
        ttl_frames = 180

        for token in tokens:
            if token in {"UP", "DOWN", "LEFT", "RIGHT"}:
                direction = token
            elif token in {"WALK", "FIGHT", "WAIT", "A", "B", "START", "SELECT"}:
                action = token
            elif token.startswith("ACTION="):
                value = token.split("=", 1)[-1]
                if value in {"UP", "DOWN", "LEFT", "RIGHT"}:
                    direction = value
                    action = "WALK"
                elif value in {"WALK", "FIGHT", "WAIT", "A", "B", "START", "SELECT"}:
                    action = value
            elif token.startswith("DURATION="):
                value = token.split("=", 1)[-1]
                if value.isdigit():
                    ttl_frames = int(value)
            elif token.isdigit():
                ttl_frames = int(token)

        if action in {"UP", "DOWN", "LEFT", "RIGHT"}:
            direction = action
            action = "WALK"
        if action == "WALK" and direction is None:
            direction = self.current_walk_direction

        ttl_frames = max(30, ttl_frames)
        adaptive_ttl = max(60, int(self.base_ttl * (1.0 - 0.6 * self.arousal)))
        ttl_frames = max(60, min(ttl_frames, adaptive_ttl))
        self.plan = Plan(
            action=action,
            direction=direction,
            remaining_frames=ttl_frames,
            issued_step=self.step_count,
            area_id=int(self.telemetry.area_id),
            in_battle=bool(self.telemetry.in_battle),
        )

    def _active_plan(self) -> bool:
        if self.plan is None:
            return False
        if self._dialog_refractory > 0 and self.plan.action in {"A", "WAIT"}:
            self.plan = None
            return False
        strong_event = (
            int(self.telemetry.area_id) != self.plan.area_id
            or (self.telemetry.in_battle and not self.plan.in_battle)
        )
        if strong_event or self.plan.remaining_frames <= 0:
            self.plan = None
            return False
        self.plan.area_id = int(self.telemetry.area_id)
        self.plan.in_battle = bool(self.telemetry.in_battle)
        return True

    def _pick_action(self) -> tuple[str, Optional[str], str]:
        if self._active_plan():
            return self.plan.action, self.plan.direction, "plan"

        default_action = self.last_decision or "WALK"
        if default_action not in {"WALK", "UP", "DOWN", "LEFT", "RIGHT", "WAIT", "FIGHT", "A", "B", "START", "SELECT"}:
            default_action = "WALK"

        if default_action == "WALK":
            direction = self.current_walk_direction
        elif default_action in {"UP", "DOWN", "LEFT", "RIGHT"}:
            direction = default_action
            default_action = "WALK"
        else:
            direction = None

        if self._dialog_refractory > 0 and default_action in {"A", "WAIT"}:
            default_action = "WALK"
            direction = self.current_walk_direction
        
        # Force menu escape if stuck in menu
        if self.telemetry is not None and self.control_tracker.phase in {"intro", "menu", "cutscene"} and not self.telemetry.in_battle:
            if self.stuck_frames > 30:  # Stuck in menu for too long
                return "B", None, "menu_escape"  # Press B to exit menu

        if self.telemetry is not None:
            habit = self.epmem.propose(int(self.telemetry.area_id), self.current_walk_direction)
            if habit in {"UP", "DOWN", "LEFT", "RIGHT", "WALK"} and self.arousal < 0.3:  # Lower threshold for exploration
                if habit == "WALK":
                    return "WALK", self.current_walk_direction, "habit"
                return "WALK", habit, "habit"

        if (
            self.goal
            and self.goal.budget > 0
            and not self.goal.done()
            and default_action == "WALK"
            and direction not in {"DOWN", "RIGHT"}
        ):
            direction = direction or "DOWN"

        return default_action, direction, "reuse"

    def _need_llm(self) -> bool:
        if (self.plan is not None and self._active_plan()) or (self.pending is not None and not self.pending.done()):
            self.last_gate_metrics = {"escalate": False, "reason": "plan_or_pending"}
            return False

        telemetry = self.telemetry
        if telemetry is None:
            self.last_gate_metrics = {}
            return True

        context = self._extract_context(telemetry)
        x, y = self._xy()
        pos = (x, y)

        in_menu = self.control_tracker.phase in {"intro", "menu", "cutscene"} and not telemetry.in_battle
        state_key = (telemetry.area_id, telemetry.in_battle, in_menu)
        self.debouncer.update(state_key)
        debounced = self.debouncer.fired()

        self.progress.push(x, y, in_menu, telemetry.in_battle)
        progress_ratio = self.progress.ratio

        pos_static = self.last_position is not None and self.last_position == pos
        self.stuck_frames = self.stuck_frames + 1 if pos_static else 0

        action_budget = self._action_budget(self.last_decision)
        over_budget = self.current_action_frames / max(1, action_budget)
        stuck_threshold = max(self.stuck_base, action_budget * 2)
        stuck_trigger = self.stuck_frames >= stuck_threshold and progress_ratio < self.stuck_progress_floor

        menu_actions = {"OPEN_MENU", "CLOSE_MENU", "START", "SELECT", "A", "B", "WAIT"}
        menu_trap = in_menu and (self.last_decision is None or self.last_decision.upper() not in menu_actions)
        # Force menu escape if stuck in menu for too long
        if in_menu and self.stuck_frames > 50:
            menu_trap = True
        flip_flop = self.flipflop_detector.update(self.last_decision)

        strong_event = False
        if debounced:
            area_changed = self.prev_area_id is None or telemetry.area_id != self.prev_area_id
            battle_started = telemetry.in_battle and not bool(self.prev_in_battle)
            strong_event = area_changed or battle_started

        score = badness_score(over_budget * 0.7, progress_ratio, menu_trap, flip_flop)

        novelty = 1.0 - max(0.0, min(1.0, progress_ratio))
        ctx = Context(game="blue", area_id=telemetry.area_id, method=telemetry.method)
        uncert = 0.0
        try:
            totals = getattr(self.knowledge, "_totals", {})
            post = getattr(self.knowledge, "_post", {})
            totals_ctx = totals.get(ctx)
            pikachu_key = (ctx, 25)
            pikachu_stats = post.get(pikachu_key)
            if totals_ctx and hasattr(totals_ctx, "alpha_r") and hasattr(totals_ctx, "beta_r"):
                a_r = float(getattr(totals_ctx, "alpha_r", 1.0))
                b_r = float(getattr(totals_ctx, "beta_r", 1.0))
                s_r = max(1.0, a_r + b_r)
                uncert += (a_r * b_r) / ((s_r**2) * (s_r + 1.0) + 1e-9)
            if pikachu_stats and hasattr(pikachu_stats, "alpha_p") and hasattr(pikachu_stats, "beta_p"):
                a_p = float(getattr(pikachu_stats, "alpha_p", 1.0))
                b_p = float(getattr(pikachu_stats, "beta_p", 1.0))
                s_p = max(1.0, a_p + b_p)
                uncert += (a_p * b_p) / ((s_p**2) * (s_p + 1.0) + 1e-9)
            uncert = min(1.0, max(0.0, uncert))
        except Exception:
            uncert = 0.0

        threat = 1.0 if (self.control_tracker.phase in {"intro", "cutscene", "menu"} and not telemetry.in_battle) else 0.0
        nov_w = 0.75 if self.mode == "explore" else 0.35
        arousal_next = self.arousal * self.arousal_decay + nov_w * novelty + 0.30 * uncert + 0.15 * threat
        self.arousal = max(0.05, min(1.0, arousal_next))
        eff_threshold = max(0.2, self.base_threshold - 0.7 * self.arousal)
        eff_cooldown = max(0, int(self.base_cooldown * (1.0 - 0.5 * self.arousal)))

        elapsed = self.step_count - self.last_llm_step
        cooldown_active = elapsed < eff_cooldown
        should_consider = elapsed >= self.min_decision_interval
        rate_limited = self._rate_limited()

        escalate = strong_event or (should_consider and (score >= eff_threshold or stuck_trigger))
        reason = "gate_triggered"
        if cooldown_active or rate_limited:
            escalate = False
            reason = "cooldown" if cooldown_active else "rate_limited"
        elif not escalate:
            reason = "threshold_not_met"

        self.current_context = context
        self.last_gate_metrics = {
            "score": float(score),
            "strong_event": bool(strong_event),
            "over_budget": float(over_budget),
            "progress_ratio": float(progress_ratio),
            "menu_trap": bool(menu_trap),
            "flip_flop": bool(flip_flop),
            "stuck_frames": int(self.stuck_frames),
            "stuck_threshold": int(stuck_threshold),
            "stuck_trigger": bool(stuck_trigger),
            "elapsed_since_llm": int(elapsed),
            "novelty": float(novelty),
            "uncertainty": float(uncert),
            "threat": float(threat),
            "arousal": float(self.arousal),
            "eff_threshold": float(eff_threshold),
            "eff_cooldown": int(eff_cooldown),
            "cooldown_active": bool(cooldown_active),
            "rate_limited": bool(rate_limited),
            "debounced_change": bool(debounced),
            "escalate": bool(escalate),
            "reason": reason,
        }
        return escalate

    def _execute(self, action: str, direction: Optional[str], source: str) -> int:
        if action in {"WALK", "UP", "DOWN", "LEFT", "RIGHT"}:
            dir_to_use = direction
            if action in {"UP", "DOWN", "LEFT", "RIGHT"}:
                dir_to_use = action
                action = "WALK"
            if dir_to_use is None:
                dir_to_use = self.current_walk_direction
            if dir_to_use is None:
                dir_to_use = "UP"
            if self.telemetry is not None:
                area_key = int(self.telemetry.area_id)
                dist = self.motion[(area_key, dir_to_use)]
                likely = max(dist.items(), key=lambda kv: kv[1])[0] if dist else None
                if likely == (0, 0):
                    order = ["UP", "RIGHT", "DOWN", "LEFT"]
                    try:
                        idx = order.index(dir_to_use)
                    except ValueError:
                        idx = 0
                    dir_to_use = order[(idx + 1) % len(order)]
            self.current_walk_direction = dir_to_use
            self.telemetry = self.adapter.step(PokemonAction(dir_to_use, {"frames": 24}))
            return 24
        if action == "FIGHT":
            self.telemetry = self.adapter.step(PokemonAction("A", {"frames": 10}))
            return 10
        if action == "WAIT":
            self.telemetry = self.adapter.step(PokemonAction("WAIT", {"frames": 24}))
            return 24
        self.telemetry = self.adapter.step(PokemonAction(action, {"frames": 8}))
        return 8

    def run(self, steps: int) -> None:
        self.step_count = 0
        self.last_decision_source = "LLM"
        self.last_position = self._xy()
        self.step_logs = []
        self.target_found = False
        self.llm_calls_total = 0
        self.llm_call_timestamps.clear()
        self.last_llm_step = -self.decision_cooldown
        self.progress.buffer.clear()
        self.debouncer.key = None
        self.debouncer.stable_for = 0
        self.flipflop_detector.actions.clear()
        self.plan = None
        if self.pending is not None:
            if not self.pending.done():
                self.pending.cancel()
            self.pending = None
        self._dialog_refractory = 0
        self._was_dialog_like = False
        if self.telemetry is not None:
            self._last_xy = self._xy()

        for step in range(steps):
            self.step_count = step
            self._observe_control()
            if self.goal:
                if self.goal.budget > 0:
                    self.goal.budget -= 1
                    if self.goal.done():
                        self.goal = None
                elif self.goal.on_fail:
                    self.goal.on_fail()
                    self.goal = None
            phase = self.control_tracker.phase

            dialog_like = phase in {"intro", "menu", "cutscene"} and not self.telemetry.in_battle
            if self._was_dialog_like and not dialog_like:
                self._dialog_refractory = max(self._dialog_refractory, 20)
                away = OPPOSITE.get(self._facing_dir(), "DOWN")
                start_pos = self._xy()
                self.telemetry = self.adapter.step(PokemonAction(away, {"frames": 24}))
                self._observe_control()
                end_pos = self._xy()
                if end_pos == start_pos:
                    order = ["UP", "RIGHT", "DOWN", "LEFT"]
                    pivot = away if away in order else "DOWN"
                    idx = order.index(pivot)
                    turn = order[(idx + 1) % 4]
                    self.telemetry = self.adapter.step(PokemonAction(turn, {"frames": 24}))
                    self._observe_control()
                    self.current_walk_direction = turn
                else:
                    self.current_walk_direction = away
                self._observe_control()
            self._was_dialog_like = dialog_like
            if self._dialog_refractory > 0:
                self._dialog_refractory -= 1

            if dialog_like:
                self.plan = None
                if self.pending is not None:
                    if not self.pending.done():
                        self.pending.cancel()
                    self.pending = None
                self._tap_button("A", down=2, up=2)
                self.current_action_frames = 0
                self.last_decision = "A"
                self.last_decision_source = "auto-dialog"
                self.stuck_frames = 0
                time.sleep(0.01)
                continue

            with torch.inference_mode():
                encoded = encode_frame(self.telemetry)
                self.last_encoded = encoded
                self.last_summary = self.srfbam.encode_step(encoded)

            need_llm = self._need_llm()
            if need_llm and self.plan is None and self.pending is None:
                self._schedule_llm()

            self._maybe_adopt_plan()

            action, direction, source = self._pick_action()
            frames_used = self._execute(action, direction, source)
            self.last_decision = action
            self.last_decision_source = source

            if source in {"LLM", "plan"}:
                self.current_action_frames = 0
            else:
                self.current_action_frames += frames_used

            if source == "plan" and self.plan is not None:
                self.plan.remaining_frames -= frames_used
                self.plan.area_id = int(self.telemetry.area_id)
                self.plan.in_battle = bool(self.telemetry.in_battle)
                if self.plan.remaining_frames <= 0:
                    self.plan = None

            time.sleep(0.0167)

            gate_decision = self.last_summary.gate_stats.get("decision") if self.last_summary else "N/A"
            curr_pos = self._xy()
            scr_pos = self._screen_xy()
            gate_metrics = self.last_gate_metrics or {}
            score = gate_metrics.get("score")
            log_msg = (
                f"[step {step:04d}] action={action:<5} src={source:<7} "
                f"map=({curr_pos[0]:02d},{curr_pos[1]:02d}) "
                f"scr=({scr_pos[0]:02d},{scr_pos[1]:02d}) "
                f"area={self.telemetry.area_id} joy={self.telemetry.extra.get('joy_ignore', 0)} "
                f"gate={gate_decision} phase={phase}"
            )
            if direction and action == "WALK":
                log_msg += f" dir={direction}"
            if score is not None:
                log_msg += f" score={score:.2f}"
            print(log_msg)

            self.knowledge.step(
                Context(game="blue", area_id=self.telemetry.area_id, method=self.telemetry.method),
                self.telemetry.in_battle,
            )

            if self.telemetry.in_battle and self.telemetry.encounter_species_id:
                self.knowledge.encounter(
                    Context(game="blue", area_id=self.telemetry.area_id, method=self.telemetry.method),
                    self.telemetry.encounter_species_id,
                    25,
                )
                if self.telemetry.encounter_species_id == 25:
                    self.target_found = True

            if self.last_summary is not None:
                log_entry = {
                    "step": step,
                    "action": action,
                    "direction": direction,
                    "source": source,
                    "phase": phase,
                    "area_id": int(self.telemetry.area_id),
                    "position": curr_pos,
                    "screen_position": scr_pos,
                    "in_grass": bool(self.telemetry.in_grass),
                    "in_battle": bool(self.telemetry.in_battle),
                    "joy_ignore": int(self.telemetry.extra.get("joy_ignore", 0)),
                    "menu_state": int(self.telemetry.extra.get("menu_state", 0)),
                    "menu_cursor": int(self.telemetry.extra.get("menu_cursor", 0)),
                    "gate_stats": dict(self.last_summary.gate_stats),
                    "llm_gate": dict(gate_metrics),
                    "context_key": self.last_summary.context_key,
                    "embedding": self.last_summary.embedding.detach().cpu().tolist(),
                    "numeric_features": self.last_summary.numeric_features.detach().cpu().flatten().tolist(),
                    "encounter_species": int(self.telemetry.encounter_species_id or -1),
                    "target_found": bool(self.target_found),
                    "goal": self.goal.name if self.goal else None,
                }
                self.step_logs.append(log_entry)

            prev_area_val = self.prev_area_id if self.prev_area_id is not None else self.telemetry.area_id
            delta_area = int(int(self.telemetry.area_id) != int(prev_area_val))
            delta_pos = int(self.last_position != curr_pos)
            stuck_penalty = 1 if self.stuck_frames > 0 else 0
            reward = (
                5.0 * delta_area
                + 0.2 * delta_pos
                - 0.05 * stuck_penalty
                + (2.0 if self.telemetry.in_battle else 0.0)
            )
            memory_event = Event(
                area_id=int(self.telemetry.area_id),
                pos=curr_pos,
                action=self.last_decision or "WALK",
                reward=float(reward),
                in_battle=bool(self.telemetry.in_battle),
                ts=int(self.step_count),
            )
            self.epmem.add(memory_event, self.current_walk_direction)

            if self._last_xy is not None and (self.last_decision in {"UP", "DOWN", "LEFT", "RIGHT", "WALK"}):
                dx = curr_pos[0] - int(self._last_xy[0])
                dy = curr_pos[1] - int(self._last_xy[1])
                area_key = int(self.telemetry.area_id)
                dir_key = self.current_walk_direction or self.last_decision
                if dir_key == "WALK":
                    dir_key = self.current_walk_direction
                if dir_key in {"UP", "DOWN", "LEFT", "RIGHT"}:
                    self.motion[(area_key, dir_key)][(dx, dy)] += 1
            self._last_xy = curr_pos

            self.prev_area_id = self.telemetry.area_id
            self.prev_in_battle = self.telemetry.in_battle
            self.prev_in_grass = self.telemetry.in_grass
            self.last_position = curr_pos

            time.sleep(0.01)

        steps_executed = min(steps, self.step_count + 1) if steps else self.step_count + 1
        avg_spacing = (steps_executed / self.llm_calls_total) if self.llm_calls_total else float("inf")
        print(f"[run] LLM calls={self.llm_calls_total} avg_spacing={avg_spacing:.1f} steps")
        self._flush_logs()

    def _build_srfbam_context(self) -> Dict[str, Any]:
        if self.last_summary is None:
            return {}
        features = self.last_summary.numeric_features.detach().cpu().flatten().tolist()
        feature_map = dict(zip(NUMERIC_FEATURE_NAMES, features))
        approx_position = (
            int(round(feature_map["x_norm"] * 256)),
            int(round(feature_map["y_norm"] * 256)),
        )
        gate_stats = {
            "decision": self.last_summary.gate_stats.get("decision"),
            "cache_hit_rate": float(self.last_summary.gate_stats.get("cache_hit_rate", 0.0)),
            "reuse_rate": float(self.last_summary.gate_stats.get("reuse_rate", 0.0)),
            "extract_rate": float(self.last_summary.gate_stats.get("extract_rate", 0.0)),
        }
        memory_state = getattr(self.srfbam, "memory_state", None)
        memory_info = None
        if memory_state is not None:
            memory_info = {
                "cache_entries": len(memory_state.frame_cache),
                "summary_files": len(memory_state.file_summary),
                "steps_recorded": memory_state.step_index,
            }
        return {
            "context_key": self.last_summary.context_key,
            "gate": gate_stats,
            "position": approx_position,
            "state_flags": {
                "in_grass": feature_map["in_grass"] >= 0.5,
                "in_battle": feature_map["in_battle"] >= 0.5,
                "joy_ignored": feature_map["joy_ignore"] >= 0.5,
            },
            "embedding_norm": float(self.last_summary.embedding.norm().item()),
            "feature_snapshot": {
                "area_fraction": feature_map["area_norm"],
                "joy_ignore_level": feature_map["joy_ignore"],
                "menu_state_level": feature_map["menu_state"],
                "menu_cursor_level": feature_map["menu_cursor"],
            },
            "memory": memory_info,
        }

    def _context_dict(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "phase": self.control_tracker.phase,
            "control_ready": self.control_tracker.has_control(),
            "area": {
                "id": int(self.telemetry.area_id),
                "mode": "battle" if self.telemetry.in_battle else "overworld",
                "step": int(self.step_count),
            },
            "recent_action": {
                "name": self.last_decision or self.current_walk_direction,
                "source": getattr(self, "last_decision_source", "LLM"),
            },
            "target_found": bool(getattr(self, "target_found", False)),
            "stuck_frames": int(self.stuck_frames),
            "dialog_refractory": int(self._dialog_refractory),
        }
        if self.goal:
            context["goal"] = {"name": self.goal.name, "budget": int(self.goal.budget)}
        srfbam_context = self._build_srfbam_context()
        if srfbam_context:
            context["srfbam"] = srfbam_context
        return context

    def _flush_logs(self) -> None:
        if not self.step_logs:
            return
        out_path = self.log_dir / f"run_{int(time.time())}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for record in self.step_logs:
                json.dump(record, handle)
                handle.write("\n")
        self.step_logs.clear()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-integrated Pokemon Blue controller")
    parser.add_argument("--rom", required=True, help="Path to Pokemon Blue ROM")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--min-decision-interval", type=int, default=45)
    parser.add_argument("--decision-cooldown", type=int, default=120)
    parser.add_argument("--state-debounce", type=int, default=6)
    parser.add_argument("--progress-window", type=int, default=60)
    parser.add_argument("--progress-floor", type=float, default=0.10)
    parser.add_argument("--escalation-threshold", type=float, default=1.6)
    parser.add_argument("--llm-rate-window-sec", type=float, default=60.0)
    parser.add_argument("--max-llm-calls-per-min", type=int, default=3)
    parser.add_argument("--default-budget", type=int, default=36)
    parser.add_argument("--stuck-base", type=int, default=48)
    parser.add_argument("--flipflop-threshold", type=int, default=3)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--verbosity", default="medium")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--mode", choices=("explore", "focus"), default="explore")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_config = LLMConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        verbosity=args.verbosity,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
    )
    controller = LLMPokemonController(
        rom_path=args.rom,
        llm_config=llm_config,
        decision_interval=args.decision_interval,
        visual=args.visual,
        min_decision_interval=args.min_decision_interval,
        decision_cooldown=args.decision_cooldown,
        state_debounce=args.state_debounce,
        progress_window=args.progress_window,
        progress_floor=args.progress_floor,
        escalation_threshold=args.escalation_threshold,
        llm_rate_window_sec=args.llm_rate_window_sec,
        max_llm_calls_per_min=args.max_llm_calls_per_min,
        default_budget=args.default_budget,
        stuck_base=args.stuck_base,
        flipflop_threshold=args.flipflop_threshold,
        mode=args.mode,
    )
    try:
        controller.start_game()
        controller.run(args.steps)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
