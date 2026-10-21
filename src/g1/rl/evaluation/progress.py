"""Best-effort worker telemetry and terminal progress, independent of evaluation data."""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import sys
import time
import unicodedata
from contextlib import suppress
from pathlib import Path

PROGRESS_ID_ENV = "G1_EVAL_PROGRESS_ID"
STAGES = {
    "starting_worker": "启动评估进程",
    "validating": "检查配置与运行环境",
    "importing": "加载仿真依赖",
    "configuring": "准备测试场景",
    "starting_simulation": "启动仿真",
    "creating_environment": "创建环境",
    "loading_policy": "加载模型",
    "evaluating": "执行评估",
    "saving_results": "保存结果",
    "closing_simulation": "关闭仿真",
    "finished": "单元完成",
    "failed": "单元失败",
    "interrupted": "执行中断",
}
_COUNTERS = (
    "episodes_completed",
    "episodes_total",
    "batches_completed",
    "batches_total",
    "batch_index",
    "batch_episodes_completed",
    "batch_episodes_total",
    "batch_step",
    "batch_steps_total",
)


class ProgressWriter:
    """Atomically publish at most twice per second; stage transitions flush immediately.

    A fresh attempt ID prevents a restarted worker from inheriting stale progress.
    Telemetry I/O is best effort and never determines episode validity or results.
    """

    def __init__(self, path: Path, job_id: str, attempt_id: str, *, interval=0.5, clock=None):
        self.path = path
        self.clock = clock or time.monotonic
        self.interval = interval
        self.started = self.clock()
        self.stage_started = self.started
        self.last_write = -math.inf
        self.state = {
            "schema_version": 1,
            "job_id": job_id,
            "attempt_id": attempt_id,
            "stage": "starting_worker",
            **dict.fromkeys(_COUNTERS, 0),
        }

    def update(self, stage: str | None = None, *, force=False, **values) -> None:
        from .jobs import write_json

        now = self.clock()
        if stage is not None and stage != self.state["stage"]:
            if stage not in STAGES:
                raise ValueError(f"Unknown evaluation progress stage: {stage}")
            self.state["stage"] = stage
            self.stage_started = now
            force = True
        self.state.update(values)
        if not force and now - self.last_write < self.interval:
            return
        payload = {
            **self.state,
            "worker_elapsed_s": max(0.0, now - self.started),
            "stage_elapsed_s": max(0.0, now - self.stage_started),
            "updated_unix_s": time.time(),
        }
        # A reader can briefly hold the file during Windows replace(). Retry
        # at the next update; the real result file remains authoritative.
        with suppress(OSError):
            write_json(self.path, payload)
        self.last_write = now


def read_progress(path: Path, job_id: str, attempt_id: str) -> dict | None:
    """Ignore missing, partially written, malformed or previous-attempt telemetry."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return None
        if value.get("job_id") != job_id or value.get("attempt_id") != attempt_id:
            return None
        if value.get("stage") not in STAGES:
            return None
        for key in _COUNTERS:
            count = value.get(key, 0)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                return None
        for key in ("worker_elapsed_s", "stage_elapsed_s", "updated_unix_s"):
            number = value.get(key)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number < 0
            ):
                return None
        for done, total in (
            ("episodes_completed", "episodes_total"),
            ("batches_completed", "batches_total"),
            ("batch_episodes_completed", "batch_episodes_total"),
            ("batch_step", "batch_steps_total"),
        ):
            if value.get(done, 0) > value.get(total, 0):
                return None
        return value
    except (OSError, ValueError, UnicodeError, OverflowError):
        return None


def finalize_progress(path: Path, job_id: str, attempt_id: str, stage: str, elapsed_s: float, **values) -> dict:
    """Parent confirms terminal state after reaping and checking the real result.

    Some simulator launchers exit the interpreter from close(), bypassing Python
    statements after the launch context. A child cannot acknowledge its own exit.
    """
    from .jobs import write_json

    if stage not in ("finished", "failed", "interrupted"):
        raise ValueError(f"Not a terminal progress stage: {stage}")
    previous = read_progress(path, job_id, attempt_id) or dict.fromkeys(_COUNTERS, 0)
    state = {
        **previous,
        **values,
        "schema_version": 1,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "stage": stage,
        "worker_elapsed_s": max(0.0, elapsed_s),
        "stage_elapsed_s": 0.0,
        "updated_unix_s": time.time(),
    }
    with suppress(OSError):
        write_json(path, state)
    return state


def _supports_ansi(stream) -> bool:
    if not getattr(stream, "isatty", lambda: False)() or os.environ.get("TERM") == "dumb":
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_ulong()
        kernel = ctypes.windll.kernel32
        kernel.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode)) and kernel.SetConsoleMode(handle, mode.value | 4))
    except (OSError, ValueError, AttributeError):
        return False


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "估算中"
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _bar(done: int, total: int, width=16) -> str:
    fraction = min(max(done / total, 0), 1) if total else 0
    filled = int(fraction * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({fraction:.0%})"


def _truncate(value: str, width: int) -> str:
    value = value.replace("\n", " ").replace("\r", " ")
    used, characters = 0, []
    for character in value:
        size = 0 if unicodedata.combining(character) else 2 if unicodedata.east_asian_width(character) in "WF" else 1
        if used + size > width:
            break
        used += size
        characters.append(character)
    return "".join(characters)


class EvaluationProgress:
    """One parent-owned display; child stdout stays in worker.log.

    Cell-duration samples include imports, simulator initialization and shutdown.
    Cached cells count toward completion but never inflate this session's ETA.
    """

    def __init__(self, total: int, *, mode="auto", stream=None, clock=None, plain_interval=15.0):
        self.total = total
        self.mode = mode
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.plain_interval = plain_interval
        self.ansi = mode == "auto" and _supports_ansi(self.stream)
        self.cached = 0
        self.completed = 0
        self.durations = []
        self.job = None
        self.job_started = self.started
        self.observed = self.started
        self.state = {}
        self.phase = "准备评估"
        self.detail = ""
        self._lines = 0
        self._last_render = -math.inf
        self._last_key = None
        self._finished = False
        self._interrupted = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if not self._finished:
            if exception_type is not None:
                self.fail_job(str(exception), interrupted=exception_type is KeyboardInterrupt)
            self.finish(success=False, interrupted=exception_type is KeyboardInterrupt)

    def set_cached(self, count: int) -> None:
        self.cached = count
        self.render(force=True)

    def start_job(self, job: dict, *, episodes: int, batches: int, steps: int) -> None:
        self.job = job
        self.job_started = self.clock()
        self.observed = self.job_started
        self.state = {
            "stage": "starting_worker",
            "stage_elapsed_s": 0.0,
            "episodes_total": episodes,
            "batches_total": batches,
            "batch_steps_total": steps,
        }
        self.phase, self.detail = "执行评估", ""
        self.render(force=True)

    def update(self, state: dict | None) -> None:
        if state is not None and state != self.state:
            self.state = state
            self.observed = self.clock()
        self.render()

    def complete_job(self) -> None:
        self.durations.append(self.clock() - self.job_started)
        self.completed += 1
        self.state.update(
            stage="finished",
            stage_elapsed_s=0.0,
            episodes_completed=self.state.get("episodes_total", 0),
            batches_completed=self.state.get("batches_total", 0),
        )
        self.observed = self.clock()
        self.render(force=True)
        self.job = None

    def fail_job(self, detail: str, *, interrupted=False) -> None:
        self._interrupted = interrupted
        self.phase = "执行中断" if interrupted else "评估失败"
        self.detail = detail
        self.state["stage"] = "interrupted" if interrupted else "failed"
        self.state["stage_elapsed_s"] = 0.0
        self.observed = self.clock()
        if self.mode == "off":
            # Quiet progress still preserves the actionable failure location.
            with suppress(OSError, UnicodeError):
                self.stream.write(f"{self.phase}：{detail}\n")
                self.stream.flush()
        self.render(force=True)

    def reporting(self) -> None:
        self.job = None
        self.phase = "生成汇总表与 TensorBoard 日志"
        self.render(force=True)

    def eta(self) -> float | None:
        remaining = self.total - self.cached - self.completed
        if remaining <= 0:
            return 0.0
        if not self.durations:
            return None
        average = statistics.fmean(self.durations[-8:])
        if self.job is None:
            return remaining * average
        current = self.clock() - self.job_started
        # A particularly slow launch can exceed the previous mean. Prefer an
        # unknown ETA over reporting zero while that cell is still running.
        if current >= average:
            return None
        return max(0.0, remaining * average - current)

    def finish(self, *, success: bool, interrupted=False) -> None:
        interrupted = interrupted or self._interrupted
        self.phase = "评估完成" if success else "执行中断" if interrupted else "评估未完成"
        if success:
            self.job = None
        self._finished = True
        self.render(force=True)

    def lines(self, *, width=120) -> list[str]:
        elapsed = self.clock() - self.started
        bar_width = 16 if width >= 110 else 8
        total = (
            f"总进度 {_bar(self.cached + self.completed, self.total, bar_width)} "
            f"| 已用 {_duration(elapsed)} | 预计剩余 {_duration(self.eta())}"
        )
        allocation = f"本次完成 {self.completed} · 已跳过 {self.cached}"
        if self.job is None:
            return [total, f"阶段：{self.phase}", allocation, self.detail or ""]
        context = (
            f"当前：{self.job['checkpoint']['label']} · {self.job['module']} · 双臂 {self.job['arm_fraction']:g} "
            f"· 腰部 {self.job['waist_fraction']:g} · {self.job['policy_output']} · seed {self.job['seed']}"
        )
        stage_time = self.state.get("stage_elapsed_s", 0) + max(0, self.clock() - self.observed)
        stage = (
            f"阶段：{STAGES[self.state['stage']]} | 单元已用 {_duration(self.clock() - self.job_started)} "
            f"| 本阶段 {_duration(stage_time)} | {allocation}"
        )
        counts = (
            f"回合 {_bar(self.state.get('episodes_completed', 0), self.state.get('episodes_total', 0))} "
            f"| 批次 {self.state.get('batches_completed', 0)}/{self.state.get('batches_total', 0)} "
            f"(当前 {self.state.get('batch_index', 0)}) "
            f"| 当前批次控制步 {self.state.get('batch_step', 0)}/{self.state.get('batch_steps_total', 0)}"
        )
        return [total, context, stage, self.detail or counts]

    def render(self, *, force=False) -> None:
        if self.mode == "off":
            return
        now = self.clock()
        total = self.state.get("episodes_total", 0)
        bucket = int(10 * self.state.get("episodes_completed", 0) / total) if total else 0
        key = (self.phase, (self.job or {}).get("job_id"), self.state.get("stage"), bucket, self.completed, self.cached)
        if not force:
            if self.ansi and now - self._last_render < 0.2:
                return
            if not self.ansi and key == self._last_key and now - self._last_render < self.plain_interval:
                return
        if self.ansi:
            width = max(1, shutil.get_terminal_size(fallback=(120, 24)).columns - 1)
            lines = self.lines(width=width)
            prefix = f"\x1b[{self._lines}F" if self._lines else ""
            content = prefix + "".join("\x1b[2K" + _truncate(line, width) + "\n" for line in lines)
            self._lines = len(lines)
        else:
            lines = self.lines()
            content = " | ".join(line for line in lines if line) + "\n"
        try:
            self.stream.write(content)
            self.stream.flush()
        except (OSError, UnicodeError):
            # Closing a log pipe or an unsupported console encoding must not
            # interrupt the model's evaluation or strand its child process.
            self.mode = "off"
        self._last_key, self._last_render = key, now
