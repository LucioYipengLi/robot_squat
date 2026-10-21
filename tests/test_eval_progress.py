"""Progress transport, terminal rendering, ETA and subprocess lifecycle regressions."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from g1.rl.evaluation import jobs, progress
from g1.rl.evaluation.config import load_suite


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _job(identifier="job-a"):
    return {
        "job_id": identifier,
        "checkpoint": {"label": "early", "path": "model.pt", "sha256": "abc"},
        "module": "tracking",
        "condition_id": "arm_0000_waist_0000",
        "arm_fraction": 0.0,
        "waist_fraction": 0.0,
        "axis_step": 0,
        "policy_output": "mean",
        "seed": 17,
    }


def test_writer_throttles_but_flushes_stage_changes_and_rejects_stale_attempts(tmp_path):
    clock = Clock()
    path = tmp_path / "progress.json"
    writer = progress.ProgressWriter(path, "job-a", "attempt-1", clock=clock)
    writer.update("evaluating", episodes_total=8, batches_total=2, batch_steps_total=360)
    writer.update(episodes_completed=1)
    assert progress.read_progress(path, "job-a", "attempt-1")["episodes_completed"] == 0
    clock.now = 0.5
    writer.update(episodes_completed=2)
    assert progress.read_progress(path, "job-a", "attempt-1")["episodes_completed"] == 2
    writer.update("saving_results")
    assert progress.read_progress(path, "job-a", "attempt-1")["stage"] == "saving_results"
    assert progress.read_progress(path, "job-a", "attempt-2") is None
    assert progress.read_progress(path, "another-job", "attempt-1") is None
    path.write_text('{"stage":', encoding="utf-8")
    assert progress.read_progress(path, "job-a", "attempt-1") is None


def test_progress_io_failure_does_not_abort_worker(tmp_path, monkeypatch):
    def locked_file(*args):
        raise PermissionError("Windows file is temporarily open")

    monkeypatch.setattr(jobs, "write_json", locked_file)
    writer = progress.ProgressWriter(tmp_path / "progress.json", "job", "attempt")
    writer.update("starting_simulation")
    writer.update("finished")


@pytest.mark.parametrize(
    "field,value",
    [
        ("episodes_completed", -1),
        ("episodes_completed", True),
        ("episodes_completed", 9),
        ("stage", "unknown"),
        ("stage_elapsed_s", float("nan")),
        ("worker_elapsed_s", "12"),
    ],
)
def test_reader_ignores_invalid_telemetry(tmp_path, field, value):
    path = tmp_path / "progress.json"
    writer = progress.ProgressWriter(path, "job-a", "attempt-1")
    writer.update(episodes_total=8, force=True)
    state = json.loads(path.read_text())
    state[field] = value
    path.write_text(json.dumps(state), encoding="utf-8")
    assert progress.read_progress(path, "job-a", "attempt-1") is None


def test_eta_includes_initialization_and_shutdown_but_excludes_cached_cells():
    clock = Clock()
    display = progress.EvaluationProgress(5, mode="off", clock=clock)
    display.set_cached(2)
    display.start_job(_job(), episodes=8, batches=2, steps=360)
    clock.now = 30
    assert display.eta() is None
    # The full cell, including boot/shutdown, takes 50 seconds.
    clock.now = 50
    display.complete_job()
    assert display.eta() == 100
    display.start_job(_job("job-b"), episodes=8, batches=2, steps=360)
    clock.now = 60
    assert display.eta() == 90
    clock.now = 110
    assert display.eta() is None  # A slower-than-average launch has an unknown finish.
    display.fail_job("failed")
    assert display.completed == 1
    assert display.cached == 2


def test_plain_log_is_bounded_has_heartbeats_and_preserves_interrupted_summary():
    clock, stream = Clock(), io.StringIO()
    with progress.EvaluationProgress(3, mode="plain", stream=stream, clock=clock) as display:
        display.set_cached(1)
        display.start_job(_job(), episodes=800, batches=7, steps=360)
        baseline_lines = len(stream.getvalue().splitlines())
        for number in range(100):
            clock.now = number / 100
            display.update(None)
        assert len(stream.getvalue().splitlines()) == baseline_lines
        clock.now = 15
        display.update(None)
        assert len(stream.getvalue().splitlines()) == baseline_lines + 1
        display.fail_job("续跑：--resume outputs/eval/test", interrupted=True)
    text = stream.getvalue()
    assert "\x1b" not in text
    assert "总进度" in text and "1/3" in text
    assert "当前批次控制步" in text
    assert "执行中断" in text and "--resume" in text
    assert display.completed == 0


@pytest.mark.parametrize("columns", [80, 120])
def test_auto_tty_updates_same_block_and_off_emits_nothing(monkeypatch, columns):
    monkeypatch.setattr(progress, "_supports_ansi", lambda stream: True)
    monkeypatch.setattr(progress.shutil, "get_terminal_size", lambda **kwargs: progress.os.terminal_size((columns, 24)))
    clock, stream = Clock(), io.StringIO()
    display = progress.EvaluationProgress(2, stream=stream, clock=clock)
    display.set_cached(0)
    display.start_job(_job(), episodes=8, batches=2, steps=360)
    assert "\x1b[4F" in stream.getvalue()
    assert "预计剩余" in stream.getvalue()
    quiet = io.StringIO()
    with progress.EvaluationProgress(1, mode="off", stream=quiet) as disabled:
        disabled.set_cached(1)
        disabled.reporting()
        disabled.finish(success=True)
    assert quiet.getvalue() == ""


def test_output_pipe_failure_cannot_abort_evaluation():
    class ClosedPipe(io.StringIO):
        def write(self, value):
            raise BrokenPipeError("consumer closed pipe")

    with progress.EvaluationProgress(1, stream=ClosedPipe(), mode="plain") as display:
        display.start_job(_job(), episodes=8, batches=2, steps=360)
        display.complete_job()
        display.finish(success=True)
    assert display.mode == "off" and display.completed == 1


def test_quiet_mode_preserves_actionable_failure_diagnostics():
    stream = io.StringIO()
    with progress.EvaluationProgress(1, mode="off", stream=stream) as display:
        display.fail_job("日志 worker.log；续跑 --resume test")
        display.finish(success=False)
    assert "worker.log" in stream.getvalue() and "--resume" in stream.getvalue()


def test_stubborn_interrupted_worker_is_killed_and_reaped():
    operations = []

    class Stubborn:
        def poll(self):
            return None

        def terminate(self):
            operations.append("terminate")

        def wait(self, timeout):
            operations.append("wait")
            if "kill" not in operations:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 1

        def kill(self):
            operations.append("kill")

    jobs._stop_worker(Stubborn())
    assert operations == ["terminate", "wait", "kill", "wait"]


def test_real_child_progress_then_resume_skips_without_a_new_process(tmp_path, monkeypatch):
    config = load_suite("g1_core")
    config["benchmark"].update(episodes_per_seed=4, seeds=[17])
    job = _job()
    manifest = {"config": config, "runtime": {"num_envs": 2}, "jobs": [job]}
    manifest_path = tmp_path / "manifest.json"
    jobs.write_json(manifest_path, manifest)
    child = tmp_path / "child.py"
    child.write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "from g1.rl.evaluation.jobs import write_json, result_path\n"
        "from g1.rl.evaluation.progress import ProgressWriter, PROGRESS_ID_ENV\n"
        "directory = Path(sys.argv[1])\n"
        "manifest = json.loads((directory/'manifest.json').read_text())\n"
        "job = manifest['jobs'][0]\n"
        "writer = ProgressWriter(result_path(directory,job).parent/'progress.json',\n"
        "                        job['job_id'],os.environ[PROGRESS_ID_ENV])\n"
        "writer.update('starting_simulation', episodes_total=4)\n"
        "time.sleep(.35)\n"
        "writer.update('evaluating',episodes_completed=2,batches_total=2,batches_completed=1,force=True)\n"
        "time.sleep(.35)\n"
        "write_json(result_path(directory,job),dict(job,status='complete'))\n"
        "writer.update('closing_simulation',episodes_completed=4)\n",
        encoding="utf-8",
    )
    real_popen, launches = subprocess.Popen, []

    def launch(command, **kwargs):
        launches.append(command)
        return real_popen([sys.executable, str(child), str(tmp_path)], **kwargs)

    monkeypatch.setattr(jobs.subprocess, "Popen", launch)
    real_read = jobs.read_result
    monkeypatch.setattr(jobs, "read_result", lambda directory, job, manifest=None: real_read(directory, job))
    stream = io.StringIO()
    with progress.EvaluationProgress(1, mode="plain", stream=stream) as display:
        records, complete = jobs.execute_jobs(tmp_path, manifest, progress=display)
        display.finish(success=complete)
    assert complete and len(records) == 1 and len(launches) == 1
    state = json.loads((jobs.result_path(tmp_path, job).parent / "progress.json").read_text(encoding="utf-8"))
    assert state["stage"] == "finished" and state["episodes_completed"] == 4
    assert "启动仿真" in stream.getvalue() and "执行评估" in stream.getvalue()
    assert "1/1 (100%)" in stream.getvalue()
    with progress.EvaluationProgress(1, mode="off") as display:
        records, complete = jobs.execute_jobs(tmp_path, manifest, progress=display)
        display.finish(success=complete)
    assert complete and display.cached == 1 and display.completed == 0
    assert len(launches) == 1


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_or_interrupted_worker_keeps_completed_count_and_reaps_process(tmp_path, monkeypatch, interrupted):
    config = load_suite("g1_core")
    config["benchmark"].update(episodes_per_seed=4, seeds=[17])
    manifest = {"config": config, "runtime": {"num_envs": 2}, "jobs": [_job()]}
    stopped = []

    class FailedProcess:
        def wait(self, timeout):
            if interrupted and not stopped:
                raise KeyboardInterrupt
            return 1

        def poll(self):
            return None

        def terminate(self):
            stopped.append("terminated")

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    stream = io.StringIO()
    with progress.EvaluationProgress(1, mode="plain", stream=stream) as display:
        records, complete = jobs.execute_jobs(tmp_path, manifest, progress=display)
        display.finish(success=complete)
    assert records == [] and not complete and display.completed == 0
    assert "1/1 (100%)" not in stream.getvalue()
    assert "--resume" in stream.getvalue()
    assert bool(stopped) == interrupted
    state = json.loads((jobs.result_path(tmp_path, _job()).parent / "progress.json").read_text(encoding="utf-8"))
    assert state["stage"] == ("interrupted" if interrupted else "failed")
