"""Opt-in end-to-end headless PhysX check using an explicitly synthetic policy.

Set G1_EVAL_PHYSICS_SMOKE=1 to run. Downloads/Isaac Sim startup are environment
dependent, so ordinary unit runs do not launch the simulator.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from g1.rl.evaluation.config import load_suite


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_EVAL_PHYSICS_SMOKE") != "1", reason="Opt-in: starts Isaac Sim and uses a GPU")
def test_real_headless_worker_and_tensorboard_artifact(tmp_path):
    """Exercise both modules/outputs, actual terminal records and source-locked resume."""
    checkpoint = tmp_path / "synthetic" / "model.pt"
    # Infer dimensions from the actual loaded USD, including any finger joints.
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_eval_physics_checkpoint.py")), str(checkpoint)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    suite = load_suite("g1_robustness")
    suite["suite"] = "g1_physics_smoke_v1"
    suite["benchmark"].update(
        episode_length_s=0.2, episodes_per_seed=4, seeds=[17], disturbance_interval_s=[0.033, 0.05]
    )
    suite["modules"][1]["arm_fractions"] = [0.2]
    suite["policy_outputs"] = ["mean", "sample"]
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(suite), encoding="utf-8")
    output = tmp_path / "evaluation"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "g1.rl",
            "eval",
            "--suite",
            str(config),
            "--checkpoints",
            str(checkpoint),
            "--num_envs",
            "4",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode:
        failures = list(output.glob("cases/*/error.json"))
        failure = failures[0].read_text(encoding="utf-8") if failures else completed.stderr
        pytest.fail(f"{completed.stdout}\n{failure}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("cases/*/result.json")]
    assert len(results) == 4
    assert all(len(result["episodes"]) == 4 for result in results)
    telemetry = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("cases/*/progress.json")]
    assert len(telemetry) == 4
    assert all(state["stage"] == "finished" and state["episodes_completed"] == 4 for state in telemetry)
    assert "总进度" in completed.stderr and "评估完成" in completed.stderr
    assert all(
        0 <= episode["disturbance_count"] <= episode["steps"] for result in results for episode in result["episodes"]
    )
    pointer = json.loads((output / "report_latest.json").read_text())
    assert list(Path(pointer["tensorboard_dir"]).rglob("events.out.tfevents.*"))
    resumed = subprocess.run(
        [sys.executable, "-m", "g1.rl", "eval", "--resume", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(list(output.glob("cases/*/result.json"))) == 4
