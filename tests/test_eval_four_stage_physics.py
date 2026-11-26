"""Opt-in four-stage PhysX smoke with a synthetic actor, not a model performance test."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from g1.rl.evaluation.config import load_suite


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_EVAL_FOUR_STAGE_PHYSICS_SMOKE") != "1", reason="Opt-in: Isaac Sim and GPU")
@pytest.mark.parametrize("suite_name", ["g1_lower_four_stage", "g1_lower_push"])
def test_four_stage_real_physics_and_resume(tmp_path, suite_name):
    checkpoint = tmp_path / "synthetic" / "model.pt"
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_eval_physics_checkpoint.py")), str(checkpoint)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    suite = load_suite(suite_name)
    suite["benchmark"].update(repetitions=2, episode_length_s=0.2, seeds=[17])
    source = tmp_path / "suite.yaml"
    source.write_text(yaml.safe_dump(suite), encoding="utf-8")
    output = tmp_path / "evaluation"
    command = [
        sys.executable,
        "-m",
        "g1.rl",
        "eval",
        "--suite",
        str(source),
        "--checkpoints",
        str(checkpoint),
        "--num_envs",
        "3",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, check=False
    )
    errors = [path.read_text(encoding="utf-8") for path in output.glob("cases/*/error.json")]
    assert completed.returncode == 0, completed.stdout + completed.stderr + "\n".join(errors)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("cases/*/result.json")]
    expected_cells = 1 if suite_name == "g1_lower_push" else 4
    assert len(results) == expected_cells
    if suite_name == "g1_lower_push":
        assert {result["module"] for result in results} == {"squat_push"}
    assert all(len(result["episodes"]) == 6 for result in results)
    for result in results:
        for episode in result["episodes"]:
            assert episode["steps"] == episode["metric_steps"] == 12
            assert episode["attitude_quaternion_order"] == "xyzw"
            assert episode["max_roll_rad"] < 0.5
            if result["module"] == "squat_push":
                assert episode["planned_push_count"] == episode["push_count"] == episode["disturbance_count"] == 2
            elif result["module"] == "walk_axes":
                assert len(episode["command_plan"]) == 2
                expected = sum(6 * (sum(x * x for x in row["command"][1:3])) ** 0.5 for row in episode["command_plan"])
                assert episode["command_speed_sum"] == pytest.approx(expected, abs=1e-6)
            else:
                assert episode["push_count"] == episode["disturbance_count"] == 0
    pointer = json.loads((output / "report_latest.json").read_text())
    report = json.loads(Path(pointer["summary_json"]).read_text(encoding="utf-8"))
    if suite_name == "g1_lower_four_stage":
        assert any("e_walk_percent" in row for row in report["rows"])
    else:
        aggregate = next(row for row in report["rows"] if row["mode"] == "aggregate")
        assert aggregate["attitude_metrics_valid"] == 1
        assert aggregate["push_count"] == 12
        assert "push_recovery_rate" in aggregate
        assert "rmse_height_m" in aggregate
    assert len([row for row in report["rows"] if row["mode"] == "height"]) == 3
    resumed = subprocess.run(
        [sys.executable, "-m", "g1.rl", "eval", "--resume", str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(list(output.glob("cases/*/result.json"))) == expected_cells
