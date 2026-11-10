"""Opt-in physical validation of Upper target-speed and substep acceleration rewards."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_UPPER_SUBSTEP_SMOKE") != "1", reason="Opt-in: requires Isaac Sim and a GPU")
@pytest.mark.parametrize("task", ["Template-G1-Upper-v0", "Template-G1-Upper-Mobile-v0"])
def test_complete_acceleration_sampling_preserves_actions_and_other_rewards(tmp_path, task):
    root = Path(__file__).resolve().parents[1]
    output, log = tmp_path / "summary.json", tmp_path / "physics.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(root / "tests/_upper_substep_physics.py"), "--task", task, "--output", str(output)],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            timeout=240,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-14000:]
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["reward_checks"] == 49
    assert data["maximum_first_substep_squared_cost"] > 0
    assert data["maximum_first_end_effector_squared_cost"] > 0
    assert data["end_effector_reference_body"] == "pelvis"
    assert data["end_effector_substeps_and_episode_logs_checked"]
    assert data["timeout_resets"] > 0
    assert data["episode_log_checks"] > 0
    assert data["maximum_target_velocity_oracle_error"] < 1e-4
    assert data["target_speed_disabled_control_steps"] == 24
    assert data["target_speed_enabled_control_steps"] == 25
    assert data["other_rewards_and_targets_unchanged"]
    assert data["zero_weight_partial_reset_checked"]
    assert data["lower_inference_calls"] == (49 if "Mobile" in task else 0)
