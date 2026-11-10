"""Opt-in real PhysX command transition and frozen inference verification."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_LOWER_COMMAND_SMOKE") != "1", reason="Opt-in: requires Isaac Sim and a GPU")
def test_frozen_lower_executes_current_commands_and_transitions(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output, log = tmp_path / "summary.json", tmp_path / "physics.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(root / "tests/_lower_command_physics.py"), "--output", str(output)],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            timeout=240,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-16000:]
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["control_steps"] == data["inference_calls"] == data["reward_checks"] == 1021
    assert data["rate_command_dimensions"] == 4
    assert data["vertical_reward_checks"] == data["control_steps"]
    assert data["maximum_vertical_oracle_error"] < 1e-4
    assert data["base_vertical_acc_weight"] == -0.05 and data["base_vertical_acc_soft_limit"] == 4.0
    assert data["partial_reset_preserved"] and data["native_lower_observations_match"]
    assert data["frozen_weights_unchanged"]
    assert data["upper_actions"] == 21 and data["upper_observations"] == 107
    assert data["base_rewards_default_zero"]
    assert data["optional_reward_checks"] == 6 * data["control_steps"]
    assert data["maximum_base_substep_oracle_error"] < 1e-3
    assert data["base_motion_reference"] == "pelvis_link_origin_world"
    assert sum(row["failures"] for row in data["phases"]) == 0, data["phases"]
    for row in data["phases"]:
        if row["phase"] in ("positive", "negative"):
            # Direction response is essential; tighter tracking accuracy is reported,
            # rather than prescribing unmeasured locomotion thresholds here.
            for axis in range(3):
                assert row["steady_actual_mean"][axis][axis] * row["commands"][axis][axis] > 0
        if row["phase"].startswith("height_"):
            assert max(row["steady_height_error_m"]) < 0.05, row
    phases = {row["phase"]: row for row in data["phases"]}
    for index in range(4):
        down = phases["height_down"]["steady_height_mean_m"][index]
        returned = phases["height_return"]["steady_height_mean_m"][index]
        assert returned - down > 0.04
