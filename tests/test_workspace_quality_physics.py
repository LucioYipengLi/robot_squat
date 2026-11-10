"""Opt-in real PhysX verification for the zero-load spatial workspace prior."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_WORKSPACE_Q_SMOKE") != "1", reason="Opt-in: requires Isaac Sim and a GPU")
@pytest.mark.parametrize("task", ["Template-G1-Upper-v0", "Template-G1-Upper-Mobile-v0"])
def test_workspace_quality_reward_and_logging_in_real_physics(tmp_path, task):
    root = Path(__file__).resolve().parents[1]
    output, log = tmp_path / "summary.json", tmp_path / "physics.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [
                sys.executable,
                str(root / "tests/_workspace_quality_physics.py"),
                "--task",
                task,
                "--output",
                str(output),
            ],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            timeout=240,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-14000:]
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["reward_checks"] == 12
    assert data["zero_weight_steps"] == data["positive_weight_steps"] == 6
    assert data["episode_log_checks"] > 0
    assert data["partial_reset_checked"] and data["physical_state_targets_and_actions_preserved"]
    assert data["lower_calls"] == (12 if "Mobile" in task else 0)
