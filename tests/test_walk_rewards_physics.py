"""Opt-in actual-contact verification of WALK shaping without long training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_WALK_REWARD_SMOKE") != "1", reason="Opt-in: requires Isaac Sim and a GPU")
def test_walk_rewards_match_real_contact_timing_and_preserve_control(tmp_path):
    root = Path(__file__).resolve().parents[1]
    log, report = tmp_path / "physics.log", tmp_path / "summary.json"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(root / "tests/_walk_rewards_physics.py"), "--output", str(report)],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            timeout=240,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-14000:]
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["reward_checks"] == 48 and data["positive_reference_frames"] > 0
    assert data["physical_resets"] > 0 and data["terminal_and_partial_reset_metrics_checked"]
    assert data["physical_state_targets_and_timers_preserved"]
