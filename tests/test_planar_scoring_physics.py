"""Opt-in independent integration oracle for mixed-length Mobile episodes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("G1_PLANAR_SCORING_SMOKE") != "1", reason="Opt-in: launches Isaac Sim and uses a GPU"
)
def test_mixed_episode_scores_and_reward_rates_match_independent_oracle(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output, log = tmp_path / "summary.json", tmp_path / "physics.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(root / "tests/_planar_scoring_physics.py"), "--output", str(output)],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            timeout=240,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-14000:]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["window_checks"] > 5 and report["reward_rate_checks"] > 5
    assert report["early_failures_checked"] == 1 and report["final_level"] >= 1
    assert report["maximum_window_error"] < 1e-6 and report["maximum_reward_rate_error"] < 1e-3
