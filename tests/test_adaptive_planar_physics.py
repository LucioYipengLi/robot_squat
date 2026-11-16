"""Opt-in GPU verification for adaptive sampling in real PhysX and the standard runner."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("G1_ADAPTIVE_SAMPLING_SMOKE") != "1", reason="Opt-in: launches Isaac Sim and uses a GPU"
)
def test_adaptive_sampling_physics_resume_and_logger(tmp_path):
    script = Path(__file__).with_name("_adaptive_planar_physics.py")
    log = tmp_path / "physics.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            cwd=script.parents[1],
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=240,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-10000:]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["gpu_draws_match"] and summary["schedule_restored"] and summary["ppo_iterations"] == 1
