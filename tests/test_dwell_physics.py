"""Opt-in PhysX verification of the Dwell reward's rolling-speed sampler."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_DWELL_SMOKE") != "1", reason="Opt-in: launches Isaac Sim and uses a GPU")
def test_dwell_matches_independent_physics_substep_oracle(tmp_path):
    project = Path(__file__).resolve().parents[1]
    output, log = tmp_path / "summary.json", tmp_path / "physics.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(project / "tests/_dwell_physics.py"), "--output", str(output)],
            cwd=project,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            timeout=240,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-14000:]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["maximum_rms_oracle_error_m_s"] < 1e-6
    assert report["positive_dwell_samples"] > 0 and report["partial_reset_preserved"]
    assert report["physics_samples_per_window"] == 12 and report["frozen_weights_unchanged"]
