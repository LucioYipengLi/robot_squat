"""Opt-in real standing and headless Upper train/play with the frozen Lower actor."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_FROZEN_STAND_SMOKE") != "1", reason="Opt-in: requires Isaac Sim and a GPU")
def test_frozen_lower_stands_and_handles_upper_motion(tmp_path):
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "standing.json"
    log = tmp_path / "standing.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(project / "tests/_frozen_standing_physics.py"), "--output", str(output)],
            cwd=project,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
            timeout=240,
            check=False,
        )
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-16000:]
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert set(summary) == {"standing", "upper_motion"}
    assert all(row["failures"] == 0 for row in summary.values())
