"""Opt-in Newton GL + PhysX test; opens a native viewer and uses a GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_UPPER_TELEOP_SMOKE") != "1", reason="Opt-in: opens Newton GL and uses a GPU")
def test_upper_teleop_native_panel_and_command_lifecycle(tmp_path):
    script = Path(__file__).with_name("_upper_teleop_physics.py")
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
    assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-14000:]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["checks"] == 65 and summary["actor_unchanged"] and summary["input_closed"]
    assert summary["native_key"] and summary["window_closed"]
    assert summary["arm_collisions_active"]
    rows = [json.loads(line) for line in (tmp_path / "session/session.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event"] == "closed"
