"""Resolve actual CLI configs in a fresh process before Kit selects its USD runtime."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("action", ["teleop", "play", "train"])
@pytest.mark.parametrize("task", ["Template-G1-Upper-v0", "Template-G1-Upper-Mobile-v0"])
def test_upper_cli_reaches_simulation_launch_without_importing_usd(action, task):
    script = textwrap.dedent("""\
        import sys

        class PrelaunchUsdGuard:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pxr" or fullname.startswith("pxr."):
                    raise RuntimeError(f"USD imported before SimulationApp: {fullname}")

        sys.meta_path.insert(0, PrelaunchUsdGuard())

        import isaaclab.app

        class LaunchReached(Exception):
            pass

        def stop_before_launch(env_cfg, args):
            assert not any(name == "pxr" or name.startswith("pxr.") for name in sys.modules)
            assert "isaaclab.sim.spawners.from_files.from_files" not in sys.modules
            assert env_cfg.scene.robot.spawn.func.__name__ == "spawn_g1_with_arm_collisions"
            raise LaunchReached

        isaaclab.app.launch_simulation = stop_before_launch

        from g1.rl.__main__ import main

        try:
            main([
                sys.argv[1], "--task", sys.argv[2],
                "--checkpoint", "logs/rsl_rl/g1_upper_mobile/2026-11-16_22-37-23/model_29999.pt",
            ])
        except LaunchReached:
            print("reached launch without USD")
        else:
            raise AssertionError("Simulation launcher was not reached")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script, action, task],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reached launch without USD" in result.stdout
