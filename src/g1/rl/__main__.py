# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run with ``uv run python -m g1.rl train|play|teleop --task ...``."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from contextlib import contextmanager
from functools import partial


@contextmanager
def _runner_adapter(backend, factory):
    """Scope the upstream constructor seam to one synchronous CLI invocation.

    The local IsaacLab entrypoints import create_rsl_rl_runner at module scope and expose no
    runner-factory argument. Replacing that reference for this call reuses their launch,
    video, logging and cleanup logic without editing or copying the upstream scripts.
    """
    from isaaclab_rl.rsl_rl import create_rsl_rl_runner

    original = backend.create_rsl_rl_runner
    if original is not create_rsl_rl_runner:
        raise RuntimeError("Unsupported IsaacLab runner constructor seam; review the project adapter")
    backend.create_rsl_rl_runner = factory
    try:
        yield
    finally:
        backend.create_rsl_rl_runner = original


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="G1 training/playback/teleop with curriculum checkpoints", add_help=False
    )
    parser.add_argument("action", choices=("train", "play", "teleop"))
    parser.add_argument("--curriculum-snapshot", help="Explicit curriculum JSON for a legacy checkpoint only")
    parser.add_argument("--rl_library", choices=("rsl_rl",), default="rsl_rl")
    parser.add_argument("--frontend", choices=("torch",), default="torch")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    selected, remaining = parser.parse_known_args(arguments)
    if "--distributed" in remaining or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Curriculum checkpoints currently support single-process training only")
    if selected.action in ("play", "teleop") and "--train_env_cfg" in remaining:
        parser.error("Curriculum playback requires play_mode(); omit --train_env_cfg")
    if selected.curriculum_snapshot and not any(
        arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in remaining
    ):
        parser.error("--curriculum-snapshot requires --checkpoint")

    from isaaclab.app.sim_launcher import fuse_kit_args
    from isaaclab.utils import configure_logging

    from .runner import create_curriculum_runner

    # The launcher applies --info/--verbose to the root logger and its handlers.
    configure_logging(save_logs_to_file=False)
    if selected.action == "teleop":
        from .teleop import run

        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0], *remaining]
            return run(fuse_kit_args(remaining), snapshot_path=selected.curriculum_snapshot)
        finally:
            sys.argv = original_argv
    backend = importlib.import_module(f"isaaclab_rl.entrypoints.backends.{selected.action}_rsl_rl")
    factory = partial(
        create_curriculum_runner,
        snapshot_path=selected.curriculum_snapshot,
        evaluation=selected.action == "play",
    )
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        with _runner_adapter(backend, factory):
            result = backend.run(fuse_kit_args(remaining))
        return result or 0
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
