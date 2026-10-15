# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility launcher for ``uv run python -m g1.rl teleop --task ...``."""

import sys

from g1.rl.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["teleop", *sys.argv[1:]]))
