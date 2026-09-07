# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms specific to the height-based balance and squat-walk tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_command_height(env: ManagerBasedRLEnv, command_name: str = "task_command") -> torch.Tensor:
    """Pelvis height offset slice (1-D) of the unified task command.

    Returns the first element of the 4-D command ``[h_offset, vx, vy, wz]``: the commanded
    pelvis height offset relative to the default standing height [m].
    """
    return env.command_manager.get_command(command_name)[:, :1]


def task_command_velocity(env: ManagerBasedRLEnv, command_name: str = "task_command") -> torch.Tensor:
    """Base-frame velocity slice (3-D) of the unified task command.

    Returns elements 1-3 of the 4-D command ``[h_offset, vx, vy, wz]``: the commanded
    base-frame linear velocities (x, y) [m/s] and yaw angular velocity [rad/s].
    """
    return env.command_manager.get_command(command_name)[:, 1:4]
