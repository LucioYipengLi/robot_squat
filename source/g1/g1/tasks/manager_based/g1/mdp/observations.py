# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms specific to the height-based balance and squat-walk tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_command_height(env: ManagerBasedRLEnv, command_name: str = "task_command") -> torch.Tensor:
    """Pelvis height offset slice (1-D) of the unified task command.

    Returns the first element of ``[h_offset, vx, vy, wz, com_dx, com_dy]``: the commanded
    pelvis height offset relative to the default standing height [m].
    """
    return env.command_manager.get_command(command_name)[:, :1]


def task_command_velocity(env: ManagerBasedRLEnv, command_name: str = "task_command") -> torch.Tensor:
    """Base-frame velocity slice (3-D) of the unified task command.

    Returns elements 1-3 of ``[h_offset, vx, vy, wz, com_dx, com_dy]``: the commanded
    base-frame linear velocities (x, y) [m/s] and yaw angular velocity [rad/s].
    """
    return env.command_manager.get_command(command_name)[:, 1:4]


def task_command_com(env: ManagerBasedRLEnv, command_name: str = "task_command") -> torch.Tensor:
    """双足水平系内的 COM 有效目标偏置，shape (N, 2) [m]。

    仅暴露限速后的目标，不暴露实际 COM、接触状态或模式/启用标志。
    行走时指令项将该切片清零，奖励另按 STAND/SQUAT 门控。
    """
    return env.command_manager.get_command(command_name)[:, 4:6]


def upper_body_joint_pos_target_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Upper-body joint PD targets relative to their default positions [rad].

    Returns the commanded PD position targets for the upper-body joints in ``asset_cfg``
    (arms + waist) minus their default positions. These targets are written by
    ``UpperBodyDisturbanceEvent`` and, per the manager-based step order (interval events fire
    before observation computation), lead the actual limb motion by at least one step. Feeding
    them to the policy gives the lower body a feed-forward cue to anticipate the momentum
    disturbance produced by upper-limb motion.

    Note:
        Only the *commanded* target enters this term. External pushes / contact act on the
        measured joint positions (already observed via ``joint_pos_rel``) and never corrupt
        this reading, keeping target uncertainty and external disturbance decoupled.
    """
    asset = env.scene[asset_cfg.name]
    target = asset.data.joint_pos_target.torch[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return target - default
