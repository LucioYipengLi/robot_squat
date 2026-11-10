"""Synchronize joint states and PD targets, with an explicit root reset for the mobile variant."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["reset_robot_joints", "reset_mobile_root"]


def reset_mobile_root(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_name: str = "robot") -> None:
    """Restore the authored root pose at each environment origin, without randomization."""
    robot = env.scene[asset_name]
    pose = robot.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    velocity = torch.zeros(len(pose), 6, device=pose.device, dtype=pose.dtype)
    robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def reset_robot_joints(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice, asset_name: str = "robot") -> None:
    """Reset all joints and PD commands together, touching only the reset environments.

    All joints start at their defaults. Action terms take ownership of their
    respective subsets on subsequent steps; this function leaves the root pose unchanged.
    """
    robot = env.scene[asset_name]
    position = robot.data.default_joint_pos.torch[env_ids].clone()
    velocity = robot.data.default_joint_vel.torch[env_ids].clone()
    robot.write_joint_position_to_sim_index(position=position, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=velocity, env_ids=env_ids)
    command = robot.actuators.target_command
    command.set_position_index(value=position, env_ids=env_ids)
    command.set_velocity_index(value=velocity, env_ids=env_ids)
    command.set_effort_index(value=torch.zeros_like(position), env_ids=env_ids)
