"""Upper-body proprioception and wrist link poses in the command's pelvis frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.observations import (
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)
from isaaclab.utils.math import compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "base_ang_vel",
    "base_lin_vel",
    "base_height",
    "end_effector_pose",
    "end_effector_pose_error",
    "active_arms",
    "generated_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "lower_velocity_command",
    "lower_height_offset",
    "projected_gravity",
    "term_action",
]


def term_action(env: ManagerBasedRLEnv, action_name: str, previous: bool = False) -> torch.Tensor:
    """One term's raw actions from manager history, zeroed on partial resets.

    Default is the most recently executed action, used in the next observation.
    ``previous=True`` selects the preceding action for difference rewards.
    """
    manager = env.action_manager
    values = manager.prev_action if previous else manager.action
    start = 0
    for name, width in zip(manager.active_terms, manager.action_term_dim, strict=True):
        if name == action_name:
            return values[:, start : start + width]
        start += width
    raise ValueError(f"Unknown action term: {action_name}")


def lower_velocity_command(env: ManagerBasedRLEnv, action_name: str = "lower_command") -> torch.Tensor:
    """Effective base-frame [vx, vy, yaw rate], in m/s, m/s, rad/s."""
    return env.action_manager.get_term(action_name).processed_actions


def lower_height_offset(env: ManagerBasedRLEnv, action_name: str = "height_offset") -> torch.Tensor:
    """Effective target height offset from default standing height [m], shape [N,1]."""
    return env.action_manager.get_term(action_name).processed_actions


def base_height(env: ManagerBasedRLEnv, asset_name: str = "robot") -> torch.Tensor:
    """Pelvis height above its environment origin [m], shape [N,1]."""
    return env.scene[asset_name].data.root_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3]


def end_effector_pose(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Both current wrist poses relative to the command's pelvis link, flattened [14]."""
    position, orientation = env.command_manager.get_term(command_name).current_pose_b()
    return torch.cat((position, orientation), dim=-1).flatten(1)


def end_effector_pose_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Left/right pose errors [12] in the live pelvis frame, zero for inactive arms.

    Each wrist contributes target-minus-current position [m, 3] and the shortest
    current-to-target rotation vector [rad, 3], both expressed along pelvis axes.
    """
    term = env.command_manager.get_term(command_name)
    position, orientation = term.current_pose_b()
    target = term.command.reshape(-1, 2, 7)
    position_error, rotation_error = compute_pose_error(
        position.reshape(-1, 3),
        orientation.reshape(-1, 4),
        target[..., :3].reshape(-1, 3),
        target[..., 3:].reshape(-1, 4),
        rot_error_type="axis_angle",
    )
    errors = torch.cat((position_error, rotation_error), dim=-1).reshape(-1, 2, 6)
    return torch.where(term.active_mask[..., None], errors, 0.0).flatten(1)


def active_arms(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Left/right target validity flags [2], independently of the pose values."""
    return env.command_manager.get_term(command_name).active_mask.float()
