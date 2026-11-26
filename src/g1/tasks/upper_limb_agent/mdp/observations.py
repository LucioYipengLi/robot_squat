"""Upper-body proprioception and wrist link poses in the command's pelvis frame."""

from __future__ import annotations

import math
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
from isaaclab.utils.math import compute_pose_error, quat_from_angle_axis, quat_mul

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
    "lower_pelvis_offset",
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


def lower_pelvis_offset(env: ManagerBasedRLEnv, action_name: str = "pelvis_offset") -> torch.Tensor:
    """Effective double-ankle-frame pelvis offset [m], shape [N,2]."""
    return env.action_manager.get_term(action_name).processed_actions


def base_height(env: ManagerBasedRLEnv, asset_name: str = "robot") -> torch.Tensor:
    """Pelvis height above its environment origin [m], shape [N,1]."""
    return env.scene[asset_name].data.root_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3]


def end_effector_pose(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Both current wrist poses relative to the command's pelvis link, flattened [14]."""
    position, orientation = env.command_manager.get_term(command_name).current_pose_b()
    return torch.cat((position, orientation), dim=-1).flatten(1)


class TargetPoseNoise:
    """Share one held world-frame target perturbation across both policy terms."""

    def __init__(self, env, cfg):
        self.period_steps = max(1, math.ceil(cfg.target_noise_period_s / env.step_dt))
        self.position_limit = cfg.target_position_noise_m
        self.rotation_limit = cfg.target_rotation_noise_rad
        self.position = torch.zeros((env.num_envs, 2, 3), device=env.device)
        self.rotation = torch.zeros((env.num_envs, 2, 4), device=env.device)
        self.rng = torch.Generator(device=env.device).manual_seed(
            env.cfg.seed if env.cfg.seed is not None else torch.initial_seed()
        )
        self.next_refresh_step = 0

    def reset(self, env_ids):
        # Project onto the unit ball so the configured bounds are vector magnitudes.
        shape = self.position[env_ids].shape
        offsets = torch.rand((2, *shape), device=self.position.device, generator=self.rng) * 2.0 - 1.0
        offsets = offsets / offsets.norm(dim=-1, keepdim=True).clamp_min(1.0)
        self.position[env_ids] = offsets[0] * self.position_limit
        rotation = offsets[1]
        self.rotation[env_ids] = quat_from_angle_axis(rotation.norm(dim=-1) * self.rotation_limit, rotation)

    def targets(self, step: int, term) -> torch.Tensor:
        if step >= self.next_refresh_step:
            self.reset(slice(None))
            self.next_refresh_step = step + self.period_steps
        goal = term.goal_pose_w
        observed = torch.cat((goal[..., :3] + self.position, quat_mul(self.rotation, goal[..., 3:])), dim=-1)
        return term.targets_in_pelvis(observed)


def observed_wrist_targets(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return env._target_pose_noise.targets(env.common_step_counter, term)


def end_effector_pose_error(env: ManagerBasedRLEnv, command_name: str, noisy_target: bool = False) -> torch.Tensor:
    """Left/right pose errors [12] in the live pelvis frame, zero for inactive arms.

    Each wrist contributes target-minus-current position [m, 3] and the shortest
    current-to-target rotation vector [rad, 3], both expressed along pelvis axes.
    """
    term = env.command_manager.get_term(command_name)
    position, orientation = term.current_pose_b()
    target = observed_wrist_targets(env, command_name) if noisy_target else term.command
    target = target.reshape(-1, 2, 7)
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
