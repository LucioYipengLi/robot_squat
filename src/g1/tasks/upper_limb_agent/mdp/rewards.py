"""Mode-aware wrist tracking and default-pose regularization for the idle arm."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.rewards import (
    action_rate_l2,
    is_terminated,
    joint_pos_limits,
    joint_torques_l2,
    joint_vel_l2,
)
from isaaclab.managers import SceneEntityCfg

from .actions import UpperJointPositionAction
from .frozen_lower_actions import FrozenLowerAction, LowerHeightCommandAction, LowerVelocityCommandAction
from .observations import term_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "action_rate_l2",
    "action_term_rate_l2",
    "base_ang_acc_l2_substeps",
    "base_ang_vel_l2_substeps",
    "base_lin_acc_l2_substeps",
    "base_lin_vel_l2_substeps",
    "base_vertical_acc_excess_l2_substeps",
    "end_effector_relative_lin_acc_l2_substeps",
    "end_effector_orientation_tracking",
    "end_effector_position_tracking",
    "idle_arm_default_pose",
    "joint_acc_l2_substeps",
    "joint_target_speed_excess_l2",
    "joint_pos_limits",
    "joint_torque_limits",
    "joint_torques_l2",
    "joint_vel_l2",
    "lower_command_magnitude_l2",
    "lower_command_overflow_l2",
    "lower_command_rate_l2",
    "is_terminated",
    "workspace_quality_bonus",
]


def action_term_rate_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Raw action difference squared sum for one named term; no dt division.

    Use ActionManager's global history, which is cleared on partial resets.
    Term offsets follow the manager's declared order, rather than a fixed slice.
    """
    return (term_action(env, action_name) - term_action(env, action_name, previous=True)).square().sum(-1)


def lower_command_rate_l2(
    env: ManagerBasedRLEnv, action_name: str = "lower_command", height_action_name: str = "height_offset"
) -> torch.Tensor:
    """Sum normalized step differences squared for vx, vy, yaw rate and height.

    Each action term caches its clipped normalized change once per control step.
    Disabled terms contribute zero; constant nonzero commands have zero change
    cost. Reads do not advance history. There is no dt division; RewardManager
    applies the shared weight and control dt once.
    """
    velocity = _lower_command_action(env, action_name)
    height = env.action_manager.get_term(height_action_name)
    if not isinstance(height, LowerHeightCommandAction):
        raise TypeError(f"{height_action_name!r} must use LowerHeightCommandAction for height-command changes")
    return velocity.change_l2 + height.change_l2


def _lower_command_action(env: ManagerBasedRLEnv, action_name: str) -> LowerVelocityCommandAction:
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, LowerVelocityCommandAction):
        raise TypeError(f"{action_name!r} must use LowerVelocityCommandAction for velocity-command rewards")
    return action


def lower_command_overflow_l2(env: ManagerBasedRLEnv, action_name: str = "lower_command") -> torch.Tensor:
    """Sum squared raw-command excess outside [-1, 1], before velocity scaling.

    Disabled command interfaces return zero. This dimensionless cost reads only
    the three command actions; RewardManager applies weight and control dt once.
    """
    action = _lower_command_action(env, action_name)
    if not action.cfg.enabled:
        return torch.zeros_like(action.change_l2)
    return (action.raw_actions.abs() - 1.0).clamp_min(0.0).square().sum(dim=-1)


def lower_command_magnitude_l2(env: ManagerBasedRLEnv, action_name: str = "lower_command") -> torch.Tensor:
    """Sum squared effective normalized commands, in [0, 3], before SI scaling.

    Constant nonzero commands retain a cost. Disabled interfaces return zero;
    RewardManager applies weight and control dt once. Reads change no history.
    """
    action = _lower_command_action(env, action_name)
    if not action.cfg.enabled:
        return torch.zeros_like(action.change_l2)
    return action.raw_actions.clamp(-1.0, 1.0).square().sum(dim=-1)


def _base_motion_cost(env: ManagerBasedRLEnv, action_name: str, component: int) -> torch.Tensor:
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, FrozenLowerAction):
        raise TypeError(f"{action_name!r} must use FrozenLowerAction for complete substep base motion rewards")
    return action.mean_base_motion_l2()[:, component]


def base_lin_vel_l2_substeps(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Mean substep squared 3-D world velocity of the base link origin [m²/s²]."""
    return _base_motion_cost(env, action_name, 0)


def base_ang_vel_l2_substeps(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Mean substep squared 3-D world angular velocity of the base [rad²/s²]."""
    return _base_motion_cost(env, action_name, 1)


def base_lin_acc_l2_substeps(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Mean substep squared 3-D world acceleration of the base link origin [m²/s⁴]."""
    return _base_motion_cost(env, action_name, 2)


def base_ang_acc_l2_substeps(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Mean substep squared 3-D world angular acceleration of the base [rad²/s⁴]."""
    return _base_motion_cost(env, action_name, 3)


def base_vertical_acc_excess_l2_substeps(
    env: ManagerBasedRLEnv, soft_limit: float = 4.0, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Soft excess of physical pelvis-link world-Z acceleration, averaged over substeps.

    The dimensionless cost is max(abs(az) / soft_limit - 1, 0)^2 for each
    physical sample. soft_limit is in m/s^2. RewardManager applies weight and
    control dt once; this read changes no command or controller target.
    """
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, FrozenLowerAction):
        raise TypeError(f"{action_name!r} must use FrozenLowerAction for base vertical acceleration rewards")
    return action.mean_base_vertical_acc_excess_l2(soft_limit)


def workspace_quality_bonus(env: ManagerBasedRLEnv, command_name: str = "wrist_pose") -> torch.Tensor:
    """Bounded target-location Q bonus after all active wrists meet position tolerance.

    Only active target positions in the pelvis frame are queried. Right queries
    reflect Y; all active targets need map support. Orientation is independent.
    RewardManager applies the positive weight and control step dt once.
    """
    return env.command_manager.get_term(command_name).workspace_quality_state().reward


def joint_target_speed_excess_l2(
    env: ManagerBasedRLEnv, soft_limit: float = 2.5, action_name: str = "joint_pos"
) -> torch.Tensor:
    """Sum squared normalized excess of processed-goal speed above soft_limit [rad/s].

    Cost = sum(max(abs(target_velocity) / soft_limit - 1, 0)^2).
    This dimensionless soft penalty leaves all position targets unchanged.
    RewardManager applies the negative weight and control dt once.
    """
    if not math.isfinite(soft_limit) or soft_limit <= 0:
        raise ValueError("soft_limit must be positive and finite [rad/s]")
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, UpperJointPositionAction):
        raise TypeError(f"{action_name!r} must use UpperJointPositionAction for processed-target speed rewards")
    return (action.joint_target_velocity.abs() / soft_limit - 1.0).clamp_min(0.0).square().sum(dim=-1)


def joint_acc_l2_substeps(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Mean physical-substep acceleration squared sum over this action's joints [rad²/s⁴].

    The action records earlier substeps; this read includes the final one before
    auto-reset. RewardManager applies the configured weight and control dt once.
    """
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, UpperJointPositionAction):
        raise TypeError(f"{action_name!r} must use UpperJointPositionAction for complete substep acceleration rewards")
    return action.mean_joint_acc_l2()


def end_effector_position_tracking(
    env: ManagerBasedRLEnv, command_name: str, arm_index: int, std: float
) -> torch.Tensor:
    """Active wrist reward, weighted by 2 / active-arm count; std is a distance scale [m]."""
    term = env.command_manager.get_term(command_name)
    position_error, _ = term.pose_error()
    weight = 2.0 * term.active_mask[:, arm_index] / term.active_mask.sum(dim=-1)
    return weight * (1.0 - torch.tanh(position_error[:, arm_index] / std))


def end_effector_orientation_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    arm_index: int,
    sigma_coarse: float,
    sigma_fine: float,
    fine_weight: float,
) -> torch.Tensor:
    """Positive two-scale exponential match of the wrist error quaternion.

    Each kernel is exp(-e_q² / sigma²), where e_q is the vector norm of the
    unit error quaternion (sin of half the shortest angle). Sigmas are
    dimensionless; fine_weight mixes the fine kernel with the coarse kernel.
    The peak is 2 / active-arm count for an active wrist and zero for an
    inactive wrist. Use a positive reward weight.
    """
    if not 0.0 < sigma_fine <= sigma_coarse < float("inf"):
        raise ValueError("Orientation sigmas must satisfy 0 < sigma_fine <= sigma_coarse < inf (dimensionless)")
    if not 0.0 <= fine_weight <= 1.0:
        raise ValueError("fine_weight must be finite and within [0, 1]")
    term = env.command_manager.get_term(command_name)
    error_squared = term.orientation_error_quat_squared()[:, arm_index]
    coarse = torch.exp(-error_squared / sigma_coarse**2)
    fine = torch.exp(-error_squared / sigma_fine**2)
    weight = 2.0 * term.active_mask[:, arm_index] / term.active_mask.sum(dim=-1)
    return weight * ((1.0 - fine_weight) * coarse + fine_weight * fine)


def idle_arm_default_pose(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Idle-arm mean squared deviation [rad²], excluding all waist and active-arm joints."""
    term = env.command_manager.get_term(command_name)
    return (term.idle_joint_deviation() * ~term.active_mask).sum(dim=-1)


def end_effector_relative_lin_acc_l2_substeps(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Mean substep squared link acceleration in the moving pelvis frame [m²/s⁴].

    The action config selects the reference link and wrist origins. RewardManager
    applies weight and control dt once. Both wrists contribute in all command modes.
    """
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, UpperJointPositionAction):
        raise TypeError(f"{action_name!r} must use UpperJointPositionAction for relative link acceleration rewards")
    return action.mean_end_effector_acc_l2()


def joint_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, soft_ratio: float = 0.8) -> torch.Tensor:
    """Sum torque excess above soft_ratio of each live joint effort limit [N·m].

    Uses post-clipping actuator telemetry. Implicit PD actuators report an
    estimate, which can differ from the torque applied internally by the solver.
    This adds a reward threshold without changing physical effort limits.
    """
    if not 0.0 < soft_ratio <= 1.0:
        raise ValueError("soft_ratio must be finite and within (0, 1]")
    robot = env.scene[asset_cfg.name]
    torque = robot.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    threshold = soft_ratio * robot.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]
    return (torque.abs() - threshold).clamp(min=0.0).sum(dim=-1)
