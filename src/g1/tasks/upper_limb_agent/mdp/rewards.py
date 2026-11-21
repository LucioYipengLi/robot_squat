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
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .actions import UpperJointPositionAction
from .frozen_lower_actions import (
    FrozenLowerAction,
    LowerHeightCommandAction,
    LowerPelvisCommandAction,
    LowerVelocityCommandAction,
)
from .observations import term_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "DwellReward",
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


class DwellReward(ManagerTermBase):
    """Current active-wrist pose readiness plus rolling world-Z base-speed readiness.

    Returns one task-level binary bonus, without per-arm scaling. Position and
    shortest-angle gates are instantaneous; only vertical speed has a history
    window. RewardManager applies the configured weight and dt once.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        for key in ("position_tolerance_m", "orientation_tolerance_rad", "vertical_speed_tolerance_m_s"):
            value = cfg.params[key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        # Per-episode pose count, bonus count, RMS sum and full-window count.
        self._sums = torch.zeros(self.num_envs, 4, device=self.device)
        self._steps = torch.zeros(self.num_envs, device=self.device)
        self._last_metric_step = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "wrist_pose",
        action_name: str = "joint_pos",
        position_tolerance_m: float = 0.01,
        orientation_tolerance_rad: float = 0.04,
        vertical_speed_tolerance_m_s: float = 0.02,
    ) -> torch.Tensor:
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, FrozenLowerAction):
            raise TypeError("DwellReward requires FrozenLowerAction for complete-substep vertical-speed RMS")
        rms, full_window = action.base_vertical_speed_rms()
        command = env.command_manager.get_term(command_name)
        position, orientation = command.pose_error()
        active = command.active_mask
        pose_ready = (((position <= position_tolerance_m) & (orientation <= orientation_tolerance_rad)) | ~active).all(
            dim=-1
        ) & active.any(dim=-1)
        sampled = env.episode_length_buf > 0
        ready = (
            pose_ready
            & full_window
            & (rms <= vertical_speed_tolerance_m_s)
            & sampled
            & ~env.termination_manager.terminated
        )
        if self._last_metric_step != env.common_step_counter:
            valid_rms = full_window & sampled
            self._steps += sampled
            self._sums[:, 0] += pose_ready & sampled
            self._sums[:, 1] += ready
            self._sums[:, 2] += torch.where(valid_rms, rms, 0.0)
            self._sums[:, 3] += valid_rms
            self._last_metric_step = env.common_step_counter
        bonus_scale = getattr(command, "wrist_bonus_scale", None)
        return ready.float() if bonus_scale is None else ready.float() * bonus_scale

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        steps, sums = self._steps[ids], self._sums[ids]
        valid = steps > 0
        log = self._env.extras.setdefault("log", {})
        if valid.any():
            for column, name in enumerate(("pose_ready_fraction", "ready_fraction")):
                log[f"Dwell/{name}"] = (sums[valid, column] / steps[valid]).mean()
        full = sums[:, 3] > 0
        if full.any():
            log["Dwell/base_vertical_speed_rms_m_s"] = (sums[full, 2] / sums[full, 3]).mean()
        self._steps[ids] = 0
        self._sums[ids] = 0
        if env_ids is None:
            self._last_metric_step = -1


def action_term_rate_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Raw action difference squared sum for one named term; no dt division.

    Use ActionManager's global history, which is cleared on partial resets.
    Term offsets follow the manager's declared order, rather than a fixed slice.
    """
    return (term_action(env, action_name) - term_action(env, action_name, previous=True)).square().sum(-1)


def lower_command_rate_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "lower_command",
    height_action_name: str = "height_offset",
    pelvis_action_name: str | None = None,
) -> torch.Tensor:
    """Sum effective-command differences squared in fixed physical reward scales.

    Each action term caches its final normalized change once per control step.
    Disabled terms contribute zero; constant nonzero commands have zero change
    cost. Reads do not advance history. There is no dt division; RewardManager
    applies the shared weight and control dt once.
    """
    velocity = _lower_command_action(env, action_name)
    height = env.action_manager.get_term(height_action_name)
    if not isinstance(height, LowerHeightCommandAction):
        raise TypeError(f"{height_action_name!r} must use LowerHeightCommandAction for height-command changes")
    cost = velocity.change_l2 + height.change_l2
    if pelvis_action_name is not None:
        pelvis = env.action_manager.get_term(pelvis_action_name)
        if not isinstance(pelvis, LowerPelvisCommandAction):
            raise TypeError("Pelvis command changes require LowerPelvisCommandAction")
        cost = cost + pelvis.change_l2
    return cost


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


def lower_command_magnitude_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "lower_command",
    command_name: str = "wrist_pose",
    near_planar_threshold: float = 0.5,
    near_planar_gain: float = 0.0,
) -> torch.Tensor:
    """Sum squared effective velocities divided by their episode range endpoints.

    Positive/negative commands use their corresponding signed bounds. Without a
    curriculum, use configured velocity ranges. Constant nonzero commands retain
    a cost; disabled interfaces return zero. After first planar near entry an
    optional excess cost applies to lateral and yaw commands. RewardManager
    applies weight and control dt once.
    """
    if not 0 <= near_planar_threshold <= 1 or not math.isfinite(near_planar_gain) or near_planar_gain < 0:
        raise ValueError("Lower magnitude requires a near threshold in [0, 1] and a finite non-negative gain")
    action = _lower_command_action(env, action_name)
    if not action.cfg.enabled:
        return torch.zeros_like(action.change_l2)
    normalized = action.magnitude_normalized
    base = normalized.square().sum(dim=-1)
    if near_planar_gain == 0:
        return base
    term = env.command_manager.get_term(command_name)
    near = term.planar_task_mask & (term.planar_first_near_step >= 0)
    excess = (normalized[:, 1:3].abs() - near_planar_threshold).clamp_min(0).square().sum(dim=-1)
    return base + near_planar_gain * torch.where(near, excess, 0.0)


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
    command = env.command_manager.get_term(command_name)
    reward = command.workspace_quality_state().reward
    bonus_scale = getattr(command, "wrist_bonus_scale", None)
    return reward if bonus_scale is None else reward * bonus_scale


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
    env: ManagerBasedRLEnv,
    command_name: str,
    arm_index: int,
    std: float,
    sigma_fine: float,
    fine_weight: float,
) -> torch.Tensor:
    """Blend broad tanh tracking with a fine position exponential for an active wrist.

    Coarse = 1 - tanh(e / std); fine = exp(-e² / sigma_fine²).
    Both scales are distances [m]. The blend peaks at 2 / active-arm count
    for an active wrist and is zero for an inactive wrist. Reads fresh world
    pose errors without modifying command or metric state.
    """
    if not 0.0 < sigma_fine <= std < float("inf"):
        raise ValueError("Position scales must satisfy 0 < sigma_fine <= std < inf [m]")
    if not 0.0 <= fine_weight <= 1.0:
        raise ValueError("fine_weight must be finite and within [0, 1]")
    term = env.command_manager.get_term(command_name)
    position_error, _ = term.pose_error()
    error = position_error[:, arm_index]
    coarse = 1.0 - torch.tanh(error / std)
    fine = torch.exp(-(error / sigma_fine).square())
    weight = 2.0 * term.active_mask[:, arm_index] / term.active_mask.sum(dim=-1)
    approach_scale = getattr(term, "wrist_reward_scale", None)
    if approach_scale is not None:
        weight = weight * approach_scale[:, arm_index]
    return weight * ((1.0 - fine_weight) * coarse + fine_weight * fine)


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
    approach_scale = getattr(term, "wrist_reward_scale", None)
    if approach_scale is not None:
        weight = weight * approach_scale[:, arm_index]
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
