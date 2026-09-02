# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv, mdp
    from isaaclab.sensors import ContactSensor

# module-level cache for the action two-step history used by action_acc_l2
# (ActionManager only keeps a_t and a_{t-1}); keyed by id(env)
_action_acc_prev2_cache: dict[int, torch.Tensor] = {}


def action_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the second-order difference of consecutive actions [action^2].

    Computes :math:`\\|a_t - 2a_{t-1} + a_{t-2}\\|_2^2` to constrain action-space acceleration,
    reducing mechanical wear and twitching. The action manager only keeps the last two actions,
    so :math:`a_{t-2}` is cached in a module-level buffer keyed per environment. Environments
    that terminated since the last reward computation are treated as freshly reset (their action
    buffers were zeroed by ``ActionManager.reset``), so their cache is cleared before use.
    """
    action_t = env.action_manager.action
    action_t1 = env.action_manager.prev_action
    # lazily allocate / grow the a_{t-2} cache for this environment
    cache = _action_acc_prev2_cache.get(id(env))
    if cache is None or cache.shape != action_t.shape:
        cache = torch.zeros_like(action_t)
        _action_acc_prev2_cache[id(env)] = cache
    # clear the cache of envs reset since the last reward computation (terminated flag is
    # one step stale at this point, matching the reset that happened in between)
    reset_mask = env.termination_manager.terminated | env.termination_manager.time_outs
    cache[reset_mask] = 0.0
    # second-order difference, then roll the history
    acc = action_t - 2.0 * action_t1 + cache
    cache.copy_(action_t)
    return torch.sum(torch.square(acc), dim=1)

def joint_torque_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize actuator torque commands that exceed the joint effort limits [Nm].

    Computes :math:`\\sum_i \\text{ReLU}(|\\tau_i| - \\tau_{max,i})` over the actuator torque
    before clipping (``computed_torque``). Joints without a finite effort limit are skipped,
    so implicit actuators that keep infinite limits contribute nothing.

    Note:
        ``computed_torque`` is only maintained for explicit actuators (e.g. DCMotor); restrict
        ``asset_cfg.joint_names`` to such joints (legs and feet for the G1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.computed_torque.torch[:, asset_cfg.joint_ids])
    effort_limit = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]
    excess = torch.relu(torque - effort_limit)
    # mask out joints with infinite limits (e.g. implicit actuators)
    excess = torch.where(torch.isinf(effort_limit), torch.zeros_like(excess), excess)
    return torch.sum(excess, dim=1)

def joint_torques_l2_normalized(
    env: ManagerBasedRLEnv, stiffness: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize squared joint torques normalized by per-joint stiffness gains [rad^2].

    Computes :math:`\\sum_i (\\tau_i / k_i)^2`, matching the official HOMIE implementation where
    torques are divided by the PD stiffness so that joints with different torque scales become
    comparable. ``stiffness`` maps joint names to gains (e.g. ``{"*_hip_.*": 100.0}``); joints
    without a matching key are skipped. Restrict ``asset_cfg.joint_names`` to the lower-body
    joints, as the official configuration does.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    # build the per-selected-joint gain vector by name matching; unmatched joints are skipped
    gains = torch.zeros(len(asset_cfg.joint_ids), device=env.device)
    for j, joint_id in enumerate(asset_cfg.joint_ids):
        name = asset.joint_names[joint_id]
        for expr, gain in stiffness.items():
            if re.fullmatch(expr, name):
                gains[j] = gain
                break
    return torch.sum(torch.square(torque / gains.clamp(min=1e-8)), dim=1)

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def standing_joint_pos_target_l2(
    env: ManagerBasedRLEnv, target: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint position deviation from a target value, gated to the standing regime.

    Same formulation as :func:`joint_pos_target_l2`, but active only when the commanded height
    is at least 0.735 m (near standing). During squatting, hip flexion and ankle dorsiflexion
    are mechanically required, so an ungated deviation penalty would fight the core task.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    deviation = torch.sum(torch.square(joint_pos - target), dim=1)
    # standing-height gate (official HOMIE design)
    height_command_w = env.command_manager.get_term(command_name).height_command_world
    gate = (height_command_w[:, 0] >= 0.735).float()
    return deviation * gate

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w.torch[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

def knee_guidance_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    soft_limits: bool = True,
) -> torch.Tensor:
    """Guide knee flexion/extension based on the sign of the pelvis height error [m·rad].

    Computes :math:`-\\|(h_r - h_{cmd}) \\cdot ((q - q_{min})/(q_{max} - q_{min}) - 1/2)\\|` over the
    knee joints. When the pelvis is above the commanded height, knee flexion (large normalized
    position) is encouraged; when below, knee extension is encouraged. Knee limits are read from
    the asset data (soft limits by default), so no manual range parameters are needed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # pelvis height error: positive when the pelvis is above the commanded height
    height_command_w = env.command_manager.get_term(command_name).height_command_world
    height_error = env.scene["robot"].data.root_pos_w.torch[:, 2] - height_command_w[:, 0]
    # normalized knee positions in [0, 1]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if soft_limits:
        limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    else:
        limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    q_norm = (joint_pos - limits[..., 0]) / (limits[..., 1] - limits[..., 0]).clamp(min=1e-6)
    # penalize the mismatch between the height error sign and the knee flexion level
    return torch.abs(height_error.unsqueeze(1) * (q_norm - 0.5)).norm(dim=1)

def feet_ground_parallel_var(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    foot_length: float = 0.18,
    foot_width: float = 0.08,
) -> torch.Tensor:
    """Penalize the height variance of the four sole corners of each foot [m^2].

    The sole is approximated by four corner points at ``(±foot_length/2, ±foot_width/2)`` in
    the foot frame. Their world heights follow from the foot body pose, and the variance across
    the four points is zero only when the sole is parallel to the ground plane. Following the
    official implementation, only feet in sustained contact (at least ``3 * dt``) contribute,
    so transient swing phases are not penalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]  # (E, B, 3)
    body_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]  # (E, B, 4)
    # four sole corner offsets in the foot frame
    half_l, half_w = foot_length / 2.0, foot_width / 2.0
    offsets = torch.tensor(
        [[half_l, half_w, 0.0], [half_l, -half_w, 0.0], [-half_l, half_w, 0.0], [-half_l, -half_w, 0.0]],
        device=env.device,
    )
    corners_w = quat_apply(
        body_quat.unsqueeze(2).expand(-1, -1, 4, -1).reshape(-1, 4),
        offsets.repeat(body_pos.shape[0] * body_pos.shape[1], 1),
    ).reshape(body_pos.shape[0], body_pos.shape[1], 4, 3)
    corner_heights = corners_w[..., 2] + body_pos[:, :, 2].unsqueeze(-1)  # (E, B, 4)
    # sustained-contact gate: feet must have been in contact for at least 3 control steps
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sustained_contact = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] >= 3.0 * env.step_dt
    # sum the gated height variance over the feet
    return (corner_heights.var(dim=-1, unbiased=False) * sustained_contact).sum(dim=1)

def feet_parallel_var(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Penalize non-parallel foot orientations via the variance of foot-pair distances [m^2].

    Each sole is approximated by three points along its length (heel, middle, toe). The set
    :math:`D` contains the three left-right distances between corresponding points, and
    :math:`\\text{Var}(D)` is zero only when the two feet are parallel, matching the official
    implementation. The term is gated to the standing regime (commanded height >= 0.735 m),
    where foot symmetry matters; during squatting the foot layout naturally differs.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]  # (E, 2, 3), [left, right]
    body_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]  # (E, 2, 4)
    # three sole points in the foot frame: heel / middle / toe
    half_l = 0.09  # half of the G1 foot length
    offsets = torch.tensor([[-half_l, 0.0, 0.0], [0.0, 0.0, 0.0], [half_l, 0.0, 0.0]], device=env.device)
    points_w = quat_apply(
        body_quat.unsqueeze(2).expand(-1, -1, 3, -1).reshape(-1, 4),
        offsets.repeat(body_pos.shape[0] * 2, 1),
    ).reshape(body_pos.shape[0], 2, 3, 3)
    points_w += body_pos.unsqueeze(2)  # (E, 2, 3, 3)
    # pairwise left-right distances and their variance (unbiased, as in the official code)
    feet_distances = torch.norm(points_w[:, 0] - points_w[:, 1], dim=2)  # (E, 3)
    feet_distances_var = feet_distances.var(dim=1)
    # standing-height gate: active only when the commanded height is near standing
    height_command_w = env.command_manager.get_term(command_name).height_command_world
    gate = (height_command_w[:, 0] >= 0.735).float()
    return feet_distances_var * gate

def feet_lateral_separation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    min_distance: float = 0.2,
    max_distance: float = 0.35,
) -> torch.Tensor:
    """Return the lateral feet separation violation in the base frame [m] (non-positive).

    Computes :math:`\\min(|\\Delta y^B| - d_{min}, 0) + \\min(-|\\Delta y^B| + d_{max}, 0) \\cdot g`:
    the lower bound (anti-crossing) is always active, while the upper bound (anti-spreading)
    is gated to the standing regime (:math:`g = [h_{cmd} \\ge 0.735]`), since a wider stance is
    allowed during squatting. Meant to be mounted with a positive weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # feet positions in the base frame; expect exactly two matched bodies (left, right)
    feet_pos_b = quat_apply_inverse(
        asset.data.root_quat_w.torch.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4),
        (asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - asset.data.root_pos_w.torch.unsqueeze(1)).reshape(-1, 3),
    ).reshape(-1, 2, 3)
    lateral_dist = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    # standing-height gate for the upper bound only
    height_command_w = env.command_manager.get_term(command_name).height_command_world
    gate = (height_command_w[:, 0] >= 0.735).float()
    return torch.clamp(lateral_dist - min_distance, max=0.0) + torch.clamp(max_distance - lateral_dist, max=0.0) * gate

def knee_lateral_separation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    min_distance: float = 0.2,
    max_distance: float = 0.35,
) -> torch.Tensor:
    """Return the lateral knee separation violation in the base frame [m] (non-positive).

    Same two-sided formulation as :func:`feet_lateral_separation` applied to the knee bodies,
    suppressing both knee valgus (crossing) and excessive spreading.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos_b = quat_apply_inverse(
        asset.data.root_quat_w.torch.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4),
        (asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - asset.data.root_pos_w.torch.unsqueeze(1)).reshape(-1, 3),
    ).reshape(-1, 2, 3)
    lateral_dist = torch.abs(knee_pos_b[:, 0, 1] - knee_pos_b[:, 1, 1])
    height_command_w = env.command_manager.get_term(command_name).height_command_world
    gate = (height_command_w[:, 0] >= 0.735).float()
    return torch.clamp(lateral_dist - min_distance, max=0.0) + torch.clamp(max_distance - lateral_dist, max=0.0) * gate

def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward keeping at least one foot grounded [0/1].

    Returns 1.0 when at least one foot maintains contact, preventing both feet from lifting
    simultaneously during the standing task. The official version gates this on a zero
    velocity command, which is always true for this task (no locomotion commands).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2] > 0.5
    return (contacts.sum(dim=1) >= 1).float()

def stand_still(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet losing contact while standing still [#feet].

    Returns the number of feet whose vertical contact force dropped below 0.1 N, i.e. airborne
    twitching or stepping-in-place is penalized for each lifted foot. The official version
    gates this on a zero velocity command, which is always true for this task.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    airborne = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2] < 0.1
    return airborne.sum(dim=1).float()

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def track_pelvis_height_exp(env, command_name: str) -> torch.Tensor:
    """Reward for tracking the commanded pelvis height using an absolute-error exponential kernel.

    Computes :math:`\\exp(-4 |h_r - h_{cmd}|)` in the world frame, matching the official
    implementation. Compared to a squared-error kernel, the constant gradient magnitude near
    zero error keeps a strong fine-adjustment signal for precise height tracking.
    """
    height_command_w = env.command_manager.get_term(command_name).height_command_world
    error = torch.abs(env.scene["robot"].data.root_pos_w.torch[:, 2] - height_command_w[:, 0])
    return torch.exp(-error * 4.0)

def zero_velocity_exp(env: ManagerBasedRLEnv, std: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for keeping the base stationary using squared-error exponential kernels.

    Sums three exponential terms over the horizontal linear velocities and the yaw rate,
    :math:`\\exp(-v_x^2/(2\\sigma^2)) + \\exp(-v_y^2/(2\\sigma^2)) + \\exp(-\\omega_{yaw}^2/(2\\sigma^2))`.
    With the default ``std = 0.25`` this reduces to
    :math:`\\exp(-4 v_x^2) + \\exp(-4 v_y^2) + \\exp(-4 \\omega_{yaw}^2)`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    denom = 2.0 * std**2
    vx = asset.data.root_lin_vel_w.torch[:, 0]
    vy = asset.data.root_lin_vel_w.torch[:, 1]
    yaw_rate = asset.data.root_ang_vel_w.torch[:, 2]
    return (
        torch.exp(-torch.square(vx) / denom)
        + torch.exp(-torch.square(vy) / denom)
        + torch.exp(-torch.square(yaw_rate) / denom)
    )

def feet_spread_x_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize horizontal (x) distance between feet and pelvis to prevent splits-style lowering."""
    asset: Articulation = env.scene[asset_cfg.name]
    pelvis_x = asset.data.root_pos_w.torch[:, 0]
    ankle_x = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 0]
    return torch.sum(torch.square(ankle_x - pelvis_x.unsqueeze(1)), dim=1)

def feet_spread_y_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize horizontal (y) distance between feet and pelvis to prevent splits-style lowering."""
    asset: Articulation = env.scene[asset_cfg.name]
    pelvis_y = asset.data.root_pos_w.torch[:, 1]
    ankle_y = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 1]
    return torch.sum(torch.square(ankle_y - pelvis_y.unsqueeze(1)), dim=1)

def base_xy_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize horizontal base velocity to suppress position drift during standing."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w.torch[:, :2]), dim=1)

def stay_at_origin_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize horizontal displacement of the root from its environment origin [m^2]."""
    root_xy = env.scene["robot"].data.root_pos_w.torch[:, :2]
    origin_xy = env.scene.env_origins[:, :2]
    return torch.sum(torch.square(root_xy - origin_xy), dim=1)

def leg_symmetry_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize asymmetry between left and right leg joint positions [rad^2].

    For each left-side joint, the squared difference to its right-side counterpart is summed.
    This suppresses one-sided leaning strategies and encourages mirrored squatting postures.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    joint_names = asset.joint_names
    reward = torch.zeros(env.num_envs, device=env.device)
    for i, name in enumerate(joint_names):
        if name.startswith("left_"):
            right_name = "right_" + name[len("left_"):]
            if right_name in joint_names:
                j = joint_names.index(right_name)
                reward = reward + torch.square(joint_pos[:, i] - joint_pos[:, j])
    return reward

def feet_air_time_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize air time of the feet; both feet should remain grounded during standing [s]."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.sum(air_time, dim=1)

def base_yaw_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize yaw angular velocity of the base to keep the heading fixed while standing."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_w.torch[:, 2])