"""Training-only body approach, wrist handoff and arm-posture shaping."""

import math
from typing import NamedTuple

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from .actions import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, WAIST_JOINT_NAMES, UpperJointPositionAction

__all__ = ["PlanarApproachCfg", "PlanarApproachReward"]


class WristApproach(NamedTuple):
    scale: torch.Tensor
    blocked: torch.Tensor
    far: torch.Tensor
    errors: torch.Tensor
    waist_gate: torch.Tensor


@configclass
class PlanarApproachCfg:
    """Station errors define a smooth handoff to wrist tracking."""

    enabled: bool = False
    near_distance_m: float = 0.20
    far_distance_m: float = 0.45
    near_heading_rad: float = math.radians(15)
    far_heading_rad: float = math.radians(45)
    far_wrist_scale: float = 0.10
    linear_reference_m_s: float = 0.40
    angular_reference_rad_s: float = 0.40
    translation_weight: float = 0.60
    heading_weight: float = 0.30
    arm_posture_weight: float = -0.25
    arm_posture_deadband_rad: float = 0.15
    arm_posture_scale_rad: float = 0.50
    arm_posture_max_cost: float = 2.0
    wrist_posture_weight: float = 0.0
    wrist_posture_deadband_rad: float = 0.15
    wrist_posture_scale_rad: float = 0.50
    wrist_posture_max_cost: float = 2.0
    waist_posture_weight: float = 0.0
    waist_posture_deadband_rad: float = 0.15
    waist_posture_scale_rad: float = 0.50
    waist_posture_max_cost: float = 2.0
    waist_posture_full_heading_rad: float = math.radians(30)
    waist_distance_axis_scale: tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, roll, pitch
    waist_large_heading_yaw_gain: float = 0.0
    waist_large_heading_full_rad: float = math.radians(65)
    geometry_enabled: bool = False
    blocked_x_m: float = 0.10
    clear_x_m: float = 0.14
    torso_half_width_m: float = 0.10
    torso_side_blend_m: float = 0.04
    wrist_reach_near_m: float = 0.50
    wrist_reach_far_m: float = 0.75
    blocked_posture_ramp_s: float = 0.25

    def validate(self):
        for name, near, far in (
            ("distance", self.near_distance_m, self.far_distance_m),
            ("heading", self.near_heading_rad, self.far_heading_rad),
            ("front clearance", self.blocked_x_m, self.clear_x_m),
            ("wrist reach", self.wrist_reach_near_m, self.wrist_reach_far_m),
        ):
            if not (math.isfinite(near) and math.isfinite(far) and 0 <= near < far):
                raise ValueError(f"planar_approach {name} requires finite 0 <= near < far")
        if self.far_heading_rad > math.pi or not 0 <= self.far_wrist_scale <= 1:
            raise ValueError("planar_approach requires far heading <= pi and wrist scale in [0, 1]")
        for name in (
            "linear_reference_m_s",
            "angular_reference_rad_s",
            "arm_posture_scale_rad",
            "arm_posture_max_cost",
            "wrist_posture_scale_rad",
            "wrist_posture_max_cost",
            "waist_posture_scale_rad",
            "waist_posture_max_cost",
            "blocked_x_m",
            "torso_half_width_m",
            "torso_side_blend_m",
            "blocked_posture_ramp_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"planar_approach {name} must be positive and finite")
        for name in (
            "translation_weight",
            "heading_weight",
            "arm_posture_deadband_rad",
            "wrist_posture_deadband_rad",
            "waist_posture_deadband_rad",
            "waist_large_heading_yaw_gain",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"planar_approach {name} must be non-negative and finite")
        for name in ("arm_posture_weight", "wrist_posture_weight", "waist_posture_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value > 0:
                raise ValueError(f"planar_approach {name} must be non-positive and finite")
        if self.waist_posture_weight < 0 and not self.near_heading_rad < self.waist_posture_full_heading_rad <= math.pi:
            raise ValueError("planar_approach waist_posture_full_heading_rad must exceed near heading and be <= pi")
        if len(self.waist_distance_axis_scale) != 3 or not all(
            math.isfinite(scale) and scale >= 0 for scale in self.waist_distance_axis_scale
        ):
            raise ValueError("waist_distance_axis_scale needs three non-negative scales in yaw, roll, pitch order")
        extended_waist = any(self.waist_distance_axis_scale) or self.waist_large_heading_yaw_gain > 0
        if extended_waist and self.waist_posture_weight == 0:
            raise ValueError("Planar waist distance and yaw gains require a negative waist_posture_weight")
        if (extended_waist or self.wrist_posture_weight < 0) and not self.geometry_enabled:
            raise ValueError("Planar wrist and extended waist posture costs require geometry_enabled")
        if self.waist_large_heading_yaw_gain > 0 and not (
            self.far_heading_rad < self.waist_large_heading_full_rad <= math.pi
        ):
            raise ValueError("planar_approach waist_large_heading_full_rad must exceed far heading and be <= pi")

    def waist_heading_gate(self, heading):
        return (
            (heading - self.near_heading_rad) / (self.waist_posture_full_heading_rad - self.near_heading_rad)
        ).clamp(0, 1)

    def wrist_geometry(self, target_xy, active):
        """Assess wrist targets in the live horizontal pelvis frame."""
        x, y = target_xy.unbind(-1)
        radius = target_xy.norm(dim=-1)
        side = ((y.abs() - self.torso_half_width_m) / self.torso_side_blend_m).clamp(0.0, 1.0)
        far = ((radius - self.wrist_reach_near_m) / (self.wrist_reach_far_m - self.wrist_reach_near_m)).clamp(0.0, 1.0)
        front_deficit = (self.clear_x_m - x).clamp_min(0.0)
        front_scale = ((x - self.blocked_x_m) / (self.clear_x_m - self.blocked_x_m)).clamp(0.0, 1.0)
        blocked = (1.0 - front_scale) * (1.0 - side) * (1.0 - far)
        # Turning can bring a genuine rear target into reach. For central
        # close targets, let backing up improve the front clearance directly.
        rear = (-x / self.blocked_x_m).clamp(0.0, 1.0)
        travel = torch.maximum(far, rear)
        scale = (1.0 - blocked) * (1.0 - rear) * (1.0 - far * (1.0 - self.far_wrist_scale))
        back = front_deficit.clamp_max(self.clear_x_m + self.blocked_x_m) * (1.0 - side) * (1.0 - far)
        distance = torch.maximum(back, (radius - self.wrist_reach_near_m).clamp_min(0.0))
        if self.waist_posture_weight < 0:
            center = (target_xy * active[..., None]).sum(dim=1)
            center_x, center_y = center.unbind(-1)
            heading = torch.atan2(center_y.abs(), center_x)
            cross_y = torch.stack((-y[:, 0], y[:, 1]), dim=-1)
            cross = ((cross_y - self.torso_half_width_m) / self.torso_side_blend_m).clamp(0, 1)
            # A close target within the torso width should retain the retreat guidance.
            turn = torch.maximum(cross, travel * (1 - blocked))
            # A common turn may make the other active wrist harder to reach.
            conflict = (active & (y.abs() > self.torso_half_width_m) & (y * center_y[:, None] < 0)).any(dim=1)
            strength = (turn * active).amax(dim=1) * ~conflict
            waist_gate = self.waist_heading_gate(heading) * strength
            errors = torch.stack(
                ((distance * active).amax(dim=1), (heading - self.near_heading_rad).clamp_min(0) * strength), dim=-1
            )
        else:
            heading = (torch.atan2(y.abs(), x) - self.near_heading_rad).clamp_min(0.0) * travel
            errors = (torch.stack((distance, heading), dim=-1) * active[..., None]).amax(dim=1)
            waist_gate = torch.zeros_like(errors[:, 1])
        return WristApproach(torch.where(active, scale, 1.0), blocked * active, travel * active, errors, waist_gate)


class PlanarApproachReward(ManagerTermBase):
    """Signed physical progress outside the dead zones and an active-arm carry cost.

    Progress starts with the first pair of consecutive physics observations.
    Posture and the wrist handoff both apply on the first control step.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.command = env.command_manager.get_term(cfg.params.get("command_name", "wrist_pose"))
        self.approach = self.command.cfg.planar_approach
        if self.approach is None or not self.approach.enabled:
            return
        self.action = env.action_manager.get_term(cfg.params.get("action_name", "joint_pos"))
        if not isinstance(self.action, UpperJointPositionAction):
            raise TypeError("PlanarApproachReward requires UpperJointPositionAction")
        self._shoulder_elbow_ids = torch.tensor(
            [
                [self.action._joint_names.index(name) for name in names[:4]]
                for names in (LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES)
            ],
            device=self.device,
        )
        self._previous_excess = torch.zeros(self.num_envs, 2, device=self.device)
        self._observed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_reward = torch.zeros(self.num_envs, device=self.device)
        self._names = ("translation_reward", "heading_reward", "posture_penalty", "wrist_handoff")
        self._planar_names = ()
        if self.approach.geometry_enabled:
            self._names += ("blocked_fraction", "far_fraction")
            self._previous_goals = torch.zeros_like(self.command.goal_pose_w)
            self._previous_active = torch.zeros_like(self.command.active_mask)
            self._previous_station = torch.zeros_like(self._observed)
            self._goal_steps = torch.zeros(self.num_envs, device=self.device)
            if self.approach.wrist_posture_weight < 0:
                self._wrist_action_ids = torch.tensor(
                    [
                        [self.action._joint_names.index(name) for name in names[4:]]
                        for names in (LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES)
                    ],
                    device=self.device,
                )
                self._planar_names += ("wrist_posture_penalty",)
            if self.approach.waist_posture_weight < 0:
                self._waist_action_ids = [self.action._joint_names.index(name) for name in WAIST_JOINT_NAMES]
                self._waist_joint_ids = self.command.robot.find_joints(list(WAIST_JOINT_NAMES), preserve_order=True)[0]
                self._names += ("waist_posture_penalty", "waist_gate")
                self._waist_distance_scale = None
                if any(self.approach.waist_distance_axis_scale):
                    self._waist_distance_scale = self._last_reward.new_tensor(self.approach.waist_distance_axis_scale)
                    self._planar_names += ("waist_distance_penalty",)
                if self.approach.waist_large_heading_yaw_gain > 0:
                    self._planar_names += ("waist_yaw_boost_penalty",)
        self._sums = torch.zeros(self.num_envs, len(self._names) + len(self._planar_names), device=self.device)
        self._steps = torch.zeros(self.num_envs, device=self.device)
        self._planar_steps = torch.zeros_like(self._steps) if self._planar_names else None
        self._last_step = None
        self._dead_zones = torch.tensor(
            [self.approach.near_distance_m, self.approach.near_heading_rad], device=self.device
        )
        self._reference_steps = env.step_dt * torch.tensor(
            [self.approach.linear_reference_m_s, self.approach.angular_reference_rad_s], device=self.device
        )

    def __call__(self, env, command_name: str = "wrist_pose", action_name: str = "joint_pos"):
        cfg = self.approach
        if cfg is None or not cfg.enabled:
            return 0.0
        if self._last_step == env.common_step_counter:
            return self._last_reward
        errors, near = self.command.planar_approach_geometry()
        planar = self.command.planar_task_mask
        excess = (errors - self._dead_zones).clamp_min(0.0)
        active = self.command.active_mask
        target = self.action.processed_actions[:, self._shoulder_elbow_ids]
        default = self.command.robot.data.default_joint_pos.torch[:, self.command.arm_joint_ids[:, :4]]
        joint_excess = ((target - default).abs() - cfg.arm_posture_deadband_rad).clamp_min(0.0)
        joint_cost = (joint_excess / cfg.arm_posture_scale_rad).square().mean(-1)
        if cfg.geometry_enabled:
            return self._geometry_reward(excess, near, planar, active, joint_cost)
        delta = ((self._previous_excess - excess) / self._reference_steps).clamp(-1.0, 1.0)
        translation, heading = torch.where((self._observed & planar)[:, None], delta, 0.0).unbind(-1)
        self._previous_excess.copy_(excess)
        self._observed.copy_(planar)

        arm_cost = (joint_cost * active).sum(-1) / active.sum(-1).clamp_min(1)
        arm_cost = arm_cost.clamp_max(cfg.arm_posture_max_cost) * (1.0 - near)

        translation = translation * cfg.translation_weight
        heading = heading * cfg.heading_weight
        posture = torch.where(planar, arm_cost * cfg.arm_posture_weight, 0.0)
        self._last_reward.copy_(translation + heading + posture)
        self._sums += torch.stack((translation, heading, posture, near * planar), dim=-1)
        self._steps += planar
        self._last_step = env.common_step_counter
        return self._last_reward

    def _geometry_reward(self, station_errors, near, planar, active, joint_cost):
        cfg, command = self.approach, self.command
        wrists = command.wrist_approach_geometry()
        station = planar & (station_errors > 0.0).any(dim=-1)
        errors = torch.where(station[:, None], station_errors, wrists.errors)
        if cfg.waist_posture_weight < 0:
            # Planar has an explicit station heading, including after entering the near field.
            errors[:, 1] = torch.where(planar, station_errors[:, 1], errors[:, 1])
        same_goal = (
            self._observed
            & (active == self._previous_active).all(dim=-1)
            & ((command.goal_pose_w == self._previous_goals).all(dim=-1) | ~active).all(dim=-1)
        )
        valid = same_goal & (station == self._previous_station)
        progress = ((self._previous_excess - errors) / self._reference_steps).clamp(-1.0, 1.0)
        translation, heading = torch.where(valid[:, None], progress, 0.0).unbind(-1)
        self._previous_excess.copy_(errors)
        self._previous_station.copy_(station)
        self._previous_goals.copy_(command.goal_pose_w)
        self._previous_active.copy_(active)
        self._observed.fill_(True)
        self._goal_steps.copy_(torch.where(same_goal, self._goal_steps + 1, 0))

        planar_carry = torch.where(planar, 1.0 - near, 0.0)[:, None]
        travel = torch.maximum(wrists.far, planar_carry)
        ramp = (self._goal_steps * self._env.step_dt / cfg.blocked_posture_ramp_s).clamp_max(1.0)
        carry = torch.maximum(travel, wrists.blocked * ramp[:, None])
        count = active.sum(-1).clamp_min(1)
        arm_cost = (joint_cost.clamp_max(cfg.arm_posture_max_cost) * carry * active).sum(-1) / count
        posture = arm_cost * cfg.arm_posture_weight
        translation *= cfg.translation_weight
        heading *= cfg.heading_weight
        blocked = (wrists.blocked > 0).any(dim=-1)
        far = ((travel > 0) & active).any(dim=-1)
        scale = command.wrist_bonus_scale
        signals = (translation, heading, posture, scale, blocked, far)
        planar_signals = ()
        reward = translation + heading + posture
        if cfg.wrist_posture_weight < 0:
            target = self.action.processed_actions[:, self._wrist_action_ids]
            default = command.robot.data.default_joint_pos.torch[:, command.arm_joint_ids[:, 4:]]
            excess = ((target - default).abs() - cfg.wrist_posture_deadband_rad).clamp_min(0.0)
            cost = (excess / cfg.wrist_posture_scale_rad).square().mean(-1).clamp_max(cfg.wrist_posture_max_cost)
            wrist_posture = cfg.wrist_posture_weight * planar * (cost * carry * active).sum(-1) / count
            reward = reward + wrist_posture
            planar_signals += (wrist_posture,)
        if cfg.waist_posture_weight < 0:
            waist_posture, waist_gate, waist_signals = self._waist_posture(planar, wrists.waist_gate)
            reward = reward + waist_posture
            signals += (waist_posture, waist_gate)
            planar_signals += waist_signals
        self._last_reward.copy_(reward)
        self._sums += torch.stack(signals + planar_signals, dim=-1)
        self._steps += 1
        if self._planar_steps is not None:
            self._planar_steps += planar
        self._last_step = self._env.common_step_counter
        return self._last_reward

    def _waist_posture(self, planar, wrist_gate):
        cfg = self.approach
        distance, heading = self.command.planar_station_errors().unbind(-1)
        gate = torch.where(planar, cfg.waist_heading_gate(heading), wrist_gate)
        target = self.action.processed_actions[:, self._waist_action_ids]
        default = self.command.robot.data.default_joint_pos.torch[:, self._waist_joint_ids]
        excess = ((target - default).abs() - cfg.waist_posture_deadband_rad).clamp_min(0)
        axis_cost = (excess / cfg.waist_posture_scale_rad).square()
        mean_cost = axis_cost.mean(dim=1)
        turn_cost = mean_cost.clamp_max(cfg.waist_posture_max_cost)
        penalty = cfg.waist_posture_weight * gate * turn_cost
        # Report only the incremental penalty each route contributes to the final cost.
        signals = ()
        if self._waist_distance_scale is not None:
            distance_gate = ((distance - cfg.near_distance_m) / (cfg.far_distance_m - cfg.near_distance_m)).clamp(0, 1)
            distance_cost = (axis_cost * self._waist_distance_scale).mean(-1).clamp_max(cfg.waist_posture_max_cost)
            distance_penalty = cfg.waist_posture_weight * planar * distance_gate * distance_cost
            combined = torch.minimum(penalty, distance_penalty)
            signals += (combined - penalty,)
            penalty = combined
        if cfg.waist_large_heading_yaw_gain > 0:
            yaw_gate = (
                (heading - cfg.far_heading_rad) / (cfg.waist_large_heading_full_rad - cfg.far_heading_rad)
            ).clamp(0, 1)
            # Yaw is the first axis in WAIST_JOINT_NAMES; keep the three-axis mean.
            yaw_cost = axis_cost[:, 0] * (cfg.waist_large_heading_yaw_gain / len(WAIST_JOINT_NAMES))
            boosted = (mean_cost + yaw_cost * yaw_gate * planar).clamp_max(cfg.waist_posture_max_cost)
            combined = torch.minimum(penalty, cfg.waist_posture_weight * gate * boosted)
            signals += (combined - penalty,)
            penalty = combined
        return penalty, gate, signals

    def reset(self, env_ids=None):
        if self.approach is None or not self.approach.enabled:
            return
        ids = slice(None) if env_ids is None else env_ids
        steps = self._steps[ids].sum()
        planar_steps = self._planar_steps[ids].sum() if self._planar_steps is not None else None
        if planar_steps is None:
            has_steps = bool(steps > 0)
            has_planar_steps = False
        else:
            has_steps, has_planar_steps = torch.stack((steps, planar_steps)).gt(0).tolist()
        if has_steps:
            log = self._env.extras.setdefault("log", {})
            sums = self._sums[ids].sum(dim=0)
            prefix = "MobileApproach" if self.approach.geometry_enabled else "PlanarApproach"
            values = sums[: len(self._names)] / steps
            for name, value in zip(self._names, values, strict=True):
                log[f"{prefix}/{name}"] = value
            if has_planar_steps:
                values = sums[len(self._names) :] / planar_steps
                for name, value in zip(self._planar_names, values, strict=True):
                    log[f"{prefix}/planar/{name}"] = value
        self._observed[ids] = False
        self._last_reward[ids] = 0
        self._sums[ids] = 0
        self._steps[ids] = 0
        if self._planar_steps is not None:
            self._planar_steps[ids] = 0
        if env_ids is None:
            self._last_step = None
