"""Single/dual wrist commands held in world space and observed in the live pelvis frame."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass, replace
from isaaclab.utils.math import (
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    subtract_frame_transforms,
    yaw_quat,
)

from .actions import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from .episode_metrics import PoseWindowMetrics, failure_per_10s, simultaneous_pose_ready
from .goal_timing import budget_steps
from .planar_targets import PlanarTaskCfg, sample_station
from .pose_override import PoseOverride, validate_pose
from .range_curriculum import RangeCurriculum, RangeCurriculumCfg
from .workspace_quality import (
    DEFAULT_POSITION_TOLERANCE_M,
    WorkspaceQualityCfg,
    WorkspaceQualityMap,
    WorkspaceQualityState,
)

__all__ = ["BimanualPoseCommand", "BimanualPoseCommandCfg"]


class BimanualPoseCommand(CommandTerm):
    """Sample single-wrist offsets or a geometrically constrained pair per interval.

    Modes are left-only, right-only and both. World goals are authoritative;
    ``command`` returns their current pelvis-relative poses, left then right.
    Inactive goals have a neutral observation and contribute no tracking reward.
    Dual goals share a center and a common orientation, with bounded separation,
    fore/aft and height differences, and a bounded relative rotation.
    Optional height tasks add one episode-wide world-Z shift shared by active
    wrists. Their complete poses can be held until reset, independently of Lower.
    """

    cfg: BimanualPoseCommandCfg
    MODE_NAMES = ("left_only", "right_only", "both")
    _teleop: PoseOverride | None = None

    def __init__(self, cfg: BimanualPoseCommandCfg, env):
        self._validate_cfg(cfg)
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.reference_body_idx = self._body_index(cfg.reference_body_name)
        self.body_indices = [self._body_index(name) for name in cfg.body_names]
        self.arm_joint_ids = torch.tensor(
            [
                self.robot.find_joints(list(names), preserve_order=True)[0]
                for names in (LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES)
            ],
            device=self.device,
            dtype=torch.long,
        )
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self.mode_id = torch.full((self.num_envs,), 2, device=self.device, dtype=torch.long)
        self._mode_masks = torch.tensor([[True, False], [False, True], [True, True]], device=self.device)
        self.height_task_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.planar_task_mask = torch.zeros_like(self.height_task_mask)
        self.height_shift = torch.zeros(self.num_envs, device=self.device)
        self.goal_pose_w = torch.zeros(self.num_envs, 2, 7, device=self.device)
        self.goal_pose_w[..., 6] = 1.0
        self.default_wrist_pose_e = self.goal_pose_w.clone()
        self._default_pelvis_pos_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.planar_station_w = torch.zeros_like(self._default_pelvis_pos_e)
        self.planar_distance_m = torch.zeros_like(self.height_shift)
        self.planar_heading_rad = torch.zeros_like(self.height_shift)
        self._episode_start_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.evaluation_start_steps = torch.zeros_like(self._episode_start_step)
        self.episode_deadline_steps = torch.zeros_like(self._episode_start_step)
        self._sampling_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._sampling_quat_w[:, 3] = 1.0
        self._reference_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._position_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self._orientation_sum = torch.zeros_like(self._position_sum)
        self._idle_deviation_sum = torch.zeros_like(self._position_sum)
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)
        self._position_ready_sum = torch.zeros(self.num_envs, device=self.device)
        self._pose_ready_sum = torch.zeros_like(self._position_ready_sum)
        self._pose_window = PoseWindowMetrics(self.num_envs, self.device)
        self.range_curriculum = None
        if cfg.range_curriculum is not None:
            actions = env.cfg.actions
            velocity = actions.lower_command
            ranges = (
                actions.height_offset.height_offset_range,
                *velocity.command_ranges,
                *actions.pelvis_offset.offset_ranges,
            )
            probability = cfg.height_task.probability if cfg.height_task is not None else 0.0
            planar_probability = cfg.planar_task.probability if cfg.planar_task is not None else 0.0
            self.range_curriculum = RangeCurriculum(
                cfg.range_curriculum,
                ranges,
                cfg.height_task.shift_range if cfg.height_task is not None else (0.0, 0.0),
                (
                    (1 - planar_probability) * (1 - probability) > 0,
                    (1 - planar_probability) * probability > 0,
                    planar_probability > 0,
                ),
                self.num_envs,
                self.device,
                planar_probability=planar_probability,
            )
            self.range_curriculum.target_contract = {
                "pose_scoring": "near_episode_mean_and_planar_fixed_window_v1",
                "left_offsets": asdict(cfg.left_offsets),
                "right_offsets": asdict(cfg.right_offsets),
                "dual_targets": asdict(cfg.dual_targets),
                "mode_probabilities": cfg.mode_probabilities,
                "class_probability": probability,
                "hold_episode": cfg.height_task.hold_episode if cfg.height_task is not None else False,
                "resampling_time_range": cfg.resampling_time_range,
                "distance_reference_speed": cfg.distance_reference_speed,
            }
        if self.planar_enabled:
            self._validate_planar_environment()
            if self.range_curriculum is not None:
                self.range_curriculum.target_contract.update(
                    planar_task=asdict(cfg.planar_task),
                    timing_control_dt=env.step_dt,
                    timing_hard_limit_s=env.cfg.episode_length_s,
                )
        self.workspace_quality_map = None
        self._workspace_sums = None
        if cfg.workspace_quality is not None:
            self.workspace_quality_map = WorkspaceQualityMap(cfg.workspace_quality.map_path, self.device)
            # Score sums/counts for target and actual positions, readiness,
            # coverage and their joint reward-availability gate.
            self._workspace_sums = torch.zeros(self.num_envs, 7, device=self.device)

    @staticmethod
    def _validate_cfg(cfg):
        if cfg.planar_task is not None:
            cfg.planar_task.validate_parameters()
        if cfg.height_task is not None:
            height = cfg.height_task
            if not math.isfinite(height.probability) or not 0 <= height.probability <= 1:
                raise ValueError("height_task.probability must be finite and in [0,1]")
            if (
                len(height.shift_range) != 2
                or not all(math.isfinite(value) for value in height.shift_range)
                or height.shift_range[0] > height.shift_range[1]
            ):
                raise ValueError("height_task.shift_range must contain two finite, ordered world-Z offsets [m]")
            if not isinstance(height.hold_episode, bool):
                raise ValueError("height_task.hold_episode must be a bool")
        if cfg.workspace_quality is not None:
            tolerance = cfg.workspace_quality.position_tolerance_m
            if not math.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("Workspace Q position tolerance must be positive and finite")
            if cfg.reference_body_name != "pelvis" or tuple(cfg.body_names) != (
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ):
                raise ValueError("Workspace Q requires pelvis-frame left/right wrist_yaw link origins")
        probabilities = cfg.mode_probabilities
        if (
            len(probabilities) != 3
            or any(not math.isfinite(p) or p < 0 for p in probabilities)
            or not math.isclose(sum(probabilities), 1.0)
        ):
            raise ValueError("mode_probabilities must contain three non-negative probabilities summing to 1")
        lo, hi = cfg.resampling_time_range
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
            raise ValueError("resampling_time_range must satisfy 0 < min <= max")
        if not math.isfinite(cfg.distance_reference_speed) or cfg.distance_reference_speed <= 0:
            raise ValueError("distance_reference_speed must be positive [m/s]")
        if len(cfg.body_names) != 2 or cfg.body_names[0] == cfg.body_names[1]:
            raise ValueError("body_names must identify two distinct wrist links, left then right")
        for label, ranges in (
            ("left_offsets", cfg.left_offsets),
            ("right_offsets", cfg.right_offsets),
            ("dual_targets", cfg.dual_targets),
        ):
            for axis in ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"):
                bounds = getattr(ranges, axis)
                if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                    raise ValueError(f"Invalid {label} offset range: {axis}={bounds}")
        dual = cfg.dual_targets
        for name in ("separation_range", "relative_pos_x", "relative_pos_z"):
            bounds = getattr(dual, name)
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid dual_targets.{name}: {bounds}")
        if dual.separation_range[0] <= 0:
            raise ValueError("dual_targets.separation_range must be positive")
        max_x = max(abs(x) for x in dual.relative_pos_x)
        max_z = max(abs(x) for x in dual.relative_pos_z)
        if math.hypot(max_x, max_z) >= dual.separation_range[0]:
            raise ValueError("dual_targets relative X/Z differences must fit inside the minimum separation")
        if not math.isfinite(dual.max_relative_angle) or not 0 <= dual.max_relative_angle <= math.pi:
            raise ValueError("dual_targets.max_relative_angle must be finite and in [0, pi] rad")

    def _body_index(self, name):
        ids, _ = self.robot.find_bodies(name)
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one link matching {name!r}, got {len(ids)}")
        return ids[0]

    @property
    def planar_enabled(self):
        forced = self._teleop is not None and self._teleop.source == "auto" and self._teleop.category == "planar"
        return self.cfg.planar_task is not None and (self.cfg.planar_task.probability > 0 or forced)

    @property
    def teleop_source(self):
        return self._teleop.source if self._teleop is not None else None

    def set_teleop(self, source, category="regular", arm_mode=2, *, replay=None):
        """Install runtime selection before reset; leave the saved training contract intact."""
        if source not in ("manual", "auto") or category not in ("regular", "height_shifted", "planar", "mixed"):
            raise ValueError("Unknown teleop source or task category")
        if arm_mode not in (None, 0, 1, 2) or (source == "manual" and arm_mode is None):
            raise ValueError("Manual teleop requires left/right/both; auto also supports sampled arms")
        if source == "auto" and category == "height_shifted" and self.cfg.height_task is None:
            raise ValueError("This environment has no height target sampler")
        if source == "auto" and category == "planar" and self.cfg.planar_task is None:
            raise ValueError("Planar targets require the Mobile task")
        previous = self._teleop
        self._teleop = PoseOverride(source, category, arm_mode, replay=replay)
        try:
            if self.planar_enabled:
                self._validate_planar_environment()
        except Exception:
            self._teleop = previous
            raise

    def set_manual_targets(self, poses_w, arm_mode):
        """Queue world targets for the next normal command update, once per control step."""
        if self.teleop_source != "manual" or arm_mode not in (0, 1, 2):
            raise ValueError("Select manual teleop and an active arm mode first")
        poses = validate_pose(poses_w, self.goal_pose_w.shape, self.device)
        self._teleop.arm_mode = arm_mode
        self._teleop.pending = poses, arm_mode

    def teleop_snapshot(self):
        return PoseOverride.snapshot(self)

    def clear_teleop(self):
        """Return to configured sampling on the next reset."""
        self._teleop = None

    def _validate_planar_environment(self):
        """Validate all levels before physics, rather than shortening a sampled window."""
        if not hasattr(self._env.cfg.actions, "lower_command") or not self._env.cfg.actions.lower_command.enabled:
            raise ValueError("planar_task requires the Mobile Lower velocity interface to be enabled")
        course = self.range_curriculum
        cfg, dt = self.cfg.planar_task, self._env.step_dt
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("planar_task requires a positive finite control dt")
        hard_steps = math.ceil(self._env.cfg.episode_length_s / dt - 1e-9)
        near_steps = math.ceil(cfg.timing.near_episode_s / dt - 1e-9)
        required = near_steps
        levels = range(course.cfg.num_levels) if course is not None else (0,)
        for level in levels:
            progress = level / (course.cfg.num_levels - 1) if course is not None else 1.0
            bounds = (
                torch.lerp(course.initial_ranges, course.final_ranges, progress).detach().cpu()[1:4]
                if course is not None
                else torch.tensor(self._env.cfg.actions.lower_command.command_ranges)
            )
            forward, yaw = bounds[0, 1], bounds[2].abs().amin()
            if forward <= 0 or yaw <= 0:
                raise ValueError("planar_task requires positive forward and bidirectional yaw limits at every level")
            extent = cfg.initial_half_extent_m + progress * (cfg.half_extent_m - cfg.initial_half_extent_m)
            angle = cfg.initial_bearing_rad + progress * (cfg.bearing_rad - cfg.initial_bearing_rad)
            _, deadline = budget_steps(
                torch.tensor(math.sqrt(2) * extent), torch.tensor(angle), forward, yaw, cfg.timing, dt
            )
            required = max(required, int(deadline))
        if hard_steps < required:
            raise ValueError(
                f"planar_task needs env.episode_length_s >= {required * dt:.3f}s for a complete evaluation window; "
                "35s is recommended for the default ranges"
            )

    def episode_age_steps(self):
        """Actual control periods since reset, independent of PPO's episode buffer."""
        return self._env.common_step_counter - self._episode_start_step

    def evaluation_window_mask(self):
        """Read-only half-open timing gate; precision never changes window timing."""
        age = self.episode_age_steps()
        return self.planar_task_mask & (age >= self.evaluation_start_steps) & (age < self.episode_deadline_steps)

    @property
    def task_ids(self):
        return torch.where(self.planar_task_mask, 2, self.height_task_mask.long())

    def elapsed_seconds(self, env_ids):
        steps = self.episode_age_steps() if hasattr(self._env, "common_step_counter") else self._env.episode_length_buf
        return steps[env_ids].float() * self._env.step_dt

    def collect_window_metrics(self, env_ids=None, position=None, orientation=None):
        """Collect once per physical control frame, including the pre-reset boundary.

        The Mobile environment calls before reset events alter the robot state.
        Ordinary command updates reuse already computed errors. Failed terminal
        frames and absent future time never contribute readiness or error samples.
        """
        if not self.planar_enabled or not hasattr(self._env, "reset_terminated"):
            return
        ids = self._env_ids[slice(None) if env_ids is None else env_ids]
        if len(ids) == 0:
            return
        if position is None:
            position, orientation = self.pose_error()
        position, orientation, active = position[ids], orientation[ids], self.active_mask[ids]
        cfg = self.range_curriculum.cfg if self.range_curriculum is not None else RangeCurriculumCfg()
        ready = simultaneous_pose_ready(
            position, orientation, active, cfg.position_tolerance_m, cfg.orientation_tolerance_rad
        )
        tolerance = (
            self.cfg.workspace_quality.position_tolerance_m
            if self.cfg.workspace_quality is not None
            else DEFAULT_POSITION_TOLERANCE_M
        )
        position_ready = ((position <= tolerance) | ~active).all(-1) & active.any(-1)
        valid = self.evaluation_window_mask()[ids] & ~self._env.reset_terminated[ids]
        self._pose_window.record(
            ids, self._env.common_step_counter, valid, position, orientation, active, ready, position_ready
        )

    def episode_quality(self, ids):
        near = self._pose_ready_sum[ids] / self._metric_steps[ids].clamp_min(1)
        window_steps = self.episode_deadline_steps[ids] - self.evaluation_start_steps[ids]
        planar = self._pose_window.quality(ids, window_steps)
        return torch.where(self.planar_task_mask[ids], planar, near)

    @property
    def active_mask(self) -> torch.Tensor:
        return self._mode_masks[self.mode_id]

    @property
    def command(self) -> torch.Tensor:
        """Live pelvis-relative targets [14]; inactive blocks are [0,0,0,0,0,0,1]."""
        position, orientation = self._pose_in_pelvis(self.goal_pose_w)
        pose = torch.cat((position, orientation), dim=-1)
        neutral = torch.zeros_like(pose)
        neutral[..., 6] = 1.0
        return torch.where(self.active_mask[..., None], pose, neutral).flatten(1)

    def _pose_in_pelvis(self, pose_w):
        reference = self.robot.data.body_link_pose_w.torch[:, self.reference_body_idx]
        position, orientation = subtract_frame_transforms(
            reference[:, None, :3].expand(-1, 2, -1),
            reference[:, None, 3:].expand(-1, 2, -1),
            pose_w[..., :3],
            pose_w[..., 3:],
        )
        return position, quat_unique(orientation)

    def current_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Actual wrist link poses in the live pelvis frame, shape [N,2,3/4]."""
        return self._pose_in_pelvis(self.robot.data.body_link_pose_w.torch[:, self.body_indices])

    def workspace_quality_state(self) -> WorkspaceQualityState:
        """Read fresh world targets and physical wrists, before command resampling.

        Reward and diagnostics share this calculation. It has no temporal cache,
        so a partial reset or a newly published goal cannot retain an old gate.
        """
        if self.workspace_quality_map is None:
            raise ValueError("Enable commands.wrist_pose.workspace_quality before using its reward")
        target_b, _ = self._pose_in_pelvis(self.goal_pose_w)
        actual_b, _ = self.current_pose_b()
        current = self.robot.data.body_link_pose_w.torch[:, self.body_indices, :3]
        errors = (current - self.goal_pose_w[..., :3]).norm(dim=-1)
        return self.workspace_quality_map.evaluate(
            target_b, actual_b, errors, self.active_mask, self.cfg.workspace_quality.position_tolerance_m
        )

    def pose_error(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw per-wrist position [m] and shortest orientation [rad] errors, shape [N,2].

        Read world poses directly: rewards run before CommandManager.compute(),
        so a cached pelvis-relative goal could belong to the previous step.
        """
        current = self.robot.data.body_link_pose_w.torch[:, self.body_indices]
        return (
            (current[..., :3] - self.goal_pose_w[..., :3]).norm(dim=-1),
            quat_error_magnitude(current[..., 3:], self.goal_pose_w[..., 3:]),
        )

    def orientation_error_quat_squared(self) -> torch.Tensor:
        """Squared vector norm of the unit error quaternion, shape [N,2].

        q_error = q_goal * conjugate(q_current), using current world poses.
        The result is dimensionless sin(shortest_angle / 2)^2 in [0,1],
        invariant to quaternion sign and stable near perfect alignment.
        """
        current = self.robot.data.body_link_pose_w.torch[:, self.body_indices, 3:]
        error = normalize(quat_mul(self.goal_pose_w[..., 3:], quat_conjugate(current)))
        return error[..., :3].square().sum(dim=-1).clamp(0.0, 1.0)

    def idle_joint_deviation(self) -> torch.Tensor:
        """Mean squared default-angle deviation per arm [rad²], excluding the waist."""
        position = self.robot.data.joint_pos.torch[:, self.arm_joint_ids]
        default = self.robot.data.default_joint_pos.torch[:, self.arm_joint_ids]
        return (position - default).square().mean(dim=-1)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
        ids = self._env_ids[slice(None) if env_ids is None else env_ids]
        if len(ids) == 0:
            return {}
        extras = self._episode_metrics(ids)
        if self.range_curriculum is not None:
            terminated = getattr(self._env, "reset_terminated", None)
            timeouts = getattr(self._env, "reset_time_outs", None)
            if terminated is not None and timeouts is not None and not getattr(self, "_restoring_ranges", False):
                completed = (terminated[ids] | timeouts[ids]) & (self._env.episode_length_buf[ids] > 0)
                ended = ids[completed]
                self.range_curriculum.complete(
                    ended,
                    self.task_ids[ended],
                    self.episode_quality(ended),
                    terminated[ended],
                    self._env.common_step_counter,
                    self.elapsed_seconds(ended),
                )
            self._env.extras.setdefault("log", {}).update(self.range_curriculum.logs())
            self.range_curriculum.begin(ids)
        for buffer in (
            self._position_sum,
            self._orientation_sum,
            self._idle_deviation_sum,
            self._metric_steps,
            self._position_ready_sum,
            self._pose_ready_sum,
        ):
            buffer[ids] = 0
        if self._workspace_sums is not None:
            self._workspace_sums[ids] = 0
        self._pose_window.reset(ids)
        self._capture_default_pose(ids)
        if self._teleop is None:
            probabilities = torch.tensor(self.cfg.mode_probabilities, device=self.device, dtype=torch.float32)
            self.mode_id[ids] = torch.multinomial(probabilities, len(ids), replacement=True)
            self._sample_task_kind(ids)
        else:
            self._teleop.prepare_reset(self, ids)
        super().reset(ids)
        return extras

    def _sample_task_kind(self, ids):
        self.planar_task_mask[ids] = False
        self.height_task_mask[ids] = False
        self.height_shift[ids] = 0
        self._episode_start_step[ids] = getattr(self._env, "common_step_counter", 0)
        if not self.planar_enabled:
            self._sample_height_task(ids)  # Preserve RNG and legacy timing when the module is off.
            return
        probability = (
            self.range_curriculum.planar_probability
            if self.range_curriculum is not None
            else self.cfg.planar_task.probability
        )
        self.planar_task_mask[ids] = (
            True if probability == 1 else torch.rand(len(ids), device=self.device) < probability
        )
        self.evaluation_start_steps[ids] = 0
        self.episode_deadline_steps[ids] = math.ceil(
            self.cfg.planar_task.timing.near_episode_s / self._env.step_dt - 1e-9
        )
        self.planar_station_w[ids] = 0
        self.planar_distance_m[ids] = 0
        self.planar_heading_rad[ids] = 0
        self._sample_height_task(ids[~self.planar_task_mask[ids]])

    def _sample_height_task(self, env_ids, *, force=False):
        """Sample only at reset; disabling height tasks preserves the old RNG sequence."""
        self.height_task_mask[env_ids] = False
        self.height_shift[env_ids] = 0
        height = self.cfg.height_task
        if height is None or (height.probability == 0 and not force):
            return
        if height.probability == 1 or force:
            self.height_task_mask[env_ids] = True
        else:
            self.height_task_mask[env_ids] = torch.rand(len(env_ids), device=self.device) < height.probability
        ids = env_ids[self.height_task_mask[env_ids]]
        if self.range_curriculum is None:
            self.height_shift[ids] = torch.empty(len(ids), device=self.device).uniform_(*height.shift_range)
        else:
            bounds = self.range_curriculum.height_shift_ranges(ids)
            self.height_shift[ids] = bounds[:, 0] + torch.rand(len(ids), device=self.device) * (
                bounds[:, 1] - bounds[:, 0]
            )

    def _capture_default_pose(self, env_ids):
        """Capture each environment once, after default joint reset and before physics steps."""
        ids = env_ids[~self._reference_initialized[env_ids]]
        if len(ids) == 0:
            return
        poses = self.robot.data.body_link_pose_w.torch
        self.default_wrist_pose_e[ids] = poses[ids][:, self.body_indices]
        self.default_wrist_pose_e[ids, :, :3] -= self._env.scene.env_origins[ids, None, :]
        self._sampling_quat_w[ids] = poses[ids, self.reference_body_idx, 3:]
        self._default_pelvis_pos_e[ids] = poses[ids, self.reference_body_idx, :3] - self._env.scene.env_origins[ids]
        if self.range_curriculum is not None:
            # Scaling separation about the authored geometry must keep every
            # intermediate level inside the same physical pair limits.
            delta = quat_apply_inverse(
                self._sampling_quat_w[ids],
                self.default_wrist_pose_e[ids, 0, :3] - self.default_wrist_pose_e[ids, 1, :3],
            )
            dual = self.cfg.dual_targets
            values = (delta.norm(dim=-1), delta[:, 0], delta[:, 2])
            bounds = (dual.separation_range, dual.relative_pos_x, dual.relative_pos_z)
            if (delta[:, 1] <= 0).any() or any(
                ((value < lo - 1e-5) | (value > hi + 1e-5)).any()
                for value, (lo, hi) in zip(values, bounds, strict=True)
            ):
                raise ValueError("Default wrist geometry must lie inside dual_targets limits for the range curriculum")
        self._reference_initialized[ids] = True

    def _resample(self, env_ids):
        """Keep height-task poses for the episode; regular goals use distance-based intervals."""
        ids = self._env_ids[env_ids]
        if self._teleop is not None:
            self._teleop.resample(self, ids)
            return
        height = self.cfg.height_task
        held = self.planar_task_mask[ids]
        if height is not None and height.hold_episode:
            held = held | self.height_task_mask[ids]
        held &= self.command_counter[ids] > 0
        self.time_left[ids[held]] = torch.inf
        ids = ids[~held]
        if len(ids) == 0:
            return
        self._resample_command(ids)
        pelvis = self.robot.data.body_link_pose_w.torch[ids, self.reference_body_idx, :3]
        distances = (self.goal_pose_w[ids, :, :3] - pelvis[:, None, :]).norm(dim=-1)
        distance = (distances * self.active_mask[ids]).amax(dim=1)
        self.time_left[ids] = (distance / self.cfg.distance_reference_speed).clamp(*self.cfg.resampling_time_range)
        if height is not None and height.hold_episode:
            # A lifecycle marker, never an observation or scalar metric. Avoid
            # floating-point timer expiry just before an episode timeout.
            self.time_left[ids[self.height_task_mask[ids]]] = torch.inf
        self.time_left[ids[self.planar_task_mask[ids]]] = torch.inf
        self.command_counter[ids] += 1

    def _resample_command(self, env_ids):
        both = self.mode_id[env_ids] == 2
        self._sample_single_targets(env_ids[~both])
        self._sample_dual_targets(env_ids[both])
        # Add after rotation: this is world vertical even if the sampling pelvis
        # is tilted. Inactive goals keep their original, masked-out poses.
        self.goal_pose_w[env_ids, :, 2] += self.height_shift[env_ids, None] * self.active_mask[env_ids]
        self._apply_planar_targets(env_ids[self.planar_task_mask[env_ids]])

    def _apply_planar_targets(self, ids):
        if len(ids) == 0:
            return
        cfg = self.cfg.planar_task
        course = self.range_curriculum
        progress = (
            course.episode_levels[ids].float() / (course.cfg.num_levels - 1)
            if course is not None
            else torch.ones(len(ids), device=self.device)
        )
        anchor = self._env.scene.env_origins[ids] + self._default_pelvis_pos_e[ids]
        local = self.goal_pose_w[ids, :, :3] - anchor[:, None]
        radius = (local.norm(dim=-1) * self.active_mask[ids]).amax(-1)
        shift, heading = sample_station(progress, radius, cfg)
        displacement = torch.cat((shift, torch.zeros(len(ids), 1, device=self.device)), dim=-1)
        station = anchor + quat_apply(yaw_quat(self._sampling_quat_w[ids]), displacement)
        zeros = torch.zeros_like(heading)
        rotation = quat_from_euler_xyz(zeros, zeros, heading)[:, None].expand(-1, 2, -1)
        self.goal_pose_w[ids, :, :3] = station[:, None] + quat_apply(rotation, local)
        self.goal_pose_w[ids, :, 3:] = quat_unique(quat_mul(rotation, self.goal_pose_w[ids, :, 3:]))
        bounds = (
            course.lower_ranges[ids, 1:4]
            if course is not None
            else torch.tensor(self._env.cfg.actions.lower_command.command_ranges, device=self.device).expand(
                len(ids), -1, -1
            )
        )
        self.evaluation_start_steps[ids], self.episode_deadline_steps[ids] = budget_steps(
            shift.norm(dim=-1), heading, bounds[:, 0, 1], bounds[:, 2].abs().amin(-1), cfg.timing, self._env.step_dt
        )
        self.planar_station_w[ids] = station
        self.planar_distance_m[ids] = shift.norm(dim=-1)
        self.planar_heading_rad[ids] = heading

    def _sample_offsets(self, ranges, count):
        offset = torch.empty(count, 3, device=self.device)
        angles = torch.empty_like(offset)
        for axis, name in enumerate(("pos_x", "pos_y", "pos_z")):
            offset[:, axis].uniform_(*getattr(ranges, name))
        for axis, name in enumerate(("roll", "pitch", "yaw")):
            angles[:, axis].uniform_(*getattr(ranges, name))
        return offset, angles

    def _sample_single_targets(self, env_ids):
        count = len(env_ids)
        if count == 0:
            return
        for side, ranges in enumerate((self.cfg.left_offsets, self.cfg.right_offsets)):
            offset, angles = self._sample_offsets(ranges, count)
            if self.range_curriculum is not None:
                offset *= self.range_curriculum.target_scale(env_ids)[:, None]
                angles *= self.range_curriculum.target_scale(env_ids, orientation=True)[:, None]
            default = self.default_wrist_pose_e[env_ids, side]
            # Position offsets use the default pelvis axes; orientation offsets use the default wrist axes.
            self.goal_pose_w[env_ids, side, :3] = (
                self._env.scene.env_origins[env_ids]
                + default[:, :3]
                + quat_apply(self._sampling_quat_w[env_ids], offset)
            )
            self.goal_pose_w[env_ids, side, 3:] = quat_unique(
                quat_mul(default[:, 3:], quat_from_euler_xyz(*angles.unbind(-1)))
            )

    def _sample_dual_targets(self, env_ids):
        """Construct a pair directly; wrist rotations do not rotate its baseline.

        Position differences use the initial pelvis axes. Separation is the
        Euclidean distance, so Y is solved from sampled distance, X and Z.
        A random relative rotation is split equally about the common wrist
        orientation; its shortest angle is bounded independently of Euler axes.
        """
        count = len(env_ids)
        if count == 0:
            return
        cfg = self.cfg.dual_targets
        offset, angles = self._sample_offsets(cfg, count)
        position_scale = orientation_scale = torch.ones(count, device=self.device)
        if self.range_curriculum is not None:
            position_scale = self.range_curriculum.target_scale(env_ids)
            orientation_scale = self.range_curriculum.target_scale(env_ids, orientation=True)
        default = self.default_wrist_pose_e[env_ids]
        frame = self._sampling_quat_w[env_ids]
        center = (
            self._env.scene.env_origins[env_ids]
            + default[..., :3].mean(dim=1)
            + quat_apply(frame, offset * position_scale[:, None])
        )
        default_delta = quat_apply_inverse(frame, default[:, 0, :3] - default[:, 1, :3])
        distance = torch.lerp(
            default_delta.norm(dim=-1),
            torch.empty(count, device=self.device).uniform_(*cfg.separation_range),
            position_scale,
        )
        delta = torch.empty(count, 3, device=self.device)
        for axis, bounds in ((0, cfg.relative_pos_x), (2, cfg.relative_pos_z)):
            delta[:, axis] = torch.lerp(
                default_delta[:, axis], torch.empty(count, device=self.device).uniform_(*bounds), position_scale
            )
        delta[:, 1] = (distance.square() - delta[:, 0].square() - delta[:, 2].square()).clamp_min(0).sqrt()
        half_delta_w = quat_apply(frame, delta) * 0.5
        self.goal_pose_w[env_ids, 0, :3] = center + half_delta_w
        self.goal_pose_w[env_ids, 1, :3] = center - half_delta_w

        # Sign-aligned quaternion midpoint gives a symmetric default reference.
        left, right = default[:, 0, 3:], default[:, 1, 3:]
        right = torch.where((left * right).sum(-1, keepdim=True) < 0, -right, right)
        common = quat_mul(
            normalize(left + right), quat_from_euler_xyz(*(angles * orientation_scale[:, None]).unbind(-1))
        )
        axis = normalize(torch.randn(count, 3, device=self.device))
        relative_angle = torch.rand(count, device=self.device) * cfg.max_relative_angle * orientation_scale
        half_rotation = quat_from_angle_axis(relative_angle * 0.5, axis)
        self.goal_pose_w[env_ids, 0, 3:] = quat_unique(quat_mul(common, half_rotation))
        self.goal_pose_w[env_ids, 1, 3:] = quat_unique(quat_mul(common, quat_conjugate(half_rotation)))

    def _update_command(self):
        # World goals stay fixed; command converts to the live pelvis frame on access.
        if self._teleop is not None:
            self._teleop.apply_pending(self)

    def _update_metrics(self):
        # Auto-reset happens before command updates. Exclude freshly reset frames;
        # the terminal physics step is not included in these episode means.
        valid = (self._env.episode_length_buf > 0).float()
        active = self.active_mask
        position, orientation = self.pose_error()
        self._position_sum += position * active * valid[:, None]
        self._orientation_sum += orientation * active * valid[:, None]
        self._idle_deviation_sum += self.idle_joint_deviation() * ~active * valid[:, None]
        self._metric_steps += valid
        if self.range_curriculum is not None:
            cfg = self.range_curriculum.cfg
            ready = simultaneous_pose_ready(
                position, orientation, active, cfg.position_tolerance_m, cfg.orientation_tolerance_rad
            )
            self._pose_ready_sum += ready * valid
        self.collect_window_metrics(position=position, orientation=orientation)
        if self._workspace_sums is not None:
            state = self.workspace_quality_state()
            target_valid = state.target_valid & active
            actual_valid = state.actual_valid & active
            self._workspace_sums += (
                torch.stack(
                    (
                        (state.target_scores * target_valid).sum(-1),
                        target_valid.sum(-1),
                        (state.actual_scores * actual_valid).sum(-1),
                        actual_valid.sum(-1),
                        state.position_ready,
                        state.map_covered,
                        state.position_ready & state.map_covered,
                    ),
                    dim=-1,
                )
                * valid[:, None]
            )
        if self.cfg.height_task is not None or self.range_curriculum is not None or self.planar_enabled:
            # Reuse Q's readiness gate when enabled. Without Q diagnostics the
            # same all-active-wrist criterion uses the default 8 mm tolerance.
            ready = (
                state.position_ready
                if self._workspace_sums is not None
                else ((position <= DEFAULT_POSITION_TOLERANCE_M) | ~active).all(dim=-1)
            )
            self._position_ready_sum += ready * valid

    def _episode_metrics(self, env_ids):
        extras = {}
        if self._workspace_sums is not None:
            ids = env_ids[self._metric_steps[env_ids] > 0]
            if len(ids):
                sums = self._workspace_sums[ids]
                for name, column in (("target_score", 0), ("actual_score", 2)):
                    covered = sums[:, column + 1] > 0
                    if covered.any():
                        extras[f"workspace_quality/{name}"] = (sums[covered, column] / sums[covered, column + 1]).mean()
                for name, column in (
                    ("position_ready_fraction", 4),
                    ("map_coverage_fraction", 5),
                    ("reward_available_fraction", 6),
                ):
                    extras[f"workspace_quality/{name}"] = (sums[:, column] / self._metric_steps[ids]).mean()
        for mode, name in enumerate(self.MODE_NAMES):
            ids = env_ids[(self.mode_id[env_ids] == mode) & (self._metric_steps[env_ids] > 0)]
            extras[f"{name}/episodes"] = torch.tensor(float(len(ids)), device=self.device)
            if len(ids) == 0:
                continue
            steps = self._metric_steps[ids]
            for side, label in enumerate(("left", "right")):
                if mode == side or mode == 2:
                    extras[f"{name}/{label}_position_error_m"] = (self._position_sum[ids, side] / steps).mean()
                    extras[f"{name}/{label}_orientation_error_rad"] = (self._orientation_sum[ids, side] / steps).mean()
                else:
                    extras[f"{name}/idle_joint_rms_rad"] = (self._idle_deviation_sum[ids, side] / steps).sqrt().mean()
        extras.update(self._task_metrics(env_ids))
        return extras

    def _task_metrics(self, env_ids):
        """Summarize completed episodes before reset changes their class or mode.

        Failure counts include one-step episodes without pose samples. Pose
        means retain the existing pre-terminal sampling convention. Explicit
        resets without a termination/timeout signal are excluded from this group.
        """
        if self.cfg.height_task is None and self.range_curriculum is None:
            return {}
        terminated = getattr(self._env, "reset_terminated", None)
        timeouts = getattr(self._env, "reset_time_outs", None)
        if terminated is None or timeouts is None:
            return {}
        completed = (terminated[env_ids] | timeouts[env_ids]) & (self._env.episode_length_buf[env_ids] > 0)
        extras = {}
        for category, name in enumerate(RangeCurriculum.CLASS_NAMES):
            if category == 2 and not self.planar_enabled:
                continue
            ids = env_ids[completed & (self.task_ids[env_ids] == category)]
            prefix = f"task/{name}"
            extras[f"{prefix}/episodes"] = torch.tensor(float(len(ids)), device=self.device)
            if len(ids) == 0:
                continue
            extras[f"{prefix}/failure_rate"] = terminated[ids].float().mean()
            seconds = self.elapsed_seconds(ids)
            extras[f"{prefix}/failure_per_10s"] = failure_per_10s(terminated[ids].sum(), seconds.sum())
            if category == 2:
                extras.update(self._planar_metrics(ids, prefix))
                continue
            if category == 1:
                extras[f"{prefix}/height_shift_mean_m"] = self.height_shift[ids].mean()
            ids = ids[self._metric_steps[ids] > 0]
            if len(ids) == 0:
                continue
            steps = self._metric_steps[ids]
            denominator = steps * self.active_mask[ids].sum(-1)
            extras[f"{prefix}/position_error_m"] = (self._position_sum[ids].sum(-1) / denominator).mean()
            extras[f"{prefix}/orientation_error_rad"] = (self._orientation_sum[ids].sum(-1) / denominator).mean()
            extras[f"{prefix}/position_ready_fraction"] = (self._position_ready_sum[ids] / steps).mean()
            if self.range_curriculum is not None:
                extras[f"{prefix}/pose_ready_fraction"] = (self._pose_ready_sum[ids] / steps).mean()
        return extras

    def _planar_metrics(self, ids, prefix):
        sums = self._pose_window.sums[ids]
        expected = self.episode_deadline_steps[ids] - self.evaluation_start_steps[ids]
        result = {
            "eval_pose_ready_fraction": self.episode_quality(ids).mean(),
            "eval_position_ready_fraction": (sums[:, 1] / expected).mean(),
            "eval_coverage_fraction": (sums[:, 4] / expected).mean(),
            "initial_distance_m": self.planar_distance_m[ids].mean(),
            "eval_start_s": (self.evaluation_start_steps[ids].float() * self._env.step_dt).mean(),
            "episode_budget_s": (self.episode_deadline_steps[ids].float() * self._env.step_dt).mean(),
        }
        observed = sums[:, 4] > 0
        if observed.any():
            result["eval_position_error_m"] = (sums[observed, 2] / sums[observed, 4]).mean()
            result["eval_orientation_error_rad"] = (sums[observed, 3] / sums[observed, 4]).mean()
        return {f"{prefix}/{name}": value for name, value in result.items()}

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis and not hasattr(self, "goal_visualizers"):
            base = self.cfg.marker_cfg.prim_path
            self.goal_visualizers = [
                VisualizationMarkers(replace(self.cfg.marker_cfg, prim_path=f"{base}/{side}/goal"))
                for side in ("left", "right")
            ]
            self.current_visualizers = [
                VisualizationMarkers(replace(self.cfg.marker_cfg, prim_path=f"{base}/{side}/current"))
                for side in ("left", "right")
            ]
        if hasattr(self, "goal_visualizers"):
            for marker in (*self.goal_visualizers, *self.current_visualizers):
                marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        current = self.robot.data.body_link_pose_w.torch[:, self.body_indices]
        for side in range(2):
            ids = self._env_ids[self.active_mask[:, side]]
            marker = self.goal_visualizers[side]
            marker.set_visibility(len(ids) > 0)
            if len(ids) > 0:
                marker.visualize(self.goal_pose_w[ids, side, :3], self.goal_pose_w[ids, side, 3:], environment_ids=ids)
            self.current_visualizers[side].visualize(
                current[:, side, :3], current[:, side, 3:], environment_ids=self._env_ids
            )


@configclass
class BimanualPoseCommandCfg(CommandTermCfg):
    class_type: type = BimanualPoseCommand
    asset_name: str = "robot"
    reference_body_name: str = "pelvis"
    body_names: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    mode_probabilities: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    resampling_time_range: tuple[float, float] = (3.0, 5.0)
    distance_reference_speed: float = 0.08
    """Goal-to-pelvis distance / this scale gives the hold time [s], clipped to the range.

    This parameter controls goal resampling, not actuator motion speed.
    """
    workspace_quality: WorkspaceQualityCfg | None = None
    """Optional spatial prior and diagnostics, independent of its reward weight."""
    range_curriculum: RangeCurriculumCfg | None = None
    """Mobile-only episode-owned ability and target ranges; None keeps fixed-base behavior."""

    @configclass
    class HeightTaskCfg:
        """Episode-level target randomization, independent of Lower's command limits."""

        probability: float = 0.3
        """Height-task probability among non-planar episodes; unchanged when planar tasks are off."""
        shift_range: tuple[float, float] = (-0.15, -0.05)
        """Additional world-Z target shift [m], shared by active wrists; positive shifts are allowed."""
        hold_episode: bool = True
        """Hold the full target pose until reset; False retains normal interval resampling."""

    height_task: HeightTaskCfg | None = None
    """None retains original sampling. Probability 0 is a regular-only ablation with class diagnostics."""
    planar_task: PlanarTaskCfg | None = None
    """Optional Mobile planar tasks; height probability applies to the remaining episodes.

    Probability zero preserves the old distribution, RNG and timeout. Planar
    quality uses the full assigned window; all sampled classes gate progression.
    """

    @configclass
    class Ranges:
        """Offsets from the default wrist pose: meters and local Euler rotations [rad]."""

        pos_x: tuple[float, float] = (-0.05, 0.15)
        pos_y: tuple[float, float] = (-0.05, 0.10)
        pos_z: tuple[float, float] = (-0.05, 0.20)
        roll: tuple[float, float] = (-0.3, 0.3)
        pitch: tuple[float, float] = (-0.3, 0.3)
        yaw: tuple[float, float] = (-0.3, 0.3)

    left_offsets: Ranges = Ranges()
    right_offsets: Ranges = Ranges(pos_y=(-0.10, 0.05))

    @configclass
    class DualTargetsCfg(Ranges):
        """Pair-center offsets and common wrist Euler rotation, plus relative limits.

        Position ranges use the initial pelvis axes. Relative positions are
        left minus right [m]; positive Y is solved to enforce separation.
        The relative angle is the full shortest wrist-to-wrist rotation [rad].
        """

        pos_y: tuple[float, float] = (-0.05, 0.05)
        separation_range: tuple[float, float] = (0.25, 0.40)
        relative_pos_x: tuple[float, float] = (-0.05, 0.05)
        relative_pos_z: tuple[float, float] = (-0.05, 0.05)
        max_relative_angle: float = 0.15

    dual_targets: DualTargetsCfg = DualTargetsCfg()
    marker_cfg: VisualizationMarkersCfg = replace(FRAME_MARKER_CFG, prim_path="/Visuals/Command/wrist_pose")

    def __post_init__(self):
        self.marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
