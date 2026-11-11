"""Single/dual wrist commands held in world space and observed in the live pelvis frame."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass, replace
from isaaclab.utils.math import (
    normalize,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    subtract_frame_transforms,
)

from .actions import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from .workspace_quality import WorkspaceQualityCfg, WorkspaceQualityMap, WorkspaceQualityState

__all__ = ["BimanualPoseCommand", "BimanualPoseCommandCfg"]


class BimanualPoseCommand(CommandTerm):
    """Sample a mode per episode and default-wrist offsets per command interval.

    Modes are left-only, right-only and both. World goals are authoritative;
    ``command`` returns their current pelvis-relative poses, left then right.
    Inactive goals have a neutral observation and contribute no tracking reward.
    Optional height tasks add one episode-wide world-Z shift shared by active
    wrists. Their complete poses can be held until reset, independently of Lower.
    """

    cfg: BimanualPoseCommandCfg
    MODE_NAMES = ("left_only", "right_only", "both")

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
        self.height_shift = torch.zeros(self.num_envs, device=self.device)
        self.goal_pose_w = torch.zeros(self.num_envs, 2, 7, device=self.device)
        self.goal_pose_w[..., 6] = 1.0
        self.default_wrist_pose_e = self.goal_pose_w.clone()
        self._sampling_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._sampling_quat_w[:, 3] = 1.0
        self._reference_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._position_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self._orientation_sum = torch.zeros_like(self._position_sum)
        self._idle_deviation_sum = torch.zeros_like(self._position_sum)
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)
        self._position_ready_sum = torch.zeros(self.num_envs, device=self.device)
        self.workspace_quality_map = None
        self._workspace_sums = None
        if cfg.workspace_quality is not None:
            self.workspace_quality_map = WorkspaceQualityMap(cfg.workspace_quality.map_path, self.device)
            # Score sums/counts for target and actual positions, readiness,
            # coverage and their joint reward-availability gate.
            self._workspace_sums = torch.zeros(self.num_envs, 7, device=self.device)

    @staticmethod
    def _validate_cfg(cfg):
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
        for ranges in (cfg.left_offsets, cfg.right_offsets):
            for axis in ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"):
                bounds = getattr(ranges, axis)
                if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                    raise ValueError(f"Invalid wrist offset range: {axis}={bounds}")

    def _body_index(self, name):
        ids, _ = self.robot.find_bodies(name)
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one link matching {name!r}, got {len(ids)}")
        return ids[0]

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
        for buffer in (
            self._position_sum, self._orientation_sum, self._idle_deviation_sum,
            self._metric_steps, self._position_ready_sum,
        ):
            buffer[ids] = 0
        if self._workspace_sums is not None:
            self._workspace_sums[ids] = 0
        self._capture_default_pose(ids)
        probabilities = torch.tensor(self.cfg.mode_probabilities, device=self.device, dtype=torch.float32)
        self.mode_id[ids] = torch.multinomial(probabilities, len(ids), replacement=True)
        self._sample_height_task(ids)
        super().reset(ids)
        return extras

    def _sample_height_task(self, env_ids):
        """Sample only at reset; disabling height tasks preserves the old RNG sequence."""
        self.height_task_mask[env_ids] = False
        self.height_shift[env_ids] = 0
        height = self.cfg.height_task
        if height is None or height.probability == 0:
            return
        if height.probability == 1:
            self.height_task_mask[env_ids] = True
        else:
            self.height_task_mask[env_ids] = torch.rand(len(env_ids), device=self.device) < height.probability
        ids = env_ids[self.height_task_mask[env_ids]]
        self.height_shift[ids] = torch.empty(len(ids), device=self.device).uniform_(*height.shift_range)

    def _capture_default_pose(self, env_ids):
        """Capture each environment once, after default joint reset and before physics steps."""
        ids = env_ids[~self._reference_initialized[env_ids]]
        if len(ids) == 0:
            return
        poses = self.robot.data.body_link_pose_w.torch
        self.default_wrist_pose_e[ids] = poses[ids][:, self.body_indices]
        self.default_wrist_pose_e[ids, :, :3] -= self._env.scene.env_origins[ids, None, :]
        self._sampling_quat_w[ids] = poses[ids, self.reference_body_idx, 3:]
        self._reference_initialized[ids] = True

    def _resample(self, env_ids):
        """Keep height-task poses for the episode; regular goals use distance-based intervals."""
        ids = self._env_ids[env_ids]
        height = self.cfg.height_task
        if height is not None and height.hold_episode:
            held = self.height_task_mask[ids] & (self.command_counter[ids] > 0)
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
        self.command_counter[ids] += 1

    def _resample_command(self, env_ids):
        count = len(env_ids)
        for side, ranges in enumerate((self.cfg.left_offsets, self.cfg.right_offsets)):
            offset = torch.empty(count, 3, device=self.device)
            angles = torch.empty_like(offset)
            for axis, name in enumerate(("pos_x", "pos_y", "pos_z")):
                offset[:, axis].uniform_(*getattr(ranges, name))
            for axis, name in enumerate(("roll", "pitch", "yaw")):
                angles[:, axis].uniform_(*getattr(ranges, name))
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
        # Add after rotation: this is world vertical even if the sampling pelvis
        # is tilted. Inactive goals keep their original, masked-out poses.
        self.goal_pose_w[env_ids, :, 2] += self.height_shift[env_ids, None] * self.active_mask[env_ids]

    def _update_command(self):
        # World goals stay fixed; command converts to the live pelvis frame on access.
        pass

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
        if self.cfg.height_task is not None:
            # Reuse Q's readiness gate when enabled. Without Q diagnostics the
            # same all-active-wrist criterion uses the standard 1 cm tolerance.
            ready = state.position_ready if self._workspace_sums is not None else (
                (position <= 0.01) | ~active
            ).all(dim=-1)
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
        if self.cfg.height_task is None:
            return {}
        terminated = getattr(self._env, "reset_terminated", None)
        timeouts = getattr(self._env, "reset_time_outs", None)
        if terminated is None or timeouts is None:
            return {}
        completed = (terminated[env_ids] | timeouts[env_ids]) & (self._env.episode_length_buf[env_ids] > 0)
        extras = {}
        for shifted, name in ((False, "regular"), (True, "height_shifted")):
            ids = env_ids[completed & (self.height_task_mask[env_ids] == shifted)]
            prefix = f"task/{name}"
            extras[f"{prefix}/episodes"] = torch.tensor(float(len(ids)), device=self.device)
            if len(ids) == 0:
                continue
            extras[f"{prefix}/failure_rate"] = terminated[ids].float().mean()
            if shifted:
                extras[f"{prefix}/height_shift_mean_m"] = self.height_shift[ids].mean()
            ids = ids[self._metric_steps[ids] > 0]
            if len(ids) == 0:
                continue
            steps = self._metric_steps[ids]
            denominator = steps * self.active_mask[ids].sum(-1)
            extras[f"{prefix}/position_error_m"] = (self._position_sum[ids].sum(-1) / denominator).mean()
            extras[f"{prefix}/orientation_error_rad"] = (self._orientation_sum[ids].sum(-1) / denominator).mean()
            extras[f"{prefix}/position_ready_fraction"] = (self._position_ready_sum[ids] / steps).mean()
        return extras

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

    @configclass
    class HeightTaskCfg:
        """Episode-level target randomization, independent of Lower's command limits."""

        probability: float = 0.3
        """Probability of selecting the height-shifted target class at episode reset."""
        shift_range: tuple[float, float] = (-0.15, -0.05)
        """Additional world-Z target shift [m], shared by active wrists; positive shifts are allowed."""
        hold_episode: bool = True
        """Hold the full target pose until reset; False retains normal interval resampling."""

    height_task: HeightTaskCfg | None = None
    """None retains original sampling. Probability 0 is a regular-only ablation with class diagnostics."""
    # TODO(curriculum): Update the offset ranges, distance_reference_speed and/or
    # resampling_time_range through the command's cfg; preserve world goals and
    # the current interval until the next sample. Keep curriculum policy outside this term.

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
    marker_cfg: VisualizationMarkersCfg = replace(FRAME_MARKER_CFG, prim_path="/Visuals/Command/wrist_pose")

    def __post_init__(self):
        self.marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
