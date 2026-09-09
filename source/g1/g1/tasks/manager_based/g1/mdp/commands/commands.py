"""
Sub-module containing command generators for the height-based balance and
squat-walk locomotion tasks.

Author:
    Lucio.YipengLi@outlook.com
"""

from __future__ import annotations


import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import SquatWalkCommandCfg, UniformHeightCommandCfg

# import logger
logger = logging.getLogger(__name__)

class UniformHeightCommand(CommandTerm):
    r"""Command generator that generates a pelvis height command in SE(1) from a uniform distribution.

    The command comprises a pelvis height offset along the z direction, expressed relative to
    the robot's default standing height. A fraction of environments (``cfg.rel_default_envs``)
    is forced to stand at the default height (zero offset) to regularize training.

    The offset form is exposed to the policy through the :attr:`command` property, while the
    equivalent absolute target height in the world frame is maintained internally for
    tracking-error metrics and debug visualization.
    """

    cfg: UniformHeightCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the lower bound of the pelvis height offset range is above the upper bound.
        """

        # Init the base class
        super().__init__(cfg, env)

        # check config
        if self.cfg.ranges.height_offset[0] > self.cfg.ranges.height_offset[1]:
            raise ValueError(
                "The pelvis height command has an invalid offset range. The lower bound is above the upper bound."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: pelvis height offset relative to the default standing height [m]
        #    (exposed to the policy via the ``command`` property)
        self.height_command_offset = torch.zeros(self.num_envs, 1, device=self.device)
        # -- command: absolute target pelvis height in world frame [m]
        #    (used for tracking-error metrics, reward, and vis)
        self.height_command_world = torch.zeros(self.num_envs, 1, device=self.device)
        # -- mask for env that must stand at default height
        self.is_default_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics: finalized per-episode means/rates, written at ``reset`` and read by the base class
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        # -- per-episode running sums (cleared at episode reset)
        self._error_height_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)
        
        # adds cmd kind and element names for leapp export
        # during export, semantic data about this command will be used to annotate the command input
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/height"
        self.cfg.element_names = self.cfg.element_names or ["pelvis_height_offset"]
    
    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range} (s)\n"
        msg += f"\tPelvis height offset range: {self.cfg.ranges.height_offset} (m)\n"
        msg += f"\tHeight success threshold: {self.cfg.height_success_threshold} (m)\n"
        msg += f"\tDefault standing probability: {self.cfg.rel_default_envs}"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        """The desired pelvis height offset relative to the default standing height [m]. Shape is (num_envs, 1)."""
        return self.height_command_offset
    
    def _update_metrics(self):
        """Accumulate the per-step height tracking error sums and the step counter.

        The per-episode mean is finalized in :meth:`reset` so the value is independent of
        episode length and ``resampling_time_range``.
        """
        error_height = torch.abs(
            self.robot.data.root_pos_w.torch[:, 2] - self.height_command_world[:, 0]
        )
        self._error_height_sum += error_height
        self._step_count += 1.0
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Finalize per-episode height tracking metrics before the base class logs them.

        The episode-mean tracking error and a binary success flag (mean error below the
        configured threshold) are written to ``self.metrics`` prior to ``super().reset()``,
        which reads, logs, and zeros them. The running accumulators are cleared afterwards
        so the next episode starts clean.

        Args:
            env_ids: The environment indices whose episodes just ended. Defaults to None, in
                which case all environments are considered.

        Returns:
            Per-episode metrics to be logged by the command manager under ``Metrics/<term_name>/``.
        """
        
        # resolve env_ids for indexing
        if env_ids is None:
            env_ids = slice(None)

        # -- stage 1: finalize the just_ended episode's metrics
        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_error_height = self._error_height_sum[env_ids] / denom
        self.metrics["error_height"][env_ids] = mean_error_height
        self.metrics["success_rate"][env_ids] = (
            mean_error_height < self.cfg.height_success_threshold
        ).float()

        # --stage 2: let the base class log, zero metrics, and resample the command
        extras = super().reset(env_ids)
        # route success_rate to the unified ``Metrics/success_rate`` path (shared
        # TensorBoard/wandb card across tasks); pop it so CommandManager does not
        # additionally log it under ``Metrics/<term_name>/success_rate``
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = extras.pop("success_rate")
        
        # -- stage 3: clear the running accumulators for the reset environments
        self._error_height_sum[env_ids] = 0.0
        self._step_count[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the pelvis height command for the specified environments.

        The height offset is sampled uniformly from ``cfg.ranges.height_offset``. Environments
        selected as default (with probability ``cfg.rel_default_envs``) are forced to a zero
        offset. The absolute target height in the world frame is then composed from the env
        origin, the default pelvis height, and the sampled offset.

        Args:
            env_ids: The environment indices to resample the command for.
        """
        # sample height commands
        h = torch.empty(len(env_ids), device=self.device)
        # -- pelvis height offset relative to the default standing height
        self.height_command_offset[env_ids, 0] = h.uniform_(*self.cfg.ranges.height_offset)
        # -- update default envs
        self.is_default_env[env_ids] = h.uniform_(0.0, 1.0) <= self.cfg.rel_default_envs
        default_env_ids = env_ids[self.is_default_env[env_ids]]
        self.height_command_offset[default_env_ids, 0] = 0.0

        # compose the absolute pelvis height target in world frame
        default_pelvis_height = self.robot.data.default_root_pose.torch[env_ids, 2]
        # -- world frame height, it will be 0.0 at flat plane.
        env_origin_z = self._env.scene.env_origins[env_ids, 2]
        self.height_command_world[env_ids, 0] = (
            env_origin_z + default_pelvis_height + self.height_command_offset[env_ids, 0]
        )

    def _update_command(self):
        """Post-processes the height command.

        Nothing to do here: default environments are already enforced at resample time and the
        height command stays constant between resamples.
        """
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_height_visualizer"):
                # -- goal pelvis height
                self.goal_height_visualizer = VisualizationMarkers(self.cfg.goal_height_visualizer_cfg)
                # -- current pelvis height
                self.current_height_visualizer = VisualizationMarkers(self.cfg.current_height_visualizer_cfg)
            # set their visibility to true
            self.goal_height_visualizer.set_visibility(True)
            self.current_height_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer.set_visibility(False)
                self.current_height_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Visualize the goal and current pelvis heights with sphere markers.

        The goal marker is placed at the robot's xy position with z set to the commanded
        absolute height; the current marker follows the actual pelvis position.
        """
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # spheres are rotation-invariant; reuse the root orientation as a valid placeholder
        root_quat_w = self.robot.data.root_quat_w.torch
        # -- goal: at the robot's xy position, but z set to the commanded absolute height
        goal_pos_w = self.robot.data.root_pos_w.torch.clone()
        goal_pos_w[:, 2] = self.height_command_world[:, 0]
        # -- current: the actual pelvis (root) position
        current_pos_w = self.robot.data.root_pos_w.torch
        # display markers
        self.goal_height_visualizer.visualize(goal_pos_w, root_quat_w)
        self.current_height_visualizer.visualize(current_pos_w, root_quat_w)


class SquatWalkCommand(CommandTerm):
    r"""Unified command generator coupling pelvis height and base velocity into one joint
    per-episode task distribution.

    The command is a 4-D vector ``[h_offset, vx, vy, wz]``: a pelvis height offset relative
    to the default standing height, plus a base-frame linear/angular velocity target. At the
    start of every episode each environment is assigned one of four discrete modes
    (:attr:`mode`, drawn with weights ``cfg.rel_mode_envs``), and the per-dimension sampling
    rules follow from it:

    =================  ============================  =============================
    Mode                 Height offset                 Velocity (vx, vy, wz)
    =================  ============================  =============================
    STAND (0)            forced 0 (default height)     forced 0
    SQUAT (1)            U(cfg.ranges.height_offset)   forced 0
    WALK (2)             forced 0 (default height)     U(cfg.ranges.lin/ang_vel_*)
    SQUAT_WALK (3)       U(cfg.ranges.height_offset)   U(cfg.ranges.lin/ang_vel_*)
    =================  ============================  =============================

    A single resampling clock drives both dimensions at different cadences: the height
    offset is redrawn at every resample (mid-episode height changes are part of the balance
    skill), while the velocity is redrawn only at the episode-start resample
    (``command_counter == 0``), so the gait target never switches mid-episode.

    The absolute world-frame height target (:attr:`height_command_world`) keeps the interface
    consumed by the existing height-gated reward terms, and :attr:`mode` is exposed for
    mode-based reward gating. The offset form and the velocity are exposed to the policy
    through the 4-D :attr:`command` property.
    """

    cfg: SquatWalkCommandCfg
    """The configuration of the command generator."""

    # per-episode task mode identifiers (values of the ``mode`` buffer)
    MODE_STAND = 0
    MODE_SQUAT = 1
    MODE_WALK = 2
    MODE_SQUAT_WALK = 3

    def __init__(self, cfg: SquatWalkCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the lower bound of the pelvis height offset range is above the upper bound.
            ValueError: If ``rel_mode_envs`` is not a 4-tuple of non-negative weights with a positive sum.
        """
        # Init the base class
        super().__init__(cfg, env)

        # check config
        if self.cfg.ranges.height_offset[0] > self.cfg.ranges.height_offset[1]:
            raise ValueError(
                "The pelvis height command has an invalid offset range. The lower bound is above the upper bound."
            )
        if (
            len(self.cfg.rel_mode_envs) != 4
            or any(p < 0.0 for p in self.cfg.rel_mode_envs)
            or sum(self.cfg.rel_mode_envs) <= 0.0
        ):
            raise ValueError(
                "The task command requires `rel_mode_envs` as a 4-tuple of non-negative mode weights"
                " (STAND, SQUAT, WALK, SQUAT_WALK) with a positive sum."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: unified 4-D task command [h_offset, vx, vy, wz]
        #    (exposed to the policy via the ``command`` property)
        self.task_command = torch.zeros(self.num_envs, 4, device=self.device)
        # -- views into the unified buffer (in-place writes propagate to ``task_command``)
        #    height offset relative to the default standing height [m]
        self.height_command_offset = self.task_command[:, :1]
        #    base-frame velocity target [m/s, m/s, rad/s]
        self.vel_command_b = self.task_command[:, 1:4]
        # -- command: absolute target pelvis height in world frame [m]
        #    (used for tracking-error metrics, reward, and vis)
        self.height_command_world = torch.zeros(self.num_envs, 1, device=self.device)
        # -- per-episode task mode buffer (one of MODE_STAND / MODE_SQUAT / MODE_WALK / MODE_SQUAT_WALK)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # -- mode of the just-ended episode, captured in ``reset`` right before ``mode`` is redrawn.
        #    It stays aligned with the finalized per-episode ``metrics`` (which describe that same
        #    ended episode), so consumers -- e.g. the command-range curriculum -- can mask a metric
        #    by the mode that produced it. ``mode`` itself already points at the new episode by then.
        self.last_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # -- mask for env commanded to the default height (STAND or WALK modes)
        self.is_default_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- normalized mode sampling weights
        self._mode_probs = torch.tensor(self.cfg.rel_mode_envs, dtype=torch.float, device=self.device)
        # -- metrics: finalized per-episode means/rates, written at ``reset`` and read by the base class
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate_vel"] = torch.zeros(self.num_envs, device=self.device)
        # -- persisted per-episode success buffers for the command-range curriculum, kept OUTSIDE
        #    ``self.metrics``: the base ``CommandTerm.reset`` zeroes every ``self.metrics`` entry right
        #    after logging it, and ``CurriculumManager.compute`` runs *before* the next command reset
        #    (see ``ManagerBasedRLEnv._reset_idx``), so a curriculum reading ``self.metrics`` would
        #    always see zeros. These hold each env's last finalized episode success, never zeroed.
        self.episode_success_rate = torch.zeros(self.num_envs, device=self.device)
        self.episode_success_rate_vel = torch.zeros(self.num_envs, device=self.device)
        # -- per-episode running sums (cleared at episode reset)
        self._error_height_sum = torch.zeros(self.num_envs, device=self.device)
        self._error_xy_sum = torch.zeros(self.num_envs, device=self.device)
        self._error_yaw_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)

        # adds cmd kind and element names for leapp export
        # during export, semantic data about this command will be used to annotate the command input
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/height_velocity"
        self.cfg.element_names = self.cfg.element_names or [
            "pelvis_height_offset",
            "lin_vel_x",
            "lin_vel_y",
            "ang_vel_z",
        ]

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "SquatWalkCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range} (s)\n"
        msg += f"\tPelvis height offset range: {self.cfg.ranges.height_offset} (m)\n"
        msg += f"\tLinear velocity range: x={self.cfg.ranges.lin_vel_x}, y={self.cfg.ranges.lin_vel_y} (m/s)\n"
        msg += f"\tAngular velocity range: z={self.cfg.ranges.ang_vel_z} (rad/s)\n"
        msg += f"\tMode weights (STAND, SQUAT, WALK, SQUAT_WALK): {self.cfg.rel_mode_envs}\n"
        msg += f"\tHeight success threshold: {self.cfg.height_success_threshold} (m)"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The unified task command ``[h_offset, vx, vy, wz]``. Shape is (num_envs, 4).

        The height offset is relative to the default standing height [m]; the velocity
        components are expressed in the robot's base frame [m/s, m/s, rad/s].
        """
        return self.task_command

    def _update_metrics(self):
        """Accumulate the per-step height and velocity tracking error sums and the step counter.

        The per-episode means are finalized in :meth:`reset` so the values are independent of
        episode length and ``resampling_time_range``.
        """
        error_height = torch.abs(
            self.robot.data.root_pos_w.torch[:, 2] - self.height_command_world[:, 0]
        )
        # velocity command is in the base frame; compare against base-frame velocities
        error_xy = torch.linalg.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1
        )
        error_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b.torch[:, 2])
        self._error_height_sum += error_height
        self._error_xy_sum += error_xy
        self._error_yaw_sum += error_yaw
        self._step_count += 1.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Draw the new episode's task modes and finalize tracking metrics.

        The episode-mean height/velocity errors and binary success flags are written to
        ``self.metrics`` prior to ``super().reset()``, which reads, logs, and zeros them and
        triggers the command resample. The per-episode mode is drawn *before* the base-class
        resample chain so that :meth:`_resample_command` consumes the fresh modes. The running
        accumulators are cleared afterwards so the next episode starts clean.

        Args:
            env_ids: The environment indices whose episodes just ended. Defaults to None, in
                which case all environments are considered.

        Returns:
            Per-episode metrics to be logged by the command manager under ``Metrics/<term_name>/``.
        """
        # resolve env_ids to a tensor index (the manager always passes tensors; None is
        # accepted for API compatibility)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # -- stage 1: finalize the just-ended episode's metrics
        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_error_height = self._error_height_sum[env_ids] / denom
        self.metrics["error_height"][env_ids] = mean_error_height
        self.metrics["success_rate"][env_ids] = (
            mean_error_height < self.cfg.height_success_threshold
        ).float()
        mean_error_xy = self._error_xy_sum[env_ids] / denom
        mean_error_yaw = self._error_yaw_sum[env_ids] / denom
        self.metrics["error_vel_xy"][env_ids] = mean_error_xy
        self.metrics["error_vel_yaw"][env_ids] = mean_error_yaw
        self.metrics["success_rate_vel"][env_ids] = (
            (mean_error_xy < self.cfg.vel_xy_success_threshold)
            & (mean_error_yaw < self.cfg.vel_yaw_success_threshold)
        ).float()

        # -- persist the finalized success for the command-range curriculum: ``super().reset()`` below
        #    zeroes every ``self.metrics`` entry, and the curriculum's ``compute`` runs before the next
        #    reset, so it must read a buffer that survives the zeroing (``episode_success_rate*``).
        self.episode_success_rate[env_ids] = self.metrics["success_rate"][env_ids]
        self.episode_success_rate_vel[env_ids] = self.metrics["success_rate_vel"][env_ids]

        # -- capture the ending episode's mode so it stays aligned with the metrics finalized above
        #    (stage 2 below overwrites ``self.mode`` with the freshly drawn new-episode mode)
        self.last_mode[env_ids] = self.mode[env_ids]

        # -- stage 2: draw the per-episode task modes before the resample chain consumes them
        num = len(env_ids)
        self.mode[env_ids] = torch.multinomial(self._mode_probs.expand(num, -1), num_samples=1).squeeze(-1)

        # -- stage 3: let the base class log, zero metrics, and resample the command
        extras = super().reset(env_ids)
        # route the height-based success_rate to the unified ``Metrics/success_rate`` path
        # (shared TensorBoard/wandb card across tasks, comparable with the height-only task);
        # pop it so CommandManager does not additionally log it under ``Metrics/<term_name>/``
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = extras.pop("success_rate")

        # -- stage 4: clear the running accumulators for the reset environments
        self._error_height_sum[env_ids] = 0.0
        self._error_xy_sum[env_ids] = 0.0
        self._error_yaw_sum[env_ids] = 0.0
        self._step_count[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the task command for the specified environments under the current modes.

        Height: sampled uniformly from ``cfg.ranges.height_offset`` and forced to zero for
        STAND/WALK modes; the absolute world-frame target is then composed from the env
        origin, the default pelvis height, and the sampled offset. Redrawn at *every*
        resample so mid-episode height changes stay in the training distribution.

        Velocity: sampled uniformly from the configured velocity ranges for WALK/SQUAT_WALK
        modes and zeroed otherwise, but *only* at the episode-start resample. The base class
        increments ``command_counter`` after ``_resample_command``, so ``counter == 0``
        identifies exactly the resample triggered by :meth:`reset`; mid-episode resamples
        leave the velocity untouched (no gait switching within an episode).

        Args:
            env_ids: The environment indices to resample the command for.
        """
        mode = self.mode[env_ids]
        squat_mask = (mode == self.MODE_SQUAT) | (mode == self.MODE_SQUAT_WALK)

        # -- height dimension: pelvis height offset relative to the default standing height
        h = torch.empty(len(env_ids), device=self.device)
        self.height_command_offset[env_ids, 0] = h.uniform_(*self.cfg.ranges.height_offset)
        # -- STAND / WALK modes stand at the default height
        self.height_command_offset[env_ids[~squat_mask], 0] = 0.0
        self.is_default_env[env_ids] = ~squat_mask

        # compose the absolute pelvis height target in world frame
        default_pelvis_height = self.robot.data.default_root_pose.torch[env_ids, 2]
        # -- world frame height, it will be 0.0 at flat plane.
        env_origin_z = self._env.scene.env_origins[env_ids, 2]
        self.height_command_world[env_ids, 0] = (
            env_origin_z + default_pelvis_height + self.height_command_offset[env_ids, 0]
        )

        # -- velocity dimension: redrawn only at the episode-start resample
        first_of_episode = self.command_counter[env_ids] == 0
        episode_start_ids = env_ids[first_of_episode]
        if len(episode_start_ids) > 0:
            walk_mask = (
                (mode == self.MODE_WALK) | (mode == self.MODE_SQUAT_WALK)
            ) & first_of_episode
            walk_ids = env_ids[walk_mask]
            still_ids = episode_start_ids[~walk_mask[first_of_episode]]
            r = torch.empty(len(walk_ids), device=self.device)
            # -- linear velocity - x direction
            self.vel_command_b[walk_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
            # -- linear velocity - y direction
            self.vel_command_b[walk_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
            # -- ang vel yaw - rotation around z
            self.vel_command_b[walk_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
            # -- STAND / SQUAT modes keep a zero velocity target
            self.vel_command_b[still_ids] = 0.0

    def _update_command(self):
        """Post-processes the task command.

        Nothing to do here: mode constraints are fully enforced at resample time and both
        command dimensions stay constant between resamples (no heading-control loop).
        """
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        Creates the four markers (goal/current height spheres and goal/current velocity
        arrows) on first enable and toggles their visibility otherwise.
        """
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_height_visualizer"):
                # -- goal / current pelvis height
                self.goal_height_visualizer = VisualizationMarkers(self.cfg.goal_height_visualizer_cfg)
                self.current_height_visualizer = VisualizationMarkers(self.cfg.current_height_visualizer_cfg)
                # -- goal / current base velocity
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_height_visualizer.set_visibility(True)
            self.current_height_visualizer.set_visibility(True)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer.set_visibility(False)
                self.current_height_visualizer.set_visibility(False)
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Visualize the goal/current pelvis heights (spheres) and velocities (arrows)."""
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # spheres are rotation-invariant; reuse the root orientation as a valid placeholder
        root_quat_w = self.robot.data.root_quat_w.torch
        # -- goal: at the robot's xy position, but z set to the commanded absolute height
        goal_pos_w = self.robot.data.root_pos_w.torch.clone()
        goal_pos_w[:, 2] = self.height_command_world[:, 0]
        # -- current: the actual pelvis (root) position
        current_pos_w = self.robot.data.root_pos_w.torch
        # display height markers
        self.goal_height_visualizer.visualize(goal_pos_w, root_quat_w)
        self.current_height_visualizer.visualize(current_pos_w, root_quat_w)
        # -- arrow anchor: above the robot base, matching the built-in velocity command style
        arrow_pos_w = self.robot.data.root_pos_w.torch.clone()
        arrow_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.vel_command_b[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b.torch[:, :2]
        )
        # display velocity markers
        self.goal_vel_visualizer.visualize(arrow_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(arrow_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

