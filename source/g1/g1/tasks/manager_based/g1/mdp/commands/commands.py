"""
Sub-module containing command generators for the height-based balance task.

Author:
    Lucio.YipengLi@outlook.com
"""

from __future__ import annotations


import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformHeightCommandCfg

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
