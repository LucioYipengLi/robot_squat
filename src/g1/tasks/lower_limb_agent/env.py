# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preserve terminal metrics and synchronize upper-body commands after physical resets."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.commands.commands import SquatWalkCommand


class LowerLimbAgentEnv(ManagerBasedRLEnv):
    """Keep terminal metrics and upper-body reset commands aligned with physical state."""

    def _reset_idx(self, env_ids: torch.Tensor | slice):
        # Base _reset_idx applies physical reset events before command_manager.reset().
        # Capture first; the per-step stamps also prevent the later compute/reset hooks
        # from counting reset states or ongoing environments for a second time.
        for name in self.command_manager.active_terms:
            term = self.command_manager.get_term(name)
            if isinstance(term, SquatWalkCommand):
                term._update_metrics()
        super()._reset_idx(env_ids)
        motion = getattr(self, "upper_body_motion", None)
        if motion is not None:
            # Preserve per-episode samples before trajectory.reset clears them.
            # RSL-RL concatenates these vectors, so differently sized reset
            # batches receive their actual episode weights.
            rows = motion.trajectory.statistics[env_ids]
            if rows.shape[0]:
                log = self.extras["log"]
                log["UpperBodyMotion/started_per_episode"] = rows[:, 0].clone()
                log["UpperBodyMotion/pending_replaced_per_episode"] = rows[:, 2].clone()
                moving = rows[rows[:, 0] > 0]
                if moving.shape[0]:
                    log["UpperBodyMotion/reference_completion_rate"] = moving[:, 1] / moving[:, 0]
                    log["UpperBodyMotion/mean_planned_duration_s"] = moving[:, 3] / moving[:, 0]
        self._reset_upper_body_targets(env_ids)

    def _reset_upper_body_targets(self, env_ids: torch.Tensor | slice) -> None:
        # The observation manager owns resolved joint indices, also in play/eval
        # where the random disturbance event is disabled. Reset the same subset.
        asset_cfg = self.observation_manager.cfg.policy.upper_body_target.params["asset_cfg"]
        robot = self.scene[asset_cfg.name]
        position = robot.data.joint_pos.torch[env_ids][:, asset_cfg.joint_ids].clone()
        if position.numel() == 0:
            return
        zero = torch.zeros_like(position)
        # Keep the randomized reset pose, but start uncontrolled upper joints at
        # rest. Lower-body reset velocities and all ongoing environments survive.
        robot.write_joint_velocity_to_sim_index(velocity=zero, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
        command = robot.actuators.target_command
        command.set_position_index(value=position, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
        command.set_velocity_index(value=zero, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
        command.set_effort_index(value=zero, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
        motion = getattr(self, "upper_body_motion", None)
        if motion is not None:
            motion.reset(env_ids)
