# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture tracking statistics before reset events overwrite terminal physical state."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.commands.commands import SquatWalkCommand, UniformHeightCommand


class LowerLimbAgentEnv(ManagerBasedRLEnv):
    """Use the standard RL environment lifecycle with one pre-reset metric hook."""

    def _reset_idx(self, env_ids: torch.Tensor | slice):
        # Base _reset_idx applies physical reset events before command_manager.reset().
        # Capture first; the per-step stamps also prevent the later compute/reset hooks
        # from counting reset states or ongoing environments for a second time.
        for name in self.command_manager.active_terms:
            term = self.command_manager.get_term(name)
            if isinstance(term, (UniformHeightCommand, SquatWalkCommand)):
                term._update_metrics()
        super()._reset_idx(env_ids)
