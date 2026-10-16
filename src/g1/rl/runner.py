# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add curriculum metadata to every upstream runner save/load operation."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import to_dict

from g1.tasks.lower_limb_agent.curriculum_state import (
    STATE_KEY,
    export_curriculum_state,
    import_curriculum_state,
    validate_curriculum_state,
)


def create_curriculum_runner(env, agent_cfg, log_dir=None, *, snapshot_path=None, evaluation=False):
    """Match IsaacLab's runner factory while preserving G1 curriculum checkpoints."""
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError("G1 curriculum checkpoints support OnPolicyRunner only")
    return CurriculumOnPolicyRunner(
        env,
        to_dict(agent_cfg),
        log_dir=log_dir,
        device=agent_cfg.device,
        snapshot_path=snapshot_path,
        evaluation=evaluation,
    )


class CurriculumOnPolicyRunner(OnPolicyRunner):
    """Reuse upstream learning while storing task distribution beside model weights."""

    def __init__(self, *args, snapshot_path=None, evaluation=False, **kwargs):
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Curriculum checkpoints currently support single-process training only")
        self._snapshot_path = snapshot_path
        self._evaluation = evaluation
        self._curriculum_loaded = False
        super().__init__(*args, **kwargs)
        if self.env.unwrapped.cfg.training_phase not in (1, 2, 3):
            raise ValueError("Curriculum runner requires a registered G1 lower-limb phase")
        from .diagnostics import install_performance_logging

        self._close_performance_logging = install_performance_logging(self.logger, self.env)

    def save(self, path, infos=None):
        metadata = {} if infos is None else copy.deepcopy(infos)
        metadata[STATE_KEY] = export_curriculum_state(self.env.unwrapped)
        super().save(path, infos=metadata)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        # Validate before changing either weights or live task distribution. The upstream
        # loader remains responsible for algorithm/optimizer semantics; this adds one CPU read.
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        infos = checkpoint.get("infos")
        state = infos.get(STATE_KEY) if isinstance(infos, dict) else None
        del checkpoint
        if state is None:
            if self._snapshot_path is None:
                raise ValueError(
                    "Checkpoint has no G1 curriculum state. Use a checkpoint saved by the project entrypoint, "
                    "or supply an explicit legacy --curriculum-snapshot JSON file; model weights cannot infer it."
                )
            state = json.loads(Path(self._snapshot_path).read_text(encoding="utf-8-sig"))
        elif self._snapshot_path is not None:
            raise ValueError("--curriculum-snapshot is only for checkpoints without curriculum metadata")
        validate_curriculum_state(self.env.unwrapped, state, evaluation=self._evaluation)
        result = super().load(
            path, load_cfg=load_cfg, strict=strict, map_location=self.device if map_location is None else map_location
        )
        import_curriculum_state(self.env.unwrapped, state, evaluation=self._evaluation)
        self._curriculum_loaded = True
        return result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.env.unwrapped.cfg.training_phase > 1 and not self._curriculum_loaded:
            raise ValueError("Phase 2/3 training requires --checkpoint from the same or immediately preceding phase")
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)
        finally:
            close = getattr(self, "_close_performance_logging", None)
            if close is not None:
                close()
