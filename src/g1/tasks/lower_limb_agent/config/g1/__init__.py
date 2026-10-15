# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-G1-v0",
    entry_point="g1.tasks.lower_limb_agent.env:LowerLimbAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LowerLimbAgentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# ── Phase 2: Static physics DR (resume from Phase 1 checkpoint) ─────────────────
gym.register(
    id="Template-G1-Phase2-v0",
    entry_point="g1.tasks.lower_limb_agent.env:LowerLimbAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_phase:LowerLimbAgentEnvCfg_Phase2",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# ── Phase 3: External push disturbance (resume from Phase 2 checkpoint) ─────────
gym.register(
    id="Template-G1-Phase3-v0",
    entry_point="g1.tasks.lower_limb_agent.env:LowerLimbAgentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_phase:LowerLimbAgentEnvCfg_Phase3",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
