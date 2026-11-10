"""Register the G1 upper-limb task without importing its configuration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Template-G1-Upper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UpperLimbAgentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-G1-Upper-Mobile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mobile_env_cfg:UpperLimbMobileEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.mobile_rsl_rl_ppo_cfg:MobilePPORunnerCfg",
    },
)
