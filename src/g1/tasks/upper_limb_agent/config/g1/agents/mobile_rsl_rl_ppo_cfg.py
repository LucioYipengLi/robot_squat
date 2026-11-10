"""Standard Upper PPO for wrist tracking and policy-selected frozen-Lower velocity commands."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class MobilePPORunnerCfg(PPORunnerCfg):
    experiment_name = "g1_upper_mobile"
