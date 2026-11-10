# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standard PPO baseline for fixed-base G1 single/dual wrist pose tracking."""

from dataclasses import field
from secrets import randbelow

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Keep the project entrypoint's random-by-default seed behavior.
    seed: int = field(default_factory=lambda: randbelow(2**31))
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "g1_upper"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    # TODO(symmetry): 研究 RslRlSymmetryCfg 的左右镜像数据增强，之后独立评估 mirror loss。
    # 同步变换双腕目标/实际位姿、任务掩码、关节状态及当前/历史动作，按模型轴向核对符号。
    # 验证镜像两次还原、运动学与奖励一致性，并通过对照训练评估样本效率和实际耗时。
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
