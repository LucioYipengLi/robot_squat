# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""机械臂随机干扰事件（loco-manipulation 鲁棒性训练）。

本模块提供 interval 模式的事件项，把**机械臂**作为智能体无法控制的随机干扰源：
周期性地在关节限位内采样机械臂 PD 目标并写入目标缓冲区，由 implicit actuator 的
PD 每步逐步驱动机械臂运动，从而对浮动基下肢产生真实的 CoM 偏移与反作用力扰动。

设计要点（与 BATCH_IK_CASE.md V1 对齐，并修正其类式 EventTerm 接口错误）：

* 机械臂关节**不在**智能体 action space（下肢 action 仅覆盖 hip/knee/ankle），
  故两者写入的 ``joint_pos_target`` 索引集合互不相交，无覆盖冲突；
* ``write_data_to_sim()`` 每步推送整个目标缓冲区（含机械臂），机械臂目标持续生效；
* interval 事件默认 ``is_global_time=False``，各 env 独立计时、异步触发，
  实现"每个 env 的机械臂独立随机干扰"。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class ArmDisturbanceEvent(ManagerTermBase):
    """机械臂随机干扰事件（interval 模式）。智能体无法控制机械臂。

    V1：在默认位姿附近随机采样机械臂关节目标（限制在关节限位内），写入 PD 目标缓冲区，
    由 PD 每步逐步驱动机械臂运动。此版本**不依赖 IK**，机械臂目标直接在关节空间采样。

    该类继承 :class:`~isaaclab.managers.ManagerTermBase`（类式 EventTerm 的强制基类），
    由 :class:`~isaaclab.managers.EventManager` 以 ``func(cfg=term_cfg, env=env)`` 实例化；
    实例化前 ``cfg.params`` 中的 :class:`SceneEntityCfg` 已被自动 resolve（``joint_ids`` 已填充）。

    .. note::
        缓存 :attr:`arm_joint_target` 为全 env 尺寸，仅在触发 env 上更新，供后续 V3
        观测（机械臂目标预告）读取。V1 阶段仅写入不使用。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """初始化干扰事件。

        Args:
            cfg: 事件项配置（``cfg.params`` 需含已 resolve 的 ``asset_cfg``）。
            env: 环境实例。

        Raises:
            RuntimeError: 若 ``asset_cfg`` 解析为全部关节（``slice(None)``），无法用于子集索引。
        """
        # 调用基类：设置 self.cfg / self._env，并启用 self.device / self.num_envs 只读 property
        super().__init__(cfg, env)

        # 读取已被 manager 自动 resolve 的 SceneEntityCfg（joint_ids 已填充）
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[self.asset_cfg.name]
        # 机械臂关节索引（单一数据源）：必须是 list[int] 子集，不能是全选优化后的 slice(None)
        self.arm_joint_ids = self.asset_cfg.joint_ids
        if not isinstance(self.arm_joint_ids, list):
            raise RuntimeError(
                "ArmDisturbanceEvent 要求 asset_cfg 选择机械臂关节子集（排除下肢），"
                f"但解析得到 joint_ids={self.arm_joint_ids}（全选会被优化成 slice(None)，无法用于子集索引）。"
            )
        self._num_arm_joints = len(self.arm_joint_ids)

        # 缓存（供 V3 观测读取）—— 全 env 尺寸，仅触发 env 被更新
        self.arm_joint_target = torch.zeros(self.num_envs, self._num_arm_joints, device=self.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        joint_noise: float,
    ) -> None:
        """Interval event 入口：为触发的 env 子集采样并写入机械臂 PD 目标。

        Args:
            env: 环境实例。
            env_ids: 计时器到期的 env 子集（异步触发）；interval 模式下由 manager 传入，
                为 None 时退化为全部 env。
            asset_cfg: 机械臂关节的 SceneEntityCfg（已在 :meth:`__init__` 解析为 ``self.asset_cfg``，
                此处为 manager 按 params 键回传的同一对象，仅为满足参数签名校验）。
            joint_noise: [rad] 默认位姿附近的随机采样幅度（对称区间 ``[-joint_noise, joint_noise]``）。
        """
        # interval 模式传入触发子集；兜底处理 None（全局时间模式或手动调用）
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)

        # 在默认位姿附近随机采样关节目标，并 clamp 到关节限位内
        # -- default_joint_pos: (num_instances, num_joints) → 取触发 env 与机械臂列 → (n, j)
        default_q = self.robot.data.default_joint_pos.torch[env_ids][:, self.arm_joint_ids]
        # -- sample_uniform 的 lower/upper 用 float 标量（tuple-tuple 会触发 unsupported operand）
        noise = sample_uniform(-joint_noise, joint_noise, (n, self._num_arm_joints), self.device)
        q_target = default_q + noise
        # -- joint_pos_limits: (num_instances, num_joints, 2) [lower, upper] → (n, j, 2)
        limits = self.robot.data.joint_pos_limits.torch[env_ids][:, self.arm_joint_ids]
        q_target = torch.clamp(q_target, limits[..., 0], limits[..., 1])

        # 仅更新触发的 env（遵守 env_ids 写入隔离，避免污染未触发 env）
        self.arm_joint_target[env_ids] = q_target
        # 写入机械臂 PD 目标缓冲区（关键字-only；target 形状 (len(env_ids), len(joint_ids))）；
        # 与下肢 action 写入索引不相交，下一步 write_data_to_sim() 时生效
        self.robot.set_joint_position_target_index(
            target=q_target, joint_ids=self.arm_joint_ids, env_ids=env_ids
        )
