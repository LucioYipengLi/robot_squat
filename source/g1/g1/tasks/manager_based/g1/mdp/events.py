# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""上肢（双臂 + 腰部）随机干扰事件（loco-manipulation 鲁棒性训练）。

本模块提供 interval 模式的事件项，把智能体**无法控制的上肢关节**（双臂 + 腰部）作为随机干扰源：
周期性地在关节限位内采样各关节组的 PD 目标并写入目标缓冲区，由 implicit actuator 的 PD 每步逐步
驱动这些关节运动，从而对浮动基下肢产生真实的 CoM 偏移与反作用力扰动。

每个关节组的采样幅度可**单独定义**（``disturb_groups`` 列表，每组含自己的 ``asset_cfg`` 与
``joint_noise_frac``）：幅度以「默认位姿到各方向关节限位的余量」的百分比 [0, 1] 表示，故同一 frac
在不同关节上自动缩放出不同的绝对弧度——非对称限位（如 elbow 的 [-1.16, +2.09]）、组内 ROM 差异
（如 waist_yaw 的 ±150° >> waist_roll 的 ±30°）都被自动吸收，无需逐关节手调弧度。

设计要点（与 BATCH_IK_CASE.md V1 对齐，并修正其类式 EventTerm 接口错误）：

* 上肢关节**不在**智能体 action space（下肢 action 仅覆盖 hip/knee/ankle），故两者写入的
  ``joint_pos_target`` 索引集合互不相交，无覆盖冲突；各组之间也应互斥（手臂 ∩ 腰部 = ∅）；
* ``write_data_to_sim()`` 每步推送整个目标缓冲区（含上肢），上肢目标持续生效；
* interval 事件默认 ``is_global_time=False``，各 env 独立计时、异步触发，实现 per-env 独立随机干扰；
* EventManager 会**递归** resolve ``params`` 中嵌套于 list/dict 的 ``SceneEntityCfg``
  （``manager_base.py:_resolve_param_value``），故 ``disturb_groups`` 里每组的 ``joint_ids``
  在本类实例化前已填充。
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


class UpperBodyDisturbanceEvent(ManagerTermBase):
    """上肢（双臂 + 腰部）随机干扰事件（interval 模式）。智能体无法控制这些关节。

    V1：在各组默认位姿附近随机采样关节目标（限制在关节限位内），写入 PD 目标缓冲区，由 PD 每步逐步
    驱动关节运动。此版本**不依赖 IK**，目标直接在关节空间采样。每个关节组（如手臂、腰部）的采样幅度
    ``joint_noise_frac``（默认位姿→各方向限位余量的百分比，[0, 1]）可在 ``disturb_groups`` 中单独定义。

    该类继承 :class:`~isaaclab.managers.ManagerTermBase`（类式 EventTerm 的强制基类），由
    :class:`~isaaclab.managers.EventManager` 以 ``func(cfg=term_cfg, env=env)`` 实例化；实例化前
    ``cfg.params["disturb_groups"]`` 里每组的 :class:`SceneEntityCfg` 已被自动 resolve（``joint_ids``
    已填充）。

    .. note::
        缓存 :attr:`joint_target` 为全 env 尺寸（列序 = 各组 ``joint_ids`` 拼接序），仅在触发 env 上
        更新，供后续 V3 观测（上肢目标预告）读取。V1 阶段仅写入不使用。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """初始化干扰事件。

        Args:
            cfg: 事件项配置。``cfg.params["disturb_groups"]`` 需为已 resolve 的分组列表，每组是一个
                dict，含 ``asset_cfg``（:class:`SceneEntityCfg`，``joint_ids`` 已填充）与
                ``joint_noise_frac``（float，[0, 1]，相对「默认位姿→各方向限位余量」的百分比幅度）。
            env: 环境实例。

        Raises:
            RuntimeError: 若某组 ``asset_cfg`` 解析为全部关节（``slice(None)``），无法用于子集索引。
        """
        # 调用基类：设置 self.cfg / self._env，并启用 self.device / self.num_envs 只读 property
        super().__init__(cfg, env)

        groups: list[dict] = cfg.params["disturb_groups"]
        self.robot: Articulation = env.scene[groups[0]["asset_cfg"].name]

        # 逐组校验并拼接关节索引（单一数据源）：每组必须是 list[int] 子集，不能是全选优化后的 slice(None)
        self._group_sizes: list[int] = []
        self._joint_ids: list[int] = []
        for i, group in enumerate(groups):
            joint_ids = group["asset_cfg"].joint_ids
            if not isinstance(joint_ids, list):
                raise RuntimeError(
                    f"UpperBodyDisturbanceEvent 要求第 {i} 组 asset_cfg 选择关节子集（如机械臂/腰部，排除下肢），"
                    f"但解析得到 joint_ids={joint_ids}（全选会被优化成 slice(None)，无法用于子集索引）。"
                )
            self._group_sizes.append(len(joint_ids))
            self._joint_ids.extend(joint_ids)
        self._num_joints = len(self._joint_ids)

        # 缓存（供 V3 观测读取）—— 全 env 尺寸，仅触发 env 被更新；列序与 self._joint_ids 一致
        self.joint_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        disturb_groups: list[dict],
    ) -> None:
        """Interval event 入口：为触发的 env 子集按组采样并写入上肢 PD 目标。

        Args:
            env: 环境实例。
            env_ids: 计时器到期的 env 子集（异步触发）；interval 模式下由 manager 传入，为 None 时
                退化为全部 env。
            disturb_groups: 已 resolve 的分组列表（每组含 ``asset_cfg`` 与 ``joint_noise_frac``）。此处
                由 manager 按 params 键回传**同一对象**——每步现读各组 ``joint_noise_frac``，故课程运行时
                mutate ``params["disturb_groups"][g]["joint_noise_frac"]`` 会在下次触发生效。
        """
        # interval 模式传入触发子集；兜底处理 None（全局时间模式或手动调用）
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)

        # 逐组现读百分比幅度，拼成与 self._joint_ids 列序对齐的逐关节 frac 向量（各组切片填同一 float）
        frac = torch.empty(self._num_joints, device=self.device)
        offset = 0
        for group, size in zip(disturb_groups, self._group_sizes):
            frac[offset : offset + size] = group["joint_noise_frac"]
            offset += size

        # 在「默认位姿 → 各方向关节限位」的余量内按 frac 采样：自动适配非对称限位（如 elbow）与组内
        # ROM 差异（如 waist_yaw >> waist_roll），frac≤1 时目标恒落在限位内。
        # -- default_joint_pos: (num_instances, num_joints) → 取触发 env 与上肢列 → (n, j)
        default_q = self.robot.data.default_joint_pos.torch[env_ids][:, self._joint_ids]
        # -- joint_pos_limits: (num_instances, num_joints, 2) [lower, upper] → (n, j, 2)
        limits = self.robot.data.joint_pos_limits.torch[env_ids][:, self._joint_ids]
        lo, hi = limits[..., 0], limits[..., 1]
        # -- 各方向余量（default→限位）；clamp_min(0) 防御 default 落在限位外导致负余量翻转采样区间
        upper_margin = (hi - default_q).clamp_min(0.0)
        lower_margin = (default_q - lo).clamp_min(0.0)
        # -- sample_uniform 支持张量边界：逐关节**非对称**采样 [-frac·lower_margin, +frac·upper_margin]
        noise = sample_uniform(
            -frac * lower_margin, frac * upper_margin, (n, self._num_joints), self.device
        )
        q_target = default_q + noise
        # -- frac≤1 时已在限位内；clamp 仅作数值安全网（兜住浮点误差与负余量退化情形）
        q_target = torch.clamp(q_target, lo, hi)

        # 仅更新触发的 env（遵守 env_ids 写入隔离，避免污染未触发 env）
        self.joint_target[env_ids] = q_target
        # 写入上肢 PD 目标缓冲区（关键字-only；target 形状 (len(env_ids), len(joint_ids))）；
        # 与下肢 action 写入索引不相交，下一步 write_data_to_sim() 时生效
        self.robot.set_joint_position_target_index(
            target=q_target, joint_ids=self._joint_ids, env_ids=env_ids
        )
