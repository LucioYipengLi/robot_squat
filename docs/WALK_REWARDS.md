# WALK：实际离地驱动的参考奖励

本机采用 [下游整改记录](WALK_REWARD_REVISION.md) 中的实际离地驱动方案，并按本机实验要求保留
接触计时阈值、高度跟踪和竖直速度约束。原始下游文件为只读记录；本文件说明本机实现与差异。

## 配置与模式

参数位于 `src/g1/tasks/lower_limb_agent/config/g1/env_cfg.py` 的 `RewardsCfg`。
Phase 2/3 通过继承获得相同配置。观测仍为每帧 130 维、五帧 650 维，动作仍为 12 维。

| 项目 | 本机配置 |
| --- | --- |
| `walk_joint_reference` | 新增，权重 +2.0 |
| `walk_support_timing` | 新增，权重 -0.75 |
| `leg_synergy_walk` | 配置已删除；静止模式的 `leg_synergy` 保留 |
| `feet_gait_phase` | 仅 SQUAT_WALK 生效，权重与频率参数保留 |
| `track_pelvis_height` | 所有模式维持权重 +2.0、原公式和原参数 |
| `vertical_bounce` | 所有模式维持 `lin_vel_z_l2`、权重 -0.5 |
| 接触计时阈值 | 显式保留本机 PhysX 原有 1.0 N |

本机原本没有 `feet_swing_height` 项，无需删除相关实现。
新增奖励只在 `mode == WALK` 且平面指令速度 > 0.1 m/s 或转向指令绝对值 > 0.2 rad/s 时生效。
六维指令仍为 `[height_offset, vx, vy, wz, pelvis_dx, pelvis_dy]`。

## 接触与参考

新增奖励和物理日志共用 `walk.py` 的计时读取：`current_contact_time > 0` 表示接触，
`current_air_time` 表示连续离地时长。由传感器按法向接触力向量模长和 1.0 N 阈值维护这两个计时器，
奖励不额外用 0.5 N 竖直力判断，也不维护独立的相位缓存。
既有 `no_fly`、防滑等项继续使用各自原有口径。

`walk_joint_reference` 以离地时长构造半正弦摆腿参考，默认参数：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `swing_time` | 0.4 s | 摆动参考时长，与时长惩罚共用 `WALK_SWING_TIME` 默认值 |
| `hip_amplitude` | 0.30 rad | 髋俯仰基准幅度 |
| `knee_amplitude` | 0.50 rad | 摆动膝屈曲基准增量 |
| `ankle_amplitude` | 0.12 rad | 踝俯仰基准幅度 |
| `sigma` | 0.20 rad | 六关节均方误差指数核的带宽 |
| `speed_reference` | 0.6 m/s | 幅度缩放参考，缩放比例限制为 0.5～1.0 |
| `command_threshold` | 0.1 | 平面运动阈值；转向阈值为其两倍 |

前进时摆动腿髋前摆、膝屈曲、踝背屈，支撑腿髋踝参考方向相反；倒退时反转髋踝方向。
纯侧移和转向仅保留摆动腿屈膝参考。只在单支撑时给分，并以摆动信号加权。
双支撑、腾空和离地超过参考时长均不获得参考奖励；0.4 s 不代表强制的完整步态周期。

`walk_support_timing` 对非单支撑施加即时单位代价；连续离地超过 0.4 s 或连续接触超过
`max_stance_time=0.6 s` 后，逐脚超时代价增加并封顶。原始代价为 `[0,2]`。
它鼓励支撑交换，但不强制左右交替，也会惩罚短暂双支撑。

所有新函数只读取物理状态，权重和控制步长由 RewardManager 施加一次。
旧 WALK 协同项的命令行覆盖应改用新名称，例如：

```text
env.rewards.walk_joint_reference.weight=1.0
```

更改摆动时长时需同步覆盖两个奖励的 `swing_time`；默认配置共用同一个常量。

## 物理指标与验收

新增三项原始标签位于 `Performance/task_command/walk/`，同时进入“02 跟踪详情”面板：

| 后缀 | 定义 |
| --- | --- |
| `single_stance_fraction` | 有效运动 WALK 控制步中恰好一脚接触的比例 |
| `swing_foot_height_m` | 实际单支撑时，空中脚 link 原点世界 Z 的均值；不是脚底净空 |
| `hip_pitch_rms_rad` | 双髋俯仰相对默认角度的均方根；不能单独区分持续偏置与交替摆动 |

统计沿用现有控制步去重标记，包含终止帧，回合重置不清除尚未输出的窗口。
无有效 WALK 帧时不输出无定义均值；无单支撑帧时不输出脚高均值。
日志的有效运动阈值固定为 0.1 m/s / 0.2 rad/s，自定义奖励阈值不会改变该物理统计口径。

验收应比较相同指令和扰动下的速度、高度误差、跌倒率及真实摆腿过程。
较高奖励、单支撑比例或髋 RMS 均不足以证明合理步频，仍需回放或接触数据分析。
建议使用独立训练目录做旧奖励与新奖励对照；本次实现验证不代替完整训练实验。

## 验证

局部数值、模式隔离和统计检查：

```powershell
uv run python -m pytest tests/test_walk_rewards.py tests/test_mdp_training_logic.py tests/test_performance_metrics.py -q
```

真实 PhysX 计时、重置及公共入口短训/回放：

```powershell
$env:G1_WALK_REWARD_SMOKE='1'
$env:G1_TRAIN_PLAY_SMOKE='1'
uv run python -m pytest tests/test_walk_rewards_physics.py tests/test_task_runners_physics.py -k 'not upper' -q
```

2026-11-12 本机验证完成：328 项相关回归通过，2 项真实物理/公共入口测试通过。
物理检查覆盖 8 个环境的 48 个控制周期，并观察到有效参考奖励；短训完成模型保存、加载、3 步回放和导出。
AST 对比确认观测、动作、指令、事件、课程及原高度/竖直速度奖励配置保留；实际运行接口为 650/12。
验证记录位于 `outputs/verification/walk_rewards_20261112.json`，这些检查不代表新奖励已改善完整训练后的步态。
