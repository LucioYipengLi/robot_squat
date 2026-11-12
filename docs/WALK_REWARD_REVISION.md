# Lower Limb Agent：WALK 奖励整改记录

日期：2026-11-12

本文记录上一轮已经落地的最终实现：保持现有观测与动作接口，删除原 WALK 协同项和足端高度项，新增由实际离地事件驱动的关节参考奖励及支撑时长惩罚。本文中的参数是当前代码配置；步态收益尚未通过新训练或仿真回放验证。

## 1. 目标与约束

整改目标是让 WALK 模式形成更明显的髋部前摆、屈膝和真实离地，改善小幅挪动的行走观感。

设计约束：

- 不新增观测项，保持每帧 130 维、五帧堆叠共 **650 维**。
- 保持 12 维下肢关节位置动作、动作比例及执行器参数。
- 采用局部、轻量实现，不新增状态机、独立相位管理器或奖励自有计时缓存。
- 新增步态引导仅用于 WALK；其他模式保留原奖励配置。
- 初始关节参考幅度和奖励权重适度激进，先观察是否产生可测量的行为变化。

## 2. 不新增相位观测后的设计调整

此前讨论过向策略提供 `sin(phase)、cos(phase)`，并按固定 episode 时钟生成轨迹。最终实现没有采用这套观测扩展。

关节角度、速度和历史可以反映实际腿部运动，但不能唯一确定独立的 episode 时钟相位。缺少相位观测会增加固定时钟轨迹的对齐难度，因此 WALK 改为：

1. 读取现有接触传感器的实际离地时长 `current_air_time`。
2. 用离地时长生成一次摆腿的半正弦参考，落地后由传感器自身的计时逻辑重新开始。
3. 用非单支撑及摆动、支撑超时惩罚，鼓励持续迈步与交替支撑。

`swing_time=0.4 s` 是摆腿参考时长，**不表示强制执行固定 0.8 s 的完整步态周期**。没有规定哪只脚必须先迈，也没有要求 WALK 与 episode 时钟锁相。

这些传感器计时只用于训练奖励，不进入策略观测。策略仍需从现有身体状态与历史间接判断运动进程；本次修改减轻了隐藏绝对时钟的问题，不构成学习效果的保证。

## 3. 按强化学习组件整理改动

| 组件 | 本轮处理 |
| --- | --- |
| 观测 | 无新增、删除或重排；保持 650 维 |
| 动作与控制 | 保持 12 维动作、动作比例及 PD 参数 |
| 奖励 | 删除 2 项、新增 2 项、调整 3 项，详见下表 |
| 指令与采样 | 保持速度范围、四模式定义、模式采样比例及指令采样逻辑 |
| 终止与成功判据 | 保持原样，包括现有 3 cm 高度成功门槛 |
| 课程学习 | 保持指令、上肢扰动和 Phase3 推扰课程 |
| PPO 与网络 | 保持网络结构和训练超参数 |
| 日志 | 新增 3 个 WALK 物理指标 |
| 三阶段配置 | Phase2/3 通过现有继承关系获得奖励调整，没有复制实现 |

### 3.1 奖励增减清单

| 操作 | 配置项 | 修改前 | 修改后 |
| --- | --- | --- | --- |
| 删除 | `leg_synergy_walk` | 权重 `+0.4`，约束关节靠近下蹲协同平面 | 删除该 WALK 配置项；STAND/SQUAT 的 `leg_synergy` 保留 |
| 删除 | `feet_swing_height` | 权重 `+0.8`，高度曲线峰值 `0.15 m`、`std=0.04` | 删除配置项、实现和导出声明 |
| 新增 | `walk_joint_reference` | 无 | 权重 **`+2.0`**，实际离地时长驱动髋膝踝参考 |
| 新增 | `walk_support_timing` | 无 | 权重 **`-0.75`**，惩罚非单支撑和持续时间超限 |
| 调整 | `feet_gait_phase` | 用于 WALK、SQUAT_WALK | `walk_modes=(3,)`，仅保留 SQUAT_WALK，原权重 `+0.5` 和频率参数保留 |
| 调整 | `track_pelvis_height` | 所有模式有效权重 `+2.0` | WALK 乘 `walk_scale=0.5`，有效权重 **`+1.0`**；其他模式仍为 `+2.0` |
| 调整 | `vertical_bounce` | 所有模式有效权重 `-0.5` | 改用 `lin_vel_z_l2_walk_scaled`，WALK 乘 `0.2`，有效权重 **`-0.1`**；其他模式仍为 `-0.5` |

这里的“有效权重”指配置权重与函数内部模式缩放的乘积。实际单步奖励还经过奖励管理器的时间步长缩放。

速度跟踪、防滑、禁止腾空、躯干姿态、足膝间距、动作一阶与二阶平滑、关节限位、力矩和终止惩罚等配置保留原样。

### 3.2 模式与运动门控

两个新增奖励均要求：

```text
mode == WALK
并且满足以下任一条件：
    sqrt(vx_cmd² + vy_cmd²) > 0.1 m/s
    abs(wz_cmd) > 0.2 rad/s
```

足部接触判定使用当前世界坐标系竖直接触力 `> 0.5 N`。关节及足部索引显式保持左右顺序。

骨盆高度与竖直速度的缩放按 WALK 模式生效，包含近零速度指令；它们不使用上述运动阈值门控。

## 4. 新增奖励的具体行为

### 4.1 `walk_joint_reference`

参数：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `weight` | `+2.0` | 参考姿态奖励权重 |
| `swing_time` | `0.4 s` | 从离地开始的半正弦参考时长 |
| `hip_amplitude` | `0.30 rad`，约 `17.2°` | 髋俯仰基准参考幅度 |
| `knee_amplitude` | `0.50 rad`，约 `28.6°` | 摆动腿膝屈曲基准增量 |
| `ankle_amplitude` | `0.12 rad`，约 `6.9°` | 踝俯仰基准参考幅度 |
| `sigma` | `0.20 rad` | 六关节均方误差指数核的宽度 |
| `speed_reference` | `0.6 m/s` | 达到完整参考幅度的等效运动指令量 |
| `command_threshold` | `0.1` | 平面运动阈值；转向阈值为其两倍 |

参考相对默认关节姿态生成，按每腿 `hip_pitch / knee / ankle_pitch` 排列：

```text
lift_i = max(sin(pi * clip(air_time_i / swing_time, 0, 1)), 0)
         * 当前脚未接触
signal_i = lift_i - lift_另一只脚

motion = norm([vx_cmd, vy_cmd]) + 0.3 * abs(wz_cmd)
scale = clip(motion / speed_reference, 0.5, 1.0)
direction = clip(vx_cmd / 0.2, -1, 1)  # 当前 command_threshold=0.1

hip_ref_i   = hip_default_i   - 0.30 * scale * direction * signal_i
knee_ref_i  = knee_default_i  + 0.50 * scale * lift_i
ankle_ref_i = ankle_default_i - 0.12 * scale * direction * signal_i
```

行为含义：

- 前进时，摆动腿髋前摆、膝屈曲，支撑腿髋踝做相反方向的参考运动。
- 倒退时反转髋踝参考方向，保留摆动腿屈膝。
- 纯侧移、原地转向时不强制前进方向的髋踝俯仰偏移，仍保留屈膝参考。
- 幅度随运动指令缩放，比例限制在 `0.5～1.0`。
- 只在实际单支撑时给分，并乘以 `max(lift_i)`，突出摆动中段。
- 双脚着地、双脚腾空和超过参考时长的持续悬空，不能获得该项参考奖励。

原始奖励为：

```text
exp(-mean((q - q_ref)²) / sigma²)
* max(lift_i)
* 单支撑门控
* WALK 运动门控
```

原始值范围为 `[0, 1]`；乘 `+2.0` 后，时间步长缩放前为 `[0, 2]`。参考角度是软奖励目标，不是强制动作或实际摆幅承诺。

### 4.2 `walk_support_timing`

该项替代了此前计划中的固定时钟接触时序惩罚，最终配置名为 `walk_support_timing`。

参数：

| 参数 | 当前值 |
| --- | --- |
| `weight` | `-0.75` |
| `swing_time` | `0.4 s` |
| `max_stance_time` | `0.6 s` |
| `command_threshold` | `0.1` |

原始代价由两部分组成：

```text
non_single = 1（接触脚数不等于 1），否则为 0

air_overdue_i = clip(air_time_i / 0.4 - 1, 0, 1)
stance_overdue_i = clip(contact_time_i / 0.6 - 1, 0, 1)
overdue = mean_i(max(air_overdue_i, stance_overdue_i))

cost = (non_single + overdue) * WALK 运动门控
```

原始代价范围 `[0, 2]`；乘权重后，时间步长缩放前的惩罚范围为 `[-1.5, 0]`。

短暂双支撑会产生瞬时代价；超时部分随实际持续时间增加并封顶。支撑脚长时间不交换也会累积代价，用于减少同一只脚反复摆动、另一只脚长期支撑的退化行为。该项不预设左右顺序，也不保证训练后必然形成理想交替步态。

两个奖励的 `swing_time` 均来自配置中的 `WALK_SWING_TIME = 0.4`，避免分别维护参考时长。

## 5. 新增物理日志指标

训练日志中的完整标记如下：

| TensorBoard 标记 | 定义 |
| --- | --- |
| `Performance/task_command/walk/single_stance_fraction` | 有效运动 WALK 控制步中，恰好一只脚接触的比例 |
| `Performance/task_command/walk/swing_foot_height_m` | 实际单支撑时，空中脚 link 原点的世界 Z 均值 |
| `Performance/task_command/walk/hip_pitch_rms_rad` | 有效运动 WALK 控制步中，双髋俯仰相对默认姿态的均方根 |

统计口径：

- 复用现有逐控制步去重标记，避免 reset、command 等钩子重复统计同一帧。
- 按日志窗口内有效样本累计；不是按 reset 批次大小平均。
- 没有有效 WALK 样本时，不输出无定义的均值；没有实际单支撑样本时，不输出摆动脚高度均值。
- 脚高是 link 世界高度，不是脚底净空或地形相对高度。
- 髋 RMS 不是峰峰值，也不能单独区分交替摆动和持续姿态偏置，需结合回放判断。
- 日志只记录物理状态，不新增策略输入，也不改变课程成功判据。

## 6. 修改文件

| 文件 | 修改内容 |
| --- | --- |
| [env_cfg.py](../src/g1/tasks/lower_limb_agent/config/g1/env_cfg.py) | 删除旧 WALK 项，配置新奖励、共享参考时长、模式缩放及 SQUAT_WALK 门控 |
| [mdp/rewards.py](../src/g1/tasks/lower_limb_agent/mdp/rewards.py) | 删除高度奖励实现；新增关节参考、支撑时长惩罚、WALK 竖直速度缩放；扩展高度奖励模式缩放 |
| [mdp/__init__.pyi](../src/g1/tasks/lower_limb_agent/mdp/__init__.pyi) | 更新奖励导出声明 |
| [mdp/commands/commands.py](../src/g1/tasks/lower_limb_agent/mdp/commands/commands.py) | 缓存日志所需足部、髋关节索引，并在现有有效控制步入口采样 |
| [performance.py](../src/g1/tasks/lower_limb_agent/performance.py) | 累计和输出三个 WALK 物理指标 |
| [tests/test_walk_rewards.py](../tests/test_walk_rewards.py) | 替代旧高度奖励测试，验证新奖励与统计口径 |
| [tests/test_mdp_training_logic.py](../tests/test_mdp_training_logic.py) | 更新测试场景，并验证新增指标沿用逐步去重逻辑 |

本轮没有修改 `mdp/observations.py`、`env_cfg_phase.py` 或 `uv.lock`；工作区中原有的其他改动不归入这次整改。

## 7. 验证结果与复现命令

上一轮已执行相关测试：**255 项通过，1 项跳过**。跳过项依赖仓库未提供的历史本地训练文件。最后针对训练逻辑的 24 项测试也单独复测通过。

覆盖内容包括：

- 前进、倒退、侧移和转向的关节参考。
- 左右腿对称、实际离地计时、双支撑与腾空门控。
- 长时间悬空、长时间固定支撑的奖励与代价。
- STAND/SQUAT/SQUAT_WALK 的奖励隔离及三阶段继承。
- 日志分母、窗口清空、终止帧及重复采样处理。
- 既有课程、checkpoint、冻结下肢策略及评估配置回归。

在项目根目录执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_walk_rewards.py \
  tests/test_mdp_training_logic.py \
  tests/test_performance_metrics.py \
  tests/test_diagnostics_logging.py \
  tests/test_push_curriculum.py \
  tests/test_curriculum_checkpoints.py \
  tests/test_frozen_lower_policy.py \
  tests/test_registration.py \
  tests/test_eval_config.py \
  tests/test_eval_scenarios.py \
  tests/test_eval_policy_semantics.py \
  -q --basetemp=/tmp/g1-walk-pytest
```

禁用 pytest 自动插件加载，是为了隔离本地环境中无关 ROS 插件的依赖问题。测试临时输出位于 `/tmp`。

已确认观测配置的 AST 与修改前一致，`git diff --check` 通过。本轮没有启动训练或仿真回放，也没有修改历史训练日志。单元测试验证的是实现逻辑，不能证明步态已改善。

## 8. Checkpoint 兼容性与后续验收

观测仍为 650 维，动作仍为 12 维，网络结构未变，因此旧 checkpoint 在输入输出维度上保持兼容。奖励目标已经改变，结构兼容不代表旧策略已经具备新步态，也不代表续训后的收益得到保证。

建议使用新的训练目录记录本次奖励版本，并在相同指令及上肢条件下比较前进、倒退、侧移、原地转向和近零指令表现。验收重点是：

1. 髋部确实形成更明显的交替摆动，脚也实际离地。
2. 速度误差、骨盆高度误差和跌倒情况没有明显恶化。
3. 没有通过持续蹲低、单脚悬空或跳跃换取更高收益。
4. STAND、SQUAT、SQUAT_WALK 的表现通过回归检查。

以物理指标与回放作为主要证据，奖励分数作为辅助证据。
