# train/play 的任务选择与 Runner 接入

`g1.rl` 的 `train/play` 根据 `--task` 的注册信息自动识别任务族，选择任务自己的 Runner。环境创建、参数覆盖、随机种子、录像、模型导出和训练循环继续复用 IsaacLab/RSL-RL。

## 1. 使用方式

通常只需指定 `--task`：

| 参数 | 含义 |
| --- | --- |
| `--task-family` | 可选的任务族校验；填写后必须与 `--task` 的实际归属一致 |
| `--task` | 在 Gym 中注册的具体环境 ID，例如某个阶段 |

目前提供的任务族：

| 任务族 | Runner | 当前状态 |
| --- | --- | --- |
| `lower_limb_agent` | `tasks/lower_limb_agent/runner.py` 中的 `CurriculumOnPolicyRunner` | 已有 Phase1/2/3 环境及课程训练 |
| `upper_limb_agent` | `tasks/upper_limb_agent/runner.py` 中的 `UpperLimbOnPolicyRunner` | 固定根节点的 G1 双腕位姿跟踪 |

下肢训练、续训和回放的原有命令继续有效：

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-v0 --max_iterations 10000 physics=isaacsim_physx

uv run --extra isaacsim python -m g1.rl play --task Template-G1-v0 --checkpoint "<phase1_model.pt>" physics=isaacsim_physx
```

上肢任务族提供 `Template-G1-Upper-v0`，采样左手、右手或双手任务，跟踪世界固定的腕部目标；策略接收实时盆骨相对位姿：

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-v0 --viz none physics=isaacsim_physx

uv run --extra isaacsim python -m g1.rl play --task Template-G1-Upper-v0 --checkpoint "<g1_upper_model.pt>" --num_envs 16 --real-time physics=isaacsim_physx
```

识别依据是 Gym 注册的 `env_cfg_entry_point` 所属模块，例如 `g1.tasks.lower_limb_agent.config.g1.env_cfg:LowerLimbAgentEnvCfg` 对应 `lower_limb_agent`。识别过程只读取注册信息，不实例化环境或 agent 配置；创建环境后，再与配置中的 `task_family` 校验。

也可以显式传入 `--task-family lower_limb_agent` 做额外校验。该参数只校验归属；与自动识别结果不一致时直接报错。未注册的环境、归属不明的注册信息或尚未接入的任务族均报错，不回退到下肢任务。

当前 IsaacLab 的 train/play 需要无窗口运行时，使用 `--viz none`。

## 2. 当前各任务的行为

### 下肢任务

下肢 Runner 迁入下肢任务包，课程状态的保存、校验、恢复，以及训练前的阶段检查沿用原实现：

- 同阶段续训恢复实际课程分布和相应统计；Phase1→2、Phase2→3 继承实际分布、重建课程门控统计。
- 仍然从新回合开始，不接续原物理轨迹。
- 原有诊断指标、TensorBoard 分组和模式对比继续由下肢 Runner 安装。
- `--curriculum-snapshot` 仍是下肢旧检查点的显式恢复选项。
- 下肢 `play` 继续要求 `play_mode()`，因此拒绝 `--train_env_cfg`。

奖励、观测、动作、事件、运动轨迹、PPO 参数与课程推进公式不因本次分发改造而调整。环境配置新增的 `task_family` 仅用于检查 Runner 选择。

### 上肢任务

`UpperLimbOnPolicyRunner` 是上肢任务自己的扩展点，目前直接继承 RSL-RL `OnPolicyRunner` 的更新、保存、加载及日志行为：

- 不要求存在下肢的 `training_phase`、`task_command`、扰动事件或性能监控器。
- 使用标准训练日志；没有下肢的四模式面板与课程快照。
- 拒绝下肢专属的 `--curriculum-snapshot`；`play --train_env_cfg` 按上游正常语义处理。
- 后续上肢课程、检查点附加状态及特殊诊断在上肢 Runner 内实现。

框架不提供“任务缺少 Runner 时自动使用普通 Runner”的回退路径。每个任务族必须明确绑定自己的实现。

### G1 上肢位姿跟踪结构

```text
tasks/upper_limb_agent/
├── __init__.py
├── runner.py
├── mdp/
│   ├── __init__.py             # 懒加载本任务的 MDP 模块
│   ├── __init__.pyi            # 导出声明；新增算子时在此登记
│   ├── actions.py             # 腰部到双臂的显式关节顺序
│   ├── commands.py            # 双腕模式、世界目标、实时盆骨相对位姿与可视化
│   ├── observations.py        # 上肢状态与末端相对位姿
│   ├── rewards.py             # 位置、姿态跟踪与正则项
│   ├── events.py              # 重置事件与执行器目标同步
│   └── terminations.py        # 终止条件扩展点
└── config/
    ├── __init__.py
    └── g1/
        ├── __init__.py         # Gym 注册，使用配置类的字符串路径
        ├── env_cfg.py          # 场景、物理、动作、观测、奖励、重置、终止
        ├── agents/
        │   ├── __init__.py
        │   └── rsl_rl_ppo_cfg.py
        └── utils/
            ├── __init__.py
            └── assets.py       # 本任务独立维护的 G1 资产配置
```

资产最初复制自下肢任务的 `config/g1/utils/assets.py`，上肢任务在自己的文件中维护参数，当前 `fix_root_link=True`。
运行时环境和 MDP 均独立于下肢课程实现。

- **动作**：17 维，顺序为腰部 yaw/roll/pitch、左臂 7 轴、右臂 7 轴；每臂依次为肩 pitch/roll/yaw、肘、腕 roll/pitch/yaw。
  位置目标为资产默认位置加 `0.5 × 策略输出`。下肢和手指保持默认 PD 目标。
- **末端**：`left_wrist_yaw_link`、`right_wrist_yaw_link`，使用腕部 link 原点及其朝向；手掌、手指或工具偏移均不纳入末端定义。
- **参考系**：显式读取 `pelvis` link 的世界位姿。当前它也是固定的 root；腰部动作改变腕部位姿。
  目标保存在世界坐标系，每次读取观测时转换为实时盆骨相对位姿。
- **指令**：由一个 `wrist_pose`（`BimanualPoseCommand`）统一管理双腕，位姿顺序为左、右，各自使用 `[x, y, z, qx, qy, qz, qw]`。
  当前环境每回合以 `(0.33, 0.33, 0.34)` 的概率选择左手、右手或双手模式，回合内模式固定；双手目标共用刷新计时器。
  无效侧的目标观测为 `[0,0,0,0,0,0,1]`，配合有效标志表示闲置；该侧实际状态继续输入策略。
- **观测**：共 93 维，按顺序为双侧目标位姿 14、左右目标有效标志 2、双侧实际末端相对位姿 14、双侧位姿误差 12、17 轴相对默认位置、17 轴关节速度、17 轴上一帧动作。
  关节速度乘以 `0.1`，位置/速度顺序与动作一致；单帧观测，关闭观测噪声。
  `wrist_pose_error` 每腕提供 6 维：目标减实际的位置差（3 维，m）与当前旋转至目标所需的最短旋转向量（3 维，rad）。
  两者均沿实时盆骨坐标轴表达，顺序为左腕、右腕；闲置侧 6 维全部置零。
  误差的范数与奖励所用的位置/姿态误差一致，并保留策略调整所需的方向信息；目标与实际位姿继续保留。
- **时间**：物理步长 `1/120 s`，控制频率 60 Hz；每回合最长 10 秒。
- **重置**：保持固定根节点相对子环境的原始位置和朝向，无 root 位置/姿态随机化；关节恢复默认位置和速度，同时同步 PD 位置、速度与力矩目标。
  只写入正在重置的环境，其他环境继续执行。
- **终止**：仅按回合时限截断；原行走配置的高度、倾斜终止条件已移除。
- **训练**：默认 4096 个环境、每次采样 24 步、训练 1500 次迭代、每 100 次迭代保存检查点；网络为 `[256, 128, 128]`。
- **种子与日志**：默认随机抽取种子，`--seed` 可覆盖；标准 TensorBoard 日志位于 `logs/rsl_rl/g1_upper/<run>/`。
- **物理**：默认 `isaacsim_physx`，另提供 `ovphysx`、`physx`、`newton_mjwarp` 和 `newton_kamino` 预设。
  实际短训验证使用 Isaac Sim PhysX。

首次默认关节重置后，缓存各子环境的双腕默认 link 位姿。位置保存为 `默认腕部世界位置 - 子环境原点`，
后续采样按 `子环境原点 + 默认腕部局部位置 + 旋转后的位置偏移` 产生世界目标，因此支持不同原点及原点平移。
默认参考只捕获一次，后续腕部运动和重复重置不会改变采样中心。

偏移范围如下：位置偏移沿首次默认盆骨的坐标轴；姿态为默认腕部朝向右乘局部 Euler 偏移对应的四元数。
目标产生后，在下一次重采样前保持世界静止。观测实时转换参考系，奖励直接比较世界位姿，避免指令更新时序造成的缓存滞后。

| 维度 | 左腕 | 右腕 |
| --- | --- | --- |
| `left_offsets/right_offsets.pos_x` (m) | `[-0.05, 0.15]` | `[-0.05, 0.15]` |
| `left_offsets/right_offsets.pos_y` (m) | `[-0.05, 0.10]` | `[-0.10, 0.05]` |
| `left_offsets/right_offsets.pos_z` (m) | `[-0.05, 0.20]` | `[-0.05, 0.20]` |
| `roll/pitch/yaw` 偏移 (rad，各轴) | `[-0.3, 0.3]` | `[-0.3, 0.3]` |

当前默认腕部相对盆骨的位姿实测约为 `(0.20, ±0.15, 0.095) m`、单位朝向。
代码读取实际默认位姿作为基准，表中值均为附加偏移。当前位置分布接近原来的工作空间；修改资产默认关节角后，采样基准也随之更新。

每次目标刷新时，取有效目标到当前盆骨的最大距离 `d`，按下式确定这一次的保持时间：

```text
保持时间 = clamp(d / distance_reference_speed, resampling_time_range.min, resampling_time_range.max)
当前环境 distance_reference_speed = 0.08 m/s，时间范围 = [2, 4] s
```

例如距离 `0.32 m` 对应 `4 s`。该参数只控制多久重采样，与 PD 或关节运动速度无关。
课程 TODO 位于 `BimanualPoseCommandCfg`：后续可修改偏移范围、距离时间比例及时间上下限，新值在下一次采样生效；当前目标和剩余保持时间继续执行。
模式概率在下一回合采样时读取。课程推进条件尚未实现。

可以通过现有 train/play 的配置覆盖项指定模式和采样参数，例如：

```bash
# 只训练左手任务；比例顺序为左手、右手、双手
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-v0 --viz none physics=isaacsim_physx "env.commands.wrist_pose.mode_probabilities=[1,0,0]"

# 全部使用双手模式，调整目标距离对应的保持时间
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-v0 --viz none physics=isaacsim_physx "env.commands.wrist_pose.mode_probabilities=[0,0,1]" env.commands.wrist_pose.distance_reference_speed=0.1
```

当前默认奖励如下，权重由 RewardManager 在每步乘以控制步长：

| 奖励项 | 目标 | 权重 |
| --- | --- | --- |
| `left_position` / `right_position` | 有效侧 `2/n × (1 - tanh(位置误差 / 0.1 m))`，`n` 为有效手臂数 | 各 `1.0` |
| `left_orientation` / `right_orientation` | 有效侧 `2/n × [0.6 G(e_q, 0.3) + 0.4 G(e_q, 0.05)]`，`e_q` 为单位误差四元数的虚部范数 | 各 `+0.3` |
| `idle_arm_pose` | 闲置臂 7 个关节偏离默认角度的平方均值，单位 rad²；双手模式为 0 | `-1.0` |
| `workspace_quality` | 所有活动腕部位置达标且目标受地图覆盖时，取目标相对骨盆的归一化 Q 均值 | `0.0`，默认只记录诊断 |
| `action_rate` | 原始策略动作的一阶差分平方和，使用 `action_rate_l2` | `-0.03` |
| `target_speed` | 关节目标变化速度超过 2.5 rad/s 的归一化平方代价 | `0.0`，默认关闭 |
| `joint_vel` | 17 个受控关节的实际速度平方和，使用 `joint_vel_l2` | `-1e-4` |
| `joint_acc` | 17 个受控关节的加速度平方和，对全部物理子步取均值 | `-1e-7` |
| `end_effector_acc` | 双腕 link 原点在骨盆坐标系中的相对线加速度平方和，对全部物理子步取均值 | `-1e-5` |
| `joint_torques` | 抑制 17 个受控关节的力矩平方和 | `-2e-6` |
| `torque_limits` | 17 个受控关节超过各自物理力矩上限 80% 的力矩绝对值超出量之和 | `-1e-3` |
| `joint_limits` | 惩罚受控关节超出软位置限制 | `-1.0` |

空间 Q 使用左臂 0 kg、5 cm 网格，右臂镜像查询；默认位置阈值为 1 cm。
数据定义、五项诊断和启用方式见 [空间 Q 奖励说明](WORKSPACE_QUALITY.md)。

`action_rate` 惩罚 `sum((a_t - a_{t-1})²)`，不除以控制步长；当前动作只有 17 个受控关节，因此覆盖范围与动作一致。
`joint_vel` 读取仿真关节速度（rad/s），惩罚 `sum(q_dot²)`，覆盖腰部和双臂的 17 个受控关节。
`joint_acc` 读取仿真关节加速度（rad/s²），只选择腰部和双臂，不计入下肢与手指。
三项分别抑制策略输出突变、关节运动速度和实际关节运动抖动；正式训练时结合跟踪误差和奖励项量级判断效果。

`end_effector_acc` 惩罚双腕相对骨盆坐标系的位置二阶导数，加速度单位为 m/s²。
末端与跟踪目标一致，取 `left_wrist_yaw_link`、`right_wrist_yaw_link` 的坐标原点；单手模式也约束闲置侧。
仿真提供 link 质心加速度，先使用刚体公式换算到腕部和骨盆的 link 原点：

```text
r = p_link - p_com                      # 均为世界坐标
a_link = a_com + alpha × r + omega × (omega × r)
```

随后扣除骨盆平移、角加速度、向心与科里奥利贡献，并转到骨盆坐标系。
整体运动且双腕相对骨盆保持不变时，理论代价为零；腰部和手臂的实际相对运动仍计入。
每个子步先计算双腕平方和，再对全部物理子步取均值；此处评价线加速度，角加速度用于坐标转换。
完整公式、配置和验证方式见 [末端加速度说明](FROZEN_LOWER_STANDING.md#末端加速度骨盆坐标系与完整子步)。
原 TensorBoard 标签保持不变，新旧参考系与采样口径的绝对值应分别解读。

`torque_limits` 使用 `sum(max(abs(tau) - soft_ratio × tau_max, 0))`，默认 `soft_ratio=0.8`。
`tau_max` 来自运行时 `robot.data.joint_effort_limits`，随各环境、各关节的物理上限变化；惩罚只作用于超出软阈值的部分，物理上限保持执行器配置值。
当前手臂上限 25 N·m，对应软阈值 20 N·m；腰部 yaw 为 70.4 N·m，roll/pitch 为 40 N·m。
力矩读取 `actuators.applied_effort`：隐式 PD 的此数据为估算值，与求解器内部实际驱动力矩可能存在差异。
`joint_torques` 持续抑制力矩大小，`torque_limits` 额外约束靠近上限的区间。
新增项由标准日志记录为 `Episode_Reward/joint_vel`、`Episode_Reward/end_effector_acc`、`Episode_Reward/torque_limits`。

跟踪项对无效侧返回 0；单手模式乘以 2，双手模式乘以 1，保留各模式之间的归一化关系。
位置项沿用 `1 - tanh(e / 0.1 m)`。朝向项使用双尺度四元数指数奖励：

```text
q_error = normalize(q_goal ⊗ conjugate(q_current))  # 四元数顺序为 (x, y, z, w)
e_q² = ||q_error.xyz||² = sin²(theta / 2)
G(e_q, sigma) = exp(-e_q² / sigma²)
orientation = (1 - fine_weight) × G(e_q, sigma_coarse) + fine_weight × G(e_q, sigma_fine)
当前 sigma_coarse = 0.3，sigma_fine = 0.05，fine_weight = 0.4
```

`theta` 为最短旋转角；`e_q` 和 sigma 均无量纲，`e_q` 范围为 `[0, 1]`。
误差使用当前腕部与目标的世界四元数直接计算，对 `q` 与 `-q` 等价；计算虚部平方和可保留接近零时的精度。
在 `e_q = sigma` 时，单个核函数值为 `exp(-1) ≈ 0.368`。

若从角度核函数 `exp(-theta² / (2 sigma_angle²))` 换算，可用
`sigma_q = sqrt(2) × sin(sigma_angle / 2)` 保留 `theta = sigma_angle` 参考点处的核函数值。
当前配置使用 `0.3`、`0.05` 两档带宽；此前观测到的 `0.11 rad` 误差对应 `e_q ≈ 0.05497`，
混合朝向奖励约为 `0.700`（模式归一化与外层权重之前）。后续根据误差与运动平滑性调整带宽。

参数在 `env_cfg.py` 的 `left_orientation`、`right_orientation` 中显式配置：大 sigma 提供粗跟踪收益，
小 sigma 提供精跟踪收益，`fine_weight` 调整精细项占比；两项占比之和为 1。
sigma 是奖励带宽，成功阈值需独立定义。支持 `0 < sigma_fine <= sigma_coarse`，两者均为有限数，`fine_weight` 在 `[0, 1]` 内。
外层 `weight=0.1` 为正，完美对齐时单手模式下该手加权值为 `0.2`，双手模式下每手为 `0.1`；最终再乘控制步长。
`Episode_Reward/left_orientation` 和 `right_orientation` 记录正向奖励。
跨奖励版本比较优先查看原始 `orientation_error_rad`、位置误差和运动平滑指标。
观测中的朝向误差仍是 3 维旋转向量（rad），日志仍记录最短旋转角（rad）；此次调整保持动作与观测维度。

闲置臂惩罚只覆盖对应手臂，腰部为共享自由度。当前默认权重为 `-1.0`，结合姿态偏差与跟踪表现校准。

标准日志通过命令管理器输出 `Metrics/wrist_pose/<模式>/...`，模式名为 `left_only`、`right_only`、`both`：

- `left/right_position_error_m`：有效侧的位置误差，先求每回合采样步均值，再对本次重置的同模式回合取均值；越低越好。
- `left/right_orientation_error_rad`：对应的最短旋转角误差；越低越好。
- `idle_joint_rms_rad`：单手模式下闲置臂相对默认角度的 RMS，先对每回合时间和 7 个关节求均方根，再对回合取均值。
- `episodes`：本次重置中具有有效统计样本的对应模式回合数；RSL-RL 在迭代窗口内求均值，因此它是样本量参考，非累计回合数。

指标按指令更新时的非重置帧累计；自动重置发生在命令更新前，终止物理步不计入这些均值。
无有效样本时只输出回合数，不写入零误差。指标通过 `pose_error()` 使用未加权的原始误差（m/rad）；
朝向奖励通过 `orientation_error_quat_squared()` 读取同一组当前世界位姿与世界目标，计算无量纲的四元数误差。
`play` 显示双腕当前坐标架和有效侧的世界目标标记；左右使用独立路径。需要扩展环境生命周期时再增加 `env.py`。

新增双腕位姿误差后，当前观测为 93 维，原 79/81 维布局的检查点需要重新训练，并使用新检查点回放。
镜像数据增强和 mirror loss 的研究计划保留在上肢 PPO 配置的 TODO 中。

Upper 的自由底座版本 `Template-G1-Upper-Mobile-v0` 复用同一任务族和 Runner，使用21维动作：
前3维选择基座速度，第4维选择高度偏置，后17维控制腰臂关节。观测为107维，冻结 Lower 执行速度与高度命令。
Mobile 的 `action_rate` 限定为17维上肢动作，`lower_command_rate=-0.10` 保留H2的3维速度与1维高度的归一化命令变化约束。
高度没有幅值和越界惩罚，保持任意恒定高度时其变化代价为零。
`base_vertical_acc=-0.05` 对骨盆link原点的世界Z加速度施加4 m/s²软阈值惩罚，覆盖全部物理子步。
范围配置、零命令对照、接口兼容与物理验证见 [冻结下肢控制说明](FROZEN_LOWER_STANDING.md)。

## 3. 新任务接入步骤

1. 在 `src/g1/tasks/<family>/` 中定义环境与 agent 配置，并在 `config/<robot>/__init__.py` 注册。`env_cfg_entry_point` 的模块路径应属于 `g1.tasks.<family>.`。
2. 在环境配置中声明任务族；其值应与注册模块的所属任务族一致：

   ```python
   task_family: str = "upper_limb_agent"
   ```

3. 在任务包内实现独立 Runner 类及工厂。工厂使用统一签名：

   ```python
   def create_runner(env, agent_cfg, log_dir=None, *, snapshot_path=None, evaluation=False):
       ...
   ```

   `env` 已经过 RSL-RL 包装；`agent_cfg` 是上游完成参数更新和版本适配后的配置。`evaluation=True` 表示回放。任务负责把配置转换成自己的 Runner 参数，并决定需要哪些额外状态与日志。

4. 在 `src/g1/rl/task_registry.py` 的 `TASK_FAMILIES` 中登记工厂的字符串路径及任务专属选项。注册阶段不导入 Runner 或启动仿真。
5. 为新任务指定自己的 `experiment_name`，例如 `g1_upper`，使训练输出目录独立。

两个任务包不互相导入训练实现。公共分发层只检查任务身份、加载所选工厂并转交参数。新任务自行设计课程和训练阶段，无须套用下肢的三阶段规则。

当前公共入口仍限定 Torch 前端、单进程和 RSL-RL `OnPolicyRunner`；扩展任务不会同时扩展这些运行能力。

## 4. teleop/eval 边界

本阶段的 `teleop/eval` 仍只支持下肢 Phase1/2/3。显式选择上肢任务族，或将上肢环境 ID 传入这两个入口，都会得到明确错误。原有下肢遥操作、评估配置和报告口径保持原样。

eval 规划和报告不因任务注册而加载仿真。现有 eval 源码校验仍覆盖整个 `src/g1`；本次代码更新后的实际评估需新建批次，原结果可继续生成报告。按任务划分源码摘要留待 eval 扩展时处理。

## 5. 验证重点

- 入口转发原始环境参数、种子与 Hydra 覆盖项，并在成功或异常时恢复上游工厂引用和 `sys.argv`。
- 下肢课程保存、同阶段恢复、相邻阶段转换、日志与退出清理继续通过原有回归测试。
- 上肢使用无下肢字段的数值环境完成真实 CPU PPO 更新、标准日志输出、保存和加载回放，并与同种子的上游 Runner 比较模型参数。
- 公共分发层不导入任务；加载上肢 Runner 不加载下肢环境、课程与诊断模块。

数值环境测试验证入口和 Runner 合同；上肢任务另有独立资产、盆骨坐标变换、四元数符号等价、局部重置及真实物理环境的训练、回放测试。
位姿跟踪的收敛效果通过正式训练评估。

可通过设置 `G1_TRAIN_PLAY_SMOKE=1` 后运行 `tests/test_task_runners_physics.py`，分别验证下肢与上肢任务的真实 PhysX 流程：4个环境短训2次迭代、加载回放3步及策略导出。
上肢测试另检查 17 维动作、93 维观测、位姿误差方向与奖励误差范数的一致性、默认双腕采样基准、双腕/盆骨 link、三种模式的可视化数据及固定根节点位置，
并缩短回合与目标保持时间，覆盖重复重置和目标刷新。
该测试默认关闭，输出写入 pytest 的临时目录；使用 `-k upper_g1` 可只运行上肢任务。
