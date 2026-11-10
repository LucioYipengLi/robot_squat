# 上肢空间 Q 先验奖励

这项奖励用于：先完成腕部位置跟踪，再鼓励世界目标落在相对骨盆更有利的空间位置。
固定底座与冻结下肢任务共用实现，默认权重为 **0.0**，先记录诊断指标。
Mobile 已开放 `vx、vy、yaw rate、height_offset` 策略动作，可主动调整基座；配置见 [冻结下肢控制说明](FROZEN_LOWER_STANDING.md)。
Q 的实际学习收益通过后续同条件训练对照评估。

## 1. 数据与分数含义

初版读取已计算的左臂、0 kg 负载先验，源文件为：

```text
logs/analysis/left_5cm.batch.metrics_0.4ef.yaw-cropped.npz
```

源数据包含 66,044 个已保存构型，对应 2,263 个有样本的空间点，网格间距为 5 cm。
Q 使用源文件中已有的综合指标，包含运动学、关节余量和动力学分量，原权重为 `0.3 / 0.3 / 0.4`。
具体原始定义随导出资源保存在 `source_composite_metrics` 中。

每个空间点取其所有已保存构型的 **0 kg Q 最大值**，保留原数据的评分口径，不额外按可行性筛选构型。
因此，该值表示“这个位置在已采样构型中能达到的最好评分”，当前实际关节构型的性能需要另外判断。
位置表没有朝向维度，Q 奖励也只处理平移；原有朝向跟踪奖励继续独立生效。

固定归一化方式为：

```text
Q_ref = max(源文件全部已保存构型的 0 kg Q) = 0.7218234970798301
score(p) = clip(Q_interpolated(p) / Q_ref, 0, 1)
```

Q_ref 来自导出时的数据，此后保持固定。各次训练使用同一资源时，可以直接比较分数。
资源版本变更后，应同时核对源数据哈希和 Q_ref。

运行时使用三线性插值。所有有贡献的邻接节点都必须有样本；缺失节点和网格外位置标记为未知。
已知节点本身可查询，权重为零的相邻缺失节点不影响结果。
内部使用很小的数值容差恢复 float32 网格节点，此容差与位置达标的 1 cm 阈值相互独立。
未知位置的奖励贡献为零，诊断统计通过有效性掩码将其与“已知但评分低”区分开。

右臂利用位置对称性查询左臂表：`(x, y, z) → (x, -y, z)`。
这一假设用于初版引导，实际左右臂性能是否一致，仍需通过实验检查。

## 2. 奖励计算

每个控制步都从当前物理状态读取腕部和骨盆位姿，使用尚未刷新的世界目标计算：

```text
p_goal_b = R_pelvis_w^T × (p_goal_w - p_pelvis_w)
e_hand = ||p_actual_w - p_goal_w||₂

ready = 所有活动手的 e_hand <= position_tolerance_m
covered = 所有活动手的 p_goal_b 都能有效查询 Q 表
reward_raw = ready × covered × mean(score(p_goal_b)，仅活动手)
reward_step = weight × reward_raw × control_dt
```

默认 `position_tolerance_m = 0.01`，单位为米。单手模式只检查活动手；闲置手继续受原有默认关节角惩罚约束。
双手模式要求两只手同时满足位置阈值和地图覆盖，原始奖励范围为 `[0, 1]`，各模式使用相同最大值。
朝向误差不参与这项门控。

奖励查询的是**目标相对骨盆的位置**。实际腕部的位置只用于位置达标判断，及单独的实际位置分数诊断。
例如，世界目标保持静止，骨盆移动后 `p_goal_b` 改变；只要腕部继续跟上目标，就能获得新的空间评分。
固定底座下，同一目标的 Q 基本固定，此时主要验证查询和门控；Mobile 可通过下肢速度和高度命令改变目标相对骨盆的位置。

门控使用硬阈值，误差在 1 cm 附近波动时奖励会切换。初版保持这一清晰定义，训练时结合达标比例判断信号是否足够。
RewardManager 统一施加权重和控制步长，函数本身只返回未加权分数。

## 3. 配置和启动

常用配置在 `src/g1/tasks/upper_limb_agent/config/g1/env_cfg.py` 中显式给出：

| 配置路径 | 默认值 | 作用 |
| --- | --- | --- |
| `env.commands.wrist_pose.workspace_quality.map_path` | `data/workspace_q0_v1.npz` | 先验表路径 |
| `env.commands.wrist_pose.workspace_quality.position_tolerance_m` | `0.01` | 所有活动手的位置达标阈值，m |
| `env.rewards.workspace_quality.weight` | `0.0` | Q 奖励权重；为零时仍记录诊断 |

相对资源路径以 `upper_limb_agent` 包目录为基准；也可以传入绝对路径。
运行时一次加载到环境设备，后续查询使用 Torch 张量，不读取原始大文件，也不调用 IK 或外部分析库。

先按现有命令训练即可记录 Q 指标。需要试验非零奖励时，例如：

```powershell
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --num_envs 512 --viz none physics=isaacsim_physx env.rewards.workspace_quality.weight=0.02
```

`0.02` 是低权重试验示例，实际效果需通过跟踪误差、达标比例及 Q 分数对照确认。
使用 `Template-G1-Upper-v0` 可以做固定底座对照（17动作/93观测）；Mobile 使用21动作/107观测。
Q 对照实验的两组应使用相同 Mobile 接口与命令变化惩罚，再改变 Q 权重。
续训仍使用项目标准 `--checkpoint` 和 `--max_iterations` 参数。

若要连诊断和资源加载一并关闭，同时传入：

```text
env.commands.wrist_pose.workspace_quality=null env.rewards.workspace_quality.weight=0.0
```

## 4. TensorBoard 指标与判断

以下五项记录在 `Metrics/wrist_pose/workspace_quality/` 下，范围均为 `[0, 1]`：

| 指标 | 定义 | 如何判断 |
| --- | --- | --- |
| `target_score` | 有效覆盖的活动手目标位置评分均值 | 描述目标相对骨盆的空间质量，升高表示落在表中更高分的位置 |
| `actual_score` | 有效覆盖的活动手实际腕部位置评分均值 | 与目标分数对照；跟踪准确时通常接近，偏离时先看位置误差 |
| `position_ready_fraction` | 所有活动手同时达到位置阈值的帧比例，与地图覆盖无关 | 衡量 Q 奖励有多少机会触发；偏低时先处理位置跟踪 |
| `map_coverage_fraction` | 所有活动手目标位置均可有效查询的帧比例，与位置误差无关 | 偏低时检查采样范围、骨盆位置和稀疏地图边界 |
| `reward_available_fraction` | 所有活动手位置达标且目标受地图覆盖的帧比例 | 直接衡量 Q 奖励可触发的时间比例；与奖励权重无关 |

`Episode_Reward/workspace_quality` 由标准奖励管理器记录，是加权、按回合时长归一化后的奖励贡献。
默认权重为零时该曲线为零，以上五项仍正常更新。

五项诊断先在各回合内汇总，再对本次结束的回合取均值。
评分只统计有覆盖的活动手样本；一个回合若没有有效评分样本，会从该项均值中排除，避免写入虚假的零分。
达标比例和覆盖比例以有效统计帧数为分母。
沿用现有命令统计时序：重置帧和终止物理步不计入这些均值；奖励管理器仍计算终止步奖励。

建议按以下顺序查看：

1. 先看原有位置误差与 `position_ready_fraction`，确认策略能完成任务。
2. 再看 `map_coverage_fraction` 和 `reward_available_fraction`，确认地图覆盖及实际联合门控情况。联合比例直接统计两个条件同时成立的帧。
3. 最后比较 `target_score` 和 Q 奖励贡献，结合相同目标分布下的跟踪误差判断是否获得收益。

评分均值只描述覆盖部分。目标分数上升同时覆盖下降时，应结合覆盖率和采样条件解释。
本次实现只建立引导与记录链路，实际改善幅度需要后续对照训练测量。

## 5. 数据导出与运行归档

项目已附带约 11 KB 的压缩表和 JSON 元信息，安装后的训练机器直接使用即可。
需要重新生成时，在项目根目录运行：

```powershell
uv run python scripts/prepare_workspace_quality.py --output outputs/workspace_quality/workspace_q0_v1.npz
```

脚本默认读取上述源文件。也可通过第一个位置参数指定同格式 NPZ；覆盖已有导出需显式添加 `--overwrite`。
导出脚本检查数据完成状态、网格和样本映射，并保护源文件及其 JSON 旁文件。
导出结果包含 `quality`、独立 `valid` 掩码和内嵌元信息，不依赖旁文件才能加载。

Runner 保存模型时将实际加载的资源归档为：

```text
<run>/artifacts/workspace_quality.npz
<run>/artifacts/workspace_quality.json
```

checkpoint 的 `infos.g1_workspace_quality` 记录资源 SHA-256、源文件 SHA-256、Q_ref、负载、坐标系、镜像轴、位置阈值和归档相对路径。
`params/env.yaml` 另记录当前路径、阈值和奖励权重。同一运行目录保存时会核对已归档 NPZ，防止混入不同地图。

旧 Upper checkpoint 可按原接口加载；加载过程按当前配置选表，不自动切换到旧归档，也不强制校验历史 Q 元信息。
复现实验时请核对上述记录，并显式将 `map_path` 指向对应的归档绝对路径。
以后机器人结构或先验定义改变时，应生成新版本资源并保留旧资源，便于实验对照。

## 6. 实现与验证

| 文件 | 职责 |
| --- | --- |
| `scripts/prepare_workspace_quality.py` | 从原始构型评分导出紧凑的空间表 |
| `mdp/workspace_quality.py` | 加载、校验、插值、镜像和位置门控 |
| `mdp/commands.py` | 提供实时目标/实际位姿，累计五项回合诊断 |
| `mdp/rewards.py` | `workspace_quality_bonus` 读取未加权奖励 |
| `config/g1/env_cfg.py` | 暴露资源、阈值、权重配置；Mobile 继承 |
| `runner.py` | 归档资源并写入 checkpoint 元信息 |

表格中 `mdp/`、`config/`、`runner.py` 均位于 `src/g1/tasks/upper_limb_agent/`。
Q 奖励保持只读查询；Mobile 的速度接口与独立命令变化代价按冻结下肢控制说明执行，PPO 更新沿用标准实现。

数值与生命周期测试：

```powershell
uv run python -m pytest tests/test_workspace_quality.py tests/test_upper_limb_task.py tests/test_task_runners.py tests/test_upper_substep_actions.py tests/test_upper_kinematics.py -q
```

真实 PhysX、原有子步奖励回归、短程 train/save/play 测试，按顺序执行：

```powershell
$env:G1_WORKSPACE_Q_SMOKE='1'
$env:G1_UPPER_SUBSTEP_SMOKE='1'
$env:G1_TRAIN_PLAY_SMOKE='1'
uv run python -m pytest tests/test_workspace_quality_physics.py tests/test_upper_substep_physics.py tests/test_task_runners_physics.py -k 'not lower_limb' -q
```

本次验证覆盖插值与缺失边界、左右镜像、三种活动模式、当前位姿读取、零/非零权重、局部重置、资源归档、冻结下肢权重及推理次数、TensorBoard 记录和策略导出。
