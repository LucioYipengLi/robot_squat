# Upper：常规目标与高度随机化目标

`Template-G1-Upper-Mobile-v0` 默认在每回合开始选择目标类别：70%常规目标、30%高度随机化目标。
[平面迁移目标与计时](UPPER_PLANAR_TARGETS.md)目前显式启用；开启后高度概率作用于非平面回合，
例如平面概率20%、高度概率30%对应56%常规、24%高度、20%平面。
左手、右手、双手模式仍按原概率独立选择；Upper继续自主输出速度、高度和腰臂动作。
固定基座 `Template-G1-Upper-v0` 使用相同的单手／双手几何采样，不添加额外高度任务。
Mobile现已接入 [统一范围课程](UPPER_RANGE_CURRICULUM.md)，本文范围表是最终等级配置。
初始常规位置／朝向偏移为表中范围的50%，额外高度偏移为[−0.05,−0.02] m，随等级扩展。
下文新23维检查点指schema 4映射；A1的schema 3检查点需在匹配旧实现中回放。

## 生成与保持方式

常规单手目标以默认腕部位姿为基准；双手目标由默认双腕中点、受限三维间距和相近朝向生成，
详细参数见[双手几何采样](TASK_RUNNERS.md#g1-上肢位姿跟踪结构)。保持时间按目标到骨盆的距离计算，默认2～4秒。
第二类先生成同样的目标，再给所有活动手叠加同一个世界Z偏移：

```text
goal_position_world = regular_goal_position_world + [0, 0, height_shift]
height_shift ~ Uniform(-0.15, -0.05) m
```

位置偏移叠加在世界坐标中，朝向使用原采样结果。单手模式下闲置侧仍使用中性目标观测，
不参与跟踪奖励，并继续受到默认关节姿态约束。
双手共享额外高度偏移，几何采样产生的高度差、三维间距和相对旋转保持原样。

第二类的完整目标位姿默认保持到回合结束（当前回合上限10秒），回合内不会刷新位置或朝向。
生成后的世界目标保持固定；机器人运动时，策略看到的实时骨盆相对目标随之变化。
每次局部重置只重新选择对应子环境的类别、高度偏移、手臂模式和目标。

“第二类”标记的是采样方式。目标可能由腰臂单独完成，也可能受益于高度或其他机身调整，
还可能超出当前策略的完成能力。采样没有IK筛选、空间Q筛选或按Lower能力裁剪。
高度偏移仅用于生成目标和诊断，不作为独立观测，也不会直接下发成Lower高度命令。

## 主要参数

配置位于 `src/g1/tasks/upper_limb_agent/config/g1/mobile_env_cfg.py` 的 `MobileCommandsCfg`：

| 配置（相对 `env.commands.wrist_pose`） | 默认值 | 含义 |
| --- | --- | --- |
| `height_task.probability` | `0.30` | 每个新回合成为第二类的概率；0表示仅常规，1表示全部第二类 |
| `height_task.shift_range` | `[-0.15, -0.05]` | 在原目标上额外叠加的世界Z偏移，m；正偏移也可使用 |
| `height_task.hold_episode` | `true` | 第二类完整目标保持一回合；false则沿用普通目标的距离保持时间 |
| `resampling_time_range` | `[2.0, 4.0]` | 常规目标保持时间上下限；关闭第二类整回合保持后也作用于第二类 |
| `mode_probabilities` | `[0.33, 0.33, 0.34]` | 左手、右手、双手的回合概率 |

类别与额外高度偏移只在重置时抽样。即使 `hold_episode=false`，额外偏移也会在该回合内保持不变，
定时更新的只是原目标部分。修改类别概率或高度偏移范围，会在下次重置时生效。
运行时修改保持方式需等对应环境重置后使用新的计时设置。

目标高度随机范围和 `env.actions.height_offset.height_offset_range=[-0.40,0.05]` 相互独立：
前者决定腕部任务，后者限制Upper能请求的根高度。
当前默认根高度为0.75 m，最终Lower高度目标范围为0.35～0.80 m；低高度终止阈值为0.25 m。

最终等级单手原目标Z偏移为 `[-0.05,0.20] m`，第二类总Z偏移范围为 `[-0.20,0.15] m`。
双手公共中心使用相同Z偏移范围，另有左右各一半的高度差（最大±0.025 m），
因此每腕相对默认双腕中点的第二类总Z偏移范围为 `[-0.225,0.175] m`。
这些是多个采样量相加的分布，最终范围内的密度并不均匀。
30%是新回合的采样概率；不同类别回合长度不同时，正在运行的环境比例可能不同。

命令行覆盖示例：

```text
env.commands.wrist_pose.height_task.probability=0.0
env.commands.wrist_pose.height_task.probability=1.0
env.commands.wrist_pose.height_task.shift_range=[-0.20,-0.10]
env.commands.wrist_pose.height_task.hold_episode=false
```

`probability=0` 保留分类诊断，目标、模式和计时采样的随机数顺序与原采样一致。
设为 `env.commands.wrist_pose.height_task=null` 则完全关闭此项及分类诊断。

## 同起点续训对照

下面两个分支从同一个新23维Upper检查点出发，使用相同预算，仅改变第二类比例。
比例变化会改变课程契约，因此显式重建课程统计；两组均从等级0重新积累。

```powershell
$heightTaskCheckpoint = "<23维Upper检查点>"

uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --checkpoint $heightTaskCheckpoint --num_envs 4096 --seed 42 --max_iterations 300 --viz none physics=isaacsim_physx agent.run_name=regular_only env.commands.wrist_pose.height_task.probability=0.0 env.commands.wrist_pose.range_curriculum.reset_on_load=true

uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --checkpoint $heightTaskCheckpoint --num_envs 4096 --seed 42 --max_iterations 300 --viz none physics=isaacsim_physx agent.run_name=height_targets env.commands.wrist_pose.height_task.probability=0.3 env.commands.wrist_pose.range_curriculum.reset_on_load=true
```

这组对照检验目标混合分布对课程和学习的影响。纯目标分布比较可另将两组均固定到相同等级并关闭推进。
H5C 及更早的 21／107 检查点需在对应的历史代码下回放或显式迁移。

play中仅查看第二类目标，可以添加：

```text
env.commands.wrist_pose.height_task.probability=1.0
env.commands.wrist_pose.range_curriculum.reset_on_load=true
```

同一检查点还可添加 `env.actions.height_offset.enabled=false`，作为固定默认站高命令的对照。
该开关固定的是高度命令，冻结Lower仍通过物理反馈保持平衡，实际根高度会随运动变化。
两个评估条件应使用相同目标、手臂模式及种子，并观察实际高度和追踪表现。

## TensorBoard分类指标

标签位于 `Metrics/wrist_pose/task/regular/` 和 `Metrics/wrist_pose/task/height_shifted/`。
本文介绍常规／高度两类；启用第三类时另外输出`task/planar/`，见[窗口评分说明](UPPER_PLANAR_TARGETS.md#窗口评分)。
类别与三种手臂模式分别统计，不交叉展开。

| 指标后缀 | 定义与观察方法 |
| --- | --- |
| `episodes` | 本次重置批次内该类实际结束的回合数，含超时与失败；用于判断样本是否充足 |
| `position_error_m` | 每回合先对活动手与有效采样步求平均，再对该类回合求平均；越低越好 |
| `orientation_error_rad` | 与上项相同聚合方式的最短朝向角误差，rad；越低越好 |
| `position_ready_fraction` | 每回合所有活动手同时满足位置容差的时间比例，再对回合求平均；越高越好 |
| `pose_ready_fraction` | 当前Mobile课程容差下，所有活动手位置与朝向同时达标的时间比例；默认1.5 cm与0.10 rad |
| `failure_rate` | 本次重置批次内该类非超时终止回合数／该类结束回合数；越低越好 |
| `failure_per_10s` | 10×批次失败次数／累计实际存活秒数；课程使用独立聚合的同口径窗口指标 |
| `height_shift_mean_m` | 仅第二类：结束回合所使用的额外高度偏移均值，m；用于核对采样范围 |

位置达标容差沿用 `workspace_quality.position_tolerance_m`，默认8 mm；关闭Q诊断时使用8 mm。
达标率只判断位置，朝向精度另看 `orientation_error_rad`。它是时间比例，不是整回合成功率。

分类失败统计在终止信号仍有效、类别尚未重新抽样时记录。
首个物理步就失败的回合仍计入失败率；若没有有效位姿采样，则省略误差和达标率，避免以零值填补。
初始重置、空类别及缺少终止／超时信号的人工重置，不产生该类失败率。
位姿采样沿用既有约定：排除刚重置帧，回合终止物理步不进入位姿均值。

RSL-RL把一个训练迭代收到的重置批次标量取平均，因此图上的 `episodes` 是平均批次数量，
`failure_rate` 是有样本批次的平均比例。它们用于训练趋势判断；精确跨回合失败率应在固定样本评估中统计。

建议先看第二类的误差和达标率是否改善，再检查第一类能力是否保持。
结合现有 `FrozenLower/height_m`、`FrozenLower/command_height_offset_m`、竖直冲击和足端诊断，
判断收益是否来自合理的高度调用。扩大范围后长期大误差样本过多时，先调整高度范围或第二类比例。

## 影响与验证边界

本功能改变Mobile训练目标分布及第二类目标保持时间。奖励函数、动作执行顺序、冻结Lower、
控制频率沿用当前实现。Mobile 的动作和观测布局见 [Upper 范围课程](UPPER_RANGE_CURRICULUM.md#快速使用)。
采样只需少量张量随机数和加法，无在线IK、候选重采样或额外策略推理。

单元测试覆盖世界坐标偏移、左右有效掩码、概率与范围、整回合保持、可选定时刷新、
局部重置、关闭时的随机数一致性和分类统计。真实PhysX短程测试覆盖train、保存、日志和play链路。
完整训练后的高度调用收益与新目标下的稳定性，需要通过上述对照实验检验。
