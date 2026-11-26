# Upper：平面迁移目标与距离时间预算

适用任务：`Template-G1-Upper-Mobile-v0`。已完成目标生成、时间预算、逐环境超时、
固定窗口评分、三类课程、实际时长日志、续训契约和可选远场引导奖励。

## 启用与当前边界

当前训练配置已启用第三类并开放三类统一课程，默认值为：

```text
env.commands.wrist_pose.planar_task.probability=0.2
env.episode_length_s=35
env.commands.wrist_pose.range_curriculum.advance=true
```

基础比例为56%常规、24%高度、20%平面。训练时默认启用
[自适应平面采样](UPPER_RANGE_CURRICULUM.md#平面任务自适应采样)，平面比例可增至50%，晋级恢复20%。
非平面回合按`height_task.probability=0.3`选择高度目标，手臂模式仍独立选择。
平面任务没有额外降高偏移。各类别在每回合开始时确定。

三类目标分别积累EMA，所有实际采样类别都达标后升一级。`advance=false`可固定等级，同时保留诊断统计；
play恢复检查点等级并冻结推进。已有按手臂模式记录的指标仍包含所有类别。
奖励全程计算，物理失败终止保持当前配置。

用现有23动作／109观测检查点查看第三类目标：

```powershell
uv run python -m g1.rl play --task Template-G1-Upper-Mobile-v0 --checkpoint "<checkpoint.pt>" --num_envs 8 --seed 42 --real-time --viz kit physics=isaacsim_physx env.episode_length_s=35 env.commands.wrist_pose.planar_task.probability=1.0 env.commands.wrist_pose.range_curriculum.advance=false env.commands.wrist_pose.range_curriculum.initial_level=0 env.commands.wrist_pose.range_curriculum.reset_on_load=true
```

`initial_level=0`用于近距离验证，改为10可查看最终空间。将概率改回0时，
若要恢复原训练时限，也应将`env.episode_length_s`改回10。

## 过渡场分组采样

设置 `env.commands.wrist_pose.planar_task.transition_sampling.enabled=true` 开启。Level 0–3
的平面回合都从站位距离至少 25 cm 处开局。距离是站位相对初始骨盆的水平径向距离。

| 组别 | Level 0–3 的平面回合占比 | 距离 | 转角 |
|---|---:|---|---|
| `walk` | 75% | 25–40 cm | ±15° |
| `turn` | 25% | 25–30 cm | 绝对角度下限 20°，上限为 Level 0–1 的 25°、Level 2–3 的 35°，左右等概率 |
| `wide` | 0% | 25 cm 起，受原等级正方形及腕目标距离上限约束 | 原等级方位扇区 |

从 Level 4 起，`wide` 按 25%、37.5%、50%、62.5%、75%、87.5%、100% 逐级增加。
`turn` 沿用 v8 同一级的绝对比例，分别为 21.43%、17.86%、14.29%、10.71%、7.14%、3.57%、0%。
其余平面回合抽 `walk`。`turn` 从 Level 4 起跟随同级 `wide` 的角度上限，Level 4–9
依次为 45°、65°、90°、115°、140°、160°，距离保持 25–30 cm。Level 10 只有 `wide`。
回合预算使用实际站位、目标朝向和该级 Lower 指令范围。

用 `transition_sampling.turn_bearing_bridge=null` 恢复旧版固定的 20–25° 转角，
再设 `transition_sampling.probability_knots=null` 可恢复 v8 原版组别比例。改变组内采样的旧 checkpoint
续训需设置 `range_curriculum.reset_on_load=true`，并显式将 `initial_level` 设为所需起点。
`failures mine` 默认恢复 checkpoint 的组内采样，指定 `--planar-sampling transition` 使用 v8 的统一分布做版本对照。

Mobile 训练开启过渡场采样后，`planar_turn_focus` 从 Level 5 起自动加密本等级新增的绝对转角区间。
默认将 Turn 组内 70%、Wide 组内 40% 的回合用于专项，左右等概率。Level 5–10 的专项分别抽
45–65°、65–90°、90–115°、115–140°、140–160°、160–180°。各组配额和给定角度下的距离规则沿用原课程。
晋级统计在每组保留同一比例的原生样本，保持原生组别分布。默认 Level 5–9 保留 30%，Level 10 保留 60%。
专项训练从同一 checkpoint 开关对照时，设 `env.commands.wrist_pose.planar_turn_focus.enabled=false` 可恢复原生训练分布。
Play、Teleop 和失败评估沿用原生采样。

训练日志的 `Metrics/wrist_pose/task/planar/<group>/` 记录已结束回合数、每 10 机器人秒失败频率、
进入近场比例、首次入场时间及完整评价窗的腕部达标率。近场以骨盆距离和朝向同时达标为准。
`Metrics/wrist_pose/task/planar/turn_focus/<group>/` 另给出专项表现，包括前 0.1 秒内的 `base_contact` 比例。
`failures mine` 在正常与失败的平面回合中均写入对应字段，并在 `summary.json` 汇总为 `planar_groups`。

用同一个 v6 起点比较旧采样和新采样。两支都固定 Level 3、20% 平面比例、开启远场奖励，
并从同一份网络和优化器重建课程统计：

```bash
checkpoint=logs/rsl_rl/g1_upper_mobile/2026-11-18_18-09-28/model_7999.pt
for transition in false true; do
  uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 \
    --checkpoint "$checkpoint" --num_envs 4096 --seed 1217707953 \
    --max_iterations 1200 --viz none --run_name "v6_l3_transition_$transition" physics=isaacsim_physx \
    agent.mirror_augmentation=true \
    env.commands.wrist_pose.range_curriculum.initial_level=3 \
    env.commands.wrist_pose.range_curriculum.reset_on_load=true \
    env.commands.wrist_pose.range_curriculum.advance=false \
    env.commands.wrist_pose.range_curriculum.planar_sampling.enabled=false \
    env.commands.wrist_pose.planar_approach.enabled=true \
    env.commands.wrist_pose.planar_task.transition_sampling.enabled="$transition"
done
```

从新采样的 checkpoint 续训或用 `play`、`teleop` 回放时，沿用该 run 的采样配置与 `initial_level`。常规续训省略
`reset_on_load=true`，恢复已保存的 Level 和课程统计。
失败采集会恢复训练时的采样设置。用下列命令评估两支时，分别替换 checkpoint 和输出路径，
统一使用 Level 4 过渡场配置。同种子和环境数使首批目标一致，后续目标序列会随自动重置时机变化。

```bash
uv run python -m g1.rl failures mine --checkpoint "<model_9198.pt>" \
  --level 4 --category planar --planar-sampling transition --policy_output mean --seed 42 \
  --num_envs 256 --episodes 600 --max_failures 600 --output "<new-output-directory>"
```

## 远场接近奖励

训练时设置 `env.commands.wrist_pose.planar_approach.enabled=true` 开启，默认关闭。
所有改动仅作用于平面回合。站位残差取实际骨盆到目标站位的水平距离及目标朝向的最短角误差。

| 项目 | 默认值 |
|---|---|
| 平移、转向进展权重 | +0.60、+0.30 |
| 进展归一化参考速度 | 0.40 m/s、0.40 rad/s |
| 平移、转向死区 | 0.20 m、15° |
| 腕部跟踪倍率 | 远场 0.10，近场 1.00。距离 0.20–0.45 m 和角误差 15–45° 各自线性过渡 |
| 活动臂肩肘姿态惩罚 | 权重 −0.25，偏离默认关节目标的容差 0.15 rad，尺度 0.50 rad，均方代价上限 2.0 |

进展用相邻物理状态的死区外误差差分，除以控制周期和参考速度，再限制到 [−1, 1]。
回合首步应用姿态代价与腕部倍率，进展从第二个物理样本开始计算。
两个近场过渡因子的乘积决定腕部跟踪的恢复程度，也决定肩肘姿态代价的退坡程度。
Q 奖励继续使用活动腕 5 cm 门槛，20 N 硬终止继续生效。

从同一个 v6 检查点各续训 1200 轮，固定 Level 3 和基础 20% 平面概率：

```bash
checkpoint=logs/rsl_rl/g1_upper_mobile/2026-11-18_18-09-28/model_7999.pt
for approach in false true; do
  uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 \
    --checkpoint "$checkpoint" --num_envs 4096 --seed 1217707953 \
    --max_iterations 1200 --viz none --run_name "v6_l3_approach_$approach" physics=isaacsim_physx \
    agent.mirror_augmentation=true \
    env.commands.wrist_pose.range_curriculum.advance=false \
    env.commands.wrist_pose.range_curriculum.planar_sampling.enabled=false \
    env.commands.wrist_pose.planar_approach.enabled="$approach"
done
```

训练日志含 `Episode_Reward/planar_approach` 和 `PlanarApproach/`，后者分别记录已结算平面回合的
平移进展、转向进展、肩肘代价及进入近场的程度。对两个模型用相同种子的 Level 6 `planar mean`
评估检查早期肩躯接触与存活，再结合完整评价窗的位姿达标指标确认到位情况。

## 按目标几何交接给 Lower

同时启用 `planar_approach.enabled` 和 `planar_approach.geometry_enabled`。前向 X 使用实时骨盆的水平朝向，
以各活动腕的世界目标为输入。位于躯干宽度内的近处目标，在 `x ≤ +0.10 m` 时关闭该腕的位置与姿态跟踪奖励，
到 `x = +0.14 m` 恢复到该几何规则允许的全量权重。横向在 0.10–0.14 m 之间放宽。
距离超过 0.50 m 时逐渐降低腕部权重，0.75 m 起为 0.10，身后目标再随背向程度降低。
平面任务的站位和朝向仍参与评价，腕部使用站位规则和目标几何规则中较小的倍率。Q 和到位稳定奖励使用活动腕中较小的倍率。

身体进展用同一目标下连续两帧的实际位姿计算。贴身时缩小目标与前向可追踪区域的差距，远处或身后目标
缩小水平距离或转向误差。已有平面站位时，优先奖励接近指定站位和朝向。换目标或改变进展度量的首帧跳过差分。

移动阶段对活动臂肩肘偏离默认关节构型计代价，每只臂的归一化均方代价上限为 2.0。
贴身让位时，构型代价在 0.25 秒内逐步给到当前门控对应的强度。
下面的训练把构型权重从 −0.25 调到 −0.35。

Mobile 的几何版默认另加腰部目标偏离代价，权重 −0.20，默认位死区 0.15 rad，归一化尺度
0.50 rad，三轴均方代价上限 2.0。Planar 用实时骨盆与站位的朝向误差控制强度，15° 以内退出，
15–30° 平滑增加。Regular 和 Height 用实时骨盆系内的活动腕目标决定是否需要转向。
跨身、远侧或明显身后的目标可启用，贴身后撤和双腕转向需求冲突的场景保留原有让位逻辑。
转向进展与腰部代价采用同一几何判断。`MobileApproach/waist_gate` 和
`MobileApproach/waist_posture_penalty` 记录启用强度与代价，可用 `waist_posture_weight=0` 做对照。

从新策略训练 30000 轮。下例延续单臂 35%、双臂 25% 的专项尝试率、镜像和近场 Lower 指令代价。

```bash
uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 \
  --num_envs 4096 --seed 1217707953 --max_iterations 30000 \
  --viz none --run_name upper_mobile_geometry_30k physics=isaacsim_physx \
  agent.mirror_augmentation=true \
  env.commands.wrist_pose.planar_task.transition_sampling.enabled=true \
  env.commands.wrist_pose.planar_approach.enabled=true \
  env.commands.wrist_pose.planar_approach.geometry_enabled=true \
  env.commands.wrist_pose.planar_approach.blocked_x_m=0.10 \
  env.commands.wrist_pose.planar_approach.arm_posture_weight=-0.35 \
  env.rewards.lower_command_magnitude.params.near_planar_gain=4.0 \
  env.commands.wrist_pose.regular_retract_sampling.probability=0.35 \
  env.commands.wrist_pose.regular_retract_sampling.dual_probability=0.25
```

`MobileApproach/` 记录平移和转向进展、姿态代价、腕部倍率，以及贴身门控和远场帧占比。
评估时一起看固定回收用例的硬碰撞和稳定到位、原生 Regular 的失败率，以及 1.2 m 目标的移动与携臂姿态。

## 入场后的 Lower 指令代价

训练命令加入以下覆盖项，可单独启用该代价。默认 `gain=0.0`。

```text
env.rewards.lower_command_magnitude.params.near_planar_threshold=0.5
env.rewards.lower_command_magnitude.params.near_planar_gain=4.0
```

平面回合首次进入近场后，对实际发给 Lower 的 `vy` 和 `wz` 计算
`max(abs(u) - threshold, 0)^2`，其中 `u` 是指令除以本回合相应符号的最大幅度。
两个轴的额外代价乘 `gain` 后并入原有 `lower_command_magnitude`，权重仍为 −0.05。
示例参数下，两轴同时顶格的额外奖励为 −0.10，RewardManager 再乘控制周期。
用普通平面采集评估策略自身表现，例如 `--level 4 --category planar --planar-sampling transition`。
训练日志的 `Metrics/wrist_pose/task/planar/near_lower/` 给出已结算回合入场后两轴超阈值的帧比例。
结合首次入场时间、入场后失败比例以及完整评价窗腕部达标率比较开关两组。

## 空间与朝向

作业站位相对初始盆骨的水平朝向系采样。最终X／Y各为±2.5 m，即5 m × 5 m；
默认等级表从 0 级的 ±0.10 m 和 ±10° 开始，到 10 级的 ±2.5 m 和 ±180°，中间范围由等级节点定义。
无课程时使用最终范围。

先按现有单手／双手几何生成局部腕目标，再将整组目标绕初始盆骨作共享水平转向，
并平移到作业站位。共享转角沿初始位置指向站位的方向。双腕三维距离、位置错位及相对旋转保持不变。
腕部roll／pitch扰动保持当前设置，当前阶段不引入全SO(3)朝向随机化。

生成后的完整位姿在世界坐标系保持到回合结束；机器人移动只改变实时盆骨相对观测。
目标刷新计时器置为无限，强制将计时器清零也不会刷新本回合的第三类目标。

各活动腕最终目标相对初始盆骨的三维距离上限为5 m。采样通过三角不等式预留局部腕目标半径，
再限制站位平移半径，直接保证此上限，使用固定次数的张量运算，无IK或重采样循环。
采样分布为方位角均匀、方向内平方根半径采样，覆盖正方形与扇区的交集；并非正方形面积均匀采样。
几何约束不等同于整机可达性保证。

## 参数

参数位于`mobile_env_cfg.py`。下表路径相对`env.commands.wrist_pose.planar_task`：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `probability` | 0.2 | 平面任务回合概率，1为纯平面验证 |
| `initial_half_extent_m / half_extent_m` | 0.25 / 2.5 | 初始／最终站位正方形半边长，m |
| `initial_bearing_rad / bearing_rad` | π/6 / π | 初始／最终目标方位角绝对上限，rad |
| `max_goal_distance_m` | 5 | 每只活动腕相对初始盆骨的三维距离上限，m |
| `timing.linear_speed / angular_speed` | 0.4 / 0.4 | 平移／转向规划速度上限，m/s、rad/s |
| `timing.linear_acceleration / angular_acceleration` | 0.3 / 0.4 | 对称启停的规划加速度，m/s²、rad/s² |
| `timing.ability_fraction` | 0.8 | 当前回合Lower前进／转向边界的规划使用比例 |
| `timing.settle_s` | 2 | 迁移后稳定余量，s |
| `timing.evaluation_s` | 10 | 固定评价窗口长度，s |
| `timing.near_episode_s` | 10 | 启用平面混合任务时，原两类回合时限，s |

以上规划参数只用于估算时间，不过滤动作、不下发速度轨迹，也不强制策略按某条路线行走。
预算假设正常站高、先转向再前进；实际策略仍自主输出六维Lower命令。

## 时间预算与窗口

以目标发布时的站位水平距离d和共享转角θ计算预算。
规划前进速度取`min(linear_speed, ability_fraction × 当前回合vx正向上限)`；
转向速度取`min(angular_speed, ability_fraction × 当前回合yaw双向较小上限)`。

对距离d、速度v、加速度a，起止速度均为零的时间为：

```text
d < v²/a：T = 2 × sqrt(d/a)
其余：    T = d/v + v/a
```

转向使用相同公式。长距离公式已经包含启停时间。

```text
评价开始步 = ceil((平移时间 + 转向时间 + 稳定余量) / control_dt)
评价窗口步数 = ceil(evaluation_s / control_dt)
回合截止步 = 评价开始步 + 评价窗口步数
```

60 Hz下10秒窗口恰为600步，取半开区间`[评价开始步, 回合截止步)`。
窗口开始由时间决定，与精度无关；预算发布后固定，不按实时距离延长。
`evaluation_window_mask()`提供只读时间门控，评分仅累计此窗口内的真实控制帧。

各子环境以实际重置时的全局控制步作为时钟起点。达到截止时间按`time_out=True`截断，
保留PPO的正常超时处理；局部重置只清理对应环境。Upper runner仅在第三类启用时关闭随机初始回合长度，
关闭第三类时沿用原PPO初始化选项与随机数路径。
保存／续训恢复课程与网络，但仿真从新回合开始，重新生成目标并分配完整预算，不恢复半个旧回合。

全局`env.episode_length_s`仍是硬上限。启用时逐级检查保守预算上界，空间或时间配置超出硬上限会报错，
不会截短评价窗口。默认设置下各类回合约10秒或12～31秒，推荐35秒硬上限。

## 兼容性与验证

平面概率为 0 时不增加目标采样随机数，超时回退到固定时限。动作和观测布局由 [Mobile 接口](UPPER_RANGE_CURRICULUM.md#快速使用) 配置。
启用后的空间、时间参数及控制步长写入课程契约。课程状态升级为schema 2，包含三类质量、失败次数和实际存活时间。
本次修改前的检查点需显式`reset_on_load=true`重建课程统计，包含未开启平面任务的旧两类检查点。
物理上限、奖励及冻结Lower执行代码不随本模块修改。

测试覆盖启停公式、逐步取整、完整窗口、所有手臂模式与课程等级、不同环境原点、
距离上限、世界固定、强制计时器到期、局部重置、PPO回合缓冲干扰、旧时序回退和契约恢复。
真实PhysX验证另覆盖train、保存续训、play和导出；该验证用于接口与计时，不代表远距离任务已训练收敛。

## 窗口评分

第三类每个控制帧判断所有活动手是否同时满足位置误差≤15 mm、最短朝向误差≤0.10 rad。
闲置手不参与判断。质量得分为：

```text
窗口质量 = 联合位姿达标步数 / 完整分配窗口步数
窗口覆盖比例 = 实际有效采样步数 / 完整分配窗口步数
```

默认分母固定为600步。提前失败、窗口内失败和未完成的未来时间都保留在分母中；
失败终止帧不贡献达标或误差样本。窗口前失败得分为0。位姿误差均值只使用真实窗口样本，
无样本时省略误差值，结合覆盖比例和失败指标判断。

末端超时步位于半开窗口之外；Mobile在重置事件修改机器人之前补采当前窗口帧，
每环境每控制帧只累计一次。当前完整窗口、提前失败、重复读取和局部重置有独立测试。
常规／高度目标继续按整个回合的有效步比例评分，排除刚重置帧和终止步。

三类质量分别进行EMA更新，系数0.2，推进门槛0.8，最少每类512个结束回合。
课程最短检查间隔3600步，即60秒／约150个PPO iteration；每次检查最多升一级。
旧等级回合保留原目标和预算，且不会计入新等级课程统计。未启用类别不参与晋级，也不输出其EMA项。
样本不足窗口保持EMA不变并重新计数；类别概率较小时，可增加检查间隔或降低每类最少样本量。

## 失败保护与日志

课程使用统一时间暴露口径：

```text
每10机器人秒失败次数 = 10 × 失败次数 / 累计实际存活秒数
```

默认上限0.01，即平均每累计1000机器人秒发生不超过1次失败。这是发生率，数值可以大于1。
例如10秒回合100次结束、发生1次失败，与30秒回合100次结束、发生3次失败，
在实际存活时间接近1000／3000秒时具有近似相同的失败发生率。
分母使用到失败／超时为止的真实时长，排除PPO虚构的初始回合长度。原回合失败率同时保留作观察。

第三类TensorBoard标签位于`Metrics/wrist_pose/task/planar/`：

| 指标 | 定义与判断 |
|---|---|
| `eval_pose_ready_fraction` | 完整评价窗口内联合位姿达标比例，与第三类课程质量口径相同 |
| `eval_position_ready_fraction` | 完整窗口内所有活动手满足8 mm位置容差的比例，朝向另看角误差 |
| `eval_coverage_fraction` | 窗口实际采样覆盖程度；提前终止会降低覆盖 |
| `eval_position_error_m / eval_orientation_error_rad` | 真实窗口样本上的活动手平均误差，无样本时省略 |
| `initial_distance_m` | 发布目标时的作业站位平移距离 |
| `eval_start_s / episode_budget_s` | 本回合预定评价开始时间／总预算 |
| `episodes / failure_rate / failure_per_10s` | 当前重置批次结束回合数、回合失败率、按存活时长计算的失败发生率 |

`Curriculum/upper_ranges/{regular,height_shifted,planar}/window_failure_per_10s`是课程实际使用的窗口失败保护值；
`window_failure_rate`保留原回合失败率。课程直接汇总真实计数与存活时间，不依赖TensorBoard的批次均值。
任务级日志经过RSL-RL重置批次平均，用于趋势判断；精确晋级条件看课程窗口指标。

Mobile的`Episode_Reward/*`现为“每个已结束回合的加权累计贡献／实际存活秒数”，再对重置批次平均。
项目内轻量环境适配层在上游清空累计值前快照，归一化采用一次矩阵运算，随后继续调用原重置流程。
训练实际收到的奖励、权重、dt及PPO更新公式保持原样。与旧版按全局回合上限归一化的奖励曲线分别解读；
`Train/mean_reward`仍是回合累计回报，比较长短任务时优先结合每秒奖励贡献与任务质量。

## 本轮验证记录（2026-11-16）

模块和总体review完成，385项CPU回归通过。真实PhysX覆盖固定Upper、原Mobile和平面Mobile的
train／保存续训／play／导出，以及独立混合回合积分核验。独立核验比较24次窗口与奖励日志，
包含一次提前失败；窗口评分误差为0，每秒奖励贡献最大误差约2.98e−8，三类课程实际推进到3级。
这些是软件链路与统计口径验证；本轮没有启动正式长训，也没有添加或改动远场奖励。
