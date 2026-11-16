# Upper六维下肢接口与范围课程

适用任务：`Template-G1-Upper-Mobile-v0`。固定底座Upper保持17动作／93观测。

## 快速使用

新Mobile使用 **23动作／109观测**：先处理速度3维、高度1维、骨盆偏移2维，再处理17维腰臂动作。
冻结Lower仍接收650维历史观测并输出12维腿部动作，每控制步推理一次，物理子步持续执行。
默认冻结模型：

```text
logs/rsl_rl/g1_flat/2026-11-11_21-45-46_lower_base_p3/model_19997.pt
```

当前映射版本为schema 4：六维Lower动作按当前回合范围解码。
A1的schema 3虽同为23／109，但使用最终大范围解码再裁剪，需在匹配的旧实现中回放，当前版本从头训练。
H5C及更早的21动作检查点也需要显式迁移才能复用。

```powershell
uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --num_envs 4096 --seed 1217707953 --max_iterations 1500 --viz none physics=isaacsim_physx agent.run_name=upper_ranges_A2

uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --checkpoint "<23维Upper检查点>" --num_envs 4096 --seed 42 --max_iterations 1200 --viz none physics=isaacsim_physx

uv run python -m g1.rl play --task Template-G1-Upper-Mobile-v0 --checkpoint "<23维Upper检查点>" --num_envs 16 --seed 42 --real-time --viz kit physics=isaacsim_physx
```

输出到 `logs/rsl_rl/g1_upper_mobile/`。`--max_iterations` 为本次训练预算，课程按表现推进，
达到最终等级所需轮数由训练结果决定。首训示例沿用A1的seed=1217707953，
先检查300／600／1300轮的贴边调用、跟踪和冲击，再决定后续预算。训练入口仍默认随机种子。

## 六维命令与物理语义

网络原始动作依次为 `[vx, vy, yaw, height, pelvis_x, pelvis_y, upper_joints × 17]`。
动作解码后，Lower接收 `[height, vx, vy, yaw, pelvis_x, pelvis_y]`。
零动作表示零速度、默认站高和零骨盆偏移，正负不对称范围分别缩放。
每个环境从自己回合所属等级读取范围：raw动作先限幅到[−1,1]，再按该范围正负分段缩放。
例如0级vx范围±0.1 m/s时，raw=0.1／0.5／1对应请求0.01／0.05／0.10 m/s。
上述为解码请求，最终有效速度仍受下述站立死区和训练包络处理。
骨盆目标也先按当前回合范围解码，再生成0.02 m/s的渐进参考。

| 命令 | 等级0 | 等级10，最终上限 |
|---|---|---|
| 高度偏移，m | −0.10～+0.01 | −0.40～+0.05 |
| vx，m/s | −0.10～+0.10 | −0.50～+1.00 |
| vy，m/s | ±0.03 | ±0.20 |
| yaw，rad/s | ±0.10 | ±0.80 |
| 骨盆X／Y偏移，m | 各±0.005 | 各±0.03 |

高度相对于默认根高度0.75 m。最终根高度目标为0.35～0.80 m，低高度失败阈值为0.25 m，
另有根倾角、膝部和躯干接触失败判定（沿用Lower的1 N净接触力阈值）。骨盆偏移位于双踝link原点中点、双脚平均水平朝向参考系，
它不是整机质量加权质心坐标。

最终范围取自冻结模型训练分布以内；模型记录的最深高度偏移为−0.405 m。
程序校验动作上限未超出该模型记录的范围，并使用其保存的高度—速度包络：

- 深度增加时，vx和yaw允许幅度线性缩小；高度偏移−0.30 m时，vx约为[−0.13,+0.26] m/s，yaw约±0.34 rad/s。
- 深于−0.324 m时三个速度分量均置零。实际允许速度取课程范围与训练包络的交集。
- 有效平移速度指令范数≤0.03 m/s且yaw指令绝对值≤0.05 rad/s时，三维速度置零，允许骨盆偏移。
  其余情况下骨盆偏移置零，匹配Lower在行走模式使用零偏移的训练语义。
- 骨盆有效参考按二维向量速度上限0.02 m/s渐进；进入行走时归零。

这几个处理均在本控制步Lower推理前完成。Upper观测读取处理后的物理指令；上肢关节目标和
原物理子步加速度采样时序继续沿用既有实现。骨盆通道不使用网络外的关节速度限制器。

## 课程参数

主要配置集中在 `config/g1/mobile_env_cfg.py`：最终动作范围在 `MobileActionsCfg`，
课程参数在 `MobileCommandsCfg.wrist_pose.range_curriculum`。命令类拥有课程状态，
在回合结算时更新；无需重写环境或PPO。配置前缀为 `env.commands.wrist_pose.range_curriculum`。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `initial_level` / `num_levels` | 0 / 11 | 0～10级，共10次扩展 |
| `advance` | true | false固定等级，仍更新诊断；play冻结推进 |
| `initial_lower_ranges` | 上表等级0六组范围 | 顺序为高度、vx、vy、yaw、骨盆X、Y |
| `initial_position_scale` | 0.50 | 初始腕部位置偏移范围系数，最终为1 |
| `initial_orientation_scale` | 0.50 | 初始腕部Euler偏移范围系数，最终为1 |
| `initial_height_shift` | [−0.05,−0.02] m | 第二类初始额外世界Z偏移，最终取height_task.shift_range |
| `position_tolerance_m` | 0.015 m | 课程联合达标位置容差 |
| `orientation_tolerance_rad` | 0.10 rad | 最短旋转角误差容差 |
| `pose_ready_gate` | 0.80 | 联合达标EMA推进门槛 |
| `failure_rate_limit` | 0.01 | 每个所需类别、每10机器人秒的窗口失败次数上限 |
| `ema_alpha` | 0.20 | 每个有效窗口更新一次 |
| `check_period_steps` | 3600 | 最短检查间隔，在回合结算时判断；当前24步PPO下约150轮，模拟时间60秒 |
| `min_episodes_per_class` | 512 | 每个窗口、每个所需类别的最低完成回合数 |
| `reset_on_load` | false | true显式重建课程及统计，保留网络／优化器恢复 |

每级按k/10插值起止范围。非平面回合按常规／高度70%／30%选择，左右单手／双手仍为33%／33%／34%。
常规目标按2～4秒刷新，高度目标完整位姿保持一回合，上限10秒；生成后的世界目标保持固定。
单手使用`left_offsets/right_offsets`，双手使用`dual_targets`；
双手几何参数见[任务说明](TASK_RUNNERS.md#g1-上肢位姿跟踪结构)。
[第三类平面目标、窗口评分与日志](UPPER_PLANAR_TARGETS.md)默认以20%基础概率参与统一课程；
训练中的自适应比例见下节。远场奖励、负载和Q奖励启用属于后续实验。

双手公共中心偏移按位置系数缩放；距离、前后错位和高度差围绕默认双腕几何缩放变化量：

```text
d = 默认间距 + position_scale × (采样间距 - 默认间距)
dx/dz = 默认差值 + position_scale × (采样差值 - 默认差值)
dy = sqrt(d² - dx² - dz²)，取正值
```

默认间距约0.30 m时，0级距离约[0.275,0.350] m、X/Z差约±0.025 m，
10级距离[0.25,0.40] m、X/Z差±0.05 m。默认几何须在配置范围内，初始化会校验。
公共Euler扰动与两腕完整相对旋转角分别按朝向系数缩放；相对角上限从0级0.075扩至10级0.15 rad。
位置连线与腕部旋转独立。额外世界Z偏移最后施加，两类目标共用这些规则。

每回合所有活动手在同一时刻同时满足位置与朝向容差，才计为联合达标。
每回合先求联合达标有效步比例，再按完成回合等权聚合窗口分数，EMA为：

```text
EMA = 0.8 × 旧EMA + 0.2 × 当前窗口分数
```

首个样本充足窗口直接初始化EMA。常规／高度类别按整个回合有效步比例统计；第三类以完整评价窗口为分母，
缺失和失败时间按未达标计入。所有实际采样类别均需满足样本量、EMA及按存活时长计算的失败保护条件。
仅启用一个类别时只检查该类。满足后每个检查窗口最多升一级。
样本不足的窗口不更新该类EMA，窗口计数重新开始。首步失败且无位姿采样的回合以零质量计入，
人工重置和初始重置排除。位姿统计沿用现有约定，排除刚重置帧及终止物理步。

升级只影响新回合。旧回合保留其等级、目标范围和下肢边界，且不进入新等级课程统计。
升级后新等级重新积累EMA和计数，课程单向推进。Lower执行误差和冲击仅作诊断，未加入额外门槛。

## 命令奖励的归一化

动作映射采用当前回合的范围，等级变化只在新回合生效。六维命令变化率使用固定参考尺度：

```text
vx / 0.30；vy / 0.10；yaw / 0.30
h < 0：h / 0.25；h >= 0：h / 0.02
pelvis_x / 0.03；pelvis_y / 0.03
```

这些变化率归一化值允许超过±1。六维变化率代价为各维归一化差分平方和，权重−0.10。
变化率每控制步只更新一次，奖励重复读取不推进历史，局部重置清理对应环境。
变化率不除dt，RewardManager统一乘控制dt一次。

速度幅值 `lower_command_magnitude` 使用本回合课程范围归一化，权重仍为−0.05：

```text
负向有效速度：q_j = c_j / abs(本回合该轴下限)
非负有效速度：q_j = c_j / 本回合该轴上限
幅值代价 = q_vx² + q_vy² + q_yaw²
```

每轴使用本方向50%范围时贡献0.25，使用完整范围时贡献1，与课程等级无关。
同一物理速度在更宽的课程范围中代价更低。每个子环境读取自己保存的回合范围，
全局升级不会改变未重置回合的分母。未启用课程时，使用动作配置中的固定速度范围；
零宽度方向贡献0，关闭速度接口时整项为0。

幅值和变化率读取经过组合包络与静止门控后、实际下发Lower的有效命令。
幅值奖励只读数据，不更新差分历史，也不覆盖高度和骨盆偏移。原始速度动作及动作映射保持原样。
`Episode_Reward/lower_command_magnitude` 标签保留；新旧公式的惩罚值需结合课程等级和物理速度解读。

原三维速度raw动作越界代价保持Σmax(|a_raw|−1,0)²，权重−0.0003。
高度和骨盆没有新增幅值惩罚。上肢17维action_rate保持−0.03，竖直冲击保持权重−0.12、
soft_limit=4 m/s²，Q权重保持0。

例如vx从0.15变成0.30 m/s，变化率原始代价恒为(1−0.5)²=0.25；
当前课程速度上限变化不会改变这段相同物理指令的代价。新增骨盆两维另行贡献变化代价。

## Dwell稳定保持奖励

Mobile新增 `env.rewards.dwell`，权重默认 `+0.2`。所有活动手当前世界位姿同时满足
位置误差≤10 mm、最短朝向角误差≤0.04 rad，且骨盆link原点世界Z速度的滑动RMS≤0.02 m/s时，
返回一份任务级奖励1，否则为0。单手和双手模式上限相同，闲置手不参与门控。
正常超时可计入最后物理步，失败终止步不发奖；RewardManager统一施加权重与控制dt。

| 配置路径 | 默认值 | 含义 |
|---|---|---|
| `env.rewards.dwell.weight` | `0.2` | 正向稳定保持奖励权重；0关闭该项及其诊断累计 |
| `env.rewards.dwell.params.position_tolerance_m` | `0.01` | 所有活动手的位置误差上限，m |
| `env.rewards.dwell.params.orientation_tolerance_rad` | `0.04` | 所有活动手的最短旋转角误差上限，rad |
| `env.rewards.dwell.params.vertical_speed_tolerance_m_s` | `0.02` | 实际骨盆世界Z线速度RMS上限，m/s |
| `env.actions.joint_pos.vertical_speed_rms_window_s` | `0.1` | RMS历史窗口，s；向上对齐完整控制周期 |

当前控制60 Hz、物理120 Hz，窗口覆盖最近6个控制步、12个物理样本：

```text
每个控制步：s_t = mean(v_z²，覆盖该步全部物理子步)
滑动RMS = sqrt(mean(s_t, s_(t-1), ..., s_(t-5)))
bonus = 当前活动手位姿达标 AND 窗口已满 AND 滑动RMS <= 0.02
```

先平方再平均，正负速度不会抵消。各回合独立累计满窗，前0.1秒不发奖；局部重置只清空对应环境。
目标刷新保留速度历史，位姿门控立即使用新目标。它表示过去0.1秒垂向安稳和当前位姿达标，
并不要求位姿也连续达标0.1秒。水平移动和转向仍可进行，指令平滑由命令变化率奖励负责。
RMS历史仅用于奖励与诊断，不改变动作／观测布局、Lower推理频率或其他奖励的原有口径。
续训从新回合开始，RMS重新满窗；没有额外的连续驻留计时器。

该项仅注册于Mobile。建议固定其他参数做对照，同时比较Dwell合格比例、8 mm位置达标比例、
角误差、失败率和垂向冲击，确认稳定保持与任务精度的实际取舍。

实现验证：309项CPU回归通过，覆盖满窗、正负反转、早期子步尖峰、局部重置、重复读取、
目标变化及既有奖励／训练接口。另以4个真实PhysX环境运行90个控制步，独立记录12样本窗口核对RMS，
最大误差约7.5e−9 m/s；局部重置、正奖励路径和冻结Lower权重检查通过。该核验用于验证采样与接口，
稳定性改善由正式训练对照评估。可用 `G1_DWELL_SMOKE=1` 运行 `tests/test_dwell_physics.py` 复查。

## 平面任务自适应采样

范围课程保持原晋级门槛，采样调度只改变新回合的任务类别概率。配置位于
`MobileCommandsCfg.wrist_pose.range_curriculum.planar_sampling`，Mobile默认启用。

同一等级下保留最近4个连续、样本足够的平面统计窗口，前2个与后2个比较。
每段用`10 × 总失败数 / 总存活秒数`计算失败频率，不对单窗口比值简单平均。
当前每窗口3600控制步、PPO每轮24步，对应约150轮检查一次，前300轮与后300轮比较。
如改变PPO rollout长度，观察轮数为`history_windows × check_period_steps / num_steps_per_env`。

```text
relative_improvement = (previous_failure - recent_failure) / max(previous_failure, failure_rate_limit, 1e-9)
```

历史已满、最近一次平面窗口仍超过失败门槛、前后改善不足10%时，提高5个百分点。
20%→25%→30%→35%→40%，达到上限后继续统计。改善明显时保持当前比例。
调整后的下一个窗口只暂停增加比例，照常更新滑动历史和范围课程；再下个检查点即可继续判断。
历史不足或样本不足不加码，样本不足会打断连续历史。

范围晋级优先于采样调度：升到下一级时恢复配置基础概率20%，清空旧等级历史、冷却和等级内调整次数。
已经运行的回合保留原任务，新比例从之后的回合重置开始生效。
`advance=false`只冻结等级，训练时采样仍可调节；play／teleop冻结调度，使用恢复出的概率。
纯平面或关闭平面的配置（概率1或0）不运行自适应调度。

| 参数（`planar_sampling.*`） | 默认值 | 用途 |
|---|---:|---|
| `enabled` | true | false使用配置中的固定基础概率，适合固定比例对照 |
| `history_windows` | 4 | 必须为至少2的偶数；默认约600轮 |
| `minimum_relative_improvement` | 0.10 | 低于该改善幅度判定下降不足；负数表示退化 |
| `probability_step` | 0.05 | 单次增加5个百分点 |
| `maximum_probability` | 0.40 | 上限，须不小于配置基础概率且小于1 |
| `cooldown_windows` | 1 | 每次增加后跳过的调整检查次数；期间仍收集数据 |

基础概率仍由`planar_task.probability`唯一配置。非平面回合的高度条件概率保持0.3：
当平面概率为p时，常规与高度分别为`0.7(1-p)`和`0.3(1-p)`，左右手比例保持原配置。
这些是回合概率，实际训练步数占比还受回合长度和提前失败影响。

调度状态保存在schema 2课程状态的可选`planar_sampling`字段，包含独立版本、配置、概率、历史和冷却。
已有schema 2检查点在其余契约匹配时可直接续训：保留等级与课程统计，从20%和空采样历史开始。
同设置恢复新检查点时完整恢复调度；采样参数变更时只重建采样调度，课程等级与统计保留。
该规则适用于`planar_sampling.*`；基础概率、目标范围等原有参数仍按原课程契约校验。
`range_curriculum.reset_on_load=true`仍表示重建整个课程，包含采样调度。

新增TensorBoard分组`Curriculum/upper_sampling`：

| 标签后缀 | 含义 |
|---|---|
| `planar_probability` | 下一次回合采样使用的平面概率 |
| `history_windows` | 已收集的有效连续窗口数，满4才比较 |
| `previous_failure_per_10s` / `recent_failure_per_10s` | 前、后半段累计失败频率；历史满后写入 |
| `relative_improvement` | 前后相对改善，低于0.10且最近窗口未达标时允许增加 |
| `cooldown_remaining` | 之后还需暂停增加的窗口数 |
| `level_adjustments` | 当前等级内已经增加比例的次数，晋级清零 |

先同时查看概率、相对改善与现有平面`window_failure_per_10s`，再看level和其他两类表现。
功能核验使用受控统计验证触发和恢复；采样增强对稳定性的收益由正式训练对照判断。

## 日志与判断

| TensorBoard标签 | 含义／用法 |
|---|---|
| `Curriculum/upper_ranges/level` | 当前可分配给新回合的等级 |
| `.../checked_level` | 最近一次窗口判断对应的等级；升级时可能落后level一级 |
| `.../range/*` | 新回合允许的六维上下限、腕部偏移系数、额外高度偏移范围 |
| `.../regular/ema`、`.../height_shifted/ema` | 当前等级的联合位姿达标EMA；升级清零 |
| `.../*/ema_valid` | 首个有效窗口前为0，此时EMA零值表示尚未初始化 |
| `.../*/window_episodes` | 最近检查窗口的真实完成回合数 |
| `.../*/window_pose_ready_fraction` | 最近窗口的回合等权联合位姿达标比例 |
| `.../*/window_failure_rate` | 最近窗口失败回合数／完成回合数 |
| `.../*/window_failure_per_10s` | 10×失败次数／实际存活秒数；课程失败保护使用此值 |
| `.../sample_ready`、`quality_ready`、`failure_ready` | 最近窗口三个推进条件，结合checked_level看 |
| `Metrics/wrist_pose/task/{regular,height_shifted}/pose_ready_fraction` | 1.5 cm＋0.10 rad联合达标时间比例，回合后记录 |
| `.../position_ready_fraction` | 所有活动手同时满足8 mm位置误差的时间比例，与Q门控共用容差 |
| `FrozenLower/pelvis_offset_error_m` | 零速度调用时实际骨盆与有效偏移命令的二维误差 |
| `FrozenLower/command_pelvis_offset_m` | 同一静止采样窗口的有效骨盆偏移模长 |
| `Episode_Reward/dwell` | 已加权、经标准回合归约的Dwell贡献 |
| `Dwell/pose_ready_fraction` | 所有活动手同时满足10 mm／0.04 rad的时间比例，包含RMS窗口尚未满的阶段 |
| `Dwell/ready_fraction` | 位姿、满窗、速度RMS和非失败条件全部满足的时间比例，包含窗口预热时间 |
| `Dwell/base_vertical_speed_rms_m_s` | 仅在窗口已满后累计的速度RMS均值；先按回合平均，再对本次重置环境平均 |

仍保留腕部位置／角误差、闲置手偏差、实际高度、高度误差、速度误差、加速度和各奖励贡献。
位置奖励使用tanh粗项与指数精细项的60%／40%混合，尺度为0.10／0.010 m，左右权重均为1。
8 mm纯位置指标与课程15 mm／0.10 rad联合门槛分别定义；本次精修不改变课程推进条件。
旧日志的纯位置达标阈值为10 mm，新旧达标比例和位置奖励贡献应按各自配置解读。
Dwell诊断仅在奖励项启用时累计，不存在有效满窗样本时不写入零RMS。
完整跟踪公式见 [任务奖励说明](TASK_RUNNERS.md#g1-上肢位姿跟踪结构)。
先看课程EMA和失败率，再看当前范围；课程停滞时看哪个条件未满足。现有回合日志经过
RSL-RL批次平均，课程自身直接聚合回合计数，推进不依赖TensorBoard均值。

## 保存、续训与固定条件回放

检查点 `infos.g1_upper_ranges`（schema 2）保存等级、三类EMA、计数、实际存活时间、检查时钟和课程契约；
`infos.g1_control_interface` 保存23／109布局、物理映射及奖励尺度。
当前控制契约为schema 4、action_mapping=episode_range，并记录三项动作的课程来源。
新增 `velocity_magnitude_normalization=episode_range` 标识幅值公式；
`velocity_reward_reference` 继续记录固定的速度变化率尺度。
奖励公式变化不改变控制接口。新增双手几何配置属于目标分布变更，
A2～A4B以及本次修改前schema 1课程检查点可复用网络／优化器，需使用下述`reset_on_load=true`重建课程统计。
旧A1的schema 3在项目train/play入口恢复权重前被拒绝，避免相同维数掩盖动作含义变化。
标准续训恢复统计和时钟，但仿真从新回合开始，未完成回合不续接；冻结Lower权重不会更新。
`play`恢复保存等级并冻结课程推进。

修改目标范围、比例、容差、双手几何配置或课程定义，以及首次加载旧双手采样检查点时，显式添加：

```text
env.commands.wrist_pose.range_curriculum.reset_on_load=true
```

这样重建课程统计。动作结构、物理映射和奖励参考尺度属于检查点控制契约，变更时需重新训练
或显式迁移；能力enabled开关允许用于play消融。

用同一个新检查点回放最终等级目标，可添加：

```text
env.commands.wrist_pose.range_curriculum.initial_level=10
env.commands.wrist_pose.range_curriculum.advance=false
env.commands.wrist_pose.range_curriculum.reset_on_load=true
```

关闭全部Lower调用时，速度、高度和骨盆三个enabled均设为false。固定Lower仍反馈维持平衡。
归档模型及其params可随Upper运行目录搬移，指定 `env.actions.joint_pos.checkpoint_path=<run>/artifacts/frozen_lower_model.pt`。

## 本次实现验证

首版已完成模块局部review、接口与课程影响review及全局review。CPU回归239项通过；
真实PhysX覆盖固定Upper和Mobile的train/play、Mobile保存续训与导出，以及四环境站立／平滑上肢运动。
站立和上肢运动各10秒均无失败，Lower观测与原实现逐项一致，冻结权重及局部重置检查通过。
日志、布局及课程恢复链路已验证；完整11级课程收敛和最大范围下的任务表现由正式训练验收。

2026-11-13映射修复：六维请求统一使用当前回合范围，移除二次课程裁剪，控制契约升至schema 4。
本轮保持EMA、范围起止值、奖励参考及权重、探索初始化、训练包络和门控配置，
用于单独验证映射修复的收益。实际稳定性改善以A2训练及同条件回放为准。
修复后CPU回归247项通过，真实PhysX固定Upper与Mobile的短训、保存、Mobile续训、play共2项测试通过；
新增验证包括六维全区间解码、回合等级隔离、跨等级奖励等价及A1检查点拒绝加载。

2026-11-14幅值改造：速度幅值按各子环境本回合的正／负向范围归一化；
保留−0.05权重、固定尺度变化率、原始动作越界及其他奖励配置，供下一轮单独对照训练。

2026-11-15双手几何采样：新增公共中心、25～40 cm三维间距、前后／高度各±5 cm错位，
以及公共朝向上的最大0.15 rad随机相对旋转。单手采样规则、动作与观测布局继续沿用。
模块及总体review完成，330项CPU回归通过；2项真实PhysX集成验证覆盖固定基座／Mobile短训、
play、Mobile保存续训与策略导出。测试强制双手模式，在实际GPU目标上校验三维间距、位置差及相对角上限。
几何参数纳入课程契约，旧采样检查点需要显式重建课程统计；完整任务收益由正式训练评估。

2026-11-16窗口评分与三类课程：第三类以完整分配窗口评分，缺失和失败时间保留在分母中；
三类分别积累EMA，失败保护使用每10机器人秒发生率，默认检查间隔3600步。
Mobile奖励贡献日志改为按各结束回合的实际存活时长归一化，训练奖励本身保持原样。
课程状态升至schema 2；385项CPU回归通过，真实混合回合核验覆盖提前失败和三类推进。
