# Upper + 冻结下肢：速度、高度命令与站立验证

`Template-G1-Upper-Mobile-v0` 使用自由底座 G1 训练 Upper，同时由冻结的 Lower 策略控制双腿。
Upper 使用 **21维动作空间**，动作顺序为 `[vx_raw, vy_raw, yaw_raw, height_raw, upper_joint_raw × 17]`。
动作管理器先处理3维速度、1维高度，再处理17维腰臂动作。桥接为冻结 Lower 的6维输入：

```text
[height_offset, vx, vy, yaw_rate, pelvis_offset_x, pelvis_offset_y]
= [Upper_height_offset, Upper_vx, Upper_vy, Upper_yaw_rate, 0, 0]
```

高度偏置为零表示默认站立高度，双腿仍会根据全身状态进行反馈调整。
速度和高度由 Upper 策略选择，冻结 Lower 将其转换为腿部关节动作；零速度时继续反馈控制平衡。
速度和高度共用 `lower_command_rate` 变化惩罚，高度不加入幅值或越界惩罚。腕部采样范围保持原配置，未加入第三类目标采样。
Mobile 另启用 `base_vertical_acc`，以软阈值约束实际竖直冲击；它对骨盆世界Z方向的运动质量计费。
原 `Template-G1-Upper-v0` 继续作为固定底座对照。

两种任务共用 [空间 Q 先验奖励](WORKSPACE_QUALITY.md)，默认权重为零并记录诊断。
Mobile 可通过速度命令调整基座位置；Q 的学习效果通过后续同条件训练对照评估。

## 快速使用

在项目根目录运行。默认模型为：

```text
logs/rsl_rl/g1_flat/2026-10-21_22-59-54/model_9999.pt
```

同目录需保留 `params/env.yaml`，建议同时保留 `params/agent.yaml`。程序直接加载训练检查点的 actor，
通过 RSL-RL 的确定性推理模块执行，包含其观测归一化；下肢 critic、优化器及课程状态不参与 Upper 训练。

先观察零动作站立：三个速度命令为零，17个上肢动作保持默认关节目标，冻结下肢策略持续执行：

```powershell
uv run --extra isaacsim isaaclab zero_agent --task Template-G1-Upper-Mobile-v0 --num_envs 4 --viz kit physics=isaacsim_physx
```

训练 Upper：

```powershell
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --num_envs 512 --max_iterations 1500 --viz none physics=isaacsim_physx
```

回放这个移动任务训练出的 Upper 检查点：

```powershell
uv run --extra isaacsim python -m g1.rl play --task Template-G1-Upper-Mobile-v0 --checkpoint "<upper_mobile_model.pt>" --num_envs 4 --real-time --viz kit physics=isaacsim_physx
```

输出目录为 `logs/rsl_rl/g1_upper_mobile/`。Mobile 的 Upper 观测为 **107维**：原93维上肢观测，
加基座线速度、角速度、投影重力及高度10维，末尾为有效速度命令3维和目标高度偏置1维。
原93维中的动作观测只读取17维上肢动作。速度命令观测使用物理单位 m/s、m/s、rad/s。
目标保持原有单左手、单右手、双手模式及世界固定目标语义。

旧 Mobile 的20动作/106观测检查点（包括R9C系列），以及更早的17动作/103观测检查点，需要重新训练或显式迁移权重。
直接加载会在恢复权重前给出接口不匹配提示；新训练使用标准train/play入口。
固定底座任务维持17动作/93观测接口。

## 命令范围与变化惩罚

常用参数在 `config/g1/mobile_env_cfg.py` 的 `MobileActionsCfg.lower_command` 中显式配置：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `command_scale` | `(0.3, 0.1, 0.3)` | 基座坐标系的 vx、vy、yaw rate 对称上限，单位 m/s、m/s、rad/s |
| `enabled` | `True` | `False` 强制下发零速度，保留21维动作接口，高度独立控制 |

策略前3维动作先限制在 `[-1, 1]`，再逐轴乘 `command_scale`。零动作严格对应零速度。
缩小速度范围的例子：

```powershell
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --viz none physics=isaacsim_physx "env.actions.lower_command.command_scale=[0.2,0.08,0.2]"
```

强制零速度、默认站高的对照需同时添加 `env.actions.lower_command.enabled=false env.actions.height_offset.enabled=false`。
单独关闭 `lower_command` 仅关闭速度输出。
此时有效速度和速度变化代价为零，高度变化仍独立计入 `lower_command_rate`；原始策略输出保留用于记录，站立平衡反馈正常执行。

Mobile 的 `action_rate` 只作用于17维上肢原始动作，权重仍为 `-0.03`。
`lower_command_rate` 统一使用速度和高度四维限幅后的归一化有效命令，原始代价为：

```text
u_v = clip([vx_raw, vy_raw, yaw_raw], -1, 1)
u_h = clip(height_raw, -1, 1)
cost = sum((u_v_t - u_v_(t-1))²) + (u_h_t - u_h_(t-1))²
step_reward = weight * cost * step_dt
```

默认权重 `-0.10`，保留H2配置；奖励管理器再乘控制步长。它是每控制步差分，不除以 dt，单位为无量纲。
四个分量直接求和，共用相同权重；原始代价范围为 `[0,16]`。
高度按正负两侧各自的范围归一化：默认高度从 `-0.15 m` 变为 `+0.02 m` 时，`u_h` 从 `-1` 变为 `+1`，高度差分代价为4。
相同的归一化差分对应相同代价；默认负向 `0.1` 对应15 mm、正向 `0.1` 对应2 mm，物理变化尺度不同。
某个接口 `enabled=false` 时，其分量和差分状态归零，另一接口继续计费；若高度范围某侧端点为0，该侧有效命令按0处理。
同一控制周期重复读取不推进历史；局部重置清空对应子环境的命令及差分状态。
不同原始输出若限幅到同一个有效命令，变化代价为零。持续保持非零命令同样没有变化代价。
TensorBoard 记录为 `Episode_Reward/lower_command_rate`；可通过 `env.rewards.lower_command_rate.weight` 调整权重。
H0及更早版本的该日志仅覆盖三维速度；新版本加入高度后，代价的统计范围变为四维，跨版本对比需区分这一口径。
21动作/107观测的H0检查点可以直接续训。已有模型只执行play时输出保持原状，新增奖励通过后续训练影响策略。

### 可选命令与基座运动惩罚

`MobileRewardsCfg` 另外注册六项惩罚。当前保留R9C速度约束：
`lower_command_overflow=-0.0003`、`lower_command_magnitude=-0.05`；四项基座运动惩罚仍为0。
下表中的越界、幅值两项只读取3维速度，高度动作不计入；四维变化由上面的 `lower_command_rate` 统一约束。

设 `a` 为前三维原始命令动作，`u=clip(a,-1,1)`；`mean_substeps` 表示先在每个物理子步计算平方和，再对该控制步的所有子步求均值。

| 配置项（`env.rewards.<项>.weight`） | 未加权代价 | 意义 / 单位 |
| --- | --- | --- |
| `lower_command_overflow` | `sum(max(abs(a)-1,0)²)` | 原始命令超出有效区间的程度，无量纲 |
| `lower_command_magnitude` | `sum(u²)` | 持续调用下肢的幅值成本，范围 `[0,3]`，无量纲 |
| `base_lin_vel` | `mean_substeps(sum(v_world²))` | 基座线速度成本，m²/s² |
| `base_ang_vel` | `mean_substeps(sum(omega_world²))` | 基座角速度成本，rad²/s² |
| `base_lin_acc` | `mean_substeps(sum(a_world²))` | 基座线加速度成本，m²/s⁴ |
| `base_ang_acc` | `mean_substeps(sum(alpha_world²))` | 基座角加速度成本，rad²/s⁴ |

基座统一使用 `MobileActionsCfg.joint_pos.base_body_name="pelvis"` 的 link 原点，覆盖三个世界坐标轴。
线速度和线加速度均对应该原点；线加速度从刚体质心加速度按刚体运动学换算，角加速度直接读取同一刚体。
这些项评价相对世界的实际基座运动；已有末端加速度项继续评价双腕相对骨盆的运动。

命令接口 `enabled=false` 时，两项新命令代价为零；基座四项仍评价真实运动。
采样缓存只读物理状态，覆盖完整物理子步，包含终止前最后一个子步。部分重置只清理对应环境；重复读取同一控制步复用缓存。
权重为零时，RewardManager 跳过奖励函数，日志 `Episode_Reward/<项>` 为零；基座子步缓存仍维护，可供测试或分析直接读取 `joint_pos.mean_base_motion_l2()`。
该读取返回四列 `[base_lin_vel, base_ang_vel, base_lin_acc, base_ang_acc]` 的未加权代价，**零权重日志不代表实际代价为零**。

启用时覆盖对应权重，例如 `env.rewards.lower_command_overflow.weight=-0.0001`。
奖励管理器统一应用权重和控制步长，函数内部不再乘 dt；加速度未使用控制步末速度差分。
建议逐类启用，并同时检查腕部跟踪、命令饱和、基座运动和失败率。

### 竖直冲击软惩罚 `base_vertical_acc`

首版默认权重为 **`-0.05`**，软阈值为 **`4.0 m/s²`**，四项三轴基座运动惩罚仍保持0。
`a_z` 是骨盆link原点相对世界Z方向的实际线加速度，使用刚体运动学从质心换算，不使用控制步末速度差分。
该项与末端相对骨盆加速度分别评价基座和上肢自身运动。

```text
substep_cost = max(abs(a_z) / soft_limit - 1, 0)²
cost = mean_over_physical_substeps(substep_cost)
step_reward = weight * cost * step_dt
```

先对每个物理子步计算超额，再求均值，保留短时冲击和正负反转。当前控制60 Hz、物理120 Hz，每次奖励覆盖两个物理子步，包含终止前最后一步。
代价无量纲；低于阈值时为0。高度位置、平面加速度和命令幅值不直接参与此项。
正常运动仍可超过软阈值并支付成本，执行目标不会被该奖励裁剪或限速。

H0～H2回放与正常运动标定给出的首版依据为：

| 条件 | 阈值4时的未加权代价均值 |
| --- | ---: |
| H2原生输出 | 0.392 |
| H2保持当时实际高度 | 0.000259 |
| 正常前进，实际约0.29 m/s | 0.00475 |
| 平移与旋转组合运动 | 0.0149 |
| 0.8秒或1.5秒平滑下蹲约15 cm | 0 |

该标定支持以小成本保留平稳调用、重点约束冲击；首版权重的学习效果仍需续训对照验证。

| 配置 | 默认值 |
| --- | ---: |
| `env.rewards.base_vertical_acc.weight` | `-0.05` |
| `env.rewards.base_vertical_acc.params.soft_limit` | `4.0` |
| `env.rewards.lower_command_rate.weight` | `-0.10` |

TensorBoard 新增 `Episode_Reward/base_vertical_acc`。它是已加权、经回合时间归一化的超额代价；为0仅表示没有超阈费用，并不表示骨盆完全静止。
关闭新项可设置 `env.rewards.base_vertical_acc.weight=0.0`；子步缓存仍维护。
诊断可读取 `joint_pos.mean_base_vertical_acc_excess_l2(soft_limit=4.0)`，原 `mean_base_motion_l2()` 四列接口保持不变。

建议先从同一H2检查点、相同种子各续训300轮，仅改变新项权重：

```powershell
uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --checkpoint .\logs\rsl_rl\g1_upper_mobile\2026-11-08_15-39-55_Lower_command_H2\model_1199.pt --num_envs 4096 --seed 1591316204 --max_iterations 300 --viz none physics=isaacsim_physx agent.run_name=H3A_vertical_off env.rewards.lower_command_rate.weight=-0.10 env.rewards.base_vertical_acc.weight=0.0

uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --checkpoint .\logs\rsl_rl\g1_upper_mobile\2026-11-08_15-39-55_Lower_command_H2\model_1199.pt --num_envs 4096 --seed 1591316204 --max_iterations 300 --viz none physics=isaacsim_physx agent.run_name=H3B_vertical_on env.rewards.lower_command_rate.weight=-0.10 env.rewards.base_vertical_acc.weight=-0.05 env.rewards.base_vertical_acc.params.soft_limit=4.0
```

评估使用相同目标，重点复查预先选定的中低位问题样本、竖直冲击、足端运动和接触切换，并同时检查腕部跟踪及高度使用分布。
新项不改变21动作/107观测布局，H2可直接续训；冻结Lower权重、指令映射和执行时序保持原样。

替换冻结下肢模型时，覆盖动作配置路径；相对路径以项目根目录解析：

```powershell
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --viz none physics=isaacsim_physx "env.actions.joint_pos.checkpoint_path=logs/rsl_rl/g1_flat/<run>/model_9999.pt"
```

## 高度偏置接口

高度配置位于 `MobileActionsCfg.height_offset`：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `height_offset_range` | `(-0.15, 0.02)` | 相对默认根高度的目标偏置，m，必须包含0 |
| `enabled` | `True` | 关闭后固定为默认站高，保留一维动作接口 |

设 `u=clip(height_raw,-1,1)`，负半轴映射到范围下限，正半轴映射到上限。
默认 `u=-1/0/+1` 对应 `-0.15/0/+0.02 m`，`u=-0.5/+0.5` 对应 `-0.075/+0.01 m`。
高度目标不额外平滑或积分，由冻结Lower执行反馈控制。零动作严格对应默认站高。
高度通过 `lower_command_rate` 参与四维归一化命令变化惩罚，保持任意恒定高度时该分量为零。
幅值和越界惩罚仍只覆盖三维速度；上肢动作变化项仍只覆盖17维腰臂动作。

调整范围或固定高度：

```text
env.actions.height_offset.height_offset_range=[-0.10,0.02]
env.actions.height_offset.enabled=false
```

## 控制与参数来源

| 部分 | 运行方式与来源 |
| --- | --- |
| Upper | PPO训练3个速度、1个高度偏置及腰部/双臂17个动作；上肢动作缩放与PD沿用 Upper 配置 |
| Lower | 冻结 actor，以确定性输出控制12个腿部关节，保持 `eval` 并关闭梯度 |
| 低层观测 | 从最新 Lower 配置构建独立观测管理器，关闭观测噪声，保留模型要求的历史排列 |
| 腿、脚执行器 | 深拷贝最新 `lower_limb_agent` 的对应配置 |
| 腰、臂、手指与自碰撞 | 继续由 Upper 自有资产配置维护；Mobile 解除根节点固定 |
| 时钟 | 同步最新 Lower 的物理步长和 decimation；本次模型为120 Hz物理、60 Hz控制 |

所选模型输入为650维、输出为12维，使用5帧、按观测项排列的历史。
低层输入包含全身43个关节的状态，包括手指；Upper直接控制腰部和双臂，通过速度与高度命令调用 Lower。
每个环境步按以下顺序执行：

1. `lower_command.process_actions()` 处理前3维动作，更新本步有效速度及变化代价。
2. `height_offset.process_actions()` 处理第4维动作，更新本步目标高度偏置。
3. `joint_pos.process_actions()` 处理后17维腰臂动作，构造目标预览，并使用本步速度/高度命令、下肢上一动作及全身状态推进一次低层历史。
4. 推理下肢12维动作，应用最新 Lower 的动作缩放及默认角度偏置。
5. 每个物理子步分别写入上肢和下肢目标；每个控制步只进行一次低层推理。

采用普通腿部关节动作执行器，不启用 Lower 任务的随机上肢扰动与轨迹控制器。
两组控制器的关节集合在初始化时检查为互不重叠。
动作管理器按配置顺序处理，`lower_command` 和 `height_offset` 均必须在 `joint_pos` 前；初始化会校验该依赖。
实现位于 `mdp/frozen_lower_actions.py`，Mobile 独立声明三项动作以固定顺序。
更新 Lower 后检查模型输入尺寸、观测项顺序、历史顺序、控制周期、动作映射及 actor结构。
参数可随最新配置更新；接口变化时，错误信息会指明差异，需要提供匹配的新模型。

保存 Upper 模型时，`infos.g1_control_interface` 记录动作顺序、维度、观测维度、上肢关节顺序、
速度缩放、高度范围、开关和控制周期，接口版本为2。加载时核对结构与时钟；范围和开关允许按实验覆盖，并在下一次保存时记录。
原有冻结 Lower 模型归档和身份检查继续生效，详见模型目录下 `artifacts/` 与 `params/env.yaml`。

## 重置与指标

每回合将根节点恢复到该子环境原点对应的默认位置和朝向，所有关节及PD目标同步重置，起始速度为零。
下肢速度及高度命令、变化代价、上一动作、观测历史及统计缓冲仅清理被重置的子环境。
第一版不随机化根位姿，因此腕部默认采样基准可在各回合复用。

Mobile 在原有上肢奖励上增加 `termination`，权重为 `-10`，由奖励管理器乘控制步长。
机身倾斜超过 `0.8 rad` 或根节点高度低于 `0.45 m` 会终止回合，超时单独处理。
默认根高度0.75 m，高度偏置范围对应目标根高度0.60–0.77 m，保留当前终止阈值。
后续开放更深下蹲或地形时，需同步调整合法高度与失败判据。

TensorBoard除原有训练和腕部指标外，记录以下冻结控制指标：

| 标签 | 含义与判断 |
| --- | --- |
| `FrozenLower/height_error_m` | 实际根高度相对“默认站高+已下发高度偏置”的绝对误差，m |
| `FrozenLower/height_m` | 实际根高度，相对子环境原点，m |
| `FrozenLower/command_height_offset_m` | 实际下发的高度偏置，m |
| `FrozenLower/planar_speed_m_s` | 实际平面速度幅值；结合下发命令判断是否符合意图 |
| `FrozenLower/tilt_rad` | 骨盆相对竖直方向的倾角，越低越直立 |
| `FrozenLower/command_speed_m_s` | 实际下发的平移命令幅值，反映主动移动程度 |
| `FrozenLower/command_yaw_rad_s` | 实际下发的转速绝对值，反映主动旋转程度 |
| `FrozenLower/velocity_error_m_s` | 实际基座平移速度与命令之差的范数，越低表示执行越接近命令 |
| `FrozenLower/yaw_error_rad_s` | 实际基座转速与命令之差的绝对值，越低表示执行越接近命令 |

各项先对每回合有效采样步求均值，再对本次重置的回合求均值。
采样在控制步开始，读取上一控制步最终物理状态，并与产生该状态的上一有效命令配对，随后才发布新命令。
跳过新回合首帧，终止物理步不包含在均值中。
结合 `Episode_Termination/fallen`、`Episode_Termination/low_height` 和回合长度判断失败。
关注腕部精度的同时，也检查这些站立指标。

## 接口与执行验证

本轮自动检查使用真实 PhysX、默认冻结 Lower checkpoint 和4个环境，分别覆盖零命令站立、
上肢运动，以及前后移动、侧移、旋转、混合命令、停止、反向切换和高度下降/上升/回站高。
Lower 输入与原版观测函数逐项比较，历史每控制步推进一次，另外核对变化代价、世界目标固定性和局部重置。

```powershell
uv run python -m pytest tests/test_lower_command_actions.py tests/test_task_runners.py -q

$env:G1_LOWER_COMMAND_SMOKE='1'
$env:G1_FROZEN_STAND_SMOKE='1'
uv run --extra isaacsim python -m pytest tests/test_lower_command_physics.py tests/test_frozen_standing_physics.py -q

$env:G1_TRAIN_PLAY_SMOKE='1'
uv run --extra isaacsim python -m pytest tests/test_task_runners_physics.py -k upper_mobile -q
```

控制执行检查使用有限的命令组合与时长。训练得到的基座使用策略，以及 Q 是否改善，需后续训练对照确认。
当前 Lower 训练代码主要在回合起点采样速度，持续策略命令带来的频繁切换应结合速度误差、失败率和回合长度观察。

导出的 Upper 网络输出21维原始动作；限幅、速度缩放、高度映射与冻结 Lower 推理由动作模块完成。
部署时需一并保留这些处理步骤及 Lower 模型，21维原始输出经该链路转换为全身关节目标。

## 上肢关节加速度的子步采样

固定底座 `Template-G1-Upper-v0` 和自由底座 `Template-G1-Upper-Mobile-v0` 共用
`UpperJointPositionAction` 的统计逻辑，保留官方关节位置动作的缩放、默认角度偏置、裁剪和目标下发。
Mobile 的冻结下肢继续每个控制步推理一次、每个物理子步下发目标。

`joint_acc` 使用 `joint_acc_l2_substeps`，按动作项实际控制的17个关节计算：

```text
周期代价 = mean_over_substeps(sum_over_controlled_joints(joint_acc ** 2))
```

先平方、再对物理子步平均，因此加速与减速都会产生代价。当前120 Hz物理、60 Hz控制，
一个控制周期包含两个8.33 ms子步。动作项在下一子步开始前采集上一子步，奖励读取时补入最后一个子步，
同周期重复读取使用缓存。新动作清空周期统计，局部重置只清空对应子环境。
奖励函数返回未加权代价，权重和控制步长仍由RewardManager统一处理。

TensorBoard继续使用 `Episode_Reward/joint_acc`，沿用奖励管理器的回合累计与时间归一化。
保存的 `params/env.yaml` 中，`joint_acc.func` 为 `joint_acc_l2_substeps` 表示采用新口径；
旧版本的官方 `joint_acc_l2` 只采样控制周期末的加速度。跨版本的物理平滑程度应使用同目标回放比较。
覆盖完整周期后，同一权重可能产生更强惩罚，权重需单独校准。
末端加速度也覆盖完整周期，采用下节说明的骨盆相对定义；其余奖励沿用原有采样方式。

这项统计要求环境逐物理子步调用动作项。当前以 `physics=isaacsim_physx` 验证；
后端启用内部decimation合并且 `decimation > 1` 时会明确报错，需另行适配该后端。
`--viz newton` 仅选择可视化，与上述物理执行方式无关。

动作和观测接口相同的任务版本间，已有Upper检查点可以继续回放；当前Mobile需匹配21动作/107观测接口。
采用新采样口径续训时，策略会按完整子步的加速度奖励优化。
子步缓存属于当前仿真状态，每次启动重新建立。

逐子步独立采样、PD目标、其他奖励及重置回归：

```powershell
$env:G1_UPPER_SUBSTEP_SMOKE="1"
uv run --extra isaacsim python -m pytest tests/test_upper_substep_physics.py -q
```

## 末端加速度：骨盆坐标系与完整子步

固定底座和 Mobile 共用 `end_effector_relative_lin_acc_l2_substeps`。
`end_effector_acc` 评价左右腕 link 原点在随骨盆平移、旋转的坐标系中的位置二阶导数：

```text
p_relative = R_pelvis.T @ (p_wrist - p_pelvis)
a_relative = d²(p_relative) / dt²
cost = mean_over_substeps(||a_left_relative||² + ||a_right_relative||²)
step_reward = weight * cost * step_dt
```

加速度单位为 m/s²，未加权代价单位为 m²/s⁴。每个子步先分别求双腕平方和，再对周期内子步取均值，
因此首次子步的冲击及加速/减速都会计入。所有单手、双手指令模式均约束双腕，包括闲置侧。

瞬时计算使用刚体解析关系，不维护位置差分历史或滤波状态。仿真提供质心加速度，先对腕部与骨盆分别换算
到 link 原点，再扣除参考系平移、角加速度、向心和科里奥利贡献。令下列向量均以世界坐标表示：

```text
a_origin = a_com + alpha × (p_origin - p_com)
                 + omega × (omega × (p_origin - p_com))
r = p_wrist - p_pelvis
u = v_wrist_origin - v_pelvis_origin - omega_pelvis × r
a_relative = R_pelvis.T @ (a_wrist_origin - a_pelvis_origin
                          - alpha_pelvis × r
                          - omega_pelvis × (omega_pelvis × r)
                          - 2 * omega_pelvis × u)
```

整体平移、旋转且上肢相对骨盆保持不变时，理论代价为零；腰部或手臂相对骨盆的运动仍计入。
物理耦合引起的真实相对摆动也计入。当前双手指令继续保持世界固定，因跟踪世界目标而发生的上肢补偿运动
会按其相对加速度产生代价。

参考 link 与腕部 link 在 Action 初始化时解析一次，配置入口为：

| 配置 | 当前值 |
| --- | --- |
| `env.actions.joint_pos.acceleration_reference_body_name` | `pelvis` |
| `env.actions.joint_pos.acceleration_body_names` | `left_wrist_yaw_link`、`right_wrist_yaw_link` |
| `env.rewards.end_effector_acc.params.action_name` | `joint_pos` |
| `env.rewards.end_effector_acc.weight` | `-1e-5`，保留为新定义的校准起点 |

`UpperJointPositionAction` 复用关节加速度的子步时序，为末端代价维护独立的周期和、均值缓存。
奖励读取补入最后子步；同周期重复读取使用缓存；局部重置清理对应槽位。权重为零而未触发奖励读取时，
局部重置仍保留其他子环境的末子步。缓存不会修改 PD 目标、动作历史或冻结下肢每控制步一次的推理时序。

TensorBoard 标签继续为 `Episode_Reward/end_effector_acc`。保存配置中的函数名和参考 link 标识新定义。
旧版本是世界加速度、控制步末采样，因此新旧曲线的绝对数值应分别解读；旧世界坐标漏采比例也只适用于旧定义。
历史 `outputs/analysis/upper_20261030_substep_rewards/` 保留旧定义的诊断证据。
权重强度与学习收益需使用同条件回放和训练对照评估，本次实现未调整其他奖励权重。

解析运动学与缓存测试：

```powershell
uv run python -m pytest tests/test_upper_kinematics.py tests/test_upper_substep_actions.py -q
```

上节的 `G1_UPPER_SUBSTEP_SMOKE` 物理测试同时验证本项：使用独立展开公式逐子步采样，
核对周期均值、奖励总和、回合日志、禁用奖励后的局部重置，以及动作和冻结下肢的回归。

2026-11-03 的实现验证：118 项相关测试、固定底座/Mobile 的两项物理采样测试及两项公共入口
短训练—保存—play—导出测试通过。模块 review、接入影响 review 和总体 review 的记录见
[验证摘要](../outputs/verification/ee_relative_acc_20261103.json)。这些检查验证实现正确性，训练收益另行评估。

## 关节目标速度超阈值惩罚

Upper的 `target_speed` 奖励读取 `UpperJointPositionAction.joint_target_velocity`。
它按动作顺序覆盖腰部与双臂17个关节，包括当前闲置手臂，使用经过缩放、默认偏置和配置裁剪后的目标：

```text
target_velocity = (processed_target[t] - processed_target[t-1]) / step_dt
cost = sum(max(abs(target_velocity) / soft_limit - 1, 0) ** 2)
step_reward = weight * cost * step_dt
```

`target_velocity` 是离散目标更新速度，单位rad/s；阈值归一化后的代价无量纲。
目标每个控制步更新一次，速度按60 Hz计算。该项仅记录、评价目标变化，保留原始动作、PD目标与执行时序；
实际关节速度和加速度仍由 `joint_vel`、`joint_acc` 评价。

每次处理动作后缓存速度和本次目标。重置时，历史目标同步到重置事件已经下发的PD目标，速度缓存清零，
仅处理对应子环境。新回合第一个策略目标相对重置目标的跳变正常计罚。
这些缓存属于仿真状态，现有网络输入和输出维度保持一致，C1000可以继续加载。

配置中的两个主要参数是：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `env.rewards.target_speed.params.soft_limit` | `2.5` | 开始增加惩罚的目标更新速度，rad/s |
| `env.rewards.target_speed.weight` | `0.0` | 默认关闭，设置负值启用惩罚 |

速度在阈值内时，此项代价为0；速度为阈值的2倍、4倍时，单关节代价分别为1、9。
权重与时间缩放仍由RewardManager统一处理一次。
TensorBoard使用 `Episode_Reward/target_speed`。权重为0时，日志贡献为0，目标速度缓存仍持续更新；
原始超阈值情况可通过诊断回放读取。

使用C1000、48个环境、默认指令采样，分别进行10秒确定性与带已学动作噪声的冻结回放，
标定结果保存在 [target_speed_calibration.json](../outputs/verification/target_speed_calibration.json)。
确定性输出约2.8%的关节样本超过2.5 rad/s，带动作噪声时约75.5%超过阈值。
因此，该项也会对探索噪声形成约束，训练时应同时观察 `Policy/mean_std` 与跟踪精度。

在这组固定权重回放上，`weight=-1e-4` 的预计平均惩罚为：

| 输出方式 | 平均惩罚，乘控制步长前 | 占正向位姿跟踪奖励 | 相当于现有action_rate代价 |
| --- | --- | --- | --- |
| 确定性 | `-0.00181` | `0.073%` | `39.8%` |
| 带动作噪声 | `-0.01861` | `0.786%` | `33.1%` |

建议将 `-1e-4` 作为初次短续训的候选权重，默认配置仍为0。
这是启用前的代价量级标定，学习后的精度与平滑收益通过独立训练对照确认。
沿用C1000的奖励设置时，可显式传入：

```text
env.rewards.joint_acc.weight=-1e-7
env.rewards.target_speed.weight=-1e-4
env.rewards.target_speed.params.soft_limit=2.5
```

## 检查点与搬移

Upper 检查点的 `infos.g1_frozen_lower` 保存模型SHA-256、迭代号及实际输入输出接口。
首次保存时归档以下文件：

```text
<upper_run>/artifacts/frozen_lower_model.pt
<upper_run>/artifacts/params/env.yaml
<upper_run>/artifacts/params/agent.yaml  # 原目录提供时一并复制
```

恢复 Upper 时校验当前冻结模型与检查点记录的模型和接口一致。
搬移完整运行目录后，可覆盖路径以使用归档模型：

```text
env.actions.joint_pos.checkpoint_path=<upper_run>/artifacts/frozen_lower_model.pt
```

回放与续训均保留 `artifacts/params/`。公共 `train/play` 入口和任务族自动识别方式保持一致，
`teleop/eval` 仍按现有边界支持下肢任务。

## 验证范围

冻结加载测试检查真实模型的批量确定性推理、与已有导出的输出一致、随机数隔离及接口变化检测。
标准PPO回归检查固定底座任务行为，并检查冻结模型归档与恢复校验。
物理验证比较桥接观测与原 Lower 观测函数在同一状态下的逐项一致性，
检查部分子环境重置及冻结权重不变。

运行有界的 headless 站立与上肢运动测试：

```powershell
$env:G1_FROZEN_STAND_SMOKE="1"
uv run --extra isaacsim python -m pytest tests/test_frozen_standing_physics.py -q
```

测试使用4个环境，分别测试10秒零动作站立和10秒平滑肩、肘、腕及腰部运动，结果写入pytest临时目录的
`standing.json`，失败时保留日志。它验证所测运动范围内的稳定性；长期训练、大幅快速动作及负载操作需要独立实验。

本次验证结果保留在 [frozen_lower_standing.json](../outputs/verification/frozen_lower_standing.json)：
两阶段均无跌倒，站立阶段最大倾角约 `0.079 rad`，上肢运动阶段约 `0.098 rad`。

公共入口的短训、保存、回放与导出验证：

```powershell
$env:G1_TRAIN_PLAY_SMOKE="1"
uv run --extra isaacsim python -m pytest tests/test_task_runners_physics.py -k upper_mobile -q
```
