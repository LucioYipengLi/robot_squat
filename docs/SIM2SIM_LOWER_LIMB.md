# 下肢策略 Sim2Sim 迁移协议

核对日期：2026-11-09。适用对象：当前 `lower_limb_agent` 的非循环 MLP Actor，迁移目标可为 Unitree_Mujoco 或其他仿真。

本文记录当前代码确定的策略接口、部署适配职责及训练环境参考。本文不提供已实现的部署程序，也不代表已通过 MuJoCo 闭环验证。

## 1. 使用边界与配置归属

迁移的核心是：部署环境状态 → 策略观测 → 推理 → 策略关节目标 → 部署控制器。导出网络只覆盖其中的推理步骤。

| 类别 | 内容 | 部署时的处理 |
| --- | --- | --- |
| 策略协议 | 输入维度、关节顺序、历史顺序、相对位置基准、动作缩放、指令含义、策略周期 | 由适配层严格实现；改变语义后不再是同一策略接口 |
| 部署环境配置 | MJCF/URDF、质量惯量、碰撞、摩擦、驱动器、限位、控制增益、物理步长 | 由部署环境独立管理，记录实际值及与训练环境的差异 |
| 训练环境参考 | 训练资产、训练 PD、电机模型、armature、随机化、初始根姿态 | 只用于解释训练分布和分析迁移误差，不自动写入或覆盖部署环境 |

**训练环境参数不得作为部署配置的隐式覆盖项。** 若需要对齐某项动力学参数，应在部署配置中显式选择、记录原因并验证，而不是从训练 `env.yaml` 整体加载机器人配置。

`q0_policy` 等策略相对量基准具有双重来源：它来自训练配置，但已参与输入和输出计算，因此是适配层必须保存的协议常量。使用它计算观测，不等于要求部署机器人修改自己的默认姿态。

本文的接口针对当前资产解析结果。历史训练、命令行覆盖和资产版本变化均可能改变语义；对其他 checkpoint，应以其同次运行配置、课程元数据和实际模型为准，不能只凭 `650 → 12` 判断兼容。

## 2. 模型侧交付物

### 2.1 文件与内容

| 文件 | 内容 | 部署需求 |
| --- | --- | --- |
| `exported/policy.pt` | TorchScript Actor：MLP、权重、观测归一化模块和确定性输出 | PyTorch/LibTorch 路线必需；可独立加载 |
| `exported/policy.onnx` | ONNX 图和部分参数 | ONNX 路线必需 |
| `exported/policy.onnx.data` | ONNX 外部权重 | 当前导出文件引用此文件，必须与 `.onnx` 同目录并保持文件名 |
| `model_<iter>.pt` | Actor、Critic、优化器、迭代号、`infos.g1_curriculum` | 推理不必需；来源核对、重新导出、续训和恢复实际课程范围时需要 |
| `params/env.yaml` | 运行时保存的环境配置 | 协议核对必需，不应作为部署环境配置整体执行 |
| `params/agent.yaml` | Actor 架构、归一化、动作裁剪等配置 | 协议及导出核对必需 |
| `run.json`、Git 版本及运行中保存的 diff | 任务和代码来源 | 随迁移包归档；本地未提交改动也可能影响训练 |

当前 `g1.rl play` 复用 IsaacLab 播放入口：先加载指定 checkpoint，再导出到该运行目录的固定 `exported/policy.*`。后续播放可能覆盖旧导出，因此文件名不包含可靠的迭代号。

交付时至少记录：源 checkpoint 相对路径及 SHA-256、迭代号、任务/阶段、代码与资产版本、各导出文件 SHA-256、导出时的软件版本、协议版本或本文件版本。建议将选定模型复制到独立迁移包后固定使用，避免依赖会被播放覆盖的目录。

### 2.2 已核对样本

当前 [默认冻结下肢配置](../src/g1/tasks/upper_limb_agent/frozen_lower.py) 引用：

```text
logs/rsl_rl/g1_flat/2026-10-21_22-59-54/model_9999.pt
SHA-256: cdaf6553b9aec20264b1588f71f1db698eac287300ba0b55c287b6063432cfad
```

实际读取该样本的结果：

| 项目 | 已核对值 |
| --- | --- |
| 架构 | `650 → 512 → 256 → 256 → 12`，隐藏层 ELU，输出层线性 |
| 输出方式 | 确定性 Actor 输出，不进行训练时的高斯随机采样 |
| 归一化 | `obs_normalizer = Identity`；当前 Actor 配置 `obs_normalization=False` |
| TorchScript | float32 输入 `[N,650]`，输出 `[N,12]`；已在 CPU 核对 `[1,650] → [1,12]` |
| ONNX | float32 输入 `obs[1,650]`，输出 `actions[1,12]`，固定 batch=1，opset 18 |
| 权重来源 | TorchScript 的全部 MLP 权重与 `model_9999.pt` 逐项完全相同 |
| ONNX 外部数据 | 图中的权重引用 `policy.onnx.data`；无自带的关节/历史协议 metadata |
| checkpoint 内容 | `actor_state_dict`、`critic_state_dict`、`optimizer_state_dict`、`iter`、`infos` |

TorchScript 的 `reset()` 对当前 MLP 是空操作。历史、上一动作和指令平滑状态均由适配层管理，调用模型 `reset()` 无法清除这些状态。

网络不包含机器人模型、观测提取、历史拼接、指令采样、默认关节角、动作缩放、PD 控制、电机限矩、上肢轨迹、奖励或课程运行逻辑。当前样本也不包含需要另行恢复的观测归一化统计。

## 3. 时间与运行协议

| 项目 | 当前策略/训练值 | 归属 |
| --- | --- | --- |
| 策略周期 | `T_policy = 1/60 s`，60 Hz | 策略协议；历史也按此周期采样 |
| 历史长度 | 5 个采样点，包含当前点 | 策略协议；最旧到最新相距 `4/60 s` |
| 训练物理步长 | `1/120 s` | 训练环境参考 |
| 训练 decimation | 2 | 训练环境参考，形成 60 Hz 策略周期 |
| 训练回合长度 | 6 s | 训练环境参考；部署无需每 6 s 重置 |

部署仿真可以使用自己的物理步长；应按仿真时间安排 60 Hz 策略调用。不能因为改为更小物理步长，就同时按每个物理步更新历史和重新推理。无法整除时，应显式安排调度并记录时间误差，不应默认为相近频率等价。

每次策略调用后保持腿部位置目标，底层控制器按部署控制周期更新。训练腿部执行器在动作应用/物理推进过程中用当前关节状态计算力矩；策略更新频率与底层 PD 更新频率是两件事。

## 4. 观测协议：130 维基础量、650 维实际输入

### 4.1 字段、单位及切片

下表索引从 0 开始，切片使用 Python 左闭右开语义。基础切片用于说明单次采样的字段；网络实际输入使用最后一列的 650 维切片。

| 顺序 | 字段 | 单次维度 | 基础切片 | 650 维输入切片 | 定义 |
| --- | --- | --- | --- | --- | --- |
| 0 | `base_lin_vel` | 3 | `0:3` | `0:15` | 根刚体质心线速度，在根 link/基座坐标系表达，m/s |
| 1 | `base_ang_vel` | 3 | `3:6` | `15:30` | 根刚体角速度，在基座坐标系表达，rad/s |
| 2 | `projected_gravity` | 3 | `6:9` | `30:45` | 基座系下的单位重力方向，无量纲 |
| 3 | `velocity_commands` | 3 | `9:12` | `45:60` | 基座系指令 `[vx,vy,wz]`，m/s、m/s、rad/s |
| 4 | `pelvis_height_cmd` | 1 | `12:13` | `60:65` | 相对默认站高的 `h_offset`，m |
| 5 | `pelvis_offset_cmd` | 2 | `13:15` | `65:75` | 双足水平系下限速后的有效 `[dx,dy]`，m |
| 6 | `joint_pos` | 43 | `15:58` | `75:290` | 按第 5 节顺序排列的 `q_policy - q0_policy`，rad |
| 7 | `joint_vel` | 43 | `58:101` | `290:505` | 同一顺序的 `dq_policy - dq0_policy`，rad/s；当前 `dq0_policy=0` |
| 8 | `actions` | 12 | `101:113` | `505:565` | 上一次网络原始输出，按动作顺序；不是目标角或力矩 |
| 9 | `upper_body_target` | 17 | `113:130` | `565:650` | 上肢最终目标减对应 `q0_policy`，rad；见第 7 节 |

当前各观测项没有额外 scale、clip 或观测延迟。不能套用其他 Unitree 策略常见的速度乘数、关节速度缩放、统一观测裁剪等配置。

下肢策略虽然只控制 12 个关节，但观测包含 43 个全身关节，并需要 17 个上肢目标；不能将其当作只读腿部编码器的独立步态网络。

### 4.2 坐标系和测量点

令 `R_WB` 表示基座坐标到世界坐标的旋转：

```text
base_lin_vel      = transpose(R_WB) * v_root_COM_world
base_ang_vel      = transpose(R_WB) * omega_world
projected_gravity = transpose(R_WB) * normalize(gravity_world)
```

在世界重力沿负 Z、基座直立时，重力观测约为 `[0,0,-1]`，不是 `[0,0,-9.81]`，也不是 IMU 加速度计输出。

当前 IsaacLab 的 `root_lin_vel_b` 是 `root_com_lin_vel_b` 的别名，COM 指**根刚体自己的质心**，不是整机质量加权质心，也不是根 link 原点。若部署侧读取同一刚体上 site S 的世界速度，应使用部署模型自己的几何关系转换：

```text
v_root_COM_world = v_site_world + omega_world × (p_root_COM_world - p_site_world)
```

随后再转到基座系。不要为此把训练模型的 COM 写入部署 MJCF。若部署根 link 轴与策略基座轴不同，还要加入固定旋转标定。

当前 IsaacLab 相关姿态数据采用 `xyzw`，MuJoCo 四元数采用 `wxyz`。应在接口边界转换，并明确根 link、惯性坐标系和 IMU site 的外参；网络本身不直接接收四元数。

### 4.3 历史排列与重置

有效配置：`history_length=5`、`flatten_history_dim=True`、`concatenate_terms=True`、`history_order="term"`。

每项先按时间从旧到新排列，每帧内部保留该项自身分量/关节顺序，然后按表中字段顺序拼接：

```text
obs_t = concat(
    base_lin_vel[t-4], ..., base_lin_vel[t],
    base_ang_vel[t-4], ..., base_ang_vel[t],
    ...,
    upper_body_target[t-4], ..., upper_body_target[t]
)
```

这里的 `t` 是策略观测采样时刻；`actions` 项在该时刻的采样值是上一控制区间的动作 `a_(t-1)`。它的 5 帧自然保留各采样时刻对应的上一动作。

不能使用 `concat(frame[t-4],...,frame[t])`，也不能对 43 个关节逐关节拼完 5 帧再换下一个关节。单个 `joint_pos` 历史块的逻辑形状为 `[5,43]`，按时间优先展平。

首次初始化或重置后：

1. 清除上一动作，令其为 12 维零。
2. 同步物理状态、当前有效任务指令和上肢目标。
3. 计算首次完整观测，将每个观测项的首次采样值复制到该项全部 5 个历史槽。
4. 后续每个策略时刻只追加一次；重复读取、日志或渲染不得推进历史。

不能在重置后长期保留前一回合历史，也不能用 4 帧全零加当前帧替代首次填充。批量仿真时，只重置对应环境的缓存。

### 4.4 训练观测噪声，仅供参考

当前训练 `enable_corruption=True`，噪声为逐元素均匀加性噪声：

| 字段 | 训练噪声范围 |
| --- | --- |
| `base_lin_vel` | ±0.1 m/s |
| `base_ang_vel` | ±0.2 rad/s |
| `projected_gravity` | ±0.05 |
| `joint_pos` | ±0.01 rad |
| `joint_vel` | ±1.5 rad/s |
| `upper_body_target` | ±0.02 rad |
| 指令、上一动作 | 无配置噪声 |

这些噪声不在导出模型内。部署接口对齐时使用无人工噪声观测；若评估传感器噪声，应在部署评估配置中单独启用并记录。

## 5. 全身关节顺序与相对量基准

下表适用于当前训练资产的 43 关节解析顺序，`joint_pos`、`joint_vel` 的每个历史采样都使用此顺序。索引不包含浮动基座自由度。

`q0_policy` 是当前 PhysX 路径下默认关节位置：先初始化为零，再应用 `init_state.joint_pos` 中的匹配项；`dq0_policy` 全部为零。若换用其他训练配置，应从对应运行核对，不能沿用本表推断。

| 索引 | 关节名 | `q0_policy` / rad |
| --- | --- | --- |
| 0 | `left_hip_pitch_joint` | -0.10 |
| 1 | `right_hip_pitch_joint` | -0.10 |
| 2 | `waist_yaw_joint` | 0 |
| 3 | `left_hip_roll_joint` | 0 |
| 4 | `right_hip_roll_joint` | 0 |
| 5 | `waist_roll_joint` | 0 |
| 6 | `left_hip_yaw_joint` | 0 |
| 7 | `right_hip_yaw_joint` | 0 |
| 8 | `waist_pitch_joint` | 0 |
| 9 | `left_knee_joint` | 0.30 |
| 10 | `right_knee_joint` | 0.30 |
| 11 | `left_shoulder_pitch_joint` | 0 |
| 12 | `right_shoulder_pitch_joint` | 0 |
| 13 | `left_ankle_pitch_joint` | -0.20 |
| 14 | `right_ankle_pitch_joint` | -0.20 |
| 15 | `left_shoulder_roll_joint` | 0 |
| 16 | `right_shoulder_roll_joint` | 0 |
| 17 | `left_ankle_roll_joint` | 0 |
| 18 | `right_ankle_roll_joint` | 0 |
| 19 | `left_shoulder_yaw_joint` | 0 |
| 20 | `right_shoulder_yaw_joint` | 0 |
| 21 | `left_elbow_joint` | 0 |
| 22 | `right_elbow_joint` | 0 |
| 23 | `left_wrist_roll_joint` | 0 |
| 24 | `right_wrist_roll_joint` | 0 |
| 25 | `left_wrist_pitch_joint` | 0 |
| 26 | `right_wrist_pitch_joint` | 0 |
| 27 | `left_wrist_yaw_joint` | 0 |
| 28 | `right_wrist_yaw_joint` | 0 |
| 29 | `left_hand_index_0_joint` | 0 |
| 30 | `left_hand_middle_0_joint` | 0 |
| 31 | `left_hand_thumb_0_joint` | 0 |
| 32 | `right_hand_index_0_joint` | 0 |
| 33 | `right_hand_middle_0_joint` | 0 |
| 34 | `right_hand_thumb_0_joint` | 0 |
| 35 | `left_hand_index_1_joint` | 0 |
| 36 | `left_hand_middle_1_joint` | 0 |
| 37 | `left_hand_thumb_1_joint` | 0 |
| 38 | `right_hand_index_1_joint` | 0 |
| 39 | `right_hand_middle_1_joint` | 0 |
| 40 | `right_hand_thumb_1_joint` | 0 |
| 41 | `left_hand_thumb_2_joint` | 0 |
| 42 | `right_hand_thumb_2_joint` | 0 |

名称顺序与已保存的 [接口审计记录](../outputs/analysis/upper_20261103_e1/parameter_audit.json) 一致；该记录是辅助证据，迁移包仍应独立携带完整列表，避免依赖本地分析产物。当前代码用正则选择关节且 `preserve_order=False`，列表顺序由实际 articulation 关节顺序决定，不是按正则声明顺序或字典序排列。

### 5.1 部署关节坐标转换

按名称建立映射，并记录关节轴方向与零点。若某关节满足：

```text
q_policy  = sign * q_deploy + zero_offset
dq_policy = sign * dq_deploy
```

则所有位置观测、速度观测、上肢目标和腿部输出都必须使用同一映射。`sign` 通常为 ±1，`zero_offset` 单位 rad；不能只重新排序输出而忽略输入。

### 5.2 缺失手指关节

本机 Unitree_Mujoco 的 `g1_29dof.xml` 缺少上述 14 个手指关节。适配层仍必须产出 43 维观测。可将缺失手指显式建模为固定在策略默认位置：`q_policy=q0_policy`、`dq_policy=0`，使其相对观测为零；应将其标为**部署近似假设**并验证。

这不等于训练时手指所有状态始终为零，也不能消除手部质量、惯量和碰撞差异。若目标模型不具备某个腿、腰、肩、肘或腕关节，不能仅靠补零宣称兼容。

## 6. 动作协议与腿部目标

当前 `LowerBodyPositionAction` 继承 `JointPositionAction` 的下肢映射；自定义部分只追加独立上肢运动推进。

```text
q_target_policy = q0_policy_lower + 0.5 * a
q_target_deploy = (q_target_policy - zero_offset_lower) / sign_lower
```

`a` 为 12 维确定性网络输出。它是默认关节位置上的偏移系数，不是当前角度增量、角速度或力矩。当前动作项 `clip=None`，Runner 的 `clip_actions=None`；不能自行增加 tanh 或假定输出在 `[-1,1]`。部署控制器若有独立限位/限幅，应显式记录，并保留上一动作观测为网络原始输出。

| 动作索引 | 关节名 | 全身观测索引 | `q0_policy_lower` / rad | 本机 MuJoCo motor 索引 |
| --- | --- | --- | --- | --- |
| 0 | `left_hip_pitch_joint` | 0 | -0.10 | 0 |
| 1 | `right_hip_pitch_joint` | 1 | -0.10 | 6 |
| 2 | `left_hip_roll_joint` | 3 | 0 | 1 |
| 3 | `right_hip_roll_joint` | 4 | 0 | 7 |
| 4 | `left_hip_yaw_joint` | 6 | 0 | 2 |
| 5 | `right_hip_yaw_joint` | 7 | 0 | 8 |
| 6 | `left_knee_joint` | 9 | 0.30 | 3 |
| 7 | `right_knee_joint` | 10 | 0.30 | 9 |
| 8 | `left_ankle_pitch_joint` | 13 | -0.20 | 4 |
| 9 | `right_ankle_pitch_joint` | 14 | -0.20 | 10 |
| 10 | `left_ankle_roll_joint` | 17 | 0 | 5 |
| 11 | `right_ankle_roll_joint` | 18 | 0 | 11 |

MuJoCo 列仅是对本机 `D:/Code/unitree_mujoco/unitree_robots/g1/g1_29dof.xml` 的静态核对结果；其他 XML、SDK 电机编号和加载后的 `qpos`/`qvel` 地址必须分别按名称检查。浮动基座有额外位置/速度坐标，不能把 motor 索引直接当作 `qpos` 索引。

## 7. 上肢目标及执行协议

### 7.1 17 维目标顺序

| 目标索引 | 关节名 | 全身观测索引 |
| --- | --- | --- |
| 0 | `waist_yaw_joint` | 2 |
| 1 | `waist_roll_joint` | 5 |
| 2 | `waist_pitch_joint` | 8 |
| 3 | `left_shoulder_pitch_joint` | 11 |
| 4 | `right_shoulder_pitch_joint` | 12 |
| 5 | `left_shoulder_roll_joint` | 15 |
| 6 | `right_shoulder_roll_joint` | 16 |
| 7 | `left_shoulder_yaw_joint` | 19 |
| 8 | `right_shoulder_yaw_joint` | 20 |
| 9 | `left_elbow_joint` | 21 |
| 10 | `right_elbow_joint` | 22 |
| 11 | `left_wrist_roll_joint` | 23 |
| 12 | `right_wrist_roll_joint` | 24 |
| 13 | `left_wrist_pitch_joint` | 25 |
| 14 | `right_wrist_pitch_joint` | 26 |
| 15 | `left_wrist_yaw_joint` | 27 |
| 16 | `right_wrist_yaw_joint` | 28 |

当前 `q0_policy_upper` 为 17 维零，但适配公式仍应保留减默认位置的语义。

### 7.2 `final_goal` 的准确含义

当前配置为 `target_source="final_goal"`，来自 `UpperBodyMotion.get_observation_goal()`：

- 正在执行某段轨迹时，观测该段终点 `goal`。
- 空闲但已有待执行目标时，观测 `pending_goal`。
- 当前段执行期间收到的新目标只替换 pending，不立即暴露；直到当前段结束后才切换可见终点。
- 没有新目标时保留当前终点。

因此观测目标不是实时测量角，也不是轨迹当前参考 `q_ref`。部署若使用不同的轨迹规划器，应显式定义可见终点何时切换，避免将尚未开始执行的排队目标提前输入策略。

训练的上肢独立控制器执行五次平滑轨迹，并下发位置和速度参考；下肢网络本身不下发上肢动作。训练轨迹参数为 `max_velocity=2.5 rad/s`、`max_acceleration=8 rad/s²`、`min_duration_s=0.3 s`，这些是参考轨迹限制，不是机器人硬件能力或部署必须使用的值。

首次迁移可采用固定上肢姿态：由部署控制器保持该姿态，同时把对应最终目标转换到策略坐标输入。不能让上肢自由下垂却始终向策略报告零目标。若以后启用上肢运动，再验证规划器速度、加速度、可见终点与真实执行的一致性。

重置时，当前环境将上肢位置目标同步到重置后的实际角，速度目标和上肢实际速度置零，并清除轨迹队列。部署也应使重置后的目标与实际状态一致。

## 8. 任务指令协议

### 8.1 六维指令定义

```text
task_command = [h_offset, vx, vy, wz, pelvis_dx, pelvis_dy]
```

| 分量 | 单位/坐标系 | 含义 |
| --- | --- | --- |
| `h_offset` | m | 相对训练默认站高的骨盆根 link 高度偏置；负值为降低 |
| `vx,vy` | m/s，基座系 | 前后、侧向速度目标 |
| `wz` | rad/s，基座系 | 绕基座 Z 轴的角速度目标 |
| `pelvis_dx,dy` | m，双足水平参考系 | 骨盆相对双踝中点的水平位置目标，输入为限速后的有效值 |

观测中的指令顺序是 `[vx,vy,wz]`、`[h_offset]`、`[pelvis_dx,dy]` 三个独立历史块，不是直接把此六维数组作为一个历史块。

高度评估语义为：`target_z = env_origin_z + 0.75 + h_offset`。其中 `0.75 m` 是当前训练默认根高度，不是部署模型应被强制设置的初始高度；也不能拿部署 XML 中不同的根初始高度自动替换该参考并声称物理目标相同。若部署需要重定义站高，应单独记录为任务语义变化并验证。

### 8.2 模式

| 模式 | 编号，仅训练管理使用 | 高度指令 | 速度指令 | 骨盆 XY 指令 |
| --- | --- | --- | --- | --- |
| STAND | 0 | 0 | 全零 | 有效 |
| SQUAT | 1 | 非默认高度 | 全零 | 有效 |
| WALK | 2 | 0 | 采样速度 | 强制为零 |
| SQUAT_WALK | 3 | 非默认高度 | 采样速度 | 强制为零 |

模式编号、接触标志、实际骨盆位置和目标误差均不作为额外输入。静止模式的零 XY 偏置仍表示骨盆居中任务，不是关闭跟踪。

当前人工指令接口以 `abs(h_offset)>1e-3` 和任一 `abs(vx,vy,wz)>1e-3` 推导模式；不会自动按训练采样包络裁剪前四维指令，部署指令管理器需要自己限制验证范围。

### 8.3 骨盆水平偏置参考系及限速

参考 link 为 `left_ankle_roll_link`、`right_ankle_roll_link`，取二者 **link 原点**的中点为原点，并非足底几何中心或压力中心。X 方向取双足 link 前向轴投影到水平面的平均方向，Y 向左；相反朝向退化时用根节点 yaw 兜底。参考系随当前双足几何更新，不按接触力选择支撑脚。

静止模式的原始偏置目标范围为每轴 `[-0.03,0.03] m`。策略观测读取二维限速后的有效值 `p_eff`：

```text
delta = p_goal - p_eff
p_eff += delta * min(1, 0.02 * T_policy / max(norm(delta), 1e-8))
```

限速为二维向量长度的 `0.02 m/s`，不是每轴各 `0.02 m/s`。行走时原始和有效偏置清零。部署命令管理器应区分原始目标与有效输入；读取观测不能再次推进限速器。

### 8.4 训练范围与已核对 checkpoint 范围

| 分量 | 当前 Phase 1 初始配置 | 已核对 `model_9999.pt` 实际课程范围 |
| --- | --- | --- |
| `h_offset` / m | `[-0.15,0.02]` | 约 `[-0.41,0.05]` |
| `vx` / m/s | `[-0.3,0.6]` | `[-0.5,1.0]` |
| `vy` / m/s | `[-0.15,0.15]` | `[-0.2,0.2]` |
| `wz` / rad/s | `[-0.8,0.8]` | `[-0.8,0.8]` |
| `pelvis_dx,dy` / m | 各 `[-0.03,0.03]` | 各 `[-0.03,0.03]` |
| 模式权重 | `(0.4,0.4,0.2,0)` | `(0.25,0.25,0.25,0.25)` |

这些是训练采样范围，不是已验证的部署能力包络，也不是部署环境物理参数。当前样本为 Phase 1，不能据此声称经过 Phase 2/3 的全部扰动训练。

当前代码还限制蹲走组合，不能将表中各轴范围任意做笛卡尔组合：

- SQUAT 高度每 2–4 s 重采样，范围为完整高度区间。
- SQUAT_WALK 高度只在回合开始采样，最深为 `0.8*h_min`，回合内保持不变；当前样本对应约 `-0.328 m`。
- 速度仅在回合开始采样；WALK/SQUAT_WALK 回合内保持，不随中途重采样变化。
- 定义深度比 `r=clamp(h_offset/h_min,0,1)`。`r` 从 0 到 0.8 时，`vx` 范围按比例线性缩小，正向上限逐渐到 `0.2 m/s`；`wz` 同理，其绝对上限逐渐到 `0.3 rad/s`。负向边界按相同比例缩放，`vy` 不缩放；超过该深度带后前向和偏航采样比例为零。
- 静止模式的骨盆偏置沿重采样时钟更新，20% 概率取零。

部署不需要复刻随机采样器，但应保留指令含义，并从已验证的温和组合开始；快速变速和任意深蹲行走组合是额外的部署验证项。

## 9. 部署适配层及控制循环

### 9.1 必须显式配置的内容

| 模块 | 适配职责 |
| --- | --- |
| 模型加载 | TorchScript 或 ONNX 二选一；检查 dtype、shape、文件 hash 和归一化方式 |
| 状态读取 | 同一仿真时刻的关节、根姿态、角速度、根刚体质心线速度 |
| 坐标映射 | 全身 43 项、动作 12 项、上肢目标 17 项的名称映射、轴方向、零点和缺失关节处理 |
| 指令管理 | 六维输入、模式规则、骨盆偏置限速、最终上肢目标可见性 |
| 观测管理 | 相对量、字段切片、5 帧历史、首次填充和重置 |
| 目标输出 | `q0_policy+0.5*a` 后转换到部署关节坐标 |
| 底层控制 | 部署自己的位置/速度/力矩接口、增益、限位和更新频率；记录与训练的差异 |
| 日志 | 状态时间戳、指令、650 维观测、原始动作、目标角、实际角速度及限幅事件 |

建议将策略协议常量与部署配置分成独立文件。加载策略不应改写部署模型的质量、COM、惯量、碰撞、驱动增益或仿真器参数。

### 9.2 控制循环示意

下列是适配层语义示意，不是项目中已有的可执行部署入口。首次初始化按第 4.3 节处理。

```text
在每个策略时刻 t（仿真时间间隔 1/60 s）：
    读取同步的部署状态，并转换到策略坐标和关节顺序
    获取本时刻有效任务指令及可见上肢最终目标
    计算各观测项；actions 项使用 a_prev
    每项历史追加一次，按 term 顺序拼成 float32 [1,650]
    a = policy(obs)
    检查输出形状为 [1,12] 且数值有限
    q_target_policy = q0_policy_lower + 0.5*a
    将目标映射到部署关节，由部署控制器保持至下个策略时刻
    a_prev = a 的副本

在各部署底层控制/物理时刻：
    使用部署控制器和当前状态执行保持中的关节目标
    按部署上肢规划器推进独立的位置/速度参考
    推进仿真，不额外追加策略历史
```

当前训练中任务命令有自己的更新钩子，上肢目标有自己的排队规则。离线逐步对照时，应复用相同的有效命令序列和采样时间戳，不能由两边各自独立采样后比较输出。

### 9.3 Unitree_Mujoco 的具体对照

本次静态核对路径为本机 `D:/Code/unitree_mujoco`；以下不是对所有版本的承诺。

| 文件/接口 | 本机核对结果 | 部署需要处理 |
| --- | --- | --- |
| `simulate_python/config.py` | 默认 `ROBOT="go2"`，`SIMULATE_DT=0.005` | 显式选择 G1 与目标场景，并独立安排 60 Hz 策略周期 |
| `unitree_robots/g1/g1_29dof.xml` | 29 个 motor，左腿 6 个后接右腿 6 个；没有 14 个手指关节 | 按第 5、6 节映射，记录缺失手指假设和动力学差异 |
| `simulate_python/unitree_sdk2py_bridge.py` | `LowCmdHandler` 中计算 `tau_ff + kp*(q_des-q) + kd*(dq_des-dq)` 后写 `ctrl` | 目标角通过 `q` 字段发送，不能将网络输出直接写 `ctrl`；部署增益独立配置 |
| 同一 Python 桥 | PD 在 LowCmd 回调中计算，不是在每个物理步自动重算 | 若只按 60 Hz 发包，会改变底层力矩刷新方式；单独决定并记录控制频率和时序 |
| `frame_vel` / 高层速度消息 | 来自 IMU site 的 `framelinvel` | 核对世界/基座表达与测量点，转换为根刚体质心速度 |
| IMU 四元数 | MuJoCo `framequat` | 转换分量顺序和外参，再构造重力方向与基座角速度 |
| motor `ctrlrange` | 固定力矩限制；上肢部分上限小于训练配置 | 保留部署模型配置，记录限幅及其对上肢执行的影响 |

也可以在单进程 MuJoCo 中直接读取状态、执行推理和控制，不一定经过 DDS；两条路线必须遵守同一策略协议。DDS 路线还应记录消息频率、时戳、延迟及过期状态处理。现有示例桥并不自动完成本策略的 650 维观测构造。

## 10. 训练环境参考参数：不自动应用于部署

以下内容用于训练分布归档和误差解释。它们不应由模型加载器写入部署模型，差异本身也不应在协议检查阶段被误报为网络维度不兼容。

### 10.1 机器人与场景

| 项目 | 当前训练配置 |
| --- | --- |
| 资产 | `${ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd`；已核对样本保存的地址属于 Isaac 6.2 资产 |
| 关节 | 当前解析为 43 个，含 14 个手指关节；配置变量虽叫 `G1_29DOF_CFG`，不代表观测只有 29 关节 |
| 浮动基座 | `fix_root_link=False` |
| 初始根位置 | `(0,0,0.75) m`，相对子环境原点 |
| 初始根旋转 | `(0,0,0.7071,0.7071)`，当前 `xyzw` 约定；此全局初始朝向不是部署必须沿用的值 |
| 自碰撞 | 训练配置关闭 |
| 软关节位置限位系数 | 0.9，用于训练相关软限位语义，不是部署限位的覆盖值 |
| PhysX 求解迭代 | position=8，velocity=4 |
| 机器人刚体线/角阻尼 | 配置为 0；不等同于关节 PD 阻尼 |
| 地形 | 基础训练为平地 |
| 质量、惯量、碰撞几何、各关节硬限位 | 来自实际 USD 资产；Python 配置和网络文件不包含完整展开值 |

若需要完整机器人参数审计，应另行从确切训练资产版本提取每个 link 的质量、COM、惯量、碰撞及关节轴/限位。本次未加载 USD 进行完整数值提取，不能把目标 MuJoCo XML 的参数误标为训练参数。

### 10.2 执行器

下表角度关节 `Kp` 单位为 Nm/rad、`Kd` 为 Nm·s/rad，力矩为 Nm、速度为 rad/s，armature 为转动惯量单位。数值是训练配置，不是部署推荐值。

| 关节组 | 训练执行器 | Kp | Kd | 配置力矩限值 | 配置速度限值 | armature |
| --- | --- | --- | --- | --- | --- | --- |
| 髋 pitch/roll/yaw | DC motor | 100 | 2.5 | 88 | 32 | 0.03 |
| 膝 | DC motor | 200 | 5.0 | 139 | 20 | 0.03 |
| 踝 pitch | DC motor | 20 | 0.2 | 50 | 37 | 0.03 |
| 踝 roll | DC motor | 20 | 0.1 | 50 | 37 | 0.03 |
| 腰 yaw | 隐式 PD | 5000 | 5 | 88 | 32 | 0.001 |
| 腰 roll/pitch | 隐式 PD | 5000 | 5 | 50 | 37 | 0.001 |
| 肩、肘、腕 | 隐式 PD | 3000 | 10 | 300 | 100 | 0.001 |
| 手指 | 隐式 PD | 20 | 2 | 300 | 100 | 0.001 |

训练 DC 电机 `saturation_effort`：髋/膝为 180 Nm，踝为 80 Nm。实际力矩不是只有固定对称限幅：先计算 PD 力矩，再结合实测转速、电机速度限值及饱和力矩求出可用力矩区间。`saturation_effort` 也不等同于表中的力矩限值。

若部署使用普通 PD 加固定限矩，应将其明确记录为执行器模型差异；不能仅凭 Kp/Kd 相同就宣称控制动力学相同。隐式 PD 与显式计算 PD 在离散求解中也可能不同，尤其是这里较高的训练上肢增益。

### 10.3 训练随机化与扰动

| 阶段 | 当前训练代码配置 |
| --- | --- |
| Phase 1 | 静摩擦 `[0.6,1.0]`，动摩擦 `[0.5,0.9]`，恢复系数 0；上肢目标扰动与课程 |
| Phase 2 | 静摩擦 `[0.5,1.5]`，动摩擦 `[0.4,1.2]`，恢复系数 `[0,0.05]`；link 质量乘 `[0.9,1.1]`；`torso_link` COM 的 X/Y 偏置 ±0.06 m、Z ±0.03 m；手掌附加质量 `[0,0.3] kg`，踝附加 `[0,0.2] kg`；腿部 Kp/Kd 乘 `[0.9,1.1]` |
| Phase 3 | 继承 Phase 2，另每 4–6 s 对根状态施加 X/Y 速度范围 ±0.7 m/s 的扰动 |

上肢扰动间隔初始为 1.5–3 s；臂组幅度初始 fraction=0.2，腰组=0.1。实际课程幅度从 checkpoint 读取，已核对样本保存的臂组 fraction 约为 0.73、腰组为 0.1；fraction 是关节目标采样参数，不是 rad。

训练阶段、运行时覆盖和课程状态决定实际经历的分布。部署不应自动施加这些随机化，也不应仅因代码存在 Phase 3 就宣称某个 Phase 1 模型已经接受相应训练。

## 11. 迁移包与验收记录

以下文件组织为建议，当前项目没有生成这套部署包的专用工具：

```text
sim2sim_bundle/
  model/
    policy.pt                         # 或 ONNX 及其 data；也可同时归档
    policy.onnx
    policy.onnx.data
  provenance/
    model_9999.pt                      # 可选归档，推理不加载
    env.yaml
    agent.yaml
    run.json
    source_revision_and_diff.txt
    hashes.txt
  protocol/
    SIM2SIM_LOWER_LIMB.md
    joint_mapping.yaml                # 三类顺序、轴方向、零点、q0 和缺失项
  deployment/
    target_model_reference.txt        # 部署模型版本和 hash
    controller_and_timing.yaml        # 部署独立参数，不从训练配置隐式覆盖
  validation/
    observation_action_samples.npz    # 对齐用状态、指令、历史、动作和目标
    results.md
```

建议按以下顺序验收，分别记录通过范围，避免将“成功加载”视为“闭环迁移成功”：

1. **文件与接口**：模型来源/hash、输入输出名称及 shape、ONNX 外部权重、归一化方式、软件版本。
2. **关节与坐标**：43/12/17 列表逐项校验；单关节扰动检查对应输入/输出，确认左右腿、轴方向、零点及浮动基座地址。
3. **观测与历史**：给每个字段、时刻、关节不同的已知数值，逐切片对照 650 维向量；核对首次填充、一次追加、重置和上一动作。
4. **同输入推理**：同一 650 维向量比较 TorchScript 与所选推理后端动作；按实际后端浮点误差设置并记录容差。
5. **动作目标**：核对零动作恢复 `q0_policy_lower`，核对 `0.5*a`、坐标反变换与 motor 映射；记录部署限位/限矩是否触发。
6. **控制时序**：记录仿真时钟、60 Hz 调用、历史采样频率、底层控制刷新和消息延迟。
7. **闭环场景**：固定上肢下站立 → 温和下蹲 → 低速行走 → 蹲走组合 → 上肢运动及部署扰动。每步记录存活、倾角、高度/速度误差、关节目标跟踪和限幅情况。

开始闭环前仍需准备：确切部署模型版本、实际关节坐标/传感器外参、部署控制器配置、训练资产展开参数（若需要动力学审计）及可重复的对齐采样数据。本文未产生这些运行数据，也未验证 MuJoCo 闭环稳定性。

## 12. 代码依据与复核入口

| 协议内容 | 当前项目入口 |
| --- | --- |
| 观测项、动作、时间、初始指令范围 | [env_cfg.py](../src/g1/tasks/lower_limb_agent/config/g1/env_cfg.py) |
| 指令切片和上肢相对目标 | [mdp/observations.py](../src/g1/tasks/lower_limb_agent/mdp/observations.py) |
| 下肢动作继承与上肢推进 | [mdp/actions.py](../src/g1/tasks/lower_limb_agent/mdp/actions.py) |
| 指令语义、人工模式、骨盆参考系、限速和采样包络 | [commands.py](../src/g1/tasks/lower_limb_agent/mdp/commands/commands.py)、[commands_cfg.py](../src/g1/tasks/lower_limb_agent/mdp/commands/commands_cfg.py) |
| 上肢可见目标、轨迹队列和状态重置 | [motion.py](../src/g1/tasks/lower_limb_agent/motion.py)、[trajectory.py](../src/g1/tasks/lower_limb_agent/trajectory.py)、[env.py](../src/g1/tasks/lower_limb_agent/env.py) |
| 训练资产和执行器参数 | [assets.py](../src/g1/tasks/lower_limb_agent/config/g1/utils/assets.py) |
| 分阶段训练随机化 | [env_cfg_phase.py](../src/g1/tasks/lower_limb_agent/config/g1/env_cfg_phase.py) |
| Actor 架构及归一化 | [rsl_rl_ppo_cfg.py](../src/g1/tasks/lower_limb_agent/config/g1/agents/rsl_rl_ppo_cfg.py) |
| checkpoint 课程元数据 | [runner.py](../src/g1/tasks/lower_limb_agent/runner.py)、[curriculum_state.py](../src/g1/tasks/lower_limb_agent/curriculum_state.py) |
| 当前完整接口构造与冻结加载校验 | [frozen_lower_actions.py](../src/g1/tasks/upper_limb_agent/mdp/frozen_lower_actions.py)、[frozen_lower.py](../src/g1/tasks/upper_limb_agent/frozen_lower.py) |

项目依赖为本地 editable IsaacLab，核对上游实现时还需检查同版本文件：

- `isaaclab/managers/observation_manager.py`、`manager_term_cfg.py`：历史覆盖、展平和拼接。
- `isaaclab/utils/buffers/circular_buffer.py`：旧到新及首次采样填充。
- `isaaclab/envs/mdp/observations.py`、`actions/joint_actions.py`：状态量、上一动作和位置目标转换。
- `isaaclab/assets/articulation/base_articulation_data.py`：根线速度的 COM 别名。
- `isaaclab_physx/assets/articulation/articulation.py`、`articulation_data.py`：默认关节位置初始化及重力方向。
- `isaaclab/actuators/actuator_pd.py`：DC 电机限矩和 PD 计算。
- `isaaclab_rl/entrypoints/backends/play_rsl_rl.py`、`rsl_rl/runners/on_policy_runner.py`、`rsl_rl/models/mlp_model.py`：播放导出、文件格式及推理封装。

相关项目说明：[课程检查点](CURRICULUM_CHECKPOINTS.md)、[上肢渐进扰动](UPPER_BODY_MOTION.md)、[冻结下肢集成](FROZEN_LOWER_STANDING.md)。
