# G1 髋高站立/蹲起任务 · 奖励函数部署说明

> 记录日期：2026-09-06
> 对应配置：`g1/source/g1/g1/tasks/manager_based/g1/g1_env_cfg.py` 中的 `RewardsCfg`
> 自定义函数实现：`g1/source/g1/g1/tasks/manager_based/g1/mdp/rewards.py`

## 一、总体说明

- **任务**：双足机器人指定髋高（命令范围约 0.60 ~ 0.77 m）原地平衡站立与蹲起，零速度指令。
- **在用奖励项**：24 项（另有 1 项 `knee_guidance` 已注释保留，作消融基线）。
- **权重生效机制**：Isaac Lab `RewardManager` 运行时按 `weight × r × dt` 累积，本项目
  `dt = 1/60 s`（`sim.dt=1/120`，`decimation=2`）。官方 HOMIE 在训练初始化时将权重乘策略
  步长 0.02 s——两者时间积分语义等效（逐秒贡献相同），**官方权重数值直接照抄，无需换算**。
- **训练规模**：`num_envs=4096`，episode 6 s（360 步），PPO（`num_steps_per_env=24`，
  `gamma=0.99`，`lr=3e-4` adaptive）。

### 来源图例

| 标记 | 含义 |
|---|---|
| 【官方】 | 源自 HOMIE 官方实现（`refer_rewards.py`），本项目在 `mdp/rewards.py` 中按官方公式复刻，权重照抄（有改动时在表中注明） |
| 【内置】 | 直接使用 Isaac Lab `isaaclab.envs.mdp` 内置函数，权重取官方对应项数值 |
| 【自研】 | 本项目自行设计，官方无对应项 |

## 二、奖励项明细

### 1. 生存与失败信号

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `alive` | `mdp.is_alive` | 【内置】 | **+0.5** | 未终止时每步 +0.5，鼓励尽量长时间不摔倒 |
| `terminating` | `mdp.is_terminated` | 【内置】 | **-200.0** | 触发终止条件（摔倒/非法接触等）时一次性重罚 |

### 2. 任务奖励（髋高追踪 + 静止约束）

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `track_pelvis_height` | `track_pelvis_height_exp` | 【官方】`tracking_base_height` | **+2.0** | 主奖励：髋高对指令的跟踪，`r = exp(-4·\|e\|)`（官方 L1 指数核，零点梯度恒为 4，保留精细到位信号）。差异：官方相对脚底测量，本项目参考系不变、用世界系骨盆高度 |
| `zero_velocity` | `zero_velocity_exp` | 【自研】 | **+1.0** | 静止零速度约束：`r = exp(-4vx²) + exp(-4vy²) + exp(-4ω_yaw²)`，静止时满分 3.0，`std=0.25` |

### 3. 下肢协同约束（2D 协同流形）

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `leg_synergy` | `leg_synergy_manifold_exp` | 【自研】数据驱动 PCA | **+0.75** | 双腿矢状面 hip/knee/ankle 三 Pitch 关节按全行程归一（相对默认位，防膝大变幅掩盖踝微调），惩罚偏离 2D 协同平面：`r = exp(-‖d⊥‖²/σ²) - 1 ∈ [-1, 0]`（返回非正值，正权重挂载）。基向量为重定向蹲起数据、以默认位 q0 为锚点的非中心化 SVD 拟合（解释 99.7% 方差）：PC1 折叠协同 `[-0.371, 0.807, -0.459]`、PC2 次级补偿模态 `[-0.548, 0.209, 0.810]`，天然内含 G1 关节符号约定（蹲下时髋/踝为负、膝为正）。`sigma=0.2` 为流形带宽（残差达 σ 时惩罚至满值 63%，≈膝关节 33° 容差）；六关节顺序由 `preserve_order=True` 按名解析保证。替代 `knee_guidance`（后者注释保留作消融基线） |
| `hip_default_deviation` | `standing_joint_default_deviation_l2` | 【官方】`deviation_hip_joint` | **-0.5**（官方 -0.2） | 髋 yaw/roll 偏离默认位平方惩罚，命令高度 ≥ 0.735 m 门控（深蹲区间放行屈髋）；默认位从资产动态读取 |
| `ankle_default_deviation` | `standing_joint_default_deviation_l2` | 【官方】`deviation_ankle_joint` | **-0.5** | 踝 pitch/roll 同上；注意 ankle_pitch 默认位为 -0.2，不可用固定 target=0 |

### 4. 躯干姿态与速度稳定性

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `flat_orientation` | `mdp.flat_orientation_l2` | 【内置】官方 `orientation` | **-1.5** | 惩罚重力在躯干 x/y 轴投影的平方和，保持直立不翻转；作用连杆 `torso_link` + `pelvis` |
| `vertical_bounce` | `mdp.lin_vel_z_l2` | 【内置】官方 `lin_vel_z` | **-0.5** | 惩罚 Z 轴线速度平方，抑制高度调整时的剧烈上下颠簸 |
| `tilt_wobble` | `mdp.ang_vel_xy_l2` | 【内置】官方 `ang_vel_xy` | **-0.025** | 惩罚 Roll/Pitch 角速度平方，防止躯干摇晃 |

### 5. 足端约束（贴地、防滑、平行、接触力、离地）

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `feet_ground_parallel` | `feet_ground_parallel_var` | 【官方】`feet_ground_parallel` | **-50.0**（官方 -2.0） | 脚掌贴地：足底四角点高度方差求和（`foot_length=0.18, foot_width=0.08`），仅持续接触 ≥ 3·dt 的脚参与（官方门控，避免抬脚瞬态误罚）。实现量级比官方小（约 1e-3 级），故权重相应放大 |
| `feet_slip` | `mdp.feet_slide` | 【内置】官方 `feet_slip` | **-0.3**（官方 -0.25） | 触地期间惩罚足端水平切向速度（防滑） |
| `feet_parallel` | `feet_parallel_var` | 【官方】 | **-3.0** | 双足平行：左右脚三点（跟/中/尖，沿足长间隔 0.09 m 合成）对应点间距集合的方差（unbiased），仅命令高度 ≥ 0.735 m 生效（深蹲区间双脚布局本就不同，不约束） |
| `feet_contact_forces` | `mdp.contact_forces` | 【内置】官方 `feet_contact_forces` | **-2.5e-4** | 足端接触力峰值惩罚 `ReLU(‖F‖ - 400 N)`，保护踝关节与真机结构 |
| `no_fly` | `no_fly` | 【官方】 | **+0.75** | 至少一脚保持接触（垂直接触力 > 0.5 N）时奖励 1.0，禁止双脚同时腾空 |
| `stand_still` | `stand_still` | 【官方】（已去门控） | **-0.15** | 惩罚离地脚数量（垂直接触力 < 0.1 N 视为离地），抑制原地踏步/离地蠕动 |

### 6. 横向安全间距

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `feet_lateral` | `feet_lateral_separation` | 【官方】`feet_distance_lateral` | **+0.5** | 双脚横向间距双边钳制在 [0.2, 0.35] m：下界防并拢（常开），上界防越分越开（仅 ≥ 0.735 m 门控，深蹲允许宽站姿）；公式整体非正，正权重挂载为惩罚语义 |
| `knee_lateral` | `knee_lateral_separation` | 【官方】`knee_distance_lateral` | **+1.0** | 双膝横向间距同上双边形式（作用 `left/right_knee_link`），防膝内扣（X 型腿）与过度外分 |

### 7. 动作平滑性（动作空间）

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `action_rate` | `mdp.action_rate_l2` | 【内置】官方 `action_rate` | **-0.01** | 一阶平滑：惩罚相邻控制帧动作差平方 `‖a_t - a_{t-1}‖²` |
| `action_smoothness` | `action_acc_l2` | 【自研】对应官方 `smoothness` | **-0.05** | 二阶平滑：惩罚动作加速度 `‖a_t - 2a_{t-1} + a_{t-2}‖²`，抑制抖动与机械磨损（内部维护 a_{t-2} 缓存） |

### 8. 能耗约束（关节空间）

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `dof_torques` | `joint_torques_l2_normalized` | 【官方】`torques`（刚度归一实现） | **-2.5e-6** | 扭矩惩罚 `Σ(τ_i/k_i)²`，按刚度归一使髋/膝/踝不同量级可比；仅下肢受控关节。注：踝实际刚度仅 20，直接归一会使踝扭矩主导惩罚项，故踝采用**等效刚度 100** 与髋对齐量级（髋 100 / 膝 200） |

### 9. 关节限位与扭矩超限（安全硬边界）

| 配置项 | 函数 | 来源 | 权重 | 中文简介 |
|---|---|---|---|---|
| `dof_pos_limits` | `mdp.joint_pos_limits` | 【内置】官方 `dof_pos_limits` | **-2.0** | 惩罚超出软限位（硬限位 × 0.9）的关节位置越界量，作用腿部全部受控关节 |
| `dof_torque_limits` | `joint_torque_limits` | 【自研】对应官方 `torque_limits` | **-0.1** | 惩罚驱动指令超出额定扭矩 `ReLU(\|τ\| - τ_max)`；仅显式执行器（DCMotor 腿/踝）生效，腰/臂/手隐式执行器限位为 inf 自动跳过 |
| `dof_vel_limits` | `mdp.joint_vel_limits` | 【内置】官方 `dof_vel_limits` | **-2e-3** | 惩罚关节速度超限（`soft_ratio=1.0`，超限量裁剪至 1 防爆炸），轻度约束 |

## 三、消融基线（已注释，未部署）

| 配置项 | 函数 | 来源 | 权重 | 说明 |
|---|---|---|---|---|
| `knee_guidance`（注释） | `knee_guidance_l1` | 【官方】`deviation_knee_joint` 复刻 | -0.75 | `r = Σ\|e·(q_norm - 0.5)\|`：高度误差加权约束膝关节偏离行程中点，防"跪式下蹲"病态解，高度到位时约束自动消失。已被 `leg_synergy`（2D 协同流形）替代；做 A/B 消融实验时取消注释、并同时注释 `leg_synergy`（两者功能重叠，不可同时挂载） |

## 四、与官方权重的差异汇总

| 项 | 官方值 | 本项目值 | 原因 |
|---|---|---|---|
| `hip_default_deviation` | -0.2 | **-0.5** | 手动加强髋 yaw/roll 回中约束（实测站姿髋部漂移） |
| `feet_slip` | -0.25 | **-0.3** | 手动加强防滑约束 |
| `feet_ground_parallel` | -2.0 | **-50.0** | 实现量级不同：本项目四角点高度方差（~1e-3 级）远小于官方 `_get_feet_heights` 口径，按等效惩罚强度放大 |
| `track_pelvis_height` | 相对脚底测高 | 世界系骨盆高度 | 参考系保持本项目原有约定（用户指定不变更） |
| `stand_still` | 带站立门控 | 已去门控 | 全高度区间抑制蠕动 |

**官方有、本项目未启用的项**：`dof_acc`（-2.5e-7，关节空间加速度正则——已由动作空间二阶平滑 `action_smoothness` 替代）、`joint_tracking_error`、`joint_power`、`action_vanish`、`contact_momentum`、`dof_vel`（均为官方 CONSIDER 级可选项，视后续训练表现按需开启）。

## 五、单步奖励预算（理想站姿，权重 × r × dt 口径）

| 项 | 单步贡献 |
|---|---|
| `alive` | +0.0083 |
| `track_pelvis_height`（e≈0） | +0.033 |
| `zero_velocity`（满分 3.0） | +0.050 |
| `no_fly` | +0.0125 |
| 合计正向基底 | **≈ +0.10 / 步** |

惩罚项在理想站姿下均接近 0（`leg_synergy` 数据残差地板 ≈ -0.0164 raw → 每步约 -2e-4）。
终止惩罚 -200 在 `gamma=0.99` 下对存活回报的威慑比约 5×，量级合理。
