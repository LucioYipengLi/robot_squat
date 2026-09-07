# G1 髋高站立/蹲起任务 · 奖励函数部署说明

> 记录日期：2026-09-06
> 对应配置：`g1/source/g1/g1/tasks/manager_based/g1/g1_env_cfg.py` 中的 `RewardsCfg`
> 自定义函数实现：`g1/source/g1/g1/tasks/manager_based/g1/mdp/rewards.py`

## 一、总体说明

- **任务**：双足机器人统一任务指令（四模式 STAND/SQUAT/WALK/SQUAT_WALK）：指定髋高
  （命令范围约 0.60 ~ 0.77 m）平衡站立与蹲起，以及机体系速度跟随行走（vx/vy/ωz）。
- **在用奖励项**：24 项（另有 1 项 `knee_guidance` 已注释保留，作消融基线）。
- **权重生效机制**：Isaac Lab `RewardManager` 运行时按 `weight × r × dt` 累积，本项目
  `dt = 1/60 s`（`sim.dt=1/120`，`decimation=2`）。官方 HOMIE 在训练初始化时将权重乘策略
  步长 0.02 s——两者时间积分语义等效（逐秒贡献相同），**官方权重数值直接照抄，无需换算**。
- **训练规模**：`num_envs=4096`，episode 6 s（360 步），PPO（`num_steps_per_env=24`，
  `gamma=0.99`，`lr=3e-4` adaptive）。

**来源标记说明**：

- `HOMIE`：源自官方 HOMIE 实现，在 `mdp/rewards.py` 中按官方公式复刻，权重照抄（有改动时注明官方原值）
- `IsaacLab`：直接使用 `isaaclab.envs.mdp` 内置函数，权重取官方对应项数值
- `Own`：本项目自行设计，官方无对应项

## 二、奖励项明细（按配置文件顺序）

### 1. alive —— 存活奖励

- 来源：IsaacLab（`mdp.is_alive`）
- 权重：+0.5
- 简介：未终止时每步 +0.5，鼓励机器人尽量长时间不摔倒。

### 2. terminating —— 终止惩罚

- 来源：IsaacLab（`mdp.is_terminated`）
- 权重：-200.0
- 简介：触发终止条件（摔倒/非法接触等）时一次性重罚 -200。

### 3. track_pelvis_height —— 髋高追踪（主奖励）

- 来源：HOMIE（`tracking_base_height`，复刻为 `track_pelvis_height_exp`）
- 权重：+2.0
- 简介：髋高对指令的跟踪，`r = exp(-4·|e|)`（官方 L1 指数核，零点梯度恒为 4，保留精细
  到位信号）。与官方的差异：官方相对脚底测高，本项目参考系不变、用世界系骨盆高度。

### 4. track_velocity —— 速度跟踪（核心任务奖励）

- 来源：Own（`track_velocity_exp`，参照 HOMIE `tracking_x/y/ang_vel` 高斯核）
- 权重：+1.0
- 简介：`r = Σ_i exp(-(v_cmd_i − v_actual_i)²/(2σ²))`，逐轴（vx/vy/ωz）高斯核，满分
  3.0，`std=0.25`。指令取 `task_command` 速度切片 `[:, 1:4]`，误差用机体系
  （`root_lin_vel_b` / `root_ang_vel_b`，与指令自带 metrics 同口径）。**单一未门控项
  统一四模式**：STAND/SQUAT 指令速度为 0 → 退化为静止约束（与旧 `zero_velocity_exp`
  逐位一致）；WALK/SQUAT_WALK → 跟踪非零指令。`zero_velocity_exp`（世界系静止惩罚）
  本体保留作消融回退，已不在配置中挂载。

### 5. leg_synergy —— 下肢 2D 协同流形

- 来源：Own（`leg_synergy_manifold_exp`，数据驱动 PCA）
- 权重：+0.75（函数返回非正值，正权重挂载为惩罚语义）
- 简介：双腿矢状面 hip/knee/ankle 三 Pitch 关节按全行程归一（相对默认位，防膝大变幅
  掩盖踝微调），惩罚偏离 2D 协同平面：`r = exp(-‖d⊥‖²/σ²) - 1 ∈ [-1, 0]`。
  - 基向量来自重定向蹲起数据、以默认位 q0 为锚点的非中心化 SVD 拟合（解释 99.7% 方差）：
    PC1 折叠协同 `[-0.371, 0.807, -0.459]`，PC2 次级补偿模态 `[-0.548, 0.209, 0.810]`，
    天然内含 G1 关节符号约定（蹲下时髋/踝为负、膝为正）；
  - `sigma=0.2` 为流形带宽：残差距离达 σ 时惩罚至满值 63%，≈膝关节 33° 容差；过小会使
    严重病态姿态饱和在 -1 附近梯度变平，训练卡死时可放宽至 0.3~0.5；
  - 六关节顺序由函数内 `preserve_order=True` 按名解析，不依赖资产内部关节序；
  - 替代 `knee_guidance`（后者注释保留作消融基线，见第三节）。

### 6. hip_default_deviation —— 髋关节回默认位

- 来源：HOMIE（`deviation_hip_joint`，复刻为 `standing_joint_default_deviation_l2`）
- 权重：-0.5（官方原值 -0.2，手动加强）
- 简介：髋 yaw/roll 偏离默认位的平方惩罚，命令高度 ≥ 0.735 m 门控（深蹲区间放行屈髋）；
  默认位从资产动态读取。

### 7. ankle_default_deviation —— 踝关节回默认位

- 来源：HOMIE（`deviation_ankle_joint`，同上函数）
- 权重：-0.5（与官方一致）
- 简介：踝 pitch/roll 偏离默认位平方惩罚，同 0.735 m 门控；注意 ankle_pitch 默认位为
  -0.2，不可用固定 target=0。

### 8. flat_orientation —— 躯干朝向惩罚

- 来源：IsaacLab（`mdp.flat_orientation_l2`，官方 `orientation` 对应项）
- 权重：-1.5
- 简介：惩罚重力在躯干 x/y 轴投影的平方和 `r = -||g_x||² - ||g_y||²`，保持直立不翻转；
  作用连杆 `torso_link` + `pelvis`。

### 9. vertical_bounce —— Z 轴线速度惩罚

- 来源：IsaacLab（`mdp.lin_vel_z_l2`，官方 `lin_vel_z`）
- 权重：-0.5
- 简介：`r = -v_z²`，抑制高度调整时的剧烈上下颠簸与冲量冲击。

### 10. tilt_wobble —— XY 角速度惩罚

- 来源：IsaacLab（`mdp.ang_vel_xy_l2`，官方 `ang_vel_xy`）
- 权重：-0.025
- 简介：`r = -||ω_xy||²`，惩罚 Roll/Pitch 方向角速度，防止躯干摇晃。

### 11. feet_ground_parallel —— 脚掌贴地约束

- 来源：HOMIE（`feet_ground_parallel`，复刻为 `feet_ground_parallel_var`）
- 权重：-50.0（官方原值 -2.0）
- 简介：`r = -Σ Var(H_i)`，以足底四角点高度方差近似（`foot_length=0.18, foot_width=0.08`），
  仅持续接触 ≥ 3·dt 的脚参与（官方门控，避免抬脚瞬态误罚）。权重差异原因：本实现量级
  （约 1e-3 级）远小于官方 `_get_feet_heights` 口径，按等效惩罚强度放大。

### 12. feet_slip —— 足部防滑惩罚

- 来源：IsaacLab（`mdp.feet_slide`，官方 `feet_slip`）
- 权重：-0.3（官方原值 -0.25，手动加强）
- 简介：`r = -Σ ||v_i|| · I_contact`，触地期间惩罚足端水平切向速度。

### 13. feet_parallel —— 双足平行约束

- 来源：HOMIE（复刻为 `feet_parallel_var`）
- 权重：-3.0（照抄官方）
- 简介：`r = -Var(D)`，D 为左右脚三点（跟/中/尖，沿足长间隔 0.09 m 合成）对应点间距
  集合，方差取 unbiased；仅命令高度 ≥ 0.735 m 生效（深蹲区间双脚布局本就不同，不约束）。

### 14. feet_contact_forces —— 足端接触力峰值惩罚

- 来源：IsaacLab（`mdp.contact_forces`，官方 `feet_contact_forces`）
- 权重：-2.5e-4（照抄官方）
- 简介：`r = -Σ ReLU(||F|| - 400 N)`，限制足端接触力超限，保护踝关节与真机结构；
  阈值 400 N 与官方 `max_contact_force` 一致。

### 15. no_fly —— 防双脚离地

- 来源：HOMIE（复刻为 `no_fly`）
- 权重：+0.75（照抄官方）
- 简介：至少一脚保持接触（垂直接触力 > 0.5 N）时奖励 1.0，禁止双脚同时腾空，
  原地任务关键项。

### 16. stand_still —— 站立蠕行惩罚

- 来源：HOMIE（复刻为 `stand_still`，已去门控）
- 权重：-0.15（照抄官方）
- 简介：惩罚离地脚数量（垂直接触力 < 0.1 N 视为离地），抑制原地踏步/离地蠕动；
  与官方差异：去掉了站立高度门控，全高度区间生效。

### 17. feet_lateral —— 双脚横向安全距离

- 来源：HOMIE（`feet_distance_lateral`，复刻为 `feet_lateral_separation`）
- 权重：+0.5（照抄官方；公式整体非正，正权重挂载为惩罚语义）
- 简介：双脚横向间距双边钳制在 [0.2, 0.35] m——下界防并拢（常开），上界防越分越开
  （仅 ≥ 0.735 m 门控，深蹲允许宽站姿）。

### 18. knee_lateral —— 双膝横向安全距离

- 来源：HOMIE（`knee_distance_lateral`，复刻为 `knee_lateral_separation`）
- 权重：+1.0（照抄官方）
- 简介：同双边钳制形式作用于膝关节连杆（`left/right_knee_link`），防膝内扣（X 型腿）
  与过度外分。

### 19. action_rate —— 动作变化率惩罚

- 来源：IsaacLab（`mdp.action_rate_l2`，官方 `action_rate`）
- 权重：-0.01
- 简介：`r = -||a_t - a_{t-1}||²`，一阶平滑，惩罚相邻控制帧动作高频跳变。

### 20. action_smoothness —— 动作二阶平滑惩罚

- 来源：Own（`action_acc_l2`，对应官方 `smoothness` 的二阶平滑语义）
- 权重：-0.05
- 简介：`r = -||a_t - 2a_{t-1} + a_{t-2}||²`，约束动作空间加速度，抑制抖动、降低机械
  磨损（内部维护 a_{t-2} 缓存）。

### 21. dof_torques —— 扭矩输出惩罚

- 来源：HOMIE（`torques`，复刻为 `joint_torques_l2_normalized` 刚度归一实现）
- 权重：-2.5e-6（照抄官方）
- 简介：`r = -Σ(τ_i / k_i)²`，按刚度归一使髋/膝/踝不同量级可比；仅下肢受控关节
  （不含腰/臂/手）。注：踝实际刚度仅 20，直接归一会使踝扭矩主导惩罚项（踝 20 Nm → 1 rad²，
  髋 100 Nm 才 1 rad²），故踝采用等效刚度 100 与髋对齐量级（髋 100 / 膝 200）。

### 22. dof_pos_limits —— 关节位置限位惩罚

- 来源：IsaacLab（`mdp.joint_pos_limits`，官方 `dof_pos_limits`）
- 权重：-2.0
- 简介：惩罚超出软限位（硬限位 × 0.9）的关节位置越界量（ReLU 求和），作用于腿部
  全部受控关节。

### 23. dof_torque_limits —— 扭矩超限惩罚

- 来源：Own（`joint_torque_limits`，对应官方 `torque_limits`）
- 权重：-0.1
- 简介：`r = -Σ ReLU(|τ| - τ_max)`，惩罚驱动指令超出额定扭矩；仅对显式执行器
  （DCMotor 腿/踝）有效，腰/臂/手的隐式执行器限位为 inf 会被自动跳过。

### 24. dof_vel_limits —— 关节速度限位惩罚

- 来源：IsaacLab（`mdp.joint_vel_limits`，官方 `dof_vel_limits`）
- 权重：-2e-3
- 简介：`r = -Σ ReLU(|ω| - ω_max·soft_ratio)`，超限量裁剪至 1 防爆炸；`soft_ratio=1.0`，
  轻度约束腿部关节速度接近极限。

## 三、消融基线（已注释，未部署）

### knee_guidance —— 膝关节引导

- 来源：HOMIE（`deviation_knee_joint` 复刻为 `knee_guidance_l1`）
- 权重：-0.75（照抄官方）
- 简介：`r = Σ|e·(q_norm - 0.5)|`，高度误差加权约束膝关节偏离行程中点，防"跪式下蹲"
  等病态解；高度到位（e≈0）时约束自动消失。
- 状态：已被 `leg_synergy`（2D 协同流形）替代，配置中注释保留。做 A/B 消融实验时取消
  注释、并同时注释 `leg_synergy`（两者功能重叠，不可同时挂载）。

## 四、与官方权重的差异汇总

- `hip_default_deviation`：官方 -0.2 → 本项目 **-0.5**。手动加强髋 yaw/roll 回中约束
  （实测站姿髋部漂移）。
- `feet_slip`：官方 -0.25 → 本项目 **-0.3**。手动加强防滑约束。
- `feet_ground_parallel`：官方 -2.0 → 本项目 **-50.0**。实现量级不同：四角点高度方差
  （~1e-3 级）远小于官方 `_get_feet_heights` 口径，按等效惩罚强度放大。
- `track_pelvis_height`：官方相对脚底测高 → 本项目用世界系骨盆高度。参考系保持本项目
  原有约定（指定不变更）。
- `stand_still`：官方带站立门控 → 本项目已去门控，全高度区间抑制蠕动。
- 官方有、本项目未启用的项：`dof_acc`（-2.5e-7，关节空间加速度正则——已由动作空间二阶
  平滑 `action_smoothness` 替代）、`joint_tracking_error`、`joint_power`、`action_vanish`、
  `contact_momentum`、`dof_vel`（均为官方 CONSIDER 级可选项，视后续训练表现按需开启）。
- 行走步态质量项（`feet_air_time` +0.05 / `feet_clearance` -0.25 / `contact_momentum`）：
  行走专属、需 WALK 门控，且参考实现基于 isaacgym（需按 Isaac Lab ContactSensor 重写），
  暂缓；待行走成为主任务时再评估（见主目录 TODO P2c）。

## 五、单步奖励预算（理想站姿，权重 × r × dt 口径）

- `alive`：+0.0083
- `track_pelvis_height`（e≈0）：+0.033
- `track_velocity`（静止满分 3.0）：+0.050
- `no_fly`：+0.0125
- **合计正向基底 ≈ +0.10 / 步**

惩罚项在理想站姿下均接近 0（`leg_synergy` 数据残差地板 ≈ -0.0164 raw → 每步约 -2e-4）。
终止惩罚 -200 在 `gamma=0.99` 下对存活回报的威慑比约 5×，量级合理。
