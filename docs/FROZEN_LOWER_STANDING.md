# Upper与冻结Lower控制

`Template-G1-Upper-Mobile-v0` 使用自由底座G1：Upper控制17个腰臂关节，并选择六维Lower指令。
Lower负责双腿执行和反馈平衡，权重冻结，不进入Upper的PPO优化器。
固定底座对照任务为 `Template-G1-Upper-v0`。

## 当前接口

网络输出23维：`[vx, vy, yaw, height, pelvis_x, pelvis_y, upper_joints × 17]`。
Lower桥接指令顺序为 `[height, vx, vy, yaw, pelvis_x, pelvis_y]`；其历史观测650维，腿部动作12维。
Upper观测109维：原93维上肢观测、10维基座状态、6维实际生效的Lower指令。

速度采用基座坐标系，高度相对默认根高度0.75 m；骨盆XY偏移相对于双踝中点、双脚平均水平朝向，
它描述骨盆位置，未计算整机质量加权质心。

默认冻结模型：

```text
logs/rsl_rl/g1_flat/2026-11-11_21-45-46_lower_base_p3/model_19997.pt
```

同目录需保留 `params/env.yaml` 和 `params/agent.yaml`。模型加载校验网络配置、历史长度、
观测项顺序、动作关节、缩放、控制频率和final_goal预览语义。

能力边界、渐进骨盆参考、深蹲速度包络、联合位姿EMA、命令奖励尺度及课程状态恢复，
统一见 [Upper范围课程说明](UPPER_RANGE_CURRICULUM.md)。当前Mobile默认开启该课程。

## 快速使用

```powershell
uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --num_envs 4096 --max_iterations 3000 --viz none physics=isaacsim_physx

uv run python -m g1.rl play --task Template-G1-Upper-Mobile-v0 --checkpoint "<23维Upper检查点>" --num_envs 16 --real-time --viz kit physics=isaacsim_physx
```

官方zero_agent可用：

```powershell
uv run isaaclab zero_agent --task Template-G1-Upper-Mobile-v0 --num_envs 4 --viz kit physics=isaacsim_physx
```

零动作对应零速度、默认站高和零骨盆偏移；Upper腰臂保持默认关节目标，Lower仍持续反馈维持平衡。
用训练模型做零调用消融，需将三个动作项的enabled均关闭：

```text
env.actions.lower_command.enabled=false
env.actions.height_offset.enabled=false
env.actions.pelvis_offset.enabled=false
```

更换冻结模型使用 `env.actions.joint_pos.checkpoint_path=<model.pt>`。
恢复Upper时，冻结模型哈希和接口必须与保存记录一致。

## 执行、观测和奖励

动作项按配置顺序处理：速度→高度→骨盆偏移→腰臂动作／Lower推理。
六维原始动作直接按当前回合能力范围解码，经过训练包络和骨盆参考处理后，再进入本步Lower观测。
Lower同时接收本步腰臂关节目标预览，自己的上一动作与观测历史独立维护。
60 Hz控制、120 Hz物理下，每控制步推理一次，两次物理子步持续施加同一组关节目标。

`lower_command_rate`读取六维有效命令的固定尺度差分平方和，幅值与raw越界惩罚继续只约束速度三维。
速度幅值按每个子环境本回合的正／负向范围归一化，权重−0.05；未启用课程时使用配置的速度范围。
上肢`action_rate`只读17维腰臂动作。关节加速度覆盖全部物理子步，腕部加速度是骨盆坐标系中的相对
线加速度并覆盖全部子步，`base_vertical_acc`读取骨盆link原点的物理世界Z加速度。
Mobile的Dwell奖励使用同一link原点的世界Z线速度，对最近0.1秒全部物理样本计算RMS。
窗口满后结合当前10 mm／0.04 rad腕部位姿门槛发放 `+0.2` 权重奖励；
实现复用基座子步采样和单次收尾缓存。参数与日志见 [Dwell说明](UPPER_RANGE_CURRICULUM.md#dwell稳定保持奖励)。
这些统计不修改网络输出。局部重置清理本环境的动作历史、Lower历史、PD目标和课程回合统计。

世界腕部目标在生成后保持固定，Upper观测实时转换至骨盆坐标；闲置手继续受到默认关节角约束。
[空间Q奖励](WORKSPACE_QUALITY.md)默认权重0，诊断保留。

## 保存和兼容性

Upper检查点记录控制布局、奖励参考尺度、课程状态、冻结Lower哈希及Q地图身份。
运行目录归档：

```text
artifacts/frozen_lower_model.pt
artifacts/params/env.yaml
artifacts/params/agent.yaml
artifacts/workspace_quality.npz
```

搬移后可将checkpoint_path指向归档Lower。只导出的Upper网络不包含Lower及动作处理；
部署需同时保留两层模型、历史观测、物理单位映射和有效指令处理。

H5C及此前21动作／107观测Mobile模型与新23／109接口不同，直接加载会在恢复权重前给出错误。
当前控制契约为schema 4；A1的schema 3具有相同维数但不同映射，当前项目入口同样拒绝加载。
新接口从头训练；固定底座17／93接口保持原样。`teleop/eval`目前仍只支持Lower任务。

## 验证入口

```powershell
$env:G1_TRAIN_PLAY_SMOKE="1"
uv run python -m pytest tests/test_task_runners_physics.py -k "upper_g1 or upper_mobile" -q

$env:G1_FROZEN_STAND_SMOKE="1"
uv run python -m pytest tests/test_frozen_standing_physics.py -q
```

第一组覆盖真实PhysX短训、保存、Mobile续训、play和导出；第二组检查真实Lower观测逐项一致、
冻结权重不变、局部重置，以及零命令站立／平滑上肢运动。完整课程收敛与带任务稳定性仍由正式训练验证。
