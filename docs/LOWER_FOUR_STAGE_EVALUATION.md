# 下肢四阶段自动测试

入口为 `python -m g1.rl eval --suite g1_lower_four_stage`。这是独立的 schema 2 测试协议；原 `g1_core`、`g1_robustness` 沿用 schema 1。

仅测试抗扰时使用 [独立推扰 suite](LOWER_PUSH_EVALUATION.md)：`--suite g1_lower_push`。它复用本协议的抗扰采集与统计，默认推扰范围为 X、Y 各 ±0.70 m/s。

## 运行

从项目根目录运行，替换检查点路径和输出目录：

```powershell
uv run --extra isaacsim python -m g1.rl eval --suite g1_lower_four_stage --checkpoints "model=logs/rsl_rl/g1_flat/<run>/model_9999.pt" --num_envs 128 --repetitions 10 --episode-length-s 6 --height-range -0.15 0.02 --vx-range -0.3 0.6 --vy-range -0.15 0.15 --wz-range -0.8 0.8 --push-heights -0.15 -0.075 0 --seeds 101 --output outputs/eval/lower_four_stage
```

- 四个阶段固定执行，每阶段重复 R 个回合；每回合运行 T 秒，M 个环境同时执行。一种动作方式、一个随机种子的预算为 **4 × R × M 个环境回合**。
- M 至少为 3，且不小于推扰高度数量。默认 128 个环境、每阶段 10 个回合、每回合 6 秒、种子 101。
- 指令范围通过上述参数设置，也可复制内置 YAML，用文件路径传给 `--suite`。高度参数均为相对默认站高的偏移，单位 m。
- `--height-range` 会自动在新范围内生成三个推扰高度；同传 `--push-heights` 时使用显式高度。
- 使用策略均值动作；可通过 `--policy-output sample/both` 更改。多模型比较沿用 `--checkpoints` 的标签、路径和通配符用法。
- 增加 `--plan-only` 只生成 manifest；`--resume <输出目录>` 继续；`--resume <输出目录> --report-only` 重新生成报告。
- 训练课程、观测噪声、物理参数随机化和上肢运动扰动关闭，骨盆水平偏移目标为零。模型观测/动作兼容性检查沿用原 eval。

## 四阶段场景

| 阶段 | 场景 | 主指标 |
| --- | --- | --- |
| `walk_axes` | 环境连续划分为直线、转弯、侧向三组；数量尽量均等，余数依次分配。每回合开头采样，在半程再独立采样一次 | `e_walk_percent` |
| `squat_tracking` | 零速度，回合前三分之一保持默认站高，中间三分之一跟踪随机高度，最后三分之一回到默认站高 | `e_squat_m` |
| `squat_walk_tracking` | 回合开头采样高度和 vx/vy/wz，整回合保持 | `e_squat_walk_m` |
| `squat_push` | 环境按高度列表连续分组，保持固定高度及零速度，每回合计划两次根节点推扰 | 按高度输出失败率、高度 RMSE、峰值偏差及恢复情况 |

直线组仅 vx 随机；转弯组 vx、wz 随机；侧向组仅 vy 随机，其余速度通道置零。行走阶段保持默认站高。半程取 `floor(总控制步数/2)`；默认 60 Hz。蹲起两次切换同样取整到控制步。

蹲走默认 `squat_walk_use_envelope: false`，严格使用输入的高度与速度范围。可在 YAML 中设为 true，沿用训练的高度—速度包络；此时高度采样下界为 `depth_frac × 高度下界`，vx 和 wz 随深度缩放，vy 不缩放。每回合实际指令计划保存到原始结果。

## 指标口径

每步使用实际生效的目标，速度使用机体坐标系，基座高度使用骨盆根刚体的世界 Z。高度目标为环境原点 Z + 默认根高度 + 高度偏移。

每回合保存四个通道的**平方误差和与实际控制步数**，汇总为 `sqrt(所有回合平方误差和 / 所有回合样本数)`，包含指令切换过渡和终止帧。

行走指标为：

```text
E_walk (%) = 100/3 × (RMSE_straight(vx)/Vx + RMSE_lateral(vy)/Vy + RMSE_turn(wz)/Wz)
```

三个 RMSE 分别只使用各自小组的数据。Vx/Vy/Wz 为配置区间端点的最大绝对值，必须大于零。`group_rmse_vx_mps`、`group_rmse_vy_mps`、`group_rmse_wz_radps` 是总分使用的三个分量；各小组另行保存全部速度通道 RMSE。误差百分比可以超过 100%。

`E_squat` 与 `E_squat-walk` 都是对应阶段的高度 RMSE，单位 m。各阶段还记录失败率、正常完成率、存活比例、样本数和终止原因。提前失败保留终止前全部数据，自动重置后的回合不重复计入。RMSE 应与失败率、实际采样时长一起比较。

多种子时先汇总平方误差和，再计算总体 RMSE；另报各种子指标的样本标准差。缺少指定种子的测试条件列入 `incomplete_groups`，不输出对应统计行。

## 推扰及恢复

默认世界系 `Δvx、Δvy` 各自均匀采样 `[-0.10, 0.10] m/s`，增加到根节点当前速度，保留其余速度分量。这与 Phase 3 初始推扰的实际实现一致。

默认两次推扰窗口为回合时长的 `[25%,40%]` 和 `[60%,75%]`，在窗口内随机选择控制步；两个窗口不重叠，且包含可执行控制步。可通过 YAML 的 `push_windows_fraction`、`push_velocity_range` 修改。

同一种子、同一重复回合的所有高度复用相同推扰时刻与速度增量，不同检查点也复用这些输入。每个环境记录 `push_plan`、实际 `pushes`、计划次数与实际次数。提前失败允许实际次数少于两次。

恢复默认要求高度误差 < 3 cm、绝对 roll/pitch 各 < 0.1 rad、水平速度 < 0.05 m/s，连续保持 0.2 s。参数位于 YAML 的 `recovery`。`recovery_s` 为从推扰到确认满足持续保持条件的时间，包含保持时间；下次推扰或回合结束前未确认恢复时保持 null。

报告输出恢复率、未恢复次数，以及已恢复事件的平均确认时间；通过恢复率核对平均时间所覆盖的事件。高度/姿态峰值覆盖整个回合，包含到达指定蹲高的初始过程。

姿态使用当前 Isaac Lab 的 `xyzw` 四元数顺序。早期记录器误按 `wxyz` 读取，导致 Roll 和恢复判定有误；没有 `attitude_quaternion_order: xyzw` 标记的旧回合在重新生成报告时省略这些指标，`attitude_metrics_valid=0`。高度、速度、失败统计仍有效。旧数据未保存逐步姿态，需重新执行测试才能获得正确的 Roll 和恢复结果。

## 输出与验证

沿用 eval 的 `manifest.json`、`cases/*/result.json`、`effective_env.yaml`、`effective_agent.yaml`、版本化报告和 `report_latest.json`。

CSV/JSON 中 `module` 表示阶段，`mode` 表示 aggregate、行走小组或 height。推扰高度行的 `axis_step` 为高度列表编号，`height_offset_m` 给出实际高度；总体行单独保留。TensorBoard 提供“四阶段指标”、失败率和实际样本等图表。

```powershell
uv run python -m pytest tests/test_eval_four_stage.py -q
$env:G1_EVAL_FOUR_STAGE_PHYSICS_SMOKE = '1'
uv run --extra isaacsim python -m pytest tests/test_eval_four_stage_physics.py -q
```

真实物理链路测试使用合成策略确认编排、采集、推扰和报告，模型性能需使用目标 checkpoint 正式评估。
