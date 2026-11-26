# 下肢独立推扰评估

`g1_lower_push` 只执行 `squat_push`，复用四阶段评估抗扰组的用例生成、场景控制器、回合收集和报告。两者在相同参数、种子和环境数下生成相同的推扰计划，保存相同的数据字段。

## 运行

```powershell
uv run --extra isaacsim python -m g1.rl eval --suite g1_lower_push --checkpoints "p3=logs/rsl_rl/g1_flat/2026-11-11_21-45-46_lower_base_p3/model_19997.pt" --num_envs 256 --repetitions 10 --output outputs/eval/lower_push_p3_07
```

默认配置：

| 项目 | 默认值 |
| --- | --- |
| 每回合时长 | 12 秒，控制频率 60 Hz |
| 重复次数 | 10 个完整并行回合 |
| 随机种子 | 101 |
| 高度偏移 | −0.25、−0.075、0 m，按环境连续分组 |
| 速度指令 | vx、vy、wz 全部为零；骨盆水平偏移为零 |
| 推扰次数 | 每个环境每回合计划两次 |
| 推扰速度增量 | 世界系 X、Y 各自均匀采样 [−0.70, +0.70] m/s |
| 推扰时刻 | 分别在回合的 25%～40%、60%～75% 窗口内随机选择控制步 |
| 恢复条件 | 高度误差 < 3 cm，绝对 roll/pitch 各 < 0.1 rad，水平速度 < 0.05 m/s，连续保持 0.2 秒 |

±0.70 m/s 对齐 `2026-11-11_21-45-46_lower_base_p3/model_19997.pt` 保存的最终训练推扰范围。原四阶段 suite 的默认推扰幅度为 ±0.10 m/s。

训练课程、观测噪声、物理参数随机化和上肢运动扰动关闭。推扰给根节点当前水平速度增加增量，保留其他速度分量；高度整回合保持。

一个检查点、一个种子、一种策略输出方式产生 **1 个评估单元**，回合预算为 `num_envs × repetitions`。上述示例为 2560 个环境回合、5120 次计划推扰。提前失败回合保留终止前的数据，实际推扰次数可能少于计划次数。

环境数至少覆盖所有高度。例如三个高度至少需要三个环境；只测一个高度时可使用一个环境：

```powershell
uv run --extra isaacsim python -m g1.rl eval --suite g1_lower_push --checkpoints "<model.pt>" --num_envs 1 --height-range 0 0 --push-heights 0 --repetitions 10 --output outputs/eval/lower_push_stand
```

## 修改条件

`--episode-length-s`、`--repetitions`、`--height-range`、`--push-heights`、`--seeds`、`--policy-output`、`--num_envs` 沿用现有 eval 参数。

更改推扰强度、推扰时间窗口或恢复阈值时，复制 [内置 suite](../src/g1/rl/evaluation/suites/g1_lower_push.yaml)，修改对应字段，并通过 `--suite <自己的.yaml>` 传入。例如将强度改为 ±0.30 m/s：

```yaml
benchmark:
  push_velocity_range:
    x: [-0.30, 0.30]
    y: [-0.30, 0.30]
```

该片段用于修改完整 suite 中的相应字段，不单独作为配置文件。

schema 2 新增可选顶层 `stages` 列表。独立推扰 suite 为 `stages: [squat_push]`；省略该字段时仍执行原四阶段。只执行静止推扰时，`ranges` 只需填写 `height_offset`，不会采样速度指令。

## 输出与查看

输出目录结构、CSV/JSON 字段及 TensorBoard 均与四阶段抗扰组一致，结果只包含 `module=squat_push` 的总体和分高度行：

- `timeout_rate`：正常完成率；`failure_rate`：失败率。
- `push_recovery_rate`：已确认恢复的推扰次数 / 实际推扰次数。
- `mean_recovery_s`：已恢复事件的平均恢复确认时间，包含 0.2 秒持续保持时间；未恢复事件保持 null。
- `rmse_height_m`：高度 RMSE，包含初始高度过渡与终止帧。
- 速度 RMSE、最大高度偏差、最大 roll/pitch、终止原因、回合与控制步数量。
- 计划/实际推扰数量、恢复/未恢复数量；每回合原始结果包含指令计划、推扰计划及每次实际推扰的速度增量、时间和恢复时间。

```powershell
$report = Get-Content "outputs/eval/lower_push_p3_07/report_latest.json" -Raw -Encoding UTF8 | ConvertFrom-Json
uv run tensorboard --logdir "$($report.tensorboard_dir)"
```

TensorBoard 沿用“00 四阶段指标”面板，run 位于 `squat_push/mean/<模型标签>`，高度横轴为高度列表编号。精确数值读取 `report_latest.json` 指向的 `summary.csv`。

增加 `--plan-only` 先生成计划；`--resume <输出目录>` 继续；`--resume <输出目录> --report-only` 只重新生成报告。参数变化需要使用新输出目录。

指标详细定义见 [四阶段评估说明](LOWER_FOUR_STAGE_EVALUATION.md#推扰及恢复)。
