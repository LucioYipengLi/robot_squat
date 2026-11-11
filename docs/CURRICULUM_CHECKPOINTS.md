# 课程检查点与三阶段续训

项目入口为 `uv run --extra isaacsim python -m g1.rl train|play --task ...`，完整命令见 [README](../README.md#运行任务)。无需修改 IsaacLab，也无需 Phase 2/3 手工复制课程终态。

人工控制使用同一入口的 `teleop` 子命令，也先恢复课程记录；输入和附加可视化见 [遥操作说明](TELEOP.md)。

## 保存内容

项目 `CurriculumOnPolicyRunner` 继承本地 RSL-RL Runner，在每次周期保存和最终保存时，将 `infos["g1_curriculum"]` 与模型、优化器、迭代号一起交给上游写入同一个 `.pt`。日志上传看到的也是包含课程记录的文件。

| 记录 | 内容 |
| --- | --- |
| 版本与兼容性 | schema 1、任务族、Phase、控制 dt、回合时长、机器人 USD 路径、动作关节与缩放、观测历史长度、有序扰动组。 |
| 真实分布 | 六维指令范围、实际 `_mode_probs`、各组实际扰动幅度；读取 manager 中的运行配置，不依赖初始配置快照。 |
| 课程进度 | 累计控制步数、SQUAT_WALK 引入锁存状态、各检查点间隔已过去的步数。 |
| 生存统计 | EMA、初始化标志、窗口内已完成回合的归一化时长和/数量、累计有效回合数。 |

Phase 3 另保存 `distribution.push_velocity_range` 和 `runtime.push_survival`，
以及 `runtime.checks["push:push_robot"]`。它们是 schema 1 的可选扩展，旧 Phase 2 检查点可直接迁入。

课程逻辑时间为“本进程控制步数 + 恢复偏移”，不改动仿真本地计数器，不从文件名或模型迭代号反推。这里的“检查点间隔”指课程检查间隔。

## 恢复规则

| 场景 | 行为 |
| --- | --- |
| 同阶段续训 | 恢复分布、逻辑进度、课程检查间隔、生存 EMA 和已完成回合汇总。 |
| Phase 1→2、2→3 | 继承真实分布、累计进度与模式锁存；保留目标阶段新增 DR，重新积累生存统计和门控结果，等待完整检查间隔再扩展。 |
| 同阶段评估 | 恢复指令范围、模式权重；沿用类的 `play_mode()` 关闭课程、观测噪声和 DR。保存的扰动幅度不重新启用事件。 |

所有加载均先校验记录，再加载模型和导入课程，然后重置环境，在新观测上开始 rollout。物理姿态、动作历史、观测历史、未完成回合、逐环境成功门控结果都会重建；不接续旧轨迹，也不恢复 RNG。环境数量可改变，但旧成功率缓存不会映射到新环境。

扰动课程同时检查存活 EMA 与全局联合成功率。联合成功率取各环境最近有效回合的结果，缓存独立于训练日志；加载时清空并从新回合重建，无有效结果时暂停幅度推进。检查点 schema 保持 1，同阶段的存活 EMA、窗口汇总和课程进度继续按上表恢复。

Phase 2/3 从头训练会报错；跳过阶段、逆向阶段迁移和跨阶段评估会报错。若 Phase 1 在引入 SQUAT_WALK 前结束，后续阶段仍会在累计逻辑时间到达阈值时引入，且只触发一次。

## Phase 3 推扰课程与调参

推扰沿世界坐标 x/y 向当前根节点速度叠加随机增量：`v_new = v_current + delta_v`。
每轴默认从 ±0.10 m/s 开始，每个合格检查窗口增加 0.05 m/s，最高 ±0.70 m/s；推扰间隔保持 4～6 s。
课程只扩展强度；质量不足时保持当前范围。两轴同时取边界时，增量向量长度可达到单轴幅度的 √2 倍。

常用参数集中在 `src/g1/tasks/lower_limb_agent/config/g1/env_cfg_phase.py` 的
`CurriculumCfg_Phase3.push_velocity_magnitude.params`：

| 参数 | 默认值 | 调整含义 |
| --- | --- | --- |
| `survival_gate` | `0.95` | 归一化回合存活时长 EMA 必须严格超过此值 |
| `task_success_gate` | `0.80` | 各环境最近有效回合的全局联合成功率必须严格超过此值 |
| `ema_beta` | `0.02` | 存活 EMA 更新系数；越大越快响应近期表现，与达标门槛不同 |
| `update_period` | `5000` | 控制步数；当前每轮 50 步，约每 100 个 PPO 迭代检查一次 |
| `velocity_step` | `0.05` | 每次增加的单轴速度增量上限，m/s |
| `velocity_final` | `0.70` | 最终单轴速度增量上限，m/s |

两个门槛是同时满足的独立条件，各自位于 `[0,1]`，无需相加等于 1。
存活统计只使用完整观察到的自然结束回合；人工重置、初始随机计数和空窗口不会触发推进。
联合成功包括正常超时完成、高度、速度/yaw 及适用模式的骨盆跟踪达标。
每个环境只贡献最近一个有效回合，口径与训练日志窗口平均值有区别。
推扰和上肢课程复用统计实现，各自维护独立的 EMA、窗口和时钟；此处调参只影响推扰课程。

可以参考 Phase 2 稳定区间的表现选取门槛，再用新 Phase 3 曲线判断推进速度。
例如，将门槛调整为 0.93 和 0.75，在原 Phase 3 训练命令后追加：

```text
env.curriculum.push_velocity_magnitude.params.survival_gate=0.93 env.curriculum.push_velocity_magnitude.params.task_success_gate=0.75
```

初始幅度位于 `EventCfg_Phase3.push_robot.params.velocity_range`，x/y 需保持相同、关于零对称。
门槛等参数按本次配置使用，不会被检查点中的旧门槛覆盖；新 Phase 3 同阶段续训则恢复实际已达到的幅度。

恢复行为：

- 旧/新 Phase 2 → 新 Phase 3：继承原有课程分布，从当前配置的推扰初值开始，等待完整检查窗口。
- 新 Phase 3 → Phase 3：恢复推扰范围、EMA、有效回合窗口与检查时钟；成功率缓存从新回合重建。
- 旧 Phase 3 缺少推扰记录：启动日志明确说明，推扰课程按当前配置重新开始；其他已有课程照常恢复。
- `play`：保留记录供追溯，继续关闭推扰和课程。

观测三项 `Curriculum/push_velocity_magnitude/` 曲线：`max_delta_v_mps`、`survival_ema`、`task_success_rate`。
全局指标可能掩盖某个模式的弱项，需同时查看各模式失败率和跟踪误差。

验证命令：

```powershell
uv run python -m pytest tests/test_push_curriculum.py tests/test_mdp_training_logic.py tests/test_curriculum_checkpoints.py tests/test_evaluation_and_torque.py -q
```

真实 PhysX 短训迁移检查（4 个环境，各次调用 2 个迭代，检查 Phase 2→3、Phase 3 续训、推扰触发及日志）：

```powershell
$env:G1_PHASE3_PUSH_SMOKE='1'
uv run python -m pytest tests/test_phase3_push_physics.py -q
```

## 旧检查点

没有课程记录的旧模型默认拒绝加载，避免把初始条件冒充训练终态。若确实保有训练时的完整状态，可提供：

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase2-v0 --checkpoint <old_phase1.pt> --curriculum-snapshot <phase1_curriculum.json> physics=isaacsim_physx
```

JSON 必须是 `export_curriculum_state(env)` 所返回的完整 schema，不能只列几个范围。可在已知正确状态的旧环境中导出：

```python
import json
from pathlib import Path
from g1.tasks.lower_limb_agent.curriculum_state import export_curriculum_state

Path("phase1_curriculum.json").write_text(
    json.dumps(export_curriculum_state(env.unwrapped), indent=2), encoding="utf-8"
)
```

若只能找回分布而没有可靠的旧进度/统计，需要明确建立一次人工迁移快照，不能自动推断。含原生课程记录的模型不接受额外 JSON 覆盖；成功迁移后的新检查点自动内嵌状态。

## 支持范围与维护

- 当前支持 Torch 环境前端、单进程 RSL-RL `OnPolicyRunner`；不支持 Warp 环境前端、分布式统计恢复或 DistillationRunner，入口明确拒绝这些用法。
- `play` 使用正常评估配置，不接受 `--train_env_cfg`。
- 标准 `uv run isaaclab train/play --task ...` 仍使用上游 Runner，不经过本适配器；需要 R04 时，训练和回放都使用项目入口。
- 本地上游后端没有 Runner 工厂参数，因此项目入口仅在当前同步调用期间替换后端 Runner 构造引用，并在 `finally` 中恢复。升级 IsaacLab/RSL-RL 后须复核入口构造点及 save/load 接口。
- 兼容性校验不是资产内容哈希或完整实验配置锁定。奖励、DR 超参数、网络等仍来自目标配置；刻意改变它们属于新实验，需自行记录。
- 自动测试覆盖序列化、真实上游 Runner 保存/加载流程和 CPU 环境替身；完整物理仿真恢复及训练效果尚需冒烟验证。
