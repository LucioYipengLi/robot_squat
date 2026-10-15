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

课程逻辑时间为“本进程控制步数 + 恢复偏移”，不改动仿真本地计数器，不从文件名或模型迭代号反推。这里的“检查点间隔”指课程检查间隔。

## 恢复规则

| 场景 | 行为 |
| --- | --- |
| 同阶段续训 | 恢复分布、逻辑进度、课程检查间隔、生存 EMA 和已完成回合汇总。 |
| Phase 1→2、2→3 | 继承真实分布、累计进度与模式锁存；保留目标阶段新增 DR，重新积累生存统计和门控结果，等待完整检查间隔再扩展。 |
| 同阶段评估 | 恢复指令范围、模式权重；沿用类的 `play_mode()` 关闭课程、观测噪声和 DR。保存的扰动幅度不重新启用事件。 |

所有加载均先校验记录，再加载模型和导入课程，然后重置环境，在新观测上开始 rollout。物理姿态、动作历史、观测历史、未完成回合、逐环境成功门控结果都会重建；不接续旧轨迹，也不恢复 RNG。环境数量可改变，但旧成功率缓存不会映射到新环境。

Phase 2/3 从头训练会报错；跳过阶段、逆向阶段迁移和跨阶段评估会报错。若 Phase 1 在引入 SQUAT_WALK 前结束，后续阶段仍会在累计逻辑时间到达阈值时引入，且只触发一次。

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
