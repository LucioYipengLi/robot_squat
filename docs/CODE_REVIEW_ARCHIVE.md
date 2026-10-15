# G1 代码审查：已修复问题归档

- 归档日期：2026-10-15。
- 当前待办与设计讨论见 [CODE_REVIEW.md](CODE_REVIEW.md)。
- 本文保存已完成的修复说明、使用方式与验证记录，不代表完整仿真验证通过。
- R04 已实现；R10 躯干约束评估和 R11 备用奖励仍保留在当前待办。

## 归档索引

| 编号 | 已完成内容 |
| --- | --- |
| R01 | 二阶动作历史滚动与回合边界清理。 |
| R02 | 物理重置前采集终止帧、步号去重、排除无效回合。 |
| R03 / R05 | 标准 play_mode 评估入口及事件/课程关闭；移除旧评估兼容类。 |
| R04 | 课程状态随模型保存；同阶段续训、相邻阶段迁移及冻结评估恢复。 |
| R06 | 固定标称刚度归一化，初始化校验与缓存。 |
| R07 | 课程到期检查去重与延迟处理。 |
| R08 | 窗口内回合等权平均后做生存 EMA。 |
| R09 | 明确骨盆统计为有效步达标率。 |
| R10 配置部分 | 移除未生效的刚体选择；是否新增躯干约束仍待评估。 |
| R12 | `stand_still` 显式限定 STAND/SQUAT，行走模式不再惩罚正常抬脚。 |
| 遥操作重构 | 旧脚本改为薄启动器，流程、输入、任务配置与几何可视化分层，接入课程恢复。 |
| 其他 | env_ids 切片兼容及 re 导入问题。 |

下列章节保留修复说明。评估配置关闭逻辑由 R03/R05 完成，真实检查点分布恢复由后续 R04 接通。

## 遥操作入口与项目组织

`scripts/play_teleop.py` 仅转发到 `python -m g1.rl teleop`。回放流程放在 `g1/rl/teleop.py`，输入状态和 Kit 适配放在 `g1/teleop/`，任务类的 `teleop_mode()` 定义环境设置，任务专用几何可视化放在任务目录。移除旁路 `cli_args`、旧版 RSL-RL 分支、硬编码控制范围和观测维数。

加载先经过 R04，再建立人工输入。保留单环境、取消超时和摔倒自动重置；不覆盖命令内部更新钩子、不重复推进观测历史。退出、窗口关闭及异常均释放输入订阅并关闭环境。使用方式与输入延迟说明见 [TELEOP.md](TELEOP.md)。

完整回归 **89 项通过**，新增模块通过 Ruff；新旧入口帮助命令均已实际执行。真实 Kit 窗口、手柄和渲染尚待交互验证，IsaacLab 仓库未修改。

## R12：站立抬脚惩罚的模式门控

[奖励配置](../src/g1/tasks/lower_limb_agent/config/g1/env_cfg.py) 为 `stand_still` 配置 `velocity_command_name="task_command"`，并显式指定 `zero_modes=(0, 1)`，对应 STAND/SQUAT。WALK/SQUAT_WALK 返回零，不再与交替抬脚的步态目标竞争。三个阶段及其评估配置继承同一设置，接触阈值和权重保持原值。

验证三个阶段训练/评估配置在四种模式下的数值结果：静止模式保留离地脚计数，行走模式输出零；完整回归 **67 项通过**。文档迁入 `docs/`，本地 Markdown 文件链接已核对。

## R04：课程检查点恢复

**状态：已实现代码链路并完成 CPU 回归，待完整物理仿真验收。**

新增 [状态协议](../src/g1/tasks/lower_limb_agent/curriculum_state.py)、[Runner](../src/g1/rl/runner.py) 和 [项目入口](../src/g1/rl/__main__.py)。范围、实际模式权重、各组实际扰动幅度、课程逻辑进度、检查间隔及生存 EMA/已完成回合汇总随模型写入同一 `.pt`，覆盖周期保存、最终保存和日志上传。

同阶段恢复课程统计；Phase 1→2、2→3 继承真实分布并重建新 DR 条件下的门控统计；评估恢复分布后维持 `play_mode()` 的关闭项。所有加载都重置轨迹和历史，重取观测后开始 rollout。Phase 2/3 的手工课程快照已移除，模式引入课程持续保留，允许 Phase 1 提前结束后继续到达触发点。

旧检查点缺少状态时明确报错，可显式提供完整 JSON；关键配置不兼容、跨越/逆向阶段、跨阶段评估均拒绝。支持改变环境数量；当前限定单进程 RSL-RL OnPolicyRunner，不恢复 RNG/物理轨迹。

**使用入口：**`uv run --extra isaacsim python -m g1.rl train|play --task ...`。标准 `isaaclab train/play` 继续使用原 Runner，不保存/恢复 R04；请勿混用来期待自动课程继承。详见 [使用说明](CURRICULUM_CHECKPOINTS.md)。IsaacLab 仓库未修改。

本轮完整测试 **67 项通过**，覆盖真实上游 Runner 保存/加载、两次相邻阶段迁移、记录校验、旧模型 JSON、评估关闭、环境数变化及入口正常/异常清理。新增代码通过完整 Ruff，训练入口 help 实际执行成功。测试使用环境替身，未启动物理仿真；其余复核结果见 [当前报告](CODE_REVIEW.md)。

## R03 / R05：评估入口与事件课程依赖

**状态：已采用各阶段类的 `play_mode()` 修复入口与关闭逻辑；检查点分布恢复现由 R04 项目入口提供。**

位置：[任务注册](../src/g1/tasks/lower_limb_agent/config/g1/__init__.py)、[分阶段评估配置](../src/g1/tasks/lower_limb_agent/config/g1/env_cfg_phase.py)、[扰动课程](../src/g1/tasks/lower_limb_agent/mdp/curriculums.py)。

三个任务的标准 play 流程通过继承调用对应阶段的 `play_mode()`。旧 `_PLAY` 兼容类已移除，今后统一使用任务 ID，不再导入单独的评估配置类。

```bash
uv run isaaclab play --task Template-G1-v0 --checkpoint latest
uv run isaaclab play --task Template-G1-Phase2-v0 --checkpoint latest
uv run isaaclab play --task Template-G1-Phase3-v0 --checkpoint latest
```

任务已注册默认 RL 库 `rsl_rl`，可以省略 `--rl_library`。`--checkpoint latest` 显式选择最近的可用检查点，也可以换成模型路径；只提供 `--task` 时由入口的默认实验目录规则查找模型。命令不会自行训练模型。

若需要由 uv 提供 Isaac Sim 依赖，可使用项目已有的启动口径：

```bash
uv run --extra isaacsim isaaclab play --task Template-G1-Phase2-v0 --checkpoint latest physics=isaacsim_physx
```

| 项目 | 常规评估行为 |
| --- | --- |
| 环境与观测 | 默认最多 10 个环境，关闭观测噪声；保留动作/观测维数、历史长度、控制周期与奖励。CLI 后续可覆盖环境数。 |
| 课程与扰动 | 移除全部课程、上肢扰动及 Phase 3 推扰，避免缺失事件引用和评估中途加难度。 |
| 物理随机化 | 关闭摩擦、质量、质心、负载及 PD 增益随机化，使用资产和地面的固定参数。 |
| 回合重置 | 保留状态重置事件，取消随机偏移和随机速度，回到默认状态；保留时间限制与终止判定。 |
| 指令分布 | 标准入口保留配置范围与模式权重；R04 项目入口进一步恢复检查点真实分布，不强制改成课程终值。 |

配置级测试覆盖三个阶段、重复调用和训练配置隔离。完整环境首次重置及仿真回放仍待运行验证。上游标准入口的 `--train_env_cfg` 可用于训练配置下的鲁棒性检查，但不恢复 R04 课程；R04 项目 play 入口只支持常规冻结评估。

## R06：力矩归一化口径

**状态：已按用户选择实现标称刚度归一化。**

位置：[奖励函数](../src/g1/tasks/lower_limb_agent/mdp/rewards.py) 的 `joint_torques_l2_normalized()`、[奖励配置](../src/g1/tasks/lower_limb_agent/config/g1/env_cfg.py) 的 `dof_torques`、[执行器配置](../src/g1/tasks/lower_limb_agent/config/g1/utils/assets.py)。

当前计算：

```text
力矩代价 = Σ(关节施加力矩 / 配置中的刚度)^2
```

奖励词条初始化时从资产的最终执行器配置取得标称刚度快照，完成名称匹配、正值校验和缓存；因此启动前的配置覆盖也能生效。踝关节默认分母为 20，膝为 200，髋为 100，不再在奖励配置中重复维护数值。

Phase 2 改变实时 PD 刚度时，分母保持这份标称快照。例如踝力矩为 10 N·m，当前代价为 `(10 / 20)^2 = 0.25`。相对旧分母 100，该关节代价增大 25 倍；奖励权重未额外调整，需要训练中观察实际量级。

每个选中关节必须匹配有限正刚度，否则初始化时明确报错，避免原先以 `1e-8` 作分母产生巨大惩罚。名称解析只进行一次，支持切片、列表和 Tensor 索引。

## 已完成项与验证边界

| 编号 | 完成内容 |
| --- | --- |
| R01 | 有状态动作历史，正确滚动二阶差分；由奖励管理器在回合边界清空。 |
| R02 | 项目环境在物理重置事件前采样终止状态，按步号去重；排除无样本回合。 |
| R03 / R05 | 各阶段定义 `play_mode()`，关闭课程、噪声、DR 和扰动，保留默认状态重置；已移除旧评估兼容类。 |
| R06 | 奖励固定使用最终资产配置中的标称刚度；初始化时校验并缓存，运行中随机化不改变分母。 |
| R07 | 到期后首次重置检查一次，不重复推进、不突发补更新。 |
| R08 | 窗口内有效完成回合等权平均，再做 EMA；排除初始化、手动截断、重复重置及人为老化的不完整首轮回合。 |
| R09 | 文档已明确为有效步达标率，统计公式及日志键不变。 |
| R10 配置部分 | 移除对 `flat_orientation_l2` 无效的 torso/pelvis 选择；躯干约束评估保留在待讨论部分。 |

必要的指标边界：R08 的 `ema_beta` 按非空更新窗口解释；首个窗口直接初始化 EMA。它是回合平均存活时长比例的时间 EMA，不是全部历史的算术平均，也不是二元超时成功率。`survival_valid=0` 表示尚无有效 EMA。高度/速度成功仍表示跟踪误差达标，不额外要求回合存活到超时。

回归范围包括[训练逻辑](../tests/test_mdp_training_logic.py)、[三个任务注册](../tests/test_registration.py)及[评估与标称刚度](../tests/test_evaluation_and_torque.py)，归档前最近一次代码验证 **33 项通过**。随兼容类移除对应的两个用例，保留三个标准阶段的评估覆盖、hook 幂等性、训练配置隔离、标称刚度来源与启动覆盖、固定分母、关节选择及非法分母处理。相关测试通过 Ruff 检查，修改配置通过未使用/未定义名称及语法检查。

**尚未完成完整仿真或训练验证**，不能将配置及 CPU 模拟场景测试等同于端到端正确性证明。

## 本地对照文件

以下路径以 `D:\Code\IsaacLab` 为根目录；该仓库保持只读。

| 用途 | 路径 |
| --- | --- |
| G1 平地与粗糙地形示例 | `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/{flat,rough}_env_cfg.py` |
| 速度任务回放实现 | `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py` |
| 环境 step/reset 顺序 | `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` |
| 默认回放行为 | `source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py` |
| 标准奖励与事件 | `source/isaaclab/isaaclab/envs/mdp/{rewards,events}.py` |
| 管理器实现 | `source/isaaclab/isaaclab/managers/{command,event,curriculum,reward,termination}_manager.py` |
| 回放配置解析 | `source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py` |
| RSL-RL 检查点加载入口 | `source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py` |

本项目当前安装的 RSL-RL 保存/加载实现另见 `.venv/Lib/site-packages/rsl_rl/runners/on_policy_runner.py`。

