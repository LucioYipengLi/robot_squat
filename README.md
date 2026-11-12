中文 | [English](README.en.md)

# g1

一个可安装的下游 Isaac Lab 任务包，采用标准的 uv `src` 目录布局。

已注册的任务有：

- `Template-G1-v0`：Phase 1。
- `Template-G1-Phase2-v0`：Phase 2，增加静态物理随机化。
- `Template-G1-Phase3-v0`：Phase 3，增加外部推力。
- `Template-G1-Upper-v0`：固定根节点，控制腰部与双臂，按左手、右手、双手三种模式跟踪世界固定目标，观测使用实时盆骨相对位姿。
- `Template-G1-Upper-Mobile-v0`：自由底座，Upper 同时控制上肢关节与三维速度命令，冻结下肢负责执行。


## 安装

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后创建项目环境。`uv sync` 不安装 Isaac Sim；
本任务的默认物理预设是 `isaacsim_physx`，运行时需要 `--extra isaacsim`，或显式选择其他已安装的后端：

```bash
uv sync
```

本项目的 `pyproject.toml` 通过可编辑的相对路径指向一份 Isaac Lab 源码检出目录。如果相关目录发生移动，请更新
`[tool.uv.sources]`。

本项目会转发用于创建它的 Isaac Lab 包所声明的全部可选 extra，涵盖物理、渲染、可视化、强化学习（RL）、遥操作和开发等
功能。请查看 `pyproject.toml` 获取完整列表，并在 `uv run` 时传入每个所需的 extra：

```bash
# 独立的 OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# 使用 PhysX 与 RTX 渲染的 Isaac Sim
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# 使用 Rerun 可视化器的 Isaac Sim
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

`all` extra 会转发 Isaac Lab 的 `all` 选项集合。`ov` extra 会同时安装 `ovphysx` 和 `ovrtx` 两种运行时；你也可以独立
地选择其中任意一个。请将 `pyproject.toml` 和 `uv.lock` 一起提交，以便协作者使用相同的环境。

## 运行任务

### 统一训练入口

训练与续训使用 `python -m g1.rl train`，模型回放使用 `play`，人工控制使用 `teleop`，固定条件批量评估使用 `eval`。
`train/play` 根据 `--task` 自动识别任务族并选择 Runner；`--task-family` 可选，仅用于核对任务归属。
每个任务族拥有自己的 Runner；下肢将模型与实际课程状态保存在同一个检查点中，Phase 2/3 自动继承上一阶段的任务分布。
目前支持 Torch 环境前端、单进程 RSL-RL `OnPolicyRunner`，无需额外指定 RL 库。

`upper_limb_agent` 提供固定根节点和冻结下肢站立两种 G1 双腕位姿跟踪环境，共用独立 Runner 和标准训练日志。
固定底座输出到 `logs/rsl_rl/g1_upper/`，冻结下肢版本输出到 `logs/rsl_rl/g1_upper_mobile/`。
机器人资产与 MDP 模块由上肢任务独立维护。
`teleop/eval` 本阶段仍只支持下肢任务。任务选择及后续任务的接入方式见 [任务 Runner 说明](docs/TASK_RUNNERS.md)。

以下命令统一使用 Isaac Sim PhysX；将引号内的 `<...>` 替换为实际检查点路径。迭代数仅为示例，可按训练计划调整。

| 参数 | 说明 |
| --- | --- |
| `--task-family` | 可选的任务族校验：`lower_limb_agent` 或 `upper_limb_agent`；默认根据 `--task` 自动识别。 |
| `--task` | 本次运行的任务 ID，下肢任务还区分训练阶段。 |
| `--checkpoint` | 加载用于续训或回放的模型；下肢跨阶段续训选择相邻上一阶段模型。无需另加 `--resume`。 |
| `--max_iterations` | 本次调用执行的训练迭代数；加载后表示继续训练多少次，不是累计迭代上限。 |
| `--seed` | 默认随机抽取本次运行种子；传入非负整数（如 `--seed 42`）固定种子，`--seed -1` 显式随机抽取。 |
| `--num_envs`、`--device` | 可选，例如 `--num_envs 1024 --device cuda:0`。 |

项目智能体配置在每次创建时从系统随机源抽取种子，适用于训练、续训、回放及下肢遥操作。
训练日志的 `params/agent.yaml` 与 `params/env.yaml` 记录最终实际使用的整数种子；需要复用时，将该值传给 `--seed`。
显式设置 `agent.seed=<整数>` 也会覆盖随机默认值，`--seed` 优先于该配置。
`--deterministic` 是独立的确定性开关，不会将默认随机种子变成固定值；同种子也不代表续训会接续原随机序列。

### G1 上肢位姿跟踪

无窗口训练与可视化回放：

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-v0 --viz none --max_iterations 1500 physics=isaacsim_physx

uv run --extra isaacsim python -m g1.rl play --task Template-G1-Upper-v0 --checkpoint "<g1_upper_model.pt>" --num_envs 16 --real-time physics=isaacsim_physx
```

默认使用 4096 个环境；可通过 `--num_envs` 调整。回放时显示双腕当前坐标架和有效侧目标。
目标在默认双腕位姿附近附加偏移采样，刷新时间与目标到骨盆的距离相关。当前观测为 93 维，包含双腕位姿误差，需要重新训练策略；
模式比例、目标范围、奖励和指标定义见 [任务 Runner 说明](docs/TASK_RUNNERS.md#g1-上肢位姿跟踪结构)。

上肢任务已接入 0 kg 空间 Q 先验：位置达标后按目标相对骨盆的位置评分，左右臂共用镜像地图。
默认奖励权重为零，诊断持续记录；配置、指标和资源归档见 [空间 Q 奖励说明](docs/WORKSPACE_QUALITY.md)。

冻结下肢的移动任务使用21维 Upper 动作（速度命令3维、高度偏置1维、上肢关节17维）和107维观测，输出到 `logs/rsl_rl/g1_upper_mobile/`：

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Upper-Mobile-v0 --num_envs 512 --viz none physics=isaacsim_physx
```

默认下肢模型为 `2026-10-21_22-59-54/model_9999.pt`，权重冻结，腿脚参数跟随最新 Lower 配置。
初始速度范围为 `vx ±0.3 m/s、vy ±0.1 m/s、ωz ±0.3 rad/s`，高度偏置范围为 `[-0.25, 0.02] m`，零动作保持默认站高。
`lower_command_rate=-0.10` 保留H2的四维归一化命令变化约束；保持恒定高度时该分量为零。
`base_vertical_acc=-0.03` 沿用H3B配置，对骨盆世界Z加速度超过4 m/s²的部分施加软惩罚，覆盖全部物理子步。
幅值与越界惩罚仍只约束速度，权重保留R9C配置。
Mobile默认混合70%常规目标、30%额外降低5～15 cm的高度目标；第二类整回合保持。
参数、分类指标和续训对照见 [Upper高度目标说明](docs/UPPER_HEIGHT_TARGETS.md)。
旧 Mobile 的20动作/106观测模型（包括R9C系列）需重新训练或显式迁移。配置、零命令对照、回放及验证见 [冻结下肢控制说明](docs/FROZEN_LOWER_STANDING.md)。

### 首次训练：Phase 1

普通 WALK 使用实际离地计时驱动的髋膝踝参考和支撑时长奖励，蹲走保留原步态时钟。
配置、本机保留参数和物理指标见 [WALK 奖励说明](docs/WALK_REWARDS.md)。

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-v0 --max_iterations 10000 physics=isaacsim_physx
```

### 同阶段续训

例如继续训练 Phase 2，加载 Phase 2 检查点，再执行 1000 次迭代：

```bash
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase2-v0 --checkpoint "<phase2_model.pt>" --max_iterations 1000 physics=isaacsim_physx
```

Phase 1/3 同理：同时替换为对应阶段的任务 ID 和检查点。同阶段恢复课程进度、生存 EMA 及已完成回合汇总。

### 跨阶段续训：Phase 1 → 2 → 3

选择目标阶段的任务 ID，并显式指定相邻上一阶段的检查点。无需手工填写上一阶段的指令范围、模式权重和扰动幅度：

```bash
# Phase 1 → Phase 2：增加静态物理随机化
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase2-v0 --checkpoint "<phase1_model.pt>" --max_iterations 5000 physics=isaacsim_physx

# Phase 2 → Phase 3：增加外部推力
uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase3-v0 --checkpoint "<phase2_model.pt>" --max_iterations 3000 physics=isaacsim_physx
```

跨阶段继承实际任务分布和累计课程进度，重新积累新阶段条件下的门控统计。Phase 2/3 必须加载检查点，不能跳过阶段或逆向迁移。

Phase 3 推扰从每轴 ±0.10 m/s 的速度增量开始，存活 EMA 和联合成功率同时达标后逐渐扩大至 ±0.70 m/s。
两个门槛可在 Phase 3 课程配置中独立调整，见 [推扰课程与调参](docs/CURRICULUM_CHECKPOINTS.md#phase-3-推扰课程与调参)。

### 模型回放

使用与检查点相同阶段的任务 ID，例如回放 Phase 3：

```bash
uv run --extra isaacsim python -m g1.rl play --task Template-G1-Phase3-v0 --checkpoint "<phase3_model.pt>" physics=isaacsim_physx
```

回放恢复指令范围与模式权重，由 `play_mode()` 关闭课程、观测噪声、物理随机化和扰动。
续训和回放均从新回合开始，不接续原物理轨迹。

检查点默认位于 `logs/rsl_rl/<experiment_name>/<run>/model_*.pt`。
标准 `uv run isaaclab train/play --task ...` 使用上游 Runner，不保存或恢复项目课程状态；上述流程请统一使用项目入口。
缺少课程记录的旧检查点需显式提供 `--curriculum-snapshot "<curriculum.json>"`，详见 [课程检查点说明](docs/CURRICULUM_CHECKPOINTS.md)。

项目训练入口额外记录逐模式性能和实际训练样本占比，自动接入 TensorBoard/W&B；指标含义、统计窗口及判读方法见 [训练性能指标](docs/TRAINING_METRICS.md)。

### 批量固定条件评估

`eval` 在 headless 模式下，用同一套固定测试用例比较多个检查点。首批模块包括基础四模式任务跟踪与上肢扰动比例扫描，支持均值动作、随机动作对照，以及中断后继续执行：

```bash
uv run --extra isaacsim python -m g1.rl eval --suite g1_robustness --checkpoints "early=<early_model.pt>" "late=<late_model.pt>" --policy-output both --num_envs 128 --headless
```

评估输出逐模式指标、汇总表与 TensorBoard 对比曲线；检查点提供策略，版本化测试套件统一控制指令、模式配额、回合长度和扰动条件。终端默认显示总体及回合进度、执行阶段和预计剩余时间，可用 `--progress plain` 或 `--progress off` 切换显示方式。内置基准采用固定初始姿态并关闭观测噪声，跨阶段比较会检查模型接口兼容性。

套件配置、自定义实验、指标解释、日志查看和恢复执行见 [统一评估框架说明](docs/EVALUATION.md)。
上肢扰动默认采用渐进参考，参数、执行统计与旧检查点说明见 [上肢渐进扰动](docs/UPPER_BODY_MOTION.md)。

### 模型遥操作与可视化

加载同阶段模型，通过键盘或 Xbox 布局手柄控制高度、速度和骨盆偏置，默认单环境、实时运行并打开 Kit 窗口：

```bash
uv run --extra isaacsim python -m g1.rl teleop --task Template-G1-v0 --checkpoint "<phase1_model.pt>" --viz kit physics=isaacsim_physx
```

W/S、A/D 控制平移，Q/E 控制转向，Z/X 控制髋高，I/K、J/L 控制骨盆偏置；SPACE 清零全部输入，C 切换骨盆与双脚几何可视化，H 显示帮助。默认恢复检查点课程范围，保留摔倒自动重置并取消超时。
原 `scripts/play_teleop.py` 保留为同一入口的薄启动器；参数、输入映射与可视化口径见 [遥操作说明](docs/TELEOP.md)。

### 环境工具

以下为上游环境工具，将占位符替换为已注册的任务：

```bash
# 列出本项目的全部环境及其可用的预设
uv run isaaclab list_envs --show_presets

# 在没有已训练策略的情况下试运行环境
uv run --extra isaacsim isaaclab zero_agent --task "<TASK_NAME>" --num_envs 16 physics=isaacsim_physx
uv run --extra isaacsim isaaclab random_agent --task "<TASK_NAME>" --num_envs 16 physics=isaacsim_physx
```

### 性能基准测试

以下以 Phase 1、16 个环境作为小规模基准，统一使用 Isaac Sim PhysX，不启用可视化。
确认可以运行后，可将 `--num_envs` 增至 1024 或 4096 测量目标规模的吞吐量；横向对比时保持环境数量、设备、物理后端及可视化设置一致。

| 类型 | 测量内容 |
| --- | --- |
| `startup` | Python 导入、配置解析、仿真启动、环境创建和首次步进耗时。 |
| `runtime` | 随机动作下的环境步进耗时与吞吐量，不训练策略。 |
| `training` | 实际训练中的采样、策略更新和整体吞吐量。 |
| `play` | 加载策略后的推理与环境步进性能，不更新模型。 |

```bash
# 1. 启动基准：无需 RL 库或步数参数
uv run --extra isaacsim isaaclab benchmark startup --task Template-G1-v0 --num_envs 16 --output_path outputs/benchmarks/startup physics=isaacsim_physx

# 2. 运行基准：预热 50 步，再测量 1000 步
uv run --extra isaacsim isaaclab benchmark runtime --task Template-G1-v0 --num_envs 16 --num_steps 1000 --warmup_steps 50 --output_path outputs/benchmarks/runtime physics=isaacsim_physx

# 3. 训练基准：显式限制为 10 次迭代，排除首次环境步的冷启动开销
uv run --extra isaacsim isaaclab benchmark training --rl_library rsl_rl --task Template-G1-v0 --num_envs 16 --max_iterations 10 --warmup_steps 1 --output_path outputs/benchmarks/training physics=isaacsim_physx

# 4. 回放基准：替换为 Phase 1 检查点路径，预热后测量 1000 步
uv run --extra isaacsim isaaclab benchmark play --rl_library rsl_rl --task Template-G1-v0 --checkpoint "<phase1_model.pt>" --num_envs 16 --num_steps 1000 --warmup_steps 50 --output_path outputs/benchmarks/play physics=isaacsim_physx
```

`training/play` 基准必须显式指定 `--rl_library`，`startup/runtime` 不需要。`--output_path` 指定性能结果目录，默认格式为 `schema`；训练日志及模型仍写入 `logs/rsl_rl/`。

这些基准使用上游 Runner，不保存或恢复项目课程状态。`training` 会更新模型，显式设置 `--max_iterations 10` 可避免按项目默认值训练 10000 次；正式三阶段续训仍使用项目入口。`play` 应选择与检查点同阶段的任务，用于测量运行性能；交互回放使用项目 `play`，固定条件的模型能力比较使用项目 `eval`。

使用 `physics=<PRESET>` 来选择 `isaaclab list_envs` 所展示的某个预设。

## 项目结构

审查待办、修复归档和课程检查点说明统一收录于 [docs 文档索引](docs/README.md)。

任务族（task families）位于 `src/g1/tasks`。在基于管理器（manager-based）的任务族中，任务通用的 MDP 词条位于 `mdp`，
而针对特定机器人的场景、注册信息和智能体配置则位于 `config/g1`。要新增另一个机器人，请创建 `config/g1` 的同级目录；
要新增另一个任务，请创建一个同级任务族目录。

## 项目资源

请将项目自有的 USD 文件及相关数据保存在 `src/g1/assets/data` 下。资源模块暴露了一个稳定的路径，无论在可编辑检出目录
还是已安装的 wheel 中都能正常工作：

```python
from g1.assets import G1_ASSETS_DIR

robot_usd_path = G1_ASSETS_DIR / "robots" / "my_robot.usd"
```

请将资源配置保留在 `g1.assets` 包内，并向那些需要字符串的配置字段传入 `str(robot_usd_path)`。USD 文件已通过项目的
`.gitattributes` 配置为使用 Git LFS；在添加大型二进制资源之前，请先安装 Git LFS。

## 开发

通过项目环境运行注册测试和代码质量检查：

```bash
uv run pytest
uv run pre-commit run --all-files
```

Isaac Lab 仓库中 `source/isaaclab_tasks/test` 下的测试辅助工具并不属于已安装的 `isaaclab_tasks` 包。请将测试固件
（test fixtures）保留在本项目中，并使用 Isaac Lab 的公开 API。如果你复制了 `env_test_utils.py`，它就会变成一份需要
你自行跟踪上游变更的内部捆绑（vendored）代码。

要配置 VS Code 或 Cursor，请运行 `setup_python_env` 任务，或直接调用其对应命令：

```bash
uv run isaaclab --editor
```

该 setup 命令会选择当前活动的解释器，并写入一个被 git 忽略的 `pyrightconfig.json`。生成的配置会继承项目中已签入的
Pyright 设置，并添加 Isaac Sim 扩展、项目 `src` 根目录，以及在当前活动 Python 环境中发现的任何 Isaac Lab 包。这同时
支持 VS Code 中的 Pylance 和 Cursor 中的 basedpyright。

在 VS Code 中，请使用 Pylance 并选择那个运行过 setup 命令的解释器。在 Cursor 中，请安装
[basedpyright 扩展](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) 而非 Pylance，选择
同一个解释器，然后重新加载窗口。两个语言服务器都会读取 `pyrightconfig.json`。

使用 `isaacsim` extra 时，请在生成编辑器配置时一并带上它，以便该命令能够发现 Isaac Sim 的安装位置：

```bash
uv run --extra isaacsim isaaclab --editor
```

对于未在项目环境中提供的 Isaac Sim 二进制（binaries）安装，请显式指定其路径：

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## 故障排查

如果 Pylance 或 basedpyright 无法解析模块，请确认所选解释器与运行 setup 命令时使用的解释器一致，然后重新加载编辑器
窗口。要添加缺失的扩展或减少索引内存占用，请编辑根目录下 `pyrightconfig.json` 中的 `extraPaths` 数组；移除项目并不
需要的模拟器扩展目录。
