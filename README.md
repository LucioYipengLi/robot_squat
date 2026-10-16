中文 | [English](README.en.md)

# g1

一个可安装的下游 Isaac Lab 任务包，采用标准的 uv `src` 目录布局。

已注册的任务有：

- `Template-G1-v0`：Phase 1。
- `Template-G1-Phase2-v0`：Phase 2，增加静态物理随机化。
- `Template-G1-Phase3-v0`：Phase 3，增加外部推力。


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

首次训练、同阶段续训和评估使用 `python -m g1.rl train/play`，人工控制使用同一入口的 `teleop` 子命令。
项目入口将模型与实际课程状态保存在同一个检查点中，Phase 2/3 自动继承上一阶段的任务分布。
目前支持 Torch 环境前端、单进程 RSL-RL `OnPolicyRunner`，无需额外指定 RL 库。

以下命令统一使用 Isaac Sim PhysX；将引号内的 `<...>` 替换为实际检查点路径。迭代数仅为示例，可按训练计划调整。

| 参数 | 说明 |
| --- | --- |
| `--task` | 本次运行的目标阶段，使用上方注册的任务 ID。 |
| `--checkpoint` | 加载的模型文件；同阶段续训选择本阶段模型，跨阶段续训选择相邻上一阶段模型。无需另加 `--resume`。 |
| `--max_iterations` | 本次调用执行的训练迭代数；加载后表示继续训练多少次，不是累计迭代上限。 |
| `--num_envs`、`--device` | 可选，例如 `--num_envs 1024 --device cuda:0`。 |

### 首次训练：Phase 1

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

### 评估与回放

使用与检查点相同阶段的任务 ID，例如评估 Phase 3：

```bash
uv run --extra isaacsim python -m g1.rl play --task Template-G1-Phase3-v0 --checkpoint "<phase3_model.pt>" physics=isaacsim_physx
```

评估恢复指令范围与模式权重，由 `play_mode()` 关闭课程、观测噪声、物理随机化和扰动。
续训和评估均从新回合开始，不接续原物理轨迹。

检查点默认位于 `logs/rsl_rl/<experiment_name>/<run>/model_*.pt`。
标准 `uv run isaaclab train/play --task ...` 使用上游 Runner，不保存或恢复项目课程状态；上述流程请统一使用项目入口。
缺少课程记录的旧检查点需显式提供 `--curriculum-snapshot "<curriculum.json>"`，详见 [课程检查点说明](docs/CURRICULUM_CHECKPOINTS.md)。

项目训练入口额外记录逐模式性能和实际训练样本占比，自动接入 TensorBoard/W&B；指标含义、统计窗口及判读方法见 [训练性能指标](docs/TRAINING_METRICS.md)。

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

这些基准使用上游 Runner，不保存或恢复项目课程状态。`training` 会更新模型，显式设置 `--max_iterations 10` 可避免按项目默认值训练 10000 次；正式三阶段续训仍使用项目入口。`play` 应选择与检查点同阶段的任务，用于测量运行性能；检查点任务分布的评估使用前文的 `python -m g1.rl play`。

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

## Isaac Sim UI 扩展

请将项目根目录添加到 Isaac Sim 扩展管理器（Extension Manager）的搜索路径中，刷新后在 `Third Party` 下启用该扩展。
请通过项目的 `isaacsim` extra 启动 Isaac Sim，以便可用的 UI 依赖就位。Kit 会通过 `config/extension.toml` 加载
`src/g1/ui_extension_example.py`。

## 故障排查

如果 Pylance 或 basedpyright 无法解析模块，请确认所选解释器与运行 setup 命令时使用的解释器一致，然后重新加载编辑器
窗口。要添加缺失的扩展或减少索引内存占用，请编辑根目录下 `pyrightconfig.json` 中的 `extraPaths` 数组；移除项目并不
需要的模拟器扩展目录。
