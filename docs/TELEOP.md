# 模型遥操作与可视化

遥操作用于人工控制训练好的策略，与常规分布评估分开。使用同阶段检查点，默认单环境、实时步进、Kit 窗口、键盘与手柄输入：

```bash
uv run --extra isaacsim python -m g1.rl teleop --task Template-G1-v0 --checkpoint "<phase1_model.pt>" --viz kit physics=isaacsim_physx
```

Phase 2/3 同时替换任务 ID 和模型路径。省略 `--checkpoint` 时选择当前任务的 `latest`；文件路径、运行目录、`latest/best` 等解析与当前上游回放入口共用。旧检查点缺少课程记录时，按 [课程检查点说明](CURRICULUM_CHECKPOINTS.md) 提供 `--curriculum-snapshot`。

旧路径保留为薄启动器，参数转交同一入口：

```bash
uv run --extra isaacsim python scripts/play_teleop.py --task Template-G1-v0 --checkpoint "<phase1_model.pt>" --viz kit physics=isaacsim_physx
```

## 运行设置

| 参数/行为 | 说明 |
| --- | --- |
| `--input keyboard/gamepad/both` | 默认 `both`；没有手柄时使用键盘。若所选输入均不可用，报错退出。手柄在启动前连接；重新连接后请重启遥操作。 |
| `--num_envs` | 仅接受 1，避免多机器人共享人工输入与单环境几何标记。 |
| `--viz kit` | 默认开启；当前输入适配器需要可见的 Isaac Sim Kit 窗口，不支持其他可视化器替代键盘窗口。推荐 `isaacsim_physx` 后端。 |
| `--show_pelvis` | 初始显示骨盆和双脚几何投影；默认隐藏，用 C 切换。 |
| `--max_steps N` | 可选，执行 N 个控制步后退出；默认持续运行，关闭窗口或 Ctrl+C 退出。 |
| `--video --video_length N` | 沿用当前回放的录像配置；录像放在模型运行目录下的 `videos/teleop/`。若同时指定 `--max_steps`，先到的步数限制生效。 |
| 终止与重置 | 取消超时；保留摔倒等终止及自动重置。`episode_length_s` 保持原值以兼容检查点。 |
| 评估设置 | 任务类的 `teleop_mode()` 调用各阶段 `play_mode()`，关闭课程、观测噪声、上肢扰动和物理随机化；不接受 `--train_env_cfg`。 |

## 控制映射

Isaac Sim 窗口需要处于焦点。键盘输入和方向键为增量式控制，松开后保留当前目标。

| 输入 | 功能 |
| --- | --- |
| W/S 或 ↑/↓ | 前进/后退速度，每次 0.1 m/s。 |
| A/D | 横向速度，每次 0.1 m/s。 |
| Q/E 或 ←/→ | 左/右转，每次 0.2 rad/s。 |
| Z/X | 升高/降低髋高目标，每次 0.05 m。 |
| I/K、J/L | 骨盆前后、左右目标，每次 0.005 m；仅 STAND/SQUAT 生效。 |
| V | 仅清零骨盆偏置。 |
| SPACE / 手柄 A | 清零键盘、方向键、摇杆及积分高度的全部输入。 |
| C / H | 切换几何可视化 / 打印帮助。切换按键不会因键盘重复事件反复开关。 |
| 手柄左摇杆 | 上下控制前进速度，左右控制 yaw；比例输入，松杆归零。 |
| 手柄方向键 | 上下累加前进速度，左右累加横向速度。 |
| 手柄右摇杆上下 | 按控制 dt 积分高度，满偏速率 0.3 m/s；松杆保持。 |

先恢复 R04，再从命令项的实际范围生成六维数值限制，不再写死课程终值。键盘与手柄的高度/速度叠加后限幅；进入行走时清除骨盆偏置，防止停步后重放旧目标。数值落在恢复范围内，不代表任意人工组合都在训练采样分布内；模式和骨盆平滑由任务指令项处理。

## 观测与可视化口径

初始化时先安装零人工指令并重置，避免第一次推理使用随机采样任务。每步使用上一步产生的观测推理，再提交新输入并执行环境步进；新目标进入下一次观测，存在一个控制周期的输入响应延迟。观测历史和骨盆平滑均只随正常环境步进推进一次，不通过重复 `get_observations()` 或覆盖命令内部方法刷新。

高度球与速度箭头复用任务自身可视化。C 控制的附加几何标记按需创建：红球是骨盆地面投影，紫球是有效骨盆目标，青球是双踝中点，黄色点是双脚足底几何角点。当前 `isaacsim.util.debug_draw` 可用时绘制绿色凸包边线，否则只显示标记点。行走时隐藏静止模式的骨盆目标和中点标记。

凸包是双脚的**几何投影包络**，没有按接触筛选，不代表实际支撑域或稳定性保证；骨盆位置不是质量加权 COM。投影平面采用环境原点高度，适用于当前平地任务。

## 代码组织与验证

| 位置 | 职责 |
| --- | --- |
| [`scripts/play_teleop.py`](../scripts/play_teleop.py) | 保留旧调用路径，只做参数转发；导入时不解析参数、不启动仿真。 |
| [`g1/rl/teleop.py`](../src/g1/rl/teleop.py) | 配置解析、课程 Runner 加载、回放时序、退出和资源清理。 |
| [`任务 env_cfg.py`](../src/g1/tasks/lower_limb_agent/config/g1/env_cfg.py) | `teleop_mode()` 定义单环境、无超时及调试标记的任务配置，三个阶段继承。 |
| [`g1/teleop/controls.py`](../src/g1/teleop/controls.py) | 可独立测试的输入状态、合成、限幅和按键映射。 |
| [`g1/teleop/kit.py`](../src/g1/teleop/kit.py) | 启动后创建的 Kit 输入订阅及释放；不在模块导入时初始化原生输入。 |
| [`任务 visualization.py`](../src/g1/tasks/lower_limb_agent/visualization.py) | 与本任务骨盆/足部语义绑定的几何可视化。 |

统一使用当前 RSL-RL 推理接口，不保留旧版本策略对象分支，也不硬编码观测维数。输入、课程恢复顺序、观测时序、几何计算和正常/异常清理通过 CPU 环境替身验证；实际 Kit 窗口、手柄和图形渲染尚需交互冒烟测试。
