# Upper腕部目标遥操作

用于交互检查Upper策略：手动编辑左／右腕世界目标，或指定任务类型自动采样。首版以
`Template-G1-Upper-Mobile-v0`为主；固定基座`Template-G1-Upper-v0`支持手动与常规采样。
Upper继续自主产生关节动作与Lower命令，冻结Lower负责执行。

## 快速使用

在项目根目录启动，提供与当前环境接口兼容的Upper检查点：

```powershell
uv run --extra isaacsim python -m g1.rl teleop --task Template-G1-Upper-Mobile-v0 --checkpoint ".\logs\rsl_rl\g1_upper_mobile\2026-11-16_00-47-24\model_9999.pt"
```

默认单环境、实时运行、键盘输入、Newton GL窗口、均值动作。`--viz newton_gl`可显式指定；旧别名
`--viz newton`也可使用。物理配置默认`isaacsim_physx`：Newton GL负责显示，PhysX继续负责仿真。
Isaac Sim物理运行时仍会启动，Kit图形窗口不参与遥操作。

省略checkpoint使用任务的`latest`。`--seed 42`可固定采样种子；`--max_steps N`可限制运行控制步数。
选用`--video`时复用现有录像流程。Upper首版使用键盘，原下肢Kit／手柄teleop继续按
[下肢说明](TELEOP.md)使用。

## 手臂碰撞模型

Upper固定基座与Mobile任务在训练、play和teleop中使用同一修复：保留G1的
`Physics=SimplifiedPhysX`，恢复左右肩部三个link、肘部、`wrist_roll`与`wrist_pitch`
共12个原本被停用的碰撞子树。标准资产的有效碰撞体由31个增至43个；下肢保留原简化碰撞几何，
Lower任务本身的资产配置保持原样。自碰撞开关继续启用。

恢复发生在机器人生成后、接触报告注册与环境复制之前，因此并行训练的每个环境均生效。
USD spawner、`pxr`及复制装饰器在生成机器人时才加载；启动前解析配置不会提前导入USD运行库，
由`SimulationApp`先初始化Kit的USD环境。
原始USD和缓存文件不被改写；缺少预期碰撞节点或有效碰撞体时会报告资产错误。
实现见[`utils/collisions.py`](../src/g1/tasks/upper_limb_agent/config/g1/utils/collisions.py)。

更新代码后，请退出并重新启动现有teleop进程，以加载恢复后的碰撞几何。
旧checkpoint保持动作与观测接口兼容；已有策略可能使用过穿模路径，恢复碰撞后的成功率需重新评估，
必要时在新碰撞条件下微调。`base_contact`仍表示`torso_link`的接触终止，该刚体包含躯干与头部碰撞体。

CPU/USD回归位于`tests/test_upper_arm_collisions.py`，覆盖实例化几何恢复、环境复制、接触报告、
资产错误和固定基座／Mobile／teleop配置。设置`G1_COLLISION_ASSET_PATH`为已解析的本地G1 USD路径，
可另外运行真实资产的31→43碰撞体核验。真实PhysX／Newton GL回放沿用下文的显式GPU测试。

## Newton面板

展开左侧Simulation中的 **Upper wrist teleop**。默认手动双手模式，初始目标为重置后的实际腕部位姿。

| 面板项 | 含义 |
| --- | --- |
| Control | Manual手动编辑；Automatic自动采样。切换控制来源会重置机器人。自动切手动时接管原目标。 |
| Active arms | Left左手、Right右手、Both双手；自动模式另有Sample，按训练手臂比例选择。 |
| Edit | 双手有效时选择编辑左目标、右目标或Pair整体；单手模式编辑有效侧。 |
| Task | regular常规、height_shifted高度、planar平面迁移、mixed训练混合分布。只显示当前任务具备的类型。 |
| Fine increments | 精细步长开关。 |
| Training stochastic actions | 使用checkpoint已学到的高斯动作分布采样；关闭时使用与play一致的均值动作。 |
| Next / reset | 重置机器人；自动模式重新采样，手动模式以当前重置腕部位姿重新初始化。 |
| Retry target | 从默认重置状态重新执行同一组目标，保留目标、有效手臂与时间预算。 |
| Current wrists | 手动模式将目标设为当前实际腕部位姿。 |
| Left/Right XYZ、RPY | 直接编辑各腕世界位置（米）和朝向（度）；内部统一使用xyzw四元数。 |

单手模式的另一侧仍由策略根据训练时的闲置手约束处理。面板将该侧误差显示为`idle`。
双手共同平移保持相对位置，共同旋转以目标中点为中心，同时旋转两腕朝向。

目标始终固定在世界空间，机器人移动后目标保持原位。双手手动目标超出配置中的间距、前后差、
高度差或相对朝向范围时，面板显示`outside configured training bounds`。该检查使用配置的双手几何边界，
只是分布提示；实际可达性、当前等级的更窄分布和碰撞情况由操作者结合演示判断。

## 键盘

Newton窗口需要处于焦点。输入框编辑、窗口失焦和暂停时屏蔽腕部按键；恢复后先松开原按键再重新按下。
按住位姿键会以固定间隔重复微调，松开后目标保持。面板暂停期间的修改在恢复步进后生效。

| 按键 | 功能 |
| --- | --- |
| J / L | 世界X负／正方向。 |
| K / I | 世界Y负／正方向。 |
| U / O | 世界Z负／正方向。 |
| Shift＋上述按键 | 绕对应世界轴负／正旋转。 |
| 1 / 2 / 3 | 左手／右手／双手有效。自动模式改变有效手臂会开始新回合。 |
| G | 精细档切换。 |
| M | 手动／自动切换。 |
| N / R / T | 下一组或重置／重试／目标设为当前腕部。 |

WASD/QE、方向键保留给Newton相机，Space继续暂停，Escape关闭窗口；也可在终端Ctrl+C退出。

可调默认值集中在[`WristTeleopCfg`](../src/g1/teleop/wrist_controls.py)：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `position_step_m` | 0.01 | 普通档位置步长，1cm。 |
| `angle_step_deg` | 2.0 | 普通档旋转步长，2°。 |
| `fine_multiplier` | 0.1 | 精细档为1mm、0.2°。 |
| `key_repeat_s` | 0.12 | 按住位姿键时的重复间隔，按控制时间累计。 |
| `status_period_s` | 0.1 | 跟踪误差等状态的刷新周期。 |
| `source/category/arm_mode` | manual/regular/2 | 初始目标来源、任务类型和有效手臂。 |

## 自动任务和时间

加载后恢复检查点课程等级与Lower动作范围，课程保持冻结。选择类别通过运行时接口覆盖，
不改动训练配置的概率和课程契约；例如当前1级checkpoint会按1级范围采样。

每回合一组目标，自动复用训练的单手／双手采样、额外高度偏移和平面变换。
`mixed`使用检查点恢复出的任务比例；旧检查点按基础20%平面选择，对应常规56%、高度24%、平面20%。
开启自适应采样的新检查点可恢复更高平面比例，遥操作期间保持该值。

| 类型 | 计时 |
| --- | --- |
| regular | 默认10秒，目标保持整个回合。 |
| height_shifted | 默认10秒，目标保持整个回合。 |
| planar | 迁移与转向估计＋稳定余量＋完整10秒评价窗口，受全局35秒硬上限校验。 |

结束后自动重置并采样下一组。时间预算独立于跟踪精度；提前达标仍保持目标。
遥操作常规目标采用一回合一目标，训练中的常规目标仍按原2～4秒刷新。
手动模式持续保持目标，取消超时，倾倒、低高度、躯干与膝接触等物理终止继续生效。
物理终止后，机器人重置并保持重置时的腕部位姿，面板保留已输入的左右目标。继续编辑或点
`Retry target` 会重新提交这组目标。可先用 Space 暂停，填好左右腕后再恢复步进。

## 诊断与记录

需要批量寻找失败目标时，使用[无窗口失败挖掘与本地三维回放](UPPER_FAILURE_MINING.md)。
该模式并行推理并保存重置前的动作与接触片段，之后通过浏览器时间轴查看，无需重新随机复现。

面板显示当前回合、课程等级、实际任务类别、执行／评价阶段、剩余时间，以及有效腕的位置和最短角误差。
Mobile另显示实际送给冻结Lower的`h/vx/vy/wz/px/py`，单位依次为m、m/s、m/s、rad/s、m、m。

均值／随机动作切换仅改变策略的动作采样方式，环境沿用play设置。它可用于检查动作探索噪声对
稳定性的影响；完整训练还包含并行采样等其他条件。手动切换发生在同一回合时，记录会注明`policy_mode`；
做对照建议在切换后用Retry target从相同初始场景重试。

记录位于`outputs/teleop/<时间-唯一编号>/session.jsonl`，不写入训练的TensorBoard事件文件。
每行一个事件：

| 事件 | 内容 |
| --- | --- |
| session | checkpoint绝对路径、种子和初始选择。 |
| target | 实际目标位姿、手臂模式、任务掩码、平面站位和时间预算。 |
| edit | 手动提交的世界目标。 |
| policy_mode | 均值／随机执行方式切换。 |
| ended | 回合时长与终止项名称，正常超时为time_out。 |
| interrupted | 操作员切换模式、重试或重置造成的中断。 |
| closed | 退出时的回合时长。 |

Retry target重新初始化物理场景并复用当前目标；同一种子仍需结合目标记录理解随机序列。
这些记录服务于交互诊断。需要横向统计时，应固定任务分布、等级和执行方式另做批量评估。

## 实现边界

- `g1.rl.teleop`复用启动、checkpoint解析和退出逻辑，按任务归属选择控制器。
- Upper命令项仅在安装运行时覆盖对象后改变采样行为；普通train/play沿用原有生命周期。
- 外部目标排队进入正常command update，存在一个控制周期响应延迟；观测历史只更新一次。
- `g1.teleop.newton`集中封装底层viewer访问、ImGui回调、按键去抖和焦点处理。
- Teleop 根据 checkpoint 选择 Mobile 观测布局。旧版使用 109 维单帧，新版使用 660 维历史观测。两种布局都使用 23 维动作和冻结 Lower。

## 验证记录

2026-11-16完成模块review和全局review：816项自动检查通过；真实物理回归覆盖下肢、固定基座Upper、
Mobile原两类路径、Mobile平面路径的训练／保存／回放。Newton窗口检查覆盖原生按键、面板绘制、
三类及混合采样、重试、超时、均值／随机执行切换和关闭清理，确认策略权重、课程等级与契约保持一致。
原下肢teleop通过输入、加载顺序和正常／异常清理回归；本次没有实测手柄硬件。

控制器与指令测试位于`tests/test_upper_teleop.py`、`tests/test_upper_teleop_commands.py`。
窗口测试`tests/test_upper_teleop_physics.py`通过`G1_UPPER_TELEOP_SMOKE=1`显式开启，默认使用上面示例的
本地模型及其冻结Lower，需要对应文件和GPU。
