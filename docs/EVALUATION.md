# eval 评估指南

`eval` 用于在固定条件下评估一个或多个兼容检查点，默认以 headless 方式运行，输出逐回合记录、汇总表和 TensorBoard 对比图。

- [第一部分：快速评估与查看结果](#第一部分快速评估与查看结果)：选择内置套件、运行评估、恢复任务、阅读 TensorBoard。
- [第二部分：创建 suite 与评估过程详解](#第二部分创建-suite-与评估过程详解)：编写自定义套件，了解参数、用例生成、模型加载和指标口径。

## 第一部分：快速评估与查看结果

### 1. 选择要回答的问题

suite（评估套件）规定测试条件、回合预算、种子和测试模块。内置套件统一为以下两个，均采用渐进上肢扰动：

| 套件 | 评估内容 | 适合回答的问题 | 默认规模：每个检查点 |
| --- | --- | --- | --- |
| [`g1_core`](../src/g1/rl/evaluation/suites/g1_core.yaml) | 基础任务跟踪，双臂和腰部目标保持默认姿态；站立、下蹲、行走比例为 33% / 33% / 34%，跳过蹲行 | 模型的基础技能如何？后期检查点是否比早期退化？ | 3 个种子，2,400 回合，3 个评估单元 |
| [`g1_robustness`](../src/g1/rl/evaluation/suites/g1_robustness.yaml) | 站立、下蹲、行走比例为 33% / 33% / 34%，跳过蹲行；先测无扰动跟踪，再扫描双臂比例 0.2、0.3、0.4、0.555，腰部固定为 0.1 | 上肢运动多大时开始影响下肢任务？哪个检查点更耐受扰动？ | 5 个条件 × 3 个种子，12,000 回合，15 个评估单元 |

两者默认都使用均值动作 `mean`，每回合上限 6 秒，每个种子、每个条件 800 回合，种子为 101、202、303。上肢比例表示关节默认姿态到限位余量的使用比例，例如 `0.555` 对应 55.5% 的目标范围。

内置高度偏移范围为 `[-0.15, 0.02] m`，采用标称初始状态，关闭观测噪声及训练随机化。它们衡量共同固定场景下的能力。复盘更深下蹲、Phase 2 物理随机化或 Phase 3 外部推扰，需要相应的自定义套件或新模块，见第二部分。

两个套件当前使用相同模式比例。比较检查点时，让所有检查点参加同一套件和相同执行配置。实际条件以本次 `manifest.json` 为准。

### 2. 运行一次评估

以下命令从项目根目录执行。示例中的检查点路径请替换为实际文件，输出目录使用新目录。`eval` 始终以 headless 运行，`--headless` 可省略。

**评估一个模型的基础能力：**

```powershell
uv run python -m g1.rl eval --suite g1_core --checkpoints "model=logs/rsl_rl/g1_flat/run_a/model_5500.pt" --output outputs/eval/core_quick
```

**比较多个检查点的上肢扰动耐受能力：**

```powershell
uv run python -m g1.rl eval --suite g1_robustness --checkpoints "early=logs/rsl_rl/g1_flat/run_a/model_5500.pt" "late=logs/rsl_rl/g1_flat/run_a/model_9999.pt" --num_envs 128 --device cuda:0 --output outputs/eval/robustness_compare
```

`--checkpoints` 接受任意数量的路径。`early=路径` 为检查点指定图例名称，建议使用简短英文名称。也可以传入带引号的通配符，例如 `"logs/rsl_rl/g1_flat/run_a/model_*.pt"`，由程序展开并去除重复文件。

动作方式通过 `--policy-output` 选择：

| 取值 | 使用方式 |
| --- | --- |
| `mean` | 使用策略均值，先判断模型学到的任务能力；内置套件的默认值 |
| `sample` | 从检查点保存的动作分布采样，观察动作随机性带来的影响 |
| `both` | 两种方式分别评估，回合数和评估单元数翻倍 |

第一次运行时，可先检查链路：

```powershell
uv run python -m g1.rl eval --suite g1_core --checkpoints "logs/rsl_rl/g1_flat/run_a/model_5500.pt" --episodes-per-seed 8 --seeds 101 --num_envs 4 --output outputs/eval/smoke
```

这个小规模测试用于确认环境、模型加载和报告生成。正式性能比较使用套件的完整预算或另行约定的多种子预算。

### 3. 预览计划、查看进度与恢复

在新建评估命令中增加 `--plan-only`，会生成计划并打印评估单元数，随后退出，适合先确认运行规模：

```powershell
uv run python -m g1.rl eval --suite g1_core --checkpoints "logs/rsl_rl/g1_flat/run_a/model_5500.pt" --plan-only --output outputs/eval/core_plan_new
uv run python -m g1.rl eval --resume outputs/eval/core_plan_new
```

默认显示总体进度、当前检查点和条件、执行阶段及回合进度：

```text
总进度 [######----------] 12/30 (40%) | 已用 00:08:20 | 预计剩余 00:12:30
当前：early · upper_body_sweep · 双臂 0.4 · 腰部 0.1 · mean · seed 101
阶段：执行评估 | 单元已用 00:00:25 | 本阶段 00:00:08 | 本次完成 10 · 已跳过 2
回合 [#########-------] 450/800 (56%) | 批次 3/7 (当前 4) | 当前批次控制步 180/360
```

一个评估单元是“检查点 × 测试条件 × 动作方式 × 种子”。总体进度按成功单元累计，回合进度包含提前失败的测试回合。回合达到 100% 后还需保存结果和关闭仿真；首个单元完成后才逐渐形成剩余时间估计。

中断后使用 `--resume` 继续，已完成单元会跳过，未完成单元从头执行：

```powershell
uv run python -m g1.rl eval --resume outputs/eval/core_quick
```

恢复沿用原计划。`--progress plain` 可改用普通文本进度，`--progress off` 可关闭进度；IDE 非交互输出会自动采用普通文本。调整测试条件、检查点或执行配置时，新建评估批次。

### 4. 打开结果与 TensorBoard

结束后，终端会打印报告位置和 TensorBoard 命令。输出目录内的 `report_latest.json` 指向最新报告，其中包含：

| 字段 | 用途 |
| --- | --- |
| `summary_csv` | 逐检查点、条件和模式的指标表，适合精确比较数值 |
| `summary_json` | 结构化汇总及缺少种子的分组信息 |
| `tensorboard_dir` | 当前报告的 TensorBoard 日志目录 |

PowerShell 可以直接读取最新报告路径，无需手动寻找报告版本。下面打开前述基础评估的结果；查看已有的 `outputs/eval/core_plan` 时，替换第一行路径即可：

```powershell
$report = Get-Content -LiteralPath "outputs/eval/core_quick/report_latest.json" -Raw -Encoding UTF8 | ConvertFrom-Json
uv run tensorboard --logdir "$($report.tensorboard_dir)"
```

打开 TensorBoard 打印的访问地址，按以下顺序查看：

1. 进入 **Custom Scalars**，使用预设的四个分组面板。
2. 在左侧 Runs 中选择同一模块和动作方式，例如 `upper_body_sweep/mean/early` 与 `upper_body_sweep/mean/late`。检查点分别对应 run，颜色随 run 区分。
3. 将平滑设为 **0**。每张图比较同一模式、同一指标下的多个检查点。
4. `tracking` 横轴为固定场景编号 0，只有一个统计点，可结合悬停数值或 `summary.csv` 比较。`upper_body_sweep` 横轴为双臂目标范围比例的千分值，200 表示 0.2，555 表示 0.555。

`--logdir` 指向**一个报告版本的 `tensorboard` 目录**。每次汇总会保留独立版本，选定版本可使显示的曲线集合保持清楚。

已有回合结果需要重新生成报告时执行：

```powershell
uv run python -m g1.rl eval --resume outputs/eval/core_quick --report-only
```

这一步读取已有结果，无需启动仿真。已产生结果但尚未完成全部指定种子的条件，会列入 `summary.json` 的 `incomplete_groups`，补齐后再输出对应统计点；完整待执行清单见 `manifest.json`。

### 5. 如何快速判断模型表现

| 面板 | 先看什么 | 观察与判断 |
| --- | --- | --- |
| `01 任务完成` | 联合成功率、失败率，其次是存活比例与最差模式联合成功率 | 成功率高且失败率低，说明任务完成和稳定性较好；总体良好而最差模式偏低，说明存在技能短板 |
| `02 跟踪精度` | 高度、平面速度、转向、骨盆水平误差 | 成功率接近时比较误差；失败率低而成功率低时，判断哪个跟踪通道拖累任务 |
| `03 失败定位` | 各通道达标率、终止原因、联合成功率跨种子标准差 | 区分精度不足与提前终止；跨种子波动较大时，增加样本并检查具体模式 |
| `04 测试条件` | 有效回合数、实际指令、扰动频率和目标范围比例 | 确认样本完整、难度一致；模型提前失败会改变实际执行时长及目标暴露 |

先比较逐模式的联合成功率与失败率，再定位误差，最后核对测试条件。联合成功要求正常完成回合并同时满足各项精度阈值，因此“存活到结束”和“完成任务”有不同含义。

扰动扫描重点看**性能从哪个幅度开始明显下降，以及下降有多快**。均值动作良好、随机动作明显下降时，动作分布方差值得关注；同一固定条件下后期检查点的均值动作也退化时，可进一步检查技能保持和训练更新。小幅差异结合回合数、区间和跨种子波动判断，详细定义见第二部分。

## 第二部分：创建 suite 与评估过程详解

### 1. 如何创建新的 suite

先确定要验证的问题及改变的变量。例如比较深蹲能力时调整高度范围，比较上肢幅度时调整扫描点，其余基准条件保持一致。

1. 复制一个接近目标的内置 YAML 到自己的配置目录，修改 `suite` 名称，例如 `g1_replay_0345_v2`。
2. 修改基准或模块参数，使用带版本的名称保留评估标准。
3. 用文件路径传入 `--suite`，先执行 `--plan-only` 检查配置和规模，再通过 `--resume` 执行。
4. 按第一部分打开报告，在同一套件下比较检查点。

`--suite` 接受内置名称以及 YAML/JSON 文件路径，`--eval-config` 是别名。只调整现有参数时，创建配置文件即可。新增外部推扰、物理参数扫描等被测变量时，需要扩展测试模块，见本部分第 8 节。

以下完整示例将高度范围扩展到 `[-0.345, 0.05] m`，用于复盘较深下蹲，并同时比较均值与随机动作。可保存为项目根目录下的 `g1_replay_0345_v2.yaml`：

```yaml
schema_version: 1
suite: g1_replay_0345_v2
task: Template-G1-v0
benchmark:
  upper_body_motion:
    max_velocity: 2.0
    max_acceleration: 8.0
    min_duration_s: 0.5
  episode_length_s: 6.0
  control_dt: 0.016666666666666666
  episodes_per_seed: 800
  mode_weights: [0.25, 0.25, 0.25, 0.25]
  seeds: [101, 202, 303]
  observation_noise: false
  reset_mode: nominal
  pelvis_target_speed_mps: 0.02
  pelvis_zero_probability: 0.2
  envelope:
    depth_frac: 0.8
    vx_floor: 0.2
    wz_floor: 0.3
  ranges:
    height_offset: [-0.345, 0.05]
    lin_vel_x: [-0.3, 0.6]
    lin_vel_y: [-0.15, 0.15]
    ang_vel_z: [-0.8, 0.8]
    pelvis_offset_x: [-0.03, 0.03]
    pelvis_offset_y: [-0.03, 0.03]
  resampling_time_range: [2.0, 4.0]
  disturbance_interval_s: [1.5, 3.0]
  thresholds:
    height_m: 0.03
    velocity_xy_mps: 0.5
    yaw_radps: 0.4
    pelvis_xy_m: 0.02
policy_outputs: [mean, sample]
modules:
  - name: tracking
    arm_fraction: 0.0
    waist_fraction: 0.0
  - name: upper_body_sweep
    arm_fractions: [0.2, 0.3, 0.4, 0.555]
    waist_fraction: 0.1
```

```powershell
uv run python -m g1.rl eval --suite g1_replay_0345_v2.yaml --checkpoints "early=logs/rsl_rl/g1_flat/run_a/model_5500.pt" "late=logs/rsl_rl/g1_flat/run_a/model_9999.pt" --plan-only --output outputs/eval/deep_compare
uv run python -m g1.rl eval --resume outputs/eval/deep_compare
```

### 2. 参数、覆盖规则与运行规模

| 字段 | 含义与约束 |
| --- | --- |
| `schema_version` / `suite` / `task` | 当前格式版本为 1；套件名称使用字母、数字、下划线、点或连字符，以字母或数字开头；任务填写已注册的环境 ID |
| `episode_length_s` / `control_dt` | 回合上限与控制周期，单位秒；回合长度为控制周期的整数倍，控制周期须与任务匹配 |
| `episodes_per_seed` | 每个检查点、测试条件、动作方式、种子的总回合数 |
| `mode_weights` | 顺序为站立、下蹲、行走、蹲行；非负权重归一化后分配回合，零权重模式跳过 |
| `seeds` | 非空、不重复的非负整数评估种子列表 |
| `observation_noise` / `reset_mode` | 当前支持 `false` 与 `nominal`，即关闭观测噪声、使用标称重置 |
| `ranges` | 高度与骨盆偏移单位 m，平移速度 m/s，转向速度 rad/s；高度下界为负，骨盆区间包含零 |
| `resampling_time_range` | 指令重采样间隔范围，单位秒 |
| `disturbance_interval_s` | 上肢目标重采样间隔范围，单位秒 |
| `upper_body_motion` | 渐进轨迹的速度、加速度和最短时长；省略时使用评估默认值 `2.0 / 8.0 / 0.5`，建议显式填写 |
| `upper_body_target_source` | 策略的上肢目标观测来源，默认 `final_goal`；此前渐进参考观测模型使用 `reference`，此项参与配置哈希及兼容性检查 |
| `pelvis_target_speed_mps` / `pelvis_zero_probability` | 有效骨盆目标的最大平滑速度（m/s）；静止模式采样零骨盆偏移的概率 |
| `envelope` | 蹲行深度比例与最深可行走高度处的前进、转向上限锚点，沿用训练的梯形包络语义 |
| `thresholds` | 回合平均误差的达标阈值，采用严格小于判断 |
| `policy_outputs` | `mean`、`sample` 的非空列表，取值不重复 |
| `modules` | 选择 `tracking`、`upper_body_sweep` 或两者；每种模块最多配置一次 |
| `arm_fraction` / `arm_fractions` / `waist_fraction` | 目标范围比例在 `[0,1]` 内，按 0.001 递增；扫描列表各点不重复，按升序执行 |

`tracking` 使用单个双臂比例和腰部比例，内置套件均为 0。`upper_body_sweep` 使用双臂比例列表，腰部比例在整个扫描中固定。关节目标围绕默认姿态，按目标方向到限位的余量缩放，保留非对称限位映射。

配置按严格字段校验，未知字段和拼写错误会报错。新批次支持以下 CLI 覆盖，最终展开配置写入 manifest：

| 类别 | 参数 | 使用约定 |
| --- | --- | --- |
| 基准覆盖 | `--task`、`--seeds`、`--episodes-per-seed`、`--policy-output` | 改变本批次的评估条件；长期复用时写入套件 |
| 执行设置 | `--num_envs`（默认 128）、`--device`（默认 `cuda:0`）、`--physics`（默认 `isaacsim_physx`） | 同次横向比较保持一致，正式跨批次比较还需对齐设备类型与软件版本 |
| 模型与输出 | `--checkpoints`、`--agent-config`、`--output` | 选择检查点、可选的共享智能体配置及新输出目录 |
| 显示设置 | `--progress auto/plain/off` | 只改变显示，恢复时也可切换 |

一个评估单元启动一个独立 worker。单元串行执行，worker 内部通过 `num_envs` 并行运行环境。因此估算规模时同时考虑测试回合和进程初始化成本：

```text
评估单元数 = 检查点数 × 所有模块的条件总数 × 动作方式数 × 种子数
总回合数   = 评估单元数 × episodes_per_seed
```

例如示例套件有 5 个条件、2 种动作方式、3 个种子，两个检查点共 60 个单元、48,000 回合。先运行 `mean` 筛选，再对候选运行 `both`，可以缩小首轮预算。

### 3. 固定用例与随机场景如何生成

**模式配额。** 程序按归一化权重计算各模式回合数，使用最大余数法补足整数；余数相同时按站立、下蹲、行走、蹲行排序。每个正权重模式至少获得一个回合，预算不足时明确报错。例如当前 core 每种子分配 264 / 264 / 272 / 0 回合，robustness 分配 200 / 200 / 200 / 200 回合。

**回合收集。** 每个 batch 同步重置，只收集各槽位被分配的第一个自然结束回合，异常终止和正常超时均计入。已完成槽位自动重置后的回合跳过；各 batch 最长运行一个回合上限。提前失败会减少实际控制步数，模式的回合占比仍符合配额。

**随机输入。** 任务目标、重采样时钟和归一化上肢目标，由评估种子、模式、用例编号和随机流名称生成。不同检查点使用相同指定用例，某个模型提前失败也保留后续用例的输入。上肢扫描复用归一化目标和事件时间，只改变幅度映射。

指令保持以下训练语义：

| 模式 | 高度与速度 | 骨盆目标 |
| --- | --- | --- |
| 站立 | 默认高度、零速度 | 原始目标随指令时钟更新，有效目标逐步平滑 |
| 下蹲 | 高度可在回合内重采样、零速度 | 同站立 |
| 行走 | 默认高度；速度在回合开始时确定并保持 | 按任务的移动模式规则处理 |
| 蹲行 | 高度、速度在回合开始时确定并保持，受高度—速度包络约束 | 按任务的移动模式规则处理 |

**控制与统计时序。** 终止状态在物理重置前提取，每步使用实际执行的有效指令计算误差。测试控制器在策略观测前发布新指令和上肢最终目标；共享运动组件在动作应用时推进渐进参考。观测按 `upper_body_target_source` 选择最终目标或当前 PD 参考，段接续时的提示时机见 [观测与续训](UPPER_BODY_MOTION.md#观测与续训)。

**固定场景。** 两套内置配置均关闭课程和训练随机化事件，包括 Phase 2 质量、摩擦、执行器增益随机化和 Phase 3 外部推扰。每个 batch 调用统一环境重置：根与关节恢复标称状态，上肢位置目标对齐重置姿态，实际速度、目标速度和目标力矩归零，运行段与待执行目标清空。增加观测噪声时，需要扩展逐用例噪声回放和配置版本。

**动作随机性。** `mean` 使用策略均值；`sample` 使用已保存的动作分布及独立 Torch 随机流，避免消耗场景随机序列。动作随机种子按 batch 和步组织，随机动作对照需保持相同 `num_envs` 和回合编排。

### 4. 模型加载与跨 Phase 比较

当前支持 RSL-RL `OnPolicyRunner`、包含 `actor_state_dict` 的检查点。评估恢复 actor 权重、观测归一化状态和动作分布参数，冻结模型。训练迭代数与阶段记录为来源信息，测试场景由套件确定。

任务兼容信息优先读取检查点内的 G1 元数据，其次读取相邻 `params/env.yaml`。基础检查覆盖机器人资源、动作关节及缩放、策略历史长度、上肢关节组和控制周期。存在 `env.yaml` 时还校验观测处理、排列、缩放与扩展动作语义；两类来源都存在时同时使用。仅有基础元数据时，结果记录扩展语义校验的范围限制。

智能体构造配置按以下顺序选择：

1. `--agent-config PATH` 指定的共享配置。
2. 各检查点旁的 `params/agent.yaml`。
3. 注册任务的默认智能体配置。

保留训练的 `params` 目录，有助于恢复对应网络和验证语义。多个检查点网络不同，可分别读取各自的相邻配置。缺少任务元数据及 `env.yaml` 的历史权重，可在确认来源后使用 `--allow-unverified-checkpoint`；该选项放行缺失的任务语义信息，actor 权重格式和形状仍严格校验。

兼容的 Phase 1/2/3 检查点可参加同一共同基准。训练阶段、回合长度和课程进度允许与套件不同。Phase 2/3 模型在 core 下的成绩表示基础技能保持，物理随机化敏感性和冲击恢复能力需专用模块测试。

### 5. 指标定义与统计口径

| 面板指标 | 汇总字段 | 定义 |
| --- | --- | --- |
| 联合成功率 ↑ | `joint_success_rate` | 正常到达回合时限，同时通过高度、平面速度和转向条件；站立、下蹲还需通过回合平均骨盆误差条件 |
| 失败率 ↓ | `failure_rate` | 出现异常终止的回合占比 |
| 存活比例 ↑ | `survival_fraction` | 各回合实际时长 / 回合上限，再对回合求平均 |
| 最差模式联合成功率 ↑ | `worst_mode_joint_success_rate` | 启用模式中最低的联合成功率，仅在总体行输出 |
| 高度误差 (m) ↓ | `error_height_m` | 每回合高度绝对误差均值，再对回合等权平均 |
| 平面速度误差 (m/s) ↓ | `error_velocity_xy_mps` | 每回合机体系平面速度误差范数均值，再对回合等权平均；静止模式用于观察漂移 |
| 转向误差 (rad/s) ↓ | `error_yaw_radps` | 每回合机体系 yaw 角速度绝对误差均值，再对回合等权平均 |
| 骨盆水平误差 (m) ↓ | `error_pelvis_xy_m` | 双踝水平参考系下，根刚体位置相对有效骨盆目标的二维误差范数，按有效静止模式控制步加权 |
| 高度达标率 ↑ | `height_tracking_success_rate` | 回合平均高度误差低于阈值的回合占比 |
| 平面速度达标率 ↑ | `velocity_xy_tracking_success_rate` | 回合平均平面速度误差低于阈值的回合占比 |
| 转向达标率 ↑ | `yaw_tracking_success_rate` | 回合平均转向误差低于阈值的回合占比 |
| 速度联合达标率 ↑ | `velocity_tracking_success_rate` | 同时通过平面速度和转向条件的回合占比 |
| 终止原因 | `termination/<name>` | 回合结束时该条件被触发的占比，多个原因可同时出现 |

默认阈值为高度 `< 0.03 m`、平面速度 `< 0.5 m/s`、转向 `< 0.4 rad/s`、静止模式回合平均骨盆误差 `< 0.02 m`。独立通道达标率包含提前失败回合，用于定位精度问题；联合成功率同时要求正常完成。失败回合较短，连续误差要结合失败率和存活比例阅读。

`timeout_rate` 表示正常到达时限；同一帧异常终止与超时同时发生时按失败计算。`pelvis_step_success_rate` 表示有效控制步中骨盆误差低于阈值的比例，与回合平均骨盆判定使用不同统计单位。行走、蹲行的骨盆指标留空。

测试条件记录有效回合数、控制步数、实际扰动频率、双臂和腰部比例，以及实际执行的高度、平面速度、转向目标。目标均值按控制步加权，实际扰动频率为事件次数 / 有效回合总时长。CSV 另有平均回合时长、模式回合占比和控制步占比等字段。

总体与不确定性按以下规则汇总：

| 字段或后缀 | 解释 |
| --- | --- |
| `aggregate` | 按配置模式权重汇总任务性能；目标、扰动暴露、骨盆误差继续按实际时长或有效步汇总 |
| `original_three` | 启用蹲行时，额外对原三模式重新归一化汇总，观察原技能保持 |
| `_lower_95` / `_upper_95` | 逐模式比例的 95% Wilson 区间；加权总体不套用普通二项 Wilson 区间 |
| `_seed_std` | 跨评估种子的样本标准差；单种子时省略 |

每个图表点汇总该条件的全部指定评估种子，已有结果但种子未齐的分组先列入 `incomplete_groups`。区间反映给定模型与基准下的抽样不确定性，跨种子标准差反映评估用例变化；多次独立训练的波动需要另外比较。

### 6. 执行、恢复、进度与文件记录

默认输出到 `logs/eval/g1_flat/<批次>`，`--output` 可指定位置：

```text
<评估目录>/
├── manifest.json
├── report_latest.json
├── cases/
│   └── <job_id>/
│       ├── worker.log
│       ├── progress.json
│       ├── effective_env.yaml
│       ├── effective_agent.yaml
│       ├── result.json
│       └── error.json              # worker 失败时生成
└── reports/
    └── <报告版本>/
        ├── summary.csv
        ├── summary.json
        └── tensorboard/
            ├── tracking/mean/<检查点名称>/events...
            └── upper_body_sweep/
                ├── mean/<检查点名称>/events...
                └── sample/<检查点名称>/events...
```

**计划与溯源。** `manifest.json` 保存规范化套件、执行设置、检查点及配置哈希、源码指纹、运行库版本和全部单元。`--plan-only` 检查配置和文件，并生成计划；模型内容与任务接口的完整兼容校验在 worker 加载时进行。每个单元保存实际环境和智能体配置，`result.json` 保存逐回合记录及运行元数据。检查点标签按不区分大小写的规则去重，冲突时追加路径摘要；路径中包含 `=` 的普通文件和通配符也可使用。

**恢复校验。** `--resume` 沿用原配置，校验 manifest、项目 Python 源码、模型与配置文件哈希、运行库版本及可编辑上游的 Git 状态。条件改变后建立新批次。已完成结果经过校验后跳过，失败或中断单元重新执行。`--report-only` 使用原有来源记录与有效结果，在分析机器上也可重新汇总，无需原模型文件和仿真运行环境。

**实时阶段。** 进度涵盖配置检查、依赖加载、仿真启动、环境创建、模型加载、回合执行、结果保存、仿真关闭及报告生成。主进程在子进程退出且结果校验通过后增加成功单元数；关闭后端时直接退出的情况也由主进程确认。失败单元保持未完成。

**时间预估。** 已用时间为本次执行的墙钟时间。剩余时间参考本次最近最多 8 个成功单元的完整耗时，包括启动和关闭成本；恢复时跳过的单元只计进度。暂无样本或当前单元超过历史平均耗时时显示“估算中”。

**显示与状态文件。** `auto` 在支持 ANSI 的交互终端原位刷新，其余场景采用 `plain`。普通文本在阶段切换、回合进度跨过约 10% 时输出，长阶段每约 15 秒补一行；`off` 保留错误和恢复提示。主进程进度写入标准错误流，仿真输出写入 `worker.log`。`progress.json` 约每 0.5 秒更新，阶段和批次边界立即更新，并用尝试标识隔离旧进度。正式成绩以 `result.json` 为准。

**报告版本。** 每次汇总新建报告目录并更新 `report_latest.json`，历史报告保留。CSV 每行对应检查点、模块、动作方式、扰动条件和模式/总体，使用英文机器字段；TensorBoard 的 Custom Scalars 使用中文图名。区间上下界等明细可在 Scalars 中查看。

### 7. 常见问题

| 现象或需求 | 处理方式 |
| --- | --- |
| worker 失败 | 查看 `cases/<job_id>/worker.log` 与 `error.json`，修复运行条件后 `--resume` |
| 提示源码、软件版本或检查点变化 | 使用相同目标配置创建新批次；旧结果仍可 `--report-only` 汇总 |
| 缺少任务元数据 | 找回相邻 `params/env.yaml`；确认来源后可选择 `--allow-unverified-checkpoint` |
| 权重形状不匹配 | 使用原训练的 `params/agent.yaml`，核对机器人、观测历史和动作接口 |
| 扰动扫描缺少图表点 | 检查 `incomplete_groups`，补齐指定种子的评估单元 |
| 改变模式比例、回合长度、指令范围或扫描点 | 创建新套件并使用新输出目录 |
| 评估其他物理后端 | 通过 `--physics` 选择已支持的配置，单独验证并保存结果 |
| 查看视频或人工控制 | 使用 `play` 或 `teleop` |

### 8. 何时需要新增测试模块

模块定义被测变量和扫描方式，suite 将模块与固定基准组合起来。现有两个模块支持基础跟踪及双臂幅度扫描；外部推扰、质量随机化等新变量需要代码支持，继续复用 `eval` 入口。

1. 在 [`config.py`](../src/g1/rl/evaluation/config.py) 定义字段与校验，在 [`modules.py`](../src/g1/rl/evaluation/modules.py) 展开条件并规定横轴单位。
2. 在任务适配器 [`evaluation.py`](../src/g1/tasks/lower_limb_agent/evaluation.py) 实现按用例隔离的回放，保留标准配额和终止收集接口。
3. 复用逐回合记录；需要恢复时间等专属指标时，在 [`reporting.py`](../src/g1/rl/evaluation/reporting.py) 补充汇总和图表。
4. 新建版本化套件，验证输入一致性、条件隔离、统计口径、恢复执行及日志显示。

### 9. 验证范围

2026-10-19 完成的项目回归测试为 272 项通过，默认跳过显式启用的 GPU 测试；另行通过 1 项真实 headless Isaac Sim/PhysX 集成测试。集成测试使用合成策略、4 个环境和 0.2 秒回合，覆盖两模块、两种动作方式、报告生成、实时进度与恢复跳过，共 4 个单元、16 个回合。它验证运行链路；模型性能通过正式套件与真实检查点评价。

普通回归测试使用 `uv run pytest`。手动启用 GPU 链路测试（PowerShell）：

```powershell
$env:G1_EVAL_PHYSICS_SMOKE = '1'
uv run --extra isaacsim pytest tests/test_eval_physics.py --basetemp outputs/pytest-eval-physx-local -q
```

## 上肢运动参数

所有套件统一使用五次渐进参考，运动速度、加速度和最短时长写入 `benchmark.upper_body_motion`。配置和执行统计见 [上肢渐进扰动](UPPER_BODY_MOTION.md)。自定义 suite 建议显式填写该字段。

```yaml
benchmark:
  upper_body_target_source: final_goal
  upper_body_motion:
    max_velocity: 2.0
    max_acceleration: 8.0
    min_duration_s: 0.5
```

`final_goal` 显示当前段的最终目标，空闲且存在待执行目标时提前显示该目标；执行器继续按既有时序追踪渐进参考。`reference` 显示当前 PD 位置参考，用于复测此前以参考位置作为观测的渐进模型。eval 会识别旧配置的观测来源，并检查它与 suite 的设置是否一致，详细规则见 [观测与续训](UPPER_BODY_MOTION.md#观测与续训)。

此前带版本后缀的内置名称统一为 `g1_core`、`g1_robustness`。旧自定义 YAML 中的 `upper_body_motion.profile` 应删除；格式字段 `schema_version: 1` 继续保留。旧输出目录保留历史结果，新源码与配置需要新建评估目录。
