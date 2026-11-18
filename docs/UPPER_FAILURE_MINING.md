# Upper Mobile 无窗口失败挖掘与本地回放

`g1.rl failures`针对`Template-G1-Upper-Mobile-v0`提供批量失败采集和浏览器三维回放。
采集运行PhysX和检查点策略，关闭窗口与渲染；查看仅使用本地记录，不启动Isaac Sim或加载策略。

## 快速使用

先用小批次确认当前机器的物理运行时和传感器正常：

```bash
uv run python -m g1.rl failures mine \
  --checkpoint "logs/rsl_rl/g1_upper_mobile/<本版训练目录>/model_XXXX.pt" \
  --num_envs 32 --episodes 100 --max_failures 10 \
  --output outputs/failures/planar_smoke
```

批量挖掘平面任务，最多完成5000回合或保存100个失败样本，保存失败前2秒：

```bash
uv run python -m g1.rl failures mine \
  --checkpoint "logs/rsl_rl/g1_upper_mobile/<本版训练目录>/model_XXXX.pt" \
  --num_envs 256 --episodes 5000 --max_failures 100 \
  --history_s 2 --category planar --policy_output mean --seed 42 \
  --output outputs/failures/planar_mean
```

使用`--policy_output sample`检查训练随机动作，另选输出目录，例如`planar_sample`。
`--category regular|height_shifted|mixed`可采集其他类别；`mixed`采用恢复的任务比例。
默认使用检查点保存的常规目标刷新范围。常规目标仍在回合中按距离刷新。
课程等级和采样调度从检查点恢复并冻结；碰撞、失败阈值、奖励和冻结Lower保持当前配置，策略不执行学习更新。
省略`--level`时使用检查点保存的当前等级。例如`2026-11-17_17-40-41/model_7999.pt`保存的是level 3，
默认整个采集过程固定在level 3，不受检查点中上一完整统计窗口`checked_level=2`影响。

使用`--level 0`至`--level 10`可以指定本次采集的全局课程等级。该选项同时改变平面站位范围、常规目标缩放、
高度任务偏移和Lower指令范围。选择其他等级时，策略权重及训练checkpoint保持原样，混合类别的采样比例仍从checkpoint恢复并冻结。
例如使用最新模型单独采集level 3的平面失败：

```bash
uv run python -m g1.rl failures mine \
  --checkpoint logs/rsl_rl/g1_upper_mobile/2026-11-17_17-40-41/model_7999.pt \
  --level 3 --category planar --policy_output mean \
  --num_envs 256 --episodes 5000 --max_failures 100 --history_s 2 \
  --output outputs/failures/1740_level3_planar_mean
```

改为`--level 2`并使用新的输出目录，可以检查同一模型在前一档难度下的失败。
`run.json`中的`checkpoint_level`保存恢复的等级，`requested_level`保存显式参数（省略时为null），
`level`保存实际采集等级；每个回合和失败片段的索引也记录实际等级。

新版分段站位课程及20 N臂躯接触定义与旧检查点的课程契约不同，采集使用对应本版配置的检查点。
旧采集目录仍可直接回放；从旧模型续训新课程时，在训练命令中加
`env.commands.wrist_pose.range_curriculum.reset_on_load=true`。

`--max_steps`默认30000个向量化控制步；达到回合数、失败数或步数上限中的任意一项结束。
Ctrl+C会保留已经写出的样本并标记为`interrupted`。
指定的输出目录必须是新目录，省略`--output`时自动生成带时间戳的目录。

采集完成后启动本地查看服务：

```bash
uv run python -m g1.rl failures view --run outputs/failures/planar_mean
```

默认打开`http://127.0.0.1:8765`。通过远程IDE使用时转发8765端口，并加`--no_browser`；
`--port 8766`可以更改端口。服务只绑定本机地址，Ctrl+C关闭服务。

## Regular 切换评估

`--category regular` 和 `--category mixed` 自动统计已接收回合中的每次常规目标切换，含安全完成的切换。
以下示例使用3万轮小网络的Level 2检查点。评估两个模型时，更换checkpoint和输出目录，其余条件取相同值。

```bash
uv run python -m g1.rl failures mine \
  --checkpoint logs/rsl_rl/g1_upper_mobile/2026-11-18_00-02-25/model_29999.pt \
  --level 2 --category regular --policy_output mean --seed 42 \
  --regular_refresh_s 2 4 --num_envs 256 --episodes 5000 --max_failures 5000 \
  --output outputs/failures/regular_switch_0002_l2
```

`--regular_refresh_s 2 4`指定距离计时的评估下限和上限。加载时先使用checkpoint课程里保存的训练范围和同目录`params/agent.yaml`，再应用评估覆盖。`run.json`记录训练与评估范围。省略开关时沿用checkpoint的刷新范围。正式对照建议检查`summary.json`中的`status`为`episodes`，两次采集均达到所需回合数。

- `switches.jsonl`逐次记录新旧目标、切换时间、方向、活动手最大位移、切换前一帧的到位与接触状态、换目标后 0.2 秒内的峰值臂躯接触力，以及 1 秒内能否达到新目标。位姿达标沿用当前课程容差，默认 15 mm 和 0.1 rad。
- `summary.json`的`regular_switches.overall`给出整体计数与比例，`by_group`按单双手、方向、跳变距离、切换前到位和接触状态分组。优先比较`arm_mode=both`、`pre_switch_ready=true`、`pre_switch_contact=false`且位移档相同的`retract`、`forward`与`lateral`。比例`rate`乘100即每百次切换的事件数。也可用`episode_id`关联原有的失败回放。
- 方向使用初始骨盆轴。任一活动手向后移动至少 3 cm 归为`retract`，其余中任一手向前移动至少 3 cm 归为`forward`。其余情况下，任一手的侧移达到 3 cm 且不少于该手的高度变化时归为`lateral`。位移分档用活动手中最大的三维目标跳变量。
- 接触统计使用切换后物理子步上的逐臂/手通道法向力模长之和，分别报告大于 1 N 和大于硬终止阈值的比例。窗口提前结束时，已观察到的事件进入分子和分母，其他观察不足的结果记为`null`。新目标到位统计把 1 秒内到位前物理终止计作失败，提前超时且尚未到位的结果记为`null`。各比例的分母见`assessed`。
- 回合起始目标单独计数。`regular_failures_before_first_switch`表示首次换目标前就终止的 regular 回合数。整体每10机器人秒的失败频率仍在`failure_per_10s_by_category.regular`。

## 左右镜像切换短评估

先从已有 regular 切换记录生成固定用例。此命令可以在无 GPU 的环境运行。

```bash
uv run python -m g1.rl failures pairs \
  --run outputs/failures/regular_com_v5_l1 \
  --output outputs/failures/v5_l1_switch_pairs.json
```

默认从 **8–12 cm** 右单手切换中抽取 100 对「向内 + 抬高」、20 对「向外 + 抬高」、20 对「向内、少抬高」。主组和对照组按几何分组后随机抽取。主组未抽到的已知早期硬碰撞可加入独立 `known_failure` 组。可通过 `--in_up`、`--controls`、`--regressions`、`--seed` 调整；`--regressions 0`关闭额外回归用例。套件保存原始目标对和来源记录编号。

再在训练机运行 headless 配对评估。

```bash
uv run python -m g1.rl failures mine \
  --checkpoint logs/rsl_rl/g1_upper_mobile/2026-11-18_12-24-35/model_7999.pt \
  --category regular --policy_output mean --seed 42 \
  --switch_pairs outputs/failures/v5_l1_switch_pairs.json --num_envs 256 \
  --output outputs/failures/v5_l1_switch_paired
```

每个用例左右各运行一次，等级取自套件。左侧目标在初始骨盆平面镜像位置和姿态扰动，并用左腕默认坐标轴表达。每个回合从标称重置姿态开始，前 **4 秒**跟踪旧目标，随后切换一次并观察 **1 秒**。臂躯硬终止用 **20 N**。该模式按套件用例数完成评估，`--max_steps`仍可设总控制步上限。正式对照使用同一套件、种子和环境数，并检查两次均以`status=episodes`结束。

- `summary.json` 中先看 `switch_pair_evaluation.by_group.in_up`。`right_only_contact`、`left_only_contact`、`both_contact`、`neither_contact`统计两侧均可比较的目标对，碰撞窗口为切换后 0.2 秒。开始切换前失败、未到位、已接触或碰撞窗口观察不足的目标对保留为 `not_comparable`；未跑满两侧标为 `incomplete`。
- 各组 `left` 和 `right` 给出切换前失败数、早期硬碰撞和 1 秒内到位的分子和分母。`by_pair`给出两侧的 `episode_id`，可与 `failures.jsonl` 的失败片段关联。原套件另存为 `switch_pairs.json`。
- 回放沿用 `uv run python -m g1.rl failures view --run outputs/failures/v5_l1_switch_paired`。

## 查看内容

- 根据任务类别、终止项选择失败样本；可切换上一例、下一例。
- 原始机器人可视网格随记录的刚体姿态回放；拖动旋转、滚轮缩放。
- 暂停、逐帧、拖动时间轴、调整播放速度。
- 腕部目标、接触点及法向力箭头；躯干净法向力与逐对象力曲线。
- 接触对象、峰值、观察到的接触时长、执行/评价阶段和目标参数。

回放显示当时记录的状态与力，不重新执行动作。因此随机动作失败也可直接查看，不依赖种子复现。
可视网格用于定位动作；物理接触依据来自PhysX传感器，网格不等价于凸包碰撞外形。
`torso_link`同时包含头部和躯干，body级接触报告通过接触点辅助定位二者，未自动拆成两个刚体。

## 接触数据与采集时序

额外传感器只监测`torso_link`，过滤对象为资产中解析出的其他机器人刚体，以及实际地面Plane路径。
训练侧的`torso_contact_pairs`负责20 N臂躯/1 N其他对象的分层终止；全机器人传感器继续负责膝部终止。
诊断记录通过RecorderManager的`record_post_step`回调，在终止计算后、自动重置前采集，包含真正的终止帧。

机器人姿态、动作与目标按控制步保存（当前60 Hz）；逐对象法向力保留每个物理子步（当前120 Hz）。
读取历史缓冲时只截取本控制步的物理样本，避免重叠历史导致冲量/时长重复计数。
另存全机器人传感器的完整躯干净法向力历史及窗口峰值。
新版臂躯终止按逐对象力的模长之和判定，可能大于该净力峰值；`run.json.torso_contact_policy`记录力的定义。
接触点为每个对象在最后物理子步内的平均接触位置，缺少接触时为NaN；查看接口转换为JSON null。
全部位置减去对应环境原点，保持世界坐标轴方向；四元数为xyzw。

历史使用有界GPU缓冲，只在回合结束时导出摘要、失败时导出轨迹。
新回合清空自身历史长度，其他并行环境继续记录，片段不跨越重置边界。
默认只保存末2秒，`--history_s 35`可保存当前任务的整回合，显存和文件体积随之增加。

## 本地文件

| 文件 | 内容 |
|---|---|
| `run.json` | 检查点路径/SHA256、执行模式、种子、课程等级、时间步、对象通道与阈值 |
| `effective_env.yaml` / `effective_agent.yaml` | 本次实际环境和策略配置 |
| `geometry.json` | 一次性导出的机器人可视网格、刚体名称与连接关系 |
| `episodes.jsonl` | 每个已接收完成回合的类别、目标、时长和全部终止原因，包含成功回合 |
| `failures.jsonl` | 失败索引、片段路径、阶段和逐对象接触统计 |
| `switches.jsonl` | regular 或 mixed 采集时写出的逐次常规目标切换摘要 |
| `clips/000000.npz` | 时间、刚体位姿、关节位置/速度、目标、原始动作、有效Lower命令、接触力/位置和原终止历史 |
| `summary.json` | 进度/停止原因、完成数、失败数、按类别汇总的存活时长和失败频率 |

NPZ只包含数值数组，读取使用`allow_pickle=False`。查看服务按需读取单个片段，不一次加载全部轨迹。
物理接触记录不包含完整求解器内部状态；本模式用于查看已有轨迹，不承诺从片段中间重启物理仿真。

## 统计解释

当前失败判据是环境的非超时终止`reset_terminated`，包括`base_contact`、`knee_contact`、`fallen`和`low_height`。
其中臂/手对躯干合力模长之和>20 N、其他对象对躯干的剩余净力>1 N会触发`base_contact`；
膝部接触>1 N、身体倾斜>0.8 rad、根部高度<0.25 m分别触发其余物理终止项。
采集按任务原始分布随机抽取目标并运行策略，在自动重置前保存触发终止的样本，不主动搜索最差位置。
仅触发`time_out`的回合记录在`episodes.jsonl`，不保存失败片段；因此到时仍未够到目标、跟踪误差大、
Q分数低或仅有1–20 N轻触而未触发其他终止的情况，本身不属于当前失败片段的保存条件。
正常到时结束表示没有物理终止，不等同于任务跟踪成功。

失败样本保留到原终止时刻，接触持续时间在该处截断，属于已观察到的部分。
`normal_impulse_ns`是片段中法向力向量模长按物理步长的积分，不包含摩擦，也不代表接触点层面的总冲量。
`peak_unattributed_normal_force_n`表示净力减去所有过滤通道后剩余向量的最大模长，用于发现遗漏或不一致的通道。
相反方向的接触力会在净力中抵消，诊断时同时查看逐对象通道。

到达停止预算时仍在运行的回合不计为成功或失败，汇总注明其数量与存活时长。
最后一个并行步中因数量上限而未接收的完成回合数另记为`unrecorded_final_step_completions`。
该流程用于快速挖掘，片段按发生顺序保留，并非均匀抽取的失败分布。
比较均值/随机动作或不同模型时，应使用相同课程等级、任务类别、碰撞模型和预算，
并使用`episodes.jsonl`中的成功/失败目标共同计算分箱失败比例。
常规任务的摘要目标为结束时目标；分析其回合中目标变化应结合片段内`goal_pose`序列。

## 验证

CPU回归：`tests/test_failure_mining.py`、`tests/test_failure_geometry.py`。
覆盖终止帧、重置边界、并行预算、时间/冲量统计、成功回合摘要、原生目标刷新、USD导入顺序、真实G1网格导出及查看API。
本次开发环境GPU不可见；实际PhysX批量采集需要在GPU机器上执行上面的smoke命令。
