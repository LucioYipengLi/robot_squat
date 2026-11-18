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
所有类别保留训练的原始采样时序，常规目标仍在回合中刷新，不采用teleop的一回合一目标覆盖。
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
