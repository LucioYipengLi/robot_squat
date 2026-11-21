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

## 平面近场 Lower 指令诊断

用同一检查点、种子和环境数各运行一组。诊断模式按 `--num_envs` 收集每个环境的首个回合，
以 `env_id` 对齐目标与手臂模式。同一批目标可用不同的侧移和转向上限评估。

```bash
checkpoint=logs/rsl_rl/g1_upper_mobile/2026-11-19_00-28-13/model_29999.pt
for scale in 1.0 0.5; do
  uv run python -m g1.rl failures mine --checkpoint "$checkpoint" \
    --level 4 --category planar --planar-sampling transition --policy_output mean \
    --seed 42 --num_envs 512 --planar-near-lower-scale "$scale" \
    --output "outputs/failures/v7_29999_l4_near_lower_${scale}" || break
done
```

`1.0`沿用原始动作，`0.5`从首次满足骨盆站位误差 20 cm 和朝向误差 15° 的下一控制步起，
将侧移和转向限制在该回合范围的 50%，直到重置。Level 4 对应 ±0.049 m/s 和 ±0.19 rad/s。
其余动作沿用策略输出。`run.json`保存倍率，`summary.json`的`planar_groups`给出入场比例、
入场后失败比例和完整评价窗腕部达标率。用`episodes.jsonl`按`env_id`对照两组首批目标。

## 平面大角度启动双臂保持对照

先使用 `--planar-startup`、`--planar-sampling transition` 和 `--policy_output mean` 采集原策略。
每个环境保留首个回合，`startup/` 保存所有 `wide` 用例的初始帧和前 0.2 秒轨迹。
将原采集目录交给 `--planar-hold-from`，沿用 checkpoint、等级、种子和并行环境数。

```bash
baseline=outputs/failures/v8_l4_wide_startup_20261120_144026
checkpoint=logs/rsl_rl/g1_upper_mobile/2026-11-19_23-59-12_upper_mobile_geometry_30k/model_29999.pt
output="outputs/failures/v8_l4_wide_hold_$(date +%Y%m%d_%H%M%S)"

uv run python -m g1.rl failures mine \
  --task Template-G1-Upper-Mobile-v0 --checkpoint "$checkpoint" \
  --level 4 --category planar --planar-sampling transition \
  --policy_output mean --seed 42 --num_envs 256 \
  --planar-hold-from "$baseline" --output "$output"
```

对照选取全部 `wide` 失败用例，以及至少一只活动手的目标跨过初始骨盆中线 10 cm 的通过用例。
每个用例按 `env_id` 核对重置姿态和目标。前 0.2 秒保持两条手臂的初始关节参考，重放原记录的
Lower 和腰部动作。原记录提前终止时，沿用最后一帧指令直到保持期结束，然后交回完整策略。

`summary.json` 的 `planar_hold.by_env_id` 报告共同窗口内两组的最大骨盆反转角、肩部及整条手臂
对躯干的接触峰值、是否超过硬接触阈值，以及保持组释放后的首次入场与位姿达标率。
`covered_baseline_window` 和 `lower_and_waist_replay_match` 均为真时，才计入可比用例。
常规 `summary.json` 的回合总数仍覆盖原始并行批次，干预用例的统计见 `planar_hold`。

### 腰部与双臂同时保持

用 `--planar-hold-waist-from` 指向已完成的双臂保持运行。命令从该归档找到原始 v8 启动轨迹，
沿用相同用例和原始 Lower 指令，前 0.2 秒保持腰部与双臂关节的初始参考，然后恢复完整策略。

```bash
arm_hold=outputs/failures/v8_l4_wide_hold_20261120_145850
checkpoint=logs/rsl_rl/g1_upper_mobile/2026-11-19_23-59-12_upper_mobile_geometry_30k/model_29999.pt
output="outputs/failures/v8_l4_wide_waist_hold_$(date +%Y%m%d_%H%M%S)"

uv run python -m g1.rl failures mine \
  --task Template-G1-Upper-Mobile-v0 --checkpoint "$checkpoint" \
  --level 4 --category planar --planar-sampling transition \
  --policy_output mean --seed 42 --num_envs 256 \
  --planar-hold-waist-from "$arm_hold" --output "$output"
```

`summary.json` 的 `planar_hold.comparison_group` 为 `arms_only`。
`by_env_id` 下的 `baseline` 是仅保持双臂的结果，`hold` 是同时保持腰部和双臂的结果，
两侧统计到各自轨迹共同覆盖的时刻。`lower_replay_match` 核对该窗口内的 Lower 输入，
保持期内终止、释放后终止和完成跟踪分别计数。

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

### 回收专项训练与对照

训练命令加 `env.commands.wrist_pose.regular_retract_sampling.probability=0.35`，在 regular
单臂回合的目标刷新时，以 35% 概率尝试定向回收。默认概率为 0。专项从当前 Level 的原生位置范围
有限次抽取候选，接受初始骨盆坐标系的向后位移至少 3 cm、三维跳变 5–12 cm 的位置。
抽不到候选时采用本次原生采样结果。方向和距离使用相邻两个指令目标。
训练日志 `Metrics/wrist_pose/task/regular/retract_sampling/` 记录单臂刷新数、尝试数、命中数、回退数和实际回收比例。

开启专项后，checkpoint 会记录采样配置。从旧分布切到新分布续训时，用 `range_curriculum.reset_on_load=true`
重建课程统计，并用保存的目标等级作为新的 `initial_level`。评估载入时会恢复训练采样，
加入 `--regular-sampling native` 可在恢复后关闭单双臂专项并使用原生刷新。比较整体能力时，在两组都指定 `native`。
`run.json` 会同时记录 checkpoint 的单双臂专项概率和评估使用的概率。

固定回收目标对可从已有 `regular` 评估生成。下例从每个距离档采 50 对，左右镜像各执行一次。

```bash
uv run python -m g1.rl failures pairs \
  --run outputs/failures/regular_l3_v7_29999_mean --suite retract --per-bin 50 --regressions 0 \
  --output outputs/failures/regular_l3_retract_pairs.json

uv run python -m g1.rl failures mine \
  --checkpoint "<checkpoint.pt>" --category regular --policy_output mean --seed 42 \
  --switch_pairs outputs/failures/regular_l3_retract_pairs.json --num_envs 200 \
  --output outputs/failures/regular_l3_retract_eval
```

在 `switch_pair_evaluation.by_group` 按 5–8 和 8–12 cm 比较两侧的 0.2 秒硬接触率和 1 秒内到位率。

### 双臂回收专项加密

`env.commands.wrist_pose.regular_retract_sampling.dual_probability=0.25` 让 regular 双臂回合从第二个目标起，
每次刷新以 25% 的概率尝试双臂回收。该参数默认为 0。以初始骨盆轴衡量，两腕目标都需后移至少 3 cm，
且两腕中较大的三维跳变需落在 `[8, 12)` cm。每次尝试最多从当前 Level 的双腕原生分布抽取 16 对完整候选，
保留同一对的双手位置和姿态。未命中时使用本次原生目标对。训练日志
`Metrics/wrist_pose/task/regular/dual_retract_sampling/` 中的 `switches`、`attempts`、`hits`、`fallbacks`
分别表示双臂刷新数、专项尝试数、命中数、回退数。`retract_fraction` 统计实际两手均后移至少 3 cm 的刷新比例，
包含原生采样命中的回收和各距离档。

从同一 checkpoint 各续训 1200 轮，固定 Level 3、20% 平面和 35% 单臂专项，只改变双臂专项尝试率。

```bash
checkpoint=logs/rsl_rl/g1_upper_mobile/2026-11-19_15-03-43_v7_from5000_near_lower_retract_8k/model_12999.pt
for dual_probability in 0.0 0.25; do
  uv run python -m g1.rl train --task Template-G1-Upper-Mobile-v0 \
    --checkpoint "$checkpoint" --num_envs 4096 --seed 1217707953 --max_iterations 1200 \
    --viz none --run_name "l3_dual_retract_$dual_probability" physics=isaacsim_physx \
    agent.mirror_augmentation=true \
    env.commands.wrist_pose.range_curriculum.initial_level=3 \
    env.commands.wrist_pose.range_curriculum.reset_on_load=true \
    env.commands.wrist_pose.range_curriculum.advance=false \
    env.commands.wrist_pose.range_curriculum.planar_sampling.enabled=false \
    env.commands.wrist_pose.planar_approach.enabled=true \
    env.commands.wrist_pose.planar_task.transition_sampling.enabled=true \
    env.rewards.lower_command_magnitude.params.near_planar_gain=4.0 \
    env.commands.wrist_pose.regular_retract_sampling.probability=0.35 \
    env.commands.wrist_pose.regular_retract_sampling.dual_probability="$dual_probability"
done
```

固定评估从已有原生 regular 记录提取两个距离档，分别检查训练档 8–12 cm 和 12 cm 及以上的泛化。
每个用例在同一回合执行一次，跟踪旧目标 4 秒后同时切换双腕，观察 1 秒。

```bash
uv run python -m g1.rl failures pairs \
  --run outputs/failures/v7_resume_compare_20261119_183442/new_regular_l3 \
  --suite dual_retract --per-bin 50 --regressions 0 \
  --output outputs/failures/regular_l3_dual_retract_pairs.json

uv run python -m g1.rl failures mine \
  --checkpoint "<checkpoint.pt>" --category regular --policy_output mean --seed 42 --num_envs 100 \
  --switch_pairs outputs/failures/regular_l3_dual_retract_pairs.json \
  --output "<new-output-directory>"
```

在 `switch_pair_evaluation.by_group` 对比切换后 0.2 秒内的 `early_hard_contacts / contact_assessed`、
1 秒内双腕的 `reached_within_1s / reach_assessed` 和切换前失败数 `failed_before_switch`。

### 固定终点的后撤诊断

同一 checkpoint、固定目标对和种子分别运行 `direct`、`hold`、`retreat`。三个模式先跟踪原目标 4 秒，
再分别直接切换、保持上肢关节参考 0.60 秒、保持上肢参考同时让 Lower 后撤 0.60 秒。两种保持模式沿用
干预开始时的高度指令，侧移、转向和骨盆站立偏移置零。后撤请求取当前 Level 的负向速度上限，骨盆实际后退
达到 5 cm 后将速度指令置零，直到相同的释放时刻。随后交回策略并发布同一个世界坐标终点。每组都观察释放后的 1 秒。

下例选择六个目标点落入切换前躯干碰撞凸包的用例，再加两个目标点在凸包外但发生过碰撞的对照。
样本来自历史失败，用于诊断原因。实际后撤量沿干预开始时的骨盆水平前向测量。

```bash
checkpoint=logs/rsl_rl/g1_upper_mobile/2026-11-19_19-30-49_v7_from5000_near_lower_dual_retract_8k/model_12999.pt
suite=outputs/failures/v7_dual_retract_12999_l3_20261119_225455/dual_retract_pairs.json
output="outputs/failures/v7_retract_retreat_$(date +%Y%m%d_%H%M%S)"
cases=(
  dual_retract_12cm_and_over-9-2
  dual_retract_12cm_and_over-277-1
  dual_retract_12cm_and_over-485-1
  dual_retract_12cm_and_over-1895-3
  dual_retract_8_to_12cm-1026-1
  dual_retract_12cm_and_over-777-1
  dual_retract_8_to_12cm-514-1
  dual_retract_8_to_12cm-1294-1
)
for mode in direct hold retreat; do
  uv run python -m g1.rl failures mine \
    --checkpoint "$checkpoint" --level 3 --category regular --policy_output mean --seed 42 \
    --num_envs 8 --switch_pairs "$suite" --switch-cases "${cases[@]}" \
    --switch-diagnostic "$mode" --output "$output/$mode" || break
done
```

在 `summary.json` 的 `switch_pair_evaluation.by_pair` 对齐同一用例，核对
`pelvis_retreat_at_switch_m`、`contact_over_hard_threshold` 和 `stable_last_0p2s`。最后一项要求
双腕在回合结束前连续 0.2 秒满足位姿容差。这批用例的分母以全部已执行回合为准，同时检查释放前失败数。
后撤会使机器人离开旧目标，三组的可比性用干预开始时的到位与接触状态判断。

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
