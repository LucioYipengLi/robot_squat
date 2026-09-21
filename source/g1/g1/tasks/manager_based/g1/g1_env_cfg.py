# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_ovphysx.sensors import ContactSensorCfg as OvPhysXContactSensorCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.utils.noise import UniformNoiseCfg as Unoise
# Terrain
from isaaclab.terrains import TerrainImporterCfg

from . import mdp, utils

##
# Pre-defined configs
##

from .utils import G1_29DOF_CFG

##
# Scene definition
##

@configclass
class VelocityEnvContactSensorCfg(PresetCfg):
    default = PhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    newton_mjwarp = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    physx = default
    ovphysx = OvPhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # sensors
    contact_forces = VelocityEnvContactSensorCfg()

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 统一任务指令项（SquatWalkCommand）：6 维 [h_offset, vx, vy, ωz, com_dx, com_dy]，髋高偏移（相对
    # 默认站立髋高）与机体系速度目标联合采样。每 episode 先抽一次四模式互斥：
    #   STAND(0.4) 原髋高站立 | SQUAT(0.4) 变髋高蹲起（零速） | WALK(0.2) 速度跟随行走（默认髋高）
    #   SQUAT_WALK(0.0) 指定髋高下行走（混合模式预留槽位，纯配置即可启用）
    # 单时钟双节奏：髋高每 (2,4) s 重采样都重抽（保留中途调髋高技能），速度仅在
    # episode 首抽一次（步态不中途切换）；奖励的模式门控见主目录 TODO P2。
    # 观测经切片消费：速度 3 维、髋高 1 维，末尾追加 COM 2 维；不加入实际 COM 或启用标志。
    # COM 仅 STAND/SQUAT 生效，双踝中点为零点，偏置各 ±1 cm；102 维观测不兼容旧 checkpoint。
    task_command = mdp.SquatWalkCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 4.0),
        rel_mode_envs=(0.4, 0.4, 0.2, 0.0),
        ranges=mdp.SquatWalkCommandCfg.Ranges(
            height_offset=(-0.15, 0.02),
            lin_vel_x=(-0.3, 0.6),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.8, 0.8),
            com_offset_x=(-0.01, 0.01),
            com_offset_y=(-0.01, 0.01),
        ),
        com_target_speed=0.02,
        com_zero_probability=0.2,
        com_success_threshold=0.005,
        height_success_threshold=0.03,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """102 维策略观测：保留原 100 维顺序，在末尾追加 COM 有效目标两维。

        原顺序为基座状态(9)、高度/速度指令(4)、全身关节状态(58)、上一步动作(12)、
        上肢 PD 目标(17)；COM 输入只含目标偏置，不含实际 COM、模式或启用标志。
        """

        # 基座本体状态（IMU 可获取量）：线速度 ±0.1、角速度 ±0.2（陀螺仪噪声更大）、
        # 投影重力 ±0.05（隐式提供躯干倾斜姿态）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # 任务指令：task_command 4 维联合指令的切片，视为真值不加噪声；
        # 速度 (vx, vy, ωz) 机体系 3 维 + 髋高偏移（相对默认站立髋高）1 维
        velocity_commands = ObsTerm(func=mdp.task_command_velocity, params={"command_name": "task_command"})
        pelvis_height_cmd = ObsTerm(func=mdp.task_command_height, params={"command_name": "task_command"})

        # 关节状态（编码器可获取量）：位置 ±0.01、速度 ±1.5（速度估计噪声远大于位置）
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # 上一步动作：支撑 PD 位置控制与平滑性
        actions = ObsTerm(func=mdp.last_action)

        # 上肢 PD 目标（前馈预判信号）：UpperBodyDisturbanceEvent 写入的上肢目标（相对 default），
        # 领先实际运动 ≥1 step，让下肢预判上肢运动产生的动量扰动；外部扰动只作用于实际关节位置，
        # 不进入此读数。Unoise ±0.02 模拟 IK 解算 / 目标读数的不确定性
        upper_body_target = ObsTerm(
            func=mdp.upper_body_joint_pos_target_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist.*", "waist_.*_joint"]
                )
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        # 末尾追加而不移动原有特征索引；行走模式输入为零，奖励独立按 mode 门控。
        com_offset_cmd = ObsTerm(func=mdp.task_command_com, params={"command_name": "task_command"})

        def __post_init__(self):
            # 训练时启用噪声注入（_PLAY 配置关闭）；所有观测项拼接为扁平向量
            self.enable_corruption = True
            self.concatenate_terms = True


    # 观测分组挂载：策略观测组，被 ObservationManager 收集
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    
    # start up
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # interval
    # 上肢随机干扰（loco-manipulation 鲁棒性）：双臂 + 腰部均不在 action space，是智能体不可控的
    # 扰动源。每 1~3 s（per-env 异步，is_global_time 默认 False）在各组「默认位姿→限位余量」内按百分比
    # 采样关节目标并 clamp 到限位，写入 PD 目标缓冲区，由 implicit actuator 逐步驱动上肢运动，对浮动基
    # 下肢产生真实的 CoM 偏移与反作用力扰动。实现见 mdp/events.py:UpperBodyDisturbanceEvent。
    # 每组幅度 joint_noise_frac 单独定义，单位是「默认位姿→各方向限位余量」的百分比 [0,1]：同一 frac 按
    # 每个关节的实际余量自动缩放绝对弧度，故非对称限位（elbow）与组内 ROM 差异（waist_yaw ±150° >>
    # waist_roll/pitch ±30°）无需逐关节手调。手臂取 0.3；腰部刚度大（200）、是平衡核心，取保守 0.12——
    # 注意纯百分比下 yaw 余量大，其绝对偏移（≈0.31 rad）明显大于 roll/pitch（≈0.06 rad）；若需分别控制，
    # 可把腰部再拆成 yaw / roll+pitch 两组各给 frac。
    upper_body_disturbance = EventTerm(
        func=mdp.UpperBodyDisturbanceEvent,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        params={
            # 分组列表：EventManager 会递归 resolve 每组嵌套的 SceneEntityCfg（填充 joint_ids）。
            # 各组须为互斥的关节子集，且均排除下肢（与下肢 action 的 joint_names 互斥，无写入冲突）。
            "disturb_groups": [
                {
                    # 组 0：双臂（肩/肘/腕，共 14 DOF）——轻质外周，可大幅随机摆动
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist.*"]),
                    "joint_noise_frac": 0.2,  # [0,1] 默认位姿→限位余量的百分比（课程爬升的 floor）
                },
                {
                    # 组 1：腰部（yaw/roll/pitch，共 3 DOF）——刚度大、平衡核心、ROM 小，保守幅度
                    "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"]),
                    "joint_noise_frac": 0.1,  # [0,1] 余量百分比；yaw≈0.31rad、roll/pitch≈0.06rad
                },
            ],
        },
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(3.0, 6.0),
    #     params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    # )
    
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(2.0, 4.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    注释约定：每项仅一行「语义 + 公式要点 + 门控」，公式推导、基向量来源、参数
    敏感性等细节见 mdp/rewards.py 各函数 docstring。两类通用门控：

    * 高度门控：命令髋高 ≥ 0.735 m（站立区间）才生效，深蹲区间放行；
    * 零速模式门控：读 task_command 的 mode 缓冲，仅 STAND/SQUAT 生效，
      WALK/SQUAT_WALK 放行（挂 ``velocity_command_name`` 参数即启用）。
    """

    # =====================================================================
    # 生存与失败信号
    # =====================================================================
    # 存活奖励：未终止时每步 +0.5，鼓励机器人尽量长时间不摔倒
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # 终止惩罚：发生终止条件（摔倒等）时一次性扣 -200，强烈约束避免摔倒
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # =====================================================================
    # 任务奖励（髋高追踪 + 速度跟踪）
    # =====================================================================
    # 髋高追踪：r = exp(-4·|e|)，绝对误差指数核（官方），零误差附近梯度恒定保留精细到位信号
    track_pelvis_height = RewTerm(
        func=mdp.track_pelvis_height_exp,
        weight=2.0,
        params={"command_name": "task_command"},
    )
    # 速度跟踪（核心任务奖励，统一四模式）：r = Σexp(-e²/(2σ²))，vx/vy/ωz 三轴机体系误差，
    # 满分 3.0；指令取 task_command 速度切片 [vx,vy,ωz]，误差用机体系（与指令 metrics 同口径）。
    # STAND/SQUAT 指令速度为 0 → 退化为静止约束（替代原 zero_velocity，无需门控）；
    # WALK/SQUAT_WALK → 跟踪非零指令。zero_velocity_exp 本体保留作回退/消融。
    track_velocity = RewTerm(
        func=mdp.track_velocity_exp,
        weight=1.0,
        params={"command_name": "task_command", "std": 0.25},
    )

    # COM 跟踪（Own）：exp(-||e_xy||²/std²)，双踝中点水平系；仅 STAND/SQUAT，核宽非安全边界。
    # 首版保留其他奖励权重；需实测零速/默认关节位/协同约束是否抵抗厘米级调整。
    track_com = RewTerm(
        func=mdp.track_com_xy_exp,
        weight=1.0,
        params={"command_name": "task_command", "std": 0.02},
    )

    # =====================================================================
    # 下肢协同约束（2D 协同流形）
    # =====================================================================
    # 膝关节引导（消融基线，注释保留）：高度误差加权约束膝偏离行程中点，防“跪式下蹲”；
    # 启用时补 "velocity_command_name": "task_command"（与 leg_synergy 门控口径一致）
    # knee_guidance = RewTerm(
    #     func=mdp.knee_guidance_l1,
    #     weight=-0.75,
    #     params={
    #         "command_name": "task_command",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
    #     },
    # )
    # 下肢协同流形（替代 knee_guidance）：r = exp(-||d⊥||²/σ²) - 1 ∈ [-1, 0]（正权重挂载），
    # 双腿 hip/knee/ankle 按全行程归一后惩罚偏离 2D 协同平面（蹲起数据非中心化 SVD 拟合，
    # 解释 99.7% 方差，内含 G1 符号约定）；sigma=0.2 为流形带宽（≈膝 33° 容差），
    # 训练卡死时可放宽至 0.3~0.5；零速模式门控（行走步态天然偏离矢状面蹲起流形）
    leg_synergy = RewTerm(
        func=mdp.leg_synergy_manifold_exp,
        weight=0.75,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_pitch_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "right_hip_pitch_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                ],
            ),
            "sigma": 0.2,
            "velocity_command_name": "task_command",
        },
    )
    # 髋关节回默认位：偏离默认位平方惩罚（官方 deviation_hip）；高度 + 零速模式双门控
    hip_default_deviation = RewTerm(
        func=mdp.standing_joint_default_deviation_l2,
        weight=-0.5,
        params={
            "command_name": "task_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]),
            "velocity_command_name": "task_command",
        },
    )
    # 踝关节回默认位：同上双门控；默认位动态读取（ankle_pitch 默认 -0.2，不可用固定 target=0）
    ankle_default_deviation = RewTerm(
        func=mdp.standing_joint_default_deviation_l2,
        weight=-0.5,
        params={
            "command_name": "task_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]),
            "velocity_command_name": "task_command",
        },
    )

    # =====================================================================
    # 躯干姿态与速度稳定性约束（使用 Isaac Lab 内置函数）
    # =====================================================================
    # 躯干朝向惩罚：r = -||g_xy||²，惩罚重力在躯干 x/y 轴投影，保持直立（内置 flat_orientation_l2）
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis"])},
    )
    # Z 轴线速度惩罚：r = -v_z²，抑制高度调整时的剧烈上下颠簸与冲量冲击（内置）
    vertical_bounce = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    # XY 角速度惩罚：r = -||ω_xy||²，惩罚 Roll/Pitch 方向角速度，防止摇晃（内置）
    tilt_wobble = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)

    # =====================================================================
    # 足端约束（贴地、防滑、平行、接触力）
    # =====================================================================
    # 脚掌贴地约束：r = -Σ Var(足底四角点高度)，量级 ~1e-3 故权重大；
    # 仅持续接触 ≥ 3·dt 的脚参与 + 零速模式门控
    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel_var,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            "foot_length": 0.18,
            "foot_width": 0.08,
            "velocity_command_name": "task_command",
        },
    )
    # 足部防滑惩罚：r = -Σ ||v_foot|| · I_contact，触地期间惩罚足端水平速度（复用 feet_slide）
    feet_slip = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
        },
    )
    # 双足平行约束（官方）：r = -Var(左右脚跟/中/尖三点间距)；高度 + 零速模式双门控，权重照抄官方
    feet_parallel = RewTerm(
        func=mdp.feet_parallel_var,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_name": "task_command",
            "velocity_command_name": "task_command",
        },
    )
    # 足端接触力峰值惩罚：r = -Σ ReLU(||F|| - 400 N)，保护踝关节与真机结构（内置，阈值同官方）
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-2.5e-4,
        params={
            "threshold": 400.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
        },
    )
    # 防双脚离地：至少一脚保持接触时奖励 1.0，禁止双脚同时腾空（官方 no_fly）
    no_fly = RewTerm(
        func=mdp.no_fly,
        weight=0.75,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )
    # 站立蠕行惩罚：惩罚离地脚数量（< 0.1 N 视为离地），抑制原地踏步（官方）；
    # 零速模式门控（行走必须抬脚，恢复官方 HOMIE 语义）
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            "velocity_command_name": "task_command",
        },
    )
    # 步态质量·摆动相时长（正奖励，行走专属）：r = min(支撑相/摆动相时长) 截断至 threshold，
    # 仅“单脚支撑 + 线速度指令非零(‖[vx,vy]‖>0.1)”时给分——自动只在 WALK/SQUAT_WALK 生效，
    # STAND/SQUAT 零速指令下恒 0（防原地抬脚刷分），无需额外门控。鼓励迈出足够长的步子、
    # 抑制高频碎步蹭行。threshold=0.5 s ≈ G1 常速半步周期；weight 为初值，按消融再调。
    # 【已停用·保留供 A/B 回调对比】min+clamp 结构只奖励"延长单支撑摆动"：封顶后悬停零成本、
    # 双支撑归零→回避落脚，教出"一腿抬起悬停刷分、另一腿快速切换"的不对称步态。已改用下方
    # feet_gait_phase（相位时钟）统一约束交替/节律/对称/防碎步。回调对比时：取消注释本项、
    # 并注释掉 feet_gait_phase 即可。
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.5,
    #     params={
    #         "command_name": "task_command",
    #         "threshold": 0.5,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
    #     },
    # )
    # 步态质量·相位时钟（正奖励，行走专属）：由 episode_length_buf×step_dt 导出相位 2πt/T，
    # 左右脚目标接触态取反相正弦（相差 180°），按高斯核匹配实际接触→一项同时约束交替/节律/
    # 对称/防碎步，根除 air_time 的"悬停刷分"。无状态（相位随 episode 自动复位、各环境天然错相）；
    # 仅 WALK/SQUAT_WALK 生效（mode 门控）。反相设计下 body_ids 左右顺序不影响交替正确性。
    # 步态周期 T 速度自适应（方案B·步频线性）：f = a + b·|vx| [Hz]，T = 1/f。a=stride_freq_intercept
    # 定零速步频（1/a≈0.84s 对齐 G1 倒立摆自然周期 pi·sqrt(L/g)），b=stride_freq_slope 定步频随速度增幅；
    # 该式全程光滑有界无奇点（优于 T=T0-k·|vx| 周期线性式，后者有限速度处撞 T=0）。速度 episode 内冻结
    # →T 逐环境恒定→相位连续无跳变。vx≈0.3 时 T≈0.70（衔接原固定值）。a/b/std/weight 为初值，按消融调。
    feet_gait_phase = RewTerm(
        func=mdp.feet_gait_phase_clock,
        weight=0.5,
        params={
            "command_name": "task_command",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "stride_freq_intercept": 1.19,
            "stride_freq_slope": 0.79,
            "std": 0.3,
        },
    )

    # =====================================================================
    # 横向安全间距（双脚 / 双膝）
    # =====================================================================
    # 双脚横向安全距离：双边钳制——下界防并拢（常开），上界防越分（仅高度门控），正权重惩罚语义
    feet_lateral = RewTerm(
        func=mdp.feet_lateral_separation,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_name": "task_command",
            "min_distance": 0.2,
            "max_distance": 0.35,
        },
    )
    # 双膝横向安全距离：同上双边形式作用膝关节连杆，防膝内扣（X 型腿）与过度外分（正权重惩罚）
    knee_lateral = RewTerm(
        func=mdp.knee_lateral_separation,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_knee_link", "right_knee_link"]),
            "command_name": "task_command",
            "min_distance": 0.2,
            "max_distance": 0.35,
        },
    )

    # =====================================================================
    # 动作平滑性约束（动作空间）
    # =====================================================================
    # 动作变化率惩罚：r = -||a_t - a_{t-1}||²，惩罚相邻控制帧动作高频跳变（内置）
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 动作二阶平滑度惩罚：r = -||a_t - 2a_{t-1} + a_{t-2}||²，约束动作空间加速度，
    # 降低机械磨损与打颤（自定义，内部维护 a_{t-2} 缓存）
    action_smoothness = RewTerm(func=mdp.action_acc_l2, weight=-0.05)

    # =====================================================================
    # 能耗约束（关节空间）
    # =====================================================================
    # 扭矩输出惩罚（官方）：r = -Σ(τ/k)²，按刚度归一使髋/踝量级可比；仅下肢关节；
    # 踝实际刚度 20 会主导惩罚，取等效刚度 100 与髋膝对齐
    dof_torques = RewTerm(
        func=mdp.joint_torques_l2_normalized,
        weight=-2.5e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
            ),
            "stiffness": {
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
                ".*_ankle_pitch_joint": 100.0,
                ".*_ankle_roll_joint": 100.0,
            },
        },
    )

    # =====================================================================
    # 关节限位与扭矩超限约束（安全硬边界）
    # =====================================================================
    # 关节位置限位：r = -Σ ReLU(软限位外超出量)，软限位 = 硬限位 × 0.9（内置）
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
            )
        },
    )
    # 扭矩超限惩罚：r = -Σ ReLU(|τ| - τ_max)；隐式执行器限位 inf 自动跳过（内置）
    dof_torque_limits = RewTerm(
        func=mdp.joint_torque_limits,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
            )
        },
    )
    # 关节速度限位：r = -Σ ReLU(|ω| - ω_max·soft_ratio)，超限量裁剪至 1（内置，权重照抄官方）
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-2e-3,
        params={
            "soft_ratio": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
            ),
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )

    knee_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_link"), "threshold": 1.0}
    )
    
    bad_ori = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.5},
    )

    base_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.2,  
            "asset_cfg": SceneEntityCfg("robot", body_names=".*pelvis"),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # 地形难度课程（平地任务，禁用）
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_levels = None

    # 指令范围课程（绩效驱动·全局，非时间驱动）：智能体稳定跟踪当前指令范围后，逐档放宽采样范围。
    # 两条轴独立推进，各自门控于指令项在 reset 时结算的逐 episode 成功率：
    # - 高度轴：全部环境 success_rate > 0.99 时，height_offset 由 [-0.15, 0.02] 扩向 [-0.45, 0.05]
    #   （更深蹲）。此轴按用户选择用“全部环境成功率”门控（= TensorBoard Metrics/success_rate）；该值被
    #   STAND/WALK 的零高度偏移平凡成功抬高，故 0.99 实际约要求“蹲起成功率 97.5%”。
    # - 速度轴：仅 WALK 模式 episode 的 success_rate_vel > 0.95 时，lin_vel_x 由 [-0.3, 0.6] 扩向
    #   [-0.5, 1.0]、lin_vel_y 由 [-0.15, 0.15] 扩向 [-0.2, 0.2]；ang_vel_z 保持 [-0.8, 0.8] 不变。
    # 扩展单调（只朝最终范围生长）且自限速（放宽→更难→成功率回落→暂停），最终范围是“上限”：
    # 若 -0.45 深蹲在 0.03 m 容差下达不到门控，范围会自适应停在中途（本版接受此停顿）。
    command_range = CurrTerm(
        func=mdp.command_range_curriculum,
        params={
            "command_name": "task_command",
            "height_success_gate": 0.99,
            "height_offset_final": (-0.50, 0.05),
            "height_step": 0.005,
            "vel_success_gate": 0.95,
            "lin_vel_x_final": (-0.5, 1.0),
            "lin_vel_y_final": (-0.2, 0.2),
            "vel_step": 0.01,
            # update_period 以环境步计：5000 = 100 训练轮 × num_steps_per_env(50)，即每 100 轮才检查一次
            # 门控，让策略在当前范围充分巩固（≈2000 万条 transition）后再微调，避免“移动目标”拖慢收敛。
            "update_period": 5000,
        },
    )

    # 上肢干扰幅度课程：随策略存活率单调爬升指定组的 joint_noise_frac（隐式 interval 课程只控触发频率、不控幅度）。
    # target_group=0 只爬升手臂组：从 EventCfg.upper_body_disturbance 手臂组当前的 frac(=0.3) 起爬，上限
    # frac_final=0.5（约用到余量一半）；腰部组（target_group=1）保持固定 0.12 不爬升（刚度大、平衡核心，更保守）。
    # survival_gate=0.95 是高门槛——仅当策略近乎满存活（EMA>5.7 s / 6 s）才放行下一次 +frac_step，把幅度爬升
    # 留给已成熟的策略；frac_step=0.005（每次 +0.5% 余量）细粒度爬升，单步过猛会把存活打回门槛下而反复暂停。
    # 切勿同时对 interval_range_s 做课程——频率已由 episode 存活时长隐式控制。
    upper_body_disturbance_magnitude = CurrTerm(
        func=mdp.upper_body_disturbance_magnitude_curriculum,
        params={
            "event_term_name": "upper_body_disturbance",
            "target_group": 0,
            "survival_gate": 0.95,
            "frac_final": 0.8,
            "frac_step": 0.005,
            "update_period": 2000,
        },
    )

    # SQUAT_WALK 模式引入课程（轮次驱动·全局一次性，非绩效驱动）：训练推进到指定步数后，把指令项的四模式
    # 权重从挂载值 (0.4, 0.4, 0.2, 0.0)（SQUAT_WALK 预留为 0）一次性改写为 target_mode_weights，让
    # STAND/SQUAT/WALK 自然成熟后再过渡进蹲走能力。选轮次驱动而非绩效驱动：操作者人工观察训练曲线选定引入
    # 点，且规避“新模式同时绑定多个绩效门控→拉低各门控指标→自我停滞”的 AND 概率坍缩。关键：改写的是指令项
    # 运行时缓存的 _mode_probs 张量（reset 只读它、从不回读 cfg.rel_mode_envs），故改 cfg 无效——详见
    # squat_walk_mode_introduction_curriculum 的 docstring。
    # trigger_step 换算：common_step_counter 每 env.step() +1，即每训练轮 +num_steps_per_env(50)；故
    # trigger_step = trigger_iter × 50，此处 400000 = 第 8000 轮引入（max_iterations=10000）。注意该计数
    # 为 env 侧、resume 时从 0 重新计（RSL-RL 只恢复迭代号 it），故本 trigger 以“本次进程内步数”为准。
    # target_mode_weights 选择：默认四模式均分 (0.25×4)；若担心遗忘成熟模式可偏向保留，如 (0.2, 0.25,
    # 0.25, 0.3)。SQUAT_WALK 槽位（index 3）必须 > 0 才真正引入。
    # 首版建议：引入前后冻结 command_range 的速度轴（或与之错峰），使包络 nom（ranges.lin_vel_x/ang_vel_z）
    # 稳定——否则速度范围同时扩宽会在新引入的 SQUAT_WALK episode 下漂移包络语义（CASE §7.3）。
    squat_walk_mode_introduction = CurrTerm(
        func=mdp.squat_walk_mode_introduction_curriculum,
        params={
            "command_name": "task_command",
            "trigger_step": 400_000,
            "target_mode_weights": (0.25, 0.25, 0.25, 0.25),
        },
    )

##
# Environment configuration
##


@configclass
class G1EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: EventCfg = EventCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 6
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

@configclass
class G1EnvCfg_PLAY(G1EnvCfg):
    """Environment configuration for playing."""
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
