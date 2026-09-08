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

    # 统一任务指令项（SquatWalkCommand）：4 维 [h_offset, vx, vy, ωz]，髋高偏移（相对
    # 默认站立髋高）与机体系速度目标联合采样。每 episode 先抽一次四模式互斥：
    #   STAND(0.4) 原髋高站立 | SQUAT(0.4) 变髋高蹲起（零速） | WALK(0.2) 速度跟随行走（默认髋高）
    #   SQUAT_WALK(0.0) 指定髋高下行走（混合模式预留槽位，纯配置即可启用）
    # 单时钟双节奏：髋高每 (2,4) s 重采样都重抽（保留中途调髋高技能），速度仅在
    # episode 首抽一次（步态不中途切换）；奖励的模式门控见主目录 TODO P2。
    # 观测经切片消费（velocity_commands 3 维 + pelvis_height_cmd 1 维，见 PolicyCfg），
    # 观测总维 83，旧 checkpoint 不兼容。
    task_command = mdp.SquatWalkCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 4.0),
        rel_mode_envs=(0.4, 0.4, 0.2, 0.0),
        ranges=mdp.SquatWalkCommandCfg.Ranges(
            height_offset=(-0.15, 0.02),
            lin_vel_x=(-0.3, 0.6),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.8, 0.8),
        ),
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
        """Policy observations, concatenated into a flat 83-D vector.

        Order: base state (9) -> task command (4) -> joint state (58) -> last action (12).
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
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "task_command",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
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
    # - 高度轴：全部环境 success_rate > 0.95 时，height_offset 由 [-0.15, 0.02] 扩向 [-0.45, 0.05]
    #   （更深蹲）。此轴按用户选择用“全部环境成功率”门控（= TensorBoard Metrics/success_rate）；该值被
    #   STAND/WALK 的零高度偏移平凡成功抬高，故 0.95 实际约要求“蹲起成功率 87.5%”。
    # - 速度轴：仅 WALK 模式 episode 的 success_rate_vel > 0.95 时，lin_vel_x 由 [-0.3, 0.6] 扩向
    #   [-0.5, 1.0]、lin_vel_y 由 [-0.15, 0.15] 扩向 [-0.2, 0.2]；ang_vel_z 保持 [-0.8, 0.8] 不变。
    # 扩展单调（只朝最终范围生长）且自限速（放宽→更难→成功率回落→暂停），最终范围是“上限”：
    # 若 -0.45 深蹲在 0.03 m 容差下达不到门控，范围会自适应停在中途（本版接受此停顿）。
    command_range = CurrTerm(
        func=mdp.command_range_curriculum,
        params={
            "command_name": "task_command",
            "height_success_gate": 0.95,
            "height_offset_final": (-0.45, 0.05),
            "height_step": 0.005,
            "vel_success_gate": 0.95,
            "lin_vel_x_final": (-0.5, 1.0),
            "lin_vel_y_final": (-0.2, 0.2),
            "vel_step": 0.01,
            "update_period": 200,
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
