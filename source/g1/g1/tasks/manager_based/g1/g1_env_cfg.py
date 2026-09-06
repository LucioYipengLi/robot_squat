# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

    pelvis_height = mdp.UniformHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 4.0),
        rel_default_envs=0.2,
        ranges=mdp.UniformHeightCommandCfg.Ranges(height_offset=(-0.15, 0.02)),
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
        """Observations for policy group."""
        """
        # observation terms (order preserved)
        # =====================================================================
        # 分组一：基座本体状态（本体感知，IMU 可获取量）
        # =====================================================================
        # 基座线速度：机体系下 xyz 三轴线速度（3 维），±0.1 均匀噪声模拟传感器误差
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # 基座角速度：机体系下 roll/pitch/yaw 角速度（3 维），±0.2 噪声（陀螺仪噪声更大）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 投影重力：重力向量投影到机体系（3 维），隐式提供躯干倾斜姿态信息，±0.05 噪声
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # =====================================================================
        # 分组二：任务指令（策略的目标输入）
        # =====================================================================
        # 速度指令：当前 base_velocity 命令（vx, vy, yaw 角速度，3 维），
        # 不加噪声（指令由上层给出，视为真值）
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # =====================================================================
        # 分组三：关节状态（本体感知，编码器可获取量）
        # =====================================================================
        # 关节位置：各关节角度相对默认位姿的偏差（全部关节），±0.01 噪声模拟编码器精度
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节速度：各关节角速度（全部关节），±1.5 噪声（速度估计噪声远大于位置）
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # =====================================================================
        # 分组四：动作历史（支撑 PD 位置控制与平滑性）
        # =====================================================================
        # 上一步动作：上一次输出的动作目标，帮助策略感知当前控制目标，支撑平滑控制
        actions = ObsTerm(func=mdp.last_action)

        # =====================================================================
        # 分组五：地形感知（平地任务已禁用）
        # =====================================================================
        # 地形高度扫描：置为 None 关闭，平地任务无需地形信息，减少观测维度
        height_scan = None
        """
        # 本体状态
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # 任务指令，当前pelvis_height的命令
        pelvis_height_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "pelvis_height"})

        # 关节状态（位置和速度）
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # 动作历史
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            # 启用噪声注入（训练时对观测加扰，提升鲁棒性；_PLAY 配置中会关闭）
            self.enable_corruption = True
            # 将所有观测项拼接成一个扁平向量，作为策略网络的单一输入
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
    """Reward terms for the MDP."""

    # =====================================================================
    # 生存与失败信号
    # =====================================================================
    # 存活奖励：未终止时每步 +0.5，鼓励机器人尽量长时间不摔倒
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # 终止惩罚：发生终止条件（摔倒等）时一次性扣 -200，强烈约束避免摔倒
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # =====================================================================
    # 任务奖励（髋高追踪 + 静止零速度约束）
    # =====================================================================
    # 髋高追踪：r = exp(-4·|e|)（官方实现的绝对误差指数核），世界系测量；
    # 零误差附近梯度恒为 4，保留精细到位信号（平方核在 e→0 时梯度趋零）
    track_pelvis_height = RewTerm(
        func=mdp.track_pelvis_height_exp,
        weight=2.0,
        params={"command_name": "pelvis_height"},
    )
    # 静止零速度约束：r = exp(-4·vx²) + exp(-4·vy²) + exp(-4·ω_yaw²)，静止时满分 3.0（本项目公式，官方无对应项）
    zero_velocity = RewTerm(
        func=mdp.zero_velocity_exp,
        weight=1.0,
        params={"std": 0.25},
    )

    # =====================================================================
    # 下肢协同约束（2D 协同流形）
    # =====================================================================
    # 膝关节引导：r = Σ|e·(q_norm - 0.5)|，
    # 高度误差加权约束膝关节偏离行程中点，防“跪式下蹲”等病态解，
    # 高度到位（e≈0）时约束自动消失（负权重挂载）；
    # knee_guidance = RewTerm(
    #     func=mdp.knee_guidance_l1,
    #     weight=-0.75,
    #     params={
    #         "command_name": "pelvis_height",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
    #     },
    # )
    # 下肢协同流形奖励（替代 knee_guidance）：
    #   公式：r = exp(-||d_⊥||²/σ²) - 1 ∈ [-1, 0]，函数返回非正值，正权重挂载；
    #   双腿矢状面 hip/knee/ankle 三 Pitch 关节按全行程归一（相对默认位，
    #   防膝大变幅掩盖踝微调），惩罚偏离 PC1 折叠协同 [-0.371,0.807,-0.459] 与
    #   PC2 次级补偿模态 [-0.548,0.209,0.810]（Gram-Schmidt 正交化）张成的 2D 协同平面；
    #   基向量来自重定向蹲起数据、以默认位为锚点的非中心化 SVD 拟合（解释 99.7% 方差），
    #   天然内含 G1 关节符号约定（蹲下时髋/踝为负、膝为正）；
    #   六关节顺序由函数内 preserve_order 按名解析，不依赖资产内部关节序；
    #   sigma 为流形带宽：残差距离达 sigma 时惩罚至满值 63%，0.2 ≈ 膝关节 33° 容差；
    #   过小会使严重病态姿态饱和在 -1 附近梯度变平，训练卡死时可放宽至 0.3~0.5
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
        },
    )
    # 髋关节回默认位：偏离默认位平方惩罚，命令高度 ≥ 0.735 m 门控（低区间放行屈髋）；
    # 默认位从资产动态读取（本项目 hip_yaw/roll 默认 0）
    hip_default_deviation = RewTerm(
        func=mdp.standing_joint_default_deviation_l2,
        weight=-0.5,
        params={
            "command_name": "pelvis_height",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]),
        },
    )
    # 踝关节回默认位：同上门控；默认位动态读取（ankle_pitch 默认 -0.2，不可用固定 target=0）
    ankle_default_deviation = RewTerm(
        func=mdp.standing_joint_default_deviation_l2,
        weight=-0.5,
        params={
            "command_name": "pelvis_height",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]),
        },
    )

    # =====================================================================
    # 躯干姿态与速度稳定性约束（使用 Isaac Lab 内置函数）
    # =====================================================================
    # 躯干朝向惩罚：r = -||g_x||² - ||g_y||²，惩罚重力在躯干 x/y 轴的投影，保持直立不翻转；
    # 内置 flat_orientation_l2 即该公式实现，取躯干连杆投影重力（注：root 为 pelvis）
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
    # 脚掌贴地约束：r = -Σ Var(H_i)，以足底四角点高度方差近似，量级较小（约 1e-3 级）
    # 故权重取大值；仅持续接触 ≥ 3·dt 的脚参与（官方门控，避免抬脚瞬态误罚）
    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel_var,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            "foot_length": 0.18,
            "foot_width": 0.08,
        },
    )
    # 足部防滑惩罚：r = -Σ ||v_i|| · I_contact，触地期间惩罚足端水平切向速度，
    # 复用既有 feet_slide（其实现即该公式，返回正惩罚量）
    feet_slip = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
        },
    )
    # 双足平行约束（官方实现）：r = -Var(D)，D 为左右脚三点（跟/中/尖）对应点间距集合，
    # 三点沿足长方向间隔 0.18 m 合成；仅在站立高度（命令 ≥ 0.735 m）生效，
    # 深蹲区间双脚布局本就不同故不约束（权重照抄官方未乘 dt 值）
    feet_parallel = RewTerm(
        func=mdp.feet_parallel_var,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_name": "pelvis_height",
        },
    )
    # 足端接触力峰值惩罚：r = -Σ ReLU(||F|| - F_max)，限制足端接触力超限，
    # 保护踝关节与真机结构（内置，阈值 400 N 与官方 max_contact_force 一致）
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
    # 站立蠕行惩罚：惩罚离地脚数量（垂直接触力 < 0.1 N 视为离地），
    # 抑制原地踏步/离地蠕动（官方 stand_still，已去门控）
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )

    # =====================================================================
    # 横向安全间距（双脚 / 双膝）
    # =====================================================================
    # 双脚横向安全距离：双边钳制——下界防并拢（常开），上界防越分越开（仅站立高度 ≥0.735 m
    # 门控，深蹲允许宽站姿）；公式整体非正，正权重作惩罚语义（与官方一致）
    feet_lateral = RewTerm(
        func=mdp.feet_lateral_separation,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_name": "pelvis_height",
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
            "command_name": "pelvis_height",
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
    # 扭矩输出惩罚（官方形式）：r = -Σ(τ_i / k_i)²，按刚度归一使髋/踝不同量级可比；
    # 仅下肢受控关节（不含腰/臂/手），防止深蹲保持时力矩爆炸（负权重惩罚）。
    # 注：踝关节实际刚度仅 20，直接归一会使踝扭矩主导整个惩罚项（踝 20 Nm → 1 rad²，
    # 而髋 100 Nm 才 1 rad²），故踝采用等效刚度 100 与髋膝对齐量级（负权重惩罚）
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
    # 关节位置限制：r = -Σ ReLU(|θ - θ_0| 越界量)，内置 joint_pos_limits 即对软限位外超出量
    # 的 ReLU 求和（软限位 = 硬限位 × 0.9），作用于腿部全部受控关节（负权重惩罚）
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
    # 扭矩超限限制：r = -Σ ReLU(|τ| - τ_max)，惩罚驱动指令超出额定扭矩；
    # 仅对显式执行器（DCMotor 腿/踝）有效，腰/臂/手的隐式执行器限位为 inf 会被自动跳过（负权重惩罚）
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
    # 关节速度限位：r = -Σ ReLU(|ω| - ω_max·soft_ratio)（内置，超限量裁剪至 1 防爆炸），
    # 官方权重极小（-2e-3），仅轻度约束腿部关节速度接近极限（负权重惩罚）
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
    #terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_levels = None

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
