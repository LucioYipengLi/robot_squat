# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 single/dual wrist tracking with world-fixed goals and pelvis-frame observations."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp
from .utils.assets import G1_29DOF_CFG


@configclass
class UpperLimbAgentPhysicsCfg(PresetCfg):
    """G1-sized physics presets, independently owned by this task."""

    isaacsim_physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    ovphysx: OvPhysxCfg = OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=95,
            nconmax=10,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: PhysxCfg = isaacsim_physx


@configclass
class UpperLimbAgentSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.UpperJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
        acceleration_reference_body_name="pelvis",
        acceleration_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    )


@configclass
class CommandsCfg:
    wrist_pose = mdp.BimanualPoseCommandCfg(
        asset_name="robot",
        reference_body_name="pelvis",
        body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
        # 每回合选择模式：左手、右手、双手；三项概率之和为 1。
        mode_probabilities=(0.33, 0.33, 0.34),
        # 保持时间 = clamp(最远有效目标到骨盆的距离 / 距离时间参数, 时间下限, 时间上限)。
        resampling_time_range=(2.0, 4.0),  # s
        distance_reference_speed=0.08,  # m/s，仅用于计算目标保持时间。
        workspace_quality=mdp.WorkspaceQualityCfg(
            map_path="data/workspace_q0_v1.npz",  # 相对 upper_limb_agent 包目录；也接受绝对路径。
            position_tolerance_m=0.01,  # 所有活动手均达标后才发放 Q 奖励；单位 m。
        ),
        # 位置偏移沿默认骨盆坐标轴（m）；姿态偏移叠加在默认腕部朝向上（rad）。
        left_offsets=mdp.BimanualPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.15),
            pos_y=(-0.05, 0.10),
            pos_z=(-0.05, 0.20),
            roll=(-0.3, 0.3),
            pitch=(-0.3, 0.3),
            yaw=(-0.3, 0.3),
        ),
        right_offsets=mdp.BimanualPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.15),
            pos_y=(-0.10, 0.05),
            pos_z=(-0.05, 0.20),
            roll=(-0.3, 0.3),
            pitch=(-0.3, 0.3),
            yaw=(-0.3, 0.3),
        ),
        debug_vis=False,  # play_mode() 中开启当前腕部与有效目标标记。
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Single-frame policy input: 93 dimensions."""

        # 双腕目标位姿（14 维，实时盆骨坐标系）与左右有效标志（2 维）
        targets = ObsTerm(func=mdp.generated_commands, params={"command_name": "wrist_pose"})
        active_arms = ObsTerm(func=mdp.active_arms, params={"command_name": "wrist_pose"})

        # 双腕实际位姿（14 维，盆骨坐标系）
        wrist_poses = ObsTerm(func=mdp.end_effector_pose, params={"command_name": "wrist_pose"})

        # 双腕跟踪误差（12 维）：每腕位置差 3 维（m）与旋转向量 3 维（rad），闲置侧置零。
        wrist_pose_error = ObsTerm(func=mdp.end_effector_pose_error, params={"command_name": "wrist_pose"})

        # 受控关节位置与速度（各 17 维，与动作顺序一致）
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
                    preserve_order=True,
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
                    preserve_order=True,
                ),
            },
            scale=0.1,
        )

        # 上一帧动作（17 维）
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # No root pose randomization: keep the authored placement relative to each environment origin.
    reset_joints = EventTerm(func=mdp.reset_robot_joints, mode="reset")


@configclass
class RewardsCfg:
    """Wrist pose tracking and control regularization."""

    # 双腕位置跟踪
    left_position = RewTerm(
        func=mdp.end_effector_position_tracking,
        weight=1.0,
        params={"command_name": "wrist_pose", "arm_index": 0, "std": 0.1},
    )
    right_position = RewTerm(
        func=mdp.end_effector_position_tracking,
        weight=1.0,
        params={"command_name": "wrist_pose", "arm_index": 1, "std": 0.1},
    )

    # 双尺度四元数朝向奖励（正权重）；e_q = ||vec(q_goal * q_current^-1)||，sigma 无量纲。
    # 核函数 G(e_q, sigma) = exp(-e_q² / sigma²)，fine_weight 为精细项占比。
    left_orientation = RewTerm(
        func=mdp.end_effector_orientation_tracking,
        weight=0.3,
        params={
            "command_name": "wrist_pose",
            "arm_index": 0,
            "sigma_coarse": 0.3,
            "sigma_fine": 0.05,
            "fine_weight": 0.4,
        },
    )
    right_orientation = RewTerm(
        func=mdp.end_effector_orientation_tracking,
        weight=0.3,
        params={
            "command_name": "wrist_pose",
            "arm_index": 1,
            "sigma_coarse": 0.3,
            "sigma_fine": 0.05,
            "fine_weight": 0.4,
        },
    )

    # Only the seven joints of the inactive arm are encouraged back to their default angles.
    idle_arm_pose = RewTerm(
        func=mdp.idle_arm_default_pose,
        weight=-1.0,
        params={"command_name": "wrist_pose"},
    )

    # 目标骨盆相对位置的 0 kg 空间 Q；默认只记录指标，正权重启用奖励。
    workspace_quality = RewTerm(
        func=mdp.workspace_quality_bonus,
        weight=0.0,
        params={"command_name": "wrist_pose"},
    )

    # 动作与实际关节运动平滑
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    # Soft penalty on processed-goal speed; enable with a negative weight for a separate experiment.
    target_speed = RewTerm(
        func=mdp.joint_target_speed_excess_l2,
        weight=0.0,
        params={"action_name": "joint_pos", "soft_limit": 2.5},  # rad/s; does not limit execution
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2_substeps,
        weight=-1.0e-7,
        # Mean squared acceleration across every visible physics substep, for the action's 17 joints.
        # Recalibrate weight separately when comparing with the former final-substep reward.
        params={"action_name": "joint_pos"},
    )
    # 双腕 link 原点在骨盆坐标系中的相对线加速度平方和，对所有物理子步取均值。
    # 所有指令模式均约束双侧；权重保留作新口径的量级校准起点。
    end_effector_acc = RewTerm(
        func=mdp.end_effector_relative_lin_acc_l2_substeps,
        weight=-1e-5,
        params={"action_name": "joint_pos"},
    )

    # 受控关节力矩与位置限制
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    # 超过物理力矩上限 80% 后按超出量惩罚；物理上限仍由执行器配置决定。
    torque_limits = RewTerm(
        func=mdp.joint_torque_limits,
        weight=-1.0e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
                preserve_order=True,
            ),
            "soft_ratio": 0.8,
        },
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class UpperLimbAgentEnvCfg(ManagerBasedRLEnvCfg):
    """Track world-fixed wrist goals in left-only, right-only and dual-arm episodes."""

    task_family: str = "upper_limb_agent"
    scene: UpperLimbAgentSceneCfg = UpperLimbAgentSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = UpperLimbAgentPhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))

    def play_mode(self) -> None:
        super().play_mode()
        self.commands.wrist_pose.debug_vis = True
