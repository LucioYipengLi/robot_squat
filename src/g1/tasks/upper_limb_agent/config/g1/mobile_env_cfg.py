"""Free-base wrist tracking with six Lower commands and an episode-owned range curriculum."""

import math

from isaaclab.envs.mdp import illegal_contact
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass, replace

from ... import mdp
from ...mdp.actions import MOBILE_OBSERVATION_JOINT_NAMES
from .env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    UpperLimbAgentEnvCfg,
    UpperLimbAgentSceneCfg,
)
from .utils.assets import mobile_robot_cfg


@configclass
class MobileSceneCfg(UpperLimbAgentSceneCfg):
    robot = mobile_robot_cfg()
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=1 / 120, history_length=3)
    torso_contact_pairs = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        update_period=1 / 120,
        history_length=3,
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{path}" for path in mdp.ARM_CONTACT_BODY_PATHS],
    )


@configclass
class MobileActionsCfg:
    # ActionManager processes in declaration order: decode all six commands
    # before joint_pos constructs this step's Lower observation and infers once.
    lower_command = mdp.LowerVelocityCommandActionCfg(
        asset_name="robot",
        command_ranges=((-0.50, 1.00), (-0.20, 0.20), (-0.80, 0.80)),
        reward_reference_scale=(0.30, 0.10, 0.30),  # Fixed H5C change-cost scales; magnitude uses episode bounds.
        range_command_name="wrist_pose",  # Decode directly into this episode's curriculum ranges.
        enabled=True,  # False forces zero velocity, independently of height_offset.
    )
    height_offset = mdp.LowerHeightCommandActionCfg(
        asset_name="robot",
        height_offset_range=(-0.40, 0.05),  # m relative to default root height; zero action keeps standing height.
        reward_reference_range=(-0.25, 0.02),  # Retain H5C's negative/positive reward scales.
        range_command_name="wrist_pose",
        enabled=True,  # False fixes height at zero and excludes it from lower_command_rate.
    )
    pelvis_offset = mdp.LowerPelvisCommandActionCfg(
        asset_name="robot",
        offset_ranges=((-0.03, 0.03), (-0.03, 0.03)),
        reward_reference_scale=(0.03, 0.03),
        range_command_name="wrist_pose",
        target_speed=0.02,  # m/s, two-dimensional norm, matching frozen Lower training.
    )
    joint_pos = mdp.FrozenLowerActionCfg(
        asset_name="robot",
        joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
        command_action_name="lower_command",
        height_action_name="height_offset",
        pelvis_action_name="pelvis_offset",
        use_training_envelope=True,
        standing_linear_threshold=0.03,  # m/s; small velocity requests map to stationary mode.
        standing_yaw_threshold=0.05,  # rad/s; stationary mode permits pelvis offsets.
        base_body_name="pelvis",  # Link origin for all three-axis world base-motion costs.
        vertical_speed_rms_window_s=0.1,  # Six control periods / twelve physical samples at 60 / 120 Hz.
        acceleration_reference_body_name=ActionsCfg().joint_pos.acceleration_reference_body_name,
        acceleration_body_names=ActionsCfg().joint_pos.acceleration_body_names,
    )


@configclass
class MobileCommandsCfg(CommandsCfg):
    wrist_pose = replace(
        CommandsCfg().wrist_pose,
        height_task=mdp.BimanualPoseCommandCfg.HeightTaskCfg(
            probability=0.30,  # 每回合：70%常规目标、30%额外高度随机化目标。
            shift_range=(-0.15, -0.05),  # 世界Z额外偏移，m；独立于Lower能力和高度命令范围。
            hold_episode=True,  # 第二类整个位姿保持一回合；常规目标仍按原间隔刷新。
        ),
        planar_task=mdp.PlanarTaskCfg(
            probability=0.2,
            initial_half_extent_m=0.10,
            half_extent_m=2.5,  # 作业站位相对初始盆骨的5 m × 5 m空间。
            initial_bearing_rad=math.radians(10),
            bearing_rad=math.pi,
            curriculum_knots=tuple(
                (level / 10, extent, math.radians(bearing))
                for level, extent, bearing in (
                    (0, 0.10, 10),
                    (1, 0.15, 15),
                    (2, 0.20, 20),
                    (3, 0.25, 30),
                    (4, 0.40, 45),
                    (5, 0.65, 65),
                    (6, 1.00, 90),
                    (7, 1.35, 115),
                    (8, 1.75, 140),
                    (9, 2.10, 160),
                    (10, 2.50, 180),
                )
            ),
            transition_sampling=mdp.PlanarTransitionSamplingCfg(
                probability_knots=((0.0, 0.0, 0.25), (0.30, 0.0, 0.25), (0.40, 0.25, 3 / 14), (1.0, 1.0, 0.0)),
                turn_bearing_bridge=(0.20, math.radians(35)),
            ),
            max_goal_distance_m=5.0,  # 各活动腕目标相对初始盆骨的三维距离上限。
            timing=mdp.GoalTimingCfg(
                linear_speed=0.4,
                linear_acceleration=0.3,
                angular_speed=0.4,
                angular_acceleration=0.4,
                ability_fraction=0.8,
                settle_s=2.0,
                evaluation_s=10.0,
                near_episode_s=10.0,
            ),
        ),
        planar_approach=mdp.PlanarApproachCfg(waist_posture_weight=-0.20),
        range_curriculum=mdp.RangeCurriculumCfg(
            initial_level=0,
            num_levels=11,
            initial_lower_ranges=(
                (-0.10, 0.01),
                (-0.10, 0.10),
                (-0.03, 0.03),
                (-0.10, 0.10),
                (-0.005, 0.005),
                (-0.005, 0.005),
            ),
            initial_position_scale=0.50,
            initial_orientation_scale=0.50,
            initial_height_shift=(-0.05, -0.02),
            position_tolerance_m=0.015,
            orientation_tolerance_rad=0.10,
            pose_ready_gate=0.80,
            failure_rate_limit=0.01,  # 每10机器人秒的失败次数上限。
            ema_alpha=0.20,
            check_period_steps=3600,  # 60秒，覆盖长回合并收集每类足量样本。
            min_episodes_per_class=512,
            planar_sampling=mdp.AdaptivePlanarSamplingCfg(
                enabled=True,
                history_windows=4,  # 当前每150轮检查一次：前300轮对比后300轮。
                minimum_relative_improvement=0.10,
                probability_step=0.05,
                maximum_probability=0.50,
                other_tasks_pose_ready_gate=0.85,  # 常规/高度任务还须分别通过当前窗口的样本量和失败频率门槛。
                cooldown_windows=1,  # 加码后下一窗口只统计，随后恢复判断。
            ),
        ),
    )


@configclass
class LegacyMobileObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """109 inputs: 93 Upper, 10 pelvis state and six effective Lower commands."""

        actions = ObsTerm(func=mdp.term_action, params={"action_name": "joint_pos"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_height = ObsTerm(func=mdp.base_height)
        lower_command = ObsTerm(func=mdp.lower_velocity_command, params={"action_name": "lower_command"})
        height_offset = ObsTerm(func=mdp.lower_height_offset, params={"action_name": "height_offset"})
        pelvis_offset = ObsTerm(func=mdp.lower_pelvis_offset, params={"action_name": "pelvis_offset"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class MobileObservationsCfg(LegacyMobileObservationsCfg):
    @configclass
    class PolicyCfg(LegacyMobileObservationsCfg.PolicyCfg):
        """132 features per step, stacked across five control steps."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(MOBILE_OBSERVATION_JOINT_NAMES), preserve_order=True
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(MOBILE_OBSERVATION_JOINT_NAMES), preserve_order=True
                )
            },
            scale=0.1,
        )
        base_height: ObsTerm | None = None

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.history_order = "term"

    policy: PolicyCfg = PolicyCfg()


@configclass
class MobileEventsCfg(EventCfg):
    # Root and joints return to the same authored placement each episode.
    reset_root = EventTerm(func=mdp.reset_mobile_root, mode="reset")


@configclass
class MobileRewardsCfg(RewardsCfg):
    planar_approach = RewTerm(
        func=mdp.PlanarApproachReward,
        weight=1.0,
        params={"command_name": "wrist_pose", "action_name": "joint_pos"},
    )
    # One shared change cost covers six effective commands in fixed physical scales.
    # Amplitude and overflow costs below still read only the three velocities.
    action_rate = RewTerm(func=mdp.action_term_rate_l2, weight=-0.03, params={"action_name": "joint_pos"})
    lower_command_rate = RewTerm(
        func=mdp.lower_command_rate_l2,
        weight=-0.10,
        params={
            "action_name": "lower_command",
            "height_action_name": "height_offset",
            "pelvis_action_name": "pelvis_offset",
        },
    )
    # Raw overflow keeps the R9C rule; magnitude measures use of each episode's velocity ranges.
    lower_command_overflow = RewTerm(
        func=mdp.lower_command_overflow_l2, weight=-3e-4, params={"action_name": "lower_command"}
    )
    lower_command_magnitude = RewTerm(
        # At a signed range endpoint, each velocity axis contributes one before weighting.
        func=mdp.lower_command_magnitude_l2,
        weight=-0.05,
        params={
            "action_name": "lower_command",
            "command_name": "wrist_pose",
            "near_planar_threshold": 0.5,
            "near_planar_gain": 0.0,
        },
    )
    base_lin_vel = RewTerm(func=mdp.base_lin_vel_l2_substeps, weight=0.0, params={"action_name": "joint_pos"})
    base_ang_vel = RewTerm(func=mdp.base_ang_vel_l2_substeps, weight=0.0, params={"action_name": "joint_pos"})
    base_lin_acc = RewTerm(func=mdp.base_lin_acc_l2_substeps, weight=0.0, params={"action_name": "joint_pos"})
    base_ang_acc = RewTerm(func=mdp.base_ang_acc_l2_substeps, weight=0.0, params={"action_name": "joint_pos"})
    # H0-H2 calibration: allow ordinary crouching/walking, penalize vertical impulses.
    base_vertical_acc = RewTerm(
        func=mdp.base_vertical_acc_excess_l2_substeps,
        weight=-0.12,
        params={"action_name": "joint_pos", "soft_limit": 4.0},
    )
    # Task-level bonus: current world-pose accuracy and a full rolling speed window.
    dwell = RewTerm(
        func=mdp.DwellReward,
        weight=0.2,
        params={
            "command_name": "wrist_pose",
            "action_name": "joint_pos",
            "position_tolerance_m": 0.01,
            "orientation_tolerance_rad": 0.04,
            "vertical_speed_tolerance_m_s": 0.02,
        },
    )
    # Failure affects Upper's return; the frozen Lower actor receives no optimizer updates.
    termination = RewTerm(func=mdp.is_terminated, weight=-10.0)
    torso_contact = RewTerm(
        func=mdp.TorsoContactPenalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("torso_contact_pairs"),
            "soft_threshold": 1.0,
            "hard_threshold": 20.0,
            "contact_cost": 0.25,
        },
    )


@configclass
class MobileTerminationsCfg(TerminationsCfg):
    time_out = DoneTerm(func=mdp.task_time_out, time_out=True, params={"command_name": "wrist_pose"})
    fallen = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    low_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    base_contact = DoneTerm(
        func=mdp.torso_contact_termination,
        params={"sensor_cfg": SceneEntityCfg("torso_contact_pairs"), "threshold": 20.0, "other_threshold": 1.0},
    )
    knee_contact = DoneTerm(
        func=illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_link"), "threshold": 1.0},
    )


@configclass
class UpperLimbMobileEnvCfg(UpperLimbAgentEnvCfg):
    """Train 23 actions (six Lower commands + 17 joints), with frozen Lower at 60 Hz."""

    scene: MobileSceneCfg = MobileSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: MobileActionsCfg = MobileActionsCfg()
    observations: MobileObservationsCfg = MobileObservationsCfg()
    commands: MobileCommandsCfg = MobileCommandsCfg()
    events: MobileEventsCfg = MobileEventsCfg()
    rewards: MobileRewardsCfg = MobileRewardsCfg()
    terminations: MobileTerminationsCfg = MobileTerminationsCfg()
    legacy_observations: bool = False

    def __post_init__(self):
        super().__post_init__()
        from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg

        lower = LowerLimbAgentEnvCfg()
        self.episode_length_s = 35.0
        self.sim.dt = lower.sim.dt
        self.decimation = lower.decimation
        self.sim.render_interval = self.decimation
