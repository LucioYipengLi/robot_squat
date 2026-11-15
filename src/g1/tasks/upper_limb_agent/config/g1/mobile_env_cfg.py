"""Free-base wrist tracking with six Lower commands and an episode-owned range curriculum."""

from isaaclab.envs.mdp import illegal_contact
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass, replace

from ... import mdp
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
            failure_rate_limit=0.01,
            ema_alpha=0.20,
            check_period_steps=1200,
            min_episodes_per_class=512,
        ),
    )


@configclass
class MobileObservationsCfg(ObservationsCfg):
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
class MobileEventsCfg(EventCfg):
    # Root and joints return to the same authored placement each episode.
    reset_root = EventTerm(func=mdp.reset_mobile_root, mode="reset")


@configclass
class MobileRewardsCfg(RewardsCfg):
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
        params={"action_name": "lower_command"},
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


@configclass
class MobileTerminationsCfg(TerminationsCfg):
    fallen = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    low_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    base_contact = DoneTerm(
        func=illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
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

    def __post_init__(self):
        super().__post_init__()
        from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg

        lower = LowerLimbAgentEnvCfg()
        self.sim.dt = lower.sim.dt
        self.decimation = lower.decimation
        self.sim.render_interval = self.decimation
