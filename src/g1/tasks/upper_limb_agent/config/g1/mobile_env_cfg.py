"""Free-base wrist tracking: Upper chooses velocity and height for a frozen Lower actor."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

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


@configclass
class MobileActionsCfg:
    # ActionManager processes in declaration order: publish velocity commands
    # and height before joint_pos constructs this step's Lower observation and infers once.
    lower_command = mdp.LowerVelocityCommandActionCfg(
        asset_name="robot",
        command_scale=(0.3, 0.1, 0.3),  # vx [m/s], vy [m/s], yaw rate [rad/s]; zero maps to zero.
        enabled=True,  # False forces zero velocity, independently of height_offset.
    )
    height_offset = mdp.LowerHeightCommandActionCfg(
        asset_name="robot",
        height_offset_range=(-0.15, 0.02),  # m relative to default root height; zero action keeps standing height.
        enabled=True,  # False fixes height at zero and excludes it from lower_command_rate.
    )
    joint_pos = mdp.FrozenLowerActionCfg(
        asset_name="robot",
        joint_names=list(mdp.UPPER_BODY_JOINT_NAMES),
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
        command_action_name="lower_command",
        height_action_name="height_offset",
        base_body_name="pelvis",  # Link origin for all three-axis world base-motion costs.
        acceleration_reference_body_name=ActionsCfg().joint_pos.acceleration_reference_body_name,
        acceleration_body_names=ActionsCfg().joint_pos.acceleration_body_names,
    )


@configclass
class MobileObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """107 inputs: 93 Upper, 10 pelvis state, 3 velocity commands and 1 height offset."""

        actions = ObsTerm(func=mdp.term_action, params={"action_name": "joint_pos"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_height = ObsTerm(func=mdp.base_height)
        lower_command = ObsTerm(func=mdp.lower_velocity_command, params={"action_name": "lower_command"})
        height_offset = ObsTerm(func=mdp.lower_height_offset, params={"action_name": "height_offset"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class MobileEventsCfg(EventCfg):
    # Root and joints return to the same authored placement each episode.
    reset_root = EventTerm(func=mdp.reset_mobile_root, mode="reset")


@configclass
class MobileRewardsCfg(RewardsCfg):
    # One shared change cost covers all four normalized velocity/height commands.
    # Amplitude and overflow costs below still read only the three velocities.
    action_rate = RewTerm(func=mdp.action_term_rate_l2, weight=-0.03, params={"action_name": "joint_pos"})
    lower_command_rate = RewTerm(
        func=mdp.lower_command_rate_l2,
        weight=-0.10,
        params={"action_name": "lower_command", "height_action_name": "height_offset"},
    )
    # Retain the R9C velocity penalties; the four three-axis base costs remain disabled.
    lower_command_overflow = RewTerm(
        func=mdp.lower_command_overflow_l2, weight=-3e-4, params={"action_name": "lower_command"}
    )
    lower_command_magnitude = RewTerm(
        func=mdp.lower_command_magnitude_l2, weight=-0.05, params={"action_name": "lower_command"}
    )
    base_lin_vel = RewTerm(func=mdp.base_lin_vel_l2_substeps, weight=0.0, params={"action_name": "joint_pos"})
    base_ang_vel = RewTerm(func=mdp.base_ang_vel_l2_substeps, weight=0.0, params={"action_name": "joint_pos"})
    base_lin_acc = RewTerm(func=mdp.base_lin_acc_l2_substeps, weight=0.0, params={"action_name": "joint_pos"})
    base_ang_acc = RewTerm(func=mdp.base_ang_acc_l2_substeps, weight=0.0, params={"action_name": "joint_pos"})
    # H0-H2 calibration: allow ordinary crouching/walking, penalize vertical impulses.
    base_vertical_acc = RewTerm(
        func=mdp.base_vertical_acc_excess_l2_substeps,
        weight=-0.05,
        params={"action_name": "joint_pos", "soft_limit": 4.0},
    )
    # Failure affects Upper's return; the frozen Lower actor receives no optimizer updates.
    termination = RewTerm(func=mdp.is_terminated, weight=-10.0)


@configclass
class MobileTerminationsCfg(TerminationsCfg):
    fallen = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    low_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})


@configclass
class UpperLimbMobileEnvCfg(UpperLimbAgentEnvCfg):
    """Train 21 actions (3 velocity + 1 height offset + 17 joints), with frozen Lower at 60 Hz."""

    scene: MobileSceneCfg = MobileSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: MobileActionsCfg = MobileActionsCfg()
    observations: MobileObservationsCfg = MobileObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
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
