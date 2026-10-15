# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp

##
# Pre-defined configs
##
from .utils import G1_29DOF_CFG


@configclass
class LowerLimbAgentPhysicsCfg(PresetCfg):
    """Physics-backend presets for the lower-limb agent environment.

    Every field is a selectable preset, chosen at launch with ``physics=<preset>``
    (``isaacsim_physx``, ``ovphysx``, ``physx``, ``newton_mjwarp``, ``newton_kamino``).
    :attr:`default` applies when no override is given; this project is launched with
    ``physics=isaacsim_physx``, so the PhysX branches carry the environment sizing.
    """

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
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: PhysxCfg = isaacsim_physx


##
# Scene definition
##


@configclass
class LowerLimbAgentSceneCfg(InteractiveSceneCfg):
    """Configuration for the lower-limb agent scene: flat ground, the G1 humanoid and its contact sensor."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = replace(G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # [h_offset, vx, vy, ωz, pelvis_dx, pelvis_dy]
    # 4 Mode: STAND / SQUAT / WALK / SQUAT_WALK
    task_command = mdp.SquatWalkCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 4.0),
        rel_mode_envs=(0.4, 0.4, 0.2, 0.0),
        ranges=mdp.SquatWalkCommandCfg.Ranges(
            height_offset=(-0.15, 0.02),
            lin_vel_x=(-0.3, 0.6),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.8, 0.8),
            pelvis_offset_x=(-0.03, 0.03),
            pelvis_offset_y=(-0.03, 0.03),
        ),
        pelvis_target_speed=0.02,
        pelvis_zero_probability=0.2,
        pelvis_success_threshold=0.02,
        height_success_threshold=0.03,
        debug_vis=False,
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
        # 510 dims = 102 dims * 5 times

        # 102 dims:
        #  - root status (9)  
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # - task command (6)
        velocity_commands = ObsTerm(func=mdp.task_command_velocity, params={"command_name": "task_command"})
        pelvis_height_cmd = ObsTerm(func=mdp.task_command_height, params={"command_name": "task_command"})
        pelvis_offset_cmd = ObsTerm(func=mdp.task_command_pelvis_xy, params={"command_name": "task_command"})

        # - body joint status (58)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # - last action (12) only lower-limb
        actions = ObsTerm(func=mdp.last_action)

        # - upper-limb target (17)
        upper_body_target = ObsTerm(
            func=mdp.upper_body_joint_pos_target_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist.*", "waist_.*_joint"]
                )
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5  # enable 5 times history

    # observation groups
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
    # upper-limb random disturbance.
    upper_body_disturbance = EventTerm(
        func=mdp.UpperBodyDisturbanceEvent,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        params={
            "disturb_groups": [
                {
                    # group 0: arms(14 DOF)
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist.*"]),
                    "joint_noise_frac": 0.2,  
                },
                {
                    # group 1: waist(3 DOF)
                    "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"]),
                    "joint_noise_frac": 0.1,  
                },
            ],
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # alive and terminal
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # command reward
    # pelvis height reward: r = exp(-4·|e|)
    track_pelvis_height = RewTerm(
        func=mdp.track_pelvis_height_exp,
        weight=2.0,
        params={"command_name": "task_command"},
    )
    # velocity reward: r = Σexp(-e²/(2σ²))
    track_velocity = RewTerm(
        func=mdp.track_velocity_exp,
        weight=1.0,
        params={"command_name": "task_command", "std": 0.25},
    )
    # pelvis horizontal plane reward: r = exp(-||e_xy||²/(2σ²))
    track_pelvis_xy = RewTerm(
        func=mdp.track_pelvis_xy_exp,
        weight=0.4,
        params={"command_name": "task_command", "std": 0.03},
    )

    # guidance item
    # lower-limb synergy: r = exp(-||d⊥||²/σ²) - 1 ∈ [-1, 0]
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
    # hip ankle deviation penalty
    hip_default_deviation = RewTerm(
        func=mdp.standing_joint_default_deviation_l2,
        weight=-0.5,
        params={
            "command_name": "task_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]),
            "velocity_command_name": "task_command",
        },
    )
    ankle_default_deviation = RewTerm(
        func=mdp.standing_joint_default_deviation_l2,
        weight=-0.5,
        params={
            "command_name": "task_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]),
            "velocity_command_name": "task_command",
        },
    )
    # hip soft deviation penalty (only hip yaw)
    hip_yaw_soft_deviation = RewTerm(
        func=mdp.hip_yaw_soft_deviation_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"]),
            "margin": 0.15,
            "velocity_command_name": "task_command",
        },
    )

    # body penalty
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    vertical_bounce = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    tilt_wobble = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)

    # foot guidance
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
    feet_slip = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
        },
    )
    feet_parallel = RewTerm(
        func=mdp.feet_parallel_var,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_name": "task_command",
            "velocity_command_name": "task_command",
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-2.5e-4,
        params={
            "threshold": 400.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
        },
    )
    no_fly = RewTerm(
        func=mdp.no_fly,
        weight=0.75,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            "velocity_command_name": "task_command",
            "zero_modes": (0, 1),  # STAND / SQUAT
        },
    )
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

    # feet and knee distance guidance
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

    # smooth
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_acc_l2, weight=-0.05)
    dof_torques = RewTerm(
        func=mdp.joint_torques_l2_normalized,
        weight=-2.5e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
            ),
        },
    )

    # joint safety
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

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # contact
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    knee_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_link"), "threshold": 1.0}
    )
    # body item
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
    """Curriculum terms for the MDP: command range, upper-limb disturbance and mode introduction."""

    terrain_levels = None

    # command curriculum
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
            "update_period": 5000,
        },
    )

    # upper-limb disturbance curriculum
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

    # mode ratio curriculum
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
class LowerLimbAgentEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 lower-limb agent environment (stand / squat / walk / squat-walk)."""
    training_phase: int = 1
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(physics=LowerLimbAgentPhysicsCfg())
    # Scene settings
    scene: LowerLimbAgentSceneCfg = LowerLimbAgentSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 6
        # visualizer camera settings
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

    def play_mode(self) -> None:
        """Evaluate the configured task distribution under nominal, undisturbed conditions.

        The standard play entry point calls this after configuration construction. Command
        ranges and mode weights are preserved; checkpoint curriculum restoration is separate.
        CLI overrides may still request another environment count after this hook.
        """
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 10)
        self.curriculum = None
        self.events.physics_material = None
        if self.events.upper_body_disturbance is not None:
            from ...curriculum_state import disturbance_layout

            self._g1_disturbance_layout = disturbance_layout(self.events.upper_body_disturbance)
        self.events.upper_body_disturbance = None
        # Keep physical resets active: removing these events would leave terminated robots
        # in their old state. Empty ranges reset to the authored default pose/velocity.
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    def teleop_mode(self) -> None:
        """Use normal evaluation with one manually controlled robot and no timeout.

        Dispatch through the phase's play_mode() so inherited mass/gain/push randomization
        is disabled too. Keep episode_length_s, action and observation layouts compatible
        with the training checkpoint; only remove the timeout termination itself.
        """
        self.play_mode()
        self.scene.num_envs = 1
        self.terminations.time_out = None
        self.commands.task_command.debug_vis = True
        self.commands.task_command.pelvis_debug_vis = False
