# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Upper-owned G1 asset; the mobile variant synchronizes Lower leg and foot actuators."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"),
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        fix_root_link=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(0, 0, 0.7071, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            actuator_effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            actuator_velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=180.0,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            actuator_effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            actuator_velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            joint_effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            joint_velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 40.0,
                "waist_pitch_joint": 40.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            joint_effort_limit=25,
            joint_velocity_limit=37,
            stiffness=40.0,
            damping=1.0,
            armature={
                ".*_shoulder_.*": 0.001,
                ".*_elbow_.*": 0.001,
                ".*_wrist_.*_joint": 0.001,
            },
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_.*",
                ".*_middle_.*",
                ".*_thumb_.*",
            ],
            joint_effort_limit=300,
            joint_velocity_limit=100,
            stiffness=20,
            damping=2,
            armature=0.001,
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)


def mobile_robot_cfg() -> ArticulationCfg:
    """Keep Upper-owned waist/arms; synchronize legs and feet with the latest Lower asset."""
    import copy

    from g1.tasks.lower_limb_agent.config.g1.utils.assets import G1_29DOF_CFG as lower_robot

    robot = copy.deepcopy(G1_29DOF_CFG)
    robot.spawn.fix_root_link = False
    for name in ("legs", "feet"):
        robot.actuators[name] = copy.deepcopy(lower_robot.actuators[name])
    return robot
