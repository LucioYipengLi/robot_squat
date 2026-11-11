# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three-stage training: inherit actual curriculum state from the predecessor checkpoint.

Phase 2 adds static physics randomization; Phase 3 adds external pushes. Neither phase
hard-codes the previous phase's command ranges, mode weights or disturbance magnitude.
The one-shot introduction curriculum remains available if Phase 1 ended before its trigger.

Use the project entrypoint for curriculum-aware saves and restores::

    uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase2-v0 \\
        --checkpoint <phase1_model.pt> --max_iterations 5000 physics=isaacsim_physx
    uv run --extra isaacsim python -m g1.rl train --task Template-G1-Phase3-v0 \\
        --checkpoint <phase2_model.pt> --max_iterations 3000 physics=isaacsim_physx

Standard play_mode() still defines nominal evaluation. Use ``python -m g1.rl play`` to
also recover the checkpoint's actual command distribution. Old checkpoints require an
explicit --curriculum-snapshot JSON; unknown training conditions are never inferred.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ... import mdp
from .env_cfg import CurriculumCfg, EventCfg, LowerLimbAgentEnvCfg


@configclass
class EventCfg_Phase2(EventCfg):
    """Phase 2 event configuration: adds static physics DR on top of Phase 1 baseline.

    Declaration order matters for mass randomization:
        link_mass (scale all) → hand_payload (add hands) → ankle_payload (add ankles)
    All three use startup mode so the caches are populated once and do not conflict.
    """

    # ── Override: widen friction/restitution ranges (Phase 1 → Phase 2) ──────────
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 256,
        },
    )

    # ── New: link mass scaling (±10%, conservative start; upgrade to ±20% when stable) ──
    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # ── New: base COM offset (torso_link) ──────────────────────────────────────────
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.06, 0.06), "y": (-0.06, 0.06), "z": (-0.03, 0.03)},
        },
    )

    # ── New: hand payload (0~0.3 kg附加质量, 模拟手持物) ───────────────────────────
    hand_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_hand_palm_link"),
            "mass_distribution_params": (0.0, 0.3),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # ── New: ankle payload (0~0.2 kg附加质量, 影响摆动动力学) ──────────────────────
    ankle_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "mass_distribution_params": (0.0, 0.2),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # ── New: PD actuator gain randomization (腿/脚关节, ±10%) ──────────────────────
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class LowerLimbAgentEnvCfg_Phase2(LowerLimbAgentEnvCfg):
    """Resume Phase 1's actual task distribution, adding static physics DR."""

    training_phase: int = 2
    events: EventCfg_Phase2 = EventCfg_Phase2()

    def play_mode(self) -> None:
        super().play_mode()
        self.events.link_mass = None
        self.events.base_com = None
        self.events.hand_payload = None
        self.events.ankle_payload = None
        self.events.actuator_gains = None


@configclass
class EventCfg_Phase3(EventCfg_Phase2):
    """Phase 3 event configuration: adds external push on top of Phase 2 static DR.

    Inherits all Phase 2 DR events (link_mass, base_com, payloads, actuator_gains,
    widened friction) and adds velocity-based push disturbance.
    """

    # World-frame velocity increments, sampled every 4~6 s. The curriculum expands
    # these initial per-axis bounds toward ±0.7 m/s as performance permits.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={"velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
    )


@configclass
class CurriculumCfg_Phase3(CurriculumCfg):
    """Keep inherited curricula and independently pace the external push magnitude."""

    push_velocity_magnitude = CurrTerm(
        func=mdp.push_velocity_magnitude_curriculum,
        params={
            "event_term_name": "push_robot",
            "command_name": "task_command",
            "survival_gate": 0.95,  # 存活比例 EMA 门槛；不是回合成功率。
            "task_success_gate": 0.80,  # 全局联合成功率门槛；与存活 EMA 同时超过才推进。
            "velocity_final": 0.70,  # m/s，单轴水平速度增量最终上限。
            "velocity_step": 0.05,  # m/s，每个合格检查窗口只增加一次。
            "ema_beta": 0.02,  # 存活 EMA 更新系数，越大越快响应近期表现。
            "update_period": 5000,  # 控制步，默认约 100 个 PPO 迭代。
        },
    )


@configclass
class LowerLimbAgentEnvCfg_Phase3(LowerLimbAgentEnvCfg_Phase2):
    """Resume Phase 2's actual task distribution, adding external pushes."""

    training_phase: int = 3
    events: EventCfg_Phase3 = EventCfg_Phase3()
    curriculum: CurriculumCfg_Phase3 = CurriculumCfg_Phase3()

    def play_mode(self) -> None:
        super().play_mode()
        self.events.push_robot = None
