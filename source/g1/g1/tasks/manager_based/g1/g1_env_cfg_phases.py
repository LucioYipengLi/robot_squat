# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-based environment configurations for multi-stage domain randomization training.

Each phase inherits from the previous one and:
1. Hard-codes the curriculum state (ranges, mode weights, disturbance magnitude) to the
   **endpoint of the previous phase** -- solving the "curriculum state not persisted in
   checkpoint" problem identified in the resume audit.
2. Adds the new domain randomization events for that phase.
3. Disables curriculum terms that are no longer relevant (e.g. one-shot mode introduction).

Usage (续训命令)::

    # Phase 2: 从 Phase 1 checkpoint 续训
    python g1/scripts/rsl_rl/train_rsl_rl.py --task Template-G1-Phase2-v0 \
        --resume --load_run <phase1_run_dir> --checkpoint model_xxxx.pt \
        --max_iterations 5000

    # Phase 3: 从 Phase 2 checkpoint 续训
    python g1/scripts/rsl_rl/train_rsl_rl.py --task Template-G1-Phase3-v0 \
        --resume --load_run <phase2_run_dir> --checkpoint model_xxxx.pt \
        --max_iterations 3000

Important:
    Before launching each phase, verify the hard-coded curriculum endpoint values below
    against the actual TensorBoard curves (``Curriculum/command_range/*``) from the
    previous phase's final run. Adjust if the curriculum stalled before reaching the target.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .g1_env_cfg import CurriculumCfg, EventCfg, G1EnvCfg

##
# Phase 2: Static physics parameter randomization
##


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
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
            ),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class CurriculumCfg_Phase2(CurriculumCfg):
    """Phase 2 curriculum: disable the one-shot mode introduction (already applied in config).

    The performance-driven curricula (command_range, upper_body_disturbance_magnitude) are
    KEPT ACTIVE -- they will automatically freeze when DR causes success rates to drop,
    then resume expansion once the policy adapts. This is the desired self-pacing behavior.
    """

    # SQUAT_WALK mode is already introduced via rel_mode_envs in Phase 2 config;
    # the one-shot trigger curriculum is no longer needed.
    squat_walk_mode_introduction = None


@configclass
class G1EnvCfg_Phase2(G1EnvCfg):
    """Phase 2: Static physics DR, resume from Phase 1 checkpoint.

    Curriculum state is hard-coded to Phase 1 endpoint values. Before launching, verify
    these against TensorBoard ``Curriculum/command_range/*`` and ``Curriculum/upper_body_
    disturbance_magnitude/joint_noise_frac`` from the Phase 1 final run.

    The performance-driven curricula remain active and will:
    - FREEZE automatically when new DR causes success rates to drop below gates
    - RESUME expansion once the policy adapts to the randomized physics
    """

    events: EventCfg_Phase2 = EventCfg_Phase2()
    curriculum: CurriculumCfg_Phase2 = CurriculumCfg_Phase2()

    def __post_init__(self) -> None:
        super().__post_init__()

        # ══════════════════════════════════════════════════════════════════════════
        # CURRICULUM STATE SNAPSHOT (from Phase 1 TensorBoard endpoint)
        # ──────────────────────────────────────────────────────────────────────────
        # TODO: 替换为 Phase 1 实际终态值（从 TensorBoard Curriculum/ 面板抄录）。
        # 以下使用课程 *_final 目标值作为占位；若 Phase 1 课程未推进到 final，
        # 需按实际值下调，否则策略会面临分布跳变。
        # ══════════════════════════════════════════════════════════════════════════

        # ── 指令采样范围：设为 Phase 1 课程实际达到的终态 ─────────────────────────
        self.commands.task_command.ranges.height_offset = (-0.50, 0.05)
        self.commands.task_command.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.task_command.ranges.lin_vel_y = (-0.2, 0.2)
        # ang_vel_z 保持不变（课程不扩展此轴）

        # ── 模式权重：SQUAT_WALK 已引入 ──────────────────────────────────────────
        self.commands.task_command.rel_mode_envs = (0.25, 0.25, 0.25, 0.25)

        # ── 上肢干扰幅度：Phase 1 课程爬升到的值 ─────────────────────────────────
        # 课程从 0.2 起爬，frac_final=0.8；Phase 1 实际达到多少取决于存活率门控。
        # TODO: 替换为 TensorBoard Curriculum/upper_body_disturbance_magnitude/joint_noise_frac 终值
        self.events.upper_body_disturbance.params["disturb_groups"][0]["joint_noise_frac"] = 0.5
        # 腰部组（group 1）课程不爬升，保持初始值 0.1


##
# Phase 3: External push disturbance
##


@configclass
class EventCfg_Phase3(EventCfg_Phase2):
    """Phase 3 event configuration: adds external push on top of Phase 2 static DR.

    Inherits all Phase 2 DR events (link_mass, base_com, payloads, actuator_gains,
    widened friction) and adds velocity-based push disturbance.
    """

    # ── New: external push (瞬时速度跳变, 模拟外部冲击) ──────────────────────────
    # interval 4~6 s, ±0.7 m/s in x/y. Strong disturbance -- deep squat postures have
    # small balance margins and are easily toppled. If too aggressive, reduce to ±0.5
    # or widen the interval to (6.0, 10.0).
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={"velocity_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7)}},
    )


@configclass
class G1EnvCfg_Phase3(G1EnvCfg_Phase2):
    """Phase 3: External push disturbance, resume from Phase 2 checkpoint.

    Inherits Phase 2's curriculum state snapshot (ranges, mode weights, disturbance
    magnitude) and adds push_robot. Before launching Phase 3, update the hard-coded
    values below to reflect Phase 2's actual endpoint (the performance-driven curricula
    may have further expanded ranges/magnitudes during Phase 2).

    Training progression:
        Phase 1 (baseline) → Phase 2 (static DR) → Phase 3 (push)

    Each phase resumes from the previous phase's checkpoint with matching task ID.
    """

    events: EventCfg_Phase3 = EventCfg_Phase3()

    def __post_init__(self) -> None:
        super().__post_init__()

        # ══════════════════════════════════════════════════════════════════════════
        # PHASE 2 → PHASE 3 CURRICULUM STATE UPDATE
        # ──────────────────────────────────────────────────────────────────────────
        # Phase 2 的绩效驱动课程可能在 DR 下继续推进了范围和幅度。
        # 续训前需从 Phase 2 TensorBoard 确认实际终态，覆盖 Phase 2 的快照值。
        # 若 Phase 2 课程因 DR 冻结（成功率低于门控），则保持 Phase 2 的初始值即可。
        #
        # TODO: 取消注释并填入 Phase 2 实际终态值（若有推进）
        # self.commands.task_command.ranges.height_offset = (-0.50, 0.05)
        # self.commands.task_command.ranges.lin_vel_x = (-0.5, 1.0)
        # self.commands.task_command.ranges.lin_vel_y = (-0.2, 0.2)
        # self.events.upper_body_disturbance.params["disturb_groups"][0]["joint_noise_frac"] = 0.6
        # ══════════════════════════════════════════════════════════════════════════
        pass


##
# Phase 2/3 PLAY configurations (for evaluation with DR disabled)
##


@configclass
class G1EnvCfg_Phase2_PLAY(G1EnvCfg_Phase2):
    """Phase 2 evaluation config: DR disabled, noise off, few envs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Disable all DR events for deterministic evaluation
        self.events.link_mass = None
        self.events.base_com = None
        self.events.hand_payload = None
        self.events.ankle_payload = None
        self.events.actuator_gains = None
        self.events.upper_body_disturbance = None
        # Restore narrow friction for flat-ground determinism
        self.events.physics_material = None


@configclass
class G1EnvCfg_Phase3_PLAY(G1EnvCfg_Phase3):
    """Phase 3 evaluation config: all DR + push disabled, noise off, few envs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Disable all DR + push events for deterministic evaluation
        self.events.link_mass = None
        self.events.base_com = None
        self.events.hand_payload = None
        self.events.ankle_payload = None
        self.events.actuator_gains = None
        self.events.upper_body_disturbance = None
        self.events.push_robot = None
        self.events.physics_material = None
