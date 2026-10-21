"""Fixed task cases shared across checkpoints; no simulator import at module scope.

Only the command term's optional evaluation hooks use this controller. Normal
training keeps its original sampling and update lifecycle.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass

import torch

from .disturbance import joint_targets_from_uniform


def prepare_env_cfg(env_cfg, benchmark: dict, condition: dict, num_envs: int, seed: int):
    """Apply a benchmark before environment construction, independent of play hooks.

    Physical resets stay enabled and return authored nominal state. Every other
    event (including Phase 2/3 randomization) is disabled. Disturbance groups are
    retained separately for the deterministic controller to resolve after startup.
    """
    if benchmark["observation_noise"]:
        raise ValueError("Evaluation v1 requires observation_noise=false for replayable test cases")
    if benchmark["reset_mode"] != "nominal":
        raise ValueError("Evaluation v1 supports only nominal physical resets")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.episode_length_s = benchmark["episode_length_s"]
    actual_dt = env_cfg.sim.dt * env_cfg.decimation
    if not math.isclose(actual_dt, benchmark["control_dt"], rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Benchmark control_dt={benchmark['control_dt']} differs from task control_dt={actual_dt}")
    env_cfg.curriculum = None
    motion = benchmark["upper_body_motion"]
    for name, value in motion.items():
        setattr(env_cfg.upper_body_motion, name, value)
    env_cfg.observations.policy.upper_body_target.params["target_source"] = benchmark.get(
        "upper_body_target_source", "final_goal"
    )
    if env_cfg.terminations.time_out is None:
        raise ValueError("Evaluation requires the time_out termination")
    disturbance = env_cfg.events.upper_body_disturbance
    if disturbance is None:
        raise ValueError("Construct evaluation from the training task configuration, before play_mode()")
    from .curriculum_state import disturbance_layout

    env_cfg._g1_disturbance_layout = disturbance_layout(disturbance)
    env_cfg._g1_eval_disturbance_groups = copy.deepcopy(disturbance.params["disturb_groups"])
    for name, event in list(vars(env_cfg.events).items()):
        if event is not None and hasattr(event, "mode") and name not in ("reset_base", "reset_robot_joints"):
            setattr(env_cfg.events, name, None)
    env_cfg.events.reset_base.params["pose_range"] = {}
    env_cfg.events.reset_base.params["velocity_range"] = {}
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    for group in vars(env_cfg.observations).values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    command = env_cfg.commands.task_command
    command.debug_vis = False
    command.pelvis_debug_vis = False
    command.rel_mode_envs = tuple(benchmark["mode_weights"])
    # The deterministic controller owns the command clock. The base term remains
    # responsible for metric sampling, reset and one pelvis smoothing step.
    command.resampling_time_range = (1e9, 1e9)
    for name, bounds in benchmark["ranges"].items():
        setattr(command.ranges, name, tuple(bounds))
    threshold_fields = {
        "height_m": "height_success_threshold",
        "velocity_xy_mps": "vel_xy_success_threshold",
        "yaw_radps": "vel_yaw_success_threshold",
        "pelvis_xy_m": "pelvis_success_threshold",
    }
    for name, field in threshold_fields.items():
        setattr(command, field, benchmark["thresholds"][name])
    for name in ("depth_frac", "vx_floor", "wz_floor"):
        setattr(command, "envelope_" + name, benchmark["envelope"][name])
    command.pelvis_target_speed = benchmark["pelvis_target_speed_mps"]
    command.pelvis_zero_probability = benchmark["pelvis_zero_probability"]
    # Keep these in the resolved task configuration for provenance inspection.
    env_cfg._g1_eval_condition = copy.deepcopy(condition)
    return env_cfg


def _rng(seed: int, mode: int, episode_id: int, stream: str) -> random.Random:
    digest = hashlib.sha256(f"g1-eval-v1:{seed}:{mode}:{episode_id}:{stream}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _interval_steps(rng: random.Random, bounds, dt: float) -> int:
    return max(1, math.ceil(rng.uniform(*bounds) / dt))


@dataclass
class CasePlan:
    """Exogenous targets indexed by completed local control steps."""

    commands: dict[int, tuple[float, ...]]
    disturbances: dict[int, tuple[float, ...]]


def make_case_plan(benchmark: dict, seed: int, mode: int, episode_id: int, num_joints: int) -> CasePlan:
    """Generate one case without global RNG, checkpoint, batch or amplitude input.

    Matches training semantics: only SQUAT redraws height within an episode;
    WALK/SQUAT_WALK velocities and SQUAT_WALK height stay fixed. Static pelvis
    raw targets share the height resampling clock and are smoothed by the term.
    """
    if mode not in range(4):
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    dt = benchmark["control_dt"]
    horizon = math.ceil(benchmark["episode_length_s"] / dt)
    ranges = benchmark["ranges"]
    envelope = benchmark["envelope"]
    target_rng = _rng(seed, mode, episode_id, "targets")
    clock_rng = _rng(seed, mode, episode_id, "command-clock")
    event_rng = _rng(seed, mode, episode_id, "disturbance-targets")
    event_clock_rng = _rng(seed, mode, episode_id, "disturbance-clock")
    height = 0.0
    if mode == 1:
        height = target_rng.uniform(*ranges["height_offset"])
    elif mode == 3:
        height = target_rng.uniform(envelope["depth_frac"] * ranges["height_offset"][0], ranges["height_offset"][1])
    velocity = [0.0, 0.0, 0.0]
    if mode >= 2:
        depth = min(max(height / ranges["height_offset"][0], 0.0), 1.0)
        ramp = min(depth / envelope["depth_frac"], 1.0)
        for index, name, floor in ((0, "lin_vel_x", "vx_floor"), (2, "ang_vel_z", "wz_floor")):
            lo, hi = ranges[name]
            scale_floor = min(envelope[floor] / hi, 1.0) if hi > 0 else 1.0
            scale = 1 - (1 - scale_floor) * ramp if depth <= envelope["depth_frac"] else 0.0
            velocity[index] = target_rng.uniform(lo * scale, hi * scale)
        velocity[1] = target_rng.uniform(*ranges["lin_vel_y"])
    commands = {}
    step = 0
    while step < horizon:
        if mode == 1 and step > 0:
            height = target_rng.uniform(*ranges["height_offset"])
        pelvis = [0.0, 0.0]
        if mode < 2:
            pelvis = [target_rng.uniform(*ranges[name]) for name in ("pelvis_offset_x", "pelvis_offset_y")]
            if target_rng.random() < benchmark["pelvis_zero_probability"]:
                pelvis = [0.0, 0.0]
        commands[step] = (height, *velocity, *pelvis)
        step += _interval_steps(clock_rng, benchmark["resampling_time_range"], dt)
    disturbances = {}
    step = _interval_steps(event_clock_rng, benchmark["disturbance_interval_s"], dt)
    while step < horizon:
        disturbances[step] = tuple(event_rng.random() for _ in range(num_joints))
        step += _interval_steps(event_clock_rng, benchmark["disturbance_interval_s"], dt)
    return CasePlan(commands, disturbances)


class EvaluationScenarioController:
    """Replay fixed first-episode cases while other simulator slots may auto-reset."""

    def __init__(self, env, benchmark: dict, condition: dict, seed: int):
        self.env = env
        self.benchmark = benchmark
        self.condition = condition
        self.seed = seed
        self.term = env.command_manager.get_term("task_command")
        self.robot = env.scene[self.term.cfg.asset_name]
        self.active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.modes = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.plans: dict[int, CasePlan] = {}
        self.case_specs: list[dict] = []
        self._context = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float64)
        self._last_recorded = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self._pending_events = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._published_event_steps = [-1] * env.num_envs
        self._joint_ids = []
        self._fractions = []
        groups = env.cfg._g1_eval_disturbance_groups
        if len(groups) != 2:
            raise ValueError("Evaluation v1 expects the task's two upper-body groups (arms, waist)")
        for group, fraction in zip(groups, (condition["arm_fraction"], condition["waist_fraction"]), strict=True):
            entity = copy.deepcopy(group["asset_cfg"])
            entity.resolve(env.scene)
            joint_ids = entity.joint_ids
            if not isinstance(joint_ids, list) or not joint_ids:
                raise ValueError("Evaluation disturbance groups must resolve nonempty joint subsets")
            self._joint_ids.extend(joint_ids)
            self._fractions.extend([fraction] * len(joint_ids))
        if len(set(self._joint_ids)) != len(self._joint_ids):
            raise ValueError("Evaluation disturbance joint groups must be disjoint")
        self._frac = torch.tensor(self._fractions, device=env.device)
        self.term._evaluation_controller = self

    def start_batch(self, case_specs: list[dict]) -> None:
        """Assign slots before env.reset(); each slot contributes its first episode."""
        if len(case_specs) > self.env.num_envs:
            raise ValueError("Evaluation batch has more cases than simulator slots")
        self.case_specs = case_specs
        self.active.zero_()
        self.active[: len(case_specs)] = True
        self.modes.zero_()
        self.plans.clear()
        self._context.zero_()
        self._last_recorded.fill_(-1)
        self._pending_events.zero_()
        self._published_event_steps = [-1] * self.env.num_envs
        for slot, spec in enumerate(case_specs):
            self.modes[slot] = spec["mode"]
            self.plans[slot] = make_case_plan(
                self.benchmark, self.seed, spec["mode"], spec["episode_id"], len(self._joint_ids)
            )

    def assign_modes(self, env_ids: torch.Tensor) -> None:
        self.term.mode[env_ids] = self.modes[env_ids]

    def _apply_commands(self, slots: list[int], commands: list[tuple[float, ...]]) -> None:
        if not slots:
            return
        ids = torch.tensor(slots, device=self.env.device)
        values = torch.tensor(commands, device=self.env.device)
        term = self.term
        term.task_command[ids, :4] = values[:, :4]
        term.pelvis_target_offset[ids] = values[:, 4:]
        term.is_default_env[ids] = (self.modes[ids] == 0) | (self.modes[ids] == 2)
        term.height_command_world[ids, 0] = (
            self.env.scene.env_origins[ids, 2] + self.robot.data.default_root_pose.torch[ids, 2] + values[:, 0]
        )

    def resample_command(self, env_ids: torch.Tensor) -> None:
        """Reset commands before initial observations are constructed."""
        slots = env_ids.tolist()
        active = self.active.tolist()
        commands = [self.plans[slot].commands[0] if active[slot] else (0.0,) * 6 for slot in slots]
        self._apply_commands(slots, commands)

    def update_commands(self) -> None:
        """Publish task commands and final goals on the deterministic case clock.

        Goals queue in the shared executor, which advances during action application. Policy
        observations select final goals or PD references via target_source.
        """
        active_ids = self.active.nonzero().flatten()
        slots, commands = [], []
        local_steps = list(zip(active_ids.tolist(), self.env.episode_length_buf[active_ids].tolist(), strict=True))
        for slot, step in local_steps:
            command = self.plans[slot].commands.get(step)
            if command is not None:
                slots.append(slot)
                commands.append(command)
        self._apply_commands(slots, commands)
        self._publish_disturbances(local_steps)

    def before_step(self) -> None:
        """Count executed targets and consume event exposure once, without mutation."""
        active_ids = self.active.nonzero().flatten()
        steps = self.env.episode_length_buf[active_ids]
        if (self._last_recorded[active_ids] == steps).any():
            raise RuntimeError("before_step() must be called exactly once per executed control step")
        self._last_recorded[active_ids] = steps
        command = self.term.task_command[active_ids]
        self._context[active_ids, 1] += command[:, 0].double()
        self._context[active_ids, 2] += torch.linalg.vector_norm(command[:, 1:3], dim=-1).double()
        self._context[active_ids, 3] += command[:, 3].abs().double()
        self._context[active_ids, 0] += self._pending_events[active_ids].double()
        self._pending_events[active_ids] = 0

    def _publish_disturbances(self, local_steps: list[tuple[int, int]]) -> None:
        event_slots, event_samples = [], []
        for slot, step in local_steps:
            normalized = self.plans[slot].disturbances.get(step)
            if normalized is None or self._published_event_steps[slot] == step:
                continue
            event_slots.append(slot)
            event_samples.append(normalized)
            self._published_event_steps[slot] = step
        if not event_slots:
            return
        ids = torch.tensor(event_slots, device=self.env.device)
        default_q = self.robot.data.default_joint_pos.torch[ids][:, self._joint_ids]
        limits = self.robot.data.joint_pos_limits.torch[ids][:, self._joint_ids]
        u = torch.tensor(event_samples, device=self.env.device)
        target = joint_targets_from_uniform(default_q, limits, self._frac, u)
        self.env.upper_body_motion.set_goal(ids, target, self._joint_ids)
        self._pending_events[ids] += 1

    def complete(self, env_ids: torch.Tensor) -> None:
        self.active[env_ids] = False

    def context(self, env_id: int) -> dict:
        values = self._context[env_id].tolist()
        result = dict(
            zip(
                ("disturbance_count", "command_height_sum", "command_speed_sum", "command_yaw_abs_sum"),
                values,
                strict=True,
            )
        )
        motion = getattr(self.env, "upper_body_motion", None)
        if motion is not None:
            result.update(
                zip(
                    (
                        "motion_started",
                        "motion_completed",
                        "motion_pending_replaced",
                        "motion_duration_sum_s",
                        "motion_time_s",
                    ),
                    motion.trajectory.statistics[env_id].tolist(),
                    strict=True,
                )
            )
        return result
