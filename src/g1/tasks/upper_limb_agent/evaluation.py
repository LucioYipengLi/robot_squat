"""Upper Mobile fixed-target scenarios and pre-reset measurement."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, yaw_quat

from g1.rl.evaluation.jobs import digest_file
from g1.rl.evaluation.policy import build_actor, policy_action, verify_checkpoint_files
from g1.rl.evaluation.upper_fixed_targets import ARM_MODES, resolve_upper_height_range
from g1.rl.evaluation.upper_q import PROTOCOL as Q_PROTOCOL
from g1.rl.evaluation.upper_q import SIDES, quaternion_xyzw

from .mdp.pose_override import PoseOverride
from .mdp.workspace_quality import WorkspaceQualityCfg
from .q_eval_recording import QWindows
from .runner import UpperLimbOnPolicyRunner
from .symmetry import FULL_OBSERVATION_DIM


class UpperEvalRecorder(RecorderTerm):
    def record_post_step(self):
        controller = getattr(self._env, "upper_eval_controller", None)
        if controller is not None:
            controller.capture()
        return None, None


@configclass
class UpperEvalRecordersCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    upper_eval = RecorderTermCfg(class_type=UpperEvalRecorder)


@dataclass(frozen=True)
class UpperEvalSetup:
    lower_sha256: str
    height_range: tuple[float, float] | None = None
    map_sha256: str | None = None


def _select_lower(checkpoint, lower, default_path):
    if override := checkpoint.get("frozen_lower_override"):
        return Path(override["path"]), override["sha256"]
    archive = Path(checkpoint["path"]).parent / "artifacts/frozen_lower_model.pt"
    default = Path(default_path).expanduser()
    original = Path(lower["checkpoint"]).expanduser() if lower.get("checkpoint") else default
    lower_path = next(
        (
            path.resolve()
            for path in (archive, original, default)
            if path.is_file() and digest_file(path) == lower["sha256"]
        ),
        None,
    )
    if lower_path is None:
        raise ValueError(
            f"Matching frozen Lower model not found for {checkpoint['path']}; expected archive at {archive}"
        )
    return lower_path, lower["sha256"]


def _lower_training_height(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = (saved.get("infos") or {}).get("g1_curriculum") or {}
    if state.get("family") != "g1.lower_limb_agent" or "actor_state_dict" not in saved:
        raise ValueError(f"Expected a frozen Lower actor checkpoint with curriculum metadata: {path}")
    height = state.get("distribution", {}).get("ranges", {}).get("height_offset")
    if height is None:
        raise ValueError(f"Frozen Lower checkpoint does not record its training height range: {path}")
    return height


def prepare_env_cfg(cfg, benchmark, checkpoint, num_envs, seed):
    verify_checkpoint_files(checkpoint)
    saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    infos = saved.get("infos") or {}
    interface = infos.get("g1_control_interface") or {}
    if interface.get("schema_version") != 4 or interface.get("policy_observation_dim") != FULL_OBSERVATION_DIM:
        raise ValueError(f"Upper eval requires a schema 4 Mobile checkpoint with {FULL_OBSERVATION_DIM} observations")
    lower = infos.get("g1_frozen_lower") or {}
    if not lower.get("sha256") or not lower.get("interface"):
        raise ValueError("Upper checkpoint must identify its frozen Lower actor and interface")
    lower_path, lower_sha256 = _select_lower(checkpoint, lower, cfg.actions.joint_pos.checkpoint_path)
    curriculum = cfg.commands.wrist_pose.range_curriculum
    height_range = None
    requested = checkpoint.get("upper_height_range")
    if checkpoint.get("frozen_lower_override") or requested is not None:
        original = cfg.actions.height_offset.height_offset_range
        if list(original) != interface.get("height_offset_range"):
            raise ValueError("Upper checkpoint control interface differs before adaptation: height_offset_range")
        height_range = resolve_upper_height_range(original, _lower_training_height(lower_path), requested)
        cfg.actions.height_offset.height_offset_range = height_range
        initial_lo, initial_hi = curriculum.initial_lower_ranges[0]
        # Preserve curriculum interpolation even when the new Lower has a smaller range than Level 0.
        curriculum.initial_lower_ranges = (
            (max(initial_lo, height_range[0]), min(initial_hi, height_range[1])),
            *curriculum.initial_lower_ranges[1:],
        )

    cfg.actions.joint_pos.checkpoint_path = str(lower_path)
    cfg.legacy_observations = False
    cfg.domain_randomization.phase = 1
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = max(cfg.episode_length_s, benchmark["episode_length_s"])
    cfg.recorders = UpperEvalRecordersCfg()
    wrist = cfg.commands.wrist_pose
    map_sha256 = None
    if benchmark.get("protocol") == Q_PROTOCOL:
        wrist.workspace_quality = WorkspaceQualityCfg()
        map_path = Path(__file__).resolve().parent / wrist.workspace_quality.map_path
        map_sha256 = benchmark["map_sha256"]
        if digest_file(map_path) != map_sha256:
            raise ValueError(f"Workspace Q map hash differs from the suite: {map_path}")
    wrist.debug_vis = False
    wrist.planar_turn_focus.enabled = False
    curriculum.initial_level = benchmark["level"]
    curriculum.advance = False
    curriculum.freeze_planar_sampling = True
    dt = cfg.sim.dt * cfg.decimation
    if not math.isclose(dt, benchmark["control_dt"], rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Upper suite requires control dt {benchmark['control_dt']}, got {dt}")
    return saved, UpperEvalSetup(lower_sha256, height_range, map_sha256)


def load_policy(env, agent_cfg, checkpoint, saved, setup, *, device):
    raw = env.unwrapped
    infos = saved["infos"]
    expected = infos["g1_control_interface"]
    if setup.height_range is not None:
        expected = {**expected, "height_offset_range": list(setup.height_range)}
    actual = UpperLimbOnPolicyRunner.mobile_control_interface(raw)
    if actual is None:
        raise ValueError("Upper evaluation requires the Mobile control interface")
    # The reward-only magnitude normalization can differ in a checkpoint.
    differing = [
        key for key, value in actual.items() if key != "velocity_magnitude_normalization" and expected.get(key) != value
    ]
    if differing:
        raise ValueError(f"Upper checkpoint control interface differs: {', '.join(differing)}")
    lower = raw.action_manager.get_term("joint_pos").frozen_policy
    expected_lower = infos["g1_frozen_lower"]
    if lower.metadata["sha256"] != setup.lower_sha256 or lower.interface != expected_lower["interface"]:
        raise ValueError("Frozen Lower model or interface differs from the Upper checkpoint")
    command = raw.command_manager.get_term("wrist_pose")
    curriculum = command.range_curriculum
    q_metadata = {}
    if setup.map_sha256 is not None:
        quality = command.workspace_quality_map
        if quality is None or quality.sha256 != setup.map_sha256:
            raise ValueError("Loaded workspace Q map differs from the suite")
        q_metadata = {
            "workspace_q_map_sha256": quality.sha256,
            "workspace_q_reference": quality.q_reference,
            "training_q_map_sha256": (infos.get("g1_workspace_quality") or {}).get("map_sha256"),
        }
    actor, cfg = build_actor(env, agent_cfg, checkpoint, saved, device=device)
    return (
        actor,
        cfg,
        {
            **q_metadata,
            "compatibility_source": "g1_control_interface+g1_frozen_lower",
            "training_iteration": saved.get("iter"),
            "training_level": (infos.get("g1_upper_ranges") or {}).get("level"),
            "training_phase": ((infos.get("g1_upper_domain_randomization") or {}).get("settings") or {}).get("phase"),
            "frozen_lower": lower.metadata,
            "frozen_lower_overridden": bool(checkpoint.get("frozen_lower_override")),
            "upper_height_range_m": actual["height_offset_range"],
            "episode_height_range_m": curriculum.lower_ranges[0, 0].tolist(),
            "actor_class": str(cfg["actor"]["class_name"]),
            "agent_config_source": checkpoint.get("agent_config") or "registered_task_default",
        },
    )


class UpperFixedTargets(PoseOverride):
    def __init__(self, env):
        super().__init__("auto", "regular", 0)
        self.env = env
        self.term = env.command_manager.get_term("wrist_pose")
        self.term.range_curriculum.evaluation = True
        ids, _ = self.term.robot.find_bodies("torso_link")
        if len(ids) != 1:
            raise ValueError("Upper eval requires one torso_link")
        self.torso_id = ids[0]
        options = {"device": env.device}
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, **options)
        self.modes = torch.zeros(env.num_envs, dtype=torch.long, **options)
        self.offsets = torch.zeros((env.num_envs, 3), **options)
        self.starts = torch.zeros_like(self.modes)
        self.deadlines = torch.zeros_like(self.modes)
        self.tracking_steps = torch.zeros_like(self.modes)
        self.window_steps = torch.zeros_like(self.modes)
        self.tilt_sum = torch.zeros((env.num_envs, 2), **options)
        self.error_sum = torch.zeros((env.num_envs, 2), **options)
        self.records = []
        self.batch_records = []
        self.specs = []

    def start_batch(self, specs):
        self.specs = specs
        self.batch_records = []
        self.active.zero_()
        selected = slice(0, len(specs))
        self.active[selected] = True
        self.modes[selected] = self.modes.new_tensor([ARM_MODES.index(spec["arm_mode"]) for spec in specs])
        self._load_targets(selected, specs)
        self.starts[selected] = self.starts.new_tensor([spec["evaluation_start_step"] for spec in specs])
        self.deadlines[selected] = self.deadlines.new_tensor([spec["deadline_step"] for spec in specs])
        for buffer in (self.tracking_steps, self.window_steps, self.tilt_sum, self.error_sum):
            buffer.zero_()

    def _load_targets(self, selected, specs):
        self.offsets[selected] = self.offsets.new_tensor([spec["offset_m"] for spec in specs])

    def prepare_reset(self, term, ids):
        super().prepare_reset(term, ids)
        selected = ids[self.active[ids]]
        term.mode_id[selected] = self.modes[selected]
        term.evaluation_start_steps[selected] = self.starts[selected]
        term.episode_deadline_steps[selected] = self.deadlines[selected]

    def resample(self, term, ids):
        fresh = ids[term.command_counter[ids] == 0]
        term.goal_pose_w[fresh] = term.default_wrist_pose_e[fresh]
        term.goal_pose_w[fresh, :, :3] += self.env.scene.env_origins[fresh, None, :]
        selected = fresh[self.active[fresh]]
        self._set_targets(term, selected)
        term.command_counter[fresh] += 1
        term.time_left[ids] = torch.inf

    def _set_targets(self, term, selected):
        world_offset = quat_apply(yaw_quat(term._sampling_quat_w[selected]), self.offsets[selected])
        term.goal_pose_w[selected, :, :3] += world_offset[:, None] * term.active_mask[selected, :, None]

    def capture(self):
        """Sample the physical step once, before any automatic reset."""
        env, term = self.env, self.term
        age = term.episode_age_steps()
        active = self.active & (age > 0)
        # Post-step ages are one-based: (start, deadline] has exactly the planned sample count.
        window = active & (age > self.starts) & (age <= self.deadlines)
        roll, pitch, _ = euler_xyz_from_quat(term.robot.data.body_link_pose_w.torch[:, self.torso_id, 3:])
        angles = torch.stack((roll, pitch), dim=-1)
        # Euler angles can be returned modulo 2 pi. Map to signed principal angles first.
        abs_tilt_deg = torch.rad2deg(torch.atan2(angles.sin(), angles.cos()).abs())
        position, orientation = term.pose_error()
        self.tracking_steps += active
        self.window_steps += window
        self.tilt_sum += torch.where(active[:, None], abs_tilt_deg, 0)
        self.error_sum += torch.where(window[:, None] & term.active_mask, position, 0)
        self._capture_step(active, window, age, position, orientation)
        ended = (active & (env.reset_terminated | env.reset_time_outs)).nonzero().flatten()
        if len(ended):
            self._finish(ended)

    def _capture_step(self, active, window, age, position, orientation):
        pass

    def _extra_records(self, ids):
        return [{} for _ in range(len(ids))]

    def _finish(self, ids):
        env, term = self.env, self.term
        goals = term.goal_pose_w[ids].clone()
        goals[:, :, :3] -= env.scene.env_origins[ids, None, :]
        steps = self.tracking_steps[ids].tolist()
        window_steps = self.window_steps[ids].tolist()
        tilt = (self.tilt_sum[ids] / self.tracking_steps[ids, None]).tolist()
        error = (1000 * self.error_sum[ids] / self.window_steps[ids, None].clamp_min(1)).tolist()
        failed = env.reset_terminated[ids].tolist()
        timed_out = env.reset_time_outs[ids].tolist()
        reasons = {
            name: env.termination_manager.get_term(name)[ids].tolist() for name in env.termination_manager.active_terms
        }
        masks = term.active_mask[ids].tolist()
        extras = self._extra_records(ids)
        for row, (slot, goal, mask) in enumerate(zip(ids.tolist(), goals.tolist(), masks, strict=True)):
            spec = self.specs[slot]
            arms = [
                value if armed and window_steps[row] else None for value, armed in zip(error[row], mask, strict=True)
            ]
            roll, pitch = tilt[row]
            expected_window_steps = spec["deadline_step"] - spec["evaluation_start_step"]
            self.batch_records.append(
                {
                    **spec,
                    **extras[row],
                    "goal_pose_env_xyzw": [value if armed else None for value, armed in zip(goal, mask, strict=True)],
                    "steps": steps[row],
                    "elapsed_s": steps[row] * env.step_dt,
                    "failed": failed[row],
                    "timed_out": timed_out[row] and not failed[row],
                    "reasons": {name: values[row] for name, values in reasons.items()},
                    "roll_abs_mean_deg": roll,
                    "pitch_abs_mean_deg": pitch,
                    "torso_angle_mean_deg": (roll + pitch) / 2,
                    "window_steps": window_steps[row],
                    "expected_window_steps": expected_window_steps,
                    "window_complete": window_steps[row] == expected_window_steps,
                    "wrist_left_error_mm": arms[0],
                    "wrist_right_error_mm": arms[1],
                    "wrist_error_mm": sum(value for value in arms if value is not None) / sum(mask)
                    if window_steps[row]
                    else None,
                }
            )
        self.active[ids] = False


class UpperQTargets(UpperFixedTargets):
    def __init__(self, env, steady_speed_mps, replay):
        super().__init__(env)
        self.q_windows = QWindows(env, steady_speed_mps)
        self.motion_replay = replay
        self.fixed_pose = None

    def _load_targets(self, selected, specs):
        if "wrist_poses" not in specs[0]:
            self.fixed_pose = None
            super()._load_targets(selected, specs)
            return
        poses = []
        for spec in specs:
            wrist = spec["wrist_poses"]
            poses.append([(*wrist[side]["position_m"], *quaternion_xyzw(wrist[side]["rpy_deg"])) for side in SIDES])
        self.fixed_pose = self.term.goal_pose_w.new_zeros(self.term.goal_pose_w.shape)
        self.fixed_pose[selected] = self.fixed_pose.new_tensor(poses)

    def _set_targets(self, term, selected):
        if self.fixed_pose is None:
            super()._set_targets(term, selected)
        else:
            term.goal_pose_w[selected] = self.fixed_pose[selected]
            term.goal_pose_w[selected, :, :3] += self.env.scene.env_origins[selected, None, :]

    def start_batch(self, specs):
        super().start_batch(specs)
        self.q_windows.reset(max(spec["deadline_step"] for spec in specs))

    def _capture_step(self, active, window, age, position, orientation):
        if self.motion_replay is not None:
            self.motion_replay.capture()
        term = self.term
        speed_square = term.robot.data.body_link_vel_w.torch[:, term.body_indices, :3].square().sum(-1)
        self.q_windows.add(active, window, age, position, orientation, speed_square, term.workspace_quality_state())

    def _extra_records(self, ids):
        slots = ids.tolist()
        if self.motion_replay is not None and 0 in slots:
            self.motion_replay.done = True
        return self.q_windows.records(
            ids, [self.specs[slot] for slot in slots], self.term.workspace_quality_map.q_reference
        )


def run_rollouts(env, policy, specs, *, seed, policy_output, progress, steady_speed_mps=None, replay=None):
    raw = env.unwrapped
    controller = UpperFixedTargets(raw) if steady_speed_mps is None else UpperQTargets(raw, steady_speed_mps, replay)
    term = controller.term
    previous = term._teleop
    term._teleop = controller
    raw.upper_eval_controller = controller
    batches = math.ceil(len(specs) / env.num_envs)
    try:
        with torch.inference_mode():
            for index, offset in enumerate(range(0, len(specs), env.num_envs)):
                batch = specs[offset : offset + env.num_envs]
                controller.start_batch(batch)
                obs, _ = env.reset()
                if replay is not None:
                    replay.capture()
                policy.reset()
                horizon = max(spec["deadline_step"] for spec in batch)
                for step in range(1, horizon + 1):
                    action_seed = int.from_bytes(
                        hashlib.sha256(f"{seed}:{index}:{step}:policy".encode()).digest()[:8], "little"
                    ) % (2**63)
                    actions = policy_action(policy, obs, stochastic=policy_output == "sample", seed=action_seed)
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Policy produced non-finite actions")
                    obs, _, dones, _ = env.step(actions)
                    policy.reset(dones)
                    completed = len(controller.batch_records)
                    progress.update(
                        "evaluating",
                        episodes_completed=len(controller.records) + completed,
                        episodes_total=len(specs),
                        batches_completed=index + int(completed == len(batch)),
                        batches_total=batches,
                        batch_index=index + 1,
                        batch_episodes_completed=completed,
                        batch_episodes_total=len(batch),
                        batch_step=step,
                        batch_steps_total=horizon,
                    )
                    if completed == len(batch):
                        break
                if len(controller.batch_records) != len(batch):
                    raise RuntimeError("Upper evaluation horizon ended before every assigned case terminated")
                controller.records.extend(controller.batch_records)
        return sorted(controller.records, key=lambda record: record["case_id"])
    finally:
        raw.upper_eval_controller = None
        term._teleop = previous
