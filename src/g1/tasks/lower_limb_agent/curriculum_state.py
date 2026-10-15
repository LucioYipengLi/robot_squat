# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable curriculum checkpoints; simulator trajectories intentionally start fresh."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch

from .mdp.curriculums import curriculum_step

STATE_KEY = "g1_curriculum"
RANGE_FIELDS = ("height_offset", "lin_vel_x", "lin_vel_y", "ang_vel_z", "pelvis_offset_x", "pelvis_offset_y")
COMMAND_NAME = "task_command"
EVENT_NAME = "upper_body_disturbance"


def disturbance_layout(event):
    """Identify ordered disturbance groups before managers resolve their joint indices."""
    return [
        {"asset": group["asset_cfg"].name, "joints": group["asset_cfg"].joint_names}
        for group in event.params["disturb_groups"]
    ]


def _compatibility(env):
    event = env.cfg.events.upper_body_disturbance
    return {
        "robot_usd": env.cfg.scene.robot.spawn.usd_path,
        "action_joints": env.cfg.actions.joint_pos.joint_names,
        "action_scale": env.cfg.actions.joint_pos.scale,
        "policy_history": env.cfg.observations.policy.history_length,
        "disturbance_groups": disturbance_layout(event) if event is not None else env.cfg._g1_disturbance_layout,
    }


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its supported range")
    return value


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def validate_curriculum_state(env, state, *, evaluation=False):
    """Validate the entire record before changing any live environment state."""
    try:
        if not isinstance(state, Mapping):
            raise ValueError("Curriculum state must be a mapping")
        if state["schema_version"] != 1 or state["family"] != "g1.lower_limb_agent":
            raise ValueError("Unsupported curriculum schema or task family")
        compatibility = _compatibility(env)
        if state["compatibility"] != compatibility:
            raise ValueError("Checkpoint robot/action/history/disturbance layout differs from this environment")
        phase = _integer(state["phase"], "phase")
        target = env.cfg.training_phase
        allowed = (target,) if evaluation else (target, target - 1)
        if phase not in (1, 2, 3) or phase not in allowed:
            raise ValueError(f"Cannot restore Phase {phase} into Phase {target}; use the same or next training phase")
        dt = _number(state["control_dt"], "control_dt", minimum=0.0)
        duration = _number(state["episode_length_s"], "episode_length_s", minimum=0.0)
        if not math.isclose(dt, env.step_dt, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Checkpoint control dt differs from this environment")
        if not math.isclose(duration, env.cfg.episode_length_s, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Checkpoint episode duration differs from this environment")
        distribution = state["distribution"]
        if set(distribution["ranges"]) != set(RANGE_FIELDS):
            raise ValueError("Checkpoint command layout differs from the six-dimensional G1 task command")
        for name, bounds in distribution["ranges"].items():
            if len(bounds) != 2:
                raise ValueError(f"Invalid bounds for {name}")
            lo, hi = (_number(value, name) for value in bounds)
            if lo > hi:
                raise ValueError(f"Reversed bounds for {name}")
            if name == "height_offset" and lo >= 0:
                raise ValueError("Height-offset lower bound must be negative for the velocity envelope")
            if name.startswith("pelvis_offset_") and not lo <= 0 <= hi:
                raise ValueError("Pelvis-offset ranges must contain zero")
        weights = distribution["mode_weights"]
        if len(weights) != 4 or sum(_number(v, "mode weight", minimum=0) for v in weights) <= 0:
            raise ValueError("Expected four nonnegative mode weights with a positive sum")
        fractions = distribution["disturbance_fractions"]
        if not fractions:
            raise ValueError("Missing disturbance groups")
        for fraction in fractions:
            _number(fraction, "disturbance fraction", minimum=0, maximum=1)
        if len(fractions) != len(compatibility["disturbance_groups"]):
            raise ValueError("Checkpoint disturbance group count differs from the target configuration")
        steps = _integer(state["progress"]["control_steps"], "control_steps")
        if not isinstance(state["progress"]["mode_introduced"], bool):
            raise ValueError("mode_introduced must be boolean")
        if state["progress"]["mode_introduced"] != (weights[3] > 0):
            raise ValueError("Mode-introduction latch does not match the stored SQUAT_WALK weight")
        for key, elapsed in state["runtime"]["checks"].items():
            if key not in (f"command_range:{COMMAND_NAME}", f"survival:{EVENT_NAME}:0"):
                raise ValueError(f"Unknown curriculum clock {key!r}")
            if _integer(elapsed, key) > steps:
                raise ValueError("A curriculum clock precedes training step zero")
        survival = state["runtime"]["survival"]
        if survival is not None:
            if not isinstance(survival["initialized"], bool):
                raise ValueError("survival.initialized must be boolean")
            _number(survival["ema"], "survival EMA", minimum=0, maximum=1)
            count = _integer(survival["episode_count"], "episode_count")
            total = _integer(survival["total_episodes"], "total_episodes")
            _number(survival["length_sum"], "length_sum", minimum=0, maximum=count)
            if total < count:
                raise ValueError("Total episode count is smaller than the pending window")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed G1 curriculum state: {exc}") from exc


def export_curriculum_state(env):
    """Return independent CPU/Python values suitable for checkpoint infos or a legacy JSON snapshot."""
    term = env.command_manager.get_term(COMMAND_NAME)
    weights = term._mode_probs.detach().cpu().tolist()
    event = env.cfg.events.upper_body_disturbance
    if event is None:
        fractions = env._g1_restored_distribution["disturbance_fractions"]
    else:
        event = env.event_manager.get_term_cfg(EVENT_NAME)
        fractions = [float(group["joint_noise_frac"]) for group in event.params["disturb_groups"]]
    runtime = getattr(env, "_g1_curriculum_state", {})
    step = curriculum_step(env)
    survival = runtime.get(f"survival:{EVENT_NAME}:0")
    if survival is not None and "ema" not in survival:
        survival = None
    state = {
        "schema_version": 1,
        "family": "g1.lower_limb_agent",
        "compatibility": copy.deepcopy(_compatibility(env)),
        "phase": env.cfg.training_phase,
        "control_dt": env.step_dt,
        "episode_length_s": env.cfg.episode_length_s,
        "distribution": {
            "ranges": {name: list(getattr(term.cfg.ranges, name)) for name in RANGE_FIELDS},
            "mode_weights": weights,
            "disturbance_fractions": list(fractions),
        },
        "progress": {
            "control_steps": step,
            "mode_introduced": bool(runtime.get(f"mode_intro:{COMMAND_NAME}", {}).get("introduced", weights[3] > 0)),
        },
        "runtime": {
            "checks": {
                key: step - value["last_update_step"] for key, value in runtime.items() if "last_update_step" in value
            },
            "survival": None
            if survival is None
            else {
                "ema": float(survival["ema"]),
                "initialized": survival["initialized"],
                "length_sum": float(survival["length_sum"]),
                "episode_count": int(survival["episode_count"]),
                "total_episodes": int(survival["total_episodes"]),
            },
        },
    }
    validate_curriculum_state(env, state, evaluation=env.cfg.curriculum is None)
    return state


def import_curriculum_state(env, state, *, evaluation=False):
    """Restore distribution/progress, preserve same-phase summaries, and reset all trajectories.

    Across phases, monitoring and clocks are restarted but global curriculum progress and the
    mode-introduction latch continue. No per-environment trajectory buffers are transported.
    """
    validate_curriculum_state(env, state, evaluation=evaluation)
    state = copy.deepcopy(state)
    distribution = state["distribution"]
    term = env.command_manager.get_term(COMMAND_NAME)
    for ranges in (term.cfg.ranges, env.cfg.commands.task_command.ranges):
        for name, bounds in distribution["ranges"].items():
            setattr(ranges, name, tuple(bounds))
    weights = distribution["mode_weights"]
    term._mode_probs.copy_(torch.tensor(weights, device=term._mode_probs.device, dtype=term._mode_probs.dtype))
    term.cfg.rel_mode_envs = tuple(weights)
    env.cfg.commands.task_command.rel_mode_envs = tuple(weights)
    if env.cfg.events.upper_body_disturbance is not None:
        for event_cfg in (env.cfg.events.upper_body_disturbance, env.event_manager.get_term_cfg(EVENT_NAME)):
            for group, fraction in zip(
                event_cfg.params["disturb_groups"], distribution["disturbance_fractions"], strict=True
            ):
                group["joint_noise_frac"] = fraction
    env._g1_restored_distribution = distribution
    step = state["progress"]["control_steps"]
    env._g1_curriculum_step_offset = step - env.common_step_counter
    runtime = {f"mode_intro:{COMMAND_NAME}": {"introduced": state["progress"]["mode_introduced"]}}
    same_phase = state["phase"] == env.cfg.training_phase
    if same_phase and not evaluation:
        for key, elapsed in state["runtime"]["checks"].items():
            runtime.setdefault(key, {})["last_update_step"] = step - elapsed
        saved = state["runtime"]["survival"]
        if saved is not None:
            runtime.setdefault(f"survival:{EVENT_NAME}:0", {}).update(
                ema=torch.tensor(saved["ema"], device=env.device),
                initialized=saved["initialized"],
                length_sum=torch.tensor(saved["length_sum"], device=env.device),
                episode_count=torch.tensor(saved["episode_count"], device=env.device, dtype=torch.long),
                total_episodes=torch.tensor(saved["total_episodes"], device=env.device, dtype=torch.long),
                start_step=torch.full((env.num_envs,), env.common_step_counter, device=env.device, dtype=torch.long),
                observed=torch.ones(env.num_envs, device=env.device, dtype=torch.bool),
            )
    # Empty monitoring on phase transfer: wait for newly completed episodes under the new DR.
    if not same_phase and not evaluation:
        runtime[f"command_range:{COMMAND_NAME}"] = {"last_update_step": step}
        runtime[f"survival:{EVENT_NAME}:0"] = {"last_update_step": step}
    env._g1_curriculum_state = runtime
    for name in (
        "_error_height_sum",
        "_error_xy_sum",
        "_error_yaw_sum",
        "_step_count",
        "_pelvis_error_sum",
        "_pelvis_success_sum",
        "_pelvis_step_count",
        "episode_success_rate",
        "episode_success_rate_vel",
        "episode_metrics_valid",
    ):
        getattr(term, name).zero_()
    term._last_metric_step.fill_(env.common_step_counter)
    env.episode_length_buf.zero_()
    env._g1_restoring_curriculum = True
    try:
        env.reset()
    finally:
        env._g1_restoring_curriculum = False
    env._g1_curriculum_loaded = True
