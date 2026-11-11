# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Performance-driven curriculum terms for the G1 squat/walk task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _curriculum_state(env: ManagerBasedRLEnv, key: str) -> dict:
    """Keep runtime state on its owning environment for the project checkpoint adapter."""
    if not hasattr(env, "_g1_curriculum_state"):
        env._g1_curriculum_state = {}
    return env._g1_curriculum_state.setdefault(key, {})


def curriculum_step(env: ManagerBasedRLEnv) -> int:
    """Training progress without changing the simulator's local step counters."""
    return env.common_step_counter + getattr(env, "_g1_curriculum_step_offset", 0)


def _update_due(env: ManagerBasedRLEnv, state: dict, update_period: int) -> bool:
    """Check on the first reset after the interval; never replay missed intervals in a burst."""
    if update_period <= 0:
        raise ValueError("update_period must be positive")
    if getattr(env, "_g1_restoring_curriculum", False):
        return False
    step = curriculum_step(env)
    last_update = state.setdefault("last_update_step", 0)
    if step < last_update:
        raise ValueError("Curriculum step counter moved backwards; reset or restore its runtime state")
    if step - last_update < update_period:
        return False
    state["last_update_step"] = step
    return True


def command_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    command_name: str = "task_command",
    height_success_gate: float = 0.95,
    height_offset_final: tuple[float, float] = (-0.45, 0.05),
    height_step: float = 0.005,
    vel_success_gate: float = 0.95,
    lin_vel_x_final: tuple[float, float] = (-0.5, 1.0),
    lin_vel_y_final: tuple[float, float] = (-0.2, 0.2),
    vel_step: float = 0.01,
    update_period: int = 200,
) -> dict[str, float]:
    """Widen the task-command sampling ranges once the agent reliably tracks the current ones.

    A performance-driven (not time-driven) curriculum for the unified :class:`SquatWalkCommand`. Two
    axes advance independently, each gated on a per-episode success metric that the command term
    finalizes at reset:

    * **Height axis** -- gated on the *all-env* height ``success_rate`` (per-episode mean pelvis-height
      tracking error below ``height_success_threshold``). On success the ``height_offset`` band creeps
      toward ``height_offset_final`` (a deeper squat band), so the robot is asked to squat lower only
      once it tracks the current depth band.
    * **Velocity axis** -- gated on the ``success_rate_vel`` of **walking-mode episodes** (masked via
      the command term's ``last_mode``). On success the ``lin_vel_x`` / ``lin_vel_y`` bands creep toward
      their finals (faster forward / wider lateral walking). ``ang_vel_z`` is intentionally left fixed.

    Expansion is **monotonic** (each bound only moves toward the final band, never back) and
    **self-pacing**: widening the band makes tracking harder, which pulls the gated success metric back
    below its threshold and pauses expansion until the agent catches up. The final band is therefore an
    *upper bound* -- it is reached only if the agent can still track it above the gate.

    Note:
        * Gates read the command term's persisted tracking flags, produced by the shared
          performance monitor's episode reduction. Nonempty-episode eligibility remains
          unchanged. This curriculum runs before command reset finalizes the current batch,
          so it sees previously finalized results; the velocity mask uses their last_mode.
        * An update is checked on the first reset at least ``update_period`` control steps after the
          previous check. Duplicate calls in one step cannot update twice; missed intervals do not
          cause a catch-up burst. Each increment therefore has time for the gate to react.

    Args:
        env: The environment instance.
        env_ids: The environment indices being reset (unused; the curriculum is global).
        command_name: Name of the :class:`SquatWalkCommand` term in the command manager.
        height_success_gate: Height success over valid finalized episodes above which the band expands.
        height_offset_final: Target ``(low, high)`` pelvis-height-offset band [m].
        height_step: Per-update increment of each height-offset bound [m].
        vel_success_gate: WALK-mode ``success_rate_vel`` above which the velocity bands expand.
        lin_vel_x_final: Target ``(low, high)`` base-frame x-velocity band [m/s].
        lin_vel_y_final: Target ``(low, high)`` base-frame y-velocity band [m/s].
        vel_step: Per-update increment of each linear-velocity bound [m/s].
        update_period: Number of control steps between range updates.

    Returns:
        Curriculum state logged under ``Curriculum/<term_name>/<key>``: the two gate signals plus each
        signal. Live sampling bounds are recorded once under TaskDistribution.
    """
    term = env.command_manager.get_term(command_name)
    ranges = term.cfg.ranges

    # Persisted flags survive episode reset; logging windows remain independent.
    valid = term.episode_metrics_valid.to(term.episode_success_rate.dtype)
    height_success = (term.episode_success_rate * valid).sum() / valid.sum().clamp_min(1.0)
    # WALK and SQUAT_WALK episodes; static modes cannot dilute the velocity gate.
    # A masked mean via sum/clamp avoids both a sync and the NaN of an empty selection.
    walk_mask = (term.last_mode == term.MODE_WALK) | (term.last_mode == term.MODE_SQUAT_WALK)
    walk_mask = walk_mask.to(height_success.dtype) * valid
    vel_success = (term.episode_success_rate_vel * walk_mask).sum() / walk_mask.sum().clamp_min(1.0)

    # -- apply the expansion at most once per ``update_period`` steps
    state = _curriculum_state(env, f"command_range:{command_name}")
    if _update_due(env, state, update_period):
        # height axis: deepen the squat band while the all-env height success holds above the gate
        if bool(height_success > height_success_gate):
            lo, hi = ranges.height_offset
            flo, fhi = height_offset_final
            ranges.height_offset = (max(lo - height_step, flo), min(hi + height_step, fhi))
        # velocity axis: widen the walk band while the WALK-mode velocity success holds above the gate
        if bool(vel_success > vel_success_gate):
            lo, hi = ranges.lin_vel_x
            flo, fhi = lin_vel_x_final
            ranges.lin_vel_x = (max(lo - vel_step, flo), min(hi + vel_step, fhi))
            lo, hi = ranges.lin_vel_y
            flo, fhi = lin_vel_y_final
            ranges.lin_vel_y = (max(lo - vel_step, flo), min(hi + vel_step, fhi))

    return {"height_success": height_success, "vel_success": vel_success}


def _update_survival_ema(env, env_ids, state, *, ema_beta, update_period) -> bool:
    """Consume real completed episodes once; return whether a nonempty window was reduced.

    Callers share this implementation while owning independent windows, EMA values and
    clocks, so tuning the push curriculum cannot change the upper-body curriculum.
    """
    if not 0.0 < ema_beta <= 1.0:
        raise ValueError("ema_beta must be in (0, 1]")
    if "ema" not in state:
        state.update(
            ema=torch.zeros((), device=env.device),
            initialized=False,
            length_sum=torch.zeros((), device=env.device),
            episode_count=torch.zeros((), dtype=torch.long, device=env.device),
            total_episodes=torch.zeros((), dtype=torch.long, device=env.device),
            start_step=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            observed=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
    episode_lengths = env.episode_length_buf[env_ids]
    if episode_lengths.numel() > 0:
        if not getattr(env, "_g1_restoring_curriculum", False):
            elapsed = env.common_step_counter - state["start_step"][env_ids]
            valid = (
                state["observed"][env_ids]
                & env.termination_manager.dones[env_ids]
                & (elapsed > 0)
                & (episode_lengths == elapsed)
            )
            finished = episode_lengths.to(torch.float32) / float(env.max_episode_length)
            state["length_sum"].add_(torch.where(valid, finished, 0.0).sum())
            count = valid.sum()
            state["episode_count"].add_(count)
            state["total_episodes"].add_(count)
        state["start_step"][env_ids] = env.common_step_counter
        state["observed"][env_ids] = True

    if not _update_due(env, state, update_period) or not bool(state["episode_count"] > 0):
        return False
    episode_mean = state["length_sum"] / state["episode_count"]
    if state["initialized"]:
        state["ema"].lerp_(episode_mean, ema_beta)
    else:
        state["ema"].copy_(episode_mean)
        state["initialized"] = True
    state["length_sum"].zero_()
    state["episode_count"].zero_()
    return True


def upper_body_disturbance_magnitude_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    event_term_name: str = "upper_body_disturbance",
    target_group: int = 0,
    survival_gate: float = 0.85,
    frac_final: float = 0.5,
    frac_step: float = 0.005,
    ema_beta: float = 0.02,
    update_period: int = 2000,
    command_name: str = "task_command",
    task_success_gate: float = 0.8,
) -> dict[str, float]:
    """Ramp one group's amplitude when survival EMA and global task success pass.

    Survival keeps its episode-weighted, normalized-duration EMA: fully observed,
    naturally ended episodes enter an update window once. Initial, manual and
    duplicate resets and artificially aged first episodes are excluded.

    Task success reuses the monitor's joint-success definition: normal timeout,
    height/velocity/yaw tracking and applicable pelvis tracking must all pass.
    Each environment contributes its latest valid finalized episode, independent
    of log drains. Before any valid result, task success is zero. The curriculum
    runs before command reset finalizes the current batch, so it reads previously
    finalized results. Checkpoint restoration clears these snapshots for recollection.

    Both signals must strictly exceed their gates. Only the selected group's
    live ``joint_noise_frac`` changes, by at most one increment per update period;
    low quality pauses the ramp. Survival EMA updates continue while quality is
    low. Windows without completed survival episodes do not update or ramp.

    Args:
        env: The environment instance.
        env_ids: The environment indices or slice being reset.
        event_term_name: Name of the :class:`UpperBodyDisturbanceEvent` term in the event manager.
        target_group: Index into ``params["disturb_groups"]`` selecting which group's ``joint_noise_frac``
            to ramp (default 0 = the arm group; the waist group is left fixed at its mounted value).
        survival_gate: Mean normalized episode length (in [0, 1]) above which ``joint_noise_frac`` ramps.
        frac_final: Target magnitude as a fraction [0, 1] of the default-to-limit margin; the ramp's
            upper bound.
        frac_step: Per-update increment of ``joint_noise_frac`` [0, 1].
        ema_beta: EMA factor per nonempty update window (smaller = slower/steadier gate).
        update_period: Number of control steps between magnitude updates (same units as
            :func:`command_range_curriculum`, which is mounted at 5000).
        command_name: Command term owning the performance monitor.
        task_success_gate: Global latest-episode joint success threshold in [0, 1].

    Returns:
        Curriculum state logged under ``Curriculum/<term_name>/<key>``: the survival EMA gate signal
        (``survival_ema``), validity flag, completed-episode count, ``task_success_rate``
        signals. Amplitude is recorded once under TaskDistribution.
    """
    # live reference to the event term cfg (get_term_cfg returns the stored object, not a copy);
    # the target group's joint_noise_frac is a plain Python float re-read by the event on every fire.
    event_cfg = env.event_manager.get_term_cfg(event_term_name)
    group = event_cfg.params["disturb_groups"][target_group]
    frac = float(group["joint_noise_frac"])

    if isinstance(task_success_gate, bool) or not 0.0 <= task_success_gate <= 1.0:
        raise ValueError("task_success_gate must be in [0, 1]")
    task_success = env.command_manager.get_term(command_name).performance.task_success_rate()
    state = _curriculum_state(env, f"survival:{event_term_name}:{target_group}")
    if (
        _update_survival_ema(env, env_ids, state, ema_beta=ema_beta, update_period=update_period)
        and bool((state["ema"] > survival_gate) & (task_success > task_success_gate))
        and frac < frac_final
    ):
        frac = min(frac + frac_step, frac_final)
        group["joint_noise_frac"] = frac

    return {
        "survival_ema": state["ema"],
        "survival_valid": float(state["initialized"]),
        "survival_episodes": state["total_episodes"],
        "task_success_rate": task_success,
    }


def _horizontal_push_magnitude(ranges) -> float:
    """Validate the curriculum's equal, symmetric world X/Y increment bounds."""
    if set(ranges) != {"x", "y"}:
        raise ValueError("Push curriculum requires velocity_range with exactly x and y")
    magnitudes = []
    for bounds in ranges.values():
        if len(bounds) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in bounds
        ):
            raise ValueError("Push bounds must contain two finite numbers")
        lo, hi = bounds
        if hi < 0 or not math.isclose(-lo, hi, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Push bounds must be symmetric about zero")
        magnitudes.append(float(hi))
    if not math.isclose(*magnitudes, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Push curriculum requires equal x/y magnitudes")
    return magnitudes[0]


def push_velocity_magnitude_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    event_term_name: str = "push_robot",
    command_name: str = "task_command",
    survival_gate: float = 0.95,
    task_success_gate: float = 0.8,
    velocity_final: float = 0.7,
    velocity_step: float = 0.05,
    ema_beta: float = 0.02,
    update_period: int = 5000,
) -> dict[str, float]:
    """Ramp the per-axis horizontal velocity increment after both quality gates pass.

    Initial bounds come from the event configuration. Each full, nonempty check
    window permits at most one increase; low quality holds the current bounds.
    Uses the same episode definitions as the arm curriculum, with its own EMA.
    """
    for name, gate in (("survival_gate", survival_gate), ("task_success_gate", task_success_gate)):
        if isinstance(gate, bool) or not 0.0 <= gate <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not math.isfinite(velocity_final) or velocity_final <= 0:
        raise ValueError("velocity_final must be positive and finite")
    if not math.isfinite(velocity_step) or velocity_step <= 0:
        raise ValueError("velocity_step must be positive and finite")
    event_cfg = env.event_manager.get_term_cfg(event_term_name)
    magnitude = _horizontal_push_magnitude(event_cfg.params["velocity_range"])
    if magnitude > velocity_final + 1e-9:
        raise ValueError("Current push magnitude exceeds velocity_final")
    state = _curriculum_state(env, f"push:{event_term_name}")
    # A transferred Phase 2 may already have many logical steps. Wait a fresh window.
    state.setdefault("last_update_step", curriculum_step(env))
    task_success = env.command_manager.get_term(command_name).performance.task_success_rate()
    if _update_survival_ema(env, env_ids, state, ema_beta=ema_beta, update_period=update_period) and bool(
        (state["ema"] > survival_gate) & (task_success > task_success_gate)
    ):
        magnitude = min(magnitude + velocity_step, velocity_final)
        for cfg in (event_cfg, getattr(env.cfg.events, event_term_name)):
            cfg.params["velocity_range"] = {axis: (-magnitude, magnitude) for axis in ("x", "y")}
    return {
        "max_delta_v_mps": magnitude,
        "survival_ema": state["ema"],
        "task_success_rate": task_success,
    }


def squat_walk_mode_introduction_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    command_name: str = "task_command",
    trigger_step: int = 400_000,
    target_mode_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> dict[str, float]:
    """Introduce the SQUAT_WALK mode at a fixed training step by rewriting the command mode weights.

    A **round-driven** (not performance-driven) curriculum that flips the unified :class:`SquatWalkCommand`
    from its mounted mode weights -- which reserve SQUAT_WALK at 0.0 -- to ``target_mode_weights`` once
    training has progressed far enough for STAND/SQUAT/WALK to be mature. Unlike :func:`command_range_curriculum`,
    it gates on the *training step counter* rather than a success metric, because the operator watches the
    curves and picks the introduction point by hand. This also sidesteps the AND-gating probability collapse
    of binding a brand-new mode to several performance gates at once (a new mode would drag each gated metric
    down and stall its own introduction).

    Why it rewrites ``term._mode_probs`` and **not** ``cfg.rel_mode_envs``:
        :class:`SquatWalkCommand.__init__` caches ``cfg.rel_mode_envs`` once into the GPU tensor
        ``self._mode_probs``, and :meth:`SquatWalkCommand.reset` draws every episode's mode from that tensor
        via ``torch.multinomial`` -- it never re-reads the cfg. Mutating ``cfg.rel_mode_envs`` at runtime is
        therefore a no-op; the live ``_mode_probs`` tensor must be overwritten in place. ``copy_`` keeps the
        existing tensor object (and its device/dtype), so the change is picked up by the very next reset.

    Timing:
        ``CurriculumManager.compute`` runs at the top of ``ManagerBasedRLEnv._reset_idx``, *before*
        ``command_manager.reset``. So on the first reset whose logical ``curriculum_step(env)`` has reached
        ``trigger_step``, the new weights are in place before that same reset draws modes -- the introduced
        distribution takes effect immediately for the envs resetting at that step.

        The logical counter advances by 1 per ``env.step()`` and includes restored progress, i.e. by
        ``num_steps_per_env`` per RSL-RL iteration. From scratch, ``trigger_step = trigger_iter *
        num_steps_per_env`` (e.g. iteration 8000 with
        ``num_steps_per_env = 50`` -> ``trigger_step = 400_000``).

    Interaction with :func:`command_range_curriculum`:
        The velocity envelope in :meth:`SquatWalkCommand._resample_command` derives its per-axis floor scale
        from the *live* ``ranges`` (nominal band). To keep that nominal stable while the new mode is being
        learned, prefer to **stagger** the two curricula or **freeze** the velocity axis around the
        introduction step (CASE 7.3); otherwise a simultaneously widening ``lin_vel_x`` shifts the envelope
        semantics under the freshly introduced SQUAT_WALK episodes.

    Note:
        * One-shot: the weights are rewritten exactly once per env (latched in environment state); later
          calls only report state. ``target_mode_weights`` need not sum to 1 (``multinomial`` normalizes).


    Args:
        env: The environment instance.
        env_ids: The environment indices being reset (unused; the switch is global and one-shot).
        command_name: Name of the :class:`SquatWalkCommand` term in the command manager.
        trigger_step: Cumulative logical control-step value at which SQUAT_WALK is introduced
            (``= trigger_iter * num_steps_per_env``).
        target_mode_weights: ``(STAND, SQUAT, WALK, SQUAT_WALK)`` weights written into ``_mode_probs`` at the
            trigger; the SQUAT_WALK slot (index 3) must be positive to actually introduce the mode.

    Returns:
        ``introduced`` (1.0 once switched, otherwise 0.0). Actual mode weights
        are logged once as sampling_probability by the performance monitor.
    """
    term = env.command_manager.get_term(command_name)

    # -- one-shot switch: rewrite the live mode-weight tensor the first time the counter reaches the trigger
    state = _curriculum_state(env, f"mode_intro:{command_name}")
    introduced = state.get("introduced", False)
    if not introduced and curriculum_step(env) >= trigger_step and not getattr(env, "_g1_restoring_curriculum", False):
        term._mode_probs.copy_(
            torch.tensor(
                target_mode_weights,
                dtype=term._mode_probs.dtype,
                device=term._mode_probs.device,
            )
        )
        state["introduced"] = True
        introduced = True

    return {
        "introduced": float(introduced),
    }
