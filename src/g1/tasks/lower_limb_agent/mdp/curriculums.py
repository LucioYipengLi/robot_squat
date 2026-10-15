# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Performance-driven curriculum terms for the G1 squat/walk task."""

from __future__ import annotations

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
        * The gates read the command term's ``episode_success_rate`` / ``episode_success_rate_vel``
          buffers -- dedicated per-episode buffers holding each env's last finalized success. They are
          kept separate from ``term.metrics`` because the base ``CommandTerm.reset`` zeroes ``metrics``
          right after logging it, while this curriculum runs *before* the next reset and would otherwise
          always read zeros. Only envs with a finalized, nonempty episode enter the mean. The velocity
          gate masks by ``last_mode`` to match the episode that produced each stored metric. Due to
          manager ordering, the batch ending now is finalized after this curriculum call; these are
          the most recently finalized buffers, not statistics from the live unfinished episodes.
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
        expanding range's live bounds (``*_lo``/``*_hi``) and width (``*_span``).
    """
    term = env.command_manager.get_term(command_name)
    ranges = term.cfg.ranges

    # -- gate signals: read the command term's *persisted* per-episode success buffers, NOT
    #    ``term.metrics`` -- the base ``CommandTerm.reset`` zeroes ``metrics`` after logging, and this
    #    curriculum runs before the next reset, so ``metrics`` would always read ~0 here (no GPU->CPU sync).
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

    return {
        "height_success": height_success,
        "height_offset_lo": ranges.height_offset[0],
        "height_offset_hi": ranges.height_offset[1],
        "height_offset_span": ranges.height_offset[1] - ranges.height_offset[0],
        "vel_success": vel_success,
        "lin_vel_x_lo": ranges.lin_vel_x[0],
        "lin_vel_x_hi": ranges.lin_vel_x[1],
        "lin_vel_x_span": ranges.lin_vel_x[1] - ranges.lin_vel_x[0],
        "lin_vel_y_hi": ranges.lin_vel_y[1],
        "lin_vel_y_span": ranges.lin_vel_y[1] - ranges.lin_vel_y[0],
    }


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
) -> dict[str, float]:
    """Ramp one disturbance group's magnitude (``joint_noise_frac``) up while the policy keeps surviving.

    A performance-driven companion to the *implicit* interval curriculum. The interval event already
    gates disturbance *frequency* on episode length (it can only fire once an episode outlives
    ``interval_range_s[0]``), which naturally withholds the disturbance until the policy can stand. This
    term adds the missing axis -- disturbance *magnitude* -- so the selected joint group can be pushed
    beyond the fixed ``joint_noise_frac`` it was mounted with, instead of switching on at full strength.

    :class:`UpperBodyDisturbanceEvent` organizes its joints into ``disturb_groups`` (arms, waist, ...),
    each with an independently mounted ``joint_noise_frac`` (a fraction in [0, 1] of each joint's
    default-to-limit margin). This curriculum ramps only the group at ``target_group`` (default 0 =
    arms); the other groups stay fixed at their mounted values.

    It reads the live event cfg via :meth:`EventManager.get_term_cfg` (a reference, not a copy) and
    mutates ``params["disturb_groups"][target_group]["joint_noise_frac"]`` in place; because
    :class:`UpperBodyDisturbanceEvent.__call__` re-reads each group's ``joint_noise_frac`` from
    ``**term_cfg.params`` on every fire, the new magnitude takes effect on the next disturbance without
    touching the event code or re-instantiating the term.

    Design mirrors :func:`command_range_curriculum`:

    * **Gate** -- episode-average survival: each naturally completed, fully observed episode has
      equal weight within an update window. Sum normalized lengths and episode counts across reset
      batches, then update the EMA once per window. Initial resets, manual truncations, duplicate
      resets and artificially aged first episodes are excluded. This is a duration fraction, not
      a binary timeout-success rate or a lifetime arithmetic mean.
    * **Monotonic + self-pacing** -- ``joint_noise_frac`` only creeps toward ``frac_final`` (never
      decays), and only while the EMA survival holds above ``survival_gate``. Raising the magnitude makes
      surviving harder, which pulls the EMA back below the gate and pauses the ramp, so it settles at the
      largest magnitude the policy can survive -- the same negative feedback as the command curriculum.
    * **Rate-limited** -- at most one increment per ``update_period`` control steps, so each bump is
      followed by enough rollouts for the gate to react before the next (avoids runaway against the
      feedback latency).

    Note:
        * Sample accumulation stays on the device. Checking the window count and gate synchronizes
          only when an update is due. Windows with no completed episodes do not update or ramp.
        * When enabling this term, set the target group's mounted ``joint_noise_frac`` to the desired
          *floor* (e.g. 0.15): the ramp starts from that live value and climbs to ``frac_final``.
        * Do **not** also curriculum ``interval_range_s`` -- frequency is already governed by the implicit
          episode-length gate, and a second controller on the same axis would fight it.

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

    Returns:
        Curriculum state logged under ``Curriculum/<term_name>/<key>``: the survival EMA gate signal
        (``survival_ema``), its validity flag, completed-episode count and disturbance magnitude.
    """
    # live reference to the event term cfg (get_term_cfg returns the stored object, not a copy);
    # the target group's joint_noise_frac is a plain Python float re-read by the event on every fire.
    event_cfg = env.event_manager.get_term_cfg(event_term_name)
    group = event_cfg.params["disturb_groups"][target_group]
    frac = float(group["joint_noise_frac"])

    if not 0.0 < ema_beta <= 1.0:
        raise ValueError("ema_beta must be in (0, 1]")
    state = _curriculum_state(env, f"survival:{event_term_name}:{target_group}")
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
    # _reset_idx calls the curriculum before clearing episode lengths. Track actual elapsed steps
    # separately so randomized initial counters cannot masquerade as fully observed episodes.
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

    ema = state["ema"]
    if _update_due(env, state, update_period) and bool(state["episode_count"] > 0):
        episode_mean = state["length_sum"] / state["episode_count"]
        if state["initialized"]:
            ema.lerp_(episode_mean, ema_beta)
        else:
            ema.copy_(episode_mean)
            state["initialized"] = True
        state["length_sum"].zero_()
        state["episode_count"].zero_()
        if bool(ema > survival_gate) and frac < frac_final:
            frac = min(frac + frac_step, frac_final)
            group["joint_noise_frac"] = frac

    return {
        "survival_ema": ema,
        "survival_valid": float(state["initialized"]),
        "survival_episodes": state["total_episodes"],
        "joint_noise_frac": frac,
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
        * No GPU->CPU sync on the hot path: the returned ``squat_walk_weight`` is a 0-dim tensor (like
          :func:`command_range_curriculum`'s signals); the latch check is pure Python.

    Args:
        env: The environment instance.
        env_ids: The environment indices being reset (unused; the switch is global and one-shot).
        command_name: Name of the :class:`SquatWalkCommand` term in the command manager.
        trigger_step: Cumulative logical control-step value at which SQUAT_WALK is introduced
            (``= trigger_iter * num_steps_per_env``).
        target_mode_weights: ``(STAND, SQUAT, WALK, SQUAT_WALK)`` weights written into ``_mode_probs`` at the
            trigger; the SQUAT_WALK slot (index 3) must be positive to actually introduce the mode.

    Returns:
        Curriculum state logged under ``Curriculum/<term_name>/<key>``: ``squat_walk_weight`` (the live
        SQUAT_WALK weight, index 3 of ``_mode_probs``) and ``introduced`` (1.0 once switched, else 0.0), so the
        switch moment is visible in TensorBoard.
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
        "squat_walk_weight": term._mode_probs[3],
        "introduced": float(introduced),
    }
