# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Performance-driven curriculum terms for the G1 squat/walk task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
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
    * **Velocity axis** -- gated on the ``success_rate_vel`` of **WALK-mode episodes only** (masked via
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
          always read zeros. Averaging over all envs gives a stable rolling global rate. The velocity
          gate masks by ``last_mode`` -- not the live ``mode`` -- because the curriculum runs before the
          command term's reset, so ``mode`` already points at the new episode while these buffers still
          describe the one that just ended.
        * The expansion is applied only once every ``update_period`` control steps (a full episode is a
          few hundred steps), so each increment is followed by enough rollouts for the gated metric to
          react before the next one -- avoiding runaway expansion against the feedback latency.

    Args:
        env: The environment instance.
        env_ids: The environment indices being reset (unused; the curriculum is global).
        command_name: Name of the :class:`SquatWalkCommand` term in the command manager.
        height_success_gate: All-env height ``success_rate`` above which the height band expands.
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
    height_success = term.episode_success_rate.mean()
    # WALK-mode episodes only (SQUAT_WALK also walks, but its proportion is 0 in the current config).
    # A masked mean via sum/clamp avoids both a sync and the NaN of an empty selection.
    walk_mask = (term.last_mode == term.MODE_WALK) | (term.last_mode == term.MODE_SQUAT_WALK)
    walk_mask = walk_mask.to(height_success.dtype)
    vel_success = (term.episode_success_rate_vel * walk_mask).sum() / walk_mask.sum().clamp_min(1.0)

    # -- apply the expansion at most once per ``update_period`` steps
    if env.common_step_counter % update_period == 0:
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
