# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish diagnostic windows directly to the installed RSL-RL writer."""

from contextlib import contextmanager
from functools import wraps

from g1.tasks.lower_limb_agent.mdp.curriculums import curriculum_step

from .tensorboard_layout import ensure_performance_layout
from .tensorboard_modes import ModeComparisonWriter

_LEGACY_COMMAND_METRICS = (
    "error_height",
    "success_rate",
    "error_vel_xy",
    "error_vel_yaw",
    "success_rate_vel",
    "error_pelvis_xy",
    "success_rate_pelvis_xy",
)
_LEGACY_RANGE_FIELDS = (
    "height_offset_lo",
    "height_offset_hi",
    "height_offset_span",
    "lin_vel_x_lo",
    "lin_vel_x_hi",
    "lin_vel_x_span",
    "lin_vel_y_hi",
    "lin_vel_y_span",
)
_TIME_AXIS_ALIASES = frozenset(("Train/mean_reward/time", "Train/mean_episode_length/time"))


def _prune_legacy_extras(logger, command_names):
    """Filter logger-owned copies, preserving environment extras and curriculum state."""
    removed = {"Metrics/success_rate"}
    removed.update(f"Metrics/{name}/{metric}" for name in command_names for metric in _LEGACY_COMMAND_METRICS)
    removed.update(f"Curriculum/command_range/{field}" for field in _LEGACY_RANGE_FIELDS)
    removed.update(
        (
            "Curriculum/upper_body_disturbance_magnitude/joint_noise_frac",
            "Curriculum/squat_walk_mode_introduction/squat_walk_weight",
        )
    )
    if hasattr(logger, "ep_extras"):
        logger.ep_extras[:] = [
            {
                key: value
                for key, value in batch.items()
                if key not in removed and not key.startswith("Episode_Termination/")
            }
            for batch in logger.ep_extras
        ]


@contextmanager
def _without_time_axis_aliases(writer):
    """Suppress duplicate scalar series during this synchronous log call only."""
    if writer is None:
        yield
        return
    original = writer.add_scalar
    had_override = "add_scalar" in vars(writer)

    @wraps(original)
    def add_scalar(tag, *args, **kwargs):
        if tag not in _TIME_AXIS_ALIASES:
            return original(tag, *args, **kwargs)
        return None

    writer.add_scalar = add_scalar
    try:
        yield
    finally:
        if had_override:
            writer.add_scalar = original
        else:
            del writer.add_scalar


def install_performance_logging(logger, env):
    """Use one sample-weighted diagnostic window per PPO log call, for any writer backend.

    RSL-RL averages reset-batch extras without their original sample counts. Writing our
    already-reduced scalars directly avoids that bias. Superseded legacy outputs are removed
    only from logging; their internal buffers and curriculum gate calculations remain intact.
    Only this runner's logger instance is adapted; upstream classes/modules stay untouched.
    """
    original = logger.log
    original_stop = getattr(logger, "stop_logging_writer", lambda: None)
    registered_log_dir = None
    comparison_writer = None

    def close_comparison():
        nonlocal comparison_writer
        if comparison_writer is not None:
            comparison_writer.close()
            comparison_writer = None

    @wraps(original_stop)
    def stop_logging_writer():
        try:
            return original_stop()
        finally:
            close_comparison()

    @wraps(original)
    def log(*args, **kwargs):
        nonlocal registered_log_dir, comparison_writer
        iteration = kwargs["it"] if "it" in kwargs else args[0]
        base = env.unwrapped
        monitored = []
        published = {}
        for name in base.command_manager.active_terms:
            term = base.command_manager.get_term(name)
            monitor = getattr(term, "performance", None)
            if monitor is None:
                continue
            monitored.append(name)
            values = monitor.drain(term._mode_probs)
            if logger.writer is None:
                continue
            for key, value in values.items():
                key = key.replace("Performance/", f"Performance/{name}/", 1)
                logger.writer.add_scalar(key, value, iteration)
                published[key] = value
            prefix = f"TaskDistribution/{name}/"
            context = {
                "phase": base.cfg.training_phase,
                "curriculum_control_steps": curriculum_step(base),
            }
            for field in vars(term.cfg.ranges):
                bounds = getattr(term.cfg.ranges, field)
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    context[f"{field}/lo"], context[f"{field}/hi"] = bounds
            event_cfg = base.cfg.events.upper_body_disturbance
            if event_cfg is not None:
                event_cfg = base.event_manager.get_term_cfg("upper_body_disturbance")
                for index, group in enumerate(event_cfg.params["disturb_groups"]):
                    context[f"disturbance_group_{index}_fraction"] = group["joint_noise_frac"]
            for key, value in context.items():
                logger.writer.add_scalar(prefix + key, value, iteration)
                published[prefix + key] = value
        if not monitored:
            return original(*args, **kwargs)
        log_dir = getattr(logger.writer, "log_dir", None)
        if log_dir is not None:
            if log_dir != registered_log_dir or comparison_writer is None:
                if comparison_writer is not None:
                    comparison_writer.close()
                ensure_performance_layout(log_dir, monitored)
                comparison_writer = ModeComparisonWriter(log_dir, monitored)
                registered_log_dir = log_dir
            comparison_writer.add_scalars(published, iteration)
        _prune_legacy_extras(logger, monitored)
        with _without_time_axis_aliases(logger.writer):
            return original(*args, **kwargs)

    logger.log = log
    logger.stop_logging_writer = stop_logging_writer
    return close_comparison
