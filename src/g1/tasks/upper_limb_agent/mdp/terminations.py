"""Timeout and free-base fall checks, enabled by each Upper environment configuration."""

import torch

from isaaclab.envs.mdp.terminations import bad_orientation, root_height_below_minimum, time_out

__all__ = ["bad_orientation", "root_height_below_minimum", "time_out", "task_time_out"]


def task_time_out(env, command_name="wrist_pose"):
    """Per-episode deadline for planar-enabled runs; legacy timeout otherwise.

    Register as time_out=True so RSL-RL bootstraps a normal time limit rather
    than treating it as a physical failure. Precision never changes this gate.
    """
    command = env.command_manager.get_term(command_name)
    if command.teleop_source == "manual":
        return torch.zeros_like(env.episode_length_buf, dtype=torch.bool)
    if command.teleop_source == "auto":
        age = command.episode_age_steps()
        return (age >= command.episode_deadline_steps) | (age >= env.max_episode_length)
    if not command.planar_enabled:
        return time_out(env)
    age = command.episode_age_steps()
    return (age >= command.episode_deadline_steps) | (age >= env.max_episode_length)
