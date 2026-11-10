"""Timeout and free-base fall checks, enabled by each Upper environment configuration."""

from isaaclab.envs.mdp.terminations import bad_orientation, root_height_below_minimum, time_out

__all__ = ["bad_orientation", "root_height_below_minimum", "time_out"]
