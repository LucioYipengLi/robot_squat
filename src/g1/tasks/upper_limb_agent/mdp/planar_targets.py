"""Square-bounded planar stations and episode-wide target timing configuration."""

import math

import torch

from isaaclab.utils import configclass

from .goal_timing import GoalTimingCfg

__all__ = ["PlanarTaskCfg"]


@configclass
class PlanarTaskCfg:
    # Opt-in until far-field reward guidance is connected.
    probability: float = 0.0
    initial_half_extent_m: float = 0.25
    half_extent_m: float = 2.5
    initial_bearing_rad: float = math.pi / 6
    bearing_rad: float = math.pi
    max_goal_distance_m: float = 5.0  # Each active wrist, from the initial pelvis; full 3D norm.
    timing: GoalTimingCfg = GoalTimingCfg()

    def validate_parameters(self):
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("planar_task.probability must be finite and in [0,1]")
        if not all(math.isfinite(x) for x in (self.initial_half_extent_m, self.half_extent_m)) or not (
            0 < self.initial_half_extent_m <= self.half_extent_m
        ):
            raise ValueError("planar_task half extents must be positive, finite and ordered")
        if not all(math.isfinite(x) for x in (self.initial_bearing_rad, self.bearing_rad)) or not (
            0 <= self.initial_bearing_rad <= self.bearing_rad <= math.pi
        ):
            raise ValueError("planar_task bearings must satisfy 0 <= initial <= final <= pi")
        if not math.isfinite(self.max_goal_distance_m) or self.max_goal_distance_m <= 0:
            raise ValueError("planar_task.max_goal_distance_m must be positive and finite")
        self.timing.validate_parameters()


def sample_station(progress, local_radius, cfg):
    """Uniform bearing and sqrt-uniform radial fraction, within square and cap.

    This covers the square/sector without rejection loops; it is NOT uniform
    square-area sampling. A triangle-inequality radius budget protects every
    active wrist after the rigid target-group transform.
    """
    extent = cfg.initial_half_extent_m + progress * (cfg.half_extent_m - cfg.initial_half_extent_m)
    bearing = cfg.initial_bearing_rad + progress * (cfg.bearing_rad - cfg.initial_bearing_rad)
    angle = (2 * torch.rand_like(progress) - 1) * bearing
    direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
    room = cfg.max_goal_distance_m - local_radius
    if (room <= 0).any():
        raise ValueError("planar_task.max_goal_distance_m leaves no room for the local wrist targets")
    radius = torch.minimum(extent / direction.abs().amax(-1), room) * torch.rand_like(progress).sqrt()
    return direction * radius[:, None], torch.where(radius > 1e-6, angle, 0.0)
