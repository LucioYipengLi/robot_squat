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
    # Optional (normalized level, half extent [m], bearing limit [rad]) knots.
    # None retains linear interpolation for explicitly configured legacy tasks.
    curriculum_knots: tuple[tuple[float, float, float], ...] | None = None
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
        if self.curriculum_knots is not None:
            knots = self.curriculum_knots
            if len(knots) < 2 or any(len(k) != 3 or not all(math.isfinite(v) for v in k) for k in knots):
                raise ValueError("planar_task.curriculum_knots requires finite (progress, extent, bearing) triples")
            if not (
                math.isclose(knots[0][0], 0.0, abs_tol=1e-9)
                and math.isclose(knots[-1][0], 1.0, abs_tol=1e-9)
                and math.isclose(knots[0][1], self.initial_half_extent_m)
                and math.isclose(knots[-1][1], self.half_extent_m)
                and math.isclose(knots[0][2], self.initial_bearing_rad)
                and math.isclose(knots[-1][2], self.bearing_rad)
            ):
                raise ValueError(
                    "planar_task.curriculum_knots endpoints must match initial/final limits at progress 0/1"
                )
            if any(b[0] <= a[0] or b[1] < a[1] or b[2] < a[2] for a, b in zip(knots[:-1], knots[1:], strict=True)):
                raise ValueError("planar_task.curriculum_knots must have increasing progress and nondecreasing limits")
        self.timing.validate_parameters()


def station_limits(progress, cfg):
    """Use one schedule for sampling, episode-budget validation and diagnostics."""
    if cfg.curriculum_knots is None:
        return (
            cfg.initial_half_extent_m + progress * (cfg.half_extent_m - cfg.initial_half_extent_m),
            cfg.initial_bearing_rad + progress * (cfg.bearing_rad - cfg.initial_bearing_rad),
        )
    progress = torch.as_tensor(progress, dtype=torch.float32)
    knots = torch.tensor(cfg.curriculum_knots, device=progress.device, dtype=progress.dtype)
    right = torch.searchsorted(knots[:, 0].contiguous(), progress.contiguous(), right=True).clamp(1, len(knots) - 1)
    lo, hi = knots[right - 1], knots[right]
    fraction = (progress - lo[..., 0]) / (hi[..., 0] - lo[..., 0])
    return torch.lerp(lo[..., 1], hi[..., 1], fraction), torch.lerp(lo[..., 2], hi[..., 2], fraction)


def sample_station(progress, local_radius, cfg):
    """Uniform bearing and sqrt-uniform radial fraction, within square and cap.

    This covers the square/sector without rejection loops; it is NOT uniform
    square-area sampling. A triangle-inequality radius budget protects every
    active wrist after the rigid target-group transform.
    """
    extent, bearing = station_limits(progress, cfg)
    angle = (2 * torch.rand_like(progress) - 1) * bearing
    direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
    room = cfg.max_goal_distance_m - local_radius
    if (room <= 0).any():
        raise ValueError("planar_task.max_goal_distance_m leaves no room for the local wrist targets")
    radius = torch.minimum(extent / direction.abs().amax(-1), room) * torch.rand_like(progress).sqrt()
    return direction * radius[:, None], torch.where(radius > 1e-6, angle, 0.0)
