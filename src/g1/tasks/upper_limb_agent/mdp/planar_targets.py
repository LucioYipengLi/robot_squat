"""Square-bounded planar stations and episode-wide target timing configuration."""

import math
from copy import deepcopy

import torch

from isaaclab.utils import configclass

from .goal_timing import GoalTimingCfg

__all__ = ["PlanarTaskCfg", "PlanarTransitionSamplingCfg", "PlanarTurnFocusCfg"]

PLANAR_SAMPLING_GROUPS = ("legacy", "walk", "turn", "wide")


@configclass
class PlanarTurnFocusCfg:
    """Training-only oversampling of newly available bearings within Turn and Wide."""

    enabled: bool = False
    start_level: int = 5
    turn_probability: float = 0.70
    wide_probability: float = 0.40

    def validate(self):
        if not isinstance(self.start_level, int) or isinstance(self.start_level, bool) or self.start_level < 1:
            raise ValueError("planar_turn_focus.start_level must be a positive integer")
        for probability in (self.turn_probability, self.wide_probability):
            if not math.isfinite(probability) or not 0 <= probability < 1:
                raise ValueError("planar_turn_focus probabilities must be in [0, 1) to retain native samples")


@configclass
class PlanarTransitionSamplingCfg:
    """Begin outside the near-field radius and gradually include the wider level ranges."""

    enabled: bool = False
    min_distance_m: float = 0.25
    walk_fraction: float = 0.75
    walk_max_distance_m: float = 0.40
    walk_max_bearing_rad: float = math.radians(15)
    turn_max_distance_m: float = 0.30
    turn_bearing_rad: tuple[float, float] = (math.radians(20), math.radians(25))
    wide_start_progress: float = 0.30
    # Optional (progress, wide probability, turn probability) knots. Walk takes the remainder.
    probability_knots: tuple[tuple[float, float, float], ...] | None = None
    # Optional (first progress, turn maximum in rad) before turn follows the wide bearing.
    turn_bearing_bridge: tuple[float, float] | None = None

    def validate(self, max_goal_distance_m):
        if not math.isfinite(self.min_distance_m) or not 0 < self.min_distance_m < max_goal_distance_m:
            raise ValueError("transition_sampling requires 0 < min_distance_m < max_goal_distance_m")
        if any(
            not math.isfinite(limit) or not self.min_distance_m <= limit <= max_goal_distance_m
            for limit in (self.walk_max_distance_m, self.turn_max_distance_m)
        ):
            raise ValueError("transition_sampling maximum distances must contain the minimum and fit the goal cap")
        if not 0 <= self.walk_fraction <= 1 or not 0 <= self.wide_start_progress < 1:
            raise ValueError("transition_sampling requires walk_fraction in [0,1] and wide_start_progress in [0,1)")
        if not 0 <= self.walk_max_bearing_rad <= math.pi:
            raise ValueError("transition_sampling walk bearing must be in [0,pi]")
        if len(self.turn_bearing_rad) != 2 or not 0 <= self.turn_bearing_rad[0] <= self.turn_bearing_rad[1] <= math.pi:
            raise ValueError("transition_sampling turn bearings must be ordered in [0,pi]")
        bridge = self.turn_bearing_bridge
        if bridge is not None and (
            len(bridge) != 2 or not 0 <= bridge[0] < 1 or not self.turn_bearing_rad[1] <= bridge[1] <= math.pi
        ):
            raise ValueError("transition_sampling turn_bearing_bridge requires valid (progress, maximum bearing)")
        if (knots := self.probability_knots) is not None:
            if len(knots) < 2 or any(
                len(k) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in k) or k[1] + k[2] > 1 for k in knots
            ):
                raise ValueError("transition_sampling requires valid (progress, wide, turn) probability triples")
            if (
                knots[0][0] != 0
                or knots[-1][0] != 1
                or any(a[0] >= b[0] for a, b in zip(knots[:-1], knots[1:], strict=True))
            ):
                raise ValueError("transition_sampling probability progress must increase from 0 to 1")

    def group_probabilities(self, progress):
        progress = torch.as_tensor(progress, dtype=torch.float32)
        if self.probability_knots is not None:
            wide, turn = _interpolate_knots(progress, self.probability_knots).unbind(-1)
            return 1 - wide - turn, turn, wide
        wide = ((progress - self.wide_start_progress) / (1 - self.wide_start_progress)).clamp(0, 1)
        walk = (1 - wide) * self.walk_fraction
        return walk, (1 - wide) * (1 - self.walk_fraction), wide

    def turn_bearing_limit(self, progress, wide_bearing, wide_probability):
        minimum, maximum = self.turn_bearing_rad
        if self.turn_bearing_bridge is None:
            return maximum
        bridge_start, bridge_limit = self.turn_bearing_bridge
        progress = torch.as_tensor(progress, device=wide_probability.device)
        limit = torch.where(progress >= bridge_start, bridge_limit, maximum)
        wide_limit = torch.as_tensor(wide_bearing, device=wide_probability.device).clamp_min(minimum)
        return torch.where(wide_probability > 0, wide_limit, limit)


@configclass
class PlanarTaskCfg:
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
    transition_sampling: PlanarTransitionSamplingCfg = PlanarTransitionSamplingCfg()

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
        if self.transition_sampling.enabled:
            self.transition_sampling.validate(self.max_goal_distance_m)
        self.timing.validate_parameters()


def _interpolate_knots(progress, values):
    progress = torch.as_tensor(progress, dtype=torch.float32)
    knots = progress.new_tensor(values)
    right = torch.searchsorted(knots[:, 0].contiguous(), progress.contiguous(), right=True).clamp(1, len(knots) - 1)
    lo, hi = knots[right - 1], knots[right]
    fraction = (progress - lo[..., 0]) / (hi[..., 0] - lo[..., 0])
    return torch.lerp(lo[..., 1:], hi[..., 1:], fraction[..., None])


def station_limits(progress, cfg):
    """Use one schedule for sampling, episode-budget validation and diagnostics."""
    if cfg.curriculum_knots is None:
        return (
            cfg.initial_half_extent_m + progress * (cfg.half_extent_m - cfg.initial_half_extent_m),
            cfg.initial_bearing_rad + progress * (cfg.bearing_rad - cfg.initial_bearing_rad),
        )
    return _interpolate_knots(progress, cfg.curriculum_knots).unbind(-1)


def sample_station(progress, local_radius, cfg):
    """Uniform bearing and sqrt-uniform radial fraction, within square and cap.

    This covers the square/sector without rejection loops; it is NOT uniform
    square-area sampling. A triangle-inequality radius budget protects every
    active wrist after the rigid target-group transform.
    """
    return _sample_station(*station_limits(progress, cfg), local_radius, cfg, progress)


def _sample_station(extent, bearing, local_radius, cfg, random_like):
    angle = (2 * torch.rand_like(random_like) - 1) * bearing
    direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
    room = cfg.max_goal_distance_m - local_radius
    if (room <= 0).any():
        raise ValueError("planar_task.max_goal_distance_m leaves no room for the local wrist targets")
    radius = torch.minimum(extent / direction.abs().amax(-1), room) * torch.rand_like(random_like).sqrt()
    return direction * radius[:, None], torch.where(radius > 1e-6, angle, 0.0)


def station_budget_bounds(progress, cfg):
    """Conservative (distance, absolute bearing) bounds for each enabled sampling group."""
    extent, bearing = station_limits(progress, cfg)
    wide_bounds = (math.sqrt(2) * extent, bearing)
    transition = cfg.transition_sampling
    if not transition.enabled:
        return (wide_bounds,)
    walk, turn, wide = transition.group_probabilities(progress)
    bounds = []
    if walk > 0:
        bounds.append((transition.walk_max_distance_m, transition.walk_max_bearing_rad))
    if turn > 0:
        bounds.append((transition.turn_max_distance_m, transition.turn_bearing_limit(progress, bearing, wide)))
    if wide > 0:
        if extent < transition.min_distance_m:
            raise ValueError("transition_sampling minimum distance exceeds the wide square half extent")
        bounds.append(wide_bounds)
    return bounds


def sample_transition_station(progress, local_radius, cfg):
    """Sample a walk, turn or wide target, with area-weighted radius in a bounded interval."""
    return _sample_transition_station(*_transition_limits(progress, cfg), local_radius, cfg, progress)[:3]


def _transition_limits(progress, cfg):
    transition = cfg.transition_sampling
    walk_fraction, _, wide_fraction = transition.group_probabilities(progress)
    extent, bearing = station_limits(progress, cfg)
    turn_max = transition.turn_bearing_limit(progress, bearing, wide_fraction)
    return extent, bearing, walk_fraction, wide_fraction, turn_max - transition.turn_bearing_rad[0]


def _sample_transition_station(
    extent, bearing, walk_fraction, wide_fraction, turn_span, local_radius, cfg, random_like, *, focus_limits=None
):
    transition = cfg.transition_sampling
    choice = torch.rand_like(random_like)
    wide = choice < wide_fraction
    walk = ~wide & (choice < wide_fraction + walk_fraction)
    signed = 2 * torch.rand_like(random_like) - 1
    sign = torch.where(signed < 0, -1.0, 1.0)
    turn_min = transition.turn_bearing_rad[0]
    turn = (turn_min + signed.abs() * turn_span) * sign
    angle = torch.where(wide, signed * bearing, torch.where(walk, signed * transition.walk_max_bearing_rad, turn))
    focused = native_reference = None
    if focus_limits is not None:
        prev_turn, prev_wide, turn_probability, wide_probability = focus_limits
        group_start = torch.where(wide, 0.0, torch.where(walk, wide_fraction, wide_fraction + walk_fraction))
        turn_fraction = 1.0 - (wide_fraction + walk_fraction)
        group_width = torch.where(wide, wide_fraction, torch.where(walk, walk_fraction, turn_fraction))
        # The within-group quantile of the group draw is uniform for every group.
        # Reserving the same top quantile in each group keeps the native mix for promotion.
        quantile = (choice - group_start) / group_width.clamp_min(torch.finfo(choice.dtype).tiny)
        probability = torch.where(wide, wide_probability, torch.where(walk, 0.0, turn_probability))
        focused = quantile < probability
        native_reference = quantile >= torch.maximum(turn_probability, wide_probability)
        low = torch.where(wide, prev_wide, prev_turn)
        high = torch.where(wide, bearing, turn_min + turn_span)
        focus_angle = torch.lerp(low, high, signed.abs()) * sign
        angle = torch.where(focused, focus_angle, angle)
    direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
    upper = torch.where(
        wide,
        extent / direction.abs().amax(-1),
        torch.where(walk, transition.walk_max_distance_m, transition.turn_max_distance_m),
    )
    upper = torch.minimum(upper, cfg.max_goal_distance_m - local_radius)
    if (upper < transition.min_distance_m).any():
        raise ValueError("transition_sampling minimum distance conflicts with the planar wrist distance cap")
    lower_sq = transition.min_distance_m**2
    radius = (lower_sq + (upper.square() - lower_sq) * torch.rand_like(random_like)).sqrt()
    group = torch.where(wide, 3, torch.where(walk, 1, 2))
    return direction * radius[:, None], angle, group, focused, native_reference


class PlanarStationSampler:
    """Batch lookup of the fixed schedule at each episode's level."""

    def __init__(self, num_levels, device):
        self._progress = (
            torch.arange(num_levels, device=device, dtype=torch.float32) / (num_levels - 1)
            if num_levels > 1
            else torch.ones(1, device=device)
        )
        self._schedule = None
        self._table = None

    def _limits(self, levels, cfg):
        transition = cfg.transition_sampling
        # Evaluation can override the sampling schedule after the environment is built.
        schedule = (
            cfg.initial_half_extent_m,
            cfg.half_extent_m,
            cfg.initial_bearing_rad,
            cfg.bearing_rad,
            cfg.curriculum_knots,
            transition.enabled,
            transition.walk_fraction,
            transition.wide_start_progress,
            transition.probability_knots,
            transition.turn_bearing_rad,
            transition.turn_bearing_bridge,
        )
        if self._schedule != schedule:
            values = (
                _transition_limits(self._progress, cfg) if transition.enabled else station_limits(self._progress, cfg)
            )
            self._table = torch.stack(
                [
                    torch.as_tensor(value, dtype=self._progress.dtype, device=self._progress.device).expand_as(
                        self._progress
                    )
                    for value in values
                ],
                dim=-1,
            )
            self._schedule = deepcopy(schedule)
        return self._table[levels].unbind(dim=-1)

    def sample(self, levels, local_radius, cfg):
        values = self._limits(levels, cfg)
        if cfg.transition_sampling.enabled:
            return _sample_transition_station(*values, local_radius, cfg, values[0])[:3]
        shift, angle = _sample_station(*values, local_radius, cfg, values[0])
        return shift, angle, None

    def sample_turn_focus(self, levels, local_radius, cfg, focus):
        values = self._limits(levels, cfg)
        extent, bearing, walk_fraction, wide_fraction, turn_span = values
        _, prev_bearing, _, _, prev_turn_span = self._table[(levels - 1).clamp_min(0)].unbind(dim=-1)
        active = levels >= focus.start_level
        turn_grew = (turn_span > prev_turn_span) & (walk_fraction + wide_fraction < 1)
        wide_grew = (bearing > prev_bearing) & (wide_fraction > 0)
        focus_limits = (
            cfg.transition_sampling.turn_bearing_rad[0] + prev_turn_span,
            prev_bearing,
            torch.where(active & turn_grew, focus.turn_probability, 0.0),
            torch.where(active & wide_grew, focus.wide_probability, 0.0),
        )
        return _sample_transition_station(*values, local_radius, cfg, extent, focus_limits=focus_limits)
