"""Rest-to-rest time estimates; these budgets never filter policy actions."""

import math

import torch

from isaaclab.utils import configclass

__all__ = ["GoalTimingCfg"]


@configclass
class GoalTimingCfg:
    linear_speed: float = 0.4  # m/s
    linear_acceleration: float = 0.3  # m/s², symmetric start/stop assumption
    angular_speed: float = 0.4  # rad/s
    angular_acceleration: float = 0.4  # rad/s²
    ability_fraction: float = 0.8
    settle_s: float = 2.0
    evaluation_s: float = 10.0
    near_episode_s: float = 10.0

    def validate_parameters(self):
        for name in (
            "linear_speed",
            "linear_acceleration",
            "angular_speed",
            "angular_acceleration",
            "settle_s",
            "evaluation_s",
            "near_episode_s",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"goal timing {name} must be positive and finite")
        if not math.isfinite(self.ability_fraction) or not 0 < self.ability_fraction <= 1:
            raise ValueError("goal timing ability_fraction must be in (0,1]")


def rest_to_rest_time(distance, speed, acceleration):
    """Triangular/trapezoidal profile duration; distance >= 0, speed/a > 0.

    Inputs broadcast; tensor inputs keep their device. The longer-distance
    expression already includes acceleration and deceleration time.
    """
    distance = torch.as_tensor(distance)
    speed = torch.as_tensor(speed, device=distance.device)
    return torch.where(
        distance < speed.square() / acceleration,
        2 * (distance / acceleration).sqrt(),
        distance / speed + speed / acceleration,
    )


def budget_steps(distance, angle, forward_limit, yaw_limit, cfg, dt):
    """Snapshot the evaluation start and deadline, rounded UP to control steps."""
    v = (forward_limit * cfg.ability_fraction).clamp(max=cfg.linear_speed)
    w = (yaw_limit * cfg.ability_fraction).clamp(max=cfg.angular_speed)
    seconds = (
        rest_to_rest_time(distance, v, cfg.linear_acceleration)
        + rest_to_rest_time(angle.abs(), w, cfg.angular_acceleration)
        + cfg.settle_s
    )
    start = torch.ceil(seconds / dt - 1e-6).long()
    return start, start + math.ceil(cfg.evaluation_s / dt - 1e-9)
