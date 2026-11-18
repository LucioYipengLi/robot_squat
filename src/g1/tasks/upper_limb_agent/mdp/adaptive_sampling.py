"""Task-level sampling feedback, updated only at complete curriculum windows."""

import math
from dataclasses import asdict

from isaaclab.utils import configclass

__all__ = ["AdaptivePlanarSamplingCfg"]


@configclass
class AdaptivePlanarSamplingCfg:
    enabled: bool = True
    history_windows: int = 4  # 4 * 3600 control steps / 24 rollout steps = 600 PPO iterations.
    minimum_relative_improvement: float = 0.10
    probability_step: float = 0.05
    maximum_probability: float = 0.30
    other_tasks_pose_ready_gate: float = 0.85
    cooldown_windows: int = 1


class AdaptivePlanarSampling:
    def __init__(self, cfg, base_probability):
        if not isinstance(cfg.history_windows, int) or cfg.history_windows < 2 or cfg.history_windows % 2:
            raise ValueError("Sampling history_windows must be an even integer >= 2")
        if not isinstance(cfg.cooldown_windows, int) or cfg.cooldown_windows < 1:
            raise ValueError("Sampling cooldown_windows must be an integer >= 1")
        if not math.isfinite(cfg.minimum_relative_improvement) or not 0 < cfg.minimum_relative_improvement <= 1:
            raise ValueError("Sampling minimum_relative_improvement must be in (0,1]")
        if not math.isfinite(cfg.probability_step) or not 0 < cfg.probability_step <= 1:
            raise ValueError("Sampling probability_step must be in (0,1]")
        if not math.isfinite(cfg.maximum_probability) or not 0 < base_probability <= cfg.maximum_probability < 1:
            raise ValueError("Sampling maximum_probability must contain the base probability and remain below 1")
        if not math.isfinite(cfg.other_tasks_pose_ready_gate) or not 0 <= cfg.other_tasks_pose_ready_gate <= 1:
            raise ValueError("Sampling other_tasks_pose_ready_gate must be finite and in [0, 1]")
        self.cfg, self.base_probability = cfg, base_probability
        self.reset()

    def reset(self):
        self.probability = self.base_probability
        self.history = []
        self.cooldown = 0
        self.adjustments = 0  # Within this curriculum level.

    def trend(self, failure_limit):
        if len(self.history) < self.cfg.history_windows:
            return None
        half = self.cfg.history_windows // 2
        rates = [
            10 * sum(x[0] for x in part) / sum(x[1] for x in part)
            for part in (self.history[:half], self.history[half:])
        ]
        improvement = (rates[0] - rates[1]) / max(rates[0], failure_limit, 1e-9)
        return *rates, improvement

    def observe(self, failures, elapsed_s, eligible, failure_limit, *, other_tasks_quality_ready: bool):
        # Cooldown consumes a check, not its samples. Keep collecting the sliding
        # history so the following check can react without another warm-up.
        cooling = self.cooldown > 0
        self.cooldown = max(0, self.cooldown - 1)
        if not eligible or elapsed_s <= 0:
            self.history.clear()  # Require consecutive sufficiently sampled windows.
            return
        self.history.append((failures, elapsed_s))
        self.history = self.history[-self.cfg.history_windows :]
        trend = self.trend(failure_limit)
        # Keep the failure history while other tasks recover; only pause increases.
        if cooling or not other_tasks_quality_ready or trend is None or 10 * failures / elapsed_s <= failure_limit:
            return
        if trend[2] < self.cfg.minimum_relative_improvement and self.probability < self.cfg.maximum_probability:
            self.probability = min(
                self.cfg.maximum_probability, round(self.probability + self.cfg.probability_step, 10)
            )
            self.cooldown = self.cfg.cooldown_windows
            self.adjustments += 1

    def logs(self, failure_limit):
        result = {
            "planar_probability": self.probability,
            "history_windows": len(self.history),
            "cooldown_remaining": self.cooldown,
            "level_adjustments": self.adjustments,
        }
        trend = self.trend(failure_limit)
        if trend is not None:
            names = ("previous_failure_per_10s", "recent_failure_per_10s", "relative_improvement")
            result.update(zip(names, trend, strict=True))
        return {f"Curriculum/upper_sampling/{key}": value for key, value in result.items()}

    def state_dict(self):
        return {
            "schema_version": 1,
            "cfg": asdict(self.cfg),
            "base_probability": self.base_probability,
            "probability": self.probability,
            "history": list(self.history),
            "cooldown": self.cooldown,
            "adjustments": self.adjustments,
        }

    def load_state_dict(self, state):
        """Old runs/settings start a fresh sampling schedule, preserving the range curriculum."""
        self.reset()
        if (
            state is None
            or state.get("cfg") != asdict(self.cfg)
            or state.get("base_probability") != self.base_probability
        ):
            return False
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported adaptive sampling state")
        probability = state.get("probability", float("nan"))
        history = state.get("history", [])
        cooldown, adjustments = state.get("cooldown"), state.get("adjustments")
        if (
            not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not self.base_probability <= probability <= self.cfg.maximum_probability
            or not isinstance(history, (list, tuple))
            or len(history) > self.cfg.history_windows
            or not isinstance(cooldown, int)
            or not 0 <= cooldown <= self.cfg.cooldown_windows
            or not isinstance(adjustments, int)
            or adjustments < 0
        ):
            raise ValueError("Invalid adaptive sampling state")
        for item in history:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], int)
                or item[0] < 0
                or not isinstance(item[1], (int, float))
                or not math.isfinite(item[1])
                or item[1] <= 0
            ):
                raise ValueError("Invalid adaptive sampling history")
        self.probability, self.history = probability, [tuple(item) for item in history]
        self.cooldown, self.adjustments = cooldown, adjustments
        return True
