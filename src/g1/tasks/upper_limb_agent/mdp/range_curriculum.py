"""Episode-owned command ranges and a windowed, pose-quality curriculum.

The command term calls ``complete`` before clearing old episode metrics, then
``begin`` before sampling new goals. No simulator state or PPO update is owned
here. Counts are episode-weighted, independent of TensorBoard log reductions.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import torch

from isaaclab.utils import configclass

__all__ = ["RangeCurriculumCfg", "RangeCurriculum"]


@configclass
class RangeCurriculumCfg:
    """Final Lower limits come from the action configs; target limits from commands."""

    initial_level: int = 0
    num_levels: int = 11
    advance: bool = True
    reset_on_load: bool = False
    # Order: height [m], vx/vy [m/s], yaw [rad/s], pelvis x/y [m].
    initial_lower_ranges: tuple = (
        (-0.10, 0.01),
        (-0.10, 0.10),
        (-0.03, 0.03),
        (-0.10, 0.10),
        (-0.005, 0.005),
        (-0.005, 0.005),
    )
    initial_position_scale: float = 0.50
    initial_orientation_scale: float = 0.50
    initial_height_shift: tuple[float, float] = (-0.05, -0.02)
    position_tolerance_m: float = 0.015
    orientation_tolerance_rad: float = 0.10
    pose_ready_gate: float = 0.80
    failure_rate_limit: float = 0.01
    ema_alpha: float = 0.20
    check_period_steps: int = 1200
    min_episodes_per_class: int = 512


class RangeCurriculum:
    """One global progression level; each running episode keeps its assigned level."""

    CLASS_NAMES = ("regular", "height_shifted")
    AXES = ("height", "vx", "vy", "yaw", "pelvis_x", "pelvis_y")

    def __init__(self, cfg, final_ranges, height_shift, required_classes, num_envs, device):
        self.cfg = cfg
        self.final_ranges = torch.tensor(final_ranges, device=device, dtype=torch.float32)
        self.initial_ranges = torch.tensor(cfg.initial_lower_ranges, device=device, dtype=torch.float32)
        if self.final_ranges.shape != (6, 2) or self.initial_ranges.shape != (6, 2):
            raise ValueError("Lower curriculum ranges must contain six (low, high) pairs")
        for ranges in (self.initial_ranges, self.final_ranges):
            if not torch.isfinite(ranges).all() or (ranges[:, 0] > 0).any() or (ranges[:, 1] < 0).any():
                raise ValueError("Lower ranges must be finite and contain zero")
        if (self.initial_ranges[:, 0] < self.final_ranges[:, 0]).any() or (
            self.initial_ranges[:, 1] > self.final_ranges[:, 1]
        ).any():
            raise ValueError("Initial Lower ranges must be inside final ranges")
        if cfg.num_levels < 2 or not 0 <= cfg.initial_level < cfg.num_levels:
            raise ValueError("Invalid curriculum level count or initial level")
        for value in (cfg.initial_position_scale, cfg.initial_orientation_scale):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Initial target scales must be in (0, 1]")
        if not 0 < cfg.ema_alpha <= 1 or not 0 <= cfg.pose_ready_gate <= 1 or not 0 <= cfg.failure_rate_limit <= 1:
            raise ValueError("Invalid curriculum EMA or gate")
        if cfg.check_period_steps < 1 or cfg.min_episodes_per_class < 1:
            raise ValueError("Curriculum windows and sample counts must be positive")
        if not all(math.isfinite(x) and x > 0 for x in (cfg.position_tolerance_m, cfg.orientation_tolerance_rad)):
            raise ValueError("Pose tolerances must be positive and finite")
        self.initial_shift = torch.tensor(cfg.initial_height_shift, device=device)
        self.final_shift = torch.tensor(height_shift, device=device)
        if any(
            x.shape != (2,) or not torch.isfinite(x).all() or x[0] > x[1]
            for x in (self.initial_shift, self.final_shift)
        ):
            raise ValueError("Invalid curriculum height shifts")
        self.required = torch.tensor(required_classes, device=device, dtype=torch.bool)
        if self.required.shape != (2,) or not self.required.any():
            raise ValueError("At least one curriculum target class is required")
        self.level = cfg.initial_level
        self.episode_levels = torch.full((num_envs,), self.level, device=device, dtype=torch.long)
        self.lower_ranges = self.final_ranges.expand(num_envs, -1, -1).clone()
        self.counts = torch.zeros(2, device=device, dtype=torch.long)
        self.score_sums = torch.zeros(2, device=device)
        self.failures = torch.zeros(2, device=device, dtype=torch.long)
        self.ema = torch.zeros(2, device=device)
        self.initialized = torch.zeros(2, device=device, dtype=torch.bool)
        self.window_counts = self.counts.clone()
        self.window_failure_rate = self.score_sums.clone()
        self.window_scores = self.score_sums.clone()
        self.sample_ready = False
        self.quality_ready = False
        self.failure_ready = False
        self.last_check = 0
        self.checked_level = self.level
        self.clock_offset = 0
        self.evaluation = False
        self.target_contract = None
        self.begin(slice(None))

    def begin(self, ids):
        self.episode_levels[ids] = self.level
        p = self.level / (self.cfg.num_levels - 1)
        self.lower_ranges[ids] = torch.lerp(self.initial_ranges, self.final_ranges, p)

    def restart(self, step):
        """Explicit new experiment: clear progress while retaining the configured contract."""
        self.level = self.cfg.initial_level
        self.checked_level = self.level
        for name in (
            "counts",
            "score_sums",
            "failures",
            "ema",
            "initialized",
            "window_counts",
            "window_failure_rate",
            "window_scores",
        ):
            getattr(self, name).zero_()
        self.clock_offset = -int(step)
        self.last_check = 0
        self.sample_ready = self.quality_ready = self.failure_ready = False
        self.begin(slice(None))

    def target_scale(self, ids, *, orientation=False):
        start = self.cfg.initial_orientation_scale if orientation else self.cfg.initial_position_scale
        return start + (1 - start) * self.episode_levels[ids].float() / (self.cfg.num_levels - 1)

    def height_shift_ranges(self, ids):
        p = self.episode_levels[ids].float()[:, None] / (self.cfg.num_levels - 1)
        return self.initial_shift + p * (self.final_shift - self.initial_shift)

    def complete(self, ids, categories, scores, failed, step):
        """Accept natural completions only; missing pose samples receive quality zero."""
        if self.evaluation or not self.cfg.advance:
            return
        current = self.episode_levels[ids] == self.level
        for c in range(2):
            mask = current & (categories == c)
            self.counts[c] += mask.sum()
            self.score_sums[c] += scores[mask].sum()
            self.failures[c] += failed[mask].sum()
        now = int(step) + self.clock_offset
        if now - self.last_check < self.cfg.check_period_steps:
            return
        self.last_check = now
        self.checked_level = self.level
        self.window_counts.copy_(self.counts)
        self.window_scores.copy_(self.score_sums / self.counts.clamp_min(1))
        self.window_failure_rate.copy_(self.failures / self.counts.clamp_min(1))
        eligible = self.counts >= self.cfg.min_episodes_per_class
        # Update EMA only from sufficiently sampled windows, once per window.
        self.ema[eligible] = torch.where(
            self.initialized[eligible],
            torch.lerp(self.ema[eligible], self.window_scores[eligible], self.cfg.ema_alpha),
            self.window_scores[eligible],
        )
        self.initialized |= eligible
        self.sample_ready = bool(eligible[self.required].all())
        self.quality_ready = bool((self.initialized & (self.ema >= self.cfg.pose_ready_gate))[self.required].all())
        self.failure_ready = bool((self.window_failure_rate <= self.cfg.failure_rate_limit)[self.required].all())
        self.counts.zero_()
        self.score_sums.zero_()
        self.failures.zero_()
        if self.sample_ready and self.quality_ready and self.failure_ready and self.level < self.cfg.num_levels - 1:
            self.level += 1
            self.ema.zero_()
            self.initialized.zero_()

    def logs(self):
        result = {
            "level": float(self.level),
            "sample_ready": float(self.sample_ready),
            "quality_ready": float(self.quality_ready),
            "failure_ready": float(self.failure_ready),
            "checked_level": float(self.checked_level),
        }
        p = self.level / (self.cfg.num_levels - 1)
        ranges = torch.lerp(self.initial_ranges, self.final_ranges, p)
        result["range/position_scale"] = self.cfg.initial_position_scale + p * (1 - self.cfg.initial_position_scale)
        result["range/orientation_scale"] = self.cfg.initial_orientation_scale + p * (
            1 - self.cfg.initial_orientation_scale
        )
        result["range/height_shift_min"], result["range/height_shift_max"] = torch.lerp(
            self.initial_shift, self.final_shift, p
        )
        for i, name in enumerate(self.AXES):
            result[f"range/{name}_min"], result[f"range/{name}_max"] = ranges[i]
        for c, name in enumerate(self.CLASS_NAMES):
            result[f"{name}/ema"] = self.ema[c].clone()
            result[f"{name}/ema_valid"] = self.initialized[c].clone().float()
            result[f"{name}/window_episodes"] = self.window_counts[c].clone().float()
            result[f"{name}/window_failure_rate"] = self.window_failure_rate[c].clone()
            result[f"{name}/window_pose_ready_fraction"] = self.window_scores[c].clone()
        return {f"Curriculum/upper_ranges/{k}": v for k, v in result.items()}

    def contract(self):
        cfg = asdict(self.cfg)
        for k in ("advance", "reset_on_load"):
            cfg.pop(k)
        contract = {
            "cfg": cfg,
            "final_ranges": self.final_ranges.cpu().tolist(),
            "height_shift": self.final_shift.cpu().tolist(),
            "required": self.required.cpu().tolist(),
            "target": self.target_contract,
        }
        return json.loads(json.dumps(contract))  # Treat Hydra lists and Python tuples identically.

    def state_dict(self, step):
        return {
            "schema_version": 1,
            "contract": self.contract(),
            "level": self.level,
            "control_steps": int(step) + self.clock_offset,
            "last_check": self.last_check,
            "checked_level": self.checked_level,
            **{
                k: getattr(self, k).detach().cpu().clone()
                for k in (
                    "counts",
                    "score_sums",
                    "failures",
                    "ema",
                    "initialized",
                    "window_counts",
                    "window_failure_rate",
                    "window_scores",
                )
            },
        }

    def load_state_dict(self, state, step):
        if state.get("schema_version") != 1 or state.get("contract") != self.contract():
            raise ValueError("Upper range curriculum differs from checkpoint; use range_curriculum.reset_on_load=true")
        level = state["level"]
        if not isinstance(level, int) or not 0 <= level < self.cfg.num_levels:
            raise ValueError("Invalid saved Upper curriculum level")
        buffers = {}
        for k in (
            "counts",
            "score_sums",
            "failures",
            "ema",
            "initialized",
            "window_counts",
            "window_failure_rate",
            "window_scores",
        ):
            value = torch.as_tensor(state[k], device=self.counts.device)
            if value.shape != (2,) or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Invalid saved Upper curriculum {k}")
            buffers[k] = value
        if (
            (buffers["ema"] > 1).any()
            or (buffers["score_sums"] > buffers["counts"] + 1e-4).any()
            or (buffers["failures"] > buffers["counts"]).any()
        ):
            raise ValueError("Invalid saved Upper curriculum counts or quality")
        if not 0 <= int(state["last_check"]) <= int(state["control_steps"]):
            raise ValueError("Invalid saved Upper curriculum clock")
        self.level = level
        self.checked_level = int(state["checked_level"])
        for k, value in buffers.items():
            getattr(self, k).copy_(value)
        self.clock_offset = int(state["control_steps"]) - int(step)
        self.last_check = int(state["last_check"])
        self.begin(slice(None))
