"""Strict, portable benchmark definitions shared by all evaluation modules.

This module deliberately avoids simulator imports: suite validation and job planning
must work before Isaac Sim starts, including on a machine used only to review results.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from ..task_registry import require_lower_limb_task

MODE_NAMES = ("stand", "squat", "walk", "squat_walk")
RANGE_FIELDS = ("height_offset", "lin_vel_x", "lin_vel_y", "ang_vel_z", "pelvis_offset_x", "pelvis_offset_y")
THRESHOLD_FIELDS = ("height_m", "velocity_xy_mps", "yaw_radps", "pelvis_xy_m")
SUITES_DIRECTORY = Path(__file__).with_name("suites")


def _mapping(value, path, required, optional=()):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    unknown = set(value) - set(required) - set(optional)
    missing = set(required) - set(value)
    if unknown:
        raise ValueError(f"Unknown fields in {path}: {', '.join(sorted(map(str, unknown)))}")
    if missing:
        raise ValueError(f"Missing fields in {path}: {', '.join(sorted(missing))}")
    return value


def _number(value, path, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{path} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    if minimum is not None and result < minimum or maximum is not None and result > maximum:
        raise ValueError(f"{path} is outside its supported range")
    if positive and result <= 0:
        raise ValueError(f"{path} must be positive")
    return result


def _integer(value, path, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return value


def _sequence(value, path, *, length=None):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    if length is not None and len(value) != length:
        raise ValueError(f"{path} must contain {length} values")
    if not value:
        raise ValueError(f"{path} must not be empty")
    return value


def _bounds(value, path, *, positive=False):
    values = _sequence(value, path, length=2)
    lo, hi = (_number(item, path, positive=positive) for item in values)
    if lo > hi:
        raise ValueError(f"{path} has reversed bounds")
    return [lo, hi]


def fraction_axis_step(value, path="disturbance fraction") -> int:
    """Encode a fraction as an exact integer per mille TensorBoard step."""
    fraction = _number(value, path, minimum=0, maximum=1)
    step = round(fraction * 1000)
    if not math.isclose(fraction * 1000, step, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{path} must use increments of 0.001")
    return step


def mode_episode_quotas(episodes_per_seed: int, mode_weights: Sequence[float]) -> list[int]:
    """Allocate exact per-mode quotas using largest remainders, with stable ties.

    Zero-weight modes receive no episodes. Every positive-weight mode must receive
    at least one episode; otherwise increase the budget or remove that mode.
    """
    count = _integer(episodes_per_seed, "benchmark.episodes_per_seed", minimum=1, maximum=2**31 - 1)
    weights = [
        _number(value, "benchmark.mode_weights", minimum=0)
        for value in _sequence(mode_weights, "benchmark.mode_weights", length=4)
    ]
    maximum = max(weights)
    if maximum == 0:
        raise ValueError("benchmark.mode_weights must have a positive sum")
    # Scaling avoids overflow when valid finite weights have a nonfinite sum.
    scaled = [weight / maximum for weight in weights]
    total = math.fsum(scaled)
    exact = [count * weight / total for weight in scaled]
    quotas = [math.floor(value) for value in exact]
    order = sorted(range(4), key=lambda index: (-(exact[index] - quotas[index]), index))
    for index in order[: count - sum(quotas)]:
        quotas[index] += 1
    if any(weight > 0 and quota == 0 for weight, quota in zip(weights, quotas, strict=True)):
        raise ValueError("Episode budget gives a positive-weight mode zero episodes; increase episodes_per_seed")
    return quotas


def _benchmark(value):
    path = "benchmark"
    fields = (
        "episode_length_s",
        "episodes_per_seed",
        "mode_weights",
        "seeds",
        "observation_noise",
        "ranges",
        "resampling_time_range",
        "disturbance_interval_s",
        "thresholds",
        "control_dt",
        "envelope",
        "pelvis_target_speed_mps",
        "pelvis_zero_probability",
        "reset_mode",
    )
    raw = _mapping(value, path, fields, optional=("upper_body_motion", "upper_body_target_source"))
    target_source = raw.get("upper_body_target_source", "final_goal")
    if target_source not in ("final_goal", "reference"):
        raise ValueError("benchmark.upper_body_target_source must be 'final_goal' or 'reference'")
    episodes = _integer(raw["episodes_per_seed"], f"{path}.episodes_per_seed", minimum=1, maximum=2**31 - 1)
    weights = [
        _number(item, f"{path}.mode_weights", minimum=0)
        for item in _sequence(raw["mode_weights"], f"{path}.mode_weights", length=4)
    ]
    mode_episode_quotas(episodes, weights)
    maximum = max(weights)
    scaled_weights = [weight / maximum for weight in weights]
    total_weight = math.fsum(scaled_weights)
    weights = [weight / total_weight for weight in scaled_weights]
    seeds = [_integer(seed, f"{path}.seeds", maximum=2**31 - 1) for seed in _sequence(raw["seeds"], f"{path}.seeds")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark.seeds must be unique")
    if not isinstance(raw["observation_noise"], bool):
        raise ValueError("benchmark.observation_noise must be boolean")
    if raw["observation_noise"]:
        raise ValueError(
            "Evaluation schema v1 supports observation_noise: false; shared noise replay is not implemented"
        )
    ranges = {
        name: _bounds(bounds, f"{path}.ranges.{name}")
        for name, bounds in _mapping(raw["ranges"], f"{path}.ranges", RANGE_FIELDS).items()
    }
    if ranges["height_offset"][0] >= 0:
        raise ValueError("benchmark.ranges.height_offset lower bound must be negative for the velocity envelope")
    for name in ("pelvis_offset_x", "pelvis_offset_y"):
        if not ranges[name][0] <= 0 <= ranges[name][1]:
            raise ValueError(f"benchmark.ranges.{name} must contain zero")
    thresholds = {
        name: _number(item, f"{path}.thresholds.{name}", positive=True)
        for name, item in _mapping(raw["thresholds"], f"{path}.thresholds", THRESHOLD_FIELDS).items()
    }
    duration = _number(raw["episode_length_s"], f"{path}.episode_length_s", positive=True)
    control_dt = _number(raw["control_dt"], f"{path}.control_dt", positive=True)
    horizon = duration / control_dt
    if not math.isfinite(horizon) or horizon < 1 or not math.isclose(horizon, round(horizon), rel_tol=0, abs_tol=1e-8):
        raise ValueError("benchmark.episode_length_s must be an integer number of control_dt steps")
    envelope = _mapping(raw["envelope"], f"{path}.envelope", ("depth_frac", "vx_floor", "wz_floor"))
    if raw["reset_mode"] != "nominal":
        raise ValueError("benchmark.reset_mode currently supports only 'nominal'")
    return {
        "episode_length_s": duration,
        "control_dt": control_dt,
        "upper_body_motion": _motion(raw.get("upper_body_motion", {})),
        "upper_body_target_source": target_source,
        "episodes_per_seed": episodes,
        "mode_weights": weights,
        "seeds": seeds,
        "observation_noise": raw["observation_noise"],
        "ranges": ranges,
        "resampling_time_range": _bounds(raw["resampling_time_range"], f"{path}.resampling_time_range", positive=True),
        "disturbance_interval_s": _bounds(
            raw["disturbance_interval_s"], f"{path}.disturbance_interval_s", positive=True
        ),
        "thresholds": thresholds,
        "envelope": {
            "depth_frac": _number(envelope["depth_frac"], f"{path}.envelope.depth_frac", positive=True, maximum=1),
            "vx_floor": _number(envelope["vx_floor"], f"{path}.envelope.vx_floor", minimum=0),
            "wz_floor": _number(envelope["wz_floor"], f"{path}.envelope.wz_floor", minimum=0),
        },
        "pelvis_target_speed_mps": _number(
            raw["pelvis_target_speed_mps"], f"{path}.pelvis_target_speed_mps", positive=True
        ),
        "pelvis_zero_probability": _number(
            raw["pelvis_zero_probability"], f"{path}.pelvis_zero_probability", minimum=0, maximum=1
        ),
        "reset_mode": raw["reset_mode"],
    }


def validate_suite(raw: Mapping) -> dict:
    """Return a detached, normalized JSON-compatible suite or raise ``ValueError``."""
    raw = _mapping(
        raw, "suite config", ("schema_version", "suite", "task", "benchmark", "modules"), ("policy_outputs",)
    )
    if _integer(raw["schema_version"], "schema_version") != 1:
        raise ValueError("Unsupported evaluation schema_version; expected 1")
    name = raw["suite"]
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError("suite must be a nonempty identifier containing letters, digits, underscores, dots or hyphens")
    task = raw["task"]
    if not isinstance(task, str) or not task.strip() or task != task.strip():
        raise ValueError("task must be a nonempty task identifier without surrounding whitespace")
    require_lower_limb_task("eval", task)
    outputs = list(_sequence(raw.get("policy_outputs", ["mean"]), "policy_outputs"))
    if any(not isinstance(item, str) or item not in ("mean", "sample") for item in outputs):
        raise ValueError("policy_outputs accepts only 'mean' and 'sample'")
    if len(set(outputs)) != len(outputs):
        raise ValueError("policy_outputs must be unique")
    modules = []
    names = set()
    for index, module in enumerate(_sequence(raw["modules"], "modules")):
        path = f"modules[{index}]"
        if not isinstance(module, Mapping) or module.get("name") not in ("tracking", "upper_body_sweep"):
            raise ValueError(f"{path}.name must be 'tracking' or 'upper_body_sweep'")
        name_value = module["name"]
        if name_value in names:
            raise ValueError(f"Duplicate evaluation module {name_value!r}")
        names.add(name_value)
        if name_value == "tracking":
            _mapping(module, path, ("name", "arm_fraction", "waist_fraction"))
            normalized = {
                "name": name_value,
                "arm_fraction": fraction_axis_step(module["arm_fraction"], f"{path}.arm_fraction") / 1000,
            }
        else:
            _mapping(module, path, ("name", "arm_fractions", "waist_fraction"))
            steps = [
                fraction_axis_step(item, f"{path}.arm_fractions")
                for item in _sequence(module["arm_fractions"], f"{path}.arm_fractions")
            ]
            if len(set(steps)) != len(steps):
                raise ValueError(f"{path}.arm_fractions must be unique")
            normalized = {"name": name_value, "arm_fractions": [step / 1000 for step in sorted(steps)]}
        normalized["waist_fraction"] = fraction_axis_step(module["waist_fraction"], f"{path}.waist_fraction") / 1000
        modules.append(normalized)
    return {
        "schema_version": 1,
        "suite": name,
        "task": task,
        "benchmark": _benchmark(raw["benchmark"]),
        "policy_outputs": outputs,
        "modules": modules,
    }


def _motion(value):
    path = "benchmark.upper_body_motion"
    fields = ("max_velocity", "max_acceleration", "min_duration_s")
    raw = _mapping(value, path, (), fields)
    return {
        name: _number(raw.get(name, default), f"{path}.{name}", positive=True)
        for name, default in (("max_velocity", 2.0), ("max_acceleration", 8.0), ("min_duration_s", 0.5))
    }


def load_suite(name_or_path: str | Path) -> dict:
    """Load a built-in suite name or an explicit YAML/JSON file and validate it."""
    value = str(name_or_path)
    if not value:
        raise ValueError("Evaluation suite name/path must not be empty")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        builtin = SUITES_DIRECTORY / f"{value}.yaml"
        path = builtin if builtin.is_file() else Path(value)
    else:
        path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation suite not found: {name_or_path}")
    try:
        with path.open(encoding="utf-8-sig") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid evaluation suite YAML in {path}: {error}") from error
    return validate_suite(raw)


def suite_config_digest(config: Mapping) -> str:
    """Hash normalized benchmark content independent of YAML formatting/key order."""
    payload = json.dumps(validate_suite(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
