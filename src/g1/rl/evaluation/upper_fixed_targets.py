"""Simulator-free definition of the Upper Mobile fixed-target suite."""

from __future__ import annotations

import math
import random
import re

from .config import _bounds, _integer, _mapping, _number, _sequence

SCHEMA_VERSION = 3
PROTOCOL = "upper_fixed_targets"
TASK = "Template-G1-Upper-Mobile-v0"
CONDITIONS = ("near", "far")
ARM_MODES = ("left_only", "right_only", "both")
CONTROL_DT = 1 / 60
AXES = ("x", "y", "z")


def validate_upper_suite(raw):
    _mapping(raw, "suite", ("schema_version", "suite", "task", "benchmark"), ("policy_outputs",))
    _integer(raw["schema_version"], "schema_version", minimum=SCHEMA_VERSION, maximum=SCHEMA_VERSION)
    name = raw["suite"]
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise ValueError("suite must be a nonempty identifier")
    if raw["task"] != TASK:
        raise ValueError(f"The Upper fixed-target suite requires task {TASK}")
    outputs = list(_sequence(raw.get("policy_outputs", ["mean"]), "policy_outputs"))
    if any(output not in ("mean", "sample") for output in outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("policy_outputs must contain unique entries from mean and sample")

    timing = ("near_episode_s", "settle_s", "evaluation_s", "far_speed_m_s", "far_acceleration_m_s2")
    fields = ("seeds", "episodes_per_seed", "level", "target_ranges_m", *timing)
    b = _mapping(raw["benchmark"], "benchmark", fields)
    seeds = [_integer(s, "benchmark.seeds", maximum=2**32 - 1) for s in _sequence(b["seeds"], "benchmark.seeds")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark.seeds must be unique")
    ranges = _mapping(b["target_ranges_m"], "benchmark.target_ranges_m", CONDITIONS)
    target_ranges = {}
    for name in CONDITIONS:
        path = f"benchmark.target_ranges_m.{name}"
        bounds = _mapping(ranges[name], path, AXES)
        target_ranges[name] = {axis: _bounds(bounds[axis], f"{path}.{axis}") for axis in AXES}
    times = {key: _number(b[key], f"benchmark.{key}", positive=True) for key in timing}
    if _steps(times["near_episode_s"]) <= _steps(times["evaluation_s"]):
        raise ValueError("near_episode_s must leave time before the evaluation window")
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": raw["suite"],
        "task": TASK,
        "policy_outputs": outputs,
        "benchmark": {
            "seeds": seeds,
            "episodes_per_seed": _integer(b["episodes_per_seed"], "benchmark.episodes_per_seed", minimum=1),
            "level": _integer(b["level"], "benchmark.level", maximum=10),
            "target_ranges_m": target_ranges,
            **times,
        },
    }


def _steps(seconds):
    return max(1, math.ceil(seconds / CONTROL_DT - 1e-9))


def _move_time(distance, speed, acceleration):
    # Same rest-to-rest translation model as Upper's goal_timing.py. The suite
    # fixes the benchmark speed so checkpoints always get the same time budget.
    if distance < speed**2 / acceleration:
        return 2 * math.sqrt(distance / acceleration)
    return distance / speed + speed / acceleration


def upper_benchmark_for_job(config, job):
    b = config["benchmark"]
    return {
        **b,
        "protocol": PROTOCOL,
        "condition": job["condition_id"],
        "seed": job["seed"],
        "control_dt": CONTROL_DT,
        "episode_length_s": max(spec["deadline_step"] for spec in episode_specs(b, job)) * CONTROL_DT,
    }


def episode_specs(benchmark, job):
    """One independent random stream per case, invariant to batching and checkpoint."""
    name, seed = job["condition_id"], job["seed"]
    ranges = benchmark["target_ranges_m"][name]
    evaluation_steps = _steps(benchmark["evaluation_s"])
    specs = []
    for index in range(benchmark["episodes_per_seed"]):
        case_id = f"{name}-{seed}-{index:06d}"
        rng = random.Random(f"{PROTOCOL}-v1:{case_id}")
        offset = [rng.uniform(*ranges[axis]) for axis in AXES]
        distance = math.hypot(*offset[:2])
        if name == "near":
            deadline = _steps(benchmark["near_episode_s"])
            start = deadline - evaluation_steps
        else:
            move_s = _move_time(distance, benchmark["far_speed_m_s"], benchmark["far_acceleration_m_s2"])
            start = _steps(move_s + benchmark["settle_s"])
            deadline = start + evaluation_steps
        specs.append(
            {
                "case_id": case_id,
                "arm_mode": ARM_MODES[index % len(ARM_MODES)],
                "offset_m": offset,
                "horizontal_distance_m": distance,
                "evaluation_start_step": start,
                "deadline_step": deadline,
            }
        )
    return specs
