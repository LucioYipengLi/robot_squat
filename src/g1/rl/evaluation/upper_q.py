"""Deterministic dual-wrist cases for the workspace Q evaluation suite."""

import math
import random
import re

from .config import _bounds, _integer, _mapping, _number, _sequence
from .upper_fixed_targets import AXES, CONTROL_DT, _steps

PROTOCOL = "upper_workspace_q"
MODES = ("random", "fixed")
SIDES = ("left", "right")
TIMELINE_WINDOW_S = 2.0


def validate_q_benchmark(raw):
    fields = (
        "modes",
        "seeds",
        "episodes_per_seed",
        "level",
        "episode_length_s",
        "evaluation_s",
        "steady_speed_mps",
        "map_sha256",
        "random_offset_ranges_m",
        "fixed_wrist_poses",
    )
    b = _mapping(raw, "benchmark", fields)
    modes = list(_sequence(b["modes"], "benchmark.modes"))
    if any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("benchmark.modes must contain unique entries from random and fixed")
    seeds = [_integer(s, "benchmark.seeds", maximum=2**32 - 1) for s in _sequence(b["seeds"], "benchmark.seeds")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark.seeds must be unique")
    duration, window = (_number(b[k], f"benchmark.{k}", positive=True) for k in ("episode_length_s", "evaluation_s"))
    if _steps(duration) <= _steps(window):
        raise ValueError("episode_length_s must leave time before the evaluation window")
    sha = b["map_sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("benchmark.map_sha256 must be the SHA-256 of the workspace Q map")
    path = "benchmark.random_offset_ranges_m"
    ranges = _mapping(b["random_offset_ranges_m"], path, AXES)
    path_pose = "benchmark.fixed_wrist_poses"
    poses = _mapping(b["fixed_wrist_poses"], path_pose, SIDES)
    fixed = {}
    for side in SIDES:
        name = f"{path_pose}.{side}"
        fields = ("position_m", "rpy_deg")
        pose = _mapping(poses[side], name, fields)
        fixed[side] = {
            key: [_number(x, f"{name}.{key}") for x in _sequence(pose[key], f"{name}.{key}", length=3)]
            for key in fields
        }
    return {
        "modes": modes,
        "seeds": seeds,
        "episodes_per_seed": _integer(b["episodes_per_seed"], "benchmark.episodes_per_seed", minimum=1),
        "level": _integer(b["level"], "benchmark.level", maximum=10),
        "episode_length_s": duration,
        "evaluation_s": window,
        "steady_speed_mps": _number(b["steady_speed_mps"], "benchmark.steady_speed_mps", positive=True),
        "map_sha256": sha,
        "random_offset_ranges_m": {axis: _bounds(ranges[axis], f"{path}.{axis}") for axis in AXES},
        "fixed_wrist_poses": fixed,
    }


def episode_specs(benchmark, job):
    mode, seed = job["condition_id"], job["seed"]
    deadline = _steps(benchmark["episode_length_s"])
    start = deadline - _steps(benchmark["evaluation_s"])
    count = 1 if mode == "fixed" else benchmark["episodes_per_seed"]
    specs = []
    for index in range(count):
        case_id = f"{mode}-{seed}-{index:06d}"
        spec = {
            "case_id": case_id,
            "arm_mode": "both",
            "evaluation_start_step": start,
            "deadline_step": deadline,
        }
        if mode == "fixed":
            spec["wrist_poses"] = benchmark["fixed_wrist_poses"]
        elif mode == "random":
            rng = random.Random(f"{PROTOCOL}-v1:{case_id}")
            spec["offset_m"] = [rng.uniform(*benchmark["random_offset_ranges_m"][axis]) for axis in AXES]
        else:
            raise ValueError(f"Unknown workspace Q condition: {mode}")
        specs.append(spec)
    return specs


def q_benchmark_for_job(config, job):
    b = config["benchmark"]
    specs = episode_specs(b, job)
    return {
        **b,
        "protocol": PROTOCOL,
        "condition": job["condition_id"],
        "seed": job["seed"],
        "control_dt": CONTROL_DT,
        "episodes_per_seed": len(specs),
    }


def quaternion_xyzw(rpy_deg):
    """Fixed-axis roll, pitch and yaw in degrees, composed Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (math.radians(math.remainder(x, 360.0)) / 2 for x in rpy_deg)
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
