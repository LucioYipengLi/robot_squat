"""Portable definitions and deterministic cases for the four-stage lower benchmark."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re

STAGES = ("walk_axes", "squat_tracking", "squat_walk_tracking", "squat_push")
SUBTASKS = ("straight", "turn", "lateral")
VELOCITY_FIELDS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def validate_four_stage(raw):
    from ..task_registry import require_lower_limb_task
    from .config import _bounds, _integer, _mapping, _motion, _number, _sequence

    _mapping(raw, "suite config", ("schema_version", "suite", "task", "benchmark"), ("policy_outputs", "stages"))
    if _integer(raw["schema_version"], "schema_version") != 2:
        raise ValueError("Four-stage suites require schema_version: 2")
    if not isinstance(raw["suite"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw["suite"]):
        raise ValueError("suite must be a nonempty identifier")
    require_lower_limb_task("eval", raw["task"])
    stages = list(_sequence(raw.get("stages", STAGES), "stages"))
    if any(not isinstance(stage, str) or stage not in STAGES for stage in stages):
        raise ValueError(f"stages must select from {', '.join(STAGES)}")
    if len(set(stages)) != len(stages):
        raise ValueError("stages must be unique")
    outputs = list(_sequence(raw.get("policy_outputs", ["mean"]), "policy_outputs"))
    if any(not isinstance(x, str) or x not in ("mean", "sample") for x in outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("policy_outputs accepts unique mean/sample values")
    b = _mapping(
        raw["benchmark"],
        "benchmark",
        ("repetitions", "episode_length_s", "ranges"),
        (
            "control_dt",
            "seeds",
            "push_height_offsets",
            "push_windows_fraction",
            "push_velocity_range",
            "squat_walk_use_envelope",
            "envelope",
            "recovery",
            "upper_body_motion",
            "upper_body_target_source",
        ),
    )
    dt = _number(b.get("control_dt", 1 / 60), "control_dt", positive=True)
    duration = _number(b["episode_length_s"], "episode_length_s", positive=True)
    horizon = duration / dt
    if not math.isfinite(horizon) or horizon < 6 or not math.isclose(horizon, round(horizon), abs_tol=1e-8, rel_tol=0):
        raise ValueError("episode_length_s must contain at least six whole control steps")
    horizon = round(horizon)
    repetitions = _integer(b["repetitions"], "repetitions", minimum=1, maximum=2**31 - 1)
    walking = "walk_axes" in stages or "squat_walk_tracking" in stages
    range_config = _mapping(
        b["ranges"],
        "ranges",
        ("height_offset", *VELOCITY_FIELDS) if walking else ("height_offset",),
        () if walking else VELOCITY_FIELDS,
    )
    ranges = {
        key: _bounds(range_config.get(key, [0, 0]), "ranges." + key) for key in ("height_offset", *VELOCITY_FIELDS)
    }
    if ("squat_tracking" in stages or "squat_walk_tracking" in stages) and ranges["height_offset"][0] >= 0:
        raise ValueError("height_offset must include negative squat offsets")
    for key in VELOCITY_FIELDS:
        if "walk_axes" in stages and max(map(abs, ranges[key])) == 0:
            raise ValueError(f"ranges.{key} needs a positive normalization scale")
    seeds = [_integer(x, "seeds", maximum=2**31 - 1) for x in _sequence(b.get("seeds", [101]), "seeds")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    lo, hi = ranges["height_offset"]
    upper = min(0, hi)
    heights = [
        _number(x, "push_height_offsets", minimum=lo, maximum=hi)
        for x in _sequence(
            b.get("push_height_offsets", list(dict.fromkeys([lo, (lo + upper) / 2, upper]))), "push_height_offsets"
        )
    ]
    if len(set(heights)) != len(heights):
        raise ValueError("push_height_offsets must be unique")
    windows = [
        _bounds(x, "push_windows_fraction")
        for x in _sequence(
            b.get("push_windows_fraction", [[0.25, 0.4], [0.6, 0.75]]), "push_windows_fraction", length=2
        )
    ]
    if not (0 < windows[0][0] <= windows[0][1] < windows[1][0] <= windows[1][1] < 1):
        raise ValueError("push windows must be ordered, separated and strictly inside the episode")
    step_windows = [(math.ceil(a * horizon), math.floor(z * horizon)) for a, z in windows]
    if any(a > z for a, z in step_windows) or step_windows[0][1] >= step_windows[1][0]:
        raise ValueError("push windows must contain separate executable control steps")
    push_ranges = {
        key: _bounds(value, "push_velocity_range." + key)
        for key, value in _mapping(
            b.get("push_velocity_range", {"x": [-0.1, 0.1], "y": [-0.1, 0.1]}), "push_velocity_range", ("x", "y")
        ).items()
    }
    use_envelope = b.get("squat_walk_use_envelope", False)
    if not isinstance(use_envelope, bool):
        raise ValueError("squat_walk_use_envelope must be boolean")
    envelope = _mapping(
        b.get("envelope", {"depth_frac": 0.8, "vx_floor": 0.2, "wz_floor": 0.3}),
        "envelope",
        ("depth_frac", "vx_floor", "wz_floor"),
    )
    envelope = {
        "depth_frac": _number(envelope["depth_frac"], "depth_frac", positive=True, maximum=1),
        "vx_floor": _number(envelope["vx_floor"], "vx_floor", minimum=0),
        "wz_floor": _number(envelope["wz_floor"], "wz_floor", minimum=0),
    }
    if "squat_walk_tracking" in stages and use_envelope and envelope["depth_frac"] * lo > hi:
        raise ValueError("height range has no overlap with the squat-walk envelope")
    recovery = _mapping(b.get("recovery", {}), "recovery", (), ("height_m", "tilt_rad", "speed_mps", "hold_s"))
    recovery = {
        key: _number(recovery.get(key, default), "recovery." + key, positive=True)
        for key, default in (("height_m", 0.03), ("tilt_rad", 0.1), ("speed_mps", 0.05), ("hold_s", 0.2))
    }
    source = b.get("upper_body_target_source", "final_goal")
    if source not in ("final_goal", "reference"):
        raise ValueError("upper_body_target_source must be final_goal/reference")
    normalized = {
        "schema_version": 2,
        "suite": raw["suite"],
        "task": raw["task"],
        "policy_outputs": outputs,
        "benchmark": {
            "repetitions": repetitions,
            "episode_length_s": duration,
            "control_dt": dt,
            "ranges": ranges,
            "seeds": seeds,
            "push_height_offsets": heights,
            "push_windows_fraction": windows,
            "push_velocity_range": push_ranges,
            "squat_walk_use_envelope": use_envelope,
            "envelope": envelope,
            "recovery": recovery,
            "upper_body_motion": _motion(b.get("upper_body_motion", {})),
            "upper_body_target_source": source,
        },
    }
    # Keep the normalized shape of existing suites/manifests unchanged when omitted.
    if "stages" in raw:
        normalized["stages"] = stages
    return normalized


def benchmark_for_job(config, job, runtime):
    """Resolve the exact per-stage budget before launch; legacy suites pass through."""
    if config["schema_version"] != 2:
        return config["benchmark"]
    benchmark = copy.deepcopy(config["benchmark"])
    stages = config.get("stages", STAGES)
    if job["module"] not in stages:
        raise ValueError("Evaluation job module is not selected by suite stages")
    count = runtime["num_envs"]
    minimum_envs = max(
        3 if "walk_axes" in stages else 1, len(benchmark["push_height_offsets"]) if "squat_push" in stages else 1
    )
    if count < minimum_envs:
        raise ValueError(f"num_envs must be >= {minimum_envs} to cover the selected groups/heights")
    mode = 2 if job["module"] == "walk_axes" else 3 if job["module"] == "squat_walk_tracking" else 1
    benchmark.update(
        protocol="four_stage",
        stage=job["module"],
        num_envs=count,
        episodes_per_seed=count * benchmark["repetitions"],
        mode_weights=[int(i == mode) for i in range(4)],
        observation_noise=False,
        reset_mode="nominal",
        thresholds={"height_m": 0.03, "velocity_xy_mps": 0.5, "yaw_radps": 0.4, "pelvis_xy_m": 0.02},
        pelvis_target_speed_mps=0.02,
        pelvis_zero_probability=1.0,
        resampling_time_range=[1e9, 1e9],
        disturbance_interval_s=[1e9, 1e9],
    )
    benchmark["ranges"].update(pelvis_offset_x=[0, 0], pelvis_offset_y=[0, 0])
    return benchmark


def protocol_specs(benchmark):
    """Each batch is one repetition of all slots, with stable contiguous groups."""
    count, stage = benchmark["num_envs"], benchmark["stage"]
    quotas = [count // 3 + int(i < count % 3) for i in range(3)]
    groups = [name for name, size in zip(SUBTASKS, quotas, strict=True) for _ in range(size)]
    heights = benchmark["push_height_offsets"]
    result = []
    for repetition in range(benchmark["repetitions"]):
        for slot in range(count):
            spec = {"episode_id": repetition * count + slot, "repetition": repetition, "slot": slot}
            if stage == "walk_axes":
                spec.update(mode=2, subtask=groups[slot])
            elif stage == "squat_walk_tracking":
                spec.update(mode=3, subtask="squat_walk")
            elif stage == "squat_tracking":
                spec.update(mode=1, subtask="squat")
            else:
                # Equal contiguous buckets; every configured height appears in each repetition.
                index = min(slot * len(heights) // count, len(heights) - 1)
                spec.update(mode=1, subtask="push", height_offset_m=heights[index], height_index=index)
            result.append(spec)
    return result


def _case_rng(seed, identity, stream):
    digest = hashlib.sha256(f"g1-four-stage-v1:{seed}:{identity}:{stream}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def protocol_plan(benchmark, seed, spec):
    """Case randomness never consumes policy/global RNG state."""
    stage, ranges = benchmark["stage"], benchmark["ranges"]
    rng = _case_rng(seed, f"{stage}:{spec['episode_id']}", "commands")
    horizon = round(benchmark["episode_length_s"] / benchmark["control_dt"])
    zero = (0.0,) * 6
    commands, pushes = {}, {}
    if stage == "walk_axes":
        for step in (0, horizon // 2):
            vx = rng.uniform(*ranges["lin_vel_x"]) if spec["subtask"] in ("straight", "turn") else 0
            vy = rng.uniform(*ranges["lin_vel_y"]) if spec["subtask"] == "lateral" else 0
            wz = rng.uniform(*ranges["ang_vel_z"]) if spec["subtask"] == "turn" else 0
            commands[step] = (0, vx, vy, wz, 0, 0)
    elif stage == "squat_tracking":
        height = rng.uniform(*ranges["height_offset"])
        commands = {0: zero, horizon // 3: (height, 0, 0, 0, 0, 0), 2 * horizon // 3: zero}
    elif stage == "squat_walk_tracking":
        lo, hi = ranges["height_offset"]
        envelope = benchmark["envelope"]
        height = rng.uniform(envelope["depth_frac"] * lo if benchmark["squat_walk_use_envelope"] else lo, hi)
        velocities = [rng.uniform(*ranges[key]) for key in VELOCITY_FIELDS]
        if benchmark["squat_walk_use_envelope"]:
            ramp = min(max(height / lo, 0) / envelope["depth_frac"], 1)
            for index, key, floor in ((0, "lin_vel_x", "vx_floor"), (2, "ang_vel_z", "wz_floor")):
                scale_floor = min(envelope[floor] / ranges[key][1], 1) if ranges[key][1] > 0 else 1
                velocities[index] *= 1 - (1 - scale_floor) * ramp
        commands[0] = (height, *velocities, 0, 0)
    else:
        commands[0] = (spec["height_offset_m"], 0, 0, 0, 0, 0)
        # Same repetition shares both pushes across heights and slots for a paired comparison.
        push_rng = _case_rng(seed, spec["repetition"], "pushes")
        for lo, hi in benchmark["push_windows_fraction"]:
            step = push_rng.randint(math.ceil(lo * horizon), math.floor(hi * horizon))
            pushes[step] = tuple(push_rng.uniform(*benchmark["push_velocity_range"][key]) for key in ("x", "y"))
    return commands, pushes
