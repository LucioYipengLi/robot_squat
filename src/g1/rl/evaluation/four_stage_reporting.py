"""Pooled squared-error summaries, subgroup walking scores and height-wise push results."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from .four_stage import STAGES, benchmark_for_job, protocol_specs


def _metrics(episodes, benchmark):
    from .reporting import _finite, _wilson

    if not episodes:
        raise ValueError("Four-stage group has no episodes")
    totals = defaultdict(float)
    reasons = defaultdict(int)
    recoveries = []
    peaks = defaultdict(float)
    attitude_valid = all(x.get("attitude_quaternion_order") == "xyzw" for x in episodes)
    for episode in episodes:
        steps = _finite(episode["metric_steps"], "metric_steps")
        if steps <= 0 or steps != episode["steps"] or not steps.is_integer():
            raise ValueError("Squared-error samples must match actual episode steps")
        duration = _finite(episode["length_s"], "length_s")
        if not math.isclose(duration, steps * benchmark["control_dt"], abs_tol=1e-5):
            raise ValueError("Episode duration does not match control steps")
        if duration > benchmark["episode_length_s"] + 1e-5:
            raise ValueError("Episode exceeds configured horizon")
        if not (episode["terminated"] or episode["time_out"]):
            raise ValueError("Four-stage records must end naturally")
        totals["steps"] += steps
        totals["duration"] += duration
        totals["failures"] += bool(episode["terminated"])
        totals["completed"] += bool(episode["time_out"] and not episode["terminated"])
        for axis in ("height", "vx", "vy", "wz"):
            value = _finite(episode[f"{axis}_squared_error_sum"], axis)
            if value < 0:
                raise ValueError("Squared-error sums must be nonnegative")
            totals[axis] += value
        for field in ("max_height_error_m", "max_roll_rad", "max_pitch_rad"):
            if field == "max_roll_rad" and not attitude_valid:
                continue
            value = _finite(episode[field], field)
            if value < 0:
                raise ValueError("Peak errors must be nonnegative")
            peaks[field] = max(peaks[field], value)
        pushes = episode["pushes"]
        expected = 2 if benchmark["stage"] == "squat_push" else 0
        if episode["planned_push_count"] != expected or episode["push_count"] != len(pushes) or len(pushes) > expected:
            raise ValueError("Invalid planned/actual push exposure")
        totals["planned_pushes"] += expected
        totals["pushes"] += len(pushes)
        for push in pushes:
            if attitude_valid and push["recovery_s"] is not None:
                recovery = _finite(push["recovery_s"], "recovery_s")
                if recovery < 0 or recovery > duration:
                    raise ValueError("Invalid push recovery time")
                recoveries.append(recovery)
        for name, active in episode["reasons"].items():
            reasons[name] += bool(active)
    count = len(episodes)
    result = {
        "episode_count": count,
        "step_count": totals["steps"],
        "episode_length_s": totals["duration"] / count,
        "survival_fraction": totals["duration"] / (count * benchmark["episode_length_s"]),
        "failure_rate": totals["failures"] / count,
        "timeout_rate": totals["completed"] / count,
        "planned_push_count": totals["planned_pushes"],
        "push_count": totals["pushes"],
        "attitude_metrics_valid": int(attitude_valid),
        **peaks,
    }
    for axis, unit in (("height", "m"), ("vx", "mps"), ("vy", "mps"), ("wz", "radps")):
        result[f"rmse_{axis}_{unit}"] = math.sqrt(totals[axis] / totals["steps"])
    for name, value in (("failure_rate", totals["failures"]), ("timeout_rate", totals["completed"])):
        result[name + "_lower_95"], result[name + "_upper_95"] = _wilson(value, count)
    result.update({"termination/" + name: value / count for name, value in reasons.items()})
    if attitude_valid:
        result["recovered_push_count"] = len(recoveries)
        result["unrecovered_push_count"] = totals["pushes"] - len(recoveries)
        if totals["pushes"]:
            result["push_recovery_rate"] = len(recoveries) / totals["pushes"]
    if recoveries:
        result["mean_recovery_s"] = statistics.fmean(recoveries)
    return result


def _stage_metrics(episodes, benchmark):
    stage = benchmark["stage"]
    result = _metrics(episodes, benchmark)
    if stage == "walk_axes":
        selected = {}
        for task, axis, field in (
            ("straight", "vx", "lin_vel_x"),
            ("lateral", "vy", "lin_vel_y"),
            ("turn", "wz", "ang_vel_z"),
        ):
            group = [x for x in episodes if x["subtask"] == task]
            stats = _metrics(group, benchmark)
            metric = f"rmse_{axis}_{'radps' if axis == 'wz' else 'mps'}"
            selected[axis] = stats[metric] / max(map(abs, benchmark["ranges"][field]))
            result["group_" + metric] = stats[metric]
        result["e_walk_percent"] = 100 * sum(selected.values()) / 3
    elif stage == "squat_tracking":
        result["e_squat_m"] = result["rmse_height_m"]
    elif stage == "squat_walk_tracking":
        result["e_squat_walk_m"] = result["rmse_height_m"]
    return result


def summarize_four_stage(config, results):
    from .reporting import _group_results

    groups, _ = _group_results(config["benchmark"], results)
    rows = []
    for cases in groups:
        reference = cases[0]
        if reference["module"] not in STAGES:
            raise ValueError("Unexpected four-stage module")
        benchmark = benchmark_for_job(config, reference, {"num_envs": reference["metadata"]["num_envs"]})
        specs = protocol_specs(benchmark)
        expected = {spec["episode_id"]: spec for spec in specs}
        pooled, per_seed = [], []
        for case in cases:
            if case["metadata"]["num_envs"] != benchmark["num_envs"]:
                raise ValueError("Four-stage environment counts differ between seeds")
            episodes = case["episodes"]
            if len(episodes) != len(expected) or len({x["episode_id"] for x in episodes}) != len(expected):
                raise ValueError("Four-stage episode budget/identity mismatch")
            for episode in episodes:
                spec = expected.get(episode["episode_id"])
                if spec is None:
                    raise ValueError("Unknown four-stage episode identity")
                for key in ("repetition", "slot", "subtask", "height_index", "height_offset_m"):
                    if key in spec and episode.get(key) != spec[key]:
                        raise ValueError(f"Four-stage case {key} mismatch")
                if episode["mode"] != ("stand", "squat", "walk", "squat_walk")[spec["mode"]]:
                    raise ValueError("Four-stage case mode mismatch")
            pooled.extend(episodes)
            per_seed.append(episodes)
        identity = {
            "checkpoint": reference["checkpoint"]["label"],
            "checkpoint_path": reference["checkpoint"]["path"],
            "checkpoint_sha256": reference["checkpoint"]["sha256"],
            **{
                key: reference[key]
                for key in ("module", "policy_output", "condition_id", "arm_fraction", "waist_fraction")
            },
            "seed_count": len(cases),
        }
        selections = [("aggregate", 0, None)]
        if benchmark["stage"] == "walk_axes":
            selections += [(task, 0, None) for task in ("straight", "turn", "lateral")]
        elif benchmark["stage"] == "squat_push":
            selections += [("height", i, height) for i, height in enumerate(benchmark["push_height_offsets"])]
        for mode, axis_step, height in selections:

            def select(episodes, mode=mode, height=height):
                if mode == "aggregate":
                    return episodes
                return (
                    [x for x in episodes if x.get("height_offset_m") == height]
                    if mode == "height"
                    else [x for x in episodes if x["subtask"] == mode]
                )

            summarizer = _stage_metrics if mode == "aggregate" else _metrics
            values = summarizer(select(pooled), benchmark)
            if len(per_seed) > 1:
                samples = [summarizer(select(episodes), benchmark) for episodes in per_seed]
                for metric in ("rmse_height_m", "e_walk_percent", "e_squat_m", "e_squat_walk_m", "failure_rate"):
                    if metric in values:
                        values[metric + "_seed_std"] = statistics.stdev(x[metric] for x in samples)
            row = identity | {"mode": mode, "axis_step": axis_step, **values}
            if height is not None:
                row["height_offset_m"] = height
            rows.append(row)
    return rows
