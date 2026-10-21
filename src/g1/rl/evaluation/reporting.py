"""Simulator-free, episode-level evaluation summaries and immutable TensorBoard reports.

Every report is a fresh generation. Point TensorBoard at the returned directory so
resuming an evaluation never appends duplicate scalar steps to an earlier report.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import re
import statistics
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

MODES = ("stand", "squat", "walk", "squat_walk")
_MODE_TITLES = dict(zip(MODES, ("站立", "下蹲", "行走", "蹲行"), strict=True))
_MODE_TITLES.update(aggregate="加权总体", original_three="原三模式")
_THRESHOLDS = {"height_m": 0.03, "velocity_xy_mps": 0.5, "yaw_radps": 0.4, "pelvis_xy_m": 0.02}
_RATES = (
    "joint_success_rate",
    "failure_rate",
    "timeout_rate",
    "height_tracking_success_rate",
    "velocity_xy_tracking_success_rate",
    "yaw_tracking_success_rate",
    "velocity_tracking_success_rate",
)
_QUALITY = (*_RATES, "survival_fraction", "error_height_m", "error_velocity_xy_mps", "error_yaw_radps")
_GROUP_FIELDS = ("module", "policy_output", "condition_id", "axis_step", "arm_fraction", "waist_fraction")
_MOTION_TOTALS = (
    "motion_started",
    "motion_completed",
    "motion_pending_replaced",
    "motion_duration_sum_s",
    "motion_time_s",
)


def _motion_metrics(result, duration):
    result["motion_started_hz"] = result["motion_started"] / duration
    result["motion_fraction"] = result["motion_time_s"] / duration
    result["motion_pending_replaced_per_episode"] = result["motion_pending_replaced"] / result["episode_count"]
    if result["motion_started"]:
        result["motion_reference_completion_rate"] = result["motion_completed"] / result["motion_started"]
        result["motion_mean_duration_s"] = result["motion_duration_sum_s"] / result["motion_started"]


def _configuration(config):
    if dataclasses.is_dataclass(config):
        config = dataclasses.asdict(config)
    if not isinstance(config, dict):
        raise TypeError("Evaluation configuration must be a dictionary or dataclass")
    return config


def _benchmark(config):
    benchmark = dict(config.get("benchmark", config))
    weights = benchmark["mode_weights"]
    if not isinstance(weights, dict):
        benchmark["mode_weights"] = dict(zip(MODES, weights, strict=True))
    return benchmark


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite evaluation value: {label}")
    return value


def _wilson(successes, count):
    """95% binomial Wilson interval; deliberately only applied to per-mode rates."""
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _group_results(config, results):
    seeds = set(config["seeds"])
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    groups, seen = defaultdict(dict), set()
    for result in results:
        if result.get("schema_version") != 1:
            raise ValueError("Unsupported evaluation result schema_version")
        checkpoint = result["checkpoint"]
        key = (checkpoint["label"], checkpoint["path"], checkpoint["sha256"], *(result[x] for x in _GROUP_FIELDS))
        seed = result["seed"]
        if seed not in seeds:
            raise ValueError(f"Unexpected evaluation seed: {seed}")
        if (key, seed) in seen:
            raise ValueError(f"Duplicate evaluation case for seed {seed}: {key}")
        seen.add((key, seed))
        groups[key][seed] = result
    complete, incomplete = [], []
    for key, cases in sorted(groups.items()):
        missing = sorted(seeds - set(cases))
        if missing:
            incomplete.append(
                {
                    "checkpoint": key[0],
                    **dict(zip(_GROUP_FIELDS, key[3:], strict=True)),
                    "completed_seeds": sorted(cases),
                    "missing_seeds": missing,
                }
            )
        else:
            complete.append([cases[seed] for seed in sorted(seeds)])
    return complete, incomplete


def _episode_metrics(episodes, config):
    thresholds = _THRESHOLDS | config.get("thresholds", {})
    horizon = _finite(config["episode_length_s"], "episode_length_s")
    if horizon <= 0:
        raise ValueError("episode_length_s must be positive")
    n = len(episodes)
    if not n:
        raise ValueError("A completed mode must contain at least one episode")
    result = dict.fromkeys(
        (
            *_RATES,
            "survival_fraction",
            "episode_length_s",
            "error_height_m",
            "error_velocity_xy_mps",
            "error_yaw_radps",
        ),
        0.0,
    )
    count_fields = (
        *_MOTION_TOTALS,
        "steps",
        "pelvis_steps",
        "pelvis_error_sum",
        "pelvis_success_steps",
        "disturbance_count",
        "command_height_sum",
        "command_speed_sum",
        "command_yaw_abs_sum",
    )
    totals = dict.fromkeys(count_fields, 0.0)
    reasons = defaultdict(int)
    seen_ids = set()
    for seed, episode in episodes:
        identity = (seed, episode["episode_id"])
        if identity in seen_ids:
            raise ValueError(f"Duplicate episode identity: {identity}")
        seen_ids.add(identity)
        steps = _finite(episode["steps"], "steps")
        duration = _finite(episode["length_s"], "length_s")
        if steps <= 0 or duration <= 0 or duration > horizon + 1e-5:
            raise ValueError("Episode steps and duration must be positive and fit the configured horizon")
        terminated, timeout = bool(episode["terminated"]), bool(episode["time_out"])
        if not (terminated or timeout):
            raise ValueError("Evaluation records must be naturally terminated or timed out")
        height, xy, yaw = (
            _finite(episode[name], name) for name in ("error_height_m", "error_velocity_xy_mps", "error_yaw_radps")
        )
        if min(height, xy, yaw) < 0:
            raise ValueError("Absolute tracking errors must be nonnegative")
        height_ok, xy_ok, yaw_ok = (
            height < thresholds["height_m"],
            xy < thresholds["velocity_xy_mps"],
            yaw < thresholds["yaw_radps"],
        )
        pelvis_ok = True
        if episode["mode"] in MODES[:2]:
            pelvis = _finite(episode["error_pelvis_xy_m"], "error_pelvis_xy_m")
            if pelvis < 0 or episode.get("pelvis_steps", 0) <= 0:
                raise ValueError("Static-mode episodes require valid pelvis samples")
            pelvis_ok = pelvis < thresholds["pelvis_xy_m"]
        values = {
            "joint_success_rate": timeout and not terminated and height_ok and xy_ok and yaw_ok and pelvis_ok,
            "failure_rate": terminated,
            "timeout_rate": timeout and not terminated,
            "height_tracking_success_rate": height_ok,
            "velocity_xy_tracking_success_rate": xy_ok,
            "yaw_tracking_success_rate": yaw_ok,
            "velocity_tracking_success_rate": xy_ok and yaw_ok,
            "survival_fraction": min(duration / horizon, 1.0),
            "episode_length_s": duration,
            "error_height_m": height,
            "error_velocity_xy_mps": xy,
            "error_yaw_radps": yaw,
        }
        for key, value in values.items():
            result[key] += value
        for key in count_fields:
            value = _finite(episode.get(key, 0), key)
            if key != "command_height_sum" and value < 0:
                raise ValueError(f"Nonnegative evaluation total expected: {key}")
            totals[key] += value
        for reason, active in episode.get("reasons", {}).items():
            reasons[reason] += bool(active)
    for name in _RATES:
        lower, upper = _wilson(result[name], n)
        result[name + "_lower_95"], result[name + "_upper_95"] = lower, upper
    result = {key: value / n if key in (*_QUALITY, "episode_length_s") else value for key, value in result.items()}
    result.update(episode_count=n, step_count=totals["steps"], disturbance_count=totals["disturbance_count"])
    result["disturbance_hz"] = totals["disturbance_count"] / (result["episode_length_s"] * n)
    has_motion = ["motion_started" in episode for _, episode in episodes]
    if any(has_motion):
        if not all(has_motion):
            raise ValueError("Mixed motion exposure schemas in one evaluation group")
        result.update({name: totals[name] for name in _MOTION_TOTALS})
        _motion_metrics(result, result["episode_length_s"] * n)
    for source, destination in (
        ("command_height_sum", "height_offset_m"),
        ("command_speed_sum", "command_speed_xy_mps"),
        ("command_yaw_abs_sum", "command_yaw_abs_radps"),
    ):
        result[destination] = totals[source] / totals["steps"]
    if totals["pelvis_steps"]:
        result["error_pelvis_xy_m"] = totals["pelvis_error_sum"] / totals["pelvis_steps"]
        result["pelvis_step_success_rate"] = totals["pelvis_success_steps"] / totals["pelvis_steps"]
    result["pelvis_step_count"] = totals["pelvis_steps"]
    result.update({"termination/" + name: value / n for name, value in reasons.items()})
    return result


def _weighted_metrics(by_mode, weights):
    """Quality uses configured mode weights; exposure and pelvis use actual measured steps."""
    total_weight = sum(weights.values())
    result = {}
    for metric in (*_QUALITY, "episode_length_s"):
        result[metric] = sum(by_mode[mode][metric] * weight for mode, weight in weights.items()) / total_weight
    for metric in ("episode_count", "step_count", "disturbance_count", "pelvis_step_count"):
        result[metric] = sum(by_mode[mode][metric] for mode in weights)
    for metric in ("height_offset_m", "command_speed_xy_mps", "command_yaw_abs_radps"):
        result[metric] = sum(by_mode[mode][metric] * by_mode[mode]["step_count"] for mode in weights)
        result[metric] /= result["step_count"]
    duration = sum(by_mode[mode]["episode_length_s"] * by_mode[mode]["episode_count"] for mode in weights)
    result["disturbance_hz"] = result["disturbance_count"] / duration
    if all("motion_started" in by_mode[mode] for mode in weights):
        for name in _MOTION_TOTALS:
            result[name] = sum(by_mode[mode][name] for mode in weights)
        _motion_metrics(result, duration)
    if result["pelvis_step_count"]:
        for metric in ("error_pelvis_xy_m", "pelvis_step_success_rate"):
            result[metric] = sum(by_mode[mode].get(metric, 0) * by_mode[mode]["pelvis_step_count"] for mode in weights)
            result[metric] /= result["pelvis_step_count"]
    reason_keys = {key for mode in weights for key in by_mode[mode] if key.startswith("termination/")}
    for key in reason_keys:
        result[key] = sum(by_mode[mode].get(key, 0) * weight for mode, weight in weights.items()) / total_weight
    result["worst_mode_joint_success_rate"] = min(by_mode[mode]["joint_success_rate"] for mode in weights)
    return result


def summarize_results(config, results):
    """Return flat per-mode and aggregate rows, excluding groups with missing seeds.

    Per-mode quality errors are episode means. Pelvis errors and target contexts use
    actual valid steps. Aggregate quality is weighted by configured mode proportions.
    Wilson intervals describe pooled binomial rates; seed standard deviations expose
    between-seed variation separately and are omitted for a single seed.
    """
    config = _benchmark(_configuration(config))
    weights = {mode: _finite(weight, "mode_weights") for mode, weight in config["mode_weights"].items() if weight > 0}
    if not weights or set(weights) - set(MODES):
        raise ValueError("mode_weights must contain supported active modes")
    groups, _ = _group_results(config, results)
    expected_quotas = None
    if "episodes_per_seed" in config:
        from .config import mode_episode_quotas

        expected_quotas = dict(
            zip(
                MODES,
                mode_episode_quotas(
                    config["episodes_per_seed"], [config["mode_weights"].get(mode, 0) for mode in MODES]
                ),
                strict=True,
            )
        )
    rows = []
    for cases in groups:
        reference = cases[0]
        identity = {
            "checkpoint": reference["checkpoint"]["label"],
            "checkpoint_path": reference["checkpoint"]["path"],
            "checkpoint_sha256": reference["checkpoint"]["sha256"],
            **{key: reference[key] for key in _GROUP_FIELDS},
        }
        episode_sets = {mode: [] for mode in weights}
        seed_metrics = {}
        for case in cases:
            buckets = {mode: [] for mode in weights}
            episode_ids = set()
            for episode in case["episodes"]:
                episode_identity = (episode["mode"], episode["episode_id"])
                if episode_identity in episode_ids:
                    raise ValueError(f"Duplicate episode identity in seed {case['seed']}: {episode['episode_id']}")
                episode_ids.add(episode_identity)
                if episode["mode"] not in weights:
                    raise ValueError(f"Episode in inactive mode: {episode['mode']}")
                buckets[episode["mode"]].append((case["seed"], episode))
            if expected_quotas:
                for mode, episodes in buckets.items():
                    if len(episodes) != expected_quotas[mode]:
                        raise ValueError(f"Completed case violates episode quota for mode {mode}")
            seed_metrics[case["seed"]] = {
                mode: _episode_metrics(episodes, config) for mode, episodes in buckets.items()
            }
            for mode, episodes in buckets.items():
                episode_sets[mode].extend(episodes)
        # Equal per-seed mode quotas keep pooled point estimates and seed means aligned.
        for mode in weights:
            quotas = {metrics[mode]["episode_count"] for metrics in seed_metrics.values()}
            if len(quotas) != 1:
                raise ValueError(f"Per-seed episode quotas differ for mode {mode}")
        metrics = {mode: _episode_metrics(episodes, config) for mode, episodes in episode_sets.items()}
        total_steps = sum(values["step_count"] for values in metrics.values())
        total_episodes = sum(values["episode_count"] for values in metrics.values())
        aggregate_sets = {"aggregate": weights}
        original = {mode: weight for mode, weight in weights.items() if mode in MODES[:3]}
        if original and set(original) != set(weights):
            aggregate_sets["original_three"] = original
        for name, selected in aggregate_sets.items():
            metrics[name] = _weighted_metrics(metrics, selected)
            for values in seed_metrics.values():
                values[name] = _weighted_metrics(values, selected)
        for mode, values in metrics.items():
            row = identity | {"mode": mode, "seed_count": len(cases), **values}
            row["step_share"] = values["step_count"] / total_steps
            row["episode_share"] = values["episode_count"] / total_episodes
            if mode in weights:
                row["sampling_probability"] = weights[mode] / sum(weights.values())
            if len(cases) > 1:
                for metric in (*_QUALITY, "error_pelvis_xy_m"):
                    samples = [values[mode][metric] for values in seed_metrics.values() if metric in values[mode]]
                    if samples:
                        row[metric + "_seed_std"] = statistics.stdev(samples)
            rows.append(row)
    return rows


def _safe_component(value):
    value = str(value)
    if not value or value in (".", "..") or not re.fullmatch(r"[\w.\-]+", value, flags=re.UNICODE):
        raise ValueError(f"Unsafe report directory component: {value!r}")
    return value


_PANELS = (
    (
        "01 任务完成",
        (
            ("联合成功率 ↑", "joint_success_rate"),
            ("失败率 ↓", "failure_rate"),
            ("存活比例 ↑", "survival_fraction"),
            ("最差模式联合成功率 ↑", "worst_mode_joint_success_rate"),
        ),
    ),
    (
        "02 跟踪精度",
        (
            ("高度误差 (m) ↓", "error_height_m"),
            ("平面速度误差 (m/s) ↓", "error_velocity_xy_mps"),
            ("转向误差 (rad/s) ↓", "error_yaw_radps"),
            ("骨盆水平误差 (m) ↓", "error_pelvis_xy_m"),
        ),
    ),
    (
        "03 失败定位",
        (
            ("高度达标率 ↑", "height_tracking_success_rate"),
            ("平面速度达标率 ↑", "velocity_xy_tracking_success_rate"),
            ("转向达标率 ↑", "yaw_tracking_success_rate"),
            ("速度联合达标率 ↑", "velocity_tracking_success_rate"),
        ),
    ),
    (
        "04 测试条件",
        (
            ("有效回合数", "episode_count"),
            ("控制步数", "step_count"),
            ("实际扰动频率 (Hz)", "disturbance_hz"),
            ("实际运动频率 (Hz)", "motion_started_hz"),
            ("参考到达率", "motion_reference_completion_rate"),
            ("平均计划运动时长 (s)", "motion_mean_duration_s"),
            ("待执行目标替换数 / 回合", "motion_pending_replaced_per_episode"),
            ("双臂目标范围比例", "arm_fraction"),
            ("腰部目标范围比例", "waist_fraction"),
            ("高度目标偏移 (m)", "height_offset_m"),
            ("目标平面速度 (m/s)", "command_speed_xy_mps"),
            ("目标转向绝对值 (rad/s)", "command_yaw_abs_radps"),
        ),
    ),
)


def _write_events(directory, rows):
    # TensorBoard is imported only when a report is actually published; no torch/Kit imports.
    from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
    from tensorboard.plugins.custom_scalar import metadata
    from tensorboard.plugins.custom_scalar.layout_pb2 import Layout
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    runs = defaultdict(list)
    for row in rows:
        run = tuple(_safe_component(row[key]) for key in ("module", "policy_output", "checkpoint"))
        runs[run].append(row)
    for (module, policy, checkpoint), entries in runs.items():
        layout = Layout()
        available = {key for row in entries for key in row}
        modes = [mode for mode in (*MODES, "aggregate", "original_three") if any(x["mode"] == mode for x in entries)]
        axis = "双臂目标范围比例‰" if module == "upper_body_sweep" else "固定场景编号"
        for index, (title, charts) in enumerate(_PANELS):
            category = layout.category.add(title=title, closed=index != 0)
            expanded_charts = list(charts)
            if index == 2:
                expanded_charts.extend(
                    ("终止原因：" + key.split("/", 1)[1], key)
                    for key in sorted(available)
                    if key.startswith("termination/")
                )
                expanded_charts.append(("联合成功率跨种子标准差", "joint_success_rate_seed_std"))
            for chart_title, metric in expanded_charts:
                for mode in modes:
                    if not any(metric in row and row["mode"] == mode for row in entries):
                        continue
                    tag = f"{title}/{mode}/{metric}"
                    category.chart.add(
                        title=f"{_MODE_TITLES[mode]} · {chart_title}（横轴：{axis}）"
                    ).multiline.tag.append("^" + re.escape(tag) + "$")
        serialized = layout.SerializeToString()
        tensor = tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING, string_val=[serialized])
        layout_value = summary_pb2.Summary.Value(
            tag=metadata.CONFIG_SUMMARY_TAG, tensor=tensor, metadata=metadata.create_summary_metadata()
        )
        writer = EventFileWriter(str(directory / module / policy / checkpoint))
        try:
            writer.add_event(
                event_pb2.Event(wall_time=time.time(), step=0, summary=summary_pb2.Summary(value=[layout_value]))
            )
            positions = set()
            for row in sorted(entries, key=lambda value: (value["axis_step"], value["mode"])):
                position = (row["mode"], row["axis_step"])
                if position in positions:
                    raise ValueError(f"Duplicate TensorBoard mode/axis position in {module}/{policy}/{checkpoint}")
                positions.add(position)
                values = []
                for panel, charts in _PANELS:
                    metrics = [metric for _, metric in charts]
                    if panel.startswith("03"):
                        metrics += [key for key in row if key.startswith("termination/")]
                        metrics += [key for key in row if key.endswith("_lower_95") or key.endswith("_upper_95")]
                        metrics += [key for key in row if key.endswith("_seed_std")]
                    for metric in metrics:
                        if metric in row:
                            values.append(
                                summary_pb2.Summary.Value(
                                    tag=f"{panel}/{row['mode']}/{metric}", simple_value=_finite(row[metric], metric)
                                )
                            )
                writer.add_event(
                    event_pb2.Event(
                        wall_time=time.time(), step=int(row["axis_step"]), summary=summary_pb2.Summary(value=values)
                    )
                )
            writer.flush()
        finally:
            writer.close()


def write_report(output_dir, config, results):
    """Publish a new immutable generation and return its report/log/summary paths.

    Only supplied groups containing all configured seeds become curves. Missing groups
    are listed in summary.json. A generation never modifies earlier reports or raw cases.
    """
    config = _configuration(config)
    results = list(results)
    rows = summarize_results(config, results)
    _, incomplete = _group_results(_benchmark(config), results)
    generation = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = Path(output_dir) / "reports" / generation
    directory.mkdir(parents=True, exist_ok=False)
    paths = {
        "report_dir": str(directory),
        "tensorboard_dir": str(directory / "tensorboard"),
        "summary_csv": str(directory / "summary.csv"),
        "summary_json": str(directory / "summary.json"),
    }
    summary = {
        "schema_version": 1,
        "configuration": config,
        "rows": rows,
        "incomplete_groups": incomplete,
        "notes": {
            "quality_aggregation": "configured-mode-weighted",
            "pelvis_and_commands": "valid-step-weighted",
            "confidence_intervals": "per-mode pooled binomial Wilson 95%; seed std reported separately",
        },
    }
    Path(paths["summary_json"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(paths["summary_csv"]).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_events(directory / "tensorboard", rows)
    return paths
