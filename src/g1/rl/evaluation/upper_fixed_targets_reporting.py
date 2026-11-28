"""Aggregate Upper metrics across complete seed groups."""

from statistics import fmean, pstdev

from .upper_fixed_targets import ARM_MODES, episode_specs
from .upper_q import PROTOCOL as Q_PROTOCOL
from .upper_q import SIDES
from .upper_q import episode_specs as q_episode_specs

GROUP_FIELDS = ("module", "policy_output", "condition_id", "axis_step")
NOTES = {
    "torso": "Per-case mean of absolute world-frame torso_link Roll and Pitch over executed control steps, in degrees.",
    "wrist": "Mean Euclidean position error in mm over active wrists and the fixed end window. "
    "Only full windows enter the report; partial windows remain in raw cases.",
    "aggregation": "Cases have equal weight within each row. Standard deviations are across cases.",
    "window_completion_rate": "Fraction of assigned cases with all planned end-window samples.",
    "window_coverage_fraction": "Fraction of all assigned end-window samples actually observed.",
}
Q_NOTES = {
    **NOTES,
    "q": "Per-frame pelvis-frame lookup on the shared 0 kg map. Right wrist is mirrored on Y. "
    "Q means use successful, full, steady windows only; per-wrist means use valid frames, "
    "and bilateral means use frames where both wrists are valid. Missing values are null.",
    "q_coverage": "Mean per-episode fraction of valid frames across complete windows, including nonsteady cases.",
    "steady": "A successful full window is steady when both world-frame wrist linear speed RMS values meet the limit.",
    "pose": "Per-wrist position/orientation means use steady windows; wrist_error_mm still uses all full windows.",
}


def _mean_known(episodes, key):
    values = [value for episode in episodes if (value := episode[key]) is not None]
    return fmean(values) if values else None


def _summarize_q(row, metadata, episodes, windows):
    steady = [record for record in windows if not record["failed"] and record["wrists_steady"]]
    row.update(
        workspace_q_map_sha256=metadata["workspace_q_map_sha256"],
        workspace_q_reference=metadata["workspace_q_reference"],
        steady_window_count=len(steady),
        steady_window_rate=len(steady) / len(episodes),
    )
    for side in SIDES:
        for metric in ("error_mm", "orientation_error_rad"):
            key = f"wrist_{side}_{metric}"
            row[key] = _mean_known(steady, key)
        key = f"wrist_{side}_speed_rms_mps"
        row[key] = _mean_known(windows, key)
    for kind in ("actual", "target"):
        for side in (*SIDES, "both"):
            key = f"wrist_{side}_q_{kind}"
            row[key] = _mean_known(steady, key)
            row[f"{key}_raw"] = _mean_known(steady, f"{key}_raw")
            row[f"{key}_coverage"] = _mean_known(windows, f"{key}_coverage")


def summarize_upper(benchmark, groups):
    rows = []
    for group in groups:
        first = group[0]
        q_eval = first["module"] == Q_PROTOCOL
        metadata = first.get("metadata", {})
        lower = metadata.get("frozen_lower") or {}
        provenance = {}
        if lower:
            provenance.update(frozen_lower_checkpoint=lower["checkpoint"], frozen_lower_sha256=lower["sha256"])
        if height := metadata.get("episode_height_range_m"):
            provenance.update(episode_height_min_m=round(height[0], 6), episode_height_max_m=round(height[1], 6))
        episodes = []
        for case in group:
            expected = (q_episode_specs if q_eval else episode_specs)(benchmark, case)
            records = case["episodes"]
            if len(records) != len(expected) or any(
                any(record.get(key) != value for key, value in spec.items())
                for spec, record in zip(expected, records, strict=True)
            ):
                raise ValueError(f"Upper evaluation cases differ from the suite for seed {case['seed']}")
            episodes.extend(records)

        for mode in ("aggregate",) if q_eval else ("aggregate", *ARM_MODES):
            selected = (
                episodes if mode == "aggregate" else [record for record in episodes if record["arm_mode"] == mode]
            )
            if not selected:
                continue
            windows = [record for record in selected if record["window_complete"]]
            row = {
                "checkpoint": first["checkpoint"]["label"],
                "checkpoint_path": first["checkpoint"]["path"],
                "checkpoint_sha256": first["checkpoint"]["sha256"],
                **provenance,
                **{field: first[field] for field in GROUP_FIELDS},
                "mode": mode,
                "seed_count": len(group),
                "episode_count": len(selected),
                "failure_rate": fmean(record["failed"] for record in selected),
                "window_complete_count": len(windows),
                "window_completion_rate": len(windows) / len(selected),
                "window_coverage_fraction": sum(record["window_steps"] for record in selected)
                / sum(record["expected_window_steps"] for record in selected),
            }
            for key, records in (
                ("roll_abs_mean_deg", selected),
                ("pitch_abs_mean_deg", selected),
                ("torso_angle_mean_deg", selected),
                ("wrist_error_mm", windows),
            ):
                values = [record[key] for record in records]
                row[key] = fmean(values) if values else None
                row[f"{key}_std"] = pstdev(values) if values else None
            if q_eval:
                _summarize_q(row, metadata, selected, windows)
            rows.append(row)
    return rows
