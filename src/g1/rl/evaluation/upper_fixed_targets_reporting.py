"""Aggregate Upper metrics across complete seed groups."""

from statistics import fmean, pstdev

from .upper_fixed_targets import ARM_MODES, episode_specs

GROUP_FIELDS = ("module", "policy_output", "condition_id", "axis_step")
NOTES = {
    "torso": "Per-case mean of absolute world-frame torso_link Roll and Pitch over executed control steps, in degrees.",
    "wrist": "Mean Euclidean position error in mm over active wrists and the fixed end window. "
    "Only full windows enter the report; partial windows remain in raw cases.",
    "aggregation": "Cases have equal weight within each row. Standard deviations are across cases.",
    "window_completion_rate": "Fraction of assigned cases with all planned end-window samples.",
    "window_coverage_fraction": "Fraction of all assigned end-window samples actually observed.",
}


def summarize_upper(benchmark, groups):
    rows = []
    for group in groups:
        first = group[0]
        episodes = []
        for case in group:
            expected = episode_specs(benchmark, case)
            records = case["episodes"]
            if len(records) != len(expected) or any(
                any(record.get(key) != value for key, value in spec.items())
                for spec, record in zip(expected, records, strict=True)
            ):
                raise ValueError(f"Upper evaluation cases differ from the suite for seed {case['seed']}")
            episodes.extend(records)

        for mode in ("aggregate", *ARM_MODES):
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
            rows.append(row)
    return rows
