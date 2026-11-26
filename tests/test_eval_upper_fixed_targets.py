"""Coverage for target replay and Upper report aggregation without physics."""

import json

import pytest

from g1.rl.evaluation.config import load_suite, validate_suite
from g1.rl.evaluation.modules import benchmark_for_job, conditions
from g1.rl.evaluation.reporting import summarize_results, write_report
from g1.rl.evaluation.upper_fixed_targets import ARM_MODES, episode_specs


def test_targets_are_replayed_across_models_and_batch_sizes():
    config = load_suite("g1_upper_fixed_targets")
    b = config["benchmark"]
    assert b["target_ranges_m"] == {
        "near": {"x": [-0.05, 0.15], "y": [-0.05, 0.1], "z": [-0.2, 0.2]},
        "far": {"x": [-2.5, 2.5], "y": [-2.5, 2.5], "z": [-0.2, 0.2]},
    }
    for condition in conditions(config):
        job = {**condition, "seed": 42, "checkpoint": "baseline"}
        planned = benchmark_for_job(config, job, {"num_envs": 11})
        other = {**job, "checkpoint": "candidate"}
        assert planned == benchmark_for_job(config, other, {"num_envs": 128})
        specs = episode_specs(planned, job)
        assert specs == episode_specs(planned, other)
        assert len({spec["case_id"] for spec in specs}) == b["episodes_per_seed"]
        assert tuple(spec["arm_mode"] for spec in specs[:3]) == ARM_MODES
        bounds = list(b["target_ranges_m"][condition["condition_id"]].values())
        for spec in specs:
            assert all(lo <= value <= hi for value, (lo, hi) in zip(spec["offset_m"], bounds, strict=True))
            assert spec["deadline_step"] - spec["evaluation_start_step"] == 120
            assert spec["deadline_step"] * planned["control_dt"] <= planned["episode_length_s"]
        assert episode_specs(planned, {**job, "seed": 43})[0]["offset_m"] != specs[0]["offset_m"]


def test_override_range_and_reject_reversed_bounds():
    config = load_suite("g1_upper_fixed_targets")
    config["benchmark"]["target_ranges_m"]["far"]["z"] = [0.1, 0.1]
    config = validate_suite(config)
    job = {**conditions(config)[1], "seed": 42}
    assert {spec["offset_m"][2] for spec in episode_specs(config["benchmark"], job)} == {0.1}
    config["benchmark"]["target_ranges_m"]["far"]["z"] = [0.2, -0.2]
    with pytest.raises(ValueError, match="target_ranges_m.far.z.*reversed bounds"):
        validate_suite(config)


def test_upper_report_accounts_for_failed_and_partial_windows(tmp_path):
    config = load_suite("g1_upper_fixed_targets")
    config["benchmark"]["episodes_per_seed"] = 3
    job = {
        **conditions(config)[0],
        "seed": 42,
        "checkpoint": {"label": "upper", "path": "upper.pt", "sha256": "test"},
        "policy_output": "mean",
    }
    records = [
        {
            **spec,
            "failed": samples < 120,
            "roll_abs_mean_deg": index,
            "pitch_abs_mean_deg": 3 * index,
            "torso_angle_mean_deg": 2 * index,
            "wrist_error_mm": error,
            "window_steps": samples,
            "expected_window_steps": 120,
            "window_complete": samples == 120,
        }
        for index, (spec, samples, error) in enumerate(
            zip(episode_specs(config["benchmark"], job), [120, 30, 0], [12.0, 1.0, None], strict=True)
        )
    ]
    result = {**job, "schema_version": 1, "episodes": records}
    rows = summarize_results(config, [result])
    assert rows[0]["mode"] == "aggregate"
    assert rows[0]["torso_angle_mean_deg"] == 2
    assert rows[0]["wrist_error_mm"] == 12
    assert rows[0]["failure_rate"] == pytest.approx(2 / 3)
    assert rows[0]["window_completion_rate"] == pytest.approx(1 / 3)
    assert rows[0]["window_coverage_fraction"] == pytest.approx(150 / 360)
    assert rows[2]["wrist_error_mm"] is None
    report = write_report(tmp_path, config, [result])
    with open(report["summary_json"], encoding="utf-8") as stream:
        assert json.load(stream)["rows"] == rows
