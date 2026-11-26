"""Push-only execution reuses the four-stage cases, records and statistical definitions."""

import copy
import json
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from g1.rl.evaluation import cli
from g1.rl.evaluation.config import load_suite, suite_config_digest, validate_suite
from g1.rl.evaluation.four_stage import benchmark_for_job, protocol_plan, protocol_specs
from g1.rl.evaluation.modules import conditions
from g1.rl.evaluation.reporting import summarize_results, write_report


def test_push_suite_defaults_one_condition_and_budget():
    config = load_suite("g1_lower_push")
    assert validate_suite(config) == config
    assert config["stages"] == ["squat_push"]
    assert [condition["module"] for condition in conditions(config)] == ["squat_push"]
    benchmark = benchmark_for_job(config, conditions(config)[0], {"num_envs": 256})
    assert benchmark["push_velocity_range"] == {"x": [-0.7, 0.7], "y": [-0.7, 0.7]}
    assert benchmark["episode_length_s"] == 12
    assert benchmark["episodes_per_seed"] == 2560
    assert benchmark["push_height_offsets"] == [-0.25, -0.075, 0]
    assert all(benchmark["ranges"][key] == [0, 0] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"))
    assert {spec["mode"] for spec in protocol_specs(benchmark)} == {1}
    with pytest.raises(ValueError, match="not selected"):
        benchmark_for_job(config, {"module": "walk_axes"}, {"num_envs": 256})


@pytest.mark.parametrize("stages", [[], ["squat_push", "squat_push"], ["unknown"], [True], "squat_push"])
def test_invalid_stage_selection_fails_before_launch(stages):
    config = load_suite("g1_lower_push")
    config["stages"] = stages
    with pytest.raises(ValueError, match="stages"):
        validate_suite(config)


def test_push_accepts_one_environment_at_one_standing_height():
    config = load_suite("g1_lower_push")
    config["benchmark"]["ranges"]["height_offset"] = [0, 0]
    config["benchmark"]["push_height_offsets"] = [0]
    config = validate_suite(config)
    benchmark = benchmark_for_job(config, conditions(config)[0], {"num_envs": 1})
    assert len(protocol_specs(benchmark)) == 10
    assert all(spec["height_offset_m"] == 0 for spec in protocol_specs(benchmark))
    with pytest.raises(ValueError, match="num_envs"):
        benchmark_for_job(load_suite("g1_lower_push"), {"module": "squat_push"}, {"num_envs": 2})


def test_stage_selection_preserves_existing_suite_shape_and_cases():
    full = load_suite("g1_lower_four_stage")
    assert "stages" not in validate_suite(full)
    selected = copy.deepcopy(full)
    selected["stages"] = ["squat_push"]
    selected = validate_suite(selected)
    assert suite_config_digest(full) != suite_config_digest(selected)
    full_benchmark = benchmark_for_job(full, {"module": "squat_push"}, {"num_envs": 7})
    push_benchmark = benchmark_for_job(selected, {"module": "squat_push"}, {"num_envs": 7})
    assert full_benchmark == push_benchmark
    assert protocol_specs(full_benchmark) == protocol_specs(push_benchmark)
    for spec in protocol_specs(push_benchmark):
        assert protocol_plan(full_benchmark, 101, spec) == protocol_plan(push_benchmark, 101, spec)


def test_push_report_has_same_fields_and_values_as_four_stage(tmp_path):
    from test_eval_four_stage import _case, _config

    full = _config()
    selected = copy.deepcopy(full)
    selected["stages"] = ["squat_push"]
    result = _case(full, "squat_push")
    for i, episode in enumerate(result["episodes"]):
        episode["push_count"] = 2
        episode["pushes"] = [{"recovery_s": 0.25}, {"recovery_s": None if i == 0 else 0.3}]
    expected = summarize_results(full, [result])
    assert summarize_results(selected, [result]) == expected
    report = write_report(tmp_path, selected, [result])
    document = json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))
    aggregate = next(row for row in document["rows"] if row["mode"] == "aggregate")
    assert aggregate["push_count"] == 6
    assert aggregate["push_recovery_rate"] == pytest.approx(5 / 6)
    assert aggregate["mean_recovery_s"] == pytest.approx((3 * 0.25 + 2 * 0.3) / 5)
    assert len([row for row in document["rows"] if row["mode"] == "height"]) == 3
    assert {row["module"] for row in document["rows"]} == {"squat_push"}
    events = EventAccumulator(str(Path(report["tensorboard_dir"]) / "squat_push/mean/model"))
    events.Reload()
    assert events.Scalars("00 四阶段指标/aggregate/push_recovery_rate")[0].value == pytest.approx(5 / 6)


def test_cli_plans_only_push_with_overrides_and_resumes_report(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"plan only")
    output = tmp_path / "push_plan"
    assert (
        cli.run(
            [
                "--suite",
                "g1_lower_push",
                "--checkpoints",
                str(checkpoint),
                "--num_envs",
                "1",
                "--repetitions",
                "2",
                "--height-range",
                "0",
                "0",
                "--push-heights",
                "0",
                "--episode-length-s",
                "8",
                "--output",
                str(output),
                "--plan-only",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) == 1
    job = manifest["jobs"][0]
    assert job["module"] == "squat_push"
    benchmark = benchmark_for_job(manifest["config"], job, manifest["runtime"])
    assert benchmark["episodes_per_seed"] == 2
    assert benchmark["episode_length_s"] == 8
    # A plan with no results can still be inspected; it cannot generate a completed report.
    assert cli.run(["--resume", str(output), "--report-only", "--progress", "off"]) == 1
