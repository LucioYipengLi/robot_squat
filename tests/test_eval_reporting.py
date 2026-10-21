"""Evaluation reductions and real TensorBoard events without importing the simulator."""

import copy
import json
import math
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.custom_scalar import metadata
from tensorboard.plugins.custom_scalar.layout_pb2 import Layout

from g1.rl.evaluation.reporting import summarize_results, write_report


def _config(seeds=(11,), weights=(1, 0, 0, 0)):
    return {
        "name": "test_v1",
        "benchmark": {
            "seeds": list(seeds),
            "mode_weights": list(weights),
            "episode_length_s": 6.0,
            "thresholds": {"height_m": 0.03, "velocity_xy_mps": 0.5, "yaw_radps": 0.4, "pelvis_xy_m": 0.02},
        },
    }


def _episode(identity=0, mode="stand", **kwargs):
    record = {
        "episode_id": identity,
        "mode": mode,
        "steps": 360,
        "length_s": 6.0,
        "terminated": False,
        "time_out": True,
        "error_height_m": 0.01,
        "error_velocity_xy_mps": 0.1,
        "error_yaw_radps": 0.2,
        "error_pelvis_xy_m": 0.01 if mode in ("stand", "squat") else None,
        "pelvis_steps": 360 if mode in ("stand", "squat") else 0,
        "pelvis_error_sum": 3.6 if mode in ("stand", "squat") else 0,
        "pelvis_success_steps": 360 if mode in ("stand", "squat") else 0,
        "disturbance_count": 2,
        "command_height_sum": -36,
        "command_speed_sum": 72,
        "command_yaw_abs_sum": 108,
        "reasons": {"bad_ori": False},
    }
    return record | kwargs


def _case(episodes=None, seed=11, checkpoint="model_a", fraction=0.2, module="upper_body_sweep"):
    return {
        "schema_version": 1,
        "job_id": f"{checkpoint}-{fraction}-{seed}",
        "checkpoint": {"label": checkpoint, "path": f"/{checkpoint}.pt", "sha256": checkpoint * 8},
        "module": module,
        "condition_id": f"arms-{fraction}",
        "axis_step": round(fraction * 1000),
        "arm_fraction": fraction,
        "waist_fraction": 0.1,
        "seed": seed,
        "policy_output": "mean",
        "episodes": [_episode()] if episodes is None else episodes,
        "metadata": {"simulator": "test"},
    }


def _row(rows, mode):
    return next(row for row in rows if row["mode"] == mode)


def test_joint_success_uses_strict_thresholds_and_natural_timeout_without_failure():
    episodes = [
        _episode(0),
        _episode(1, error_height_m=0.03),
        _episode(2, error_yaw_radps=0.4),
        _episode(3, error_velocity_xy_mps=0.5),
        _episode(4, error_pelvis_xy_m=0.02),
        _episode(5, terminated=True, time_out=True, length_s=3),
    ]
    row = _row(summarize_results(_config(), [_case(episodes)]), "stand")
    assert row["joint_success_rate"] == pytest.approx(1 / 6)
    assert row["failure_rate"] == pytest.approx(1 / 6)
    assert row["timeout_rate"] == pytest.approx(5 / 6)
    assert row["height_tracking_success_rate"] == pytest.approx(5 / 6)
    assert row["yaw_tracking_success_rate"] == pytest.approx(5 / 6)
    assert row["velocity_xy_tracking_success_rate"] == pytest.approx(5 / 6)
    assert row["velocity_tracking_success_rate"] == pytest.approx(4 / 6)
    assert row["survival_fraction"] == pytest.approx(5.5 / 6)
    assert 0 < row["joint_success_rate_lower_95"] < 1 / 6 < row["joint_success_rate_upper_95"] < 1


def test_motion_exposure_counts_and_durations_are_pooled_before_ratios():
    episodes = [
        _episode(
            0,
            motion_started=2,
            motion_completed=1,
            motion_pending_replaced=3,
            motion_duration_sum_s=3.0,
            motion_time_s=2.0,
        ),
        _episode(
            1,
            motion_started=1,
            motion_completed=1,
            motion_pending_replaced=0,
            motion_duration_sum_s=1.0,
            motion_time_s=1.0,
        ),
    ]
    rows = summarize_results(_config(), [_case(episodes)])
    for mode in ("stand", "aggregate"):
        row = _row(rows, mode)
        assert row["motion_started"] == 3
        assert row["motion_started_hz"] == pytest.approx(3 / 12)
        assert row["motion_reference_completion_rate"] == pytest.approx(2 / 3)
        assert row["motion_mean_duration_s"] == pytest.approx(4 / 3)
        assert row["motion_fraction"] == pytest.approx(3 / 12)
        assert row["motion_pending_replaced_per_episode"] == pytest.approx(1.5)


def test_episode_errors_step_context_and_pelvis_use_their_own_denominators():
    short = _episode(
        0,
        steps=60,
        length_s=1,
        terminated=True,
        time_out=False,
        error_height_m=0.05,
        error_pelvis_xy_m=0.04,
        pelvis_steps=60,
        pelvis_error_sum=2.4,
        pelvis_success_steps=0,
        disturbance_count=1,
        command_height_sum=-12,
        command_speed_sum=6,
        command_yaw_abs_sum=12,
    )
    row = _row(summarize_results(_config(), [_case([short, _episode(1)])]), "stand")
    assert row["error_height_m"] == pytest.approx(0.03)
    assert row["error_pelvis_xy_m"] == pytest.approx(6 / 420)
    assert row["pelvis_step_success_rate"] == pytest.approx(360 / 420)
    assert row["height_offset_m"] == pytest.approx(-48 / 420)
    assert row["command_speed_xy_mps"] == pytest.approx(78 / 420)
    assert row["disturbance_hz"] == pytest.approx(3 / 7)
    assert row["episode_count"] == 2
    assert row["step_count"] == 420


def test_aggregates_use_configured_mode_weights_and_preserve_original_three():
    episodes = [
        _episode(0),
        _episode(1, mode="walk", error_yaw_radps=1),
        _episode(2, mode="squat_walk", error_yaw_radps=1),
    ]
    rows = summarize_results(_config(weights=(0.6, 0, 0.3, 0.1)), [_case(episodes)])
    combined = _row(rows, "aggregate")
    assert combined["joint_success_rate"] == pytest.approx(0.6)
    assert combined["worst_mode_joint_success_rate"] == 0
    assert combined["episode_count"] == 3
    assert combined["error_yaw_radps"] == pytest.approx(0.52)
    assert _row(rows, "original_three")["joint_success_rate"] == pytest.approx(2 / 3)
    assert "joint_success_rate_lower_95" not in combined
    assert "error_pelvis_xy_m" not in _row(rows, "walk")
    assert "pelvis_step_success_rate" not in _row(rows, "squat_walk")


def test_missing_seeds_are_excluded_and_between_seed_variation_is_preserved(tmp_path):
    config = _config(seeds=(11, 22))
    cases = [_case(), _case(seed=22, episodes=[_episode(error_yaw_radps=0.7)]), _case(fraction=0.4)]
    rows = summarize_results(config, cases)
    row = _row(rows, "stand")
    assert row["joint_success_rate"] == 0.5
    assert row["joint_success_rate_seed_std"] == pytest.approx(math.sqrt(0.5))
    assert {row["axis_step"] for row in rows} == {200}
    assert row["seed_count"] == 2
    report = write_report(tmp_path, config, cases)
    summary = json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))
    assert summary["incomplete_groups"][0]["missing_seeds"] == [22]
    assert summary["incomplete_groups"][0]["axis_step"] == 400


def test_tensorboard_has_checkpoint_runs_sorted_real_axes_and_chinese_panels(tmp_path):
    cases = [_case(checkpoint=name, fraction=fraction) for name in ("model_a", "model_b") for fraction in (0.555, 0.2)]
    paths = write_report(tmp_path, _config(), cases)
    logdir = Path(paths["tensorboard_dir"])
    tag = "01 任务完成/stand/joint_success_rate"
    for name in ("model_a", "model_b"):
        events = EventAccumulator(str(logdir / "upper_body_sweep" / "mean" / name)).Reload()
        assert [(x.step, x.value) for x in events.Scalars(tag)] == [(200, 1), (555, 1)]
        assert not any("Loss" in value for value in events.Tags()["scalars"])
        tensor = events.Tensors(metadata.CONFIG_SUMMARY_TAG)[0].tensor_proto
        layout = Layout.FromString(tensor.string_val[0])
        assert [x.title for x in layout.category] == ["01 任务完成", "02 跟踪精度", "03 失败定位", "04 测试条件"]
        assert not layout.category[0].closed
        assert all(category.closed for category in layout.category[1:])
        assert "双臂目标范围比例‰" in layout.category[0].chart[0].title
    # Re-publishing a resumed evaluation leaves both source results and past reports intact.
    before = {path: path.read_bytes() for path in logdir.rglob("*") if path.is_file()}
    second = write_report(tmp_path, _config(), cases)
    assert paths["report_dir"] != second["report_dir"]
    assert all(path.read_bytes() == content for path, content in before.items())
    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert summary["configuration"] == _config()
    assert Path(paths["summary_csv"]).read_text(encoding="utf-8-sig").startswith("checkpoint,")


def test_empty_partial_report_never_invents_zero_performance(tmp_path):
    report = write_report(tmp_path, _config(seeds=(11, 22)), [_case()])
    summary = json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))
    assert summary["rows"] == []
    assert len(summary["incomplete_groups"]) == 1
    assert not list(Path(report["report_dir"]).rglob("events.out.tfevents.*"))


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"error_height_m": float("nan")}, "Non-finite"),
        ({"length_s": 0}, "positive"),
        ({"terminated": False, "time_out": False}, "naturally"),
        ({"pelvis_steps": 0}, "pelvis"),
    ],
)
def test_invalid_episode_data_is_rejected(mutation, message):
    with pytest.raises(ValueError, match=message):
        summarize_results(_config(), [_case([_episode(**mutation)])])


def test_duplicate_cases_and_unequal_seed_quotas_are_rejected():
    case = _case()
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        summarize_results(_config(), [case, copy.deepcopy(case)])
    with pytest.raises(ValueError, match="quotas differ"):
        summarize_results(_config(seeds=(11, 22)), [case, _case(seed=22, episodes=[_episode(0), _episode(1)])])


def test_fixed_quota_and_mode_local_episode_identity_are_validated():
    config = _config(weights=(1, 0, 1, 0))
    config["benchmark"]["episodes_per_seed"] = 4
    with pytest.raises(ValueError, match="quota"):
        summarize_results(config, [_case([_episode(0), _episode(1, mode="walk")])])
    with pytest.raises(ValueError, match="Duplicate episode identity"):
        summarize_results(_config(), [_case([_episode(0), _episode(0)])])
    rows = summarize_results(_config(weights=(1, 0, 1, 0)), [_case([_episode(0), _episode(0, mode="walk")])])
    assert _row(rows, "aggregate")["episode_count"] == 2
