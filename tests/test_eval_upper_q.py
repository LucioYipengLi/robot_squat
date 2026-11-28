"""Workspace Q case replay and report semantics without simulation."""

import json
import math

import pytest

from g1.rl.evaluation import cli
from g1.rl.evaluation.config import load_suite
from g1.rl.evaluation.modules import benchmark_for_job, conditions
from g1.rl.evaluation.reporting import summarize_results, write_report
from g1.rl.evaluation.upper_q import episode_specs, quaternion_xyzw


def test_q_cases_replay_across_checkpoints_and_preserve_fixed_pose():
    config = load_suite("g1_upper_workspace_q")
    random, fixed = conditions(config)
    job = {**random, "seed": 42}
    initial = episode_specs(benchmark_for_job(config, job, {"num_envs": 1}), job)
    assert len(initial) == 24
    assert len({tuple(case["offset_m"]) for case in initial}) == len(initial)
    assert all(case["arm_mode"] == "both" and case["evaluation_start_step"] == 1080 for case in initial)
    assert initial == episode_specs(config["benchmark"], {**job, "checkpoint": "another"})
    assert initial[0]["offset_m"] != episode_specs(config["benchmark"], {**job, "seed": 43})[0]["offset_m"]
    (case,) = episode_specs(config["benchmark"], {**fixed, "seed": 42})
    assert case["wrist_poses"] == {
        "left": {"position_m": [-0.1, 0.3, 0.75], "rpy_deg": [0.003, 0.003, 89.989]},
        "right": {"position_m": [0.1, 0.3, 0.75], "rpy_deg": [-0.003, 0.003, 90.011]},
    }
    assert quaternion_xyzw([0, 0, 90]) == pytest.approx((0, 0, math.sqrt(0.5), math.sqrt(0.5)))


def test_cli_fixed_q_plan_persists_custom_pose_and_one_job_per_model(tmp_path):
    paths = [tmp_path / f"{name}.pt" for name in ("q0p0", "q0p2", "q1p2")]
    for path in paths:
        path.write_bytes(path.stem.encode())
    output = tmp_path / "plan"
    args = [
        "--suite",
        "g1_upper_workspace_q",
        "--checkpoints",
        *map(str, paths),
        "--q-mode",
        "fixed",
        "--episode-length-s",
        "12",
        "--left-wrist-pose",
        "-0.15",
        "0.3",
        "0.75",
        "0",
        "0",
        "90",
        "--plan-only",
        "--output",
        str(output),
    ]
    assert cli.run(args) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) == 3
    config = manifest["config"]
    for job in manifest["jobs"]:
        (case,) = episode_specs(config["benchmark"], job)
        assert case["case_id"] == "fixed-42-000000"
        assert case["deadline_step"] == 720 and case["evaluation_start_step"] == 600
        assert case["wrist_poses"]["left"] == {"position_m": [-0.15, 0.3, 0.75], "rpy_deg": [0.0, 0.0, 90.0]}


def test_q_report_keeps_missing_coverage_and_filters_nonsteady_windows(tmp_path):
    config = load_suite("g1_upper_workspace_q")
    config["benchmark"]["episodes_per_seed"] = 2
    job = {
        **conditions(config)[0],
        "seed": 42,
        "policy_output": "mean",
        "checkpoint": {"label": "q0p0", "path": "model.pt", "sha256": "test"},
    }
    episodes = []
    for index, spec in enumerate(episode_specs(config["benchmark"], job)):
        record = {
            **spec,
            "failed": False,
            "wrists_steady": index == 0,
            "window_steps": 120,
            "window_complete": True,
            "expected_window_steps": 120,
            "roll_abs_mean_deg": 0.0,
            "pitch_abs_mean_deg": 0.0,
            "torso_angle_mean_deg": 0.0,
            "wrist_error_mm": 6.0,
        }
        for side in ("left", "right"):
            record[f"wrist_{side}_error_mm"] = 6.0 + index * 6
            record[f"wrist_{side}_orientation_error_rad"] = 0.04
            record[f"wrist_{side}_speed_rms_mps"] = 0.02 + index * 0.03
        for kind in ("actual", "target"):
            for side in ("left", "right", "both"):
                key = f"wrist_{side}_q_{kind}"
                record[key] = (0.5 if side == "left" else None) if index == 0 else 0.9
                record[f"{key}_raw"] = record[key] * 0.72 if record[key] is not None else None
                record[f"{key}_coverage"] = float(record[key] is not None)
        episodes.append(record)
    result = {
        **job,
        "schema_version": 1,
        "episodes": episodes,
        "metadata": {"workspace_q_map_sha256": config["benchmark"]["map_sha256"], "workspace_q_reference": 0.72},
    }
    (row,) = summarize_results(config, [result])
    assert row["steady_window_rate"] == 0.5
    assert row["wrist_left_error_mm"] == 6.0
    assert row["wrist_left_q_actual"] == 0.5
    assert row["wrist_both_q_actual"] is None
    assert row["wrist_both_q_actual_coverage"] == 0.5
    report = write_report(tmp_path, config, [result])
    assert report["tensorboard_dir"]
