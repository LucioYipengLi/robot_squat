"""Check pre-reset time bins and portable motion archive compatibility."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from g1.rl.evaluation.jobs import write_json
from g1.rl.evaluation.q_outputs import write_q_outputs
from g1.rl.failures.viewer import read_index, sample_payload
from g1.tasks.upper_limb_agent.q_eval_recording import QWindows


def test_q_time_bins_preserve_end_window_and_early_cutoff():
    windows = QWindows(SimpleNamespace(step_dt=1.0, num_envs=2, device="cpu"), 0.03)
    windows.reset(5)
    for step in range(1, 6):
        state = SimpleNamespace(
            target_scores=torch.tensor([[0.5, 0.9], [0.5, 0.9]]),
            target_valid=torch.tensor([[True, step != 4], [True, True]]),
            actual_scores=torch.tensor([[0.6, 0.8], [0.6, 0.8]]),
            actual_valid=torch.ones((2, 2), dtype=torch.bool),
        )
        active = torch.tensor([True, step <= 3])
        end = torch.tensor([step >= 4, False])
        pose = torch.full((2, 2), step / 1000)
        speed_square = torch.full((2, 2), 0.02**2 if step == 5 else 0.04**2)
        windows.add(active, end, torch.tensor([step, step]), pose, pose, speed_square, state)
    first, second = windows.records(torch.tensor([0, 1]), [{"deadline_step": 5}] * 2, 0.72)
    assert [(w["start_s"], w["end_s"], w["frames"]) for w in first["timeline"]] == [(0, 2, 2), (2, 4, 2), (4, 5, 1)]
    assert first["wrist_left_error_mm"] == pytest.approx(4.5)
    assert first["wrist_both_q_target"] == pytest.approx(0.7)
    assert first["wrist_both_q_target_coverage"] == 0.5
    assert not first["wrists_steady"]
    assert second["timeline"][-1]["frames"] == 1 and second["timeline"][-1]["expected_frames"] == 2
    assert second["wrist_both_q_actual"] is None and second["wrists_steady"] is None


def test_timeline_report_publishes_clip_readable_by_viewer(tmp_path):
    cell = tmp_path / "cases" / "job"
    cell.mkdir(parents=True)
    report = tmp_path / "report"
    report.mkdir()
    names = ["pelvis"]
    write_json(tmp_path / "replay_geometry.json", {"body_names": names, "meshes": [], "edges": []})
    np.savez_compressed(
        cell / "q_replay.npz", time=[0, 0.01], body_pose=np.zeros((2, 1, 7)), goal_pose=np.zeros((2, 2, 7))
    )
    case = {
        "job_id": "job",
        "checkpoint": {"label": "q0p0"},
        "condition_id": "fixed",
        "seed": 42,
        "policy_output": "mean",
        "episodes": [{"case_id": "fixed-42", "steps": 1, "timeline": [{"start_s": 0.0, "wrist_both_q_actual": None}]}],
        "metadata": {
            "control_dt": 0.01,
            "level": 7,
            "q_replay": {"case_id": "fixed-42", "frames": 2, "body_names": names, "joint_names": []},
        },
    }
    paths = write_q_outputs(tmp_path, report, [case])
    with open(paths["timeline_csv"], encoding="utf-8-sig", newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["checkpoint"] == "q0p0" and row["wrist_both_q_actual"] == ""
    (sample,) = read_index(paths["replay_dir"])["samples"]
    assert sample["checkpoint"] == "q0p0"
    assert sample_payload(paths["replay_dir"], sample["sample_id"])["time"] == [0, 0.01]
