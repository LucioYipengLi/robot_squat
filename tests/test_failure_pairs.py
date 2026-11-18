"""Fixed target reflection, switch timing and paired failure accounting."""

import json

import pytest
import torch
from test_planar_timing import make_command

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul

from g1.rl.failures.cli import parse_args, run
from g1.rl.failures.paired_recording import FixedSwitchPairs, mirror_right_poses
from g1.rl.failures.pairs import COORDINATE_FRAME, load_switch_pairs
from g1.rl.failures.storage import FailureArchive
from g1.tasks.upper_limb_agent.mdp.terminations import task_time_out


def test_mirror_follows_reset_pelvis_and_destination_wrist_axes():
    origin = torch.tensor([0.05, -0.02, 0.75])
    pelvis = quat_from_euler_xyz(*torch.tensor([0.2, 0.1, 1.0]))
    right_rest = quat_from_euler_xyz(*torch.tensor([-0.1, 0.2, -0.2]))
    left_rest = quat_from_euler_xyz(*torch.tensor([0.3, -0.1, 0.5]))
    offset = torch.tensor([0.23, -0.15, 0.17])
    local_rot = torch.tensor([0.1, -0.12, 0.18, 0.96])
    local_rot /= local_rot.norm()
    reflected_rot = local_rot * torch.tensor([-1, 1, -1, 1])
    pelvis_inv = pelvis * torch.tensor([-1, -1, -1, 1])
    right = torch.cat(
        (origin + quat_apply(pelvis, offset), quat_mul(quat_mul(quat_mul(pelvis, local_rot), pelvis_inv), right_rest))
    )
    expected = torch.cat(
        (
            origin + quat_apply(pelvis, offset * torch.tensor([1, -1, 1])),
            quat_mul(quat_mul(quat_mul(pelvis, reflected_rot), pelvis_inv), left_rest),
        )
    )
    left = mirror_right_poses(right, origin, pelvis, right_rest, left_rest)
    torch.testing.assert_close(left, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        mirror_right_poses(left, origin, pelvis, left_rest, right_rest), right, atol=1e-6, rtol=1e-6
    )


def test_fixed_pairs_switch_once_at_four_seconds_and_keep_partial_resets_separate():
    term = make_command(probability=0.2, level=1)
    env = term._env
    old = term.default_wrist_pose_e[1, 1].clone()
    new = old.clone()
    new[:3] += quat_apply(term._sampling_quat_w[1], torch.tensor([0.04, 0.05, 0.06]))
    pair = {"id": "one", "group": "in_up", "right_old_pose": old.tolist(), "right_new_pose": new.tolist()}
    program = FixedSwitchPairs(term, {"pairs": [pair, {**pair, "id": "two"}]})
    term._teleop = program
    term.reset()
    assert program.trial_ids.tolist() == [0, 1, 2]
    assert program.metadata(2) == {"pair_id": "two", "pair_group": "in_up", "pair_side": "left"}
    torch.testing.assert_close(term.goal_pose_w[1, 1, :3] - env.scene.env_origins[1], old[:3])

    env.common_step_counter = 60
    term.reset(torch.tensor([2]))  # One trial ends early; other environments retain their schedule.
    assert program.trial_ids.tolist() == [0, 1, 3]
    for step in (239, 240, 299, 300, 360):
        env.common_step_counter = step
        term.compute(env.step_dt)
        assert term.command_counter.tolist() == [1 + int(step >= 240)] * 2 + [1 + int(step >= 300)]
        assert task_time_out(env).tolist() == [step >= 300, step >= 300, step >= 360]
    torch.testing.assert_close(term.goal_pose_w[1, 1, :3] - env.scene.env_origins[1], new[:3])
    left_delta = term.goal_pose_w[0, 0, :3] - env.scene.env_origins[0] - term._default_pelvis_pos_e[0]
    right_delta = new[:3] - term._default_pelvis_pos_e[0]
    torch.testing.assert_close(
        quat_apply_inverse(term._sampling_quat_w[0], left_delta),
        quat_apply_inverse(term._sampling_quat_w[0], right_delta) * torch.tensor([1, -1, 1]),
    )
    term.reset(torch.tensor([0]))
    assert program.trial_ids.tolist() == [-1, 1, 3]


def test_suite_generation_and_pair_summary_keep_early_failures_visible(tmp_path):
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"level": 1, "coordinate_frame": COORDINATE_FRAME}))
    old = [[0, 0, 0, 0, 0, 0, 1], [0.15, -0.15, 0.8, 0, 0, 0, 1]]
    rows = []
    for i, (dy, dz) in enumerate(((0.05, 0.07), (-0.05, 0.07), (0.08, 0.01))):
        rows.append(
            {
                "episode_id": i,
                "switch_index": 1,
                "arm_mode": "right_only",
                "level": 1,
                "pre_switch_ready": True,
                "pre_switch_contact": False,
                "position_jump_bin": "8_to_12cm",
                "contact_over_hard_threshold": i == 0,
                "active_wrist_delta_initial_pelvis_m": [[0.02, dy, dz]],
                "old_goal_pose": old,
                "new_goal_pose": [old[0], [0.17, -0.15 + dy, 0.8 + dz, *old[1][3:]]],
            }
        )
    (run_dir / "switches.jsonl").write_text("\n".join(map(json.dumps, rows)))
    suite_path = tmp_path / "suite.json"
    assert run(["pairs", "--run", str(run_dir), "--output", str(suite_path), "--in_up", "1", "--controls", "1"]) == 0
    suite = load_switch_pairs(suite_path)
    assert sorted(pair["group"] for pair in suite["pairs"]) == ["in_flat", "in_up", "out_up"]
    archive = FailureArchive(tmp_path / "eval", {"switch_pair_evaluation": {"cases": suite["pairs"]}})
    pairs = {p["group"]: p["id"] for p in suite["pairs"]}

    def record(group, side, hard=None, *, failed=False):
        switches = (
            (
                {
                    "pre_switch_ready": True,
                    "pre_switch_contact": False,
                    "contact_over_hard_threshold": hard,
                    "new_goal_reached_within_1s": not hard,
                },
            )
            if hard is not None
            else ()
        )
        archive.add_episode(
            {
                "pair_id": pairs[group],
                "pair_side": side,
                "category": "regular",
                "failed": failed,
                "elapsed_s": 1.0,
                "reasons": [],
            },
            switches=switches,
        )

    record("in_up", "right", True, failed=True)
    record("in_up", "left", False)
    record("out_up", "right", failed=True)
    record("out_up", "left", False)
    record("in_flat", "right", False)
    archive.progress("interrupted")
    archive.close()
    summary = json.loads((archive.directory / "summary.json").read_text())["switch_pair_evaluation"]["overall"]
    assert summary["right_only_contact"] == summary["not_comparable"] == summary["incomplete"] == 1
    assert summary["right"]["failed_before_switch"] == 1 and summary["left"]["reached_within_1s"] == 2


def test_fixed_pairs_requires_regular_mean_and_no_refresh_override():
    base = ["mine", "--checkpoint", "model.pt", "--switch_pairs", "suite.json"]
    assert parse_args([*base, "--category", "regular"]).switch_pairs.name == "suite.json"
    for extra in (
        [],
        ["--category", "regular", "--regular_refresh_s", "2", "4"],
        ["--category", "regular", "--policy_output", "sample"],
    ):
        with pytest.raises(SystemExit):
            parse_args([*base, *extra])
