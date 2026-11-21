"""Fixed target reflection, switch timing and paired failure accounting."""

import json
from types import SimpleNamespace

import pytest
import torch
from test_planar_timing import make_command

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul

from g1.rl.failures.cli import parse_args, run
from g1.rl.failures.paired_recording import FixedSwitchPairs, mirror_right_poses
from g1.rl.failures.pairs import COORDINATE_FRAME, PairedSwitchSummary, load_switch_pairs
from g1.rl.failures.storage import FailureArchive
from g1.tasks.upper_limb_agent.mdp.terminations import task_time_out


def test_retract_suite_uses_both_jump_bins_and_only_backward_targets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.json").write_text(json.dumps({"level": 3, "coordinate_frame": COORDINATE_FRAME}))
    old = [[0, 0, 0, 0, 0, 0, 1], [0.2, -0.15, 0.8, 0, 0, 0, 1]]
    rows = []
    for i, (direction, delta, label) in enumerate(
        (("retract", -0.07, "5_to_8cm"), ("retract", -0.09, "8_to_12cm"), ("forward", 0.09, "8_to_12cm"))
    ):
        rows.append(
            {
                "episode_id": i,
                "switch_index": 1,
                "arm_mode": "right_only",
                "level": 3,
                "pre_switch_ready": True,
                "pre_switch_contact": False,
                "direction": direction,
                "position_jump_bin": label,
                "contact_over_hard_threshold": i == 0,
                "old_goal_pose": old,
                "new_goal_pose": [old[0], [0.2 + delta, *old[1][1:]]],
            }
        )
    (source / "switches.jsonl").write_text("\n".join(map(json.dumps, rows)))
    path = tmp_path / "retract.json"
    assert run(["pairs", "--run", str(source), "--output", str(path), "--suite", "retract", "--per-bin", "1"]) == 0
    pairs = load_switch_pairs(path)["pairs"]
    assert {pair["group"] for pair in pairs} == {"retract_5_to_8cm", "retract_8_to_12cm"}
    assert {pair["source_episode_id"] for pair in pairs} == {0, 1}


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


def test_dual_retract_suite_switches_both_wrists_in_one_trial(tmp_path):
    term = make_command(probability=0.2, level=3)
    env = term._env
    old = term.default_wrist_pose_e[0].clone()
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.json").write_text(json.dumps({"level": 3, "coordinate_frame": COORDINATE_FRAME}))
    rows = []
    for index, (left, right, label) in enumerate(
        ((-0.09, -0.08, "8_to_12cm"), (-0.13, -0.06, "12cm_and_over"), (-0.10, 0.01, "8_to_12cm"))
    ):
        delta = torch.tensor([[left, 0.0, 0.0], [right, 0.0, 0.0]])
        new = old.clone()
        new[:, :3] += quat_apply(term._sampling_quat_w[0], delta)
        rows.append(
            {
                "episode_id": index,
                "switch_index": 1,
                "arm_mode": "both",
                "level": 3,
                "pre_switch_ready": True,
                "pre_switch_contact": False,
                "direction": "retract",
                "position_jump_bin": label,
                "active_wrist_delta_initial_pelvis_m": delta.tolist(),
                "contact_over_hard_threshold": False,
                "old_goal_pose": old.tolist(),
                "new_goal_pose": new.tolist(),
            }
        )
    (source / "switches.jsonl").write_text("\n".join(map(json.dumps, rows)))
    path = tmp_path / "dual.json"
    assert run(["pairs", "--run", str(source), "--output", str(path), "--suite", "dual_retract", "--per-bin", "1"]) == 0
    suite = load_switch_pairs(path)
    assert suite["schema_version"] == 2
    assert {case["source_episode_id"] for case in suite["pairs"]} == {0, 1}
    program = FixedSwitchPairs(term, suite)
    term._teleop = program
    term.reset()
    assert program.trial_ids.tolist() == [0, 1, -1]
    assert program.metadata(0)["pair_side"] == "both" and term.active_mask[:2].all()
    for step in (239, 240, 299, 300):
        env.common_step_counter = step
        term.compute(env.step_dt)
        assert term.command_counter[:2].tolist() == [1 + int(step >= 240)] * 2
        assert task_time_out(env)[:2].tolist() == [step >= 300] * 2
        key = "new_poses" if step >= 240 else "old_poses"
        expected = torch.tensor([case[key] for case in suite["pairs"]])
        actual = term.goal_pose_w[:2].clone()
        actual[..., :3] -= env.scene.env_origins[:2, None]
        torch.testing.assert_close(actual, expected)

    archive = FailureArchive(
        tmp_path / "eval", {"switch_pair_evaluation": {"cases": suite["pairs"], "trial_mode": "dual_arm"}}
    )
    for index in range(2):
        archive.add_episode(
            {
                **program.metadata(index),
                "category": "regular",
                "failed": index == 1,
                "elapsed_s": 4.0,
                "reasons": [],
            },
            switches=(
                {
                    "pre_switch_ready": True,
                    "pre_switch_contact": False,
                    "contact_over_hard_threshold": False,
                    "new_goal_reached_within_1s": True,
                },
            )
            if index == 0
            else (),
        )
    archive.progress("episodes")
    archive.close()
    summary = json.loads((archive.directory / "summary.json").read_text())["switch_pair_evaluation"]["overall"]
    assert summary["cases"] == summary["completed"] == 2
    assert summary["failed_before_switch"] == summary["not_comparable"] == 1
    assert summary["reached_within_1s"] == summary["reach_assessed"] == summary["contact_assessed"] == 1


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


@pytest.mark.parametrize("mode,release_step", [("direct", 240), ("hold", 276), ("retreat", 276)])
def test_switch_diagnostic_releases_same_goal_and_records_actual_retreat(mode, release_step):
    term = make_command(probability=0.2, level=3)
    env = term._env
    previous_actions = torch.arange(69, dtype=torch.float32).reshape(3, 23)
    env.action_manager = SimpleNamespace(
        active_terms=["lower_command", "height_offset", "pelvis_offset", "joint_pos"],
        action_term_dim=[3, 1, 2, 17],
        action=previous_actions.clone(),
    )
    old = term.default_wrist_pose_e[0].clone()
    new = old.clone()
    new[..., 0] -= 0.09
    pair = {"id": "one", "group": "retract", "old_poses": old.tolist(), "new_poses": new.tolist()}
    program = FixedSwitchPairs(term, {"schema_version": 2, "pairs": [pair]}, diagnostic=mode)
    term._teleop = program
    term.reset()
    diagnostic = program.diagnostic
    assert program.switch_step == release_step
    assert program.end_step == release_step + 60
    pelvis = torch.zeros(3, 7)
    pelvis[:, 6] = 1
    pelvis[0, 3:] = quat_from_euler_xyz(*torch.tensor([0.0, 0.0, torch.pi / 2]))
    status = torch.zeros(3, dtype=torch.bool)
    actions = torch.zeros_like(previous_actions)
    env.common_step_counter = 240
    diagnostic.observe(pelvis, ~status, status, status)
    term.compute(env.step_dt)
    out = diagnostic.intervene(actions)
    if mode == "direct":
        assert out is actions
    else:
        assert term.command_counter[0] == 1
        assert out[0, :3].tolist() == ([-1.0, 0.0, 0.0] if mode == "retreat" else [0.0] * 3)
        torch.testing.assert_close(out[0, 6:], previous_actions[0, 6:])
        assert out[0, 3:6].tolist() == [3.0, 0.0, 0.0]
        torch.testing.assert_close(out[1:], actions[1:])
        env.common_step_counter = 260
        pelvis[0, 1] = -0.06
        diagnostic.observe(pelvis, status, status, status)
        assert diagnostic.intervene(actions)[0, 0] == 0
        env.common_step_counter = release_step
        diagnostic.observe(pelvis, status, status, status)
        term.compute(env.step_dt)
    assert term.command_counter[0] == 2
    assert diagnostic.intervene(actions) is actions
    actual = term.goal_pose_w[0].clone()
    actual[..., :3] -= env.scene.env_origins[0]
    torch.testing.assert_close(actual, new)

    for step in range(program.end_step - diagnostic.stable_steps + 1, program.end_step + 1):
        env.common_step_counter = step
        diagnostic.observe(pelvis, ~status, status, status)
    metadata = program.metadata(0)
    assert metadata["switch_diagnostic"] == {
        "ready_before_intervention": True,
        "contact_before_intervention": False,
        "pelvis_retreat_at_switch_m": pytest.approx(0.0 if mode == "direct" else 0.06),
        "stable_last_0p2s": True,
    }
    summary = PairedSwitchSummary([pair], dual=True)
    summary.add(
        {**metadata, "episode_id": 0, "failed": False},
        [
            {
                "pre_switch_ready": mode == "direct",
                "contact_over_hard_threshold": False,
                "new_goal_reached_within_1s": True,
            }
        ],
    )
    overall = summary.report()["overall"]
    assert overall["comparable_cases"] == overall["stable_last_0p2s"] == 1
    term.reset(torch.tensor([0]))
    assert not diagnostic.started.any() and not diagnostic.ready_steps.any()


def test_switch_diagnostic_requires_a_fixed_suite():
    base = ["mine", "--checkpoint", "model.pt", "--category", "regular"]
    with pytest.raises(SystemExit):
        parse_args([*base, "--switch-diagnostic", "retreat"])
    args = parse_args(
        [*base, "--switch_pairs", "suite.json", "--switch-cases", "one", "two", "--switch-diagnostic", "hold"]
    )
    assert args.switch_cases == ["one", "two"] and args.switch_diagnostic == "hold"
