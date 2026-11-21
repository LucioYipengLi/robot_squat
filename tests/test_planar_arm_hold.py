"""Verify the paired startup intervention without starting PhysX."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from g1.rl.failures.cli import parse_args
from g1.rl.failures.planar_hold import PlanarArmHold
from g1.rl.failures.storage import FailureArchive
from g1.tasks.upper_limb_agent.mdp.actions import UPPER_BODY_JOINT_NAMES
from g1.tasks.upper_limb_agent.mdp.contacts import ARM_CONTACT_BODY_PATHS
from g1.tasks.upper_limb_agent.symmetry import ACTION_DIMS, ACTION_TERMS


def make_trace(steps, goal_y):
    body_pose = np.tile(np.float32([0, 0, 0.75, 0, 0, 0, 1]), (steps + 1, 1, 1))
    goal_pose = body_pose.repeat(2, axis=1)
    goal_pose[..., 0] = 0.4
    goal_pose[..., 1] = goal_y
    action = np.zeros((steps + 1, sum(ACTION_DIMS)), dtype=np.float32)
    action[1:] = np.arange(1, steps + 1)[:, None]
    forces = np.zeros((steps + 1, 2, len(ARM_CONTACT_BODY_PATHS), 3), dtype=np.float32)
    forces[0] = np.nan
    return {
        "time": np.arange(steps + 1) * 0.02,
        "body_pose": body_pose,
        "goal_pose": goal_pose,
        "joint_pos": np.full((steps + 1, 17), 0.2, dtype=np.float32),
        "joint_vel": np.zeros((steps + 1, 17), dtype=np.float32),
        "action": action,
        "lower_command": action[:, :6].copy(),
        "pair_force": forces,
    }


def episode(env_id, elapsed_s, *, failed):
    return {
        "env_id": env_id,
        "category": "planar",
        "arm_mode": 0,
        "elapsed_s": elapsed_s,
        "failed": failed,
        "reasons": ["base_contact"] if failed else ["time_out"],
        "planar_sampling_group": "wide",
        "planar_distance_m": 0.4,
        "planar_heading_rad": 0.5,
        "planar_entered_near": not failed,
        "planar_first_near_s": None if failed else 0.5,
        "planar_eval_pose_ready_fraction": 0.0 if failed else 1.0,
    }


@pytest.fixture
def baseline(tmp_path):
    manifest = {
        "task": "Template-G1-Upper-Mobile-v0",
        "checkpoint_sha256": "test",
        "seed": 42,
        "policy_output": "mean",
        "level": 4,
        "num_envs": 3,
        "planar_sampling": "transition",
        "control_dt": 0.02,
        "physics_dt": 0.01,
        "physics_substeps": 2,
        "body_names": ["pelvis"],
        "joint_names": list(UPPER_BODY_JOINT_NAMES),
        "partners": list(ARM_CONTACT_BODY_PATHS),
        "terminations": {"base_contact": 20.0},
        "torso_contact_policy": {"other_threshold_n": 1.0},
        "source_usd": "test",
        "variants": {},
        "planar_startup": {"window_s": 0.2},
    }
    failed_trace = make_trace(2, -0.2)
    failed_trace["body_pose"][-1, 0, 5:] = [math.sin(-0.1), math.cos(-0.1)]
    failed_trace["pair_force"][-1, -1, 0, 0] = 30
    traces = [failed_trace, make_trace(10, -0.2), make_trace(10, 0.2)]
    archive = FailureArchive(tmp_path / "baseline", manifest)
    for env_id, trace in enumerate(traces):
        archive.add_episode(episode(env_id, 0.04 if env_id == 0 else 1.0, failed=env_id == 0), startup_trace=trace)
    archive.progress("episodes")
    archive.close()
    return archive.directory, manifest, traces


def prepare_hold(hold, traces):
    action = SimpleNamespace(_joint_names=UPPER_BODY_JOINT_NAMES, _joint_ids=slice(None), _offset=0.1, _scale=0.5)
    manager = SimpleNamespace(
        active_terms=[name for name, _ in ACTION_TERMS],
        action_term_dim=ACTION_DIMS,
        get_term=lambda _: action,
    )
    env = SimpleNamespace(action_manager=manager, device="cpu")
    term = SimpleNamespace(mode_id=torch.zeros(3, dtype=torch.long), planar_sampling_group=torch.full((3,), 3))
    motion = {key: torch.tensor(np.stack([trace[key][0] for trace in traces])) for key in traces[0]}
    hold.prepare(env, term, motion)
    return env, term, motion


def test_replay_holds_only_arms_and_pads_baseline_input(baseline):
    directory, manifest, traces = baseline
    hold = PlanarArmHold(directory)
    hold.validate_run(manifest)
    assert hold.settings()["env_ids"] == [0, 1]
    with pytest.raises(ValueError, match="seed"):
        hold.validate_run({**manifest, "seed": 43})

    env, term, motion = prepare_hold(hold, traces)

    incoming = torch.zeros((3, sum(ACTION_DIMS)))
    pending = torch.tensor([True, True, True])
    result = hold.intervene(incoming, step=3, pending=pending)
    assert result[:, 0].tolist() == [2, 4, 0]  # Env 0 pads the final recorded A command.
    torch.testing.assert_close(result[:2, :9], torch.tensor([[2.0] * 9, [4.0] * 9]))
    torch.testing.assert_close(result[:2, 9:], torch.full((2, 14), 0.2))
    assert torch.count_nonzero(result[2]) == torch.count_nonzero(incoming) == 0
    pending[0] = False
    assert torch.count_nonzero(hold.intervene(incoming, step=3, pending=pending)[0]) == 0
    assert hold.intervene(incoming, step=10, pending=pending) is incoming
    motion["goal_pose"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="env_id 0"):
        hold.prepare(env, term, motion)


def test_report_compares_same_time_and_tracks_post_release(baseline):
    directory, _, _ = baseline
    hold = PlanarArmHold(directory)
    trace = make_trace(10, -0.2)
    trace["pair_force"][3, 1, 0, 0] = 18
    hold.add_episode({**episode(0, 1.0, failed=False), "episode_id": 5}, trace)
    hold.add_episode({**episode(1, 0.3, failed=True), "episode_id": 6}, make_trace(10, -0.2))
    result = hold.report()
    assert result["selected_cases"] == result["completed_cases"] == result["comparable_cases"] == 2
    assert result["comparable_baseline_hard_contacts"] == 1
    assert result["comparable_hold_hard_contacts"] == 0
    assert result["hold_failures_during_hold"] == 0
    assert result["hold_failures_after_release"] == result["hold_full_pose_ready"] == 1
    case = result["by_env_id"][0]
    assert case["matched_window_s"] == pytest.approx(0.04)
    assert case["baseline"]["max_reverse_yaw_deg"] == pytest.approx(math.degrees(0.2), abs=1e-5)
    assert case["baseline"]["shoulder_peak_n"] == 30
    assert case["hold"]["shoulder_peak_n"] == 0  # 18 N occurred after the common window.

    hold.add_episode({**episode(1, 0.02, failed=True), "episode_id": 7}, make_trace(1, -0.2))
    assert hold.report()["comparable_cases"] == 1


def test_hold_cli_requires_unmodified_mean_transition_sampler():
    args = ["mine", "--checkpoint", "model.pt", "--planar-hold-from", "baseline", "--planar-sampling", "transition"]
    assert parse_args(args).planar_hold_from.name == "baseline"
    for flags in (["--policy_output", "sample"], ["--planar-near-lower-scale", "0.5"]):
        with pytest.raises(SystemExit):
            parse_args([*args, *flags])


def test_waist_hold_replays_original_lower_and_compares_arms_only(baseline):
    directory, manifest, traces = baseline
    arms_only = PlanarArmHold(directory)
    archive = FailureArchive(directory.parent / "arms_only", {**manifest, "planar_hold": arms_only.settings()})
    for env_id in (0, 1):
        trace = make_trace(1, -0.2)
        trace["pair_force"][1, -1, 0, 0] = 50
        archive.add_episode(episode(env_id, 0.02, failed=True), startup_trace=trace)
    archive.close()

    waist_hold = PlanarArmHold(archive.directory, hold_waist=True)
    prepare_hold(waist_hold, traces)
    incoming = torch.zeros((3, sum(ACTION_DIMS)))
    pending = torch.ones(3, dtype=torch.bool)
    actions = waist_hold.intervene(incoming, step=3, pending=pending)
    # C uses A's inputs after B's earlier crash, and replaces all 17 upper targets.
    torch.testing.assert_close(actions[:2, :6], torch.tensor([[2.0] * 6, [4.0] * 6]))
    torch.testing.assert_close(actions[:2, 6:], torch.full((2, 17), 0.2))
    assert waist_hold.intervene(incoming, step=10, pending=pending) is incoming

    c_trace = make_trace(10, -0.2)
    c_trace["action"][1:, 6:] = 0.2
    waist_hold.add_episode({**episode(0, 1.0, failed=False), "episode_id": 5}, c_trace)
    report = waist_hold.report()
    assert report["comparison_group"] == "arms_only"
    assert report["selected_baseline_failures"] == 2
    assert report["comparable_cases"] == report["comparable_baseline_hard_contacts"] == 1
    assert report["comparable_hold_hard_contacts"] == 0
    result = report["by_env_id"][0]
    assert result["matched_window_s"] == pytest.approx(0.02)
    assert result["lower_replay_match"]
    assert result["baseline"]["arm_peak_n"] == 50


def test_waist_hold_cli_requires_exclusive_baseline_options():
    args = ["mine", "--checkpoint", "model.pt", "--planar-sampling", "transition", "--planar-hold-waist-from", "hold"]
    assert parse_args(args).planar_hold_waist_from.name == "hold"
    with pytest.raises(SystemExit):
        parse_args([*args, "--planar-hold-from", "baseline"])
