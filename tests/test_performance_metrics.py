"""Numerical checks for diagnostic denominators and episode/rollout boundaries."""

from types import SimpleNamespace

import pytest
import torch

from g1.tasks.lower_limb_agent.performance import ModePerformanceMonitor


def _monitor(modes):
    mode = torch.tensor(modes)
    monitor = ModePerformanceMonitor(len(modes), "cpu", 0.1, (0.03, 0.5, 0.4, 0.02))
    monitor.reset(torch.arange(len(modes)), mode, 0)
    return monitor, mode


def _sample(monitor, mode, *, height=None, valid=None):
    count = len(mode)
    monitor.sample(
        mode,
        torch.ones(count, dtype=torch.bool) if valid is None else valid,
        torch.zeros(count) if height is None else torch.tensor(height),
        torch.zeros(count),
        torch.zeros(count),
        torch.zeros(count),
        torch.zeros(count, 6),
    )


def _finish(monitor, ids, step, lengths, *, failures=None, timeouts=None):
    count = len(lengths)
    failures = torch.zeros(count, dtype=torch.bool) if failures is None else torch.tensor(failures)
    timeouts = ~failures if timeouts is None else torch.tensor(timeouts)
    monitor.finish(
        torch.tensor(ids),
        step=step,
        lengths=torch.tensor(lengths),
        terminated=failures,
        time_outs=timeouts,
        reasons={"bad_ori": failures, "time_out": timeouts},
        max_steps=3,
    )


def _drain(monitor):
    return monitor.drain(torch.ones(4))


def test_different_reset_batch_sizes_are_episode_weighted_and_failures_do_not_pass_joint_success():
    monitor, mode = _monitor([0, 0, 0, 0])
    _sample(monitor, mode, height=[0.0, 0.0, 0.0, 0.0])
    _finish(monitor, [0], 1, [1, 1, 1, 1], failures=[True, False, False, False])
    for _ in range(2):
        _sample(monitor, mode, valid=torch.tensor([False, True, True, True]))
    _finish(monitor, [1, 2, 3], 3, [1, 3, 3, 3])
    metrics = _drain(monitor)
    assert metrics["Performance/stand/episode_count"] == 4
    assert metrics["Performance/stand/failure_rate"] == 0.25
    assert metrics["Performance/stand/height_tracking_success_rate"] == 1
    assert metrics["Performance/stand/joint_success_rate"] == 0.75
    assert metrics["Performance/stand/episode_length_s"] == pytest.approx(0.25)
    assert metrics["Performance/stand/termination/bad_ori"] == 0.25
    assert metrics["Performance/stand/step_count"] == 10


def test_mode_buckets_step_share_and_no_sample_means_are_not_filled_with_zero():
    monitor, mode = _monitor([0, 3])
    _sample(monitor, mode)
    _finish(monitor, [0], 1, [1, 1])
    _sample(monitor, mode, valid=torch.tensor([False, True]))
    _finish(monitor, [1], 2, [1, 2])
    metrics = _drain(monitor)
    assert metrics["Performance/stand/step_share"] == pytest.approx(1 / 3)
    assert metrics["Performance/squat_walk/step_share"] == pytest.approx(2 / 3)
    assert metrics["Performance/squat_walk/episode_count"] == 1
    assert "Performance/squat_walk/error_pelvis_xy_m" not in metrics
    assert metrics["Performance/squat/episode_valid"] == 0
    assert "Performance/squat/failure_rate" not in metrics
    assert "Performance/squat/error_height_m" not in metrics
    empty = _drain(monitor)
    assert empty["Performance/stand/episode_count"] == 0
    assert "Performance/stand/step_share" not in empty


def test_episode_accumulators_survive_rollout_drains_and_duplicate_finish_is_ignored():
    monitor, mode = _monitor([2])
    _sample(monitor, mode, height=[0.1])
    assert _drain(monitor)["Performance/walk/episode_count"] == 0
    _sample(monitor, mode, height=[0.3])
    _finish(monitor, [0], 2, [2])
    _finish(monitor, [0], 2, [2])
    metrics = _drain(monitor)
    assert metrics["Performance/walk/episode_count"] == 1
    assert metrics["Performance/walk/error_height_m"] == pytest.approx(0.2)
    assert metrics["Performance/walk/step_count"] == 1


def test_manual_partial_and_mixed_episodes_are_excluded_but_executed_steps_remain_visible():
    monitor, mode = _monitor([0, 1, 2])
    _sample(monitor, mode)
    mode[2] = 3
    _sample(monitor, mode)
    _finish(monitor, [0, 1, 2], 2, [2, 6, 2], timeouts=[False, True, True])
    metrics = _drain(monitor)
    assert metrics["Performance/excluded/manual_episodes"] == 1
    assert metrics["Performance/excluded/partial_episodes"] == 1
    assert metrics["Performance/excluded/mixed_mode_episodes"] == 1
    assert sum(metrics[f"Performance/{name}/episode_count"] for name in ("stand", "squat", "walk", "squat_walk")) == 0
    assert sum(metrics[f"Performance/{name}/step_count"] for name in ("stand", "squat", "walk", "squat_walk")) == 6


def test_timeout_with_failure_is_not_a_success_and_disturbances_are_counted_at_actual_firing():
    monitor, mode = _monitor([1])
    _sample(monitor, mode)
    monitor.disturbance(mode, torch.tensor([0]), 0.2)
    monitor.disturbance(mode, torch.tensor([0]), 0.4)
    _finish(monitor, [0], 1, [1], failures=[True], timeouts=[True])
    metrics = _drain(monitor)
    assert metrics["Performance/squat/timeout_rate"] == 0
    assert metrics["Performance/squat/joint_success_rate"] == 0
    assert metrics["Performance/squat/disturbance_count"] == 2
    assert metrics["Performance/squat/disturbance_arm_fraction"] == pytest.approx(0.3)
    assert metrics["Performance/squat/disturbance_hz"] == pytest.approx(20)


def test_latest_task_success_survives_log_drains_and_resets_then_replaces_each_environment():
    monitor, mode = _monitor([0, 1, 2])
    assert monitor.task_success_rate() == 0
    assert not monitor.latest_joint_success_valid.any()
    _sample(monitor, mode)
    _finish(monitor, [0], 1, [1, 1, 1])
    assert monitor.task_success_rate() == 1  # Unfinished environments are excluded.
    _finish(monitor, [1], 1, [1, 1, 1], failures=[False, True, False])
    assert monitor.task_success_rate() == 0.5
    rng = torch.get_rng_state()
    _drain(monitor)
    monitor.reset(torch.tensor([0]), mode, 1)
    _drain(monitor)
    assert monitor.task_success_rate() == 0.5
    assert torch.equal(rng, torch.get_rng_state())
    _sample(monitor, mode, valid=torch.tensor([True, False, False]))
    _finish(monitor, [0], 2, [1, 1, 1], failures=[True, False, False])
    assert monitor.task_success_rate() == 0  # The latest failure replaces env 0's success.
    _finish(monitor, [0], 2, [1, 1, 1])  # Duplicate finalization cannot replace it again.
    assert monitor.task_success_rate() == 0
    monitor.clear_task_success()
    assert not monitor.latest_joint_success_valid.any()


@pytest.mark.parametrize("failure", ["height", "velocity", "yaw", "pelvis", "termination"])
def test_curriculum_snapshot_and_logged_joint_success_share_the_same_definition(failure):
    monitor, mode = _monitor([0, 1, 2])
    zeros = torch.zeros(3)
    error = torch.tensor([0.0, 1.0, 0.0])
    monitor.sample(
        mode,
        torch.ones(3, dtype=torch.bool),
        error if failure == "height" else zeros,
        error if failure == "velocity" else zeros,
        error if failure == "yaw" else zeros,
        error if failure == "pelvis" else zeros,
        torch.zeros(3, 6),
    )
    _finish(monitor, [0, 1, 2], 1, [1, 1, 1], failures=[False, failure == "termination", False])
    assert monitor.task_success_rate() == pytest.approx(2 / 3)
    metrics = _drain(monitor)
    assert metrics["Performance/stand/joint_success_rate"] == 1
    assert metrics["Performance/squat/joint_success_rate"] == 0
    assert metrics["Performance/walk/joint_success_rate"] == 1


def test_ineligible_episodes_keep_the_previous_valid_snapshot():
    monitor, mode = _monitor([0, 1, 2])
    _sample(monitor, mode)
    _finish(monitor, [0, 1, 2], 1, [1, 1, 1])
    monitor.reset(torch.arange(3), mode, 1)
    _sample(monitor, mode)
    mode[2] = 3
    _sample(monitor, mode)
    _finish(monitor, [0, 1, 2], 3, [2, 6, 2], timeouts=[False, True, True])
    assert monitor.latest_joint_success.all()
    assert monitor.latest_joint_success_valid.all()
    assert monitor.task_success_rate() == 1
    monitor.clear_task_success()
    assert not monitor.latest_joint_success_valid.any()
    assert monitor.task_success_rate() == 0


@pytest.mark.parametrize("selection", [None, slice(None), slice(1, 2), [1]])
def test_actual_disturbance_observer_preserves_targets_and_rng(monkeypatch, selection):
    from g1.tasks.lower_limb_agent.mdp.events import UpperBodyDisturbanceEvent

    monitor, mode = _monitor([0, 3])
    term = SimpleNamespace(performance=monitor, mode=mode)
    env = SimpleNamespace(
        command_manager=SimpleNamespace(active_terms=["task"], get_term=lambda name: term),
        upper_body_motion=SimpleNamespace(set_goal=lambda *args: None),
    )
    event = UpperBodyDisturbanceEvent.__new__(UpperBodyDisturbanceEvent)
    event._env = SimpleNamespace(device="cpu")
    event._joint_ids, event._num_joints, event._group_sizes = [0], 1, [1]
    event.joint_target = torch.zeros(2, 1)
    event.robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=torch.zeros(2, 1)),
            joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-1.0, 1.0]], [[-1.0, 1.0]]])),
        ),
        set_joint_position_target_index=lambda **kwargs: None,
    )
    groups = [{"joint_noise_frac": 0.2}]
    rng = torch.get_rng_state()
    event(env, selection, groups)
    targets, after = event.joint_target.clone(), torch.get_rng_state()
    metrics = _drain(monitor)
    expected = 2 if selection is None or selection == slice(None) else 1
    assert (
        metrics["Performance/stand/disturbance_count"] + metrics["Performance/squat_walk/disturbance_count"] == expected
    )
    monkeypatch.setattr(monitor, "disturbance", lambda *args: None)
    torch.set_rng_state(rng)
    event(env, selection, groups)
    torch.testing.assert_close(event.joint_target, targets)
    assert torch.equal(after, torch.get_rng_state())
