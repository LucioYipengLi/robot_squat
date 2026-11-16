"""Fixed-window scores, exposure-normalized gates and pre-reset reward rates."""

from types import SimpleNamespace

import pytest
import torch
from test_planar_timing import make_command
from test_upper_range_curriculum import FINAL

from isaaclab.envs import ManagerBasedRLEnv

from g1.tasks.upper_limb_agent.env import UpperLimbMobileEnv
from g1.tasks.upper_limb_agent.mdp.episode_metrics import failure_per_10s
from g1.tasks.upper_limb_agent.mdp.range_curriculum import RangeCurriculum, RangeCurriculumCfg


def scoring_command():
    term = make_command()
    term.mode_id[:] = torch.tensor([0, 1, 2])
    term.evaluation_start_steps[:] = 2
    term.episode_deadline_steps[:] = 6
    term.pose_error = lambda: (
        torch.tensor([[0.001, 99], [99, 0.001], [0.001, 0.001]]),
        torch.tensor([[0.01, 3], [3, 0.01], [0.01, 0.01]]),
    )
    return term


def test_full_window_has_exact_sample_count_and_repeated_reads_do_not_double_count():
    term = scoring_command()
    for step in range(7):
        term._env.common_step_counter = step
        term.collect_window_metrics()
        term.collect_window_metrics()
    torch.testing.assert_close(term._pose_window.sums[:, 4], torch.full((3,), 4.0))
    torch.testing.assert_close(term.episode_quality(torch.arange(3)), torch.ones(3))
    logs = term._planar_metrics(torch.arange(3), "task/planar")
    assert logs["task/planar/eval_coverage_fraction"] == 1
    assert logs["task/planar/eval_position_error_m"] == pytest.approx(0.001)
    assert logs["task/planar/eval_orientation_error_rad"] == pytest.approx(0.01)


def test_initial_reset_before_first_physics_step_has_no_termination_buffers_or_samples():
    term = scoring_command()
    del term._env.reset_terminated
    term.collect_window_metrics()
    assert not term._pose_window.sums.any()


@pytest.mark.parametrize("failure_step,score,coverage", [(1, 0, 0), (3, 0.25, 0.25), (5, 0.75, 0.75)])
def test_early_failure_and_missing_future_window_are_not_removed_from_denominator(failure_step, score, coverage):
    term = scoring_command()
    for step in range(failure_step + 1):
        term._env.common_step_counter = step
        term._env.reset_terminated[:] = step == failure_step
        term.collect_window_metrics()
    torch.testing.assert_close(term.episode_quality(torch.arange(3)), torch.full((3,), float(score)))
    logs = term._planar_metrics(torch.arange(3), "p")
    assert logs["p/eval_coverage_fraction"] == coverage
    if failure_step == 1:
        assert "p/eval_position_error_m" not in logs
    else:
        assert logs["p/eval_position_error_m"] == pytest.approx(0.001)


def test_partial_reset_clears_window_only_for_selected_environment():
    term = scoring_command()
    term._env.common_step_counter = 2
    term.collect_window_metrics()
    before = term._pose_window.sums.clone()
    term.reset(torch.tensor([0]))
    assert not term._pose_window.sums[0].any()
    torch.testing.assert_close(term._pose_window.sums[1:], before[1:])


def test_actual_elapsed_time_ignores_ppo_fabricated_episode_offset_even_without_planar():
    term = make_command(probability=0)
    term._env.episode_length_buf.fill_(999999)
    term._env.common_step_counter = 120
    torch.testing.assert_close(term.elapsed_seconds(torch.arange(3)), torch.full((3,), 2.0))


def test_position_readiness_is_recorded_when_height_task_is_disabled():
    term = scoring_command()
    term.cfg.height_task = None
    term._env.common_step_counter = 2
    term._update_metrics()
    torch.testing.assert_close(term._position_ready_sum, torch.ones(3))


def test_failure_frequency_counts_real_robot_seconds_not_number_of_episodes():
    result = failure_per_10s(torch.tensor([1, 3, 6]), torch.tensor([1000.0, 3000.0, 3000.0]))
    torch.testing.assert_close(result, torch.tensor([0.01, 0.01, 0.02]))
    assert failure_per_10s(torch.tensor(0), torch.tensor(0.0)) == 0


def course(**changes):
    cfg = RangeCurriculumCfg(check_period_steps=10, min_episodes_per_class=1, ema_alpha=1, **changes)
    return RangeCurriculum(cfg, FINAL, (-0.15, -0.05), (True, True, True), 300, "cpu")


def complete_three(c, failures=3, planar_score=1):
    categories = torch.arange(3).repeat_interleave(100)
    elapsed = torch.where(categories == 2, 30.0, 10.0)
    failed = torch.zeros(300, dtype=torch.bool)
    failed[0] = failed[100] = True
    failed[200 : 200 + failures] = True
    scores = torch.where(categories == 2, float(planar_score), 1.0)
    c.complete(torch.arange(300), categories, scores, failed, 10, elapsed)


def test_three_class_progression_checks_pose_quality_and_equal_time_exposure():
    c = course()
    complete_three(c)
    assert c.level == 1 and c.sample_ready and c.quality_ready and c.failure_ready
    torch.testing.assert_close(c.window_failure_rate, torch.tensor([0.01, 0.01, 0.03]))
    torch.testing.assert_close(c.window_failure_per_10s, torch.full((3,), 0.01))
    c = course()
    complete_three(c, failures=4)
    assert c.level == 0 and c.quality_ready and not c.failure_ready
    c = course()
    complete_three(c, failures=0, planar_score=0.6)
    assert c.level == 0 and not c.quality_ready and c.failure_ready


def test_frozen_level_still_updates_three_class_diagnostics_and_state_restores_exposure():
    c = course(advance=False)
    complete_three(c)
    assert c.level == 0 and c.quality_ready and c.failure_ready
    assert "Curriculum/upper_ranges/planar/window_failure_per_10s" in c.logs()
    c.complete(torch.tensor([0]), torch.tensor([2]), torch.ones(1), torch.tensor([True]), 11, torch.tensor([7.0]))
    saved = c.state_dict(11)
    restored = course(advance=False)
    restored.load_state_dict(saved, 0)
    torch.testing.assert_close(restored.elapsed_s, torch.tensor([0.0, 0.0, 7.0]))
    torch.testing.assert_close(restored.window_failure_per_10s, c.window_failure_per_10s)
    assert restored.clock_offset == 11 and saved["schema_version"] == 2
    saved["schema_version"] = 1
    with pytest.raises(ValueError, match="reset_on_load"):
        restored.load_state_dict(saved, 0)


@pytest.mark.parametrize("slice_ids", [False, True])
def test_reset_adapter_reads_terminal_pose_before_reset_and_normalizes_each_episode(monkeypatch, slice_ids):
    term = scoring_command()
    term._env.common_step_counter = 5
    for step in (2, 3, 4):
        term._env.common_step_counter = step
        term.collect_window_metrics()
    term._env.common_step_counter = 5
    raw = object.__new__(UpperLimbMobileEnv)
    raw.command_manager = SimpleNamespace(get_term=lambda name: term)
    values = torch.tensor([100.0, 4.0, 20.0])
    raw.reward_manager = SimpleNamespace(_episode_sum_buf=values[:, None], active_terms=["test"])
    raw.extras = {"log": {}}
    term.elapsed_seconds = lambda ids: torch.tensor([0.0, 2.0, 5.0])[ids]

    def parent_reset(self, ids):
        self.reward_manager._episode_sum_buf.zero_()
        term.pose_error = lambda: (torch.full((3, 2), 99.0), torch.full((3, 2), 3.0))
        self.extras["log"] = {"Episode_Reward/test": -99.0, "other": 2}

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", parent_reset)
    raw._reset_idx(slice(None) if slice_ids else torch.arange(3))
    assert raw.extras["log"]["Episode_Reward/test"] == 3  # mean(4/2, 20/5), excluding the initial reset.
    assert raw.extras["log"]["other"] == 2 and not values.any()
    torch.testing.assert_close(term.episode_quality(torch.arange(3)), torch.ones(3))
