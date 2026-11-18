"""Task sampling feedback: real exposure, consecutive windows, cooldown and resume."""

import copy

import pytest
import torch
from test_planar_timing import make_command
from test_upper_range_curriculum import FINAL

from g1.tasks.upper_limb_agent.mdp.adaptive_sampling import AdaptivePlanarSampling, AdaptivePlanarSamplingCfg
from g1.tasks.upper_limb_agent.mdp.range_curriculum import RangeCurriculum, RangeCurriculumCfg


def sampler(**changes):
    return AdaptivePlanarSampling(AdaptivePlanarSamplingCfg(**changes), 0.2)


def observe(s, failures=3, seconds=1000, eligible=True, other_tasks_quality_ready=True):
    s.observe(failures, seconds, eligible, 0.01, other_tasks_quality_ready=other_tasks_quality_ready)


def test_flat_failure_increases_small_steps_keeps_history_and_skips_exactly_one_check():
    s = sampler()
    for _ in range(3):
        observe(s)
        assert s.probability == 0.2
    observe(s)
    assert s.probability == 0.25 and s.cooldown == 1 and len(s.history) == 4
    observe(s, 4)
    assert s.probability == 0.25 and s.cooldown == 0 and s.history[-1][0] == 4
    observe(s, 4)
    assert s.probability == 0.3 and len(s.history) == 4
    for _ in range(30):
        observe(s, 4)
    assert s.probability == 0.3 and s.adjustments == 2


def test_other_task_quality_blocks_increases_but_keeps_history_and_resumes_without_warmup():
    s = sampler()
    for _ in range(6):
        observe(s, other_tasks_quality_ready=False)
    assert s.probability == 0.2 and s.adjustments == 0 and len(s.history) == 4
    observe(s)
    assert s.probability == 0.25 and s.cooldown == 1
    observe(s, other_tasks_quality_ready=False)
    assert s.probability == 0.25 and s.cooldown == 0
    observe(s)
    assert s.probability == 0.3


def test_meaningful_progress_blocks_increase_even_while_failure_gate_is_not_met():
    s = sampler()
    for n in (8, 8, 5, 5):
        observe(s, n)
    assert s.probability == 0.2
    assert s.trend(0.01) == pytest.approx((0.08, 0.05, 0.375))
    # A sample increase can raise raw failure counts while its normalized rate improves.
    s = sampler()
    for count, exposure in [(10, 1000), (10, 1000), (15, 2000), (15, 2000)]:
        observe(s, count, exposure)
    assert s.probability == 0.2 and s.trend(0.01)[2] == pytest.approx(0.25)


def test_current_failure_gate_stops_increase_and_insufficient_windows_break_history():
    s = sampler()
    for _ in range(3):
        observe(s, 0)
    observe(s, 1)  # Exactly the failure threshold, despite a worsening relative trend.
    assert s.probability == 0.2
    observe(s, eligible=False)
    assert not s.history and s.trend(0.01) is None


def test_trend_pools_exposure_instead_of_averaging_window_ratios():
    s = sampler()
    for count, seconds in [(1, 10), (1, 990), (2, 1000), (2, 1000)]:
        observe(s, count, seconds)
    assert s.trend(0.01) == pytest.approx((0.02, 0.02, 0))
    assert s.probability == 0.25


def course(advance=True, required=(True, True, True)):
    cfg = RangeCurriculumCfg(
        min_episodes_per_class=1,
        check_period_steps=10,
        ema_alpha=1,
        advance=advance,
        planar_sampling=AdaptivePlanarSamplingCfg(),
    )
    return RangeCurriculum(cfg, FINAL, (-0.15, -0.05), required, 300, "cpu", planar_probability=0.2)


def complete(c, failures=3, scores=(1.0, 1.0, 1.0), counts=(100, 100, 100)):
    categories = torch.arange(3).repeat_interleave(torch.tensor(counts))
    ids = torch.arange(len(categories))
    c.begin(ids)
    failed = torch.zeros(len(ids), dtype=torch.bool)
    start = sum(counts[:2])
    failed[start : start + failures] = True
    step = c.last_check + c.cfg.check_period_steps - c.clock_offset
    c.complete(ids, categories, torch.tensor(scores)[categories], failed, step, torch.full((len(ids),), 10.0))


@pytest.mark.parametrize("category", [0, 1])
def test_each_other_task_must_meet_ema_gate_and_exact_threshold_allows_sampling(category):
    c = course()
    scores = [0.85, 0.85, 0.1]  # Planar quality does not gate sampling increases.
    scores[category] = 0.849
    for _ in range(4):
        complete(c, scores=scores)
    assert c.planar_probability == 0.2
    assert c.logs()["Curriculum/upper_sampling/other_tasks_quality_ready"] == 0
    scores[category] = 0.85
    complete(c, scores=scores)
    assert c.planar_probability == 0.25
    assert c.logs()["Curriculum/upper_sampling/other_tasks_quality_ready"] == 1


@pytest.mark.parametrize("category", [0, 1])
@pytest.mark.parametrize("initialized", [False, True])
def test_missing_other_task_samples_block_sampling_even_with_previously_good_ema(category, initialized):
    c = course()
    if initialized:
        complete(c)
        assert c.ema[category] == 1
    counts = [100, 100, 100]
    counts[category] = 0
    for _ in range(4):
        complete(c, counts=counts)
    assert c.planar_probability == 0.2
    assert c.logs()["Curriculum/upper_sampling/other_tasks_quality_ready"] == 0
    complete(c)
    assert c.planar_probability == 0.25


def test_disabled_height_task_does_not_block_sampling():
    c = course(required=(True, False, True))
    for _ in range(4):
        complete(c, scores=(0.85, 0.0, 0.1), counts=(100, 0, 100))
    assert c.planar_probability == 0.25


def test_range_promotion_still_uses_original_quality_gate_below_sampling_protection_gate():
    c = course()
    complete(c, failures=0, scores=(0.82, 0.82, 0.82))
    assert c.level == 1 and c.planar_probability == 0.2
    assert c.logs()["Curriculum/upper_sampling/other_tasks_quality_ready"] == 0


def test_level_up_takes_priority_over_cooldown_and_restores_base_and_fresh_history():
    c = course()
    for _ in range(4):
        complete(c)
    assert c.planar_probability == 0.25 and c.planar_sampling.cooldown == 1
    complete(c, 0)
    assert c.level == 1 and c.planar_probability == 0.2
    assert not c.planar_sampling.history and c.planar_sampling.cooldown == 0
    assert c.logs()["Curriculum/upper_sampling/other_tasks_quality_ready"] == 0
    for _ in range(3):
        complete(c)
    assert c.planar_probability == 0.2


def test_fixed_training_level_can_adapt_and_evaluation_freezes_both_states():
    c = course(advance=False)
    for _ in range(4):
        complete(c)
    assert c.level == 0 and c.planar_probability == 0.25
    state = copy.deepcopy(c.state_dict(c.last_check))
    c.evaluation = True
    for _ in range(4):
        complete(c)
    assert c.planar_sampling.state_dict() == state["planar_sampling"]
    torch.testing.assert_close(c.counts, state["counts"])


def test_resume_preserves_partial_window_history_and_cooldown_and_supports_old_checkpoints():
    c = course()
    for _ in range(4):
        complete(c)
    c.complete(torch.tensor([0]), torch.tensor([2]), torch.ones(1), torch.tensor([True]), 41, torch.tensor([3.0]))
    saved = c.state_dict(41)
    restored = course()
    restored.load_state_dict(saved, 0)
    assert restored.planar_sampling.state_dict() == c.planar_sampling.state_dict()
    torch.testing.assert_close(restored.elapsed_s, c.elapsed_s)
    complete(restored)
    assert restored.planar_probability == 0.25 and restored.planar_sampling.cooldown == 0
    complete(restored)
    assert restored.planar_probability == 0.3
    legacy = copy.deepcopy(saved)
    legacy.pop("planar_sampling")
    restored.load_state_dict(legacy, 0)
    assert restored.level == c.level and restored.planar_probability == 0.2
    assert not restored.planar_sampling.history
    changed = course()
    changed.planar_sampling.cfg.probability_step = 0.1
    changed.load_state_dict(saved, 0)
    assert changed.planar_probability == 0.2 and not changed.planar_sampling.history


def test_old_forty_percent_schedule_restarts_without_losing_range_statistics():
    c = course()
    for _ in range(4):
        complete(c)
    saved = copy.deepcopy(c.state_dict(c.last_check))
    saved["planar_sampling"]["cfg"].pop("other_tasks_pose_ready_gate")
    saved["planar_sampling"]["cfg"]["maximum_probability"] = 0.4
    saved["planar_sampling"]["probability"] = 0.4
    restored = course()
    restored.load_state_dict(saved, 0)
    assert restored.level == c.level and restored.planar_probability == 0.2
    assert not restored.planar_sampling.history
    torch.testing.assert_close(restored.ema, saved["ema"])
    torch.testing.assert_close(restored.window_counts, saved["window_counts"])


def test_sampler_probability_drives_only_new_resets_without_mutating_contract():
    def configure(cfg, env):
        cfg.range_curriculum.planar_sampling = AdaptivePlanarSamplingCfg()

    term = make_command(probability=0.2, configure=configure)
    contract = copy.deepcopy(term.range_curriculum.contract())
    term.range_curriculum.planar_sampling.probability = 0.3
    # Identical draws isolate the class-selection threshold from the unchanged sampler.
    torch.manual_seed(1)
    draws = torch.rand(3)
    torch.manual_seed(1)
    term._sample_task_kind(torch.arange(3))
    torch.testing.assert_close(term.planar_task_mask, draws < 0.3)
    goals, classes = term.goal_pose_w.clone(), term.task_ids.clone()
    term.range_curriculum.planar_sampling.probability = 0.25
    torch.testing.assert_close(term.goal_pose_w, goals)
    torch.testing.assert_close(term.task_ids, classes)
    assert term.cfg.planar_task.probability == 0.2 and term.range_curriculum.contract() == contract


@pytest.mark.parametrize("probability", [0, 1])
def test_pure_task_configs_skip_adaptation(probability):
    def configure(cfg, env):
        cfg.range_curriculum.planar_sampling = AdaptivePlanarSamplingCfg()

    term = make_command(probability=probability, configure=configure)
    assert term.range_curriculum.planar_sampling is None


def test_disabled_or_warming_sampler_preserves_base_goals_and_rng():
    baseline = make_command(probability=0.2)
    torch.manual_seed(119)
    baseline.reset()
    goals, state = baseline.goal_pose_w.clone(), torch.get_rng_state()
    for enabled in (False, True):

        def configure(cfg, env, enabled=enabled):
            cfg.range_curriculum.planar_sampling = AdaptivePlanarSamplingCfg(enabled=enabled)

        other = make_command(probability=0.2, configure=configure)
        torch.manual_seed(119)
        other.reset()
        torch.testing.assert_close(other.goal_pose_w, goals, atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("history_windows", 3),
        ("history_windows", 0),
        ("cooldown_windows", -1),
        ("probability_step", float("nan")),
        ("probability_step", 0),
        ("maximum_probability", 0.1),
        ("maximum_probability", 1),
        ("other_tasks_pose_ready_gate", -0.01),
        ("other_tasks_pose_ready_gate", 1.01),
        ("other_tasks_pose_ready_gate", float("nan")),
        ("minimum_relative_improvement", 0),
    ],
)
def test_invalid_configuration_fails_at_creation(name, value):
    with pytest.raises(ValueError):
        sampler(**{name: value})


def test_bad_checkpoint_history_is_rejected_and_logs_have_a_separate_group():
    s = sampler()
    for _ in range(4):
        observe(s)
    assert s.logs(0.01)["Curriculum/upper_sampling/planar_probability"] == 0.25
    saved = s.state_dict()
    saved["history"] = [(1, 0)]
    with pytest.raises(ValueError, match="history"):
        sampler().load_state_dict(saved)
