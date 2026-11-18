"""Planar target geometry, immutable time budgets and reset-safe deadlines."""

import math
from types import SimpleNamespace

import pytest
import torch
from test_upper_limb_task import pose_command as _pose_command

from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, quat_from_euler_xyz, quat_mul

from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import (
    MobileActionsCfg,
    MobileCommandsCfg,
    MobileTerminationsCfg,
)
from g1.tasks.upper_limb_agent.mdp.commands import BimanualPoseCommand, BimanualPoseCommandCfg
from g1.tasks.upper_limb_agent.mdp.goal_timing import GoalTimingCfg, budget_steps, rest_to_rest_time
from g1.tasks.upper_limb_agent.mdp.planar_targets import PlanarTaskCfg, sample_station, station_limits
from g1.tasks.upper_limb_agent.mdp.range_curriculum import RangeCurriculumCfg
from g1.tasks.upper_limb_agent.mdp.terminations import task_time_out


def make_command(probability=1.0, level=0, hard_s=35, advance=False, configure=None):
    base = _pose_command.__wrapped__()
    env = base._env
    env.step_dt = 1 / 60
    env.cfg = SimpleNamespace(actions=MobileActionsCfg(), episode_length_s=hard_s)
    env.max_episode_length = math.ceil(hard_s / env.step_dt)
    env.common_step_counter = 0
    env.extras = {}
    env.reset_terminated = torch.zeros(3, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(3, dtype=torch.bool)
    cfg = base.cfg.copy()
    cfg.height_task = BimanualPoseCommandCfg.HeightTaskCfg()
    cfg.planar_task = PlanarTaskCfg(probability=probability)
    cfg.range_curriculum = RangeCurriculumCfg(initial_level=level, advance=advance)
    if configure is not None:
        configure(cfg, env)
    term = BimanualPoseCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    term.reset()
    return term


@pytest.mark.parametrize("speed,acceleration", [(0.08, 0.3), (0.4, 0.3), (0.4, 0.4)])
def test_rest_to_rest_profile_integrates_to_distance_and_has_no_double_counted_start_stop(speed, acceleration):
    distance = torch.tensor([0, 0.001, speed**2 / acceleration, 1.0, 3.54, 5.0])
    duration = rest_to_rest_time(distance, speed, acceleration)
    peak = torch.minimum(torch.full_like(distance, speed), duration * acceleration / 2)
    accelerating = peak / acceleration
    cruising = (duration - 2 * accelerating).clamp_min(0)
    travelled = acceleration * accelerating.square() + peak * cruising
    torch.testing.assert_close(travelled, distance)
    assert duration[0] == 0 and torch.isfinite(duration).all()
    assert (duration[1:] > duration[:-1]).all()


def test_budget_rounds_up_and_keeps_full_independent_evaluation_window():
    cfg, dt = GoalTimingCfg(), 1 / 60
    d, angle = torch.tensor([0.0, 3.54, 5.0]), torch.tensor([0.0, 0.0, math.pi])
    start, end = budget_steps(d, angle, torch.ones(3), torch.ones(3), cfg, dt)
    assert start[0] == 120 and end[0] == 720
    assert ((end - start) == 600).all()
    assert end[2] * dt == pytest.approx(34.7, abs=dt)
    slow_start, _ = budget_steps(d, angle, torch.full((3,), 0.1), torch.full((3,), 0.1), cfg, dt)
    assert (slow_start >= start).all()


@pytest.mark.parametrize("progress", [0.0, 0.5, 1.0])
def test_station_sampler_covers_bearing_sector_inside_square_and_radius(progress):
    cfg = PlanarTaskCfg()
    p = torch.full((8192,), progress)
    shift, angle = sample_station(p, torch.full_like(p, 0.4), cfg)
    half = 0.25 + 2.25 * progress
    bearing = math.pi / 6 + progress * (math.pi - math.pi / 6)
    assert (shift.abs() <= half + 1e-6).all()
    assert (shift.norm(dim=-1) <= 4.6 + 1e-6).all()
    assert (angle.abs() <= bearing + 1e-6).all()
    assert angle.min() < -bearing * 0.98 and angle.max() > bearing * 0.98
    assert shift[:, 0].max() > half * 0.95
    assert torch.isfinite(shift).all()


@pytest.mark.parametrize("level", range(11))
def test_mobile_planar_schedule_matches_knots_bounds_and_episode_budget(level):
    cfg = MobileCommandsCfg().wrist_pose.planar_task
    cfg.validate_parameters()
    extent, angle = station_limits(level / 10, cfg)
    assert float(extent) == pytest.approx(cfg.curriculum_knots[level][1])
    assert float(angle) == pytest.approx(cfg.curriculum_knots[level][2])
    p = torch.full((1024,), level / 10)
    shift, heading = sample_station(p, torch.full_like(p, 0.4), cfg)
    assert (shift.abs() <= extent + 1e-6).all()
    assert (heading.abs() <= angle + 1e-6).all()
    term = make_command(level=level, configure=lambda c, env: setattr(c, "planar_task", cfg))
    assert (term.episode_deadline_steps <= 35 * 60).all()
    assert ((term.episode_deadline_steps - term.evaluation_start_steps) == 600).all()


def test_mobile_planar_schedule_interpolates_monotonically_and_checkpoint_records_it():
    cfg = MobileCommandsCfg().wrist_pose.planar_task
    extent, angle = station_limits(torch.linspace(0, 1, 101), cfg)
    assert (extent.diff() >= 0).all() and (angle.diff() >= 0).all()
    middle_extent, middle_angle = station_limits(0.15, cfg)
    assert float(middle_extent) == pytest.approx(0.175)
    assert float(middle_angle) == pytest.approx(math.radians(17.5))
    assert float(extent[0]) == pytest.approx(0.1) and float(extent[-1]) == pytest.approx(2.5)
    term = make_command(configure=lambda c, env: setattr(c, "planar_task", cfg))
    saved = term.range_curriculum.state_dict(100)
    assert saved["contract"]["target"]["planar_task"]["curriculum_knots"] is not None
    with pytest.raises(ValueError, match="reset_on_load"):
        make_command().range_curriculum.load_state_dict(saved, 0)


def test_contact_threshold_change_rejects_old_curriculum_failure_statistics():
    def configure(cfg, env):
        env.cfg.terminations = MobileTerminationsCfg()

    term = make_command(configure=configure)
    saved = term.range_curriculum.state_dict(100)
    assert saved["contract"]["target"]["torso_contact"]["threshold_n"] == 20
    other = make_command(configure=configure)
    other.range_curriculum.target_contract["torso_contact"]["threshold_n"] = 10
    with pytest.raises(ValueError, match="reset_on_load"):
        other.range_curriculum.load_state_dict(saved, 0)


@pytest.mark.parametrize(
    "knots",
    [
        ((0, 0.1, 0),),
        ((0, 0.1, 0), (1, 2.5, math.pi)),  # First knot disagrees with the configured initial limits.
        ((0, 0.25, math.pi / 6), (0.5, 2, 1), (0.4, 2, 2), (1, 2.5, math.pi)),
        ((0, 0.25, math.pi / 6), (0.5, 0.1, 1), (1, 2.5, math.pi)),
        ((0, 0.25, math.pi / 6), (1, 2.5, float("nan"))),
    ],
)
def test_invalid_planar_schedule_is_rejected(knots):
    with pytest.raises(ValueError, match="curriculum_knots"):
        PlanarTaskCfg(curriculum_knots=knots).validate_parameters()


@pytest.mark.parametrize("level", [0, 5, 10])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_rigid_targets_preserve_local_poses_for_single_and_dual_across_world_origins(level, mode):
    def configure(cfg, env):
        cfg.mode_probabilities = tuple(float(i == mode) for i in range(3))

    term = make_command(level=level, configure=configure)
    assert term.planar_task_mask.all() and not term.height_task_mask.any()
    assert (term.command_counter == 1).all() and torch.isinf(term.time_left).all()
    # Undo the whole-group heading and the authored initial yaw. Neutral fixture
    # offsets then recover the known wrist coordinates, independently of origin.
    heading = quat_from_euler_xyz(torch.zeros(3), torch.zeros(3), term.planar_heading_rad)
    work_frame = quat_mul(heading, term._sampling_quat_w)[:, None].expand(-1, 2, -1)
    local = quat_apply_inverse(work_frame, term.goal_pose_w[..., :3] - term.planar_station_w[:, None])
    expected = torch.tensor([[0.2, 0.15, 0.1], [0.2, -0.15, 0.1]]).expand(3, -1, -1)
    torch.testing.assert_close(local, expected, atol=2e-6, rtol=1e-5)
    anchor = term._env.scene.env_origins + term._default_pelvis_pos_e
    assert ((term.goal_pose_w[..., :3] - anchor[:, None]).norm(dim=-1) <= 5).all()
    assert ((term.episode_deadline_steps - term.evaluation_start_steps) == 600).all()
    assert (term.episode_deadline_steps <= term._env.max_episode_length).all()
    if mode == 2:
        torch.testing.assert_close(
            (term.goal_pose_w[:, 0, :3] - term.goal_pose_w[:, 1, :3]).norm(dim=-1), torch.full((3,), 0.3)
        )
        torch.testing.assert_close(
            quat_error_magnitude(term.goal_pose_w[:, 0, 3:], term.goal_pose_w[:, 1, 3:]), torch.zeros(3)
        )


def test_timing_is_read_only_precision_independent_and_immune_to_random_episode_buffer():
    term = make_command()
    env = term._env
    start, end = int(term.evaluation_start_steps[0]), int(term.episode_deadline_steps[0])
    budget, goals = term.episode_deadline_steps.clone(), term.goal_pose_w.clone()
    env.episode_length_buf.fill_(999999)
    for age, in_window, timed_out in (
        (start - 1, False, False),
        (start, True, False),
        (end - 1, True, False),
        (end, False, True),
    ):
        env.common_step_counter = age
        assert bool(term.evaluation_window_mask()[0]) == in_window
        assert bool(task_time_out(env)[0]) == timed_out
        assert bool(task_time_out(env)[0]) == timed_out
    env.scene["robot"].data.body_link_pose_w.torch[:, 1, :3] += 3
    term.time_left.zero_()  # Forced expiry still cannot replace an episode-held target.
    term._resample(torch.arange(3))
    torch.testing.assert_close(term.goal_pose_w, goals, atol=0, rtol=0)
    torch.testing.assert_close(term.episode_deadline_steps, budget, atol=0, rtol=0)
    assert (term.command_counter == 1).all()


def test_partial_reset_clears_only_selected_timer_and_restarts_a_full_budget():
    term = make_command()
    env = term._env
    before = {
        name: getattr(term, name).clone()
        for name in (
            "goal_pose_w",
            "_episode_start_step",
            "episode_deadline_steps",
            "evaluation_start_steps",
            "planar_station_w",
        )
    }
    env.common_step_counter = 100
    term.reset(torch.tensor([0]))
    assert term._episode_start_step[0] == 100
    assert term.episode_age_steps()[0] == 0 and not task_time_out(env)[0]
    assert term.episode_deadline_steps[0] - term.evaluation_start_steps[0] == 600
    for name, value in before.items():
        torch.testing.assert_close(getattr(term, name)[1:], value[1:], atol=0, rtol=0)


def test_zero_probability_preserves_rng_goals_and_legacy_short_timeout():
    base = _pose_command.__wrapped__()
    torch.manual_seed(200)
    base.reset()
    expected = base.goal_pose_w.clone(), torch.get_rng_state()
    base.cfg.planar_task = PlanarTaskCfg(probability=0)
    torch.manual_seed(200)
    base.reset()
    torch.testing.assert_close(base.goal_pose_w, expected[0], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected[1], atol=0, rtol=0)
    base._env.max_episode_length = 3
    base._env.episode_length_buf[:] = torch.tensor([2, 3, 4])
    torch.testing.assert_close(task_time_out(base._env), torch.tensor([False, True, True]))


def test_each_task_has_its_own_category_and_planar_uses_fixed_window_score(monkeypatch):
    term = make_command()
    env = term._env
    term.planar_task_mask[:] = torch.tensor([True, False, False])
    term.height_task_mask[:] = torch.tensor([False, False, True])
    env.reset_time_outs.fill_(True)
    env.episode_length_buf.fill_(10)
    env.common_step_counter = 10
    calls = []
    monkeypatch.setattr(term.range_curriculum, "complete", lambda ids, *args: calls.append(ids.tolist()))
    logs = term._task_metrics(torch.arange(3))
    assert logs["task/regular/episodes"] == logs["task/height_shifted/episodes"] == 1
    term.reset()
    assert calls == [[0, 1, 2]]


def test_enabled_planar_contract_is_saved_and_legacy_requires_restart():
    term = make_command()
    state = term.range_curriculum.state_dict(100)
    assert state["contract"]["target"]["planar_task"]["timing"]["evaluation_s"] == 10
    assert state["contract"]["target"]["timing_hard_limit_s"] == 35
    other = make_command().range_curriculum
    other.load_state_dict(state, 0)
    assert other.clock_offset == 100
    del state["contract"]["target"]["planar_task"]
    with pytest.raises(ValueError, match="reset_on_load"):
        other.load_state_dict(state, 0)


def test_short_hard_limit_fails_and_planar_course_advancement_can_be_enabled():
    with pytest.raises(ValueError, match="complete evaluation"):
        make_command(hard_s=10)
    assert make_command(advance=True).range_curriculum.cfg.advance


def test_goal_distance_cap_is_checked_on_active_wrists_and_rejects_no_room():
    term = make_command(configure=lambda cfg, env: setattr(cfg.planar_task, "max_goal_distance_m", 0.3))
    anchor = term._env.scene.env_origins + term._default_pelvis_pos_e
    assert ((term.goal_pose_w[..., :3] - anchor[:, None]).norm(dim=-1) <= 0.3 + 1e-6).all()
    with pytest.raises(ValueError, match="no room"):
        make_command(configure=lambda cfg, env: setattr(cfg.planar_task, "max_goal_distance_m", 0.1))


@pytest.mark.parametrize(
    "name,value",
    [
        ("probability", -0.1),
        ("probability", float("nan")),
        ("half_extent_m", 0),
        ("initial_half_extent_m", 3),
        ("bearing_rad", math.pi + 0.1),
        ("initial_bearing_rad", -1),
        ("max_goal_distance_m", float("inf")),
    ],
)
def test_invalid_space_configuration_fails_early(name, value):
    cfg = PlanarTaskCfg()
    setattr(cfg, name, value)
    with pytest.raises(ValueError, match="planar_task"):
        cfg.validate_parameters()


@pytest.mark.parametrize(
    "name",
    [
        "linear_speed",
        "angular_speed",
        "linear_acceleration",
        "angular_acceleration",
        "settle_s",
        "evaluation_s",
        "near_episode_s",
        "ability_fraction",
    ],
)
def test_invalid_time_configuration_fails_early(name):
    cfg = GoalTimingCfg()
    setattr(cfg, name, 0)
    with pytest.raises(ValueError, match="goal timing"):
        cfg.validate_parameters()


def test_mobile_defaults_enable_three_class_training_and_keep_runner_offset_setting():
    from g1.tasks.upper_limb_agent.config.g1.agents.mobile_rsl_rl_ppo_cfg import MobilePPORunnerCfg

    cfg = MobileCommandsCfg().wrist_pose.planar_task
    assert cfg.probability == 0.2 and cfg.half_extent_m == 2.5
    assert MobilePPORunnerCfg().init_at_random_ep_len


@pytest.mark.parametrize("enabled,requested,expected", [(True, True, False), (False, True, True), (True, False, False)])
def test_runner_disables_random_episode_offset_only_for_planar_tasks(monkeypatch, enabled, requested, expected):
    from rsl_rl.runners import OnPolicyRunner

    from g1.tasks.upper_limb_agent.runner import UpperLimbOnPolicyRunner

    command = SimpleNamespace(planar_enabled=enabled)
    manager = SimpleNamespace(active_terms=["wrist_pose"], get_term=lambda name: command)
    runner = object.__new__(UpperLimbOnPolicyRunner)
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=manager))
    calls = []
    monkeypatch.setattr(
        OnPolicyRunner, "learn", lambda self, n, init_at_random_ep_len: calls.append((n, init_at_random_ep_len))
    )
    runner.learn(2, init_at_random_ep_len=requested)
    assert calls == [(2, expected)]
