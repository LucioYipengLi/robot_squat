"""Planar target geometry, immutable time budgets and reset-safe deadlines."""

import math
from types import SimpleNamespace

import pytest
import torch
from test_upper_limb_task import pose_command as _pose_command

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude, quat_from_euler_xyz, quat_mul

from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import (
    MobileActionsCfg,
    MobileCommandsCfg,
    MobileTerminationsCfg,
)
from g1.tasks.upper_limb_agent.mdp.commands import BimanualPoseCommand, BimanualPoseCommandCfg
from g1.tasks.upper_limb_agent.mdp.goal_timing import GoalTimingCfg, budget_steps, rest_to_rest_time
from g1.tasks.upper_limb_agent.mdp.planar_targets import (
    PlanarStationSampler,
    PlanarTaskCfg,
    sample_station,
    sample_transition_station,
    station_budget_bounds,
    station_limits,
)
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


def test_station_schedule_lookup_preserves_draws_and_observes_live_overrides():
    cfg = MobileCommandsCfg().wrist_pose.planar_task.copy()
    transition = cfg.transition_sampling
    original_knots, original_bridge = transition.probability_knots, transition.turn_bearing_bridge
    sampler = PlanarStationSampler(11, "cpu")
    levels = torch.arange(11).repeat_interleave(31)
    progress = levels.float() / 10
    local_radius = torch.full_like(progress, 0.4)
    for enabled, knots, bridge in (
        (False, None, None),
        (True, None, None),
        (True, original_knots, original_bridge),
        (False, None, None),
    ):
        transition.enabled, transition.probability_knots, transition.turn_bearing_bridge = enabled, knots, bridge
        torch.manual_seed(42)
        if enabled:
            expected = sample_transition_station(progress, local_radius, cfg)
        else:
            expected = (*sample_station(progress, local_radius, cfg), None)
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(42)
        actual = sampler.sample(levels, local_radius, cfg)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert torch.equal(torch.random.get_rng_state(), expected_rng)


@pytest.mark.parametrize("level,low_deg,high_deg", [(5, 45, 65), (6, 65, 90), (10, 160, 180)])
def test_turn_focus_samples_new_bearings_and_reserves_the_native_group_mix(level, low_deg, high_deg):
    command = MobileCommandsCfg().wrist_pose
    cfg = command.planar_task
    focus = command.planar_turn_focus
    cfg.transition_sampling.enabled = True
    sampler = PlanarStationSampler(11, "cpu")
    levels = torch.full((8192,), level)
    local_radius = torch.full_like(levels, 0.4, dtype=torch.float32)

    torch.manual_seed(42)
    native_shift, native_angle, native_group = sampler.sample(levels, local_radius, cfg)
    native_rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    shift, angle, group, focused, native_reference = sampler.sample_turn_focus(levels, local_radius, cfg, focus)
    assert torch.equal(torch.random.get_rng_state(), native_rng)
    assert torch.equal(group, native_group)
    torch.testing.assert_close(shift[~focused], native_shift[~focused], atol=0, rtol=0)
    torch.testing.assert_close(angle[~focused], native_angle[~focused], atol=0, rtol=0)
    assert not (focused & native_reference).any()
    assert (angle[focused].abs() >= math.radians(low_deg) - 1e-6).all()
    assert (angle[focused].abs() <= math.radians(high_deg) + 1e-6).all()
    assert ((angle[focused] > 0).float().mean()).item() == pytest.approx(0.5, abs=0.03)

    turn, wide = group == 2, group == 3
    assert not (focused & (group == 1)).any()
    if turn.any():
        assert focused[turn].float().mean().item() == pytest.approx(0.70, abs=0.04)
        assert (shift[turn].norm(dim=-1) <= 0.30 + 1e-6).all()
    assert focused[wide].float().mean().item() == pytest.approx(0.40, abs=0.04)
    assert (shift[wide].abs() <= station_limits(level / 10, cfg)[0] + 1e-6).all()
    for index in (1, 2, 3):
        native_share = (group == index).float().mean().item()
        assert (group[native_reference] == index).float().mean().item() == pytest.approx(native_share, abs=0.03)
    assert native_reference.float().mean().item() == pytest.approx(0.30 if level < 10 else 0.60, abs=0.03)


def configure_turn_focus(cfg, _env):
    mobile = MobileCommandsCfg().wrist_pose
    cfg.planar_task = mobile.planar_task.copy()
    cfg.planar_task.probability = 1.0
    cfg.planar_task.transition_sampling.enabled = True
    cfg.planar_turn_focus = mobile.planar_turn_focus.copy()


@pytest.mark.parametrize("level,evaluation", [(level, False) for level in range(5)] + [(6, True)])
def test_turn_focus_preserves_native_command_targets_and_rng_before_level_five_and_in_evaluation(level, evaluation):
    term = make_command(level=level, configure=configure_turn_focus)
    term.range_curriculum.evaluation = evaluation

    torch.manual_seed(311)
    term.reset()
    goal = term.goal_pose_w.clone()
    heading = term.planar_heading_rad.clone()
    expected_rng = torch.random.get_rng_state()
    assert not term._planar_turn_focused.any()
    assert term._planar_native_reference.all()

    term.cfg.planar_turn_focus.enabled = False
    torch.manual_seed(311)
    term.reset()
    torch.testing.assert_close(term.goal_pose_w, goal, atol=0, rtol=0)
    torch.testing.assert_close(term.planar_heading_rad, heading, atol=0, rtol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


def test_turn_focus_uses_only_native_reference_for_promotion_and_logs_early_contact():
    def configure(cfg, env):
        configure_turn_focus(cfg, env)
        cfg.planar_task.probability = 0.50

    term = make_command(level=5, configure=configure)
    env, course = term._env, term.range_curriculum
    saved = course.state_dict(0)
    assert "planar_turn_focus" not in saved["contract"]["target"]
    term.cfg.planar_turn_focus.enabled = False
    course.load_state_dict(saved, 0)
    term.cfg.planar_turn_focus.enabled = True

    term.planar_task_mask[:] = torch.tensor([True, True, False])
    term.height_task_mask[:] = False
    term.planar_sampling_group[:] = torch.tensor([2, 3, 0])
    term._planar_turn_focused[:] = torch.tensor([True, False, False])
    term._planar_native_reference[:] = ~term._planar_turn_focused
    term.planar_first_near_step[1] = 3
    env.common_step_counter = 6
    env.reset_terminated[:] = torch.tensor([True, False, False])
    env.reset_time_outs[:] = ~env.reset_terminated
    env.termination_manager = SimpleNamespace(active_terms=["base_contact"], get_term=lambda _: env.reset_terminated)

    extras = term.reset()
    assert course.counts.tolist() == [1, 0, 1]
    assert course.failures.tolist() == [0, 0, 0]
    assert extras["task/planar/turn_focus/turn/episodes"] == 1
    assert extras["task/planar/turn_focus/turn/early_base_contact_fraction"] == 1
    assert extras["task/planar/turn_focus/turn/near_entry_fraction"] == 0


def test_regular_retract_refresh_is_bounded_per_arm_and_falls_back_at_rear_edge():
    def configure(cfg, _):
        cfg.mode_probabilities = (1.0, 0.0, 0.0)
        cfg.height_task.probability = 0.0
        for arm in (cfg.left_offsets, cfg.right_offsets):
            arm.pos_x, arm.pos_y, arm.pos_z = (-0.10, 0.10), (-0.03, 0.03), (-0.03, 0.03)
        cfg.regular_retract_sampling.probability = 1.0

    torch.manual_seed(42)
    term = make_command(probability=0.0, level=3, configure=configure)
    assert not term._regular_retract_counts.any()
    term.mode_id[:] = torch.tensor([0, 1, 2])
    ids, sides = torch.arange(2), torch.arange(2)
    anchor = term._env.scene.env_origins[ids] + term.default_wrist_pose_e[ids, sides, :3]
    frame = term._sampling_quat_w[ids]
    previous = anchor + quat_apply(frame, torch.tensor([[0.064, 0.0, 0.0]]).expand(2, -1))
    term.goal_pose_w[ids, sides, :3] = previous
    term._resample(torch.arange(3))
    new = term.goal_pose_w[ids, sides, :3]
    delta = quat_apply_inverse(frame, new - previous)
    assert (delta[:, 0] <= -0.03).all() and ((delta.norm(dim=1) >= 0.05) & (delta.norm(dim=1) < 0.12)).all()
    assert (quat_apply_inverse(frame, new - anchor).abs() <= torch.tensor([0.065, 0.0195, 0.0195]) + 1e-6).all()
    assert term._regular_retract_counts.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]

    term.goal_pose_w[0, 0, :3] = anchor[0] + quat_apply(frame[0], torch.tensor([-0.065, 0.0, 0.0]))
    term._resample(torch.tensor([0]))
    assert term._regular_retract_counts[0].tolist() == [2, 2, 1, 1]
    log = term.reset(torch.tensor([0]))
    assert (
        log["task/regular/retract_sampling/fallbacks"] == 1
        and log["task/regular/retract_sampling/retract_fraction"] == 0.5
    )
    assert term._regular_retract_counts.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]


def test_dual_retract_uses_complete_native_candidates_and_preserves_fallback():
    def configure(cfg, _):
        cfg.mode_probabilities = (0.0, 0.0, 1.0)
        cfg.height_task.probability = 0.0
        cfg.regular_retract_sampling.dual_probability = 1.0
        cfg.regular_retract_sampling.candidates = 128

    torch.manual_seed(42)
    term = make_command(probability=0.0, level=3, configure=configure)
    assert not term._dual_retract_counts.any()
    ids = torch.arange(3)
    frame = term._sampling_quat_w
    anchor = term._env.scene.env_origins[:, None] + term.default_wrist_pose_e[..., :3]
    offsets = torch.tensor([[0.0975, 0.0, 0.0], [-0.08, 0.0, 0.0], [0.0, 0.0, 0.0]])
    previous = anchor + quat_apply(frame, offsets)[:, None]
    term.goal_pose_w[..., :3] = previous
    term.planar_task_mask[2] = True
    rng_state = torch.random.get_rng_state()
    native_pos, native_rot = term._draw_dual_targets(ids)
    torch.rand(2)  # One Bernoulli draw per eligible Regular goal refresh.
    candidates = term.cfg.regular_retract_sampling.candidates
    candidate_pos, candidate_rot = term._draw_dual_targets(ids[:2].repeat_interleave(candidates))
    expected_rng = torch.random.get_rng_state()
    delta = quat_apply_inverse(frame[0], candidate_pos[:candidates] - previous[0])
    jump = delta.norm(dim=-1).amax(dim=-1)
    first = ((delta[..., 0] <= -0.03).all(dim=-1) & (jump >= 0.08) & (jump < 0.12)).nonzero()[0, 0]
    torch.random.set_rng_state(rng_state)

    term._sample_dual_targets(ids)

    delta = quat_apply_inverse(frame[0], term.goal_pose_w[0, :, :3] - previous[0])
    assert (delta[:, 0] <= -0.03).all()
    assert 0.08 <= delta.norm(dim=-1).max() < 0.12
    torch.testing.assert_close(term.goal_pose_w[0, :, :3], candidate_pos[first], rtol=0, atol=0)
    torch.testing.assert_close(term.goal_pose_w[0, :, 3:], candidate_rot[first], rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    torch.testing.assert_close(term.goal_pose_w[1:, :, :3], native_pos[1:])
    torch.testing.assert_close(term.goal_pose_w[1:, :, 3:], native_rot[1:])
    assert term._dual_retract_counts.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]
    log = term.reset(torch.tensor([1]))
    assert log["task/regular/dual_retract_sampling/fallbacks"] == 1
    assert term._dual_retract_counts.tolist() == [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_regular_retract_sampling_is_only_in_enabled_checkpoint_contracts():
    control = make_command()
    active = make_command(configure=lambda cfg, _: setattr(cfg.regular_retract_sampling, "probability", 0.3))
    saved = control.range_curriculum.state_dict(0)
    assert "regular_retract_sampling" not in saved["contract"]["target"]
    single = active.range_curriculum.contract()["target"]["regular_retract_sampling"]
    assert single["probability"] == 0.3
    assert "dual_probability" not in single and "dual_jump_range_m" not in single
    with pytest.raises(ValueError, match="reset_on_load"):
        active.range_curriculum.load_state_dict(saved, 0)
    dual_active = make_command(configure=lambda cfg, _: setattr(cfg.regular_retract_sampling, "dual_probability", 0.25))
    dual = dual_active.range_curriculum.contract()["target"]["regular_retract_sampling"]
    assert dual["dual_probability"] == 0.25 and dual["dual_jump_range_m"] == [0.08, 0.12]


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


@pytest.mark.parametrize("level", range(11))
def test_transition_sampling_keeps_the_distance_floor_and_opens_wide_groups_gradually(level):
    cfg = MobileCommandsCfg().wrist_pose.planar_task
    cfg.transition_sampling.enabled = True
    cfg.validate_parameters()
    torch.manual_seed(7)
    progress = torch.full((8192,), level / 10)
    shift, heading, group = sample_transition_station(progress, torch.full_like(progress, 0.4), cfg)
    distance = shift.norm(dim=-1)
    bearing = heading.abs()
    assert (distance >= 0.25 - 1e-6).all()
    walk, turn, wide = group == 1, group == 2, group == 3
    assert (distance[walk] <= 0.40 + 1e-6).all() and (bearing[walk] <= math.radians(15)).all()
    assert (distance[turn] <= 0.30 + 1e-6).all()
    extent, maximum_bearing = station_limits(level / 10, cfg)
    assert (shift[wide].abs() <= extent + 1e-6).all() and (bearing[wide] <= maximum_bearing).all()
    if level <= 3:
        assert not wide.any()
    if level == 10:
        assert wide.all()
    else:
        turn_limit = (25, 25, 35, 35, 45, 65, 90, 115, 140, 160)[level]
        assert (bearing[turn] >= math.radians(20)).all()
        assert float(bearing[turn].max()) == pytest.approx(math.radians(turn_limit), abs=math.radians(5))
        assert (bearing[turn] <= math.radians(turn_limit) + 1e-6).all()
        turn_distance, turn_budget_bearing = station_budget_bounds(level / 10, cfg)[1]
        assert turn_distance == pytest.approx(0.30)
        assert float(turn_budget_bearing) == pytest.approx(math.radians(turn_limit))
    turn_probability = (1 - max(0, (level - 3) / 7)) * 0.25
    wide_probability = 0 if level < 4 else 0.25 + (level - 4) * 0.125
    assert float(wide.float().mean()) == pytest.approx(wide_probability, abs=0.02)
    assert float(turn.float().mean()) == pytest.approx(turn_probability, abs=0.02)
    assert float(walk.float().mean()) == pytest.approx(1 - wide_probability - turn_probability, abs=0.02)


def test_transition_sampling_contract_is_opt_in_and_uses_the_existing_full_evaluation_window():
    cfg = MobileCommandsCfg().wrist_pose.planar_task
    cfg.probability = 1.0
    legacy = make_command(configure=lambda c, env: setattr(c, "planar_task", cfg.copy()))
    saved = legacy.range_curriculum.state_dict(100)
    assert "transition_sampling" not in saved["contract"]["target"]["planar_task"]
    cfg.transition_sampling.enabled = True
    active = make_command(configure=lambda c, env: setattr(c, "planar_task", cfg))
    active_transition = active.range_curriculum.contract()["target"]["planar_task"]["transition_sampling"]
    assert active_transition["turn_bearing_bridge"] == pytest.approx([0.2, math.radians(35)])
    assert (active.planar_distance_m >= 0.25 - 1e-6).all()
    assert (active.planar_sampling_group > 0).all()
    assert (active.episode_deadline_steps <= 35 * 60).all()
    assert ((active.episode_deadline_steps - active.evaluation_start_steps) == 600).all()
    with pytest.raises(ValueError, match="reset_on_load"):
        active.range_curriculum.load_state_dict(saved, 0)
    previous_cfg = cfg.copy()
    previous_cfg.transition_sampling.turn_bearing_bridge = None
    previous = make_command(configure=lambda c, env: setattr(c, "planar_task", previous_cfg))
    old_transition = previous.range_curriculum.contract()["target"]["planar_task"]["transition_sampling"]
    assert "probability_knots" in old_transition and "turn_bearing_bridge" not in old_transition
    assert station_budget_bounds(0.4, previous_cfg)[1] == pytest.approx((0.30, math.radians(25)))
    with pytest.raises(ValueError, match="reset_on_load"):
        active.range_curriculum.load_state_dict(previous.range_curriculum.state_dict(100), 0)

    previous_cfg.transition_sampling.probability_knots = None
    v8 = make_command(configure=lambda c, env: setattr(c, "planar_task", previous_cfg))
    v8_transition = v8.range_curriculum.contract()["target"]["planar_task"]["transition_sampling"]
    assert "probability_knots" not in v8_transition
    assert float(previous_cfg.transition_sampling.group_probabilities(0.4)[2]) == pytest.approx(1 / 7)


def test_planar_group_metrics_preserve_first_near_time_until_partial_reset():
    term = make_command()
    env = term._env
    pelvis = term.robot.data.body_link_pose_w.torch[:, term.reference_body_idx]
    pelvis[0, :3] = term.planar_station_w[0]
    zero = torch.zeros(1)
    pelvis[0, 3:] = quat_mul(quat_from_euler_xyz(zero, zero, term.planar_heading_rad[:1]), term._sampling_quat_w[:1])[0]
    term.planar_sampling_group[0] = 1
    env.common_step_counter = 1
    term.collect_window_metrics()
    assert term.planar_first_near_step[0] == 1
    env.common_step_counter = 2
    term.collect_window_metrics()
    env.reset_time_outs[0] = True
    metrics = term.reset(torch.tensor([0]))
    prefix = "task/planar/walk/"
    assert metrics[prefix + "near_entry_fraction"] == 1
    assert metrics[prefix + "first_near_s"] == pytest.approx(env.step_dt)
    assert metrics[prefix + "eval_pose_ready_fraction"] == 0
    assert term.planar_first_near_step[0] == -1


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
