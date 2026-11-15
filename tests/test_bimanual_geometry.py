"""Bounded pair sampling in frozen pelvis axes, with full quaternion angle limits."""

import math
from types import SimpleNamespace

import pytest
import torch
from test_upper_limb_task import pose_command as _pose_command

from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
)

from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import MobileActionsCfg
from g1.tasks.upper_limb_agent.mdp.commands import BimanualPoseCommand, BimanualPoseCommandCfg
from g1.tasks.upper_limb_agent.mdp.range_curriculum import RangeCurriculumCfg


@pytest.fixture
def pair():
    return _pose_command.__wrapped__()


def geometry(term):
    goals = term.goal_pose_w
    delta = quat_apply_inverse(term._sampling_quat_w, goals[:, 0, :3] - goals[:, 1, :3])
    center = quat_apply_inverse(
        term._sampling_quat_w,
        goals[..., :3].mean(dim=1) - term._env.scene.env_origins - term.default_wrist_pose_e[..., :3].mean(dim=1),
    )
    angle = quat_error_magnitude(goals[:, 0, 3:], goals[:, 1, 3:])
    return center, delta, angle


def with_curriculum(pair, cfg=None):
    env = pair._env
    env.cfg = SimpleNamespace(actions=MobileActionsCfg())
    env.extras = {}
    env.common_step_counter = 0
    env.reset_terminated = torch.zeros(3, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(3, dtype=torch.bool)
    cfg = pair.cfg.copy() if cfg is None else cfg
    cfg.range_curriculum = RangeCurriculumCfg()
    # Neutral fixture ranges have a fixed 30 cm spacing; production limits
    # include the authored distance despite floating-point pose conversion.
    cfg.dual_targets = BimanualPoseCommandCfg.DualTargetsCfg()
    term = BimanualPoseCommand(cfg, env)
    term.reset()
    return term


def test_many_random_pairs_obey_full_distance_signed_offsets_and_relative_angle(pair):
    env = pair._env
    count = 3072
    env.num_envs = count
    env.scene.env_origins = env.scene.env_origins.repeat(count // 3, 1)
    data = env.scene["robot"].data
    data.body_link_pose_w.torch = data.body_link_pose_w.torch.repeat(count // 3, 1, 1)
    for name in ("joint_pos", "default_joint_pos"):
        buffer = getattr(data, name)
        buffer.torch = buffer.torch.repeat(count // 3, 1)
    env.episode_length_buf = torch.ones(count, dtype=torch.long)
    torch.manual_seed(20261115)
    term = BimanualPoseCommand(BimanualPoseCommandCfg(mode_probabilities=(0, 0, 1)), env)
    term.reset()
    center, delta, angle = geometry(term)
    distance = delta.norm(dim=-1)
    assert torch.isfinite(term.goal_pose_w).all()
    assert (distance >= 0.25 - 1e-6).all() and (distance <= 0.40 + 1e-6).all()
    assert (delta[:, 1] > 0).all()
    for axis in (0, 2):
        assert (delta[:, axis].abs() <= 0.05 + 1e-6).all()
        assert delta[:, axis].min() < -0.04 and delta[:, axis].max() > 0.04
    for axis, (lo, hi) in enumerate(((-0.05, 0.15), (-0.05, 0.05), (-0.05, 0.20))):
        assert (center[:, axis] >= lo - 1e-6).all() and (center[:, axis] <= hi + 1e-6).all()
    assert (angle <= 0.15 + 1e-6).all() and angle.max() > 0.14
    assert angle.mean() == pytest.approx(0.075, abs=0.004)
    torch.testing.assert_close(term.goal_pose_w[..., 3:].norm(dim=-1), torch.ones(count, 2))


@pytest.mark.parametrize("right_sign", [-1, 1])
def test_exact_three_dimensional_geometry_and_symmetric_half_rotations(pair, monkeypatch, right_sign):
    cfg = pair.cfg.dual_targets
    for name, value in (("pos_x", 0.1), ("pos_y", -0.02), ("pos_z", 0.07)):
        setattr(cfg, name, (value, value))
    cfg.separation_range = (0.35, 0.35)
    cfg.relative_pos_x, cfg.relative_pos_z = (0.04, 0.04), (-0.03, -0.03)
    cfg.roll, cfg.pitch, cfg.yaw = (0.2, 0.2), (-0.1, -0.1), (0.3, 0.3)
    cfg.max_relative_angle = 0.15
    pair.default_wrist_pose_e[:, 1, 3:] *= right_sign
    monkeypatch.setattr(torch, "rand", lambda count, device: torch.ones(count, device=device))
    monkeypatch.setattr(
        torch, "randn", lambda count, width, device: torch.tensor([0.0, 0.0, 1.0], device=device).expand(count, width)
    )
    pair._resample_command(torch.arange(3))
    center, delta, angle = geometry(pair)
    torch.testing.assert_close(center, torch.tensor([0.1, -0.02, 0.07]).expand(3, -1))
    torch.testing.assert_close(delta, torch.tensor([0.04, math.sqrt(0.12), -0.03]).expand(3, -1))
    torch.testing.assert_close(angle, torch.full((3,), 0.15), atol=1e-6, rtol=1e-5)
    euler = torch.tensor([0.2, -0.1, 0.3]).expand(3, -1)
    common = quat_mul(pair.default_wrist_pose_e[:, 0, 3:], quat_from_euler_xyz(*euler.unbind(-1)))
    for side in range(2):
        torch.testing.assert_close(
            quat_error_magnitude(common, pair.goal_pose_w[:, side, 3:]), torch.full((3,), 0.075), atol=1e-6, rtol=1e-5
        )
    # In the common frame, the relative wrist rotations are equal/opposite about Z.
    from isaaclab.utils.math import quat_conjugate

    for side, sign in ((0, 1), (1, -1)):
        relative = quat_mul(quat_conjugate(common), pair.goal_pose_w[:, side, 3:])
        torch.testing.assert_close(relative[:, 2], torch.full((3,), sign * math.sin(0.15 / 4)))


def test_course_scales_pair_about_default_geometry_and_uses_each_episode_level(pair):
    term = with_curriculum(pair)
    course = term.range_curriculum
    course.episode_levels[:] = torch.tensor([0, 5, 10])
    torch.manual_seed(401)
    term._resample_command(torch.arange(3))
    center, delta, angle = geometry(term)
    original_goals = term.goal_pose_w.clone()
    scales = course.target_scale(torch.arange(3))
    # Repeat the same random samples with full ranges as an independent endpoint.
    course.episode_levels.fill_(10)
    torch.manual_seed(401)
    term._resample_command(torch.arange(3))
    full_center, full_delta, full_angle = geometry(term)
    nominal = quat_apply_inverse(
        term._sampling_quat_w, term.default_wrist_pose_e[:, 0, :3] - term.default_wrist_pose_e[:, 1, :3]
    )
    torch.testing.assert_close(center, full_center * scales[:, None], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        delta.norm(dim=-1), nominal.norm(dim=-1) + scales * (full_delta.norm(dim=-1) - nominal.norm(dim=-1))
    )
    for axis in (0, 2):
        torch.testing.assert_close(delta[:, axis], nominal[:, axis] + scales * (full_delta[:, axis] - nominal[:, axis]))
    torch.testing.assert_close(angle, full_angle * scales, atol=1e-6, rtol=1e-5)
    course.episode_levels[:] = torch.tensor([0, 5, 10])
    course.level = 9  # Existing episodes retain their assigned ranges.
    torch.manual_seed(401)
    term._resample_command(torch.arange(3))
    torch.testing.assert_close(term.goal_pose_w, original_goals, atol=0, rtol=0)


@pytest.mark.parametrize("mode", [0, 1])
def test_single_hand_sampling_is_independent_of_dual_configuration(pair, mode):
    pair.mode_id.fill_(mode)
    pair.cfg.left_offsets = BimanualPoseCommandCfg.Ranges()
    pair.cfg.right_offsets = BimanualPoseCommandCfg.Ranges(pos_y=(-0.1, 0.05))
    torch.manual_seed(29)
    pair._resample_command(torch.arange(3))
    expected = pair.goal_pose_w.clone()
    pair.cfg.dual_targets = BimanualPoseCommandCfg.DualTargetsCfg(pos_x=(2, 3), max_relative_angle=1.0)
    torch.manual_seed(29)
    pair._resample_command(torch.arange(3))
    torch.testing.assert_close(pair.goal_pose_w, expected, atol=0, rtol=0)


def test_partial_dual_resampling_keeps_other_modes_and_world_goals_fixed(pair):
    pair.cfg.dual_targets = BimanualPoseCommandCfg.DualTargetsCfg()
    pair.mode_id[:] = torch.tensor([0, 1, 2])
    before = pair.goal_pose_w.clone()
    pair._resample(torch.tensor([2]))
    torch.testing.assert_close(pair.goal_pose_w[:2], before[:2], atol=0, rtol=0)
    goal, observed = pair.goal_pose_w.clone(), pair.command.clone()
    pair.robot.data.body_link_pose_w.torch[:, 1, :3] += 2
    pair.robot.data.body_link_pose_w.torch[:, 1, 3:] = quat_from_euler_xyz(
        torch.zeros(3), torch.zeros(3), torch.full((3,), 0.8)
    )
    assert not torch.allclose(pair.command, observed)
    torch.testing.assert_close(pair.goal_pose_w, goal, atol=0, rtol=0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("pos_y", (0.1, -0.1)),
        ("roll", (float("nan"), 0.1)),
        ("separation_range", (0.0, 0.4)),
        ("separation_range", (0.4, 0.25)),
        ("separation_range", (0.3,)),
        ("relative_pos_x", (-0.3, 0.3)),
        ("relative_pos_x", (float("inf"), 0.1)),
        ("relative_pos_z", (0.1, -0.1)),
        ("max_relative_angle", -0.1),
        ("max_relative_angle", float("nan")),
        ("max_relative_angle", math.pi + 0.01),
    ],
)
def test_invalid_geometry_fails_before_environment_initialization(name, value):
    cfg = BimanualPoseCommandCfg()
    setattr(cfg.dual_targets, name, value)
    with pytest.raises(ValueError, match="dual_targets"):
        BimanualPoseCommand(cfg, None)


def test_curriculum_rejects_authored_pair_outside_limits(pair):
    term = with_curriculum(pair)
    term.cfg.dual_targets.separation_range = (0.35, 0.4)
    term._reference_initialized.zero_()
    with pytest.raises(ValueError, match="Default wrist geometry"):
        term.reset()


def test_geometry_contract_changes_require_explicit_curriculum_restart(pair):
    term = with_curriculum(pair)
    course = term.range_curriculum
    saved = course.state_dict(100)
    assert saved["contract"]["target"]["dual_targets"]["max_relative_angle"] == 0.15
    restored = with_curriculum(pair).range_curriculum
    restored.load_state_dict(saved, 0)
    assert restored.clock_offset == 100
    changed = with_curriculum(pair).range_curriculum
    changed.target_contract["dual_targets"]["max_relative_angle"] = 0.2
    with pytest.raises(ValueError, match="differs from checkpoint"):
        changed.load_state_dict(saved, 0)
    legacy = course.state_dict(100)
    del legacy["contract"]["target"]["dual_targets"]
    with pytest.raises(ValueError, match="reset_on_load"):
        course.load_state_dict(legacy, 0)


def test_production_config_exposes_pair_defaults_and_mobile_inherits_them():
    from g1.tasks.upper_limb_agent.config.g1.env_cfg import CommandsCfg
    from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import MobileCommandsCfg

    fixed = CommandsCfg().wrist_pose.dual_targets
    mobile = MobileCommandsCfg().wrist_pose.dual_targets
    assert fixed == mobile and fixed is not mobile
    assert fixed.relative_pos_x == fixed.relative_pos_z == (-0.05, 0.05)
    assert fixed.separation_range == (0.25, 0.4) and fixed.max_relative_angle == 0.15
