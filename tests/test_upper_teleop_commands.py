"""External targets exercise the real command lifecycle without a simulator."""

import copy

import pytest
import torch
from test_planar_timing import make_command

from g1.tasks.upper_limb_agent.mdp.terminations import task_time_out


def test_manual_targets_are_queued_world_fixed_and_survive_forced_resampling():
    term = make_command()
    term.set_teleop("manual", arm_mode=0)
    term.reset()
    current = term.robot.data.body_link_pose_w.torch[:, term.body_indices].clone()
    torch.testing.assert_close(term.goal_pose_w, current)
    poses = current.clone()
    poses[:, 0, 0] += 1
    poses[..., 3:] *= 2  # External input is normalized once at the boundary.
    term.set_manual_targets(poses, 0)
    torch.testing.assert_close(term.goal_pose_w, current)
    term._update_command()
    assert (term.active_mask == torch.tensor([True, False])).all()
    assert (term.goal_pose_w[:, 0, 0] == current[:, 0, 0] + 1).all()
    expected = term.goal_pose_w.clone()
    term.robot.data.body_link_pose_w.torch[:, 0, 0] += 2
    term.time_left.zero_()
    term._resample(torch.arange(3))
    torch.testing.assert_close(term.goal_pose_w, expected)
    term._env.episode_length_buf.fill_(10000)
    term._env.common_step_counter = 10000
    assert not task_time_out(term._env).any()


@pytest.mark.parametrize("category", ["regular", "height_shifted", "planar", "mixed"])
def test_forced_category_holds_one_goal_and_does_not_mutate_course_contract(category):
    term = make_command(probability=0.2)
    term.range_curriculum.evaluation = True
    contract = copy.deepcopy(term.range_curriculum.contract())
    term.set_teleop("auto", category, arm_mode=1)
    term.reset()
    assert term.range_curriculum.contract() == contract
    assert (term.mode_id == 1).all()
    if category != "mixed":
        assert (term.task_ids == ["regular", "height_shifted", "planar"].index(category)).all()
    before = term.teleop_snapshot()
    term.time_left.zero_()
    term._resample(torch.arange(3))
    for name, value in before.items():
        torch.testing.assert_close(getattr(term, name), value)
    term._env.common_step_counter = int(term.episode_deadline_steps.max())
    assert task_time_out(term._env).all()


def test_retry_restores_geometry_timing_and_arms_after_a_real_reset():
    term = make_command()
    term.set_teleop("auto", "planar", arm_mode=2)
    term.reset()
    snapshot = term.teleop_snapshot()
    term._env.common_step_counter = 500
    term.set_teleop("auto", "planar", 2, replay=snapshot)
    term.reset()
    for name, value in snapshot.items():
        torch.testing.assert_close(getattr(term, name), value)
    assert (term.episode_age_steps() == 0).all()
    assert term._teleop.replay is None


def test_invalid_poses_rejected_and_training_can_resume_with_original_rng_path():
    term = make_command()
    term.set_teleop("manual")
    term.reset()
    poses = term.goal_pose_w.clone()
    poses[..., 3:] = 0
    with pytest.raises(ValueError, match="nonzero"):
        term.set_manual_targets(poses, 2)
    term.clear_teleop()
    torch.manual_seed(124)
    term.reset()
    expected, rng = term.goal_pose_w.clone(), torch.get_rng_state()
    other = make_command()
    torch.manual_seed(124)
    other.reset()
    torch.testing.assert_close(other.goal_pose_w, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
