# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical regression checks for episode boundaries, metrics and curriculum scheduling."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg

from g1.tasks.lower_limb_agent.env import LowerLimbAgentEnv
from g1.tasks.lower_limb_agent.mdp.commands.commands import SquatWalkCommand, UniformHeightCommand
from g1.tasks.lower_limb_agent.mdp.commands.commands_cfg import SquatWalkCommandCfg, UniformHeightCommandCfg
from g1.tasks.lower_limb_agent.mdp.curriculums import (
    command_range_curriculum,
    upper_body_disturbance_magnitude_curriculum,
)
from g1.tasks.lower_limb_agent.mdp.rewards import action_acc_l2


class _Scene(dict):
    def __init__(self, robot, num_envs):
        super().__init__(robot=robot)
        self.env_origins = torch.zeros(num_envs, 3)


def _command(kind, num_envs=2):
    root_pos = torch.zeros(num_envs, 3)
    root_pos[:, 2] = 0.75
    root_pose = torch.zeros(num_envs, 7)
    root_pose[:, 2] = 0.75
    root_pose[:, 6] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos),
            default_root_pose=SimpleNamespace(torch=root_pose),
            root_lin_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
            root_ang_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
        ),
        find_bodies=lambda *args, **kwargs: ([0, 1], ["left", "right"]),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=_Scene(robot, num_envs),
        extras={},
        common_step_counter=0,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        step_dt=1 / 60,
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None)),
    )
    if kind == "height":
        cfg = UniformHeightCommandCfg(
            asset_name="robot",
            debug_vis=False,
            ranges=UniformHeightCommandCfg.Ranges(height_offset=(0.0, 0.0)),
        )
        term = UniformHeightCommand(cfg, env)
    else:
        cfg = SquatWalkCommandCfg(
            asset_name="robot",
            debug_vis=False,
            rel_mode_envs=(1.0, 0.0, 0.0, 0.0),
            ranges=SquatWalkCommandCfg.Ranges(height_offset=(-0.15, 0.0)),
        )
        term = SquatWalkCommand(cfg, env)
        term.pelvis_error_xy = lambda: torch.zeros(num_envs, 2)
    term.height_command_world.fill_(0.75)
    return term, env


def test_action_acceleration_and_subset_resets():
    env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.zeros(2, 1), prev_action=torch.zeros(2, 1)))
    reward = action_acc_l2(RewardTermCfg(func=action_acc_l2, weight=-1.0), env)
    for action, previous, expected in ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 2, 0)):
        env.action_manager.action.fill_(action)
        env.action_manager.prev_action.fill_(previous)
        torch.testing.assert_close(reward(env), torch.full((2,), float(expected)))
    # Reset one env after its terminal reward; the other must retain its history.
    reward.reset(torch.tensor([0]))
    env.action_manager.action[:] = torch.tensor([[1.0], [4.0]])
    env.action_manager.prev_action[:] = torch.tensor([[0.0], [3.0]])
    torch.testing.assert_close(reward(env), torch.tensor([1.0, 0.0]))
    reward.reset(slice(None))
    env.action_manager.action.fill_(2)
    env.action_manager.prev_action.zero_()
    torch.testing.assert_close(reward(env), torch.tensor([4.0, 4.0]))


@pytest.mark.parametrize("kind", ["height", "squat_walk"])
def test_initial_reset_has_no_success_sample(kind):
    term, env = _command(kind)
    assert term.reset(slice(None)) == {}
    assert "Metrics/success_rate" not in env.extras.get("log", {})
    assert term._step_count.sum() == 0
    if kind == "squat_walk":
        assert not term.episode_metrics_valid.any()


@pytest.mark.parametrize("kind", ["height", "squat_walk"])
def test_terminal_frame_is_counted_once_and_survivors_are_not_double_counted(kind):
    term, env = _command(kind)
    env.common_step_counter = 1
    env.episode_length_buf.fill_(1)
    env.scene["robot"].data.root_pos_w.torch[0, 2] = 0.5
    extras = term.reset(torch.tensor([0]))
    torch.testing.assert_close(extras["error_height"], torch.tensor(0.25))
    torch.testing.assert_close(env.extras["log"]["Metrics/success_rate"], torch.tensor(0.0))
    if kind == "squat_walk":
        assert term.episode_metrics_valid.tolist() == [True, False]
    # Reproduce the manager's post-reset compute hook in this same control step.
    env.episode_length_buf[0] = 0
    term._update_metrics()
    torch.testing.assert_close(term._step_count, torch.tensor([0.0, 1.0]))
    env.common_step_counter = 2
    env.episode_length_buf[:] = torch.tensor([1, 2])
    term._update_metrics()
    torch.testing.assert_close(term._step_count, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("kind", ["height", "squat_walk"])
def test_empty_episodes_do_not_dilute_mixed_reset_statistics(kind):
    term, env = _command(kind)
    env.common_step_counter = 1
    env.episode_length_buf[0] = 1
    env.scene["robot"].data.root_pos_w.torch[0, 2] = 0.5
    extras = term.reset(slice(None))
    torch.testing.assert_close(extras["error_height"], torch.tensor(0.25))
    torch.testing.assert_close(env.extras["log"]["Metrics/success_rate"], torch.tensor(0.0))


@pytest.mark.parametrize("kind", ["height", "squat_walk"])
def test_episode_mean_includes_earlier_and_terminal_samples(kind):
    term, env = _command(kind)
    env.common_step_counter = 1
    env.episode_length_buf[0] = 1
    term._update_metrics()
    env.common_step_counter = 2
    env.episode_length_buf[0] = 2
    env.scene["robot"].data.root_pos_w.torch[0, 2] = 0.55
    extras = term.reset(torch.tensor([0]))
    torch.testing.assert_close(extras["error_height"], torch.tensor(0.1))


def test_terminal_velocity_and_pelvis_errors_are_included():
    term, env = _command("squat_walk")
    env.common_step_counter = 1
    env.episode_length_buf[0] = 1
    term.vel_command_b[0] = torch.tensor([1.0, 0.0, 1.0])
    term.pelvis_error_xy = lambda: torch.tensor([[0.04, 0.0], [0.0, 0.0]])
    extras = term.reset(torch.tensor([0]))
    torch.testing.assert_close(extras["error_vel_xy"], torch.tensor(1.0))
    torch.testing.assert_close(extras["error_vel_yaw"], torch.tensor(1.0))
    torch.testing.assert_close(term.episode_success_rate_vel[0], torch.tensor(0.0))
    assert extras["error_pelvis_xy"] == pytest.approx(0.04)
    assert extras["success_rate_pelvis_xy"] == 0


@pytest.mark.parametrize("kind", ["height", "squat_walk"])
def test_environment_captures_terminal_state_before_physical_reset_events(kind, monkeypatch):
    term, env = _command(kind)
    env.common_step_counter = 1
    env.episode_length_buf[0] = 1
    env.scene["robot"].data.root_pos_w.torch[0, 2] = 0.5
    project_env = LowerLimbAgentEnv.__new__(LowerLimbAgentEnv)
    project_env.command_manager = SimpleNamespace(active_terms=["task"], get_term=lambda name: term)
    logged = {}

    def physical_reset_then_command_reset(self, ids):
        # This is the important ordering in IsaacLab's base implementation.
        env.scene["robot"].data.root_pos_w.torch[ids, 2] = 0.75
        logged.update(term.reset(ids))
        env.episode_length_buf[ids] = 0

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", physical_reset_then_command_reset)
    project_env._reset_idx(torch.tensor([0]))
    torch.testing.assert_close(logged["error_height"], torch.tensor(0.25))
    term._update_metrics()
    torch.testing.assert_close(term._step_count, torch.zeros(2))


def _range_env():
    term = SimpleNamespace(
        cfg=SimpleNamespace(
            ranges=SimpleNamespace(
                height_offset=(-0.15, 0.02),
                lin_vel_x=(-0.3, 0.6),
                lin_vel_y=(-0.15, 0.15),
            )
        ),
        episode_success_rate=torch.ones(2),
        episode_success_rate_vel=torch.ones(2),
        episode_metrics_valid=torch.ones(2, dtype=torch.bool),
        last_mode=torch.full((2,), 2),
        MODE_WALK=2,
        MODE_SQUAT_WALK=3,
    )
    return SimpleNamespace(common_step_counter=0, command_manager=SimpleNamespace(get_term=lambda name: term)), term


def test_range_update_handles_late_resets_duplicates_and_no_catchup_burst():
    env, term = _range_env()
    for step, expected in (
        (0, -0.15),
        (9, -0.15),
        (11, -0.155),
        (11, -0.155),
        (20, -0.155),
        (21, -0.16),
        (100, -0.165),
        (100, -0.165),
    ):
        env.common_step_counter = step
        command_range_curriculum(env, slice(None), update_period=10)
        assert term.cfg.ranges.height_offset[0] == pytest.approx(expected)


def test_range_gate_ignores_uninitialized_episode_buffers():
    env, term = _range_env()
    term.episode_metrics_valid.zero_()
    env.common_step_counter = 10
    result = command_range_curriculum(env, slice(None), update_period=10)
    assert result["height_success"] == 0
    assert term.cfg.ranges.height_offset[0] == -0.15


def _survival_env(num_envs=3):
    group = {"joint_noise_frac": 0.2}
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        max_episode_length=10,
        common_step_counter=0,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        termination_manager=SimpleNamespace(dones=torch.zeros(num_envs, dtype=torch.bool)),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params={"disturb_groups": [group]})),
    )
    return env, group


def _survival(env, ids=slice(None), **kwargs):
    return upper_body_disturbance_magnitude_curriculum(env, ids, update_period=10, **kwargs)


def test_survival_episode_weighting_and_window_ema():
    env, group = _survival_env()
    assert _survival(env)["survival_valid"] == 0
    env.common_step_counter = 2
    env.episode_length_buf[0] = 2
    env.termination_manager.dones[0] = True
    _survival(env, torch.tensor([0]), ema_beta=0.5)
    env.common_step_counter = 10
    env.episode_length_buf[1:] = 10
    env.termination_manager.dones[:] = torch.tensor([False, True, True])
    result = _survival(env, slice(1, 3), ema_beta=0.5)
    torch.testing.assert_close(result["survival_ema"], torch.tensor((0.2 + 1.0 + 1.0) / 3))
    assert result["survival_episodes"] == 3
    assert group["joint_noise_frac"] == 0.2
    # Repeated reset does not add samples or move the EMA again.
    _survival(env, slice(1, 3), ema_beta=0.5)
    assert result["survival_episodes"] == 3
    env.common_step_counter = 20
    env.episode_length_buf[1:] = 10
    result = _survival(env, slice(1, 3), ema_beta=0.5)
    torch.testing.assert_close(result["survival_ema"], torch.tensor(((2.2 / 3) + 1) / 2))
    assert group["joint_noise_frac"] == pytest.approx(0.205)


def test_survival_excludes_manual_resets_random_initial_lengths_and_empty_windows():
    env, group = _survival_env()
    _survival(env)
    env.common_step_counter = 4
    env.episode_length_buf[:] = torch.tensor([4, 10, 0])
    env.termination_manager.dones[:] = torch.tensor([False, True, False])
    _survival(env, slice(0, 2))
    env.common_step_counter = 10
    result = _survival(env, torch.tensor([], dtype=torch.long))
    assert result["survival_valid"] == 0
    assert result["survival_episodes"] == 0
    assert group["joint_noise_frac"] == 0.2
    env.common_step_counter = 14
    env.episode_length_buf[:2] = 10
    env.termination_manager.dones[:2] = True
    _survival(env, slice(0, 2))
    env.common_step_counter = 20
    result = _survival(env, slice(0, 0))
    assert result["survival_valid"] == 1
    torch.testing.assert_close(result["survival_ema"], torch.tensor(1.0))
    assert result["survival_episodes"] == 2
    assert group["joint_noise_frac"] == pytest.approx(0.205)
