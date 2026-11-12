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
from g1.tasks.lower_limb_agent.mdp.commands.commands import SquatWalkCommand
from g1.tasks.lower_limb_agent.mdp.commands.commands_cfg import SquatWalkCommandCfg
from g1.tasks.lower_limb_agent.mdp.curriculums import (
    command_range_curriculum,
    upper_body_disturbance_magnitude_curriculum,
)
from g1.tasks.lower_limb_agent.mdp.rewards import action_acc_l2
from g1.tasks.lower_limb_agent.performance import ModePerformanceMonitor


class _Scene(dict):
    def __init__(self, robot, num_envs):
        super().__init__(robot=robot)
        self.env_origins = torch.zeros(num_envs, 3)
        self.sensors = {
            "contact_forces": SimpleNamespace(
                find_bodies=lambda *args, **kwargs: ([0, 1], ["left", "right"]),
                data=SimpleNamespace(
                    current_air_time=SimpleNamespace(torch=torch.zeros(num_envs, 2)),
                    current_contact_time=SimpleNamespace(torch=torch.ones(num_envs, 2)),
                ),
            ),
        }


def _command(num_envs=2):
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
            body_link_pose_w=SimpleNamespace(torch=torch.zeros(num_envs, 2, 7)),
            joint_pos=SimpleNamespace(torch=torch.zeros(num_envs, 2)),
            default_joint_pos=SimpleNamespace(torch=torch.zeros(num_envs, 2)),
        ),
        find_bodies=lambda *args, **kwargs: ([0, 1], ["left", "right"]),
        find_joints=lambda *args, **kwargs: ([0, 1], ["left_hip_pitch_joint", "right_hip_pitch_joint"]),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=_Scene(robot, num_envs),
        extras={},
        common_step_counter=0,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        max_episode_length=360,
        termination_manager=SimpleNamespace(
            active_terms=["time_out", "bad_ori"],
            get_term=lambda name: torch.full((num_envs,), name == "time_out", dtype=torch.bool),
            get_term_cfg=lambda name: SimpleNamespace(time_out=name == "time_out"),
        ),
        step_dt=1 / 60,
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None)),
    )
    cfg = SquatWalkCommandCfg(
        asset_name="robot",
        debug_vis=False,
        rel_mode_envs=(1.0, 0.0, 0.0, 0.0),
        ranges=SquatWalkCommandCfg.Ranges(height_offset=(-0.15, 0.0)),
    )
    term = SquatWalkCommand(cfg, env)
    term.pelvis_error_xy = lambda: torch.zeros(num_envs, 2)
    term.height_command_world.fill_(0.75)
    term.performance.reset(torch.arange(num_envs), term.mode, 0)
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


def test_walk_physical_metrics_use_the_existing_step_stamp_and_include_terminal_frame():
    term, env = _command()
    term.mode.fill_(2)
    term.task_command[:, 1] = 0.6
    sensor = env.scene.sensors["contact_forces"]
    sensor.data.current_contact_time.torch[:] = torch.tensor([0.0, 0.2])
    sensor.data.current_air_time.torch[:] = torch.tensor([0.2, 0.0])
    env.scene["robot"].data.body_link_pose_w.torch[:, 0, 2] = 0.2
    env.common_step_counter = 1
    env.episode_length_buf.fill_(1)
    term._update_metrics()
    term._update_metrics()
    assert term.performance.walk[0] == 2
    term.reset(torch.tensor([0]))  # Repeated terminal sampling in reset must not add a frame.
    assert term.performance.walk[0] == 2
    metrics = term.performance.drain(term._mode_probs)
    assert metrics["Performance/walk/swing_foot_height_m"] == pytest.approx(0.2)


def test_initial_reset_has_no_success_sample():
    term, env = _command()
    assert term.reset(slice(None)) == {}
    assert "Metrics/success_rate" not in env.extras.get("log", {})
    assert term.performance.episode[:, 0].sum() == 0
    assert not term.episode_metrics_valid.any()


def test_terminal_frame_is_counted_once_and_survivors_are_not_double_counted():
    term, env = _command()
    env.common_step_counter = 1
    env.episode_length_buf.fill_(1)
    env.scene["robot"].data.root_pos_w.torch[0, 2] = 0.5
    extras = term.reset(torch.tensor([0]))
    assert extras == {}
    assert term.episode_success_rate[0] == 0
    assert term.performance.drain(term._mode_probs)["Performance/stand/error_height_m"] == pytest.approx(0.25)
    assert term.episode_metrics_valid.tolist() == [True, False]
    # Reproduce the manager's post-reset compute hook in this same control step.
    env.episode_length_buf[0] = 0
    term._update_metrics()
    torch.testing.assert_close(term.performance.episode[:, 0], torch.tensor([0.0, 1.0]))
    env.common_step_counter = 2
    env.episode_length_buf[:] = torch.tensor([1, 2])
    term._update_metrics()
    torch.testing.assert_close(term.performance.episode[:, 0], torch.tensor([1.0, 2.0]))


def test_empty_episodes_do_not_dilute_mixed_reset_statistics():
    term, env = _command()
    env.common_step_counter = 1
    env.episode_length_buf[0] = 1
    env.scene["robot"].data.root_pos_w.torch[0, 2] = 0.5
    extras = term.reset(slice(None))
    assert extras == {}
    assert term.episode_success_rate[0] == 0
    assert term.performance.drain(term._mode_probs)["Performance/stand/error_height_m"] == pytest.approx(0.25)


def test_episode_mean_includes_earlier_and_terminal_samples():
    term, env = _command()
    env.common_step_counter = 1
    env.episode_length_buf[0] = 1
    term._update_metrics()
    env.common_step_counter = 2
    env.episode_length_buf[0] = 2
    env.scene["robot"].data.root_pos_w.torch[0, 2] = 0.55
    extras = term.reset(torch.tensor([0]))
    assert extras == {}
    assert term.episode_success_rate[0] == 0
    assert term.performance.drain(term._mode_probs)["Performance/stand/error_height_m"] == pytest.approx(0.1)


def test_terminal_velocity_and_pelvis_errors_are_included():
    term, env = _command()
    env.common_step_counter = 1
    env.episode_length_buf[0] = 1
    term.vel_command_b[0] = torch.tensor([1.0, 0.0, 1.0])
    term.pelvis_error_xy = lambda: torch.tensor([[0.04, 0.0], [0.0, 0.0]])
    extras = term.reset(torch.tensor([0]))
    assert extras == {}
    assert term.episode_success_rate_vel[0] == 0
    assert not term.metrics  # No superseded command-metric producer remains.
    values = term.performance.drain(term._mode_probs)
    assert values["Performance/stand/error_velocity_xy_mps"] == 1
    assert values["Performance/stand/error_yaw_radps"] == 1
    assert values["Performance/stand/error_pelvis_xy_m"] == pytest.approx(0.04)


def test_performance_uses_old_mode_and_raw_failure_before_reset():
    term, env = _command()
    term.reset(slice(None))
    term.mode[:] = torch.tensor([2, 3])
    term.performance.reset(torch.arange(2), term.mode, 0)
    env.common_step_counter = 1
    env.episode_length_buf.fill_(1)
    env.termination_manager.get_term = lambda name: (
        torch.tensor([True, False]) if name == "bad_ori" else torch.zeros(2, dtype=torch.bool)
    )
    term.reset(torch.tensor([0]))
    assert term.mode[0] == 0  # next episode is forced to STAND by the fixture
    metrics = term.performance.drain(term._mode_probs)
    assert metrics["Performance/walk/episode_count"] == 1
    assert metrics["Performance/walk/failure_rate"] == 1
    assert metrics["Performance/walk/joint_success_rate"] == 0
    assert metrics["Performance/stand/episode_count"] == 0
    assert term.last_mode[0] == 2
    assert term.episode_success_rate[0] == 1  # legacy curriculum semantics are preserved


def test_diagnostics_do_not_consume_rng_or_change_command_sampling(monkeypatch):
    first, first_env = _command()
    second, second_env = _command()
    for env in (first_env, second_env):
        env.common_step_counter = 1
        env.episode_length_buf.fill_(1)
    second.performance.drain(second._mode_probs)
    rng = torch.get_rng_state()
    first.reset(slice(None))
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    second.reset(slice(None))
    torch.testing.assert_close(first.command, second.command)
    torch.testing.assert_close(first.mode, second.mode)
    assert torch.equal(torch.get_rng_state(), after)


def test_environment_captures_terminal_state_before_physical_reset_events(monkeypatch):
    term, env = _command()
    term.reset(slice(None))
    env.common_step_counter = 1
    env.episode_length_buf[0] = 1
    env.scene["robot"].data.root_pos_w.torch[0, 2] = 0.5
    env.termination_manager.get_term = lambda name: (
        torch.tensor([True, False]) if name == "bad_ori" else torch.zeros(2, dtype=torch.bool)
    )
    project_env = LowerLimbAgentEnv.__new__(LowerLimbAgentEnv)
    project_env.command_manager = SimpleNamespace(active_terms=["task"], get_term=lambda name: term)
    logged = {}
    reset_order = []

    def physical_reset_then_command_reset(self, ids):
        # This is the important ordering in IsaacLab's base implementation.
        env.scene["robot"].data.root_pos_w.torch[ids, 2] = 0.75
        logged.update(term.reset(ids))
        env.episode_length_buf[ids] = 0
        reset_order.append("physical")

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", physical_reset_then_command_reset)
    monkeypatch.setattr(LowerLimbAgentEnv, "_reset_upper_body_targets", lambda self, ids: reset_order.append("targets"))
    project_env._reset_idx(torch.tensor([0]))
    assert reset_order == ["physical", "targets"]
    assert logged == {}
    metrics = term.performance.drain(term._mode_probs)
    assert metrics["Performance/stand/error_height_m"] == pytest.approx(0.25)
    assert metrics["Performance/stand/failure_rate"] == 1
    term._update_metrics()
    torch.testing.assert_close(term.performance.episode[:, 0], torch.zeros(2))


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
    monitor = ModePerformanceMonitor(num_envs, "cpu", 0.1, (0.03, 0.5, 0.4, 0.02))
    # Keep legacy tests focused on the unchanged survival weighting and EMA.
    monitor.latest_joint_success.fill_(True)
    monitor.latest_joint_success_valid.fill_(True)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        max_episode_length=10,
        common_step_counter=0,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        termination_manager=SimpleNamespace(dones=torch.zeros(num_envs, dtype=torch.bool)),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params={"disturb_groups": [group]})),
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(performance=monitor)),
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


@pytest.mark.parametrize(
    "survival_passes,successful_envs,advances",
    [
        (True, 5, True),
        (True, 4, False),
        (True, 0, False),
        (False, 5, False),
    ],
)
def test_disturbance_requires_both_survival_and_strict_task_success_gates(survival_passes, successful_envs, advances):
    env, group = _survival_env(5)
    monitor = env.command_manager.get_term("task_command").performance
    monitor.latest_joint_success[:] = torch.arange(5) < successful_envs
    env.max_episode_length = 10 if survival_passes else 20
    _survival(env)
    env.common_step_counter = 10
    env.episode_length_buf.fill_(10)
    env.termination_manager.dones.fill_(True)
    result = _survival(env)
    assert result["task_success_rate"] == pytest.approx(successful_envs / 5)
    assert result["survival_ema"] == (1.0 if survival_passes else 0.5)
    assert group["joint_noise_frac"] == pytest.approx(0.205 if advances else 0.2)


def test_missing_quality_pauses_ramp_without_freezing_survival_ema_and_can_recover():
    env, group = _survival_env()
    monitor = env.command_manager.get_term("task_command").performance
    monitor.clear_task_success()
    _survival(env)
    env.common_step_counter = 10
    env.episode_length_buf.fill_(10)
    env.termination_manager.dones.fill_(True)
    result = _survival(env)
    assert result["survival_ema"] == 1
    assert result["survival_valid"] == 1
    assert result["task_success_rate"] == 0
    assert group["joint_noise_frac"] == 0.2
    monitor.latest_joint_success.fill_(True)
    monitor.latest_joint_success_valid.fill_(True)
    env.common_step_counter = 20
    assert _survival(env)["task_success_rate"] == 1
    assert group["joint_noise_frac"] == pytest.approx(0.205)
    _survival(env)  # Duplicate reset in the same step must not advance again.
    assert group["joint_noise_frac"] == pytest.approx(0.205)


@pytest.mark.parametrize("gate", [-0.1, 1.1, float("nan"), True])
def test_disturbance_rejects_invalid_task_success_gate(gate):
    env, _ = _survival_env()
    with pytest.raises(ValueError, match="task_success_gate"):
        _survival(env, task_success_gate=gate)


def test_curriculum_consumes_previous_finalized_quality_without_changing_reset_order():
    term, env = _command()
    env.max_episode_length = 10
    group = {"joint_noise_frac": 0.2}
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params={"disturb_groups": [group]}))
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    env.termination_manager.dones = torch.ones(2, dtype=torch.bool)
    env.termination_manager.get_term = lambda name: torch.full((2,), name == "time_out", dtype=torch.bool)
    term.reset(slice(None))
    _survival(env)
    for ending_step in (10, 20):
        for step in range(ending_step - 9, ending_step + 1):
            env.common_step_counter = step
            env.episode_length_buf.fill_(step - ending_step + 10)
            term._update_metrics()
        result = _survival(env)
        assert result["task_success_rate"] == (0 if ending_step == 10 else 1)
        assert group["joint_noise_frac"] == pytest.approx(0.2 if ending_step == 10 else 0.205)
        term.reset(slice(None))
        assert term.performance.task_success_rate() == 1
