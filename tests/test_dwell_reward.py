"""Physical-substep rolling RMS, current-goal dwell gates and reset-safe diagnostics."""

from types import SimpleNamespace

import pytest
import torch
from test_mobile_motion_rewards import make_mobile

from isaaclab.managers import RewardManager

from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import MobileActionsCfg, MobileRewardsCfg
from g1.tasks.upper_limb_agent.mdp.rewards import DwellReward


def make_dwell(decimation=2):
    action, calls = make_mobile(decimation)
    env = action._env
    env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    env.termination_manager = SimpleNamespace(terminated=torch.zeros(3, dtype=torch.bool))
    position, orientation = torch.zeros(3, 2), torch.zeros(3, 2)
    command = SimpleNamespace(
        range_curriculum=None,
        active_mask=torch.tensor([[True, False], [False, True], [True, True]]),
        pose_error=lambda: (position, orientation),
    )
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    cfg = MobileRewardsCfg().dwell
    return action, calls, command, position, orientation, DwellReward(cfg, env)


def speed_period(action, samples, finalize=True):
    env = action._env
    env.common_step_counter += 1
    env.episode_length_buf += 1
    action.process_actions(torch.zeros(3, 2))
    for speed in samples:
        action.apply_actions()
        action._asset.data.body_link_vel_w.torch[:, action._base_body_id, 2] = torch.as_tensor(speed)
    return action.base_vertical_speed_rms() if finalize else None


def fill_window(action, speed=0.0):
    for _ in range(action._vertical_speed_window_steps):
        speed_period(action, [speed] * action._substep_count)


@pytest.mark.parametrize("decimation", [1, 2, 4])
def test_rms_waits_for_full_window_and_captures_early_substep_spikes(decimation):
    action, calls, _, _, _, reward = make_dwell(decimation)
    assert action._vertical_speed_window_steps * decimation == 12
    assert not reward(action._env).any()
    assert not reward._steps.any()
    for _ in range(action._vertical_speed_window_steps - 1):
        _, full = speed_period(action, [0.0] * decimation)
        assert not full.any() and not reward(action._env).any()
    rms, full = speed_period(action, [0.12] + [0.0] * (decimation - 1))
    assert full.all()
    torch.testing.assert_close(rms, torch.full((3,), (0.12**2 / 12) ** 0.5))
    assert not reward(action._env).any()
    assert calls["inference"] == action._vertical_speed_window_steps
    assert calls["lower_apply"] == 12
    for _ in range(action._vertical_speed_window_steps - 1):
        rms, _ = speed_period(action, [0.0] * decimation)
        assert (rms > 0.02).all()
    rms, full = speed_period(action, [0.0] * decimation)
    assert full.all() and not rms.any()
    torch.testing.assert_close(reward(action._env), torch.ones(3))


def test_rms_squares_before_averaging_and_reads_selected_world_link_without_resampling():
    action, _, _, _, _, reward = make_dwell()
    for _ in range(6):
        speed_period(action, [0.03, -0.03])
    rms, full = action.base_vertical_speed_rms()
    torch.testing.assert_close(rms, torch.full((3,), 0.03))
    assert full.all() and not reward(action._env).any()
    history, count = action._vertical_speed_history.clone(), action._vertical_speed_history_count.clone()
    index = action._vertical_speed_history_index
    action._asset.data.body_link_vel_w.torch.fill_(100)
    for _ in range(3):
        reward(action._env)
        action.mean_base_vertical_acc_excess_l2()
        action.mean_base_motion_l2()
    torch.testing.assert_close(action.base_vertical_speed_rms()[0], rms)
    torch.testing.assert_close(action._vertical_speed_history, history, atol=0, rtol=0)
    torch.testing.assert_close(action._vertical_speed_history_count, count)
    assert action._vertical_speed_history_index == index
    speed_period(action, [0.0, 0.0])
    # Other links and world XY velocity do not enter vertical-speed RMS.
    action._asset.data.body_link_vel_w.torch[:, [0, 2]] = 1000
    action._asset.data.body_link_vel_w.torch[:, 1, :2] = 1000
    for _ in range(6):
        speed_period(action, [0.0, 0.0])
    assert not action.base_vertical_speed_rms()[0].any()


def test_rms_advances_even_without_reward_reads():
    action, *_ = make_dwell()
    for _ in range(6):
        speed_period(action, [0.01, 0.01], finalize=False)
    rms, full = action.base_vertical_speed_rms()
    assert full.all()
    torch.testing.assert_close(rms, torch.full((3,), 0.01))


@pytest.mark.parametrize("finalized", [False, True])
def test_partial_reset_preserves_other_rms_histories_and_requires_refill(finalized):
    action, _, _, _, _, reward = make_dwell()
    fill_window(action, torch.tensor([0.01, 0.03, 0.0]))
    speed_period(action, [torch.tensor([0.01, 0.03, 0.0])] * 2, finalize=finalized)
    action.reset(torch.tensor([1]))
    reward.reset(torch.tensor([1]))
    rms, full = action.base_vertical_speed_rms()
    torch.testing.assert_close(rms, torch.tensor([0.01, 0.0, 0.0]))
    assert full.tolist() == [True, False, True]
    for _ in range(5):
        speed_period(action, [0.0, 0.0])
        assert not action.base_vertical_speed_rms()[1][1]
    speed_period(action, [0.0, 0.0])
    assert action.base_vertical_speed_rms()[1].all()
    action.reset()
    assert not action.base_vertical_speed_rms()[0].any()
    assert not action.base_vertical_speed_rms()[1].any()


def test_dwell_uses_joint_active_pose_gates_current_targets_and_excludes_failures():
    action, _, command, position, orientation, reward = make_dwell()
    fill_window(action)
    position[:] = torch.tensor([[0.01, 100.0], [100.0, 0.009], [0.009, 0.009]])
    orientation[:] = torch.tensor([[0.04, 100.0], [100.0, 0.03], [0.03, 0.03]])
    torch.testing.assert_close(reward(action._env), torch.ones(3))
    counts = reward._steps.clone()
    position[0, 0], orientation[1, 1], position[2, 1] = 0.0101, 0.0401, 0.0101
    assert not reward(action._env).any()
    assert action.base_vertical_speed_rms()[1].all()  # Goal refresh does not erase physical history.
    torch.testing.assert_close(reward._steps, counts)
    position.zero_()
    orientation.zero_()
    action._env.termination_manager.terminated[1] = True
    torch.testing.assert_close(reward(action._env), torch.tensor([1.0, 0.0, 1.0]))
    command.active_mask[0] = False
    assert reward(action._env)[0] == 0


def test_dwell_speed_tolerance_and_shortest_angle_are_in_physical_units():
    action, _, _, _, orientation, reward = make_dwell()
    fill_window(action, torch.tensor([0.019, 0.021, 0.0]))
    orientation[2] = 0.06  # A quaternion-vector threshold of .04 would incorrectly admit this angle.
    torch.testing.assert_close(reward(action._env), torch.tensor([1.0, 0.0, 0.0]))


def test_dwell_weight_dt_diagnostics_and_selected_episode_reset():
    action, _, _, _, _, _ = make_dwell()
    env = action._env
    manager = RewardManager({"dwell": MobileRewardsCfg().dwell}, env)
    for _ in range(6):
        speed_period(action, [0.0, 0.0])
        manager.compute(env.step_dt)
    cfg = manager.get_term_cfg("dwell")
    reward = cfg.func
    torch.testing.assert_close(reward(env), torch.ones(3))
    torch.testing.assert_close(reward._steps, torch.full((3,), 6.0))
    before = reward._sums[1:].clone()
    logs = manager.reset(torch.tensor([0]))
    torch.testing.assert_close(logs["Episode_Reward/dwell"], torch.tensor(0.2 * env.step_dt / 10))
    assert env.extras["log"]["Dwell/pose_ready_fraction"] == 1
    assert env.extras["log"]["Dwell/ready_fraction"] == pytest.approx(1 / 6)
    assert env.extras["log"]["Dwell/base_vertical_speed_rms_m_s"] == 0
    assert not reward._sums[0].any() and reward._steps[0] == 0
    torch.testing.assert_close(reward._sums[1:], before, atol=0, rtol=0)
    assert action.base_vertical_speed_rms()[1].all()  # Reward reset does not own physical history.


@pytest.mark.parametrize("window", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_rms_windows_fail(window):
    action, *_ = make_dwell()
    action.cfg.vertical_speed_rms_window_s = window
    with pytest.raises(ValueError, match="positive and finite"):
        action._initialize_base_motion_buffers()


@pytest.mark.parametrize("key", ["position_tolerance_m", "orientation_tolerance_rad", "vertical_speed_tolerance_m_s"])
@pytest.mark.parametrize("value", [0.0, -0.01, float("nan"), float("inf")])
def test_invalid_dwell_tolerances_fail(key, value):
    cfg = MobileRewardsCfg().dwell
    cfg.params[key] = value
    with pytest.raises(ValueError, match=key):
        DwellReward(cfg, None)


def test_dwell_is_mobile_only_and_exported_without_changing_other_weights():
    from g1.tasks.upper_limb_agent import mdp
    from g1.tasks.upper_limb_agent.config.g1.env_cfg import RewardsCfg

    cfg = MobileRewardsCfg().dwell
    assert cfg.func is DwellReward and mdp.DwellReward is DwellReward and cfg.weight == 0.2
    assert MobileActionsCfg().joint_pos.vertical_speed_rms_window_s == 0.1
    assert not hasattr(RewardsCfg(), "dwell")
    assert MobileRewardsCfg().base_vertical_acc.weight == -0.12
    assert MobileRewardsCfg().lower_command_rate.weight == -0.10
