"""Arm/torso force cancellation, substep costs and reset-safe contact diagnostics."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardManager, RewardTermCfg, SceneEntityCfg

from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import UpperLimbMobileEnvCfg
from g1.tasks.upper_limb_agent.mdp.contacts import (
    ARM_CONTACT_BODY_PATHS,
    TorsoContactPenalty,
    torso_contact_termination,
)


def contact_env(n=3):
    pair = torch.zeros(n, 3, 1, len(ARM_CONTACT_BODY_PATHS), 3)
    net = torch.zeros(n, 3, 1, 3)
    data = SimpleNamespace(
        normal_force_matrix_w_history=SimpleNamespace(torch=pair),
        net_normal_forces_w_history=SimpleNamespace(torch=net),
    )
    return (
        SimpleNamespace(
            num_envs=n,
            device="cpu",
            common_step_counter=1,
            step_dt=0.02,
            cfg=SimpleNamespace(decimation=2, sim=SimpleNamespace(dt=0.01)),
            scene={"torso_contact_pairs": SimpleNamespace(data=data)},
            extras={},
        ),
        pair,
        net,
    )


def penalty(env, **params):
    cfg = RewardTermCfg(func=TorsoContactPenalty, weight=-2, params=params)
    return TorsoContactPenalty(cfg, env)


def test_hard_contact_uses_arm_load_and_retains_strict_non_arm_threshold():
    env, pair, net = contact_env()
    pair[0, :, 0, 0, 0] = 20
    pair[1, 2, 0, 0, 0] = 20.01  # Older history still belongs to the hard gate.
    net.copy_(pair.sum(dim=-2))
    net[2, 0, 0, 2] = 1.01  # Ground / non-arm force remains a failure.
    torch.testing.assert_close(torso_contact_termination(env), torch.tensor([False, True, True]))
    pair.zero_()
    net.zero_()
    assert not torso_contact_termination(env).any()


def test_opposing_arm_contacts_cannot_cancel_the_hard_gate_or_cost():
    env, pair, net = contact_env()
    pair[:, :2, 0, 0, 0] = 11
    pair[:, :2, 0, 1, 0] = -11
    net.copy_(pair.sum(dim=-2))
    assert not net.any() and torso_contact_termination(env).all()
    torch.testing.assert_close(penalty(env)(env), torch.ones(3))


def test_ground_contact_is_checked_after_removing_simultaneous_arm_force():
    env, pair, net = contact_env()
    pair[:, :2, 0, 0, 0] = 10
    net.copy_(pair.sum(dim=-2))
    net[:, 0, 0, 0] -= 2
    assert torso_contact_termination(env).all()


def test_cost_is_persistent_force_scaled_and_uses_each_current_substep_once():
    env, pair, net = contact_env()
    pair[0, :2, 0, 0, 0] = 1  # Exact soft threshold has no cost.
    pair[1, :2, 0, 0, 0] = 10.5
    pair[2, 0, 0, 0, 0] = 20  # Half a control period in contact.
    pair[:, 2, 0, 0, 0] = 200  # Previous control step is not charged again.
    net.copy_(pair.sum(dim=-2))
    reward = penalty(env)
    expected = torch.tensor([0, 0.625, 0.5])
    torch.testing.assert_close(reward(env), expected)
    sums = reward._sums.clone()
    torch.testing.assert_close(reward(env), expected)
    torch.testing.assert_close(reward._sums, sums)
    env.common_step_counter += 1
    torch.testing.assert_close(reward(env), expected)
    reward.reset()
    logs = env.extras["log"]
    assert logs["Contact/torso/self_contact_fraction"] == pytest.approx(0.5)
    assert logs["Contact/torso/episode_peak_self_force_n"] == pytest.approx((1 + 10.5 + 20) / 3)
    assert logs["Contact/torso/self_force_over_20n_fraction"] == 0


def test_contact_onsets_continue_across_steps_and_partial_resets_keep_other_environments():
    env, pair, net = contact_env()
    pair[:, :2, 0, 0, 0] = 2
    net.copy_(pair.sum(dim=-2))
    reward = penalty(env)
    reward(env)
    env.common_step_counter += 1
    reward(env)
    torch.testing.assert_close(reward._onsets, torch.ones(3))
    saved = reward._sums.clone()
    reward.reset(torch.tensor([0]))
    assert not reward._sums[0].any() and not reward._previous_contact[0]
    torch.testing.assert_close(reward._sums[1:], saved[1:])
    env.common_step_counter += 1
    reward(env)
    torch.testing.assert_close(reward._onsets, torch.ones(3))
    torch.testing.assert_close(reward._sums[:, 0], torch.tensor([0.02, 0.06, 0.06]))
    reward.reset()
    assert not reward._onsets.any() and not reward._previous_contact.any()


def test_reward_manager_applies_weight_and_control_dt_exactly_once():
    env, pair, net = contact_env()
    pair[:, :2, 0, 0, 0] = 10.5
    net.copy_(pair.sum(dim=-2))
    env.max_episode_length_s = 10
    env.sim = SimpleNamespace(is_playing=lambda: True)
    env.scene["torso_contact_pairs"].find_bodies = lambda *args, **kw: ([0], ["torso_link"])
    cfg = RewardTermCfg(
        func=TorsoContactPenalty, weight=-2, params={"sensor_cfg": SceneEntityCfg("torso_contact_pairs")}
    )
    manager = RewardManager({"torso_contact": cfg}, env)
    torch.testing.assert_close(manager.compute(env.step_dt), torch.full((3,), -0.625 * 2 * env.step_dt))


def test_mobile_config_connects_both_gates_to_same_filtered_sensor():
    cfg = UpperLimbMobileEnvCfg()
    sensor = cfg.scene.torso_contact_pairs
    assert len(sensor.filter_prim_paths_expr) == len(set(sensor.filter_prim_paths_expr)) == 30
    assert sensor.history_length >= cfg.decimation and sensor.update_period == cfg.sim.dt
    assert cfg.terminations.base_contact.func is torso_contact_termination
    assert cfg.terminations.base_contact.params["threshold"] == 20
    assert cfg.terminations.base_contact.params["other_threshold"] == 1
    assert cfg.rewards.torso_contact.func is TorsoContactPenalty and cfg.rewards.torso_contact.weight == -2
    assert (
        cfg.rewards.torso_contact.params["sensor_cfg"].name == cfg.terminations.base_contact.params["sensor_cfg"].name
    )
    assert cfg.terminations.knee_contact.params["threshold"] == 1


@pytest.mark.parametrize("params", [{"soft_threshold": -1}, {"hard_threshold": 1}, {"contact_cost": 0}])
def test_invalid_contact_cost_configuration_is_rejected(params):
    env, _, _ = contact_env()
    with pytest.raises(ValueError):
        penalty(env, **params)
