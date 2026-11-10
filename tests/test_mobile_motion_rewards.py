"""Mobile base link kinematics, complete substeps, cache resets and reward registration."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardManager, RewardTermCfg

from g1.tasks.upper_limb_agent.mdp.actions import UpperJointPositionAction
from g1.tasks.upper_limb_agent.mdp.frozen_lower_actions import FrozenLowerAction, FrozenLowerActionCfg
from g1.tasks.upper_limb_agent.mdp.rewards import (
    base_ang_acc_l2_substeps,
    base_ang_vel_l2_substeps,
    base_lin_acc_l2_substeps,
    base_lin_vel_l2_substeps,
    base_vertical_acc_excess_l2_substeps,
)

BASE_REWARDS = {
    "base_lin_vel": base_lin_vel_l2_substeps,
    "base_ang_vel": base_ang_vel_l2_substeps,
    "base_lin_acc": base_lin_acc_l2_substeps,
    "base_ang_acc": base_ang_acc_l2_substeps,
}


class JointIds:
    def __init__(self):
        self.torch = torch.tensor([0, 1])

    def __len__(self):
        return 2


def make_mobile(decimation=2):
    """Use real Upper processing and Mobile sampling; stub only asset and frozen inference."""
    body_names = ["left_wrist_yaw_link", "pelvis", "right_wrist_yaw_link"]
    pose = torch.zeros(3, 3, 7)
    pose[..., 6] = 1
    data = SimpleNamespace(
        default_joint_pos=SimpleNamespace(torch=torch.zeros(3, 3)),
        joint_acc=SimpleNamespace(torch=torch.zeros(3, 3)),
        body_link_pose_w=SimpleNamespace(torch=pose),
        body_com_pose_w=SimpleNamespace(torch=pose.clone()),
        body_link_vel_w=SimpleNamespace(torch=torch.zeros(3, 3, 6)),
        body_com_vel_w=SimpleNamespace(torch=torch.zeros(3, 3, 6)),
        body_com_acc_w=SimpleNamespace(torch=torch.zeros(3, 3, 6)),
    )
    position_target = torch.zeros(3, 3)
    calls = {"inference": 0, "lower_apply": 0, "upper_targets": []}

    def find_bodies(query, **kwargs):
        names = [query] if isinstance(query, str) else query
        return [body_names.index(name) for name in names], names

    def set_target(target, joint_ids):
        position_target[:, joint_ids] = target
        calls["upper_targets"].append(target.clone())

    robot = SimpleNamespace(
        data=data,
        num_joints=3,
        find_bodies=find_bodies,
        find_joints=lambda query, **kwargs: (JointIds(), ["j0", "j1"]),
        set_joint_position_target_index=set_target,
        actuators=SimpleNamespace(target_command=SimpleNamespace(position=SimpleNamespace(torch=position_target))),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene={"robot": robot},
        cfg=SimpleNamespace(decimation=decimation),
        step_dt=decimation / 120,
        max_episode_length_s=10,
        extras={"log": {}},
        sim=SimpleNamespace(physics_manager=SimpleNamespace(handles_decimation=lambda: False), is_playing=lambda: True),
    )
    action = object.__new__(FrozenLowerAction)
    cfg = FrozenLowerActionCfg(asset_name="robot", joint_names=["j0", "j1"], preserve_order=True, scale=0.5)
    UpperJointPositionAction.__init__(action, cfg, env)
    action._base_body_id = 1  # Deliberately not the first link.
    action._base_motion_sum, action._base_motion_mean = torch.zeros(3, 4), None
    action._base_vertical_acc_samples = torch.zeros(3, decimation)
    action._motion_sum, action._motion_steps = torch.zeros(3, 9), torch.zeros(3)
    action._update_motion_metrics = lambda: None
    action.lower_command, action.lower_actions = torch.zeros(3, 6), torch.zeros(3, 1)
    action.last_lower_observations = torch.zeros(3, 650)
    action.lower_observations = SimpleNamespace(
        compute_group=lambda *args, **kwargs: torch.zeros(3, 650), reset=lambda env_ids: None
    )

    def infer(observations):
        calls["inference"] += 1
        return torch.zeros(3, 1)

    def lower_apply():
        calls["lower_apply"] += 1

    action.frozen_policy = infer
    action.lower_action = SimpleNamespace(
        process_actions=lambda values: None,
        apply_actions=lower_apply,
        reset=lambda env_ids: None,
        _joint_ids=torch.tensor([2]),
        _processed_actions=torch.zeros(3, 1),
    )
    command = SimpleNamespace(processed_actions=torch.zeros(3, 3))
    height = SimpleNamespace(processed_actions=torch.zeros(3, 1))
    terms = {"joint_pos": action, "lower_command": command, "height_offset": height}
    env.action_manager = SimpleNamespace(get_term=terms.__getitem__)
    return action, calls


def run_period(action, factors):
    action.process_actions(torch.zeros(3, 2))
    data = action._asset.data
    for factor in factors:
        action.apply_actions()
        velocity = torch.tensor([1.0, 0, 0, 0, 2, 0]) * factor
        acceleration = torch.tensor([0.0, 0, 3, 4, 0, 0]) * factor
        data.body_link_vel_w.torch[:, 1] = velocity
        data.body_com_vel_w.torch[:, 1] = velocity
        data.body_com_acc_w.torch[:, 1] = acceleration


@pytest.mark.parametrize("decimation", [1, 2, 4])
def test_base_motion_covers_all_substeps_and_keeps_inference_and_targets(decimation):
    action, calls = make_mobile(decimation)
    action._asset.data.body_com_acc_w.torch.fill_(1e6)  # Previous period must be excluded.
    assert not action.mean_base_motion_l2().any()
    factors = list(range(1, decimation + 1))
    run_period(action, factors)
    mean_squared = sum(f * f for f in factors) / decimation
    expected = torch.tensor([1.0, 4, 9, 16]).expand(3, -1) * mean_squared
    torch.testing.assert_close(action.mean_base_motion_l2(), expected)
    assert calls["inference"] == 1 and calls["lower_apply"] == decimation
    assert len(calls["upper_targets"]) == decimation
    assert not any(target.any() for target in calls["upper_targets"])
    for index, reward in enumerate(BASE_REWARDS.values()):
        torch.testing.assert_close(reward(action._env), expected[:, index])


def test_first_substep_spikes_and_reversals_do_not_disappear():
    action, _ = make_mobile()
    run_period(action, [1000, 0])
    torch.testing.assert_close(action.mean_base_motion_l2(), torch.tensor([1.0, 4, 9, 16]).expand(3, -1) * 500000)
    run_period(action, [1000, -1000])
    torch.testing.assert_close(action.mean_base_motion_l2(), torch.tensor([1.0, 4, 9, 16]).expand(3, -1) * 1000000)


def test_base_costs_use_selected_link_origin_and_three_world_axes_without_state_writes():
    action, _ = make_mobile(decimation=1)
    run_period(action, [0])
    data = action._asset.data
    data.body_link_pose_w.torch[:, 1, 0] = 1  # COM-to-origin r = [1, 0, 0].
    data.body_link_vel_w.torch[:, 1] = torch.tensor([2.0, 3, 4, 0, 0, 2])
    data.body_com_vel_w.torch[:, 1, 3:] = torch.tensor([0.0, 0, 2])
    data.body_com_acc_w.torch[:, 1] = torch.tensor([1.0, 2, 3, 0, 0, 3])
    data.body_com_acc_w.torch[:, [0, 2]] = 1e6
    before = {name: value.torch.clone() for name, value in vars(data).items()}
    # a_origin = [1,2,3] + [0,3,0] + [-4,0,0] = [-3,5,3].
    torch.testing.assert_close(action.mean_base_motion_l2(), torch.tensor([29.0, 4, 43, 9]).expand(3, -1))
    for name, expected in before.items():
        torch.testing.assert_close(getattr(data, name).torch, expected, rtol=0, atol=0)


@pytest.mark.parametrize("finalized", [False, True])
def test_base_cache_survives_partial_reset_and_starts_fresh_next_period(finalized):
    action, _ = make_mobile()
    run_period(action, [1, 3])
    if finalized:
        base_lin_vel_l2_substeps(action._env)
    action.reset(torch.tensor([1]))
    expected = torch.tensor([5.0, 20, 45, 80]).expand(3, -1).clone()
    expected[1] = 0
    action._asset.data.body_com_acc_w.torch.fill_(1e6)
    torch.testing.assert_close(action.mean_base_motion_l2(), expected)
    run_period(action, [2, 2])
    torch.testing.assert_close(action.mean_base_motion_l2(), torch.tensor([4.0, 16, 36, 64]).expand(3, -1))
    action.reset()
    assert not action.mean_base_motion_l2().any()


def test_incomplete_substeps_fail_and_repeated_reads_do_not_resample():
    action, _ = make_mobile()
    action.process_actions(torch.zeros(3, 2))
    action.apply_actions()
    with pytest.raises(RuntimeError, match="expected 2 physical substeps, received 1"):
        action.mean_base_motion_l2()
    action.apply_actions()
    expected = action.mean_base_motion_l2().clone()
    action._asset.data.body_link_vel_w.torch.fill_(100)
    action._asset.data.body_com_acc_w.torch.fill_(100)
    torch.testing.assert_close(action.mean_base_motion_l2(), expected)
    with pytest.raises(RuntimeError, match="more times"):
        action.apply_actions()


@pytest.mark.parametrize("reward", list(BASE_REWARDS.values()))
def test_base_rewards_require_mobile_action(reward):
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: object()))
    with pytest.raises(TypeError, match="FrozenLowerAction"):
        reward(env)


def test_base_reward_weights_dt_and_episode_logs_are_applied_once():
    action, _ = make_mobile()
    run_period(action, [1, 3])
    weights = [-0.1, -0.02, -1e-4, -1e-5]
    cfg = {
        name: RewardTermCfg(func=func, weight=weight)
        for (name, func), weight in zip(BASE_REWARDS.items(), weights, strict=True)
    }
    manager = RewardManager(cfg, action._env)
    costs = torch.tensor([5.0, 20, 45, 80])
    expected = (costs * torch.tensor(weights)).sum() * action._env.step_dt
    torch.testing.assert_close(manager.compute(action._env.step_dt), expected.expand(3))
    logs = manager.reset()
    for index, name in enumerate(BASE_REWARDS):
        torch.testing.assert_close(
            logs[f"Episode_Reward/{name}"], costs[index] * weights[index] * action._env.step_dt / 10
        )


def test_mobile_retains_velocity_weights_and_enables_only_the_vertical_soft_base_cost():
    from g1.tasks.upper_limb_agent import mdp
    from g1.tasks.upper_limb_agent.config.g1.env_cfg import RewardsCfg
    from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import MobileActionsCfg, MobileRewardsCfg
    from g1.tasks.upper_limb_agent.mdp.rewards import lower_command_magnitude_l2, lower_command_overflow_l2

    rewards = {
        "lower_command_overflow": lower_command_overflow_l2,
        "lower_command_magnitude": lower_command_magnitude_l2,
        **BASE_REWARDS,
    }
    mobile, fixed = MobileRewardsCfg(), RewardsCfg()
    for name, function in rewards.items():
        cfg = getattr(mobile, name)
        expected_weight = {"lower_command_overflow": -3e-4, "lower_command_magnitude": -0.05}.get(name, 0.0)
        assert cfg.weight == expected_weight and cfg.func is function
        assert getattr(mdp, function.__name__) is function
        assert cfg.params == {"action_name": "lower_command" if name.startswith("lower_command") else "joint_pos"}
        assert not hasattr(fixed, name)
    assert MobileActionsCfg().joint_pos.base_body_name == "pelvis"
    assert mobile.lower_command_rate.weight == -0.10 and mobile.action_rate.weight == -0.03
    assert mobile.base_vertical_acc.func is base_vertical_acc_excess_l2_substeps
    assert mobile.base_vertical_acc.weight == -0.05
    assert mobile.base_vertical_acc.params == {"action_name": "joint_pos", "soft_limit": 4.0}
    assert mdp.base_vertical_acc_excess_l2_substeps is base_vertical_acc_excess_l2_substeps
    assert not hasattr(fixed, "base_vertical_acc")


def test_zero_weight_base_rewards_skip_readers_and_preserve_existing_reward():
    action, _ = make_mobile()
    env = action._env
    cfg = {name: RewardTermCfg(func=reward, weight=0.0) for name, reward in BASE_REWARDS.items()}
    cfg["base_vertical_acc"] = RewardTermCfg(func=base_vertical_acc_excess_l2_substeps, weight=0.0)
    cfg["existing"] = RewardTermCfg(func=lambda env: torch.full((3,), 2.0), weight=1.0)
    manager = RewardManager(cfg, env)

    def unexpected_read():
        pytest.fail("Zero-weight rewards must be skipped by RewardManager")

    action.mean_base_motion_l2 = unexpected_read
    action.mean_base_vertical_acc_excess_l2 = lambda *_: unexpected_read()
    torch.testing.assert_close(manager.compute(env.step_dt), torch.full((3,), 2 * env.step_dt))
    logs = manager.reset()
    assert all(logs[f"Episode_Reward/{name}"] == 0 for name in BASE_REWARDS)
    assert logs["Episode_Reward/base_vertical_acc"] == 0


@pytest.mark.parametrize("decimation", [1, 2, 4])
def test_vertical_acc_covers_all_substeps_and_preserves_original_motion_interface(decimation):
    action, calls = make_mobile(decimation)
    assert not base_vertical_acc_excess_l2_substeps(action._env).any()
    action._asset.data.body_com_acc_w.torch.fill_(1e6)
    factors = list(range(1, decimation + 1))
    run_period(action, factors)
    costs = [(max(abs(3 * f) / 4 - 1, 0)) ** 2 for f in factors]
    torch.testing.assert_close(
        base_vertical_acc_excess_l2_substeps(action._env), torch.full((3,), sum(costs) / decimation)
    )
    assert action.mean_base_motion_l2().shape == (3, 4)
    assert calls["inference"] == 1 and calls["lower_apply"] == decimation
    assert len(calls["upper_targets"]) == decimation and not any(t.any() for t in calls["upper_targets"])


@pytest.mark.parametrize("factors,expected", [([8 / 3, 0], 0.5), ([0, 8 / 3], 0.5), ([8 / 3, -8 / 3], 1.0)])
def test_vertical_threshold_precedes_averaging_and_preserves_impulses_and_sign_reversals(factors, expected):
    action, _ = make_mobile()
    run_period(action, factors)
    torch.testing.assert_close(base_vertical_acc_excess_l2_substeps(action._env), torch.full((3,), expected))


def test_vertical_acc_uses_selected_link_origin_world_z_without_charging_xy_or_velocity():
    action, _ = make_mobile(decimation=1)
    run_period(action, [0])
    data = action._asset.data
    data.body_link_pose_w.torch[:, 1, 0] = 1
    data.body_link_pose_w.torch[:, 1, 3:] = torch.tensor([0, 2**-0.5, 0, 2**-0.5])
    data.body_link_vel_w.torch[:, 1, :3] = torch.tensor([100.0, 200.0, 0.5])
    data.body_com_acc_w.torch[:, 1] = torch.tensor([1e6, 1e6, 0.0, 0, -8.0, 0])
    data.body_com_acc_w.torch[:, [0, 2]] = 1e6
    before = {name: value.torch.clone() for name, value in vars(data).items()}
    # COM az=0; alpha x r contributes world az=8. Body orientation is not projected onto local Z.
    torch.testing.assert_close(base_vertical_acc_excess_l2_substeps(action._env), torch.ones(3))
    for name, expected in before.items():
        torch.testing.assert_close(getattr(data, name).torch, expected, rtol=0, atol=0)


def test_vertical_acc_repeated_reads_and_different_limits_reuse_signed_physical_samples():
    action, _ = make_mobile()
    run_period(action, [8 / 3, 0])
    torch.testing.assert_close(action.mean_base_vertical_acc_excess_l2(), torch.full((3,), 0.5))
    action._asset.data.body_com_acc_w.torch.fill_(1e6)
    torch.testing.assert_close(action.mean_base_vertical_acc_excess_l2(2), torch.full((3,), 4.5))
    torch.testing.assert_close(action.mean_base_vertical_acc_excess_l2(), torch.full((3,), 0.5))


@pytest.mark.parametrize("finalized_by", [None, "base", "vertical"])
def test_vertical_samples_survive_partial_reset_even_when_reward_was_disabled(finalized_by):
    action, _ = make_mobile()
    run_period(action, [8 / 3, -8 / 3])
    if finalized_by == "base":
        action.mean_base_motion_l2()
    elif finalized_by == "vertical":
        action.mean_base_vertical_acc_excess_l2()
    action.reset(torch.tensor([1]))
    action._asset.data.body_com_acc_w.torch.fill_(1e6)
    torch.testing.assert_close(action.mean_base_vertical_acc_excess_l2(), torch.tensor([1.0, 0, 1.0]))
    run_period(action, [0, 0])
    assert not action.mean_base_vertical_acc_excess_l2().any()
    action.reset()
    assert not action.mean_base_vertical_acc_excess_l2().any()


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf")])
def test_vertical_acc_rejects_invalid_soft_limits(limit):
    action, _ = make_mobile()
    with pytest.raises(ValueError, match="positive and finite"):
        action.mean_base_vertical_acc_excess_l2(limit)


def test_vertical_acc_requires_complete_substeps_and_a_mobile_action():
    action, _ = make_mobile()
    action.process_actions(torch.zeros(3, 2))
    action.apply_actions()
    with pytest.raises(RuntimeError, match="expected 2 physical substeps, received 1"):
        base_vertical_acc_excess_l2_substeps(action._env)
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: object()))
    with pytest.raises(TypeError, match="FrozenLowerAction"):
        base_vertical_acc_excess_l2_substeps(env)


def test_vertical_reward_weight_dt_and_episode_log_are_applied_once():
    action, _ = make_mobile()
    run_period(action, [8 / 3, 0])
    cfg = {"base_vertical_acc": RewardTermCfg(func=base_vertical_acc_excess_l2_substeps, weight=-0.05)}
    manager = RewardManager(cfg, action._env)
    expected = -0.05 * 0.5 * action._env.step_dt
    torch.testing.assert_close(manager.compute(action._env.step_dt), torch.full((3,), expected))
    torch.testing.assert_close(manager.reset()["Episode_Reward/base_vertical_acc"], torch.tensor(expected / 10))
