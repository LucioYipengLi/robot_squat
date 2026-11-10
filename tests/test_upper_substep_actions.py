"""Target speeds, physical-substep acceleration coverage and standard Upper actions."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import RewardManager, RewardTermCfg

from g1.tasks.upper_limb_agent.mdp.actions import UpperJointPositionAction, UpperJointPositionActionCfg
from g1.tasks.upper_limb_agent.mdp.rewards import (
    end_effector_relative_lin_acc_l2_substeps,
    joint_acc_l2_substeps,
    joint_target_speed_excess_l2,
)


class JointIds:
    def __init__(self, values):
        self.torch = torch.tensor(values)

    def __len__(self):
        return len(self.torch)


def make_action(decimation=2, *, batched=False, clip=None, scale=0.5, **overrides):
    names = [f"j{i}" for i in range(6)]
    defaults = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10
    targets = []
    position_target = defaults.clone()
    body_names = ["left_wrist_yaw_link", "torso", "pelvis", "right_wrist_yaw_link"]
    pose = torch.zeros(3, 4, 7)
    pose[..., 6] = 1

    def find_bodies(query, **kwargs):
        keys = [query] if isinstance(query, str) else query
        return [body_names.index(key) for key in keys], keys

    def find_joints(query, **kwargs):
        assert kwargs == {"preserve_order": True, "as_proxy": True}
        return JointIds([names.index(name) for name in query]), query

    def set_target(target, joint_ids):
        targets.append((target.clone(), joint_ids.clone()))
        position_target[:, joint_ids] = target

    robot = SimpleNamespace(
        find_joints=find_joints,
        find_bodies=find_bodies,
        num_joints=len(names),
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=defaults),
            joint_acc=SimpleNamespace(torch=torch.zeros_like(defaults)),
            body_link_pose_w=SimpleNamespace(torch=pose),
            body_com_pose_w=SimpleNamespace(torch=pose.clone()),
            body_link_vel_w=SimpleNamespace(torch=torch.zeros(3, 4, 6)),
            body_com_vel_w=SimpleNamespace(torch=torch.zeros(3, 4, 6)),
            body_com_acc_w=SimpleNamespace(torch=torch.zeros(3, 4, 6)),
        ),
        set_joint_position_target_index=set_target,
        actuators=SimpleNamespace(target_command=SimpleNamespace(position=SimpleNamespace(torch=position_target))),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene={"robot": robot},
        cfg=SimpleNamespace(decimation=decimation),
        step_dt=decimation / 120,
        sim=SimpleNamespace(physics_manager=SimpleNamespace(handles_decimation=lambda: batched)),
    )
    cfg = UpperJointPositionActionCfg(
        asset_name="robot", joint_names=["j4", "j1"], preserve_order=True, scale=scale, clip=clip, **overrides
    )
    return UpperJointPositionAction(cfg, env), targets


def test_target_velocity_uses_signed_processed_targets_and_control_dt_without_limiting():
    action, _ = make_action()
    values = torch.tensor([[0.1, -0.3], [0.4, 0.2], [-0.2, 0.5]])
    assert not action.joint_target_velocity.any()
    action.process_actions(values)
    torch.testing.assert_close(action.joint_target_velocity, 0.5 * values / action._env.step_dt)
    expected_target = action._offset + values * 0.5
    torch.testing.assert_close(action.processed_actions, expected_target)
    before = action.joint_target_velocity.clone()
    action.apply_actions()
    action.apply_actions()
    torch.testing.assert_close(action.joint_target_velocity, before)
    action.process_actions(values * 2)
    torch.testing.assert_close(action.joint_target_velocity, 0.5 * values / action._env.step_dt)


def test_target_velocity_respects_per_joint_scaling_and_full_reset():
    action, _ = make_action(scale={"j4": 0.25, "j1": 1.0})
    action.process_actions(torch.ones(3, 2))
    expected = torch.tensor([0.25, 1.0]).expand(3, -1) / action._env.step_dt
    torch.testing.assert_close(action.joint_target_velocity, expected)
    action.apply_actions()
    action.apply_actions()
    action._asset.actuators.target_command.position.torch.copy_(action._asset.data.default_joint_pos.torch)
    action.reset()
    assert not action.joint_target_velocity.any()
    action.process_actions(torch.zeros(3, 2))
    assert not action.joint_target_velocity.any()


def test_target_velocity_follows_clipping_and_reading_actual_motion_does_not_affect_it():
    action, _ = make_action(clip={".*": (-0.2, 0.2)})
    initial = action._asset.actuators.target_command.position.torch[:, [4, 1]].clone()
    action.process_actions(torch.full((3, 2), 100.0))
    torch.testing.assert_close(action.joint_target_velocity, (action.processed_actions - initial) / action._env.step_dt)
    action._asset.data.joint_acc.torch.fill_(1e5)
    action.process_actions(torch.full((3, 2), 200.0))
    assert not action.joint_target_velocity.any()  # Both raw actions clip to exactly the same goal.


@pytest.mark.parametrize("decimation", [1, 2, 4])
def test_target_velocity_uses_control_period_rather_than_physics_period(decimation):
    action, _ = make_action(decimation)
    action.process_actions(torch.ones(3, 2))
    torch.testing.assert_close(action.joint_target_velocity, torch.full((3, 2), 0.5 / (decimation / 120)))


def test_target_velocity_partial_reset_rebases_on_pd_goal_and_penalizes_first_new_target():
    action, _ = make_action()
    action.process_actions(torch.ones(3, 2))
    old_velocity = action.joint_target_velocity.clone()
    # Reset event synchronizes the selected row's physical PD target before ActionTerm.reset().
    pd_target = action._asset.actuators.target_command.position.torch
    pd_target[1] = action._asset.data.default_joint_pos.torch[1]
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.joint_target_velocity[[0, 2]], old_velocity[[0, 2]])
    assert not action.joint_target_velocity[1].any()
    action.process_actions(torch.ones(3, 2))
    torch.testing.assert_close(action.joint_target_velocity[[0, 2]], torch.zeros(2, 2))
    torch.testing.assert_close(action.joint_target_velocity[1], torch.full((2,), 0.5 / action._env.step_dt))


def test_target_speed_cost_has_a_dead_zone_normalizes_units_and_preserves_targets():
    action, _ = make_action()
    env = action._env
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    speeds = torch.tensor([[0.0, 2.5], [5.0, -5.0], [10.0, -10.0]])
    values = speeds * env.step_dt / action.cfg.scale
    action.process_actions(values)
    before = action.raw_actions.clone(), action.processed_actions.clone(), action.joint_target_velocity.clone()
    expected = torch.tensor([0.0, 2.0, 18.0])
    torch.testing.assert_close(joint_target_speed_excess_l2(env), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(joint_target_speed_excess_l2(env), expected, atol=1e-5, rtol=1e-5)
    for current, saved in zip(
        (action.raw_actions, action.processed_actions, action.joint_target_velocity), before, strict=True
    ):
        torch.testing.assert_close(current, saved, rtol=0, atol=0)


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf")])
def test_target_speed_rejects_invalid_soft_limits(limit):
    with pytest.raises(ValueError, match="positive and finite"):
        joint_target_speed_excess_l2(SimpleNamespace(), soft_limit=limit)


@pytest.mark.parametrize("weight", [0.0, -1e-4])
def test_target_speed_reward_applies_weight_and_time_once_and_keeps_velocity_when_disabled(weight):
    action, _ = make_action()
    env = action._env
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    env.max_episode_length_s = 10.0
    env.sim.is_playing = lambda: True
    # Each of the two selected joints moves its goal at 5 rad/s: normalized total cost = 2.
    action.process_actions(torch.full((3, 2), 5.0 * env.step_dt / action.cfg.scale))
    velocity = action.joint_target_velocity.clone()
    cfg = RewardTermCfg(func=joint_target_speed_excess_l2, weight=weight, params={"soft_limit": 2.5})
    manager = RewardManager({"target_speed": cfg}, env)
    expected = torch.full((3,), 2 * weight * env.step_dt)
    torch.testing.assert_close(manager.compute(env.step_dt), expected)
    torch.testing.assert_close(action.joint_target_velocity, velocity)
    torch.testing.assert_close(manager.reset()["Episode_Reward/target_speed"], expected.mean() / 10)


def run_period(action, samples):
    action.process_actions(torch.zeros(3, 2))
    for sample in samples:
        action.apply_actions()
        action._asset.data.joint_acc.torch.copy_(sample)


@pytest.mark.parametrize("decimation", [1, 2, 4])
def test_mean_covers_every_substep_and_only_controlled_joints(decimation):
    action, _ = make_action(decimation)
    data = action._asset.data.joint_acc.torch
    data.fill_(1e6)  # Previous control-period data must not leak into the new one.
    samples = []
    for k in range(decimation):
        sample = torch.full_like(data, 1e5)  # Ignore unselected joints, including Lower.
        sample[:, 4] = (k + 1) * torch.tensor([1.0, 2.0, 3.0])
        sample[:, 1] = -sample[:, 4] * 2
        samples.append(sample)
    expected = torch.stack(samples)[:, :, [4, 1]].square().sum(dim=-1).mean(dim=0)
    run_period(action, samples)
    torch.testing.assert_close(action.mean_joint_acc_l2(), expected)
    torch.testing.assert_close(data, samples[-1])


def test_first_substep_spike_is_penalized_and_opposite_signs_do_not_cancel():
    action, _ = make_action()
    spike = torch.zeros(3, 6)
    spike[:, 4] = 1000
    run_period(action, [spike, torch.zeros_like(spike)])
    torch.testing.assert_close(action.mean_joint_acc_l2(), torch.full((3,), 500000.0))
    run_period(action, [spike, -spike])
    torch.testing.assert_close(action.mean_joint_acc_l2(), torch.full((3,), 1000000.0))


def test_repeated_read_is_cached_and_next_control_period_starts_fresh():
    action, _ = make_action()
    assert not action.mean_joint_acc_l2().any()
    sample = torch.ones(3, 6)
    run_period(action, [sample, sample * 3])
    expected = torch.full((3,), 10.0)
    torch.testing.assert_close(action.mean_joint_acc_l2(), expected)
    action._asset.data.joint_acc.torch.fill_(100)
    torch.testing.assert_close(action.mean_joint_acc_l2(), expected)
    run_period(action, [sample * 2, sample * 2])
    torch.testing.assert_close(action.mean_joint_acc_l2(), torch.full((3,), 8.0))


@pytest.mark.parametrize("finalized", [False, True])
def test_partial_reset_preserves_other_environments_with_or_without_reward_read(finalized):
    action, _ = make_action()
    sample = torch.ones(3, 6)
    run_period(action, [sample, sample * 3])
    if finalized:
        action.mean_joint_acc_l2()
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.mean_joint_acc_l2(), torch.tensor([10.0, 0.0, 10.0]))
    run_period(action, [sample * 2, sample * 2])
    torch.testing.assert_close(action.mean_joint_acc_l2(), torch.full((3,), 8.0))
    action.reset()
    assert not action.mean_joint_acc_l2().any()


def test_sampling_leaves_parent_processing_targets_and_action_history_unchanged():
    action, targets = make_action(clip={".*": (-0.2, 0.2)})
    parent = JointPositionAction(action.cfg, action._env)
    values = torch.tensor([[0.1, -0.3], [0.4, 0.2], [-0.2, 0.5]])
    action.process_actions(values)
    parent.process_actions(values)
    torch.testing.assert_close(action.raw_actions, parent.raw_actions)
    torch.testing.assert_close(action.processed_actions, parent.processed_actions)
    for _ in range(2):
        action.apply_actions()
        parent.apply_actions()
        torch.testing.assert_close(targets[-2][0], targets[-1][0])
        torch.testing.assert_close(targets[-2][1], targets[-1][1])
    action.mean_joint_acc_l2()
    action.mean_end_effector_acc_l2()
    torch.testing.assert_close(action.raw_actions, values)
    torch.testing.assert_close(action.processed_actions, parent.processed_actions)


def test_sampling_requires_complete_visible_substeps():
    action, _ = make_action()
    action.process_actions(torch.zeros(3, 2))
    action.apply_actions()
    with pytest.raises(RuntimeError, match="expected 2 physical substeps, received 1"):
        action.mean_joint_acc_l2()
    with pytest.raises(RuntimeError, match="expected 2 physical substeps, received 1"):
        action.mean_end_effector_acc_l2()
    action.apply_actions()
    with pytest.raises(RuntimeError, match="more times"):
        action.apply_actions()
    with pytest.raises(ValueError, match="batches decimation internally"):
        make_action(batched=True)
    single, _ = make_action(decimation=1, batched=True)
    run_period(single, [torch.ones(3, 6)])
    torch.testing.assert_close(single.mean_joint_acc_l2(), torch.full((3,), 2.0))


def test_reward_is_weighted_and_time_scaled_once_and_keeps_the_existing_tag():
    action, _ = make_action()
    env = action._env
    env.action_manager = SimpleNamespace(get_term=lambda name: action if name == "joint_pos" else None)
    env.max_episode_length_s = 10.0
    env.sim.is_playing = lambda: True
    spike = torch.zeros(3, 6)
    spike[:, 4] = 1000
    run_period(action, [spike, torch.zeros_like(spike)])
    weight, dt = -3e-7, 1 / 60
    cfg = RewardTermCfg(func=joint_acc_l2_substeps, weight=weight, params={"action_name": "joint_pos"})
    manager = RewardManager({"joint_acc": cfg}, env)
    expected = torch.full((3,), 500000.0 * weight * dt)
    torch.testing.assert_close(manager.compute(dt), expected)
    assert manager.active_terms == ["joint_acc"]
    torch.testing.assert_close(manager.reset()["Episode_Reward/joint_acc"], expected.mean() / 10)
    env.action_manager.get_term = lambda name: object()
    with pytest.raises(TypeError, match="UpperJointPositionAction"):
        joint_acc_l2_substeps(env)


def run_end_effector_period(action, samples):
    action.process_actions(torch.zeros(3, 2))
    for sample in samples:
        action.apply_actions()
        action._asset.data.body_com_acc_w.torch.copy_(sample)


@pytest.mark.parametrize("decimation", [1, 2, 4])
def test_end_effector_mean_covers_substeps_and_removes_common_base_acceleration(decimation):
    action, _ = make_action(decimation)
    data = action._asset.data.body_com_acc_w.torch
    data.fill_(1e6)  # Old-period state must not enter the new period.
    samples, expected = [], torch.zeros(3)
    for k in range(decimation):
        sample = torch.zeros_like(data)
        sample[:, :, :3] = 100  # Shared base acceleration.
        relative = torch.tensor([1.0, 2.0, 3.0]) * (k + 1)
        sample[:, 0, 0] += relative  # Left wrist.
        sample[:, 3, 1] -= 2 * relative  # Right wrist, including an idle wrist.
        sample[:, 1, :3] = 1e6  # Unselected torso.
        samples.append(sample)
        expected += 5 * relative.square() / decimation
    run_end_effector_period(action, samples)
    torch.testing.assert_close(action.mean_end_effector_acc_l2(), expected)
    torch.testing.assert_close(data, samples[-1])


def test_end_effector_first_substep_spike_and_sign_reversal_are_preserved():
    action, _ = make_action()
    spike = torch.zeros(3, 4, 6)
    spike[:, 0, 0] = 1000
    run_end_effector_period(action, [spike, torch.zeros_like(spike)])
    torch.testing.assert_close(action.mean_end_effector_acc_l2(), torch.full((3,), 500000.0))
    run_end_effector_period(action, [spike, -spike])
    torch.testing.assert_close(action.mean_end_effector_acc_l2(), torch.full((3,), 1000000.0))


@pytest.mark.parametrize("finalized", [False, True])
def test_end_effector_cache_partial_reset_and_next_period_are_independent(finalized):
    action, _ = make_action()
    assert not action.mean_end_effector_acc_l2().any()
    sample = torch.zeros(3, 4, 6)
    sample[:, 0, 0] = 1
    run_end_effector_period(action, [sample, sample * 3])
    if finalized:
        action.mean_end_effector_acc_l2()
    action.reset(torch.tensor([1]))
    expected = torch.tensor([5.0, 0.0, 5.0])
    torch.testing.assert_close(action.mean_end_effector_acc_l2(), expected)
    action._asset.data.body_com_acc_w.torch.fill_(100)
    torch.testing.assert_close(action.mean_end_effector_acc_l2(), expected)
    run_end_effector_period(action, [sample * 2, sample * 2])
    torch.testing.assert_close(action.mean_end_effector_acc_l2(), torch.full((3,), 4.0))
    action.reset()
    assert not action.mean_end_effector_acc_l2().any()


def test_reading_either_acceleration_reward_does_not_finalize_the_other():
    action, _ = make_action()
    sample = torch.zeros(3, 4, 6)
    run_end_effector_period(action, [sample, sample])
    action.mean_joint_acc_l2()
    assert action._end_effector_acc_mean is None
    action.process_actions(torch.zeros(3, 2))
    for _ in range(2):
        action.apply_actions()
    action.mean_end_effector_acc_l2()
    assert action._joint_acc_mean is None


@pytest.mark.parametrize("weight", [0.0, -1e-5])
def test_end_effector_reward_weight_dt_tag_and_disabled_cache(weight):
    action, _ = make_action()
    env = action._env
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    env.max_episode_length_s = 10.0
    env.sim.is_playing = lambda: True
    spike = torch.zeros(3, 4, 6)
    spike[:, 0, 0] = 1000
    run_end_effector_period(action, [spike, torch.zeros_like(spike)])
    cfg = RewardTermCfg(func=end_effector_relative_lin_acc_l2_substeps, weight=weight)
    manager = RewardManager({"end_effector_acc": cfg}, env)
    expected = torch.full((3,), 500000.0 * weight * env.step_dt)
    torch.testing.assert_close(manager.compute(env.step_dt), expected)
    assert (action._end_effector_acc_mean is None) == (weight == 0)
    torch.testing.assert_close(manager.reset()["Episode_Reward/end_effector_acc"], expected.mean() / 10)
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.mean_end_effector_acc_l2(), torch.tensor([500000.0, 0.0, 500000.0]))
    env.action_manager.get_term = lambda name: object()
    with pytest.raises(TypeError, match="UpperJointPositionAction"):
        end_effector_relative_lin_acc_l2_substeps(env)


@pytest.mark.parametrize(
    "overrides",
    [
        {"acceleration_body_names": ()},
        {"acceleration_body_names": ("left_wrist_yaw_link", "left_wrist_yaw_link")},
        {"acceleration_body_names": ("pelvis",)},
    ],
)
def test_invalid_acceleration_body_selection_is_rejected(overrides):
    with pytest.raises(ValueError, match="distinct"):
        make_action(**overrides)
