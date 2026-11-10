"""Normalized velocity/height decoding, shared change costs and command ordering."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ActionManager, ActionTermCfg, RewardManager, RewardTermCfg

from g1.tasks.upper_limb_agent.mdp.actions import UpperJointPositionAction
from g1.tasks.upper_limb_agent.mdp.frozen_lower_actions import (
    FrozenLowerAction,
    FrozenLowerActionCfg,
    LowerHeightCommandAction,
    LowerHeightCommandActionCfg,
    LowerVelocityCommandAction,
    LowerVelocityCommandActionCfg,
)
from g1.tasks.upper_limb_agent.mdp.observations import lower_height_offset, lower_velocity_command, term_action
from g1.tasks.upper_limb_agent.mdp.rewards import (
    action_term_rate_l2,
    lower_command_magnitude_l2,
    lower_command_overflow_l2,
    lower_command_rate_l2,
)


def make_command(**kwargs):
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": object()})
    return LowerVelocityCommandAction(LowerVelocityCommandActionCfg(asset_name="robot", **kwargs), env)


def make_height(**kwargs):
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": object()})
    return LowerHeightCommandAction(LowerHeightCommandActionCfg(asset_name="robot", **kwargs), env)


def test_height_zero_asymmetric_bounds_clipping_and_partial_reset():
    term = make_height()
    assert term.action_dim == 1 and not term.processed_actions.any()
    values = torch.tensor([[-2.0], [0.0], [2.0]])
    term.process_actions(values)
    torch.testing.assert_close(term.processed_actions, torch.tensor([[-0.15], [0.0], [0.02]]))
    torch.testing.assert_close(term.raw_actions, values)
    term.process_actions(torch.tensor([[-0.5], [0.5], [1.0]]))
    torch.testing.assert_close(term.processed_actions, torch.tensor([[-0.075], [0.01], [0.02]]))
    term.apply_actions()
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.processed_actions, torch.tensor([[-0.075], [0.0], [0.02]]))
    assert not term.raw_actions[1].any()
    term.cfg.enabled = False
    term.process_actions(values)
    assert not term.processed_actions.any()
    torch.testing.assert_close(term.raw_actions, values)
    term.reset()
    assert not term.raw_actions.any()


@pytest.mark.parametrize("bounds", [(-0.15,), (0.01, 0.1), (-0.2, -0.1), (0, 0), (-0.1, float("nan"))])
def test_height_rejects_invalid_or_nonzero_centered_ranges(bounds):
    with pytest.raises(ValueError, match="contain zero"):
        make_height(height_offset_range=bounds)


def test_height_change_uses_effective_normalization_and_preserves_history_on_reads_and_partial_reset():
    term = make_height()
    term.process_actions(torch.tensor([[-2.0], [0.5], [2.0]]))
    torch.testing.assert_close(term.change_l2, torch.tensor([1.0, 0.25, 1.0]))
    # Overflow changes cannot change the clipped target or its difference cost.
    term.process_actions(torch.tensor([[-8.0], [0.5], [8.0]]))
    assert not term.change_l2.any()
    term.process_actions(torch.tensor([[1.0], [-0.5], [-1.0]]))
    torch.testing.assert_close(term.change_l2, torch.tensor([4.0, 1.0, 4.0]))
    term.apply_actions()
    term.apply_actions()
    torch.testing.assert_close(term.change_l2, torch.tensor([4.0, 1.0, 4.0]))
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.change_l2, torch.tensor([4.0, 0.0, 4.0]))
    term.process_actions(torch.tensor([[1.0], [-0.5], [-1.0]]))
    torch.testing.assert_close(term.change_l2, torch.tensor([0.0, 0.25, 0.0]))
    term.cfg.enabled = False
    term.process_actions(torch.ones(3, 1))
    assert not term.processed_actions.any() and not term.change_l2.any()
    term.cfg.enabled = True
    term.process_actions(torch.ones(3, 1))
    torch.testing.assert_close(term.change_l2, torch.ones(3))
    term.reset()
    assert not term.raw_actions.any() and not term.processed_actions.any() and not term.change_l2.any()


@pytest.mark.parametrize("bounds,blocked", [((0.0, 0.02), -1), ((-0.15, 0.0), 1)])
def test_height_change_ignores_the_zero_range_side(bounds, blocked):
    term = make_height(height_offset_range=bounds)
    term.process_actions(torch.full((3, 1), 2.0 * blocked))
    assert not term.processed_actions.any() and not term.change_l2.any()
    term.process_actions(torch.full((3, 1), 8.0 * blocked))
    assert not term.change_l2.any()
    term.process_actions(torch.full((3, 1), -0.5 * blocked))
    torch.testing.assert_close(term.change_l2, torch.full((3,), 0.25))
    term.process_actions(torch.full((3, 1), 8.0 * blocked))
    torch.testing.assert_close(term.change_l2, torch.full((3,), 0.25))
    assert not term.processed_actions.any()


def test_zero_axis_scaling_clipping_and_no_physics_writes():
    term = make_command()
    assert term.action_dim == 3 and not term.processed_actions.any()
    values = torch.tensor([[1.0, 0, 0], [0, -2.0, 0], [0, 0, 0.5]])
    saved = values.clone()
    term.process_actions(values)
    torch.testing.assert_close(term.processed_actions, torch.tensor([[0.3, 0, 0], [0, -0.1, 0], [0, 0, 0.15]]))
    torch.testing.assert_close(term.raw_actions, values)
    torch.testing.assert_close(values, saved)
    term.apply_actions()  # The asset has no joint writers; this term writes none.
    torch.testing.assert_close(term.change_l2, torch.tensor([1.0, 1.0, 0.25]))


def test_change_uses_clipped_normalized_values_and_stays_read_only_between_steps():
    term = make_command(command_scale=(0.6, 0.2, 0.8))
    term.process_actions(torch.tensor([[2.0, 0, 0]]).expand(3, -1))
    term.process_actions(torch.tensor([[3.0, 0, 0]]).expand(3, -1))
    assert not term.change_l2.any()  # Different raw outputs, same effective command.
    term.process_actions(torch.tensor([[-1.0, 0, 0]]).expand(3, -1))
    torch.testing.assert_close(term.change_l2, torch.full((3,), 4.0))
    before = term.processed_actions.clone()
    term.apply_actions()
    term.apply_actions()
    torch.testing.assert_close(term.change_l2, torch.full((3,), 4.0))
    torch.testing.assert_close(term.processed_actions, before)


def test_force_zero_keeps_interface_and_partial_reset_preserves_ongoing_costs():
    term = make_command()
    term.process_actions(torch.ones(3, 3))
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.change_l2, torch.tensor([3.0, 0, 3.0]))
    assert not term.raw_actions[1].any() and not term.processed_actions[1].any()
    term.process_actions(torch.ones(3, 3))
    torch.testing.assert_close(term.change_l2, torch.tensor([0.0, 3.0, 0.0]))
    term.reset()
    term.cfg.enabled = False
    term.process_actions(torch.full((3, 3), 5.0))
    assert term.action_dim == 3 and not term.processed_actions.any() and not term.change_l2.any()
    assert (term.raw_actions == 5).all()


@pytest.mark.parametrize("scale", [(0.0, 0.1, 0.3), (-0.3, 0.1, 0.3), (0.3, float("nan"), 0.3), (1, 2)])
def test_invalid_scales_fail_before_asset_initialization(scale):
    with pytest.raises(ValueError, match="three positive finite"):
        make_command(command_scale=scale)


def test_command_costs_separate_raw_overflow_from_effective_magnitude():
    term = make_command(command_scale=(0.6, 0.2, 0.8))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))
    values = torch.tensor([[0.5, -1.0, 0.0], [2.0, -3.0, 0.5], [10.0, -10.0, 1.0]])
    term.process_actions(values)
    torch.testing.assert_close(lower_command_overflow_l2(env), torch.tensor([0.0, 5.0, 162.0]))
    torch.testing.assert_close(lower_command_magnitude_l2(env), torch.tensor([1.25, 2.25, 3.0]))
    saved = [value.clone() for value in (term.raw_actions, term.processed_actions, term.change_l2)]
    lower_command_overflow_l2(env)
    lower_command_magnitude_l2(env)
    for value, before in zip((term.raw_actions, term.processed_actions, term.change_l2), saved, strict=True):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    # Holding the same command eliminates only the difference cost.
    term.process_actions(values)
    assert not term.change_l2.any()
    assert (lower_command_magnitude_l2(env) > 0).all()
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(lower_command_overflow_l2(env), torch.tensor([0.0, 0.0, 162.0]))
    torch.testing.assert_close(lower_command_magnitude_l2(env), torch.tensor([1.25, 0.0, 3.0]))
    term.cfg.enabled = False
    term.process_actions(values)
    assert not lower_command_overflow_l2(env).any() and not lower_command_magnitude_l2(env).any()


@pytest.mark.parametrize("reward", [lower_command_overflow_l2, lower_command_magnitude_l2])
def test_command_costs_require_the_velocity_action_type(reward):
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: object()))
    with pytest.raises(TypeError, match="LowerVelocityCommandAction"):
        reward(env)


@pytest.mark.parametrize("missing", [False, True])
def test_reversed_or_missing_command_fails_before_loading_model(missing):
    joint = FrozenLowerActionCfg(asset_name="robot", joint_names=["joint"])
    configured = {"joint_pos": joint}
    if not missing:
        configured["lower_command"] = LowerVelocityCommandActionCfg(asset_name="robot")
    env = SimpleNamespace(cfg=SimpleNamespace(actions=configured))
    with pytest.raises(ValueError, match="before joint_pos"):
        FrozenLowerAction(joint.copy(), env)  # ActionManager deep-copies configs.


@pytest.mark.parametrize("missing", [False, True])
def test_reversed_or_missing_height_fails_before_loading_model(missing):
    joint = FrozenLowerActionCfg(asset_name="robot", joint_names=["joint"])
    configured = {"lower_command": LowerVelocityCommandActionCfg(asset_name="robot"), "joint_pos": joint}
    if not missing:
        configured["height_offset"] = LowerHeightCommandActionCfg(asset_name="robot")
    env = SimpleNamespace(cfg=SimpleNamespace(actions=configured))
    with pytest.raises(ValueError, match="before joint_pos"):
        FrozenLowerAction(joint.copy(), env)


def make_bridge(monkeypatch):
    """Exercise the actual manager and bridge process method without loading an asset."""
    data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0, 0.8]]).expand(3, -1)),
        default_root_pose=SimpleNamespace(torch=torch.tensor([[0.0, 0, 0.8, 0, 0, 0, 1]]).expand(3, -1)),
        root_lin_vel_b=SimpleNamespace(torch=torch.zeros(3, 3)),
        root_ang_vel_b=SimpleNamespace(torch=torch.zeros(3, 3)),
        projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.0, 0, -1.0]]).expand(3, -1)),
    )
    scene = {"robot": SimpleNamespace(data=data)}

    class Scene(dict):
        env_origins = torch.zeros(3, 3)

    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene=Scene(scene),
        sim=SimpleNamespace(is_playing=lambda: True),
        episode_length_buf=torch.ones(3),
        max_episode_length_s=10,
    )
    bridge = object.__new__(FrozenLowerAction)
    bridge._env = env
    bridge.cfg = FrozenLowerActionCfg(asset_name="robot", joint_names=["joint"])
    bridge._asset = scene["robot"]
    bridge._num_joints = 17
    bridge._raw_actions = torch.zeros(3, 17)
    bridge._processed_actions = torch.zeros(3, 17)
    bridge.lower_command = torch.zeros(3, 6)
    bridge.lower_actions = torch.zeros(3, 12)
    bridge.last_lower_observations = torch.zeros(3, 650)
    bridge._motion_sum, bridge._motion_steps = torch.zeros(3, 9), torch.zeros(3)
    bridge._base_motion_sum, bridge._base_motion_mean = torch.zeros(3, 4), None
    bridge._base_vertical_acc_samples = torch.zeros(3, 2)
    calls = []

    def upper_process(term, values):
        term._raw_actions.copy_(values)
        term._processed_actions.copy_(values * 0.5)

    monkeypatch.setattr(UpperJointPositionAction, "process_actions", upper_process)

    def compute(group, update_history):
        assert group == "policy" and update_history
        return torch.cat((bridge.lower_command, bridge.processed_actions, torch.zeros(3, 627)), -1)

    def infer(observations):
        calls.append(observations.clone())
        return observations[:, :1].expand(-1, 12).clone()

    bridge.lower_observations = SimpleNamespace(compute_group=compute)
    bridge.frozen_policy = infer
    bridge.lower_action = SimpleNamespace(process_actions=lambda values: None)
    manager_cfg = {
        "lower_command": LowerVelocityCommandActionCfg(asset_name="robot"),
        "height_offset": LowerHeightCommandActionCfg(asset_name="robot"),
        "joint_pos": ActionTermCfg(asset_name="robot", class_type=lambda cfg, env: bridge),
    }
    env.action_manager = ActionManager(SimpleNamespace(**manager_cfg), env)
    return env, bridge, calls


def test_manager_publishes_current_commands_and_upper_preview_before_one_lower_inference(monkeypatch):
    env, bridge, calls = make_bridge(monkeypatch)
    values = torch.zeros(3, 21)
    values[:, :3] = torch.tensor([0.5, -1.0, 1.0])
    values[:, 3] = -0.5
    values[:, 4:] = 0.2
    env.action_manager.process_action(values)
    assert len(calls) == 1
    expected = torch.tensor([-0.075, 0.15, -0.1, 0.3, 0, 0]).expand(3, -1)
    torch.testing.assert_close(calls[0][:, :6], expected)
    torch.testing.assert_close(calls[0][:, 6:23], torch.full((3, 17), 0.1))
    torch.testing.assert_close(bridge.last_lower_observations, calls[0])
    torch.testing.assert_close(term_action(env, "joint_pos"), values[:, 4:])
    torch.testing.assert_close(lower_velocity_command(env), expected[:, 1:4])
    torch.testing.assert_close(lower_height_offset(env), expected[:, :1])
    # The state at this boundary was produced by the previous zero command.
    assert not bridge._motion_sum[:, 3:7].any()
    env.scene["robot"].data.root_lin_vel_b.torch[:, :2] = expected[:, 1:3]
    env.scene["robot"].data.root_ang_vel_b.torch[:, 2] = expected[:, 3]
    env.action_manager.process_action(torch.zeros_like(values))
    assert len(calls) == 2 and not calls[-1][:, :23].any()
    assert (bridge._motion_sum[:, 3:5] > 0).all()
    assert not bridge._motion_sum[:, 5:7].any()  # Response paired with the command that produced it.


def test_named_upper_cost_excludes_velocity_and_lower_cost_gets_weight_dt_once(monkeypatch):
    env, _, _ = make_bridge(monkeypatch)
    first = torch.zeros(3, 21)
    first[:, :3], first[:, 4:] = 1, 0.1
    first[:, 3] = -0.5
    env.action_manager.process_action(first)
    torch.testing.assert_close(action_term_rate_l2(env, "joint_pos"), torch.full((3,), 17 * 0.01))
    torch.testing.assert_close(lower_command_rate_l2(env), torch.full((3,), 3.25))
    rewards = RewardManager({"rate": RewardTermCfg(func=lower_command_rate_l2, weight=-0.01)}, env)
    torch.testing.assert_close(rewards.compute(1 / 60), torch.full((3,), -0.01 * 3.25 / 60))
    # Repeated reads preserve history; changing only Lower leaves the Upper cost zero.
    rewards.compute(1 / 60)
    second = first.clone()
    second[:, :3] = -1
    second[:, 3] = 0.5
    env.action_manager.process_action(second)
    assert not action_term_rate_l2(env, "joint_pos").any()
    torch.testing.assert_close(lower_command_rate_l2(env), torch.full((3,), 13.0))
    with pytest.raises(ValueError, match="Unknown action term"):
        action_term_rate_l2(env, "missing")


def test_height_changes_only_affect_shared_rate_and_keep_velocity_amplitude_and_overflow_costs(monkeypatch):
    env, _, calls = make_bridge(monkeypatch)
    values = torch.zeros(3, 21)
    values[:, :3] = torch.tensor([2.0, -3.0, 0.5])
    values[:, 3] = 8.0
    env.action_manager.process_action(values)
    values[:, 3] = -8.0
    env.action_manager.process_action(values)
    torch.testing.assert_close(calls[-1][:, 0], torch.full((3,), -0.15))
    torch.testing.assert_close(lower_command_rate_l2(env), torch.full((3,), 4.0))
    assert not action_term_rate_l2(env, "joint_pos").any()
    torch.testing.assert_close(lower_command_overflow_l2(env), torch.full((3,), 5.0))
    torch.testing.assert_close(lower_command_magnitude_l2(env), torch.full((3,), 2.25))
    env.action_manager.process_action(values)
    assert not lower_command_rate_l2(env).any()  # Holding any nonzero height is free.


@pytest.mark.parametrize(
    "velocity_enabled,height_enabled,expected", [(True, True, 4), (False, True, 1), (True, False, 3), (False, False, 0)]
)
def test_shared_rate_handles_independently_disabled_interfaces_and_partial_resets(
    monkeypatch, velocity_enabled, height_enabled, expected
):
    env, bridge, _ = make_bridge(monkeypatch)
    # This fixture stubs joint processing without physical substep state. Keep
    # its reset inert while exercising real manager/command reset semantics.
    bridge.reset = lambda env_ids=None: None
    env.action_manager.get_term("lower_command").cfg.enabled = velocity_enabled
    env.action_manager.get_term("height_offset").cfg.enabled = height_enabled
    values = torch.zeros(3, 21)
    values[:, :4] = 2.0
    env.action_manager.process_action(values)
    torch.testing.assert_close(lower_command_rate_l2(env), torch.full((3,), float(expected)))
    env.action_manager.reset(torch.tensor([1]))
    torch.testing.assert_close(lower_command_rate_l2(env), torch.tensor([expected, 0, expected], dtype=torch.float))
    env.action_manager.process_action(values)
    torch.testing.assert_close(lower_command_rate_l2(env), torch.tensor([0, expected, 0], dtype=torch.float))


def test_shared_rate_accepts_named_terms_and_rejects_wrong_height_type():
    velocity, height = make_command(), make_height()
    velocity.process_actions(torch.tensor([[0.5, 0, 0]]).expand(3, -1))
    height.process_actions(torch.full((3, 1), -0.5))
    terms = {"velocity": velocity, "height": height}
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=terms.__getitem__))
    torch.testing.assert_close(lower_command_rate_l2(env, "velocity", "height"), torch.full((3,), 0.5))
    terms["height"] = object()
    with pytest.raises(TypeError, match="LowerHeightCommandAction"):
        lower_command_rate_l2(env, "velocity", "height")


def test_height_tracking_metric_pairs_physical_height_with_previous_target(monkeypatch):
    env, bridge, _ = make_bridge(monkeypatch)
    values = torch.zeros(3, 21)
    values[:, 3] = -1.0
    env.action_manager.process_action(values)
    assert not bridge._motion_sum[:, 0].any()
    data = env.scene["robot"].data
    data.root_pos_w.torch = data.root_pos_w.torch.clone()
    data.root_pos_w.torch[:, 2] = 0.65  # Default 0.8 plus previous -0.15 command.
    env.action_manager.process_action(torch.zeros_like(values))
    torch.testing.assert_close(bridge._motion_sum[:, 0], torch.zeros(3), atol=1e-7, rtol=0)
    torch.testing.assert_close(bridge._motion_sum[:, 8], torch.full((3,), -0.15))


def test_command_penalties_use_named_term_and_reward_manager_weight_dt_once(monkeypatch):
    env, _, _ = make_bridge(monkeypatch)
    values = torch.full((3, 21), 100.0)
    values[:, :3] = torch.tensor([2.0, -3.0, 0.5])
    env.action_manager.process_action(values)
    cfg = {
        "overflow": RewardTermCfg(func=lower_command_overflow_l2, weight=-1e-4),
        "magnitude": RewardTermCfg(func=lower_command_magnitude_l2, weight=-0.01),
    }
    rewards = RewardManager(cfg, env)
    expected = torch.full((3,), (-1e-4 * 5 - 0.01 * 2.25) / 60)
    torch.testing.assert_close(rewards.compute(1 / 60), expected)
    torch.testing.assert_close(rewards.reset()["Episode_Reward/overflow"], torch.tensor(-1e-4 * 5 / 60 / 10))


def test_upper_action_observation_uses_reset_manager_history_not_retained_term_buffers(monkeypatch):
    env, bridge, _ = make_bridge(monkeypatch)
    env.action_manager.process_action(torch.full((3, 21), 0.5))
    # Custom Upper reset retains its private raw buffer. ActionManager owns and
    # clears the authoritative action history used by observations and rewards.
    bridge.reset = lambda env_ids: None
    env.action_manager.reset(torch.tensor([1]))
    assert bridge.raw_actions[1].any()
    assert not term_action(env, "joint_pos")[1].any()
    torch.testing.assert_close(term_action(env, "joint_pos")[[0, 2]], torch.full((2, 17), 0.5))
