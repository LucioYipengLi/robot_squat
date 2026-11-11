"""Phase 3 push progression, episode eligibility and independent quality gates."""

from types import SimpleNamespace

import pytest
import torch

from g1.tasks.lower_limb_agent.config.g1.env_cfg_phase import LowerLimbAgentEnvCfg_Phase3
from g1.tasks.lower_limb_agent.mdp.curriculums import push_velocity_magnitude_curriculum
from g1.tasks.lower_limb_agent.performance import ModePerformanceMonitor


@pytest.fixture
def push_env():
    cfg = LowerLimbAgentEnvCfg_Phase3()
    event = cfg.events.push_robot.copy()
    monitor = ModePerformanceMonitor(5, "cpu", 0.1, (0.03, 0.5, 0.4, 0.02))
    monitor.latest_joint_success.fill_(True)
    monitor.latest_joint_success_valid.fill_(True)
    return SimpleNamespace(
        cfg=cfg,
        event=event,
        monitor=monitor,
        device="cpu",
        num_envs=5,
        max_episode_length=10,
        common_step_counter=0,
        episode_length_buf=torch.zeros(5, dtype=torch.long),
        termination_manager=SimpleNamespace(dones=torch.zeros(5, dtype=torch.bool)),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: event),
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(performance=monitor)),
    )


def _push(env, ids=slice(None), **overrides):
    params = env.cfg.curriculum.push_velocity_magnitude.params | {"update_period": 10} | overrides
    return push_velocity_magnitude_curriculum(env, ids, **params)


def _finish(env, step):
    state = env._g1_curriculum_state["push:push_robot"]
    env.common_step_counter = step
    env.episode_length_buf.copy_(step - state["start_step"])
    env.termination_manager.dones.fill_(True)


def test_waits_full_window_at_large_restored_step_and_does_not_double_count(push_env):
    env = push_env
    env._g1_curriculum_step_offset = 500_000
    assert _push(env)["max_delta_v_mps"] == 0.1
    env.common_step_counter = 9
    assert _push(env, slice(0, 0))["max_delta_v_mps"] == 0.1
    _finish(env, 10)
    result = _push(env)
    assert result["max_delta_v_mps"] == pytest.approx(0.15)
    assert result["survival_ema"] == 1 and result["task_success_rate"] == 1
    assert env.cfg.events.push_robot.params == env.event.params
    assert _push(env)["max_delta_v_mps"] == pytest.approx(0.15)
    assert env._g1_curriculum_state["push:push_robot"]["total_episodes"] == 5


@pytest.mark.parametrize("full_survival,successes,advances", [(True, 5, True), (True, 4, False), (False, 5, False)])
def test_both_strict_quality_gates_are_required(push_env, full_survival, successes, advances):
    env = push_env
    env.max_episode_length = 10 if full_survival else 20
    env.monitor.latest_joint_success[:] = torch.arange(5) < successes
    _push(env)
    _finish(env, 10)
    assert _push(env)["max_delta_v_mps"] == pytest.approx(0.15 if advances else 0.1)


def test_quality_thresholds_are_tunable_without_freezing_ema(push_env):
    env = push_env
    env.max_episode_length = 20
    env.monitor.latest_joint_success[:] = torch.arange(5) < 3
    _push(env)
    _finish(env, 10)
    assert _push(env)["survival_ema"] == 0.5
    assert env.event.params["velocity_range"]["x"] == (-0.1, 0.1)
    _finish(env, 20)
    result = _push(env, survival_gate=0.4, task_success_gate=0.5)
    assert result["max_delta_v_mps"] == pytest.approx(0.15)


def test_empty_manual_partial_and_restore_resets_do_not_advance(push_env):
    env = push_env
    _push(env)
    env.common_step_counter = 5
    env.episode_length_buf[:] = torch.tensor([5, 10, 0, 0, 0])
    env.termination_manager.dones[1] = True
    _push(env, slice(0, 2))  # Manual reset / artificially aged episode.
    env.common_step_counter = 10
    assert _push(env, slice(0, 0))["max_delta_v_mps"] == 0.1
    assert env._g1_curriculum_state["push:push_robot"]["total_episodes"] == 0
    env._g1_restoring_curriculum = True
    env.episode_length_buf.fill_(10)
    env.termination_manager.dones.fill_(True)
    assert _push(env)["max_delta_v_mps"] == 0.1


def test_caps_at_final_and_skipped_intervals_do_not_catch_up(push_env):
    env = push_env
    _push(env)
    _finish(env, 10)
    assert _push(env, velocity_step=0.4)["max_delta_v_mps"] == 0.5
    env._g1_curriculum_state["push:push_robot"]["start_step"].fill_(90)
    _finish(env, 100)
    assert _push(env, velocity_step=0.05)["max_delta_v_mps"] == pytest.approx(0.55)
    assert _push(env, velocity_step=0.4)["max_delta_v_mps"] == pytest.approx(0.55)
    _finish(env, 110)
    assert _push(env, velocity_step=0.4)["max_delta_v_mps"] == 0.7
    _finish(env, 120)
    assert _push(env, velocity_step=0.4)["max_delta_v_mps"] == 0.7


@pytest.mark.parametrize(
    "overrides",
    [
        {"survival_gate": -0.1},
        {"task_success_gate": 1.1},
        {"survival_gate": True},
        {"velocity_step": 0},
        {"velocity_final": float("nan")},
        {"ema_beta": 0},
        {"update_period": 0},
    ],
)
def test_invalid_parameters_fail(push_env, overrides):
    with pytest.raises(ValueError):
        _push(push_env, **overrides)


@pytest.mark.parametrize(
    "bounds",
    [
        {"x": (-0.1, 0.2), "y": (-0.1, 0.1)},
        {"x": (-0.1, 0.1), "y": (-0.2, 0.2)},
        {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0, 0)},
        {"x": (-0.8, 0.8), "y": (-0.8, 0.8)},
    ],
)
def test_unsupported_push_ranges_fail(push_env, bounds):
    push_env.event.params["velocity_range"] = bounds
    with pytest.raises(ValueError):
        _push(push_env)
