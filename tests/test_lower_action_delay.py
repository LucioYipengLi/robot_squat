"""Verify target delivery, independent environment resets, and nominal overrides."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from g1.rl.evaluation.config import load_suite
from g1.rl.evaluation.modules import conditions
from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg
from g1.tasks.lower_limb_agent.config.g1.env_cfg_phase import (
    LowerLimbAgentEnvCfg_Phase2,
    LowerLimbAgentEnvCfg_Phase3,
)
from g1.tasks.lower_limb_agent.evaluation import prepare_env_cfg
from g1.tasks.lower_limb_agent.mdp.actions import LowerBodyPositionAction


def _action(bounds, *, count=3, handles_decimation=False):
    local_pos = torch.full((count, 4), -5.0)
    output = torch.full((count, 2), float("nan"))

    def write(*, target, joint_ids, env_ids=None):
        torch.testing.assert_close(joint_ids, torch.tensor([0, 3]))
        output[slice(None) if env_ids is None else env_ids] = target

    env = SimpleNamespace(
        cfg=SimpleNamespace(lower_action_delay_steps=bounds, decimation=2, seed=17),
        _physics_handles_decimation=handles_decimation,
        upper_body_motion=SimpleNamespace(advance=lambda _: None),
        physics_dt=1 / 120,
        num_envs=count,
        device="cpu",
    )
    action = LowerBodyPositionAction.__new__(LowerBodyPositionAction)
    action._env = env
    action._motion_dt = env.physics_dt
    action._joint_ids = torch.tensor([0, 3])
    action._raw_actions = torch.zeros_like(output)
    action._processed_actions = torch.zeros_like(output)
    action._asset = SimpleNamespace(
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=local_pos)),
        set_joint_position_target_index=write,
    )
    action._delay = action._make_delay(env)
    return action, local_pos, output


def test_fixed_five_substeps_and_partial_reset_do_not_replay_old_episode_targets():
    action, position, output = _action((5, 5))
    reset_ids = torch.tensor([1])
    action.reset()
    assert action.completed_delay_ms(reset_ids).numel() == 0
    for tick in range(8):
        action._processed_actions[:] = 10 + tick // 2
        action.apply_actions()
        expected = -5 if tick < 5 else 10 + (tick - 5) // 2
        torch.testing.assert_close(output, torch.full_like(output, expected))
    assert action.completed_delay_ms(reset_ids).item() == pytest.approx(1000 * 5 / 120)

    position[1, [0, 3]] = torch.tensor([-2.0, -3.0])
    action.reset(reset_ids)
    torch.testing.assert_close(output[1], torch.tensor([-2.0, -3.0]))
    for tick in range(6):
        action._processed_actions[:] = 30 + tick // 2
        action.apply_actions()
        expected_continuing = 10 + (tick + 3) // 2 if tick < 5 else 30
        assert output[0, 0] == output[2, 1] == expected_continuing
        expected_reset = [-2.0, -3.0] if tick < 5 else [30.0, 30.0]
        torch.testing.assert_close(output[1], torch.tensor(expected_reset))


def test_random_lag_stays_fixed_during_rollout_and_draws_only_for_reset_rows():
    action, _, _ = _action((0, 5), count=128)
    replica, _, _ = _action((0, 5), count=128)
    global_rng = torch.get_rng_state()
    for term in (action, replica):
        term.reset()
    before = action._delay.time_lags.clone()
    assert set(before.tolist()) == set(range(6))
    torch.testing.assert_close(before, replica._delay.time_lags)
    for _ in range(8):
        action.apply_actions()
    torch.testing.assert_close(action._delay.time_lags, before)
    ids = torch.arange(64)
    action.reset(ids)
    replica.reset(ids)
    torch.testing.assert_close(action._delay.time_lags, replica._delay.time_lags)
    torch.testing.assert_close(action._delay.time_lags[64:], before[64:])
    assert (action._delay.time_lags[:64] != before[:64]).any()
    assert torch.equal(global_rng, torch.get_rng_state())


def test_zero_delay_keeps_original_dispatch_and_backend_compatibility():
    action, _, output = _action((0, 0), handles_decimation=True)
    assert action._delay is None
    action.reset()
    assert output.isnan().all()
    action._processed_actions.fill_(0.2)
    action.apply_actions()
    torch.testing.assert_close(output, torch.full_like(output, 0.2))
    with pytest.raises(ValueError, match="every physics substep"):
        _action((0, 5), handles_decimation=True)
    with pytest.raises(ValueError, match="ordered non-negative integers"):
        _action((0, 1.5))


def test_training_phase_defaults_and_evaluation_override():
    assert LowerLimbAgentEnvCfg().lower_action_delay_steps == (0, 0)
    assert LowerLimbAgentEnvCfg_Phase2().lower_action_delay_steps == (0, 5)
    play_cfg = LowerLimbAgentEnvCfg_Phase3()
    assert play_cfg.lower_action_delay_steps == (0, 5)
    play_cfg.play_mode()
    assert play_cfg.lower_action_delay_steps == (0, 0)

    suite = load_suite("g1_lower_delay")
    jobs = conditions(suite)
    assert [job["axis_step"] for job in jobs] == list(range(6))
    assert all(job["arm_fraction"] == job["waist_fraction"] == 0 for job in jobs)
    for condition, expected in (({}, (0, 0)), (jobs[-1], (5, 5))):
        cfg = prepare_env_cfg(LowerLimbAgentEnvCfg_Phase3(), suite["benchmark"], condition, 4, 17)
        assert cfg.lower_action_delay_steps == expected


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_LOWER_DELAY_PHYSICS_SMOKE") != "1", reason="Opt-in: starts Isaac Sim")
def test_real_physx_substeps_receive_delayed_targets_and_reset_isolated():
    run = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_lower_action_delay_physics.py"))],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    assert run.returncode == 0 and "LOWER_ACTION_DELAY_PHYSX_VERIFIED" in run.stdout, run.stdout + run.stderr
