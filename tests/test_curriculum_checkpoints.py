# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise real serialization and upstream periodic saves without a physics runtime."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from g1.rl.runner import CurriculumOnPolicyRunner
from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg
from g1.tasks.lower_limb_agent.config.g1.env_cfg_phase import LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3
from g1.tasks.lower_limb_agent.curriculum_state import (
    COMMAND_NAME,
    EVENT_NAME,
    STATE_KEY,
    export_curriculum_state,
    import_curriculum_state,
)
from g1.tasks.lower_limb_agent.mdp.curriculums import curriculum_step, upper_body_disturbance_magnitude_curriculum


class _Environment:
    def __init__(self, phase=1, num_envs=2, evaluation=False):
        self.cfg = (LowerLimbAgentEnvCfg, LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3)[phase - 1]()
        if evaluation:
            self.cfg.play_mode()
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = self.cfg.sim.dt * self.cfg.decimation
        self.max_episode_length = round(self.cfg.episode_length_s / self.step_dt)
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.termination_manager = SimpleNamespace(dones=torch.zeros(num_envs, dtype=torch.bool))
        self.term = SimpleNamespace(
            cfg=copy.deepcopy(self.cfg.commands.task_command),
            _mode_probs=torch.tensor(self.cfg.commands.task_command.rel_mode_envs),
            last_mode=torch.zeros(num_envs, dtype=torch.long),
            MODE_WALK=2,
            MODE_SQUAT_WALK=3,
            _last_metric_step=torch.full((num_envs,), -1, dtype=torch.long),
        )
        for name in (
            "_error_height_sum",
            "_error_xy_sum",
            "_error_yaw_sum",
            "_step_count",
            "_pelvis_error_sum",
            "_pelvis_success_sum",
            "_pelvis_step_count",
            "episode_success_rate",
            "episode_success_rate_vel",
        ):
            setattr(self.term, name, torch.zeros(num_envs))
        self.term.episode_metrics_valid = torch.zeros(num_envs, dtype=torch.bool)
        self.command_manager = SimpleNamespace(get_term=lambda name: self.term)
        self.event = copy.deepcopy(self.cfg.events.upper_body_disturbance)
        self.event_manager = SimpleNamespace(get_term_cfg=lambda name: self.event)
        self.resets = []
        self.reset()

    def reset(self):
        if self.cfg.curriculum is not None:
            for entry in vars(self.cfg.curriculum).values():
                if entry is not None:
                    entry.func(self, slice(None), **entry.params)
        self.resets.append(
            {
                "ranges": copy.deepcopy(vars(self.term.cfg.ranges)),
                "weights": self.term._mode_probs.clone(),
                "modes": torch.multinomial(self.term._mode_probs.expand(self.num_envs, -1), 1),
            }
        )
        self.episode_length_buf.zero_()
        return torch.zeros(self.num_envs, 3), {}


def _trained(phase=1):
    env = _Environment(phase)
    env.common_step_counter = 500_000
    env.term.cfg.ranges.height_offset = (-0.31, 0.04)
    env.term.cfg.ranges.lin_vel_x = (-0.42, 0.83)
    env.term._mode_probs[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    env.event.params["disturb_groups"][0]["joint_noise_frac"] = 0.43
    env._g1_curriculum_state[f"mode_intro:{COMMAND_NAME}"] = {"introduced": True}
    env._g1_curriculum_state[f"command_range:{COMMAND_NAME}"]["last_update_step"] = 499_900
    survival = env._g1_curriculum_state[f"survival:{EVENT_NAME}:0"]
    survival.update(
        ema=torch.tensor(0.91),
        initialized=True,
        length_sum=torch.tensor(1.5),
        episode_count=torch.tensor(2),
        total_episodes=torch.tensor(99),
        last_update_step=499_800,
    )
    return env


def test_same_phase_roundtrip_is_portable_and_starts_fresh_episodes():
    source = _trained()
    saved = export_curriculum_state(source)
    assert json.loads(json.dumps(saved)) == saved
    target = _Environment(num_envs=5)
    target.term._error_height_sum.fill_(100)
    import_curriculum_state(target, saved)
    assert curriculum_step(target) == 500_000
    assert target.common_step_counter == 0
    assert target.resets[-1]["ranges"]["height_offset"] == (-0.31, 0.04)
    assert (target.resets[-1]["modes"] == 3).all()
    assert target.event.params["disturb_groups"][0]["joint_noise_frac"] == 0.43
    assert not target.term.episode_metrics_valid.any()
    assert not target.term._error_height_sum.any()
    survival = target._g1_curriculum_state[f"survival:{EVENT_NAME}:0"]
    assert survival["observed"].shape == (5,)
    assert survival["observed"].all()
    assert int(survival["episode_count"]) == 2
    assert survival["ema"] == pytest.approx(0.91)
    assert export_curriculum_state(target) == saved
    source.term.cfg.ranges.height_offset = (-0.4, 0.05)
    assert saved["distribution"]["ranges"]["height_offset"] == [-0.31, 0.04]


@pytest.mark.parametrize("phase", [2, 3])
def test_phase_transfer_inherits_distribution_but_rebuilds_monitoring(phase):
    source = _trained(phase - 1)
    target = _Environment(phase)
    import_curriculum_state(target, export_curriculum_state(source))
    assert target.term.cfg.ranges.height_offset == (-0.31, 0.04)
    assert (target.resets[-1]["modes"] == 3).all()
    assert target.cfg.events.actuator_gains is not None
    if phase == 3:
        assert target.cfg.events.push_robot is not None
    survival = target._g1_curriculum_state[f"survival:{EVENT_NAME}:0"]
    assert not survival["initialized"]
    assert survival["observed"].all()
    assert survival["total_episodes"] == 0
    assert survival["last_update_step"] == 500_000
    # The first fully observed new episode is counted, without inheriting old DR statistics.
    target.common_step_counter = 100
    target.episode_length_buf.fill_(100)
    target.termination_manager.dones.fill_(True)
    result = upper_body_disturbance_magnitude_curriculum(target, slice(None), update_period=2000)
    assert result["survival_episodes"] == target.num_envs


def test_transfer_before_mode_introduction_keeps_the_future_trigger():
    source = _Environment()
    source.common_step_counter = 1000
    target = _Environment(2)
    import_curriculum_state(target, export_curriculum_state(source))
    assert target.term._mode_probs[3] == 0
    target.common_step_counter = 399_000
    target.reset()
    torch.testing.assert_close(target.term._mode_probs, torch.full((4,), 0.25))


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_evaluation_recovers_ranges_and_modes_without_enabling_disturbances(phase):
    target = _Environment(phase, evaluation=True)
    saved = export_curriculum_state(_trained(phase))
    import_curriculum_state(target, saved, evaluation=True)
    assert target.cfg.curriculum is None
    assert target.cfg.events.upper_body_disturbance is None
    assert target.resets[-1]["ranges"]["height_offset"] == (-0.31, 0.04)
    assert (target.resets[-1]["modes"] == 3).all()
    target.common_step_counter += 1000
    target.reset()
    assert target.term.cfg.ranges.height_offset == (-0.31, 0.04)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda state: state.update(schema_version=99),
        lambda state: state.update(control_dt=0.1),
        lambda state: state["compatibility"].update(policy_history=1),
        lambda state: state["compatibility"]["disturbance_groups"].reverse(),
        lambda state: state["distribution"]["ranges"].update(height_offset=[float("nan"), 0]),
        lambda state: state["distribution"]["ranges"].update(height_offset=[0, 0.1]),
        lambda state: state["distribution"]["ranges"].update(pelvis_offset_x=[0.01, 0.02]),
        lambda state: state["distribution"].update(disturbance_fractions=[0.4]),
        lambda state: state["distribution"].update(mode_weights=[0, 0, 0, 0]),
        lambda state: state["progress"].update(mode_introduced=False),
        lambda state: state["runtime"]["checks"].update(unknown=0),
        lambda state: state["runtime"]["survival"].update(length_sum=3),
    ],
)
def test_invalid_records_fail_before_mutating_the_environment(corrupt):
    target = _Environment()
    saved = export_curriculum_state(_trained())
    corrupt(saved)
    before = copy.deepcopy(vars(target.term.cfg.ranges))
    with pytest.raises(ValueError):
        import_curriculum_state(target, saved)
    assert vars(target.term.cfg.ranges) == before
    assert not getattr(target, "_g1_curriculum_loaded", False)


def test_phase_skips_and_wrong_phase_evaluation_are_rejected():
    saved = export_curriculum_state(_trained())
    with pytest.raises(ValueError, match="Phase"):
        import_curriculum_state(_Environment(3), saved)
    with pytest.raises(ValueError, match="Phase"):
        import_curriculum_state(_Environment(2, evaluation=True), saved, evaluation=True)


class _Algorithm:
    def __init__(self):
        self.loads = 0
        self.learning_rate = 0.0003

    def save(self):
        return {"model": torch.tensor([42.0])}

    def load(self, state, load_cfg, strict):
        self.loads += 1
        assert state["model"].item() == 42
        return True

    def train_mode(self):
        pass

    def act(self, obs):
        return torch.zeros(obs.shape[0], 1)

    def process_env_step(self, *args):
        pass

    def compute_returns(self, obs):
        pass

    def update(self):
        return {}

    def get_policy(self):
        return SimpleNamespace(output_std=torch.ones(1))


def _runner(env, path, *, evaluation=False):
    runner = CurriculumOnPolicyRunner.__new__(CurriculumOnPolicyRunner)

    def step(actions):
        env.common_step_counter += 1
        env.episode_length_buf += 1
        return torch.zeros(env.num_envs, 3), torch.zeros(env.num_envs), torch.zeros(env.num_envs), {}

    runner.env = SimpleNamespace(
        unwrapped=env, device="cpu", get_observations=lambda: torch.zeros(env.num_envs, 3), step=step
    )
    runner.alg = _Algorithm()
    runner.cfg = {"num_steps_per_env": 1, "algorithm": {}, "save_interval": 1, "check_for_nan": False}
    runner.current_learning_iteration = 0
    runner.device = "cpu"
    runner.is_distributed = False
    runner._evaluation = evaluation
    runner._snapshot_path = None
    runner._curriculum_loaded = False

    def uploaded(checkpoint, iteration):
        assert STATE_KEY in torch.load(checkpoint, weights_only=False)["infos"]

    runner.logger = SimpleNamespace(
        log_dir=str(path),
        writer=True,
        save_model=uploaded,
        init_logging_writer=lambda: None,
        process_env_step=lambda *args: None,
        log=lambda **kwargs: None,
        stop_logging_writer=lambda: None,
    )
    return runner


def test_real_runner_periodic_and_final_saves_include_curriculum(tmp_path):
    runner = _runner(_Environment(), tmp_path)
    runner.learn(3)
    for iteration in range(3):
        saved = torch.load(tmp_path / f"model_{iteration}.pt", weights_only=False)
        assert saved["infos"][STATE_KEY]["progress"]["control_steps"] == iteration + 1


def test_runner_save_load_preserves_other_infos_and_restores_before_next_rollout(tmp_path):
    path = tmp_path / "model.pt"
    source = _runner(_trained(), tmp_path)
    source.save(path, infos={"other": {"value": 7}})
    target = _runner(_Environment(2), tmp_path)
    result = target.load(path)
    assert result["other"] == {"value": 7}
    assert target.alg.loads == 1
    assert target.env.unwrapped.term.cfg.ranges.height_offset == (-0.31, 0.04)
    assert target._curriculum_loaded
    assert not source._curriculum_loaded


def test_legacy_checkpoint_requires_explicit_snapshot(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"model": torch.tensor([42.0]), "iter": 10, "infos": None}, path)
    runner = _runner(_Environment(2), tmp_path)
    with pytest.raises(ValueError, match="no G1 curriculum"):
        runner.load(path)
    assert runner.alg.loads == 0
    snapshot = tmp_path / "curriculum.json"
    snapshot.write_text(json.dumps(export_curriculum_state(_trained())), encoding="utf-8")
    runner._snapshot_path = snapshot
    runner.load(path)
    assert runner._curriculum_loaded


def test_later_phases_cannot_start_learning_without_restore(tmp_path):
    runner = _runner(_Environment(2), tmp_path)
    with pytest.raises(ValueError, match="requires --checkpoint"):
        runner.learn(1)
