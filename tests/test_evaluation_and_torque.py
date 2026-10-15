# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation configuration and fixed nominal-stiffness reward regressions."""

import copy
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg
from g1.tasks.lower_limb_agent.config.g1.env_cfg_phase import (
    LowerLimbAgentEnvCfg_Phase2,
    LowerLimbAgentEnvCfg_Phase3,
)
from g1.tasks.lower_limb_agent.mdp.rewards import joint_torques_l2_normalized


@pytest.mark.parametrize(
    "config_type", [LowerLimbAgentEnvCfg, LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3]
)
def test_play_mode_disables_randomization_without_removing_physical_resets(config_type):
    cfg = config_type()
    training_cfg = config_type()
    ranges = copy.deepcopy(vars(cfg.commands.task_command.ranges))
    weights = cfg.commands.task_command.rel_mode_envs
    cfg.scene.num_envs = 2
    cfg.play_mode()
    cfg.play_mode()  # Repeated evaluation setup must be idempotent.
    assert cfg.scene.num_envs == 2
    assert not cfg.observations.policy.enable_corruption
    assert cfg.curriculum is None
    assert vars(cfg.commands.task_command.ranges) == ranges
    assert cfg.commands.task_command.rel_mode_envs == weights
    assert cfg.terminations.time_out is not None
    for event in vars(cfg.events).values():
        if event is not None:
            assert event.mode == "reset"
    assert cfg.events.reset_base.params["pose_range"] == {}
    assert cfg.events.reset_base.params["velocity_range"] == {}
    assert cfg.events.reset_robot_joints.params["position_range"] == (0.0, 0.0)
    assert cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    # Config mutation must not leak into independently created training configurations.
    assert training_cfg.curriculum is not None
    assert training_cfg.events.physics_material is not None
    assert training_cfg.events.upper_body_disturbance is not None
    assert training_cfg.events.reset_base.params["velocity_range"]["x"] == (-0.5, 0.5)
    if config_type is not LowerLimbAgentEnvCfg:
        assert training_cfg.events.actuator_gains is not None
    if config_type is LowerLimbAgentEnvCfg_Phase3:
        assert training_cfg.events.push_robot is not None


@pytest.mark.parametrize(
    "config_type", [LowerLimbAgentEnvCfg, LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3]
)
def test_torque_reference_comes_from_nominal_actuator_config(config_type):
    cfg = config_type()
    names = ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "right_ankle_roll_joint"]
    asset = SimpleNamespace(
        cfg=cfg.scene.robot,
        joint_names=names,
        data=SimpleNamespace(applied_torque=SimpleNamespace(torch=torch.ones(1, len(names)))),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": asset})
    reward_cfg = copy.deepcopy(cfg.rewards.dof_torques)
    reward_cfg.params["asset_cfg"].joint_ids = slice(None)
    reward = joint_torques_l2_normalized(reward_cfg, env)
    torch.testing.assert_close(reward._nominal_gains, torch.tensor([100.0, 200.0, 20.0, 20.0]))
    # Startup configuration overrides are part of the nominal reference, before manager init.
    cfg.scene.robot.actuators["feet"].stiffness[".*_ankle_pitch_joint"] = 22.0
    updated_reward = joint_torques_l2_normalized(reward_cfg, env)
    torch.testing.assert_close(updated_reward._nominal_gains, torch.tensor([100.0, 200.0, 22.0, 20.0]))
    # The already initialized term retains its original snapshot.
    torch.testing.assert_close(reward._nominal_gains, torch.tensor([100.0, 200.0, 20.0, 20.0]))


def _torque_reward(stiffness, joint_ids=slice(None)):
    asset = SimpleNamespace(
        joint_names=["left_ankle_pitch_joint", "left_knee_joint"],
        data=SimpleNamespace(applied_torque=SimpleNamespace(torch=torch.tensor([[10.0, 20.0], [20.0, 40.0]]))),
    )
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": asset})
    cfg = RewardTermCfg(
        func=joint_torques_l2_normalized,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_ids=joint_ids), "stiffness": stiffness},
    )
    return joint_torques_l2_normalized(cfg, env), env, cfg.params


@pytest.mark.parametrize("joint_ids", [slice(None), [0, 1], torch.tensor([0, 1], dtype=torch.int32)])
def test_torque_reward_uses_fixed_nominal_gains(joint_ids):
    reward, env, params = _torque_reward({".*ankle.*": 20.0, ".*knee.*": 200.0}, joint_ids)
    torch.testing.assert_close(reward(env, **params), torch.tensor([0.26, 1.04]))
    # Changing live stiffness must not change the nominal reference penalty.
    env.scene["robot"].data.joint_stiffness = SimpleNamespace(torch=torch.tensor([[22.0, 180.0], [18.0, 220.0]]))
    torch.testing.assert_close(reward(env, **params), torch.tensor([0.26, 1.04]))
    reward.reset(torch.tensor([0]))
    torch.testing.assert_close(reward(env, **params), torch.tensor([0.26, 1.04]))


def test_torque_reward_respects_joint_selection_and_order():
    reward, env, params = _torque_reward({".*ankle.*": 20.0}, [0])
    torch.testing.assert_close(reward(env, **params), torch.tensor([0.25, 1.0]))
    reward, env, params = _torque_reward({".*ankle.*": 20.0, ".*knee.*": 200.0}, [1, 0])
    torch.testing.assert_close(reward(env, **params), torch.tensor([0.26, 1.04]))


@pytest.mark.parametrize("gain", [0.0, -1.0, float("nan"), float("inf"), None])
def test_torque_reward_rejects_invalid_nominal_gains(gain):
    with pytest.raises(ValueError, match="finite and positive"):
        _torque_reward({".*": gain})


def test_torque_reward_rejects_unmatched_selected_joints():
    with pytest.raises(ValueError, match="left_knee_joint"):
        _torque_reward({".*ankle.*": 20.0})
