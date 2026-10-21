"""Upper-body reset commands match physical state and stay local to reset slots."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

from g1.tasks.lower_limb_agent.env import LowerLimbAgentEnv
from g1.tasks.lower_limb_agent.mdp.events import UpperBodyDisturbanceEvent
from g1.tasks.lower_limb_agent.motion import UpperBodyMotion, UpperBodyMotionCfg


def _write(buffer, value, env_ids, joint_ids):
    rows = torch.arange(buffer.shape[0])[env_ids].long()
    columns = torch.as_tensor(joint_ids).long()
    assert value.shape == (rows.numel(), columns.numel())
    buffer[rows[:, None], columns[None, :]] = value


class _Command:
    def __init__(self, shape):
        self.position = torch.full(shape, -9.0)
        self.velocity = torch.full(shape, 7.0)
        self.effort = torch.full(shape, 11.0)

    def set_position_index(self, *, value, joint_ids, env_ids):
        _write(self.position, value, env_ids, joint_ids)

    def set_velocity_index(self, *, value, joint_ids, env_ids):
        _write(self.velocity, value, env_ids, joint_ids)

    def set_effort_index(self, *, value, joint_ids, env_ids):
        _write(self.effort, value, env_ids, joint_ids)


class _Robot:
    def __init__(self):
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10),
            joint_vel=SimpleNamespace(torch=torch.full((4, 5), 0.4)),
            default_joint_pos=SimpleNamespace(torch=torch.zeros(4, 5)),
        )
        self.actuators = SimpleNamespace(target_command=_Command((4, 5)))

    def write_joint_velocity_to_sim_index(self, *, velocity, joint_ids, env_ids):
        _write(self.data.joint_vel.torch, velocity, env_ids, joint_ids)


@pytest.mark.parametrize(
    "env_ids",
    [
        torch.tensor([3, 1], dtype=torch.int32),
        torch.tensor([2]),
        slice(None),
        slice(1, None, 2),
        torch.tensor([], dtype=torch.long),
        slice(2, 2),
    ],
)
@pytest.mark.parametrize("disturbance_enabled", [True, False])
def test_reset_discards_stale_commands_after_physical_reset(monkeypatch, env_ids, disturbance_enabled):
    robot = _Robot()
    project_env = LowerLimbAgentEnv.__new__(LowerLimbAgentEnv)
    project_env.scene = {"robot": robot}
    project_env.command_manager = SimpleNamespace(active_terms=[])
    # Use a reordered, noncontiguous device selection as finalized SceneEntityCfg does.
    upper_ids = torch.tensor([4, 1], dtype=torch.int32)
    entity = SimpleNamespace(name="robot", joint_ids=upper_ids)
    project_env.observation_manager = SimpleNamespace(
        cfg=SimpleNamespace(policy=SimpleNamespace(upper_body_target=SimpleNamespace(params={"asset_cfg": entity})))
    )
    event = None
    if disturbance_enabled:
        groups = [
            {"asset_cfg": SimpleNamespace(name="robot", joint_ids=[1])},
            {"asset_cfg": SimpleNamespace(name="robot", joint_ids=[4])},
        ]
        event = UpperBodyDisturbanceEvent(SimpleNamespace(params={"disturb_groups": groups}), project_env)
        torch.testing.assert_close(event.joint_target, robot.data.joint_pos.torch[:, [1, 4]])
        event.joint_target.fill_(-5)

    # Deliberately different from both the old pose and authored defaults.
    reset_pose = robot.data.joint_pos.torch.clone() + 0.03
    reset_velocity = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 100 + 0.1

    def physical_reset(self, ids):
        robot.data.joint_pos.torch[ids] = reset_pose[ids]
        robot.data.joint_vel.torch[ids] = reset_velocity[ids]
        if event is not None:
            event.reset(ids)

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", physical_reset)
    expected_position = robot.data.joint_pos.torch.clone()
    expected_velocity = robot.data.joint_vel.torch.clone()
    expected_position[env_ids] = reset_pose[env_ids]
    expected_velocity[env_ids] = reset_velocity[env_ids]
    command = robot.actuators.target_command
    expected_targets = command.position.clone()
    expected_velocity_targets = command.velocity.clone()
    expected_effort = command.effort.clone()
    _write(expected_targets, reset_pose[env_ids][:, upper_ids], env_ids, upper_ids)
    zero = torch.zeros_like(reset_pose[env_ids][:, upper_ids])
    _write(expected_velocity, zero, env_ids, upper_ids)
    _write(expected_velocity_targets, zero, env_ids, upper_ids)
    _write(expected_effort, zero, env_ids, upper_ids)
    rng_state = torch.get_rng_state()

    project_env._reset_idx(env_ids)

    torch.testing.assert_close(robot.data.joint_pos.torch, expected_position)
    torch.testing.assert_close(robot.data.joint_vel.torch, expected_velocity)
    torch.testing.assert_close(command.position, expected_targets)
    torch.testing.assert_close(command.velocity, expected_velocity_targets)
    torch.testing.assert_close(command.effort, expected_effort)
    assert torch.equal(torch.get_rng_state(), rng_state)
    if event is not None:
        expected_cache = torch.full_like(event.joint_target, -5)
        expected_cache[env_ids] = reset_pose[env_ids][:, [1, 4]]
        torch.testing.assert_close(event.joint_target, expected_cache)


def test_event_reset_none_clears_every_cached_target():
    robot = _Robot()
    env = SimpleNamespace(scene={"robot": robot})
    cfg = SimpleNamespace(params={"disturb_groups": [{"asset_cfg": SimpleNamespace(name="robot", joint_ids=[1, 4])}]})
    event = UpperBodyDisturbanceEvent(cfg, env)
    event.joint_target.fill_(-5)
    robot.data.joint_pos.torch.add_(0.02)

    event.reset()

    torch.testing.assert_close(event.joint_target, robot.data.joint_pos.torch[:, [1, 4]])


def test_reset_log_preserves_each_episode_sample_before_statistics_are_cleared(monkeypatch):
    robot = _Robot()
    env = LowerLimbAgentEnv.__new__(LowerLimbAgentEnv)
    env.scene = {"robot": robot}
    env.command_manager = SimpleNamespace(active_terms=[])
    entity = SimpleNamespace(name="robot", joint_ids=torch.tensor([4, 1]))
    env.observation_manager = SimpleNamespace(
        cfg=SimpleNamespace(policy=SimpleNamespace(upper_body_target=SimpleNamespace(params={"asset_cfg": entity})))
    )
    env.upper_body_motion = UpperBodyMotion(robot, entity.joint_ids, UpperBodyMotionCfg())
    stats = env.upper_body_motion.trajectory.statistics
    stats[1] = torch.tensor([2, 1, 4, 3, 0.5])
    stats[3] = torch.tensor([1, 1, 0, 2, 0.5])

    def physical_reset(self, ids):
        self.extras = {"log": {}}

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", physical_reset)
    env._reset_idx(torch.tensor([1, 3]))
    log = env.extras["log"]
    torch.testing.assert_close(log["UpperBodyMotion/started_per_episode"], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(log["UpperBodyMotion/reference_completion_rate"], torch.tensor([0.5, 1.0]))
    assert not stats[[1, 3]].any()
