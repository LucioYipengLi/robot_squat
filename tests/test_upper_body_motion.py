"""Simulator adapter, joint ordering and unchanged lower-body action mapping."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

from g1.tasks.lower_limb_agent.mdp.actions import LowerBodyPositionAction
from g1.tasks.lower_limb_agent.mdp.observations import upper_body_joint_pos_target_rel
from g1.tasks.lower_limb_agent.motion import UpperBodyMotion, UpperBodyMotionCfg


def _motion(profile="quintic_v1", device="cpu"):
    positions, velocities = torch.full((3, 5), -9.0, device=device), torch.full((3, 5), -9.0, device=device)

    def write(buffer, value, joint_ids, env_ids=None):
        rows = torch.arange(3, device=device) if env_ids is None else torch.arange(3, device=device)[env_ids]
        buffer[rows[:, None], joint_ids[None, :]] = value

    command = SimpleNamespace(
        set_position_index=lambda *, value, joint_ids, env_ids=None: write(positions, value, joint_ids, env_ids),
        set_velocity_index=lambda *, value, joint_ids, env_ids=None: write(velocities, value, joint_ids, env_ids),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.zeros(3, 5, device=device)),
            joint_pos_target=SimpleNamespace(torch=positions),
            default_joint_pos=SimpleNamespace(torch=torch.zeros(3, 5, device=device)),
        ),
        actuators=SimpleNamespace(target_command=command),
    )
    motion = UpperBodyMotion(robot, torch.tensor([3, 1], device=device), UpperBodyMotionCfg(profile=profile))
    return motion, positions, velocities


def test_goal_order_is_resolved_once_and_only_references_are_written():
    motion, positions, velocities = _motion()
    motion.set_goal(torch.tensor([1]), torch.tensor([[0.4, -0.3]]), [1, 3])
    assert (positions == -9).all()
    rng = torch.get_rng_state()
    for _ in range(200):
        motion.advance(1 / 120)
    torch.testing.assert_close(positions[1, [1, 3]], torch.tensor([0.4, -0.3]))
    assert not velocities[:, [1, 3]].any()
    assert (positions[:, [0, 2, 4]] == -9).all()
    assert not motion.robot.data.joint_pos.torch.any()
    assert torch.equal(rng, torch.get_rng_state())


def test_nominal_goals_do_not_create_artificial_movements():
    motion, _, _ = _motion()
    motion.set_goal(slice(None), torch.zeros(3, 2), [1, 3])
    motion.advance(1 / 120)
    assert not motion.trajectory.statistics.any()


def test_reset_discards_pending_motion_and_keeps_other_slots_running():
    motion, _, _ = _motion()
    motion.set_goal(slice(None), torch.ones(3, 2), [1, 3])
    motion.advance(0.1)
    motion.set_goal(slice(None), -torch.ones(3, 2), [1, 3])
    motion.robot.data.joint_pos.torch[1, [1, 3]] = torch.tensor([0.02, -0.03])
    motion.reset(torch.tensor([1]))
    torch.testing.assert_close(motion.get_observation_goal()[1], torch.tensor([-0.03, 0.02]))
    torch.testing.assert_close(motion.get_observation_goal()[[0, 2]], torch.ones(2, 2))
    motion.advance(0.1)
    torch.testing.assert_close(motion.trajectory.position[1], torch.tensor([-0.03, 0.02]))
    assert not motion.trajectory.statistics[1].any()
    assert motion.trajectory.active[0]


def test_step_profile_preserves_immediate_target_baseline():
    motion, positions, _ = _motion("step_v1")
    motion.set_goal(torch.tensor([0]), torch.tensor([[0.1, 0.2]]), [1, 3])
    torch.testing.assert_close(positions[0, [1, 3]], torch.tensor([0.1, 0.2]))
    snapshot = positions.clone()
    motion.advance(0.1)
    torch.testing.assert_close(positions, snapshot)


@pytest.mark.parametrize("dt", [1 / 120, 1 / 60])
def test_action_hook_delegates_lower_body_and_advances_exactly_once(monkeypatch, dt):
    calls = []
    monkeypatch.setattr(JointPositionAction, "apply_actions", lambda self: calls.append("lower"))
    action = LowerBodyPositionAction.__new__(LowerBodyPositionAction)
    action._env = SimpleNamespace(upper_body_motion=SimpleNamespace(advance=lambda value: calls.append(value)))
    action._motion_dt = dt
    action.apply_actions()
    assert calls == ["lower", dt]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA observation check")
        ),
    ],
)
def test_goal_preview_tracks_executable_segment_without_changing_trajectory(device):
    motion, positions, velocities = _motion(device=device)
    curve = motion.trajectory
    ids = torch.tensor([1], device=device)
    a = torch.tensor([[0.4, -0.3]], device=device)
    b, c = -a, a * 0.5

    def read(expected):
        state = {name: value.clone() for name, value in vars(curve).items() if isinstance(value, torch.Tensor)}
        commands = positions.clone(), velocities.clone()
        rng = torch.get_rng_state()
        if device != "cpu":
            cuda_rng = torch.cuda.get_rng_state(device)
        torch.testing.assert_close(motion.get_observation_goal()[ids], expected[:, [1, 0]])
        for name, value in state.items():
            torch.testing.assert_close(getattr(curve, name), value)
        torch.testing.assert_close(positions, commands[0])
        torch.testing.assert_close(velocities, commands[1])
        assert torch.equal(rng, torch.get_rng_state())
        if device != "cpu":
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))

    read(torch.zeros_like(a))
    motion.set_goal(ids, a, [1, 3])
    read(a)  # Preview before the first action, while the reference is unchanged.
    assert (positions == -9).all()
    motion.advance(0.1)
    read(a)
    assert not torch.allclose(curve.position[ids], a[:, [1, 0]])
    motion.set_goal(ids, b, [1, 3])
    motion.set_goal(ids, c, [1, 3])
    read(a)  # Replacing the queued target does not alter the running endpoint.
    motion.advance(curve.duration[1, 0].item())
    assert not curve.active[1] and curve.pending[1]
    read(c)  # Preview at rest, without starting the next segment.
    assert curve.statistics[1, 0] == 1
    motion.advance(0.01)
    read(c)
    motion.advance(curve.duration[1, 0].item())
    assert not curve.active[1] and not curve.pending[1]
    curve.pending_goal[1] = 99  # Inactive pending buffers must never leak.
    read(c)


@pytest.mark.parametrize("profile", ["quintic_v1", "step_v1"])
def test_observation_keeps_joint_order_default_offsets_and_reference_option(profile):
    motion, positions, _ = _motion(profile)
    motion.robot.data.default_joint_pos.torch[:] = torch.tensor([0.0, 0.1, 0.0, -0.2, 0.0])
    entity = SimpleNamespace(name="robot", joint_ids=motion.joint_ids)
    env = SimpleNamespace(scene={"robot": motion.robot}, upper_body_motion=motion)
    motion.set_goal(slice(None), torch.tensor([[0.4, -0.3]]).expand(3, -1), [1, 3])
    expected = torch.tensor([[-0.1, 0.3]]).expand(3, -1)
    torch.testing.assert_close(upper_body_joint_pos_target_rel(env, entity), expected)
    motion.advance(0.1)
    torch.testing.assert_close(upper_body_joint_pos_target_rel(env, entity), expected)
    reference = positions[:, motion.joint_ids] - motion.robot.data.default_joint_pos.torch[:, motion.joint_ids]
    torch.testing.assert_close(upper_body_joint_pos_target_rel(env, entity, "reference"), reference)
    if profile == "quintic_v1":
        assert not torch.allclose(reference, expected)
    with pytest.raises(ValueError, match="target source"):
        upper_body_joint_pos_target_rel(env, entity, "unknown")
