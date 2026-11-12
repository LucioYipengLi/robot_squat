"""Actual-liftoff WALK reference, support timing, sensor semantics and mode isolation."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from g1.tasks.lower_limb_agent.config.g1.env_cfg import WALK_SWING_TIME, LowerLimbAgentEnvCfg
from g1.tasks.lower_limb_agent.config.g1.env_cfg_phase import LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3
from g1.tasks.lower_limb_agent.mdp.rewards import walk_joint_reference, walk_support_timing
from g1.tasks.lower_limb_agent.performance import ModePerformanceMonitor


class _Scene(dict):
    pass


@pytest.fixture
def scene():
    # Deliberately scramble asset indices: the configured order must be used.
    ids = [4, 1, 5, 0, 3, 2]
    default = torch.zeros(4, 6)
    default[:, ids] = torch.tensor([-0.1, 0.3, -0.2, -0.1, 0.3, -0.2])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=default.clone()),
            default_joint_pos=SimpleNamespace(torch=default),
        )
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_air_time=SimpleNamespace(torch=torch.tensor([[0.2, 0.0]]).repeat(4, 1)),
            current_contact_time=SimpleNamespace(torch=torch.tensor([[0.0, 0.3]]).repeat(4, 1)),
            # Deliberately contradict the timers: new rewards must use the sensor's timer state.
            net_normal_forces_w=SimpleNamespace(torch=torch.full((4, 2, 3), 100.0)),
        )
    )
    world = _Scene(robot=robot)
    world.sensors = {"contact_forces": sensor}
    command = torch.zeros(4, 6)
    command[:, 1] = 0.6
    term = SimpleNamespace(mode=torch.full((4,), 2), command=command)
    env = SimpleNamespace(
        scene=world,
        command_manager=SimpleNamespace(get_term=lambda name: term),
        episode_length_buf=torch.zeros(4),
        step_dt=1 / 60,
    )
    asset_cfg = SceneEntityCfg("robot", joint_ids=ids)
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])
    return env, term, robot, sensor, asset_cfg, sensor_cfg


def _reference(scene, **kwargs):
    env, _, _, _, asset, sensor = scene
    return walk_joint_reference(env, asset, sensor, **kwargs)


def _support(scene, **kwargs):
    env, _, _, _, _, sensor = scene
    return walk_support_timing(env, sensor, **kwargs)


@pytest.mark.parametrize(
    "velocity,offset",
    [
        ([0.6, 0.0, 0.0], [-0.30, 0.50, -0.12, 0.30, 0.0, 0.12]),
        ([-0.6, 0.0, 0.0], [0.30, 0.50, 0.12, -0.30, 0.0, -0.12]),
        ([0.0, 0.6, 0.0], [0.0, 0.50, 0.0, 0.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.25, 0.0, 0.0, 0.0, 0.0]),
        ([0.11, 0.0, 0.0], [-0.0825, 0.25, -0.033, 0.0825, 0.0, 0.033]),
    ],
)
def test_known_mid_swing_reference_for_each_command_direction(scene, velocity, offset):
    _, term, robot, _, asset, _ = scene
    term.command[:, 1:4] = torch.tensor(velocity)
    robot.data.joint_pos.torch[:, asset.joint_ids] += torch.tensor(offset)
    torch.testing.assert_close(_reference(scene), torch.ones(4))
    assert not _support(scene).any()


def test_right_swing_is_the_mirror_and_reference_is_independent_of_episode_clock(scene):
    env, _, robot, sensor, asset, _ = scene
    sensor.data.current_air_time.torch[:] = torch.tensor([0.0, 0.2])
    sensor.data.current_contact_time.torch[:] = torch.tensor([0.3, 0.0])
    robot.data.joint_pos.torch[:, asset.joint_ids] += torch.tensor([0.3, 0.0, 0.12, -0.3, 0.5, -0.12])
    first = _reference(scene)
    env.episode_length_buf.fill_(9999)
    torch.testing.assert_close(_reference(scene), first, rtol=0, atol=0)
    torch.testing.assert_close(first, torch.ones(4))


def test_only_moving_walk_is_selected_with_correct_six_dimensional_command(scene):
    _, term, _, _, _, _ = scene
    term.mode[:] = torch.arange(4)
    assert (_reference(scene) > 0).tolist() == [False, False, True, False]
    term.mode.fill_(2)
    term.command.zero_()
    term.command[0, 0] = -0.5  # Height must not be mistaken for velocity.
    term.command[1, 3] = 0.21  # Pure yaw must use the actual yaw channel.
    term.command[2, 1] = 0.1
    term.command[3, 3] = 0.2
    assert (_reference(scene) > 0).tolist() == [False, True, False, False]


def test_double_support_flight_and_overdue_swing_receive_no_reference_bonus(scene):
    _, _, _, sensor, _, _ = scene
    sensor.data.current_air_time.torch[:] = torch.tensor([[0.0, 0.0], [0.2, 0.2], [0.4, 0.0], [0.8, 0.0]])
    sensor.data.current_contact_time.torch[:] = torch.tensor([[0.1, 0.1], [0.0, 0.0], [0.0, 0.3], [0.0, 0.3]])
    assert not _reference(scene).any()
    torch.testing.assert_close(_support(scene), torch.tensor([1.0, 1.0, 0.0, 0.5]))


def test_stance_overdue_cost_is_bounded_and_mode_gated(scene):
    _, term, _, sensor, _, _ = scene
    sensor.data.current_contact_time.torch[:] = torch.tensor([[0.0, 1.2], [1.2, 1.2], [0.0, 5.0], [0.0, 5.0]])
    term.mode[3] = 3
    torch.testing.assert_close(_support(scene), torch.tensor([0.5, 2.0, 0.5, 0.0]))


def test_repeated_reward_reads_preserve_physical_buffers_and_rng(scene):
    _, term, robot, sensor, _, _ = scene
    arrays = [
        term.command,
        robot.data.joint_pos.torch,
        robot.data.default_joint_pos.torch,
        sensor.data.current_air_time.torch,
        sensor.data.current_contact_time.torch,
    ]
    before = [value.clone() for value in arrays]
    rng = torch.get_rng_state()
    reference, support = _reference(scene), _support(scene)
    for _ in range(2):
        torch.testing.assert_close(_reference(scene), reference, rtol=0, atol=0)
        torch.testing.assert_close(_support(scene), support, rtol=0, atol=0)
    for value, saved in zip(arrays, before, strict=True):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "params",
    [{"sigma": 0}, {"swing_time": -1}, {"hip_amplitude": -0.1}, {"speed_reference": 0}, {"command_threshold": 0}],
)
def test_invalid_reference_parameters_fail(scene, params):
    with pytest.raises(ValueError):
        _reference(scene, **params)


def test_missing_timer_or_wrong_joint_count_fails_clearly(scene):
    _, _, _, sensor, asset, _ = scene
    asset.joint_ids = asset.joint_ids[:5]
    with pytest.raises(ValueError, match="six"):
        _reference(scene)
    sensor.data.current_air_time = None
    with pytest.raises(ValueError, match="track_air_time"):
        _support(scene)


@pytest.mark.parametrize("cls", [LowerLimbAgentEnvCfg, LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3])
def test_stage_configs_keep_original_vertical_constraints_and_sensor_threshold(cls):
    cfg = cls()
    assert cfg.scene.contact_forces.force_threshold == 1.0
    assert cfg.rewards.track_pelvis_height.weight == 2.0
    assert cfg.rewards.track_pelvis_height.params == {"command_name": "task_command"}
    assert cfg.rewards.vertical_bounce.weight == -0.5
    assert cfg.rewards.vertical_bounce.func.__name__ == "lin_vel_z_l2"
    assert cfg.rewards.leg_synergy.weight == 0.75 and not hasattr(cfg.rewards, "leg_synergy_walk")
    assert cfg.rewards.feet_gait_phase.params["walk_modes"] == (3,)
    assert cfg.rewards.walk_joint_reference.weight == 2.0
    assert cfg.rewards.walk_joint_reference.params["asset_cfg"].preserve_order
    assert cfg.rewards.walk_joint_reference.params["sensor_cfg"].preserve_order
    assert cfg.rewards.walk_support_timing.weight == -0.75
    assert cfg.rewards.walk_joint_reference.params["swing_time"] == WALK_SWING_TIME
    assert cfg.rewards.walk_support_timing.params["swing_time"] == WALK_SWING_TIME


def test_walk_metrics_use_frame_counts_single_support_denominator_and_true_rms():
    monitor = ModePerformanceMonitor(3, "cpu", 1 / 60, (0.03, 0.5, 0.4, 0.02))
    mode, valid = torch.tensor([2, 2, 3]), torch.tensor([True, True, True])
    command = torch.tensor([[0.0, 0.6, 0.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    contacts = torch.tensor([[False, True], [True, True], [False, True]])
    height = torch.tensor([[0.23, 0.05], [0.05, 0.05], [9.0, 9.0]])
    hips = torch.tensor([[0.3, -0.1], [0.1, 0.1], [9.0, 9.0]])
    monitor.sample_walk(mode, valid, command, contacts, height, hips)
    monitor.reset(torch.tensor([0]), mode, 1)  # Episode reset must not erase the logging window.
    metrics = monitor.drain(torch.ones(4))
    assert metrics["Performance/walk/single_stance_fraction"] == 0.5
    assert metrics["Performance/walk/swing_foot_height_m"] == pytest.approx(0.23)
    assert metrics["Performance/walk/hip_pitch_rms_rad"] == pytest.approx(0.03**0.5)
    assert "Performance/walk/single_stance_fraction" not in monitor.drain(torch.ones(4))
    valid.zero_()
    monitor.sample_walk(mode, valid, command, contacts, height, hips)
    assert "Performance/walk/hip_pitch_rms_rad" not in monitor.drain(torch.ones(4))
    valid[:] = torch.tensor([True, False, False])
    contacts.fill_(True)
    monitor.sample_walk(mode, valid, command, contacts, height, hips)
    metrics = monitor.drain(torch.ones(4))
    assert metrics["Performance/walk/single_stance_fraction"] == 0
    assert "Performance/walk/swing_foot_height_m" not in metrics
