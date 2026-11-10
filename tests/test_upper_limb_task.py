"""World-fixed bimanual commands, mode-aware rewards and local default-state resets."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_from_euler_xyz, quat_mul

from g1.tasks.upper_limb_agent.mdp.actions import UPPER_BODY_JOINT_NAMES
from g1.tasks.upper_limb_agent.mdp.commands import BimanualPoseCommand, BimanualPoseCommandCfg
from g1.tasks.upper_limb_agent.mdp.events import reset_robot_joints
from g1.tasks.upper_limb_agent.mdp.observations import active_arms, end_effector_pose, end_effector_pose_error
from g1.tasks.upper_limb_agent.mdp.rewards import (
    end_effector_orientation_tracking,
    end_effector_position_tracking,
    idle_arm_default_pose,
    joint_torque_limits,
    workspace_quality_bonus,
)
from g1.tasks.upper_limb_agent.mdp.workspace_quality import WorkspaceQualityCfg, WorkspaceQualityMap


@pytest.fixture
def pose_command():
    """Real command lifecycle with different environment origins and nontrivial wrist frames."""
    count = 3
    zero = torch.zeros(count)
    origins = torch.tensor([[2.0, 3.0, 0.0], [-3.0, 1.0, 0.0], [5.0, -2.0, 0.0]])
    reference_pos = origins + torch.tensor([0.0, 0.0, 0.75])
    reference_quat = quat_from_euler_xyz(zero, zero, torch.tensor([0.7, -1.2, 1.57]))
    poses = torch.zeros(count, 4, 7)
    poses[..., 6] = 1.0
    poses[:, 1] = torch.cat((reference_pos, reference_quat), dim=-1)
    for side, y in enumerate((0.15, -0.15)):
        wrist_pos = torch.tensor([0.2, y, 0.1]).expand(count, -1)
        wrist_quat = quat_from_euler_xyz(zero + 0.15, zero - 0.1, zero)
        pos, quat = combine_frame_transforms(reference_pos, reference_quat, wrist_pos, wrist_quat)
        poses[:, side + 2] = torch.cat((pos, quat), dim=-1)
    # Robot joint indices deliberately differ from 17-dimensional action indices.
    names = ("other", *UPPER_BODY_JOINT_NAMES, "finger")
    defaults = torch.linspace(-0.1, 0.1, len(names)).expand(count, -1).clone()
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=poses),
            joint_pos=SimpleNamespace(torch=defaults.clone()),
            default_joint_pos=SimpleNamespace(torch=defaults),
        ),
        find_bodies=lambda name: ([{"pelvis": 1, "left_wrist_yaw_link": 2, "right_wrist_yaw_link": 3}[name]], [name]),
        find_joints=lambda query, preserve_order: ([names.index(name) for name in query], query),
        is_initialized=True,
    )

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = origins
    env = SimpleNamespace(
        num_envs=count,
        device="cpu",
        scene=scene,
        episode_length_buf=torch.ones(count, dtype=torch.long),
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda obj: None)),
    )
    ranges = BimanualPoseCommandCfg.Ranges(
        pos_x=(0, 0), pos_y=(0, 0), pos_z=(0, 0), roll=(0, 0), pitch=(0, 0), yaw=(0, 0)
    )
    cfg = BimanualPoseCommandCfg(mode_probabilities=(0, 0, 1), left_offsets=ranges, right_offsets=ranges.copy())
    term = BimanualPoseCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    term.reset()
    return term


def test_default_wrist_anchor_world_goal_and_pelvis_observations_agree(pose_command):
    term = pose_command
    current = term.robot.data.body_link_pose_w.torch[:, [2, 3]]
    torch.testing.assert_close(term.goal_pose_w, current)
    torch.testing.assert_close(end_effector_pose(term._env, "wrist_pose"), term.command, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        term.default_wrist_pose_e[..., :3] + term._env.scene.env_origins[:, None], current[..., :3]
    )
    position_error, orientation_error = term.pose_error()
    torch.testing.assert_close(position_error, torch.zeros(3, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(orientation_error, torch.zeros(3, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(end_effector_pose_error(term._env, "wrist_pose"), torch.zeros(3, 12), atol=1e-6, rtol=0)


def test_pose_error_observation_has_signed_pelvis_vectors_and_matches_reward_error_norms(pose_command):
    term = pose_command
    poses = term.robot.data.body_link_pose_w.torch
    pelvis_quat = poses[:, 1, 3:][:, None].expand(-1, 2, -1)
    delta = torch.tensor([[0.03, -0.02, 0.01], [-0.04, 0.01, -0.02]]).expand(3, -1, -1)
    rotations = torch.tensor([[0.25, 0.0, 0.0], [0.0, -0.2, 0.0]]).expand(3, -1, -1)
    _, current_orientation = term.current_pose_b()
    # Apply separate rotations along pelvis X/Y, with non-identity current wrist orientations.
    target_orientation_b = quat_mul(quat_from_euler_xyz(*rotations.unbind(-1)), current_orientation)
    term.goal_pose_w[..., :3] = poses[:, [2, 3], :3] + quat_apply(pelvis_quat, delta)
    term.goal_pose_w[..., 3:] = quat_mul(pelvis_quat, target_orientation_b)
    errors = end_effector_pose_error(term._env, "wrist_pose").reshape(3, 2, 6)
    torch.testing.assert_close(errors[..., :3], delta, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(errors[..., 3:], rotations, atol=1e-6, rtol=1e-6)
    position_norm, orientation_norm = term.pose_error()
    torch.testing.assert_close(errors[..., :3].norm(dim=-1), position_norm, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(errors[..., 3:].norm(dim=-1), orientation_norm, atol=1e-6, rtol=1e-6)
    term.goal_pose_w[..., 3:] *= -1  # Equivalent quaternion signs must give the same error.
    torch.testing.assert_close(
        end_effector_pose_error(term._env, "wrist_pose").reshape(3, 2, 6), errors, atol=1e-6, rtol=1e-6
    )


def test_pose_error_observation_masks_inactive_targets_in_all_modes(pose_command):
    term = pose_command
    term.mode_id[:] = torch.tensor([0, 1, 2])
    term.goal_pose_w[..., 0] += 0.1
    delta = quat_from_euler_xyz(torch.full((3, 2), 0.3), torch.zeros(3, 2), torch.zeros(3, 2))
    term.goal_pose_w[..., 3:] = quat_mul(term.goal_pose_w[..., 3:], delta)
    errors = end_effector_pose_error(term._env, "wrist_pose").reshape(3, 2, 6)
    assert torch.all(errors[term.active_mask].norm(dim=-1) > 0)
    torch.testing.assert_close(errors[~term.active_mask], torch.zeros(2, 6))


def test_world_goals_hold_while_live_pelvis_observations_change_without_command_update(pose_command):
    term = pose_command
    goal, before = term.goal_pose_w.clone(), term.command.clone()
    poses = term.robot.data.body_link_pose_w.torch
    poses[:, 1, :3] += torch.tensor([0.3, -0.2, 0.1])
    poses[:, 1, 3:] = quat_from_euler_xyz(torch.zeros(3), torch.zeros(3), torch.full((3,), 0.4))
    assert not torch.allclose(term.command, before)
    term._update_command()
    torch.testing.assert_close(term.goal_pose_w, goal)
    torch.testing.assert_close(term.pose_error()[0], torch.zeros(3, 2), atol=1e-6, rtol=0)
    # Rewards see current world geometry even before the command manager runs.
    poses[:, 2, 0] += 0.04
    expected = 1 - torch.tanh(torch.full((3,), 0.04) / 0.1)
    torch.testing.assert_close(end_effector_position_tracking(term._env, "wrist_pose", 0, 0.1), expected)


def test_offsets_use_default_wrist_and_environment_origin_not_current_robot_pose(pose_command):
    term = pose_command
    anchor, goal = term.default_wrist_pose_e.clone(), term.goal_pose_w.clone()
    initial_quat = term._sampling_quat_w.clone()
    term.robot.data.body_link_pose_w.torch[:, 1:, :3] += 10
    term._env.scene.env_origins[0] += torch.tensor([1.0, 2.0, 0.0])
    term.cfg.left_offsets.pos_x = (0.1, 0.1)
    term.cfg.left_offsets.yaw = (0.2, 0.2)
    term.reset(torch.tensor([0]))
    expected = (
        anchor[0, 0, :3] + term._env.scene.env_origins[0] + quat_apply(initial_quat[0], torch.tensor([0.1, 0.0, 0.0]))
    )
    torch.testing.assert_close(term.goal_pose_w[0, 0, :3], expected)
    rotation = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([0.2]))
    torch.testing.assert_close(term.goal_pose_w[0, 0, 3:], quat_mul(anchor[0:1, 0, 3:], rotation)[0])
    torch.testing.assert_close(term.default_wrist_pose_e, anchor)
    torch.testing.assert_close(term.goal_pose_w[1:], goal[1:])
    torch.testing.assert_close(term.goal_pose_w[..., 3:].norm(dim=-1), torch.ones(3, 2))


@pytest.mark.parametrize("mode", [0, 1, 2], ids=["left", "right", "both"])
def test_modes_have_explicit_flags_neutral_inactive_goals_and_stay_fixed_within_episode(pose_command, mode):
    term = pose_command
    term.cfg.mode_probabilities = tuple(float(i == mode) for i in range(3))
    term.reset()
    expected = torch.tensor([(mode != 1), (mode != 0)]).expand(3, -1)
    torch.testing.assert_close(active_arms(term._env, "wrist_pose"), expected.float())
    before = term.mode_id.clone()
    term.time_left.zero_()
    term.compute(0.02)
    torch.testing.assert_close(term.mode_id, before)
    pose = term.command.reshape(3, 2, 7)
    if mode != 2:
        neutral = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(3, -1)
        torch.testing.assert_close(pose[:, 1 - mode], neutral)


def test_distance_duration_uses_farthest_active_goal_and_new_cfg_only_at_next_sample(pose_command):
    term = pose_command
    term.cfg.resampling_time_range = (0.01, 20.0)
    term.cfg.distance_reference_speed = 0.1
    term.cfg.right_offsets.pos_x = (0.5, 0.5)
    term.mode_id[:] = torch.tensor([0, 1, 2])
    term._resample(torch.arange(3))
    left_distance = (0.2**2 + 0.15**2 + 0.1**2) ** 0.5
    right_distance = (0.7**2 + 0.15**2 + 0.1**2) ** 0.5
    torch.testing.assert_close(term.time_left, torch.tensor([left_distance, right_distance, right_distance]) / 0.1)
    before = term.time_left.clone()
    term.cfg.distance_reference_speed = 0.2
    term.compute(0.01)
    torch.testing.assert_close(term.time_left, before - 0.01)
    term._resample(torch.arange(3))
    torch.testing.assert_close(term.time_left, before / 2)
    term.cfg.resampling_time_range = (3.0, 5.0)
    term.cfg.distance_reference_speed = 0.01
    term._resample(torch.arange(3))
    torch.testing.assert_close(term.time_left, torch.full((3,), 5.0))
    term.cfg.distance_reference_speed = 10.0
    term._resample(torch.arange(3))
    torch.testing.assert_close(term.time_left, torch.full((3,), 3.0))


def test_rewards_gate_inactive_targets_keep_dual_scale_and_ignore_quaternion_sign(pose_command):
    term = pose_command
    term.mode_id[:] = torch.tensor([0, 1, 2])
    term.goal_pose_w[..., 0] += 0.04
    term.goal_pose_w[..., 3:] *= -1
    expected = 1 - torch.tanh(torch.full((3,), 0.04) / 0.1)
    for side, weights in enumerate(([2.0, 0.0, 1.0], [0.0, 2.0, 1.0])):
        torch.testing.assert_close(
            end_effector_position_tracking(term._env, "wrist_pose", side, 0.1), expected * torch.tensor(weights)
        )
        torch.testing.assert_close(
            end_effector_orientation_tracking(term._env, "wrist_pose", side, 0.2462, 0.05655, 0.4),
            torch.tensor(weights),
        )
    delta = quat_from_euler_xyz(torch.full((3, 2), 0.3), torch.zeros(3, 2), torch.zeros(3, 2))
    term.goal_pose_w[..., 3:] = quat_mul(term.goal_pose_w[..., 3:], delta)
    orientation_reward = end_effector_orientation_tracking(term._env, "wrist_pose", 0, 0.2462, 0.05655, 0.4)
    assert 0 < orientation_reward[2] < 1
    torch.testing.assert_close(orientation_reward[0], orientation_reward[2] * 2)
    assert orientation_reward[1] == 0


def test_quaternion_error_matches_half_angle_at_small_and_half_turn_rotations(pose_command):
    term = pose_command
    angle = torch.tensor([0.0, 0.0001, torch.pi])[:, None].expand(-1, 2)
    delta = quat_from_euler_xyz(angle, torch.zeros_like(angle), torch.zeros_like(angle))
    term.goal_pose_w[..., 3:] = quat_mul(term.goal_pose_w[..., 3:], delta)
    expected = torch.sin(angle / 2).square()
    torch.testing.assert_close(term.orientation_error_quat_squared(), expected, atol=1e-11, rtol=0.002)
    # Either quaternion sign can change without changing the physical error.
    term.goal_pose_w[:, 0, 3:] *= -1
    term.robot.data.body_link_pose_w.torch[:, term.body_indices[1], 3:] *= -1
    torch.testing.assert_close(term.orientation_error_quat_squared(), expected, atol=1e-11, rtol=0.002)


def test_orientation_exponentials_use_quaternion_sigma_and_tunable_precision(pose_command):
    term = pose_command
    # Set quaternion vector errors exactly to each bandwidth; G(sigma) = exp(-1).
    sigma_coarse, sigma_fine = 0.2462, 0.05655
    error = torch.tensor([0.0, sigma_fine, sigma_coarse])[:, None].expand(-1, 2)
    angle = 2 * torch.asin(error)
    delta = quat_from_euler_xyz(angle, torch.zeros_like(angle), torch.zeros_like(angle))
    term.goal_pose_w[..., 3:] = quat_mul(term.goal_pose_w[..., 3:], delta)
    coarse = end_effector_orientation_tracking(term._env, "wrist_pose", 0, sigma_coarse, sigma_fine, 0.0)
    fine = end_effector_orientation_tracking(term._env, "wrist_pose", 0, sigma_coarse, sigma_fine, 1.0)
    mixture = end_effector_orientation_tracking(term._env, "wrist_pose", 0, sigma_coarse, sigma_fine, 0.4)
    torch.testing.assert_close(coarse[2], torch.tensor(0.36787944), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fine[1], torch.tensor(0.36787944), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mixture, 0.6 * coarse + 0.4 * fine)
    assert mixture[0] == pytest.approx(1.0)
    assert torch.all(mixture[:-1] > mixture[1:])
    tighter = end_effector_orientation_tracking(term._env, "wrist_pose", 0, sigma_coarse, sigma_fine / 2, 0.4)
    torch.testing.assert_close(tighter[0], mixture[0])
    assert tighter[1] < mixture[1]
    # Far outside the fine band, its contribution can be below float32 resolution.
    assert torch.all(tighter[1:] <= mixture[1:])


def test_quaternion_sigmas_preserve_previous_reference_angle_rewards(pose_command):
    term = pose_command
    angle = torch.tensor([0.08, 0.11, 0.35])[:, None].expand(-1, 2)
    delta = quat_from_euler_xyz(angle, torch.zeros_like(angle), torch.zeros_like(angle))
    term.goal_pose_w[..., 3:] = quat_mul(term.goal_pose_w[..., 3:], delta)
    coarse = end_effector_orientation_tracking(term._env, "wrist_pose", 0, 0.2462, 0.05655, 0.0)
    fine = end_effector_orientation_tracking(term._env, "wrist_pose", 0, 0.2462, 0.05655, 1.0)
    torch.testing.assert_close(coarse[2], torch.tensor(0.60653066), atol=1e-4, rtol=0)
    torch.testing.assert_close(fine[0], torch.tensor(0.60653066), atol=1e-4, rtol=0)
    # The reported 0.11 rad accuracy retains roughly the same reward baseline.
    torch.testing.assert_close(0.6 * coarse[1] + 0.4 * fine[1], torch.tensor(0.7263), atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "sigma_coarse,sigma_fine,fine_weight",
    [
        (0.35, 0, 0.4),
        (0.35, -0.08, 0.4),
        (0.08, 0.35, 0.4),
        (float("inf"), 0.08, 0.4),
        (0.35, float("nan"), 0.4),
        (0.35, 0.08, -0.1),
        (0.35, 0.08, 1.1),
        (0.35, 0.08, float("nan")),
    ],
)
def test_orientation_tracking_rejects_invalid_sigmas_and_mixture_weights(sigma_coarse, sigma_fine, fine_weight):
    with pytest.raises(ValueError):
        end_effector_orientation_tracking(None, "wrist_pose", 0, sigma_coarse, sigma_fine, fine_weight)


def test_idle_penalty_uses_only_idle_arm_actual_indices_and_default_angles(pose_command):
    term = pose_command
    term.mode_id[:] = torch.tensor([0, 1, 2])
    position = term.robot.data.joint_pos.torch
    position[:, 1:4] += 2  # Shared waist must not be part of the idle penalty.
    torch.testing.assert_close(idle_arm_default_pose(term._env, "wrist_pose"), torch.zeros(3))
    position[:, term.arm_joint_ids[0]] += 0.2
    position[:, term.arm_joint_ids[1]] += 0.3
    torch.testing.assert_close(idle_arm_default_pose(term._env, "wrist_pose"), torch.tensor([0.09, 0.04, 0]))


def test_torque_limits_penalize_only_excess_in_both_directions_and_follow_live_limits():
    torque = torch.tensor([[1000.0, 16.0, 1000.0, -36.0], [1000.0, -21.0, 1000.0, 35.0]])
    limits = torch.tensor([[10.0, 20.0, 30.0, 40.0], [10.0, 25.0, 30.0, 50.0]])
    robot = SimpleNamespace(
        data=SimpleNamespace(joint_effort_limits=SimpleNamespace(torch=limits)),
        actuators=SimpleNamespace(applied_effort=SimpleNamespace(torch=torque)),
    )
    env = SimpleNamespace(scene={"robot": robot})
    cfg = SceneEntityCfg("robot", joint_ids=[3, 1])
    torch.testing.assert_close(joint_torque_limits(env, cfg, 0.8), torch.tensor([4.0, 1.0]))
    torque.neg_()
    torch.testing.assert_close(joint_torque_limits(env, cfg, 0.8), torch.tensor([4.0, 1.0]))
    limits[0, 3] = 50
    limits[1, 1] = 20
    torch.testing.assert_close(joint_torque_limits(env, cfg, 0.8), torch.tensor([0.0, 5.0]))


@pytest.mark.parametrize("ratio", [0.0, -0.8, 1.01, float("inf"), float("nan")])
def test_torque_limits_reject_invalid_soft_ratios(ratio):
    with pytest.raises(ValueError, match="soft_ratio"):
        joint_torque_limits(None, SceneEntityCfg("robot"), ratio)


def test_metrics_average_valid_steps_by_mode_and_partial_reset_preserves_other_state(pose_command):
    term = pose_command
    term.mode_id[:] = torch.tensor([0, 1, 2])
    term.goal_pose_w[..., 0] += 0.04
    term._env.episode_length_buf[:] = torch.tensor([1, 1, 0])
    term._update_metrics()
    term.goal_pose_w[..., 0] += 0.04
    term._update_metrics()
    before = (term.mode_id.clone(), term.goal_pose_w.clone(), term.time_left.clone(), term._metric_steps.clone())
    logs = term.reset(torch.tensor([0]))
    assert logs["left_only/episodes"] == 1
    torch.testing.assert_close(logs["left_only/left_position_error_m"], torch.tensor(0.06))
    assert "left_only/right_position_error_m" not in logs
    assert "right_only/right_position_error_m" not in logs  # Missing samples are not logged as zero error.
    for current, old in zip((term.mode_id, term.goal_pose_w, term.time_left, term._metric_steps), before, strict=True):
        torch.testing.assert_close(current[1:], old[1:])
    assert term._metric_steps[0] == 0 and term._metric_steps[2] == 0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"mode_probabilities": (0, 0, 0)}, "mode_probabilities"),
        ({"mode_probabilities": (-1, 1, 1)}, "mode_probabilities"),
        ({"distance_reference_speed": 0}, "distance_reference_speed"),
        ({"resampling_time_range": (5, 3)}, "resampling_time_range"),
        ({"body_names": ("same", "same")}, "body_names"),
    ],
)
def test_invalid_command_configuration_fails_early(change, match):
    with pytest.raises(ValueError, match=match):
        BimanualPoseCommand(BimanualPoseCommandCfg(**change), None)


def test_upper_asset_is_owned_independently_of_lower_asset(monkeypatch):
    from g1.tasks.lower_limb_agent.config.g1.utils.assets import G1_29DOF_CFG as lower
    from g1.tasks.upper_limb_agent.config.g1.utils.assets import G1_29DOF_CFG as upper

    assert lower is not upper
    before = lower.actuators["arms"].stiffness
    monkeypatch.setattr(upper.actuators["arms"], "stiffness", before + 1.0)
    assert lower.actuators["arms"].stiffness == before


@pytest.mark.parametrize("env_ids", [torch.tensor([0, 2]), slice(1, 3)], ids=["indices", "slice"])
def test_joint_reset_synchronizes_pd_without_touching_ongoing_environments(env_ids):
    default_pos = torch.arange(12, dtype=torch.float).reshape(3, 4)
    default_vel = torch.zeros_like(default_pos)
    buffers = {
        name: torch.full_like(default_pos, 99.0) for name in ("position", "velocity", "pd_pos", "pd_vel", "effort")
    }

    def write(name, value, env_ids):
        buffers[name][env_ids] = value

    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=default_pos), default_joint_vel=SimpleNamespace(torch=default_vel)
        ),
        write_joint_position_to_sim_index=lambda position, env_ids: write("position", position, env_ids),
        write_joint_velocity_to_sim_index=lambda velocity, env_ids: write("velocity", velocity, env_ids),
        actuators=SimpleNamespace(
            target_command=SimpleNamespace(
                set_position_index=lambda value, env_ids: write("pd_pos", value, env_ids),
                set_velocity_index=lambda value, env_ids: write("pd_vel", value, env_ids),
                set_effort_index=lambda value, env_ids: write("effort", value, env_ids),
            )
        ),
    )
    reset_robot_joints(SimpleNamespace(scene={"robot": robot}), env_ids)
    expected_pos, expected_zero = torch.full_like(default_pos, 99.0), torch.full_like(default_pos, 99.0)
    expected_pos[env_ids] = default_pos[env_ids]
    expected_zero[env_ids] = 0
    for name in ("position", "pd_pos"):
        torch.testing.assert_close(buffers[name], expected_pos)
    for name in ("velocity", "pd_vel", "effort"):
        torch.testing.assert_close(buffers[name], expected_zero)
    # A writer must not retain an alias to the authored default state.
    buffers["position"].zero_()
    torch.testing.assert_close(default_pos, torch.arange(12, dtype=torch.float).reshape(3, 4))


@pytest.fixture
def quality_command(pose_command):
    term = pose_command
    term.cfg.workspace_quality = WorkspaceQualityCfg()
    term.workspace_quality_map = WorkspaceQualityMap(term.cfg.workspace_quality.map_path, "cpu")
    term._workspace_sums = torch.zeros(3, 7)
    term._env.episode_length_buf = torch.ones(3, dtype=torch.long)
    term.mode_id[:] = torch.tensor([0, 1, 2])
    return term


def test_workspace_reward_uses_fresh_world_goals_masks_idle_and_ignores_orientation(quality_command):
    term = quality_command
    original = workspace_quality_bonus(term._env).clone()
    assert (original > 0).all()
    term.goal_pose_w[0, 1, :3] += 100  # Idle right target has no influence.
    term.goal_pose_w[1, 0, :3] += 100  # Idle left target has no influence.
    term.goal_pose_w[..., 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(workspace_quality_bonus(term._env), original)
    term.goal_pose_w[0, 0, 0] += 0.02
    term.goal_pose_w[2, 1, 0] += 0.02
    result = workspace_quality_bonus(term._env)
    assert result[0] == result[2] == 0
    torch.testing.assert_close(result[1], original[1])


def test_workspace_goal_relative_query_changes_with_pelvis_but_not_arm_tracking_adjustment(quality_command):
    term = quality_command
    previous = term.workspace_quality_state()
    poses = term.robot.data.body_link_pose_w.torch
    reference = poses[:, term.reference_body_idx]
    local_shift = torch.tensor([0.005, 0, 0]).expand(3, -1)
    poses[:, term.body_indices[0], :3] += quat_apply(reference[:, 3:], local_shift)
    current = term.workspace_quality_state()
    torch.testing.assert_close(current.target_scores, previous.target_scores, rtol=0, atol=0)
    torch.testing.assert_close(current.reward, previous.reward)
    # Translate the pelvis while keeping the world goals and physical wrists fixed.
    poses[:, term.reference_body_idx, :3] += quat_apply(reference[:, 3:], local_shift * 2)
    moved = term.workspace_quality_state()
    expected_b = torch.tensor([[[0.19, 0.15, 0.1], [0.19, -0.15, 0.1]]]).expand(3, -1, -1).clone()
    expected_b[:, 1, 1] *= -1
    expected, covered = term.workspace_quality_map.query(expected_b)
    torch.testing.assert_close(moved.target_scores, expected, atol=1e-6, rtol=1e-6)
    assert covered.all() and moved.position_ready.all()


@pytest.mark.parametrize("weight", [0.0, 0.02])
def test_workspace_weight_dt_logs_and_partial_reset(quality_command, weight):
    from isaaclab.managers import RewardManager, RewardTermCfg

    term, env = quality_command, quality_command._env
    env.max_episode_length_s = 10.0
    env.sim = SimpleNamespace(is_playing=lambda: True)
    state = term.workspace_quality_state()
    manager = RewardManager({"workspace_quality": RewardTermCfg(func=workspace_quality_bonus, weight=weight)}, env)
    torch.testing.assert_close(manager.compute(1 / 60), state.reward * weight / 60)
    for _ in range(3):
        term._update_metrics()
    assert term._workspace_sums[:, 1].sum() > 0  # Diagnostics also run for weight=0.
    other = term._workspace_sums[1:].clone()
    extras = term.reset(torch.tensor([0]))
    assert extras["workspace_quality/position_ready_fraction"] == 1
    assert extras["workspace_quality/map_coverage_fraction"] == 1
    assert extras["workspace_quality/reward_available_fraction"] == 1
    assert extras["workspace_quality/target_score"] > 0
    assert not term._workspace_sums[0].any()
    torch.testing.assert_close(term._workspace_sums[1:], other, rtol=0, atol=0)


def test_workspace_unknown_targets_do_not_log_zero_as_a_valid_score(quality_command):
    term = quality_command
    term.goal_pose_w[..., :3] += 100
    term._update_metrics()
    extras = term._episode_metrics(torch.arange(3))
    assert "workspace_quality/target_score" not in extras
    assert extras["workspace_quality/map_coverage_fraction"] == 0
    assert extras["workspace_quality/position_ready_fraction"] == 0
    assert extras["workspace_quality/reward_available_fraction"] == 0
    assert extras["workspace_quality/actual_score"] > 0


def test_workspace_reward_availability_counts_joint_events_not_marginal_products(quality_command, monkeypatch):
    term = quality_command
    # One frame is position-ready but uncovered; another is covered but not
    # position-ready. Both marginal proportions are 1/2; joint proportion is 0.
    state = term.workspace_quality_state()
    state.position_ready[:] = True
    state.map_covered[:] = False
    monkeypatch.setattr(term, "workspace_quality_state", lambda: state)
    term._update_metrics()
    state.position_ready[:] = False
    state.map_covered[:] = True
    term._update_metrics()
    extras = term._episode_metrics(torch.arange(3))
    assert extras["workspace_quality/position_ready_fraction"] == 0.5
    assert extras["workspace_quality/map_coverage_fraction"] == 0.5
    assert extras["workspace_quality/reward_available_fraction"] == 0


def test_workspace_command_validation_checks_frame_and_tolerance(pose_command):
    cfg = pose_command.cfg.copy()
    cfg.workspace_quality = WorkspaceQualityCfg(position_tolerance_m=-1)
    with pytest.raises(ValueError, match="positive and finite"):
        BimanualPoseCommand(cfg, pose_command._env)
    cfg.workspace_quality.position_tolerance_m = 0.01
    cfg.reference_body_name = "torso"
    with pytest.raises(ValueError, match="pelvis-frame"):
        BimanualPoseCommand(cfg, pose_command._env)
