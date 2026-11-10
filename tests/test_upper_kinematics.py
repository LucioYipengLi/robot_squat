"""Analytical oracles for wrist acceleration in a translating and rotating pelvis frame."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from g1.tasks.upper_limb_agent.mdp.kinematics import (
    link_origin_linear_acceleration_w,
    relative_link_linear_acceleration,
)


def moving_frame_data(*, rotating=True, translating=True, relative_velocity=True, relative_acceleration=True):
    """Synthesize consistent COM telemetry from known local polynomial motion."""
    generator = torch.Generator().manual_seed(124)

    def random(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    batch, count = 3, 4
    q = quat_from_euler_xyz(*random(3, batch))
    local_position, local_velocity, local_acceleration = (random(batch, 2, 3) for _ in range(3))
    if not relative_velocity:
        local_velocity.zero_()
    if not relative_acceleration:
        local_acceleration.zero_()
    omega, alpha = random(batch, 1, 3), random(batch, 1, 3)
    if not rotating:
        omega.zero_()
        alpha.zero_()
    p_ref, v_ref, a_ref = (random(batch, 1, 3) for _ in range(3))
    if not translating:
        v_ref.zero_()
        a_ref.zero_()
    rotation = q[:, None].expand(-1, 2, -1)
    r, u, relative_acc_w = (
        quat_apply(rotation, value) for value in (local_position, local_velocity, local_acceleration)
    )
    wrist_acc = (
        a_ref
        + torch.linalg.cross(alpha, r)
        + torch.linalg.cross(omega, torch.linalg.cross(omega, r))
        + 2 * torch.linalg.cross(omega, u)
        + relative_acc_w
    )
    # Noncontiguous body order also checks reference/body selection.
    body_ids, reference_id = [3, 0], 2
    pose = torch.zeros(batch, count, 7, dtype=torch.float64)
    pose[..., 3:] = q[:, None]
    pose[:, reference_id, :3] = p_ref[:, 0]
    pose[:, body_ids, :3] = p_ref + r
    velocity = random(batch, count, 6)
    velocity[:, reference_id, :3] = v_ref[:, 0]
    velocity[:, reference_id, 3:] = omega[:, 0]
    velocity[:, body_ids, :3] = v_ref + torch.linalg.cross(omega, r) + u
    link_acc = random(batch, count, 6)
    link_acc[:, reference_id, :3] = a_ref[:, 0]
    link_acc[:, reference_id, 3:] = alpha[:, 0]
    link_acc[:, body_ids, :3] = wrist_acc
    # Every origin, including the pelvis, is offset from its COM.
    origin_from_com = random(batch, count, 3) * 0.1
    com_pose = pose.clone()
    com_pose[..., :3] -= origin_from_com
    com_velocity = velocity.clone()
    com_velocity[..., :3] -= torch.linalg.cross(velocity[..., 3:], origin_from_com)
    com_acc = link_acc.clone()
    com_acc[..., :3] -= torch.linalg.cross(link_acc[..., 3:], origin_from_com) + torch.linalg.cross(
        velocity[..., 3:], torch.linalg.cross(velocity[..., 3:], origin_from_com)
    )
    data = SimpleNamespace(
        **{
            name: SimpleNamespace(torch=value)
            for name, value in {
                "body_link_pose_w": pose,
                "body_link_vel_w": velocity,
                "body_com_pose_w": com_pose,
                "body_com_vel_w": com_velocity,
                "body_com_acc_w": com_acc,
            }.items()
        }
    )
    return data, body_ids, reference_id, local_acceleration, wrist_acc


@pytest.mark.parametrize("rotating,translating", [(False, True), (True, False), (True, True)])
def test_rigid_base_motion_has_zero_relative_acceleration(rotating, translating):
    data, ids, ref, expected, _ = moving_frame_data(
        rotating=rotating, translating=translating, relative_velocity=False, relative_acceleration=False
    )
    torch.testing.assert_close(relative_link_linear_acceleration(data, ids, ref), expected, atol=1e-12, rtol=1e-12)


def test_constant_relative_velocity_in_rotating_frame_removes_coriolis_acceleration():
    data, ids, ref, expected, _ = moving_frame_data(relative_acceleration=False)
    torch.testing.assert_close(relative_link_linear_acceleration(data, ids, ref), expected, atol=1e-12, rtol=1e-12)


def test_known_relative_acceleration_includes_waist_or_arm_motion_and_both_origin_shifts():
    data, ids, ref, expected, wrist_acc = moving_frame_data()
    torch.testing.assert_close(link_origin_linear_acceleration_w(data, ids), wrist_acc, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(relative_link_linear_acceleration(data, ids, ref), expected, atol=1e-12, rtol=1e-12)
    # Unselected torso telemetry cannot affect this measurement.
    data.body_com_acc_w.torch[:, 1] = 1e6
    torch.testing.assert_close(relative_link_linear_acceleration(data, ids, ref), expected, atol=1e-12, rtol=1e-12)


def test_static_reference_agrees_with_world_acceleration_cost():
    data, ids, ref, _, wrist_acc = moving_frame_data(rotating=False, translating=False)
    relative = relative_link_linear_acceleration(data, ids, ref)
    torch.testing.assert_close(relative.square().sum((1, 2)), wrist_acc.square().sum((1, 2)))


def test_common_world_rotation_and_translation_do_not_change_relative_acceleration():
    data, ids, ref, expected, _ = moving_frame_data()
    q = quat_from_euler_xyz(*torch.tensor([[0.4], [-0.7], [1.2]], dtype=torch.float64))
    rotation = q[:, None].expand(3, 4, -1)
    for name in ("body_link_pose_w", "body_com_pose_w"):
        pose = getattr(data, name).torch
        pose[..., :3] = quat_apply(rotation, pose[..., :3]) + torch.tensor([100.0, -200.0, 30.0])
        pose[..., 3:] = quat_mul(rotation, pose[..., 3:])
    for name in ("body_link_vel_w", "body_com_vel_w", "body_com_acc_w"):
        value = getattr(data, name).torch
        value[..., :3] = quat_apply(rotation, value[..., :3])
        value[..., 3:] = quat_apply(rotation, value[..., 3:])
    torch.testing.assert_close(relative_link_linear_acceleration(data, ids, ref), expected, atol=1e-11, rtol=1e-11)
