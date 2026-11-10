"""Instantaneous link-origin kinematics, independent of reward and action state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData

__all__ = ["link_origin_linear_acceleration_w", "relative_link_linear_acceleration"]


def link_origin_linear_acceleration_w(data: BaseArticulationData, body_ids: list[int]) -> torch.Tensor:
    """World linear acceleration at each selected link origin [m/s²], shape (envs, links, 3).

    Physics reports COM acceleration. For the rigid COM-to-origin offset r,
    a_origin = a_com + alpha x r + omega x (omega x r).
    """
    acceleration = data.body_com_acc_w.torch[:, body_ids]
    omega = data.body_com_vel_w.torch[:, body_ids, 3:]
    offset = data.body_link_pose_w.torch[:, body_ids, :3] - data.body_com_pose_w.torch[:, body_ids, :3]
    return (
        acceleration[..., :3]
        + torch.linalg.cross(acceleration[..., 3:], offset)
        + torch.linalg.cross(omega, torch.linalg.cross(omega, offset))
    )


def relative_link_linear_acceleration(
    data: BaseArticulationData, body_ids: list[int], reference_body_id: int
) -> torch.Tensor:
    """Second derivative of link positions in the moving reference frame [m/s²].

    Returns (envs, links, 3). With r = p_link - p_ref in world coordinates,
    u = v_link - v_ref - omega_ref x r is the relative velocity expressed in
    world axes. The reference-frame acceleration is
    R_ref.T @ (a_link - a_ref - alpha_ref x r
               - omega_ref x (omega_ref x r) - 2 * omega_ref x u).

    Both acceleration origins match the link-origin positions and velocities.
    Common rigid translation/rotation contributes zero; waist and arm motion
    relative to the pelvis still contributes. This reads current physics only.
    """
    ids = [reference_body_id, *body_ids]
    pose = data.body_link_pose_w.torch[:, ids]
    velocity = data.body_link_vel_w.torch[:, ids]
    acceleration = link_origin_linear_acceleration_w(data, ids)
    alpha = data.body_com_acc_w.torch[:, reference_body_id : reference_body_id + 1, 3:]
    omega = velocity[:, :1, 3:]
    offset = pose[:, 1:, :3] - pose[:, :1, :3]
    relative_velocity_w = velocity[:, 1:, :3] - velocity[:, :1, :3] - torch.linalg.cross(omega, offset)
    relative_acceleration_w = (
        acceleration[:, 1:]
        - acceleration[:, :1]
        - torch.linalg.cross(alpha, offset)
        - torch.linalg.cross(omega, torch.linalg.cross(omega, offset))
        - 2.0 * torch.linalg.cross(omega, relative_velocity_w)
    )
    reference_orientation = pose[:, :1, 3:].expand(-1, len(body_ids), -1)
    return quat_apply_inverse(reference_orientation, relative_acceleration_w)
