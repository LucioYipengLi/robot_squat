"""Reflect the Upper Mobile observation/action interface across the pelvis XZ plane."""

import torch
from tensordict import TensorDict

# Polar vectors follow S. Angular quantities follow det(S) S. The G1 wrist
# frames are co-oriented with the pelvis at zero joints, so xyzw rotations
# reflect by the same axial signs on xyz (R -> S R S), with w unchanged.
POLAR = (1, -1, 1)
AXIAL = (-1, 1, -1)
POSE = (*POLAR, *AXIAL, 1)
POSE_ERROR = (*POLAR, *AXIAL)

# USD revolute axes in action order: waist Z X Y, arms Y X Z Y X Y Z.
# Waist stays in place; corresponding arm joints exchange sides.
WAIST = (-1, -1, 1)
ARM = (1, -1, -1, 1, -1, 1, -1)
VELOCITY_COMMAND = (1, -1, -1)  # vx, vy, yaw rate
PELVIS_OFFSET = (1, -1)

OBS_TERMS = (
    ("targets", 14),
    ("active_arms", 2),
    ("wrist_poses", 14),
    ("wrist_pose_error", 12),
    ("joint_pos", 17),
    ("joint_vel", 17),
    ("actions", 17),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("base_height", 1),
    ("lower_command", 3),
    ("height_offset", 1),
    ("pelvis_offset", 2),
)
ACTION_TERMS = (("lower_command", 3), ("height_offset", 1), ("pelvis_offset", 2), ("joint_pos", 17))
OBS_DIMS = tuple(width for _, width in OBS_TERMS)
ACTION_DIMS = tuple(width for _, width in ACTION_TERMS)


def _sign(values, signs):
    return values * values.new_tensor(signs)


def _swap_pair(values, signs):
    return _sign(values.unflatten(-1, (2, len(signs))).flip(-2), signs).flatten(-2)


def mirror_upper_joints(values):
    waist, arms = values.split((len(WAIST), 2 * len(ARM)), dim=-1)
    return torch.cat((_sign(waist, WAIST), _swap_pair(arms, ARM)), dim=-1)


def mirror_mobile_policy(policy):
    targets, active, wrists, errors, pos, vel, previous, lin, ang, gravity, height, cmd, height_cmd, pelvis = (
        policy.split(OBS_DIMS, dim=-1)
    )
    return torch.cat(
        (
            _swap_pair(targets, POSE),
            active.flip(-1),
            _swap_pair(wrists, POSE),
            _swap_pair(errors, POSE_ERROR),
            mirror_upper_joints(pos),
            mirror_upper_joints(vel),
            mirror_upper_joints(previous),
            _sign(lin, POLAR),
            _sign(ang, AXIAL),
            _sign(gravity, POLAR),
            height,
            _sign(cmd, VELOCITY_COMMAND),
            height_cmd,
            _sign(pelvis, PELVIS_OFFSET),
        ),
        dim=-1,
    )


def mirror_mobile_actions(actions):
    velocity, height, pelvis, joints = actions.split(ACTION_DIMS, dim=-1)
    return torch.cat(
        (_sign(velocity, VELOCITY_COMMAND), height, _sign(pelvis, PELVIS_OFFSET), mirror_upper_joints(joints)), dim=-1
    )


def validate_mobile_mirror_interface(env):
    """Catch reordered policy tensors or an incompatible action envelope before training."""
    from .mdp.actions import UPPER_BODY_JOINT_NAMES

    raw = env.unwrapped
    obs = raw.observation_manager
    names, dims = zip(*OBS_TERMS, strict=True)
    if (
        set(obs.active_terms) != {"policy"}
        or tuple(obs.active_terms["policy"]) != names
        or tuple(obs.group_obs_term_dim["policy"]) != tuple((width,) for width in dims)
        or not obs.group_obs_concatenate["policy"]
    ):
        raise ValueError("Upper Mobile mirror augmentation requires the standard 109-D policy observations")
    actions = raw.action_manager
    if (
        tuple(zip(actions.active_terms, actions.action_term_dim, strict=True)) != ACTION_TERMS
        or tuple(actions.get_term("joint_pos")._joint_names) != UPPER_BODY_JOINT_NAMES
    ):
        raise ValueError("Upper Mobile mirror augmentation requires the standard 23-D action and joint order")
    curriculum = raw.command_manager.get_term("wrist_pose").range_curriculum
    # Vx and height may have asymmetric bounds because they retain their sign.
    for ranges in (curriculum.initial_ranges, curriculum.final_ranges):
        reflected_axes = ranges[[2, 3, 5]]  # vy, yaw, pelvis_y
        if not torch.allclose(reflected_axes[:, 0], -reflected_axes[:, 1]):
            raise ValueError("Upper Mobile mirroring requires symmetric vy, yaw and pelvis_y command ranges")


def augment_mobile_batch(env, obs=None, actions=None):
    """RSL-RL callback returning the original mini-batch followed by its reflection."""
    if obs is not None:
        policy = obs["policy"]
        policy = torch.cat((policy, mirror_mobile_policy(policy)), dim=0)
        obs = TensorDict({"policy": policy}, batch_size=policy.shape[:1], device=policy.device)
    if actions is not None:
        actions = torch.cat((actions, mirror_mobile_actions(actions)), dim=0)
    return obs, actions
