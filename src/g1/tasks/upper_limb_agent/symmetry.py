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
# Hip pitch/roll/yaw, knee, ankle pitch/roll use Y/X/Z/Y/Y/X axes.
LEG = (1, -1, -1, 1, 1, -1)
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
FULL_OBS_TERMS = tuple(
    (name, 29 if name in {"joint_pos", "joint_vel"} else width) for name, width in OBS_TERMS if name != "base_height"
)
OBS_HISTORY = 5
LEGACY_OBSERVATION_DIM = sum(OBS_DIMS)
FULL_OBSERVATION_DIM = OBS_HISTORY * sum(width for _, width in FULL_OBS_TERMS)


def _sign(values, signs):
    return values * values.new_tensor(signs)


def _swap_pair(values, signs):
    return _sign(values.unflatten(-1, (2, len(signs))).flip(-2), signs).flatten(-2)


def mirror_upper_joints(values):
    waist, arms = values.split((len(WAIST), 2 * len(ARM)), dim=-1)
    return torch.cat((_sign(waist, WAIST), _swap_pair(arms, ARM)), dim=-1)


def _mirror_observed_joints(values):
    upper = mirror_upper_joints(values[..., :17])
    if values.shape[-1] == 17:
        return upper
    return torch.cat((upper, _swap_pair(values[..., 17:], LEG)), dim=-1)


def mirror_mobile_policy(policy):
    terms, history = _policy_layout(policy.shape[-1])
    widths = tuple(width for _, width in terms)
    blocks = policy.split(tuple(width * history for width in widths), dim=-1)
    targets, active, wrists, errors, pos, vel, previous, lin, ang, gravity, *tail = (
        block.unflatten(-1, (history, width)) for block, width in zip(blocks, widths, strict=True)
    )
    *height, cmd, height_cmd, pelvis = tail
    reflected = (
        _swap_pair(targets, POSE),
        active.flip(-1),
        _swap_pair(wrists, POSE),
        _swap_pair(errors, POSE_ERROR),
        _mirror_observed_joints(pos),
        _mirror_observed_joints(vel),
        mirror_upper_joints(previous),
        _sign(lin, POLAR),
        _sign(ang, AXIAL),
        _sign(gravity, POLAR),
        *height,
        _sign(cmd, VELOCITY_COMMAND),
        height_cmd,
        _sign(pelvis, PELVIS_OFFSET),
    )
    return torch.cat(tuple(block.flatten(-2) for block in reflected), dim=-1)


def _policy_layout(width):
    if width == LEGACY_OBSERVATION_DIM:
        return OBS_TERMS, 1
    if width == FULL_OBSERVATION_DIM:
        return FULL_OBS_TERMS, OBS_HISTORY
    raise ValueError(f"Unsupported Upper Mobile policy observation dimension: {width}")


def expand_legacy_observation_columns(weight, fill=0):
    """Move the retained input columns to the newest frame in the term-major layout."""
    if weight.ndim != 2 or weight.shape[1] != LEGACY_OBSERVATION_DIM:
        raise ValueError("Expected the standard 109-D Upper Mobile first layer")
    blocks = dict(zip((name for name, _ in OBS_TERMS), weight.split(OBS_DIMS, dim=-1), strict=True))
    expanded = weight.new_full((weight.shape[0], FULL_OBSERVATION_DIM), fill)
    offset = 0
    for name, width in FULL_OBS_TERMS:
        block = blocks[name]
        start = offset + (OBS_HISTORY - 1) * width
        expanded[:, start : start + block.shape[1]] = block
        offset += OBS_HISTORY * width
    return expanded


def mirror_mobile_actions(actions):
    velocity, height, pelvis, joints = actions.split(ACTION_DIMS, dim=-1)
    return torch.cat(
        (_sign(velocity, VELOCITY_COMMAND), height, _sign(pelvis, PELVIS_OFFSET), mirror_upper_joints(joints)), dim=-1
    )


def validate_mobile_observation_layout(raw):
    """Validate the stock observation layouts used by mirroring and checkpoint migration."""
    from .mdp.actions import MOBILE_OBSERVATION_JOINT_NAMES

    obs = raw.observation_manager
    sizes = tuple(obs.group_obs_term_dim.get("policy", ()))
    terms, history = _policy_layout(sum(dim[0] for dim in sizes if len(dim) == 1))
    names, dims = zip(*terms, strict=True)
    if (
        set(obs.active_terms) != {"policy"}
        or tuple(obs.active_terms["policy"]) != names
        or sizes != tuple((width * history,) for width in dims)
        or not obs.group_obs_concatenate["policy"]
    ):
        raise ValueError(f"Upper Mobile requires the standard {sum(dims) * history}-D policy observations")
    if history == OBS_HISTORY:
        policy = raw.cfg.observations.policy
        if policy.history_order != "term" or any(
            tuple(getattr(policy, name).params["asset_cfg"].joint_names) != MOBILE_OBSERVATION_JOINT_NAMES
            for name in ("joint_pos", "joint_vel")
        ):
            raise ValueError("Upper Mobile mirroring requires term-major history and the standard observed joint order")


def validate_mobile_mirror_interface(env):
    """Catch reordered policy tensors or an incompatible action envelope before training."""
    from .mdp.actions import UPPER_BODY_JOINT_NAMES

    raw = env.unwrapped
    validate_mobile_observation_layout(raw)
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
