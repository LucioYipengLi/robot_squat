"""Deterministic joint-target mapping shared by training and evaluation."""

import torch


def joint_targets_from_uniform(
    default_q: torch.Tensor,
    joint_limits: torch.Tensor,
    fraction: torch.Tensor | float,
    samples: torch.Tensor,
) -> torch.Tensor:
    """Map uniform samples to targets using each direction's available joint range.

    ``default_q`` and ``samples`` have shape (envs, joints); ``joint_limits``
    adds a final (lower, upper) axis. ``fraction`` broadcasts over these joints.
    Callers supply samples in [0, 1] on the same device and own all randomness,
    joint selection, timing and goal submission.

    For fractions in [0, 1] and defaults within limits, sample 0 selects the
    lower scaled endpoint, sample 1 the upper endpoint, and fraction 0 keeps
    the default. Negative margins are treated as zero and targets are clamped
    to the physical limits, including when a default lies outside them.
    """
    lo, hi = joint_limits[..., 0], joint_limits[..., 1]
    lower_margin = (default_q - lo).clamp_min(0.0)
    upper_margin = (hi - default_q).clamp_min(0.0)
    lower = -fraction * lower_margin
    upper = fraction * upper_margin
    # Preserve training's sample_uniform arithmetic order for seeded replay.
    noise = samples * (upper - lower) + lower
    return torch.clamp(default_q + noise, lo, hi)
