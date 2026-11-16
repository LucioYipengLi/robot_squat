"""Shared pose-window statistics and duration-normalized failure accounting."""

import torch

__all__ = []


def failure_per_10s(failures, elapsed_s):
    """Failure occurrences per ten aggregate robot-seconds, not a probability."""
    return 10 * failures / elapsed_s.clamp_min(1e-9)


def simultaneous_pose_ready(position, orientation, active, position_tolerance, orientation_tolerance):
    return (((position <= position_tolerance) & (orientation <= orientation_tolerance)) | ~active).all(-1) & active.any(
        -1
    )


class PoseWindowMetrics:
    """Observed sums only; the caller supplies the full assigned-window denominator.

    Columns: joint pose readiness, position readiness, mean active-arm position
    error, mean active-arm angle error, observed samples. Missing/failed time
    contributes zero readiness, and never fabricates zero-valued error samples.
    """

    def __init__(self, num_envs, device):
        self.sums = torch.zeros(num_envs, 5, device=device)
        self.last_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)

    def record(self, ids, step, valid, position, orientation, active, pose_ready, position_ready):
        valid = valid & (self.last_step[ids] != step)
        arms = active.sum(-1).clamp_min(1)
        values = torch.stack(
            (
                pose_ready,
                position_ready,
                (position * active).sum(-1) / arms,
                (orientation * active).sum(-1) / arms,
                torch.ones_like(arms),
            ),
            dim=-1,
        )
        self.sums[ids] += values * valid[:, None]
        self.last_step[ids] = step

    def quality(self, ids, window_steps):
        return self.sums[ids, 0] / window_steps.clamp_min(1)

    def reset(self, ids):
        self.sums[ids] = 0
        self.last_step[ids] = -1
