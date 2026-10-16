# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only, sample-weighted diagnostics, independent of curriculum gate buffers."""

from __future__ import annotations

import torch

MODE_NAMES = ("stand", "squat", "walk", "squat_walk")
STEP_FIELDS = ("height_offset_m", "command_speed_xy_mps", "command_yaw_abs_radps")
EPISODE_FIELDS = (
    "episode_length_s",
    "survival_fraction",
    "failure_rate",
    "timeout_rate",
    "height_tracking_success_rate",
    "velocity_tracking_success_rate",
    "joint_success_rate",
    "error_height_m",
    "error_velocity_xy_mps",
    "error_yaw_radps",
)


class ModePerformanceMonitor:
    """Accumulate each real step and finalized episode once; drain once per logging interval.

    Partial, manual and mixed-mode episodes are excluded from episode statistics, but their
    executed steps remain in the measured sample allocation. Draining never clears live
    episode accumulators. This object does not sample randomness or change training state.
    """

    def __init__(self, num_envs: int, device: str, step_dt: float, thresholds: tuple[float, float, float, float]):
        self.step_dt = step_dt
        self.thresholds = thresholds
        self.episode = torch.zeros((num_envs, 7), device=device)
        self.start_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.start_mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.observed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.mixed = torch.zeros_like(self.observed)
        self.reported = torch.zeros_like(self.observed)
        self.steps = torch.zeros((4, 4), device=device)
        self.episodes = torch.zeros((4, 1 + len(EPISODE_FIELDS)), device=device)
        self.pelvis = torch.zeros((4, 3), device=device)
        self.events = torch.zeros((4, 2), device=device)
        self.terminations: dict[str, torch.Tensor] = {}
        self.excluded = torch.zeros(3, device=device)

    def reset(self, env_ids: torch.Tensor, mode: torch.Tensor, step: int) -> None:
        self.episode[env_ids] = 0
        self.start_step[env_ids] = step
        self.start_mode[env_ids] = mode[env_ids]
        self.observed[env_ids] = True
        self.mixed[env_ids] = False
        self.reported[env_ids] = False

    def sample(self, mode, valid, height_error, xy_error, yaw_error, pelvis_error, command) -> None:
        """Consume the command term's existing once-per-executed-step mask."""
        still = (mode < 2) & valid
        values = torch.stack(
            (
                valid.float(),
                torch.where(valid, height_error, 0),
                torch.where(valid, xy_error, 0),
                torch.where(valid, yaw_error, 0),
                torch.where(still, pelvis_error, 0),
                still.float(),
                (still & (pelvis_error < self.thresholds[3])).float(),
            ),
            dim=1,
        )
        self.episode.add_(values)
        self.mixed |= valid & (mode != self.start_mode)
        context = torch.stack((command[:, 0], command[:, 1:3].norm(dim=1), command[:, 3].abs()), dim=1)
        self.steps.index_add_(0, mode, torch.cat((valid[:, None].float(), context * valid[:, None]), dim=1))

    def finish(self, env_ids, *, step, lengths, terminated, time_outs, reasons, max_steps) -> None:
        """Use accumulated terminal samples and capture raw terms before termination-manager reset."""
        ids = env_ids[~self.reported[env_ids]]
        count = self.episode[ids, 0]
        has_samples = count > 0
        natural = terminated[ids] | time_outs[ids]
        full = self.observed[ids] & (count == lengths[ids]) & (count == step - self.start_step[ids])
        mixed = self.mixed[ids]
        self.excluded.add_(
            torch.stack(
                (
                    (has_samples & ~natural).sum(),
                    (has_samples & natural & ~full).sum(),
                    (has_samples & natural & full & mixed).sum(),
                )
            )
        )
        ids = ids[has_samples & natural & full & ~mixed]
        sums = self.episode[ids]
        count = sums[:, 0]
        means = sums[:, 1:4] / count[:, None]
        height_ok = means[:, 0] < self.thresholds[0]
        velocity_ok = (means[:, 1] < self.thresholds[1]) & (means[:, 2] < self.thresholds[2])
        complete = time_outs[ids] & ~terminated[ids]
        pelvis_mean = sums[:, 4] / sums[:, 5].clamp_min(1)
        pelvis_ok = (sums[:, 5] == 0) | (pelvis_mean < self.thresholds[3])
        values = torch.stack(
            (
                torch.ones_like(count),
                count * self.step_dt,
                (count / max_steps).clamp(max=1),
                terminated[ids].float(),
                complete.float(),
                height_ok.float(),
                velocity_ok.float(),
                (complete & height_ok & velocity_ok & pelvis_ok).float(),
                means[:, 0],
                means[:, 1],
                means[:, 2],
            ),
            dim=1,
        )
        modes = self.start_mode[ids]
        self.episodes.index_add_(0, modes, values)
        self.pelvis.index_add_(0, modes, sums[:, [5, 4, 6]])
        for name, active in reasons.items():
            if name not in self.terminations:
                self.terminations[name] = torch.zeros_like(self.events[:, 0])
            self.terminations[name].index_add_(0, modes, active[ids].float())
        self.reported[env_ids] = True

    def disturbance(self, mode, env_ids, arm_fraction: float) -> None:
        """Count actual upper-body event firings, rather than inferred exposure from survival."""
        if env_ids is None or isinstance(env_ids, slice):
            selection = slice(None) if env_ids is None else env_ids
            env_ids = torch.arange(len(mode), device=mode.device)[selection]
        values = torch.ones((len(env_ids), 2), device=self.events.device)
        values[:, 1] *= arm_fraction
        self.events.index_add_(0, mode[env_ids], values)

    def drain(self, probabilities: torch.Tensor) -> dict[str, float]:
        """Return independent scalars, omitting undefined means instead of logging false zeros."""
        # Transfer all mode buckets together, rather than synchronizing once per metric.
        reason_names = list(self.terminations)
        reason_values = [self.terminations[name][:, None] for name in reason_names]
        rows = torch.cat((self.steps, self.episodes, self.pelvis, self.events, *reason_values), dim=1).cpu().tolist()
        probs = (probabilities / probabilities.sum()).detach().cpu().tolist()
        excluded = self.excluded.cpu().tolist()
        result = {}
        total_steps = sum(row[0] for row in rows)
        total_episodes = sum(row[4] for row in rows)
        for index, (mode, row) in enumerate(zip(MODE_NAMES, rows, strict=True)):
            prefix = f"Performance/{mode}/"
            steps, episodes = row[0], row[4]
            result[prefix + "episode_count"] = episodes
            result[prefix + "episode_valid"] = float(episodes > 0)
            result[prefix + "step_count"] = steps
            result[prefix + "sampling_probability"] = probs[index]
            if total_steps:
                result[prefix + "step_share"] = steps / total_steps
            if total_episodes:
                result[prefix + "episode_share"] = episodes / total_episodes
            if steps:
                for name, value in zip(STEP_FIELDS, row[1:4], strict=True):
                    result[prefix + name] = value / steps
            if episodes:
                for name, value in zip(EPISODE_FIELDS, row[5:15], strict=True):
                    result[prefix + name] = value / episodes
                for name, value in zip(reason_names, row[20:], strict=True):
                    result[prefix + "termination/" + name] = value / episodes
            pelvis_steps, pelvis_error, pelvis_success = row[15:18]
            if pelvis_steps:
                result[prefix + "error_pelvis_xy_m"] = pelvis_error / pelvis_steps
                result[prefix + "pelvis_step_success_rate"] = pelvis_success / pelvis_steps
            events, fraction = row[18:20]
            result[prefix + "disturbance_count"] = events
            if steps:
                result[prefix + "disturbance_hz"] = events / (steps * self.step_dt)
            if events:
                result[prefix + "disturbance_arm_fraction"] = fraction / events
        for name, value in zip(("manual_episodes", "partial_episodes", "mixed_mode_episodes"), excluded, strict=True):
            result["Performance/excluded/" + name] = value
        self.steps.zero_()
        self.episodes.zero_()
        self.pelvis.zero_()
        self.events.zero_()
        self.excluded.zero_()
        for values in self.terminations.values():
            values.zero_()
        return result
