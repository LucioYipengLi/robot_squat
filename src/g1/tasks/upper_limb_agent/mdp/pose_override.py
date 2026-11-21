"""Opt-in teleop lifecycle; ordinary training never creates this helper."""

import math
from dataclasses import dataclass

import torch

__all__ = []

SNAPSHOT_FIELDS = (
    "goal_pose_w",
    "mode_id",
    "height_task_mask",
    "planar_task_mask",
    "height_shift",
    "planar_station_w",
    "planar_distance_m",
    "planar_heading_rad",
    "planar_sampling_group",
    "evaluation_start_steps",
    "episode_deadline_steps",
)


def validate_pose(value, shape, device):
    value = torch.as_tensor(value, dtype=torch.float32, device=device).clone()
    if value.shape != shape or not torch.isfinite(value).all():
        raise ValueError(f"Expected finite world wrist poses of shape {shape}")
    length = value[..., 3:].norm(dim=-1, keepdim=True)
    if (length < 1e-6).any():
        raise ValueError("Wrist orientation must be a nonzero xyzw quaternion")
    value[..., 3:] /= length
    return value


@dataclass
class PoseOverride:
    source: str
    category: str
    arm_mode: int | None
    replay: dict | None = None
    pending: tuple | None = None

    def prepare_reset(self, term, ids):
        term.planar_task_mask[ids] = False
        term.height_task_mask[ids] = False
        term.height_shift[ids] = 0
        term.planar_station_w[ids] = 0
        term.planar_distance_m[ids] = 0
        term.planar_heading_rad[ids] = 0
        term._episode_start_step[ids] = term._env.common_step_counter
        term.evaluation_start_steps[ids] = 0
        timing = term.cfg.planar_task.timing if term.cfg.planar_task is not None else None
        seconds = timing.near_episode_s if timing is not None else term._env.cfg.episode_length_s
        term.episode_deadline_steps[ids] = math.ceil(seconds / term._env.step_dt - 1e-9)
        self.pending = None
        if self.replay is not None:
            for name in SNAPSHOT_FIELDS:
                getattr(term, name)[ids] = self.replay[name][ids].to(term.device)
            return
        if self.arm_mode is None:
            probabilities = torch.tensor(term.cfg.mode_probabilities, device=term.device)
            term.mode_id[ids] = torch.multinomial(probabilities, len(ids), replacement=True)
        else:
            term.mode_id[ids] = self.arm_mode
        if self.source == "auto":
            if self.category == "mixed":
                term._sample_task_kind(ids)
            elif self.category == "height_shifted":
                term._sample_height_task(ids, force=True)
            elif self.category == "planar":
                term.planar_task_mask[ids] = True

    def resample(self, term, ids):
        # One goal per teleop episode, including regular tasks. A forced timer
        # expiry must not silently replace a target the operator is inspecting.
        fresh = ids[term.command_counter[ids] == 0]
        if len(fresh):
            if self.replay is not None:
                self.replay = None  # Reuse once; subsequent automatic episodes sample anew.
            elif self.source == "manual":
                term.goal_pose_w[fresh] = term.robot.data.body_link_pose_w.torch[fresh][:, term.body_indices]
            else:
                term._resample_command(fresh)
            term.command_counter[fresh] += 1
        term.time_left[ids] = torch.inf

    def apply_pending(self, term):
        if self.pending is not None:
            poses, arm_mode = self.pending
            term.goal_pose_w.copy_(poses)
            term.mode_id.fill_(arm_mode)
            self.pending = None

    @staticmethod
    def snapshot(term):
        return {name: getattr(term, name).detach().clone() for name in SNAPSHOT_FIELDS}
