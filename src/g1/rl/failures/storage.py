"""Portable, non-pickle failure archives and bounded device-side histories."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value, *, indent=2):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, allow_nan=False, indent=indent), encoding="utf-8")
    temporary.replace(path)


class TensorHistory:
    """One shared cursor; per-environment lengths prevent clips crossing reset boundaries."""

    def __init__(self, capacity, num_envs):
        self.capacity, self.num_envs = capacity, num_envs
        self.cursor = 0
        self.buffers = {}
        self.lengths = None

    def push(self, frame):
        import torch

        if not self.buffers:
            self.buffers = {k: v.new_empty((self.capacity, *v.shape)) for k, v in frame.items()}
            self.lengths = torch.zeros(self.num_envs, dtype=torch.long, device=next(iter(frame.values())).device)
        for key, value in frame.items():
            self.buffers[key][self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.lengths.add_(1).clamp_(max=self.capacity)

    def reset(self, ids):
        if self.lengths is not None:
            self.lengths[ids] = 0

    def extract(self, env_id):
        import torch

        count = int(self.lengths[env_id])
        indices = (torch.arange(count, device=self.lengths.device) + self.cursor - count) % self.capacity
        return {k: v[indices, env_id].detach().cpu().numpy().copy() for k, v in self.buffers.items()}


def contact_statistics(trace, partners, physics_dt, threshold=1.0):
    """Count disjoint physics samples once, retaining cancellation and coverage diagnostics."""
    force = np.linalg.norm(trace["pair_force"], axis=-1).reshape(-1, len(partners))
    net = np.linalg.norm(trace["net_force"], axis=-1).reshape(-1)
    residual = trace["net_force"] - trace["pair_force"].sum(axis=-2)
    result = []
    for index, name in enumerate(partners):
        values = force[:, index]
        touching = values > threshold
        result.append(
            {
                "partner": name,
                "peak_normal_force_n": float(values.max(initial=0)),
                "normal_impulse_ns": float(values.sum() * physics_dt),
                "time_above_1n_s": float(touching.sum() * physics_dt),
                "contact_onsets": int(np.count_nonzero(touching & ~np.r_[False, touching[:-1]])),
            }
        )
    result.sort(key=lambda item: item["peak_normal_force_n"], reverse=True)
    return {
        "partners": result,
        "peak_net_normal_force_n": float(net.max(initial=0)),
        "peak_unattributed_normal_force_n": float(np.linalg.norm(residual, axis=-1).max(initial=0)),
        "duration_is_censored_at_failure": True,
    }


class FailureArchive:
    def __init__(self, directory, manifest, *, planar_hold=None):
        self.planar_hold = planar_hold
        if planar_hold is not None:
            planar_hold.validate_run(manifest)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "clips").mkdir()
        if manifest.get("planar_startup") is not None:
            (self.directory / "startup").mkdir()
        self.manifest = {"schema_version": 1, **manifest}
        self.completed = self.failed = self.saved = 0
        self.classes, self.reasons = Counter(), Counter()
        self.failed_classes, self.elapsed_classes = Counter(), Counter()
        self.planar_groups = {}
        self._episodes = (self.directory / "episodes.jsonl").open("w", encoding="utf-8", buffering=1)
        self._failures = (self.directory / "failures.jsonl").open("w", encoding="utf-8", buffering=1)
        self._switches = self.switch_summary = None
        self.pair_summary = None
        if (paired := manifest.get("switch_pair_evaluation")) is not None:
            from .pairs import PairedSwitchSummary

            self.pair_summary = PairedSwitchSummary(paired["cases"], dual=paired.get("trial_mode") == "dual_arm")
        if manifest.get("regular_switch_evaluation") is not None:
            from .switches import SwitchSummary

            self._switches = (self.directory / "switches.jsonl").open("w", encoding="utf-8", buffering=1)
            self.switch_summary = SwitchSummary()
        write_json(self.directory / "run.json", self.manifest)
        self.progress("starting")

    def _write_trace(self, clip, trace):
        temporary = self.directory / f"{clip}.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **trace)
        temporary.replace(self.directory / clip)

    def add_episode(self, episode, trace=None, *, switches=(), startup_trace=None):
        episode = {"episode_id": self.completed, **episode}
        if startup_trace is not None:
            episode["planar_startup_clip"] = f"startup/{episode['env_id']:06d}.npz"
            self._write_trace(episode["planar_startup_clip"], startup_trace)
            if self.planar_hold is not None:
                self.planar_hold.add_episode(episode, startup_trace)
        if trace is not None:
            sample_id = f"{self.saved:06d}"
            clip = f"clips/{sample_id}.npz"
            self._write_trace(clip, trace)
            sample = {"sample_id": sample_id, "clip": clip, **episode}
            self._failures.write(json.dumps(plain(sample), ensure_ascii=False, allow_nan=False) + "\n")
            self.saved += 1
        self._episodes.write(json.dumps(plain(episode), ensure_ascii=False, allow_nan=False) + "\n")
        if self._switches is not None and episode["category"] == "regular":
            self._switches.writelines(
                json.dumps(plain({"episode_id": self.completed, "level": episode["level"], **row}), allow_nan=False)
                + "\n"
                for row in switches
            )
            self.switch_summary.add_episode(switches, episode["failed"])
        if self.pair_summary is not None:
            self.pair_summary.add(episode, switches)
        self.completed += 1
        self.failed += int(episode["failed"])
        self.classes[episode["category"]] += 1
        self.failed_classes[episode["category"]] += int(episode["failed"])
        self.elapsed_classes[episode["category"]] += episode["elapsed_s"]
        self.reasons.update(episode["reasons"])
        if (name := episode.get("planar_sampling_group")) is not None:
            group = self.planar_groups.setdefault(name, Counter())
            group["episodes"] += 1
            group["failures"] += int(episode["failed"])
            group["robot_seconds"] += episode["elapsed_s"]
            group["near_entries"] += int(episode["planar_entered_near"])
            group["failures_after_near"] += int(episode["planar_entered_near"] and episode["failed"])
            group["near_seconds"] += episode["planar_first_near_s"] or 0.0
            group["pose_ready_sum"] += episode["planar_eval_pose_ready_fraction"]

    def progress(self, status, **extra):
        if self.switch_summary is not None:
            extra["regular_switches"] = self.switch_summary.report()
        if self.pair_summary is not None:
            extra["switch_pair_evaluation"] = self.pair_summary.report()
        if self.planar_hold is not None:
            extra["planar_hold"] = self.planar_hold.report()
        if self.planar_groups:
            extra["planar_groups"] = {
                name: {
                    "episodes": group["episodes"],
                    "failures": group["failures"],
                    "failure_per_10s": 10 * group["failures"] / max(group["robot_seconds"], 1e-9),
                    "near_entry_fraction": group["near_entries"] / group["episodes"],
                    "failures_after_near": group["failures_after_near"],
                    "failure_after_near_fraction": group["failures_after_near"] / group["near_entries"]
                    if group["near_entries"]
                    else None,
                    "first_near_s": group["near_seconds"] / group["near_entries"] if group["near_entries"] else None,
                    "eval_pose_ready_fraction": group["pose_ready_sum"] / group["episodes"],
                }
                for name, group in self.planar_groups.items()
            }
        write_json(
            self.directory / "summary.json",
            {
                "status": status,
                "completed_episodes": self.completed,
                "failed_episodes": self.failed,
                "saved_failures": self.saved,
                "episodes_by_category": dict(self.classes),
                "termination_counts": dict(self.reasons),
                "completed_episode_robot_seconds": dict(self.elapsed_classes),
                "failure_per_10s_by_category": {
                    name: 10 * self.failed_classes[name] / max(seconds, 1e-9)
                    for name, seconds in self.elapsed_classes.items()
                },
                **extra,
            },
        )

    def close(self):
        self._episodes.close()
        self._failures.close()
        if self._switches is not None:
            self._switches.close()
