"""Compare startup interventions that hold arm or whole upper-body references."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from g1.tasks.upper_limb_agent.mdp.actions import UPPER_BODY_JOINT_NAMES, WAIST_JOINT_NAMES
from g1.tasks.upper_limb_agent.mdp.contacts import ARM_CONTACT_BODY_PATHS
from g1.tasks.upper_limb_agent.mdp.planar_targets import PLANAR_SAMPLING_GROUPS
from g1.tasks.upper_limb_agent.symmetry import ACTION_TERMS


def _yaw(pose):
    x, y, z, w = pose[..., 3:].T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _cross_body_distance(trace, pelvis_id, arm_mode):
    pelvis = trace["body_pose"][0, pelvis_id]
    dx, dy = (trace["goal_pose"][0, :, :2] - pelvis[:2]).T
    yaw = _yaw(pelvis)
    lateral = -np.sin(yaw) * dx + np.cos(yaw) * dy
    return float(max((-lateral[0], lateral[1])[side] for side in ((0,), (1,), (0, 1))[arm_mode]))


def _wide_episodes(directory):
    with (directory / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            episode = json.loads(line)
            if episode.get("planar_sampling_group") == "wide":
                yield episode


class PlanarArmHold:
    """The original first batch recreates the initial conditions by environment ID."""

    def __init__(self, directory, *, hold_waist=False):
        self.hold_waist = hold_waist
        self.arms_only_path = directory.resolve(strict=True) if hold_waist else None
        if self.arms_only_path is not None:
            arms_only_manifest = json.loads((self.arms_only_path / "run.json").read_text())
            arms_only_settings = arms_only_manifest.get("planar_hold")
            if not arms_only_settings or arms_only_settings.get("hold_waist"):
                raise ValueError("--planar-hold-waist-from requires an arms-only hold archive")
            directory = Path(arms_only_settings["baseline"])
        self.directory = directory.resolve(strict=True)
        self.baseline = json.loads((self.directory / "run.json").read_text())
        settings = self.baseline.get("planar_startup")
        if (
            not settings
            or self.baseline.get("planar_hold")
            or self.baseline["planar_sampling"] != "transition"
            or self.baseline["policy_output"] != "mean"
            or self.baseline.get("planar_near_lower_scale") is not None
        ):
            raise ValueError("Use an unmodified transition/mean --planar-startup archive as the baseline")
        self.hold_s = float(settings["window_s"])
        self.hold_steps = math.ceil(self.hold_s / self.baseline["control_dt"])
        self.pelvis_id = self.baseline["body_names"].index("pelvis")
        partners = self.baseline["partners"]
        self.arm_channels = [partners.index(name) for name in ARM_CONTACT_BODY_PATHS]
        self.shoulder_channels = [channel for channel in self.arm_channels if "shoulder" in partners[channel]]
        self.threshold = self.baseline["terminations"]["base_contact"]
        self.held_joint_start = 0 if hold_waist else len(WAIST_JOINT_NAMES)
        self.replay_width = sum(width for _, width in ACTION_TERMS[:-1]) + self.held_joint_start
        self.replay_match_key = "lower_replay_match" if hold_waist else "lower_and_waist_replay_match"

        self.cases = {}
        for episode in _wide_episodes(self.directory):
            env_id = episode["env_id"]
            trace = self._load_trace(self.directory, episode)
            cross_m = _cross_body_distance(trace, self.pelvis_id, episode["arm_mode"])
            if not episode["failed"] and cross_m <= 0.10:
                continue
            if env_id in self.cases:
                raise ValueError(f"Duplicate baseline startup trace for env_id {env_id}")
            self.cases[env_id] = episode, trace, cross_m
        if not self.cases:
            raise ValueError("The baseline contains no wide failures or successful cross-body controls")
        self.results = {}
        self.arms_only_cases = {}
        if self.arms_only_path is not None:
            self.validate_run(arms_only_manifest)
            if (
                set(arms_only_settings["env_ids"]) != self.cases.keys()
                or arms_only_settings["control_steps"] != self.hold_steps
            ):
                raise ValueError("The arms-only archive must match the original startup cases and hold window")
            for episode in _wide_episodes(self.arms_only_path):
                env_id = episode["env_id"]
                if env_id not in self.cases:
                    continue
                if env_id in self.arms_only_cases:
                    raise ValueError(f"Duplicate arms-only startup trace for env_id {env_id}")
                trace = self._load_trace(self.arms_only_path, episode)
                original, original_trace, _ = self.cases[env_id]
                if episode["arm_mode"] != original["arm_mode"] or any(
                    not np.allclose(trace[key][0], original_trace[key][0], rtol=0, atol=1e-4)
                    for key in ("goal_pose", "body_pose", "joint_pos", "joint_vel")
                ):
                    raise ValueError(f"Arms-only reset state or target differs for env_id {env_id}")
                self.arms_only_cases[env_id] = episode, trace
            if missing := self.cases.keys() - self.arms_only_cases.keys():
                raise ValueError(f"Missing arms-only startup traces for env_ids {sorted(missing)}")

    def _load_trace(self, directory, episode):
        with np.load(directory / episode["planar_startup_clip"], allow_pickle=False) as clip:
            trace = dict(clip)
        steps = len(trace["time"]) - 1
        env_id = episode["env_id"]
        if not 1 <= steps <= self.hold_steps:
            raise ValueError(f"Invalid first-episode startup trace for env_id {env_id}")
        if not np.allclose(trace["time"], np.arange(steps + 1) * self.baseline["control_dt"], atol=1e-6):
            raise ValueError(f"Missing control frames in startup trace for env_id {env_id}")
        if steps < self.hold_steps and not np.isclose(episode["elapsed_s"], trace["time"][-1], atol=1e-6):
            raise ValueError(f"Incomplete startup trace for env_id {env_id}")
        return trace

    def settings(self):
        return {
            "baseline": str(self.directory),
            "hold_s": self.hold_s,
            "control_steps": self.hold_steps,
            "cross_body_control_min_m": 0.10,
            "env_ids": sorted(self.cases),
            "hold_waist": self.hold_waist,
            "comparison_archive": str(self.arms_only_path or self.directory),
        }

    def validate_run(self, manifest):
        keys = (
            "task",
            "checkpoint_sha256",
            "seed",
            "policy_output",
            "level",
            "num_envs",
            "planar_sampling",
            "control_dt",
            "physics_dt",
            "physics_substeps",
            "body_names",
            "joint_names",
            "partners",
            "terminations",
            "torso_contact_policy",
            "source_usd",
            "variants",
        )
        mismatch = [key for key in keys if manifest[key] != self.baseline[key]]
        if mismatch:
            raise ValueError(f"Planar arm hold must match the baseline on {', '.join(mismatch)}")

    def prepare(self, env, term, motion):
        manager = env.action_manager
        arm_action = manager.get_term("joint_pos")
        layout = tuple(zip(manager.active_terms, manager.action_term_dim, strict=True))
        if layout != ACTION_TERMS or tuple(arm_action._joint_names) != UPPER_BODY_JOINT_NAMES:
            raise ValueError("The replay requires the Upper Mobile Lower, waist and arm action layout")

        self.env_ids = torch.tensor(sorted(self.cases), device=env.device)
        mode_ids = term.mode_id[self.env_ids].tolist()
        sampling_groups = term.planar_sampling_group[self.env_ids].tolist()
        fields = ("goal_pose", "body_pose", "joint_pos", "joint_vel")
        initial = {key: motion[key][self.env_ids].detach().cpu().numpy() for key in fields}
        commands = []
        for index, env_id in enumerate(sorted(self.cases)):
            episode, trace, _ = self.cases[env_id]
            if (
                mode_ids[index] != episode["arm_mode"]
                or PLANAR_SAMPLING_GROUPS[sampling_groups[index]] != "wide"
                or any(not np.allclose(initial[key][index], trace[key][0], rtol=0, atol=1e-4) for key in fields)
            ):
                raise ValueError(f"Reset state or target differs from the baseline for env_id {env_id}")
            recorded = trace["action"][1:, : self.replay_width]
            padded_indices = np.minimum(np.arange(self.hold_steps), len(recorded) - 1)
            commands.append(recorded[padded_indices])
        self.commands = torch.as_tensor(np.stack(commands, axis=1), device=env.device)

        # Invert the position action transform so the requested target equals the actual reset pose.
        joint_pos = motion["joint_pos"][:, arm_action._joint_ids]
        references = (joint_pos - arm_action._offset) / arm_action._scale
        self.hold_actions = references[self.env_ids, self.held_joint_start :]

    def intervene(self, actions, step, pending):
        if step >= self.hold_steps:
            return actions
        active = pending[self.env_ids]
        ids = self.env_ids[active]
        actions = actions.clone()
        actions[ids, : self.replay_width] = self.commands[step, active]
        actions[ids, self.replay_width :] = self.hold_actions[active]
        return actions

    def _metrics(self, episode, trace, stop):
        force = np.linalg.norm(trace["pair_force"][1:stop], axis=-1)
        arm_peak = float(force[..., self.arm_channels].sum(axis=-1).max(initial=0))
        shoulder_peak = float(force[..., self.shoulder_channels].sum(axis=-1).max(initial=0))
        yaw = np.unwrap(_yaw(trace["body_pose"][:stop, self.pelvis_id]))
        signed_yaw = np.sign(episode["planar_heading_rad"]) * (yaw - yaw[0])
        return {
            "max_reverse_yaw_deg": float(np.rad2deg(-signed_yaw.min())),
            "arm_peak_n": arm_peak,
            "shoulder_peak_n": shoulder_peak,
            "over_hard_threshold": arm_peak > self.threshold,
            "failed_by_window_end": episode["failed"] and episode["elapsed_s"] <= trace["time"][stop - 1] + 1e-6,
        }

    def add_episode(self, episode, trace):
        env_id = episode["env_id"]
        if env_id not in self.cases:
            return
        original, original_trace, cross_m = self.cases[env_id]
        reference, baseline = self.arms_only_cases.get(env_id, (original, original_trace))
        stop = min(len(baseline["time"]), len(trace["time"]))
        commands_match = np.allclose(baseline["lower_command"][1:stop], trace["lower_command"][1:stop], atol=1e-5)
        actions_match = np.array_equal(
            baseline["action"][1:stop, : self.replay_width], trace["action"][1:stop, : self.replay_width]
        )
        self.results[env_id] = {
            "baseline_episode_id": reference["episode_id"],
            "hold_episode_id": episode["episode_id"],
            "cross_body_m": cross_m,
            "arm_mode": original["arm_mode"],
            "station_distance_m": original["planar_distance_m"],
            "station_heading_deg": math.degrees(original["planar_heading_rad"]),
            "matched_window_s": float(baseline["time"][stop - 1]),
            "covered_baseline_window": stop == len(baseline["time"]),
            self.replay_match_key: bool(commands_match and actions_match),
            "baseline": self._metrics(reference, baseline, stop),
            "hold": self._metrics(episode, trace, stop),
            "baseline_failed": reference["failed"],
            "baseline_elapsed_s": reference["elapsed_s"],
            "hold_failed": episode["failed"],
            "hold_elapsed_s": episode["elapsed_s"],
            "hold_reasons": episode["reasons"],
            "hold_first_near_s": episode["planar_first_near_s"],
            "hold_pose_ready_fraction": episode["planar_eval_pose_ready_fraction"],
        }

    def report(self):
        cases = list(self.results.values())
        comparable = [r for r in cases if r[self.replay_match_key] and r["covered_baseline_window"]]
        baseline_failed = (
            sum(episode["failed"] for episode, _ in self.arms_only_cases.values())
            if self.hold_waist
            else sum(episode["failed"] for episode, _, _ in self.cases.values())
        )
        return {
            "comparison_group": "arms_only" if self.hold_waist else "original",
            "selected_cases": len(self.cases),
            "selected_baseline_failures": baseline_failed,
            "completed_cases": len(cases),
            "comparable_cases": len(comparable),
            "shorter_hold_windows": sum(not r["covered_baseline_window"] for r in cases),
            ("lower_replay_mismatches" if self.hold_waist else "lower_or_waist_replay_mismatches"): sum(
                not r[self.replay_match_key] for r in cases
            ),
            "comparable_baseline_hard_contacts": sum(r["baseline"]["over_hard_threshold"] for r in comparable),
            "comparable_hold_hard_contacts": sum(r["hold"]["over_hard_threshold"] for r in comparable),
            "hold_failures_during_hold": sum(
                r["hold_failed"] and r["hold_elapsed_s"] <= self.hold_s + 1e-6 for r in cases
            ),
            "hold_failures_after_release": sum(
                r["hold_failed"] and r["hold_elapsed_s"] > self.hold_s + 1e-6 for r in cases
            ),
            "hold_near_entries": sum(r["hold_first_near_s"] is not None for r in cases),
            "hold_full_pose_ready": sum(r["hold_pose_ready_fraction"] == 1.0 and not r["hold_failed"] for r in cases),
            "by_env_id": self.results,
        }
