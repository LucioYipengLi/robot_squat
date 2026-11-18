"""Bounded per-episode metrics for native regular target refreshes."""

import math
from collections import Counter, defaultdict

import numpy as np
import torch

from isaaclab.utils.math import quat_apply_inverse

from g1.tasks.upper_limb_agent.mdp.episode_metrics import simultaneous_pose_ready

CONTACT_WINDOW_S = 0.2
REACH_WINDOW_S = 1.0
CONTACT_THRESHOLD_N = 1.0
DIRECTION_THRESHOLD_M = 0.03
JUMP_BINS_M = (0.05, 0.08, 0.12)
JUMP_LABELS = ("under_5cm", "5_to_8cm", "8_to_12cm", "12cm_and_over")
ARM_MODES = ("left_only", "right_only", "both")


class RegularSwitchTracker:
    def __init__(self, env, term, hard_threshold_n):
        self.term = term
        self.dt, self.physics_dt, self.substeps = env.step_dt, env.cfg.sim.dt, env.cfg.decimation
        self.hard_threshold_n = hard_threshold_n
        self.contact_ticks = math.floor(CONTACT_WINDOW_S / self.physics_dt + 1e-8)
        self.reach_steps = math.floor(REACH_WINDOW_S / self.dt + 1e-8)
        counter = term.command_counter
        self.env_ids = torch.arange(len(counter), device=counter.device)
        self.previous_counter = torch.zeros_like(counter)
        self.previous_goals = torch.zeros_like(term.goal_pose_w)
        self.previous_ready = torch.zeros_like(counter, dtype=torch.bool)
        self.previous_contact = torch.zeros_like(self.previous_ready)
        min_steps = max(1, math.floor(term.cfg.resampling_time_range[0] / self.dt))
        shape = (len(counter), env.max_episode_length // min_steps + 1)
        self.start = counter.new_zeros(shape)
        self.observed_steps = torch.zeros_like(self.start)
        self.first_reached = torch.zeros_like(self.start)
        self.pre_ready = self.start.bool()
        self.pre_contact = self.start.bool()
        self.terminated = self.start.bool()
        self.peak = term.goal_pose_w.new_zeros(shape)
        self.goal_pairs = term.goal_pose_w.new_zeros((*shape, 2, 2, 7))
        self.delta_pelvis = term.goal_pose_w.new_zeros((*shape, 2, 3))

    def settings(self):
        cfg = self.term.range_curriculum.cfg
        return {
            "contact_window_s": CONTACT_WINDOW_S,
            "reach_window_s": REACH_WINDOW_S,
            "contact_threshold_n": CONTACT_THRESHOLD_N,
            "hard_contact_threshold_n": self.hard_threshold_n,
            "pose_position_tolerance_m": cfg.position_tolerance_m,
            "pose_orientation_tolerance_rad": cfg.orientation_tolerance_rad,
            "direction_threshold_m": DIRECTION_THRESHOLD_M,
            "position_jump_bin_edges_m": JUMP_BINS_M,
        }

    def reset(self, ids):
        self.previous_counter[ids] = 0
        self.start[ids] = 0
        self.observed_steps[ids] = 0
        self.first_reached[ids] = 0
        self.terminated[ids] = False
        self.peak[ids] = 0

    def observe(self, goals, arm_force, terminated):
        term = self.term
        counter = term.command_counter
        age = term.episode_age_steps()
        regular = term.task_ids == 0
        active = term.active_mask
        pos_error, rot_error = term.pose_error()
        cfg = term.range_curriculum.cfg
        ready = simultaneous_pose_ready(
            pos_error, rot_error, active, cfg.position_tolerance_m, cfg.orientation_tolerance_rad
        )

        # The recorder runs after physics and before reset or command refresh. The first
        # frame with a new counter was driven by that goal for one full control step.
        slot = (counter - 2).clamp_min(0)
        ids = (regular & (counter >= 2) & (counter == self.previous_counter + 1)).nonzero().flatten()
        index = (ids, slot[ids])
        old_goals = self.previous_goals[ids]
        self.start[index] = age[ids]
        self.pre_ready[index] = self.previous_ready[ids]
        self.pre_contact[index] = self.previous_contact[ids]
        self.goal_pairs[index] = torch.stack((old_goals, goals[ids]), dim=1)
        delta = goals[ids, :, :3] - old_goals[:, :, :3]
        self.delta_pelvis[index] = quat_apply_inverse(term._sampling_quat_w[ids, None, :], delta)

        index = (self.env_ids, slot)
        start = self.start[index]
        elapsed = (age - start + 1).clamp_min(0)
        observed = regular & (start > 0) & (counter >= 2)
        watch_reach = observed & (elapsed <= self.reach_steps)
        ticks = (elapsed[:, None] - 1) * self.substeps + torch.arange(1, self.substeps + 1, device=age.device)
        peak = torch.where(observed[:, None] & (ticks <= self.contact_ticks), arm_force, 0.0).amax(dim=1)
        self.peak[index] = torch.maximum(self.peak[index], peak)
        self.observed_steps[index] = torch.where(observed, elapsed, self.observed_steps[index])
        reached = self.first_reached[index]
        self.first_reached[index] = torch.where(watch_reach & ready & ~terminated & (reached == 0), elapsed, reached)
        self.terminated[index] |= watch_reach & terminated

        self.previous_counter.copy_(counter)
        self.previous_goals.copy_(goals)
        self.previous_ready.copy_(ready)
        self.previous_contact.copy_(arm_force[:, -1] > CONTACT_THRESHOLD_N)

    def episode(self, env_id, arm_mode):
        active = [0, 1] if arm_mode == 2 else [arm_mode]
        slots = self.start[env_id].nonzero().flatten()
        if not len(slots):
            return []
        fields = (
            self.start,
            self.observed_steps,
            self.first_reached,
            self.pre_ready,
            self.pre_contact,
            self.terminated,
            self.peak,
            self.goal_pairs,
            self.delta_pelvis,
        )
        rows = []
        values = (v[env_id, slots].tolist() for v in fields)
        for i, start, steps, first_reached, pre_ready, pre_contact, terminated, peak, goals, delta in zip(
            slots.tolist(), *values, strict=True
        ):
            delta = np.asarray(delta)[active]
            dx, dy, dz = delta.T
            if np.any(dx <= -DIRECTION_THRESHOLD_M):
                direction = "retract"
            elif np.any(dx >= DIRECTION_THRESHOLD_M):
                direction = "forward"
            elif np.any((np.abs(dy) >= DIRECTION_THRESHOLD_M) & (np.abs(dy) >= np.abs(dz))):
                direction = "lateral"
            else:
                direction = "other"
            jump = float(np.linalg.norm(delta, axis=1).max())
            contact_ticks = min(steps * self.substeps, self.contact_ticks)
            full_contact_window = contact_ticks == self.contact_ticks
            soft_contact = peak > CONTACT_THRESHOLD_N
            hard_contact = peak > self.hard_threshold_n
            rows.append(
                {
                    "switch_index": i + 1,
                    "switch_time_s": (start - 1) * self.dt,
                    "arm_mode": ARM_MODES[arm_mode],
                    "direction": direction,
                    "position_jump_bin": JUMP_LABELS[np.searchsorted(JUMP_BINS_M, jump, side="right")],
                    "max_position_jump_m": jump,
                    "active_wrist_delta_initial_pelvis_m": delta.tolist(),
                    "old_goal_pose": goals[0],
                    "new_goal_pose": goals[1],
                    "pre_switch_ready": pre_ready,
                    "pre_switch_contact": pre_contact,
                    "contact_observed_s": contact_ticks * self.physics_dt,
                    "peak_arm_force_0p2s_n": peak,
                    "contact_over_1n": soft_contact if full_contact_window or soft_contact else None,
                    "contact_over_hard_threshold": hard_contact if full_contact_window or hard_contact else None,
                    "new_goal_reached_within_1s": bool(first_reached)
                    if first_reached or steps >= self.reach_steps or terminated
                    else None,
                    "new_goal_reach_time_s": first_reached * self.dt if first_reached else None,
                    "terminated_within_1s": terminated,
                }
            )
        return rows


class SwitchSummary:
    OUTCOMES = ("contact_over_1n", "contact_over_hard_threshold", "new_goal_reached_within_1s")
    GROUPS = ("arm_mode", "direction", "position_jump_bin", "pre_switch_ready", "pre_switch_contact")

    def __init__(self):
        self.overall = Counter()
        self.groups = defaultdict(Counter)
        self.no_switch_episodes = self.failures_before_first_switch = 0

    def add_episode(self, switches, failed):
        if not switches:
            self.no_switch_episodes += 1
            self.failures_before_first_switch += int(failed)
        for row in switches:
            key = tuple(row[field] for field in self.GROUPS)
            for counts in (self.overall, self.groups[key]):
                counts["switches"] += 1
                counts["pre_switch_ready_count"] += int(row["pre_switch_ready"])
                for name in self.OUTCOMES:
                    if row[name] is not None:
                        counts[name + "_assessed"] += 1
                        counts[name] += int(row[name])

    def _rates(self, counts):
        total, ready = counts["switches"], counts["pre_switch_ready_count"]
        result = {
            "switches": total,
            "pre_switch_ready_count": ready,
            "pre_switch_ready_rate": ready / total if total else None,
        }
        for name in self.OUTCOMES:
            assessed = counts[name + "_assessed"]
            result[name] = {
                "count": counts[name],
                "assessed": assessed,
                "rate": counts[name] / assessed if assessed else None,
            }
        return result

    def report(self):
        return {
            "regular_episodes_without_switch": self.no_switch_episodes,
            "regular_failures_before_first_switch": self.failures_before_first_switch,
            "overall": self._rates(self.overall),
            "by_group": [
                dict(zip(self.GROUPS, key, strict=True), **self._rates(counts))
                for key, counts in sorted(self.groups.items())
            ],
        }
