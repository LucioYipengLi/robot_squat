"""Evaluation-only playback of one fixed A-to-B switch per trial."""

import math

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, quat_unique, yaw_quat

from g1.tasks.upper_limb_agent.mdp.pose_override import PoseOverride

from .pairs import AFTER_S, SWITCH_S

HOLD_S = 0.60
RETREAT_M = 0.05
STABLE_S = 0.20


def mirror_right_poses(poses, pelvis_pos, pelvis_rot, default_right_rot, default_left_rot):
    """Reflect about the reset pelvis plane, with the destination wrist's local axes."""
    local_pos = quat_apply_inverse(pelvis_rot, poses[..., :3] - pelvis_pos)
    local_pos[..., 1] *= -1
    pos = pelvis_pos + quat_apply(pelvis_rot, local_pos)

    # Reflect the rotation from the nominal right wrist in pelvis axes. Apply
    # the mirrored rotation to the nominal left wrist, whose link frame may
    # use different local axes. Quaternions are xyzw; rotation axes are axial.
    rot = poses[..., 3:]
    pelvis_rot = pelvis_rot.expand_as(rot)
    pelvis_inv = quat_conjugate(pelvis_rot)
    right_delta = quat_mul(rot, quat_conjugate(default_right_rot.expand_as(rot)))
    local_delta = quat_mul(quat_mul(pelvis_inv, right_delta), pelvis_rot)
    local_delta[..., 0] *= -1
    local_delta[..., 2] *= -1
    left_delta = quat_mul(quat_mul(pelvis_rot, local_delta), pelvis_inv)
    return torch.cat((pos, quat_unique(quat_mul(left_delta, default_left_rot.expand_as(rot)))), dim=-1)


class FixedSwitchPairs(PoseOverride):
    def __init__(self, term, suite, *, diagnostic=None):
        self.dual = suite.get("schema_version") == 2
        super().__init__("auto", "regular", 2 if self.dual else 0)
        self.pairs = suite["pairs"]
        self.trials_per_pair = 1 if self.dual else 2
        self.trial_count = self.trials_per_pair * len(self.pairs)
        self.next_trial = 0
        self.trial_ids = torch.full_like(term.command_counter, -1)
        old_key, new_key = ("old_poses", "new_poses") if self.dual else ("right_old_pose", "right_new_pose")
        self.poses = term.goal_pose_w.new_tensor([[p[old_key], p[new_key]] for p in self.pairs])
        rotations = self.poses[..., 3:]
        rotations.div_(rotations.norm(dim=-1, keepdim=True))
        self.new_pose_w = torch.zeros_like(term.goal_pose_w if self.dual else term.goal_pose_w[:, 0])
        dt = term._env.step_dt
        start_step = math.ceil(SWITCH_S / dt - 1e-9)
        hold_steps = math.ceil(HOLD_S / dt - 1e-9) if diagnostic in ("hold", "retreat") else 0
        self.switch_step = start_step + hold_steps
        self.end_step = self.switch_step + math.ceil(AFTER_S / dt - 1e-9)
        if self.end_step > term._env.max_episode_length:
            raise ValueError("Episode time limit must allow the full fixed switch trial")
        self.diagnostic = (
            SwitchDiagnostic(term, self.trial_ids, diagnostic, start_step, self.switch_step) if diagnostic else None
        )

    def prepare_reset(self, term, ids):
        super().prepare_reset(term, ids)
        if self.diagnostic is not None:
            self.diagnostic.reset(ids)
        self.trial_ids[ids] = -1
        count = min(len(ids), self.trial_count - self.next_trial)
        assigned = ids[:count]
        trials = torch.arange(self.next_trial, self.next_trial + count, device=self.trial_ids.device)
        self.trial_ids[assigned] = trials
        term.mode_id[assigned] = 2 if self.dual else trials % 2
        term.evaluation_start_steps[ids] = self.switch_step
        term.episode_deadline_steps[ids] = self.end_step
        self.next_trial += count

    def resample(self, term, ids):
        fresh = ids[term.command_counter[ids] == 0]
        origins = term._env.scene.env_origins
        defaults = term.default_wrist_pose_e
        term.goal_pose_w[fresh] = defaults[fresh]
        term.goal_pose_w[fresh, :, :3] += origins[fresh, None, :]

        active = fresh[self.trial_ids[fresh] >= 0]
        targets = self.poses[self.trial_ids[active] // self.trials_per_pair].clone()
        if self.dual:
            targets[..., :3] += origins[active, None, None, :]
            term.goal_pose_w[active] = targets[:, 0]
        else:
            sides = term.mode_id[active]
            mirrored = mirror_right_poses(
                targets,
                term._default_pelvis_pos_e[active, None],
                term._sampling_quat_w[active, None],
                defaults[active, 1:2, 3:],
                defaults[active, 0:1, 3:],
            )
            targets = torch.where((sides == 0)[:, None, None], mirrored, targets)
            targets[..., :3] += origins[active, None, :]
            term.goal_pose_w[active, sides] = targets[:, 0]
        self.new_pose_w[active] = targets[:, 1]
        term.command_counter[fresh] += 1
        term.time_left[ids] = torch.inf

    def apply_pending(self, term):
        # CommandManager updates after the step's recorder and before the next
        # observation, so the final A frame supplies the switch's ready status.
        due = (self.trial_ids >= 0) & (term.command_counter == 1) & (term.episode_age_steps() >= self.switch_step)
        ids = due.nonzero().flatten()
        if self.dual:
            term.goal_pose_w[ids] = self.new_pose_w[ids]
        else:
            term.goal_pose_w[ids, term.mode_id[ids]] = self.new_pose_w[ids]
        term.command_counter[ids] += 1

    def metadata(self, env_id):
        trial_id = int(self.trial_ids[env_id])
        pair = self.pairs[trial_id // self.trials_per_pair]
        side = "both" if self.dual else ("left", "right")[trial_id % 2]
        result = {"pair_id": pair["id"], "pair_group": pair["group"], "pair_side": side}
        if self.diagnostic is not None:
            result["switch_diagnostic"] = self.diagnostic.metadata(env_id)
        return result


class SwitchDiagnostic:
    """Hold the previous Upper reference while testing a matched Lower intervention."""

    def __init__(self, term, trial_ids, mode, start_step, release_step):
        self.term, self.trial_ids = term, trial_ids
        self.mode, self.start_step, self.release_step = mode, start_step, release_step
        env = term._env
        self.dt = env.step_dt
        self.stable_steps = math.ceil(STABLE_S / self.dt - 1e-9)
        manager = env.action_manager
        self.action_manager = manager
        offset, slices = 0, {}
        for name, width in zip(manager.active_terms, manager.action_term_dim, strict=True):
            slices[name] = slice(offset, offset + width)
            offset += width
        self.velocity, self.pelvis_offset = slices["lower_command"], slices["pelvis_offset"]
        if self.velocity.stop - self.velocity.start != 3 or self.pelvis_offset.stop - self.pelvis_offset.start != 2:
            raise ValueError("Switch diagnostic requires the Mobile Lower velocity and pelvis offset actions")
        self.held = torch.zeros_like(manager.action)
        self.start_xy = term.goal_pose_w.new_zeros((len(trial_ids), 2))
        self.forward_xy = torch.zeros_like(self.start_xy)
        self.retreat_m = self.start_xy.new_zeros(len(trial_ids))
        self.started = torch.zeros_like(trial_ids, dtype=torch.bool)
        self.start_ready = torch.zeros_like(self.started)
        self.start_contact = torch.zeros_like(self.started)
        self.retreat_reached = torch.zeros_like(self.started)
        self.ready_steps = torch.zeros_like(trial_ids)

    def settings(self):
        return {
            "mode": self.mode,
            "start_s": self.start_step * self.dt,
            "hold_s": (self.release_step - self.start_step) * self.dt,
            "retreat_target_m": RETREAT_M if self.mode == "retreat" else None,
            "stable_s": self.stable_steps * self.dt,
        }

    def reset(self, ids):
        self.started[ids] = False
        self.retreat_reached[ids] = False
        self.ready_steps[ids] = 0
        self.retreat_m[ids] = 0

    def observe(self, pelvis, ready, contact, terminated):
        age = self.term.episode_age_steps()
        active = self.trial_ids >= 0
        start = active & (age == self.start_step)
        self.start_xy[start] = pelvis[start, :2]
        forward = pelvis.new_tensor([1.0, 0.0, 0.0])
        self.forward_xy[start] = quat_apply(yaw_quat(pelvis[start, 3:]), forward)[:, :2]
        self.held[start] = self.action_manager.action[start]
        self.start_ready[start] = ready[start]
        self.start_contact[start] = contact[start]
        self.started |= start
        observe = self.started & (age <= self.release_step)
        retreat = ((self.start_xy - pelvis[:, :2]) * self.forward_xy).sum(dim=1)
        self.retreat_m.copy_(torch.where(observe, retreat, self.retreat_m))
        self.retreat_reached |= observe & (retreat >= RETREAT_M)
        tracking = active & (self.term.command_counter == 2)
        self.ready_steps.copy_(torch.where(tracking & ready & ~terminated, self.ready_steps + 1, 0))

    def intervene(self, actions):
        if self.mode == "direct":
            return actions
        ids = (self.started & (self.term.episode_age_steps() < self.release_step)).nonzero().flatten()
        if not len(ids):
            return actions
        actions = actions.clone()
        actions[ids] = self.held[ids]
        actions[ids, self.velocity] = 0.0
        actions[ids, self.pelvis_offset] = 0.0
        if self.mode == "retreat":
            # -1 is the current episode's negative velocity endpoint.
            actions[ids, self.velocity.start] = torch.where(self.retreat_reached[ids], 0.0, -1.0)
        return actions

    def metadata(self, env_id):
        switched = int(self.term.command_counter[env_id]) == 2
        return {
            "ready_before_intervention": bool(self.start_ready[env_id]) if self.started[env_id] else None,
            "contact_before_intervention": bool(self.start_contact[env_id]) if self.started[env_id] else None,
            "pelvis_retreat_at_switch_m": float(self.retreat_m[env_id]) if switched else None,
            "stable_last_0p2s": bool(self.ready_steps[env_id] >= self.stable_steps),
        }
