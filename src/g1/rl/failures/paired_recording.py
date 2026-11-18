"""Evaluation-only playback of one fixed A-to-B switch per trial."""

import math

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, quat_unique

from g1.tasks.upper_limb_agent.mdp.pose_override import PoseOverride

from .pairs import AFTER_S, SWITCH_S


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
    def __init__(self, term, suite):
        super().__init__("auto", "regular", 0)
        self.pairs = suite["pairs"]
        self.trial_count = 2 * len(self.pairs)
        self.next_trial = 0
        self.trial_ids = torch.full_like(term.command_counter, -1)
        self.right_poses = term.goal_pose_w.new_tensor([[p["right_old_pose"], p["right_new_pose"]] for p in self.pairs])
        rotations = self.right_poses[..., 3:]
        rotations.div_(rotations.norm(dim=-1, keepdim=True))
        self.new_pose_w = torch.zeros_like(term.goal_pose_w[:, 0])
        dt = term._env.step_dt
        self.switch_step = math.ceil(SWITCH_S / dt - 1e-9)
        self.end_step = self.switch_step + math.ceil(AFTER_S / dt - 1e-9)
        if self.end_step > term._env.max_episode_length:
            raise ValueError("Episode time limit must allow the full fixed switch trial")

    def prepare_reset(self, term, ids):
        super().prepare_reset(term, ids)
        self.trial_ids[ids] = -1
        count = min(len(ids), self.trial_count - self.next_trial)
        assigned = ids[:count]
        trials = torch.arange(self.next_trial, self.next_trial + count, device=self.trial_ids.device)
        self.trial_ids[assigned] = trials
        term.mode_id[assigned] = trials % 2
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
        targets = self.right_poses[self.trial_ids[active] // 2].clone()
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
        term.goal_pose_w[ids, term.mode_id[ids]] = self.new_pose_w[ids]
        term.command_counter[ids] += 1

    def metadata(self, env_id):
        trial_id = int(self.trial_ids[env_id])
        pair = self.pairs[trial_id // 2]
        return {"pair_id": pair["id"], "pair_group": pair["group"], "pair_side": ("left", "right")[trial_id % 2]}
