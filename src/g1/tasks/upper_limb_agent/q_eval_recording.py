"""Accumulate Q windows and capture one matched case before automatic reset."""

import math

import numpy as np
import torch

from g1.rl.evaluation.upper_q import SIDES, TIMELINE_WINDOW_S
from g1.rl.failures.storage import TensorHistory


class QWindows:
    def __init__(self, env, steady_speed_mps):
        self.dt = env.step_dt
        self.width = round(TIMELINE_WINDOW_S / self.dt)
        self.steady_speed_mps = steady_speed_mps
        self.slots = torch.arange(env.num_envs, device=env.device)

    def reset(self, deadline):
        # Bin 0 keeps the original end-window metric; remaining bins span the whole episode.
        shape = (len(self.slots), 1 + math.ceil(deadline / self.width))
        options = {"device": self.slots.device}
        self.frames = torch.zeros(shape, dtype=torch.long, **options)
        self.pose_sums = torch.zeros((*shape, 3, 2), **options)  # position, angle, linear speed squared
        q_shape = (*shape, 2, 3)  # actual/target, then left/right/both
        self.q_sums = torch.zeros(q_shape, **options)
        self.q_counts = torch.zeros(q_shape, dtype=torch.long, **options)

    def add(self, active, end_window, age, position, orientation, speed_square, state):
        pose = torch.stack((position, orientation, speed_square), dim=1)
        scores = torch.stack((state.actual_scores, state.target_scores), dim=1)
        masks = torch.stack((state.actual_valid, state.target_valid), dim=1)
        scores = torch.cat((scores, scores.mean(dim=-1, keepdim=True)), dim=-1)
        masks = torch.cat((masks, masks.all(dim=-1, keepdim=True)), dim=-1)
        bins = 1 + (age - 1).clamp_min(0) // self.width
        bins = bins.clamp_max(self.frames.shape[1] - 1)
        for index, mask in (((self.slots, bins), active), ((slice(None), 0), end_window)):
            self.frames[index] += mask
            self.pose_sums[index] += torch.where(mask[:, None, None], pose, 0)
            valid = mask[:, None, None] & masks
            self.q_counts[index] += valid
            self.q_sums[index] += torch.where(valid, scores, 0)

    def records(self, ids, specs, q_ref):
        frames = self.frames[ids].tolist()
        poses = self.pose_sums[ids].tolist()
        q_counts = self.q_counts[ids].tolist()
        q_sums = self.q_sums[ids].tolist()
        result = []
        for row, spec in enumerate(specs):

            def summarize(index, row=row):
                n = frames[row][index]
                position, orientation, speed_square = poses[row][index]
                data = {}
                speed = [math.sqrt(s / n) for s in speed_square] if n else [None, None]
                data["wrists_steady"] = max(speed) <= self.steady_speed_mps if n else None
                for col, side in enumerate(SIDES):
                    data[f"wrist_{side}_error_mm"] = 1000 * position[col] / n if n else None
                    data[f"wrist_{side}_orientation_error_rad"] = orientation[col] / n if n else None
                    data[f"wrist_{side}_speed_rms_mps"] = speed[col]
                for kind, tag in enumerate(("actual", "target")):
                    for col, side in enumerate((*SIDES, "both")):
                        covered = q_counts[row][index][kind][col]
                        score = q_sums[row][index][kind][col] / covered if covered else None
                        key = f"wrist_{side}_q_{tag}"
                        data[key] = score
                        data[f"{key}_raw"] = score * q_ref if score is not None else None
                        data[f"{key}_coverage"] = covered / n if n else None
                return data

            end = summarize(0)
            timeline = []
            for index, start in enumerate(range(0, spec["deadline_step"], self.width), start=1):
                if not frames[row][index]:
                    break
                stop = min(start + self.width, spec["deadline_step"])
                timeline.append(
                    {
                        "start_s": start * self.dt,
                        "end_s": stop * self.dt,
                        "frames": frames[row][index],
                        "expected_frames": stop - start,
                        **summarize(index),
                    }
                )
            result.append({**end, "timeline": timeline})
        return result


class QReplay:
    """Record the first case of each job, so all checkpoints replay the same input."""

    def __init__(self, env, term, spec):
        self.env, self.term = env, term
        self.case_id = spec["case_id"]
        self.history = TensorHistory(spec["deadline_step"] + 1, 1)
        self.done = False

    def capture(self):
        if self.done:
            return
        env, term = self.env, self.term
        data = term.robot.data
        body_pose = data.body_link_pose_w.torch[:1].clone()
        goal_pose = term.goal_pose_w[:1].clone()
        origin = env.scene.env_origins[:1, None, :]
        body_pose[..., :3] -= origin
        goal_pose[..., :3] -= origin
        self.history.push(
            {
                "time": term.episode_age_steps()[:1].float() * env.step_dt,
                "body_pose": body_pose,
                "goal_pose": goal_pose,
                "joint_pos": data.joint_pos.torch[:1],
                "joint_vel": data.joint_vel.torch[:1],
                "action": env.action_manager.action[:1],
                "lower_command": env.action_manager.get_term("joint_pos").lower_command[:1],
            }
        )

    def save(self, directory):
        if not self.done:
            raise RuntimeError("Q replay case did not finish")
        trace = self.history.extract(0)
        np.savez_compressed(directory / "q_replay.npz", **trace)
        return {
            "case_id": self.case_id,
            "frames": len(trace["time"]),
            "body_names": list(self.term.robot.body_names),
            "joint_names": list(self.term.robot.joint_names),
        }
