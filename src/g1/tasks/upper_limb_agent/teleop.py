"""Upper-only interactive playback; inference, command updates and physics stay in their normal order."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch

from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    return value


class UpperPolicyPlayback:
    def __init__(self, env, policy, controller, *, checkpoint, seed, output_dir=None):
        self.env, self.policy, self.controller = env, policy, controller
        self.raw = env.unwrapped
        self.term = self.raw.command_manager.get_term("wrist_pose")
        self.course = self.term.range_curriculum
        if self.course is not None:
            self.course.evaluation = True
        kinds = ["regular"]
        if self.term.cfg.height_task is not None:
            kinds.append("height_shifted")
        if self.term.cfg.planar_task is not None and hasattr(self.raw.cfg.actions, "lower_command"):
            kinds.append("planar")
        controller.available_categories = kinds + ["mixed"]
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        self.output_dir = Path(output_dir) if output_dir is not None else Path("outputs/teleop") / stamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (self.output_dir / "session.jsonl").open("w", encoding="utf-8", buffering=1)
        self._trial = 0
        self._trial_steps = 0
        self._snapshot = None
        self._status_age = 0.0
        self._source = None
        self._write("session", checkpoint=str(Path(checkpoint).resolve()), seed=seed)
        try:
            self._reset(reason="startup")
        except BaseException:
            self._stream.close()
            raise
        print(f"[TELEOP] Session records: {self.output_dir / 'session.jsonl'}")

    def _write(self, event, **values):
        c = self.controller
        row = {
            "event": event,
            "trial": self._trial,
            "control_step": int(self.raw.common_step_counter),
            "source": self._source or c.source,
            "category_selection": c.category,
            "arm_selection": c.arm_mode,
            "stochastic": c.stochastic,
            "level": self.course.level if self.course is not None else None,
            **values,
        }
        self._stream.write(json.dumps(_json_value(row), ensure_ascii=False) + "\n")

    def _begin(self, *, keep_manual_targets=False):
        self._trial += 1
        self._trial_steps = 0
        self._snapshot = self.term.teleop_snapshot()
        self._manual_targets_pending = keep_manual_targets
        if keep_manual_targets:
            self._write("target", target=self._snapshot, retained_manual_goal_pose_w=self.controller.poses[None])
        else:
            self.controller.adopt(self.term.goal_pose_w)
            self._write("target", target=self._snapshot)
        self.update_status()

    def _reset(self, *, reason, replay=None, carry=None):
        if self._snapshot is not None:
            self._write("interrupted", reason=reason, elapsed_s=self._trial_steps * self.raw.step_dt)
        c = self.controller
        self.term.set_teleop(c.source, c.category, c.arm_mode, replay=replay)
        self.obs, _ = self.env.reset()
        self.policy.reset()
        self._source = c.source
        self._begin()
        if carry is not None:
            c.adopt(carry)
            c.changed = True

    def _requests(self):
        c = self.controller
        requests = list(dict.fromkeys(c.requests))
        c.requests.clear()
        for action in requests:
            if action == "configure":
                if self._source == c.source == "manual":
                    c.changed = True  # Changing active arms alone needs no physical reset.
                else:
                    carry = self.term.goal_pose_w.clone() if c.source == "manual" else None
                    self._reset(reason="mode_change", carry=carry)
            elif action == "next":
                self._reset(reason="next")
            elif action == "retry":
                replay = self.term.teleop_snapshot()
                if c.source == "manual":
                    replay["goal_pose_w"].copy_(c.poses.to(self.term.device)[None])
                    replay["mode_id"].fill_(c.arm_mode)
                self._reset(reason="retry", replay=replay)
            elif action == "restore" and c.source == "manual":
                c.adopt(self.term.robot.data.body_link_pose_w.torch[:, self.term.body_indices])
                c.changed = True
            elif action == "policy":
                self._write("policy_mode")

    def step(self):
        with torch.inference_mode():
            self._requests()
            c = self.controller
            # Infer from the last completed observation; submit edits for the
            # upcoming command update, with the same one-step latency as Lower teleop.
            actions = self.policy(self.obs, stochastic_output=True) if c.stochastic else self.policy(self.obs)
            if c.source == "manual" and c.changed:
                self.term.set_manual_targets(c.poses[None], c.arm_mode)
                self._write("edit", goal_pose_w=c.poses[None])
                c.changed = False
                self._manual_targets_pending = False
            self.obs, _, dones, infos = self.env.step(actions)
            self.policy.reset(dones)
            self._trial_steps += 1
            if bool(dones.any()):
                reasons = [
                    name.removeprefix("Episode_Termination/")
                    for name, value in infos.get("log", {}).items()
                    if name.startswith("Episode_Termination/") and float(value) > 0
                ]
                self._write("ended", reasons=reasons, elapsed_s=self._trial_steps * self.raw.step_dt)
                # Keep the editor's values while the reset pose holds the robot.
                self._begin(keep_manual_targets=self._source == "manual")
            elif c.source == "auto":
                c.adopt(self.term.goal_pose_w)
            self._status_age += self.raw.step_dt
            if self._status_age >= c.cfg.status_period_s:
                self._status_age = 0
                self.update_status()

    def update_status(self):
        c, term = self.controller, self.term
        position, orientation = term.pose_error()
        age = int(term.episode_age_steps()[0])
        deadline = int(term.episode_deadline_steps[0])
        category = ["regular", "height_shifted", "planar"][int(term.task_ids[0])]
        stage = (
            ("targets kept; edit or Retry target to apply" if self._manual_targets_pending else "manual hold")
            if c.source == "manual"
            else (
                "evaluation"
                if bool(term.planar_task_mask[0]) and age >= int(term.evaluation_start_steps[0])
                else "executing"
            )
        )
        level = self.course.level if self.course is not None else "-"
        active = term.active_mask[0].tolist()
        status = {
            "Trial / level / task": f"{self._trial} / {level} / {category}",
            "Stage": stage,
            "Remaining": "held" if c.source == "manual" else f"{max(0, deadline - age) * self.raw.step_dt:.1f} s",
            "Position L/R [m]": ", ".join(
                f"{x:.4f}" if active[i] else "idle" for i, x in enumerate(position[0].tolist())
            ),
            "Angle L/R [rad]": ", ".join(
                f"{x:.4f}" if active[i] else "idle" for i, x in enumerate(orientation[0].tolist())
            ),
        }
        if c.source == "manual" and c.arm_mode == 2:
            delta = quat_apply_inverse(term._sampling_quat_w[:1].cpu(), (c.poses[0, :3] - c.poses[1, :3])[None])[0]
            pair = term.cfg.dual_targets
            angle = float(quat_error_magnitude(c.poses[:1, 3:], c.poses[1:, 3:])[0])
            valid = (
                pair.separation_range[0] <= float(delta.norm()) <= pair.separation_range[1]
                and pair.relative_pos_x[0] <= float(delta[0]) <= pair.relative_pos_x[1]
                and pair.relative_pos_z[0] <= float(delta[2]) <= pair.relative_pos_z[1]
                and float(delta[1]) > 0
                and angle <= pair.max_relative_angle
            )
            status["Pair geometry"] = "within configured bounds" if valid else "outside configured training bounds"
        if "lower_command" in self.raw.action_manager.active_terms:
            command = self.raw.action_manager.get_term("joint_pos").lower_command[0]
            status["Lower h/vx/vy/wz/px/py"] = ", ".join(f"{x:.3f}" for x in command.tolist())
        c.status = status

    def close(self):
        if not self._stream.closed:
            self._write("closed", elapsed_s=self._trial_steps * self.raw.step_dt)
            self._stream.close()
