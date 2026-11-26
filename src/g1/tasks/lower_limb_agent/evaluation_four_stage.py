"""Four-stage targets, pre-observation root pushes and terminal-safe statistics."""

from __future__ import annotations

import math

import torch

from g1.rl.evaluation.four_stage import protocol_plan

from .evaluation import CasePlan, EvaluationScenarioController


class FourStageScenarioController(EvaluationScenarioController):
    def __init__(self, env, benchmark, condition, seed):
        super().__init__(env, benchmark, condition, seed)
        self.squared = torch.zeros((env.num_envs, 5), device=env.device, dtype=torch.float64)
        self.peaks = torch.zeros((env.num_envs, 3), device=env.device)
        self.push_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.recovery_streak = torch.zeros_like(self.push_step)
        self.recovering = torch.zeros_like(self.active)
        self.push_records = [[] for _ in range(env.num_envs)]
        self._last_push_step = [-1] * env.num_envs

    def make_plan(self, spec):
        commands, pushes = protocol_plan(self.benchmark, self.seed, spec)
        return CasePlan(commands, {}, pushes)

    def start_batch(self, case_specs):
        super().start_batch(case_specs)
        self.squared.zero_()
        self.peaks.zero_()
        self.push_step.fill_(-1)
        self.recovery_streak.zero_()
        self.recovering.zero_()
        self.push_records = [[] for _ in range(self.env.num_envs)]
        self._last_push_step = [-1] * self.env.num_envs

    def update_commands(self):
        super().update_commands()
        # CommandManager runs after the previous state's metrics and before the next
        # observation/history update. Publishing here makes the push visible to the actor.
        slots, increments = [], []
        for slot in self.active.nonzero().flatten().tolist():
            step = int(self.env.episode_length_buf[slot])
            delta = self.plans[slot].pushes.get(step)
            if delta is None or self._last_push_step[slot] == step:
                continue
            slots.append(slot)
            increments.append(delta)
            self._last_push_step[slot] = step
            self.push_step[slot] = step
            self.recovery_streak[slot] = 0
            self.recovering[slot] = True
            self.push_records[slot].append(
                {
                    "step": step,
                    "time_s": step * self.benchmark["control_dt"],
                    "delta_vx_mps": delta[0],
                    "delta_vy_mps": delta[1],
                    "recovery_s": None,
                }
            )
        if slots:
            ids = torch.tensor(slots, device=self.env.device)
            velocity = self.robot.data.root_vel_w.torch[ids].clone()
            velocity[:, :2] += torch.tensor(increments, device=self.env.device, dtype=velocity.dtype)
            self.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
            self._pending_events[ids] += 1

    def sample_metrics(self, valid, height_error):
        """Called with the command term's once-per-control-step mask, before reset."""
        selected = valid & self.active
        velocity_error = torch.cat(
            (
                self.robot.data.root_lin_vel_b.torch[:, :2] - self.term.task_command[:, 1:3],
                self.robot.data.root_ang_vel_b.torch[:, 2:3] - self.term.task_command[:, 3:4],
            ),
            dim=1,
        )
        errors = torch.cat((height_error[:, None], velocity_error), dim=1).double()
        self.squared[:, 0] += selected
        self.squared[:, 1:] += torch.where(selected[:, None], errors.square(), 0)
        quat = self.robot.data.root_quat_w.torch
        # Isaac Lab's current tensor API uses xyzw, including root_quat_w.
        x, y, z, w = quat.unbind(dim=1)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square())).abs()
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1)).abs()
        values = torch.stack((height_error, roll, pitch), dim=1)
        self.peaks = torch.maximum(self.peaks, torch.where(selected[:, None], values, 0))
        recovery = self.benchmark["recovery"]
        speed = self.robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
        good = (height_error < recovery["height_m"]) & (roll < recovery["tilt_rad"])
        good &= (pitch < recovery["tilt_rad"]) & (speed < recovery["speed_mps"])
        observing = selected & self.recovering
        self.recovery_streak = torch.where(
            observing, torch.where(good, self.recovery_streak + 1, 0), self.recovery_streak
        )
        recovered = observing & (self.recovery_streak >= math.ceil(recovery["hold_s"] / self.benchmark["control_dt"]))
        for slot in recovered.nonzero().flatten().tolist():
            elapsed = int(self.env.episode_length_buf[slot] - self.push_step[slot])
            self.push_records[slot][-1]["recovery_s"] = elapsed * self.benchmark["control_dt"]
        self.recovering[recovered] = False

    def context(self, env_id):
        context = super().context(env_id)
        spec, plan = self.case_specs[env_id], self.plans[env_id]
        context.update({key: value for key, value in spec.items() if key not in ("episode_id", "mode")})
        context.update(
            zip(
                (
                    "metric_steps",
                    "height_squared_error_sum",
                    "vx_squared_error_sum",
                    "vy_squared_error_sum",
                    "wz_squared_error_sum",
                ),
                self.squared[env_id].tolist(),
                strict=True,
            )
        )
        context.update(
            zip(("max_height_error_m", "max_roll_rad", "max_pitch_rad"), self.peaks[env_id].tolist(), strict=True)
        )
        context.update(
            attitude_quaternion_order="xyzw",
            command_plan=[{"step": step, "command": command} for step, command in plan.commands.items()],
            push_plan=[
                {"step": step, "delta_vx_mps": delta[0], "delta_vy_mps": delta[1]}
                for step, delta in plan.pushes.items()
            ],
            planned_push_count=len(plan.pushes),
            push_count=len(self.push_records[env_id]),
            pushes=self.push_records[env_id],
        )
        return context
