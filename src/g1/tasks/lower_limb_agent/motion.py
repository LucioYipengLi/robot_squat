"""Shared upper-body target execution for training, play and evaluation."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .trajectory import QuinticTrajectory


@configclass
class UpperBodyMotionCfg:
    """Reference limits, not physical actuator limits. Units: rad/s, rad/s², s."""

    profile: str = "quintic_v1"
    max_velocity: float = 2.5
    max_acceleration: float = 8.0
    min_duration_s: float = 0.3


class UpperBodyMotion:
    """Own one articulation subset and publish references in its resolved order."""

    def __init__(self, robot, joint_ids, cfg: UpperBodyMotionCfg):
        if cfg.profile not in ("quintic_v1", "step_v1"):
            raise ValueError(f"Unsupported upper-body motion profile: {cfg.profile}")
        self.robot, self.joint_ids, self.cfg = robot, joint_ids, cfg
        host_ids = joint_ids.tolist() if isinstance(joint_ids, torch.Tensor) else list(joint_ids)
        if not host_ids or len(set(host_ids)) != len(host_ids):
            raise ValueError("Upper-body motion requires a nonempty, unique joint subset")
        self._columns = {joint: column for column, joint in enumerate(host_ids)}
        position = robot.data.joint_pos.torch[:, joint_ids]
        self.trajectory = QuinticTrajectory(
            position,
            max_velocity=cfg.max_velocity,
            max_acceleration=cfg.max_acceleration,
            min_duration_s=cfg.min_duration_s,
        )
        self._input_orders = {}

    def reset(self, env_ids) -> None:
        # Physical state and actuator commands are synchronized by the environment
        # before this call; the same reset works with the interval event disabled.
        self.trajectory.reset(env_ids, self.robot.data.joint_pos.torch[env_ids][:, self.joint_ids])

    def get_observation_goal(self) -> torch.Tensor:
        """Read the segment endpoint, previewing a pending goal only while idle.

        Columns follow the observation joint order resolved by the action term.
        Reading this target never starts a segment or consumes the pending goal.
        """
        if self.cfg.profile == "step_v1":
            return self.robot.data.joint_pos_target.torch[:, self.joint_ids]
        curve = self.trajectory
        return torch.where(~curve.active & curve.pending, curve.pending_goal, curve.goal)

    def set_goal(self, env_ids, goal, joint_ids) -> None:
        """Accept a complete subset in event/eval order, without resampling."""
        key = tuple(joint_ids)
        if key not in self._input_orders:
            if len(key) != len(self._columns) or set(key) != self._columns.keys():
                raise ValueError("Disturbance targets must cover exactly the configured upper-body joints")
            self._input_orders[key] = torch.tensor(
                [key.index(joint) for joint in self._columns], device=goal.device, dtype=torch.long
            )
        ordered = goal[:, self._input_orders[key]]
        if self.cfg.profile == "step_v1":
            self.robot.actuators.target_command.set_position_index(
                value=ordered, joint_ids=self.joint_ids, env_ids=env_ids
            )
        else:
            self.trajectory.set_goal(env_ids, ordered)

    def advance(self, dt: float) -> None:
        if self.cfg.profile == "step_v1":
            return
        self.trajectory.advance(dt)
        command = self.robot.actuators.target_command
        command.set_position_index(value=self.trajectory.position, joint_ids=self.joint_ids)
        command.set_velocity_index(value=self.trajectory.velocity, joint_ids=self.joint_ids)
