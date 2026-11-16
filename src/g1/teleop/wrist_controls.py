"""World-pose editing and UI requests, independent of a particular viewer."""

import math
from collections import deque
from dataclasses import dataclass

import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_mul


@dataclass
class WristTeleopCfg:
    position_step_m: float = 0.01
    angle_step_deg: float = 2.0
    fine_multiplier: float = 0.1
    key_repeat_s: float = 0.12
    status_period_s: float = 0.1
    source: str = "manual"
    category: str = "regular"
    arm_mode: int = 2


# Leave Newton's WASD/QE, arrows, Space, F/H and Escape to the viewer.
POSE_KEYS = {"L": (0, 1), "J": (0, -1), "I": (1, 1), "K": (1, -1), "O": (2, 1), "U": (2, -1)}


class WristController:
    def __init__(self, cfg=None):
        self.cfg = cfg or WristTeleopCfg()
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.cfg.position_step_m,
                self.cfg.angle_step_deg,
                self.cfg.fine_multiplier,
                self.cfg.key_repeat_s,
                self.cfg.status_period_s,
            )
        ):
            raise ValueError("Wrist teleop increments and intervals must be positive and finite")
        self.source, self.category, self.arm_mode = self.cfg.source, self.cfg.category, self.cfg.arm_mode
        self.editor = 2
        self.fine = self.stochastic = self.changed = False
        self.poses = torch.zeros(2, 7)
        self.poses[:, 6] = 1
        self.requests = deque()
        self.status = {}

    def adopt(self, poses):
        self.poses.copy_(torch.as_tensor(poses).detach().cpu().reshape(2, 7))
        self.changed = False

    def request(self, name):
        self.requests.append(name)

    def nudge(self, axis, direction, *, rotate=False):
        if self.source != "manual":
            return
        selected = [self.arm_mode] if self.arm_mode in (0, 1) else ([0, 1] if self.editor == 2 else [self.editor])
        factor = self.cfg.fine_multiplier if self.fine else 1.0
        if rotate:
            angles = torch.zeros(3)
            angles[axis] = math.radians(self.cfg.angle_step_deg) * direction * factor
            delta = quat_from_euler_xyz(*angles.unbind())
            center = self.poses[selected, :3].mean(0)
            self.poses[selected, :3] = center + quat_apply(
                delta.expand(len(selected), -1), self.poses[selected, :3] - center
            )
            self.poses[selected, 3:] = quat_mul(delta.expand(len(selected), -1), self.poses[selected, 3:])
        else:
            self.poses[selected, axis] += self.cfg.position_step_m * direction * factor
        self.changed = True

    def angles_deg(self, side):
        angles = torch.stack(euler_xyz_from_quat(self.poses[side : side + 1, 3:]), -1)[0]
        return ((angles * (180 / math.pi) + 180) % 360 - 180).tolist()

    def set_pose(self, side, position, angles_deg):
        position, angles = torch.tensor(position), torch.tensor(angles_deg) * (math.pi / 180)
        if not torch.isfinite(position).all() or not torch.isfinite(angles).all():
            return False
        self.poses[side, :3] = position
        self.poses[side, 3:] = quat_from_euler_xyz(*angles.unbind())
        self.changed = True
        return True

    def key(self, name, *, rotate=False):
        if name in POSE_KEYS:
            self.nudge(*POSE_KEYS[name], rotate=rotate)
        elif name in ("1", "2", "3"):
            self.arm_mode = int(name) - 1
            self.request("configure")
        elif name == "N":
            self.request("next")
        elif name == "R":
            self.request("retry")
        elif name == "T":
            self.request("restore")
        elif name == "M":
            self.source = "auto" if self.source == "manual" else "manual"
            if self.source == "manual" and self.arm_mode is None:
                self.arm_mode = 2
            self.request("configure")
        elif name == "G":
            self.fine = not self.fine
