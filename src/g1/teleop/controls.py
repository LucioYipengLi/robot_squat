"""Six-dimensional G1 command input, independent of the simulator and input devices."""

from __future__ import annotations

import math
from dataclasses import dataclass

RANGE_FIELDS = ("height_offset", "lin_vel_x", "lin_vel_y", "ang_vel_z", "pelvis_offset_x", "pelvis_offset_y")
AXES = ("LSTICK_UP", "LSTICK_DOWN", "LSTICK_LEFT", "LSTICK_RIGHT", "RSTICK_UP", "RSTICK_DOWN")
KEY_STEPS = {
    "W": (1, 0.1),
    "UP": (1, 0.1),
    "S": (1, -0.1),
    "DOWN": (1, -0.1),
    "A": (2, 0.1),
    "D": (2, -0.1),
    "Q": (3, 0.2),
    "LEFT": (3, 0.2),
    "E": (3, -0.2),
    "RIGHT": (3, -0.2),
    "Z": (0, 0.05),
    "X": (0, -0.05),
    "I": (4, 0.005),
    "K": (4, -0.005),
    "J": (5, 0.005),
    "L": (5, -0.005),
}
DPAD_STEPS = {"DPAD_UP": (1, 0.1), "DPAD_DOWN": (1, -0.1), "DPAD_LEFT": (2, 0.1), "DPAD_RIGHT": (2, -0.1)}

HELP_TEXT = """
G1 teleop (focus the Isaac Sim window)
  W/S or Up/Down: forward/back       A/D: lateral velocity
  Q/E or Left/Right: yaw             Z/X: raise/lower height
  I/K: pelvis forward/back           J/L: pelvis left/right
  V: clear pelvis offset             SPACE: reset ALL input
  C: toggle pelvis/feet geometry      H: show this help
  Gamepad: left stick = vx/yaw; D-pad = vx/vy steps;
           right stick vertical = height; A = reset ALL input.
  Close the window or press Ctrl+C to exit.
"""


def clamp(value, bounds):
    return max(bounds[0], min(bounds[1], value))


@dataclass(frozen=True)
class CommandLimits:
    bounds: tuple[tuple[float, float], ...]

    @classmethod
    def from_ranges(cls, ranges):
        bounds = tuple(tuple(getattr(ranges, name)) for name in RANGE_FIELDS)
        if any(len(pair) != 2 or not all(math.isfinite(v) for v in pair) or pair[0] > pair[1] for pair in bounds):
            raise ValueError("Teleop requires six finite, ordered command ranges")
        # Zero is also a valid static-mode command, even when a sampled band excludes it.
        return cls(tuple((min(0.0, lo), max(0.0, hi)) for lo, hi in bounds))


class CommandController:
    """Combine latched keyboard/D-pad commands with proportional and integrated sticks."""

    def __init__(self, limits: CommandLimits, *, show_pelvis=False):
        self.limits = limits
        self.show_pelvis = show_pelvis
        self.keyboard = [0.0] * 6
        self.gamepad = [0.0] * 4
        self.axes = dict.fromkeys(AXES, 0.0)
        self._buttons_down = set()

    def key(self, name: str, *, repeat=False):
        if name in KEY_STEPS:
            index, delta = KEY_STEPS[name]
            self.keyboard[index] = clamp(self.keyboard[index] + delta, self.limits.bounds[index])
        elif not repeat:
            if name == "SPACE":
                self.reset()
            elif name == "V":
                self.keyboard[4:] = [0.0, 0.0]
            elif name == "C":
                self.show_pelvis = not self.show_pelvis
            elif name == "H":
                return "help"
        return None

    def gamepad_event(self, name: str, value: float):
        if name in self.axes:
            self.axes[name] = clamp(value, (0.0, 1.0))
            return
        if value <= 0:
            self._buttons_down.discard(name)
            return
        if name in self._buttons_down:
            return
        self._buttons_down.add(name)
        if name == "FACE_SOUTH":
            self.reset()
            self._buttons_down.add(name)
        elif name in DPAD_STEPS:
            index, delta = DPAD_STEPS[name]
            self.gamepad[index] = clamp(self.gamepad[index] + delta, self.limits.bounds[index])

    def reset(self):
        self.keyboard[:] = [0.0] * 6
        self.gamepad[:] = [0.0] * 4
        self.axes = dict.fromkeys(AXES, 0.0)
        self._buttons_down.clear()

    def clear_gamepad(self):
        """Drop stale stick/D-pad values when the device disconnects."""
        self.gamepad[:] = [0.0] * 4
        self.axes = dict.fromkeys(AXES, 0.0)
        self._buttons_down.clear()

    def sample(self, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Teleop dt must be finite and positive")

        def axis(positive, negative):
            value = self.axes[positive] - self.axes[negative]
            return value if abs(value) >= 0.08 else 0.0

        def proportional(value, bounds):
            return value * (bounds[1] if value >= 0 else -bounds[0])

        self.gamepad[0] = clamp(self.gamepad[0] + axis("RSTICK_UP", "RSTICK_DOWN") * 0.3 * dt, self.limits.bounds[0])
        values = self.keyboard.copy()
        values[:4] = [a + b for a, b in zip(values[:4], self.gamepad, strict=True)]
        values[1] += proportional(axis("LSTICK_UP", "LSTICK_DOWN"), self.limits.bounds[1])
        values[3] += proportional(axis("LSTICK_LEFT", "LSTICK_RIGHT"), self.limits.bounds[3])
        values = [clamp(value, bounds) for value, bounds in zip(values, self.limits.bounds, strict=True)]
        if any(abs(value) > 1e-3 for value in values[1:4]):
            # Do not replay old pelvis offsets when walking stops.
            self.keyboard[4:] = values[4:] = [0.0, 0.0]
        return tuple(values)
