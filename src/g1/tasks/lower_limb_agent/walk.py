"""Shared WALK motion selection and sensor-owned contact timing; no gait state."""

import math

WALK_COMMAND_THRESHOLD = 0.1


def moving_walk_mask(mode, command, threshold=WALK_COMMAND_THRESHOLD):
    """Select ordinary WALK with planar motion or yaw in the six-dimensional command."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("WALK command threshold must be positive and finite")
    moving = (command[:, 1:3].norm(dim=-1) > threshold) | (command[:, 3].abs() > 2 * threshold)
    return (mode == 2) & moving


def foot_contact_timing(sensor, body_ids):
    """Read air/contact timers and their own contact state in resolved left/right order.

    The sensor backend owns the threshold and reset lifecycle. Rewards and logs
    deliberately avoid an independent force threshold or contact-history test.
    """
    air, contact = sensor.data.current_air_time, sensor.data.current_contact_time
    if air is None or contact is None:
        raise ValueError("WALK rewards require contact sensor track_air_time=True")
    air, contact = air.torch[:, body_ids], contact.torch[:, body_ids]
    if air.shape[1] != 2 or contact.shape != air.shape:
        raise ValueError("WALK contact configuration must select exactly two feet, left then right")
    return air, contact, contact > 0
