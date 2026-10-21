"""Fused CUDA execution of the reference algorithm in trajectory.py."""

import warp as wp


@wp.kernel(enable_backward=False)
def advance_quintic(
    position: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    start: wp.array2d(dtype=wp.float32),
    goal: wp.array2d(dtype=wp.float32),
    pending_goal: wp.array2d(dtype=wp.float32),
    elapsed: wp.array2d(dtype=wp.float32),
    duration: wp.array2d(dtype=wp.float32),
    pending_duration: wp.array2d(dtype=wp.float32),
    active: wp.array2d(dtype=wp.bool),
    pending: wp.array2d(dtype=wp.bool),
    statistics: wp.array2d(dtype=wp.float32),
    dt: float,
):
    # One thread owns every joint and flag of its environment; no atomics or
    # cross-thread races between segment activation and joint interpolation.
    row = wp.tid()
    ready = not active[row, 0] and pending[row, 0]
    if ready:
        duration[row, 0] = pending_duration[row, 0]
        elapsed[row, 0] = 0.0
        pending[row, 0] = False
        active[row, 0] = True
        statistics[row, 0] = statistics[row, 0] + 1.0
        statistics[row, 3] = statistics[row, 3] + duration[row, 0]
    if active[row, 0]:
        t = wp.min(elapsed[row, 0] + dt, duration[row, 0])
        elapsed[row, 0] = t
        s = t / duration[row, 0]
        blend = wp.clamp(s * s * s * (10.0 + s * (-15.0 + 6.0 * s)), 0.0, 1.0)
        speed = 30.0 * s * s * (1.0 - s) * (1.0 - s) / duration[row, 0]
        done = t >= duration[row, 0]
        for joint in range(position.shape[1]):
            if ready:
                start[row, joint] = position[row, joint]
                goal[row, joint] = pending_goal[row, joint]
            delta = goal[row, joint] - start[row, joint]
            if done:
                position[row, joint] = goal[row, joint]
                velocity[row, joint] = 0.0
            else:
                position[row, joint] = start[row, joint] + delta * blend
                velocity[row, joint] = delta * speed
        statistics[row, 4] = statistics[row, 4] + dt
        if done:
            statistics[row, 1] = statistics[row, 1] + 1.0
            active[row, 0] = False
