"""Batched rest-to-rest quintic references, independent of simulator APIs."""

from __future__ import annotations

import math

import torch


class QuinticTrajectory:
    """One running segment and one latest pending goal per environment.

    All joints in a slot share a duration. Analytic peak derivatives determine
    that duration: max(h')=15/8 and max(abs(h''))=10/sqrt(3).
    ``advance`` performs only fixed-shape device operations, with no CPU reads.
    Statistics columns: started, completed, pending replacements, planned
    duration sum [s], and executed movement time [s]. Completion describes the
    reference reaching its goal, not the physical joint tracking it.
    """

    def __init__(self, position, *, max_velocity=2.0, max_acceleration=8.0, min_duration_s=0.5, accelerated=True):
        for name, value in (
            ("max_velocity", max_velocity),
            ("max_acceleration", max_acceleration),
            ("min_duration_s", min_duration_s),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if position.ndim != 2 or position.shape[1] == 0:
            raise ValueError("Trajectory positions must have shape (environments, nonempty joints)")
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.min_duration_s = min_duration_s
        self.position = position.clone()
        self.velocity = torch.zeros_like(position)
        self.start = position.clone()
        self.goal = position.clone()
        self.pending_goal = position.clone()
        clock = position[:, :1]
        self.elapsed = torch.zeros_like(clock)
        self.duration = torch.full_like(clock, min_duration_s)
        self.pending_duration = self.duration.clone()
        self.active = torch.zeros_like(clock, dtype=torch.bool)
        self.pending = torch.zeros_like(self.active)
        self.statistics = position.new_zeros((position.shape[0], 5))
        self._warp = None
        if accelerated and position.is_cuda and position.dtype == torch.float32:
            import warp as wp

            from .trajectory_kernel import advance_quintic

            wp.init()
            self._warp = wp
            self._kernel = advance_quintic
            self._kernel_buffers = [
                wp.from_torch(buffer)
                for buffer in (
                    self.position,
                    self.velocity,
                    self.start,
                    self.goal,
                    self.pending_goal,
                    self.elapsed,
                    self.duration,
                    self.pending_duration,
                    self.active,
                    self.pending,
                    self.statistics,
                )
            ]

    def reset(self, env_ids, position) -> None:
        """Drop the ending episode's segment, pending goal and statistics."""
        for buffer in (self.position, self.start, self.goal, self.pending_goal):
            buffer[env_ids] = position
        self.velocity[env_ids] = 0
        self.elapsed[env_ids] = 0
        self.duration[env_ids] = self.min_duration_s
        self.pending_duration[env_ids] = self.min_duration_s
        self.active[env_ids] = False
        self.pending[env_ids] = False
        self.statistics[env_ids] = 0

    def set_goal(self, env_ids, goal) -> None:
        """Replace only the pending goal; the running segment remains intact."""
        origin = torch.where(self.active[env_ids], self.goal[env_ids], self.position[env_ids])
        distance = (goal - origin).abs().amax(dim=1, keepdim=True)
        duration = torch.maximum(
            distance * (1.875 / self.max_velocity),
            (distance * (10 / math.sqrt(3) / self.max_acceleration)).sqrt(),
        ).clamp_min(self.min_duration_s)
        self.statistics[env_ids, 2] += self.pending[env_ids, 0].to(self.statistics.dtype)
        self.pending_goal[env_ids] = goal
        self.pending_duration[env_ids] = duration
        self.pending[env_ids] = distance > 1e-8

    def advance(self, dt: float) -> None:
        """Advance once per action-application interval, including its endpoint."""
        if self._warp is not None:
            # Use Torch's current stream, so targets and resets remain ordered
            # without a device-wide synchronization or extra staging buffers.
            stream = self._warp.stream_from_torch(torch.cuda.current_stream(self.position.device))
            self._warp.launch(
                self._kernel, dim=self.position.shape[0], inputs=[*self._kernel_buffers, dt], stream=stream
            )
            return
        ready = ~self.active & self.pending
        self.start.copy_(torch.where(ready, self.position, self.start))
        self.goal.copy_(torch.where(ready, self.pending_goal, self.goal))
        self.duration.copy_(torch.where(ready, self.pending_duration, self.duration))
        self.elapsed.masked_fill_(ready, 0)
        self.active |= ready
        self.pending &= ~ready
        self.statistics[:, 0].add_(ready[:, 0])
        self.statistics[:, 3].add_((ready * self.duration)[:, 0])
        self.statistics[:, 4].add_(self.active[:, 0] * dt)

        self.elapsed.copy_(torch.where(self.active, torch.minimum(self.elapsed + dt, self.duration), self.elapsed))
        s = self.elapsed / self.duration
        blend = (s**3 * (10 + s * (-15 + 6 * s))).clamp_(0, 1)
        speed = 30 * s.square() * (1 - s).square() / self.duration
        delta = self.goal - self.start
        self.position.copy_(torch.where(self.active, self.start + delta * blend, self.position))
        self.velocity.copy_(torch.where(self.active, delta * speed, 0.0))
        done = self.active & (self.elapsed >= self.duration)
        self.position.copy_(torch.where(done, self.goal, self.position))
        self.velocity.masked_fill_(done.expand_as(self.velocity), 0)
        self.statistics[:, 1].add_(done[:, 0])
        self.active &= ~done
