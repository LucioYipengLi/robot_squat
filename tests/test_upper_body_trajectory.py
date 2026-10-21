"""Reference continuity, derivative limits, queue semantics and reset isolation."""

import math

import pytest
import torch

from g1.tasks.lower_limb_agent.trajectory import QuinticTrajectory


@pytest.mark.parametrize("dt", [1 / 120, 1 / 60])
def test_reference_is_monotonic_bounded_and_finishes_at_rest(dt):
    curve = QuinticTrajectory(torch.zeros(2, 3, dtype=torch.float64), max_velocity=1, max_acceleration=3)
    goal = torch.tensor([[0.1, -0.5, 2], [1.0, 0.2, -0.1]], dtype=torch.float64)
    curve.set_goal(slice(None), goal)
    durations = curve.pending_duration.clone()
    previous, previous_v = curve.position.clone(), curve.velocity.clone()
    for _ in range(math.ceil(durations.max().item() / dt) + 2):
        curve.advance(dt)
        assert curve.velocity.abs().max() <= 1 + 1e-9
        assert ((curve.velocity - previous_v) / dt).abs().max() <= 3 + 1e-9
        assert (curve.position - previous).abs().max() <= dt + 1e-9
        assert ((curve.position - previous) * goal >= -1e-10).all()
        previous, previous_v = curve.position.clone(), curve.velocity.clone()
    torch.testing.assert_close(curve.position, goal)
    assert not curve.velocity.any()
    assert not curve.active.any()
    torch.testing.assert_close(curve.statistics[:, :2], torch.ones(2, 2, dtype=torch.float64))


def test_latest_pending_goal_waits_for_segment_end_without_reversal_jump():
    curve = QuinticTrajectory(torch.zeros(2, 1), min_duration_s=0.5)
    curve.set_goal(torch.tensor([0]), torch.tensor([[0.3]]))
    curve.advance(0.1)
    curve.set_goal(torch.tensor([0]), torch.tensor([[-0.1]]))
    curve.set_goal(torch.tensor([0]), torch.tensor([[-0.2]]))
    assert curve.goal[0] == pytest.approx(0.3)
    for _ in range(20):
        curve.advance(0.1)
    assert curve.position[0] == pytest.approx(-0.2)
    torch.testing.assert_close(curve.statistics[0, :3], torch.tensor([2.0, 2.0, 1.0]))
    assert curve.position[1] == 0


def test_partial_reset_discards_motion_and_pending_goal_only_for_selected_slots():
    curve = QuinticTrajectory(torch.zeros(3, 2))
    curve.set_goal(slice(None), torch.ones(3, 2))
    curve.advance(0.1)
    curve.set_goal(slice(None), -torch.ones(3, 2))
    ongoing = curve.position[[0, 2]].clone()
    curve.reset(torch.tensor([1]), torch.tensor([[0.03, -0.02]]))
    assert not curve.pending[1] and not curve.active[1]
    assert not curve.statistics[1].any()
    curve.advance(0.1)
    torch.testing.assert_close(curve.position[1], torch.tensor([0.03, -0.02]))
    assert (curve.position[[0, 2]] > ongoing).all()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_limits_fail_at_construction(value):
    with pytest.raises(ValueError, match="finite and positive"):
        QuinticTrajectory(torch.zeros(1, 1), max_velocity=value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity check")
def test_fused_cuda_kernel_matches_tensor_reference_with_pending_and_resets():
    initial = torch.zeros(5, 17, device="cuda:0")
    fused, reference = QuinticTrajectory(initial), QuinticTrajectory(initial, accelerated=False)
    ids = torch.tensor([3, 1], device="cuda:0", dtype=torch.int32)
    goal = torch.linspace(-0.7, 0.7, 34, device="cuda:0").reshape(2, 17)
    for step in range(200):
        for curve in (fused, reference):
            if step in (0, 1, 2, 85):
                curve.set_goal(ids, goal * (1 if step % 2 == 0 else -1))
            if step == 90:
                curve.reset(ids[:1], torch.zeros(1, 17, device="cuda:0"))
            curve.advance(1 / 120)
        torch.testing.assert_close(fused.position, reference.position, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(fused.velocity, reference.velocity, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(fused.statistics, reference.statistics)
        assert torch.equal(fused.active, reference.active)
