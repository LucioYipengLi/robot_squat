"""Joint-range edge cases and seeded training compatibility for shared targets."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import sample_uniform

from g1.tasks.lower_limb_agent.disturbance import joint_targets_from_uniform
from g1.tasks.lower_limb_agent.mdp.events import UpperBodyDisturbanceEvent

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.mark.parametrize("device", DEVICES)
def test_asymmetric_endpoints_and_out_of_limit_defaults(device):
    default = torch.tensor([[0.25, 0.1]] * 3 + [[3.0, -1.0]], device=device)
    limits = torch.tensor([[[-1.0, 2.0], [-0.5, 0.5]]] * 4, device=device)
    fractions = torch.tensor([0.2, 0.5], device=device)
    samples = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.0, 1.0]], device=device)
    expected = torch.tensor([[0.0, -0.2], [0.6, 0.3], [0.3, 0.05], [2.0, -0.25]], device=device)
    torch.testing.assert_close(joint_targets_from_uniform(default, limits, fractions, samples), expected)
    torch.testing.assert_close(
        joint_targets_from_uniform(default, limits, 0.0, samples),
        torch.tensor([[0.25, 0.1]] * 3 + [[2.0, -0.5]], device=device),
    )
    full_range = joint_targets_from_uniform(default[:3], limits[:3], 1.0, samples[:3])
    torch.testing.assert_close(full_range[:2], torch.tensor([[-1.0, -0.5], [2.0, 0.5]], device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("selection", [None, slice(1, 3), [2, 0], []])
def test_training_targets_and_rng_match_original_sampling(device, selection):
    # Reordered, non-contiguous joints and per-environment defaults/limits.
    default = torch.tensor([[0.1, 0.0, 0.2, -0.1], [0.2, 0.0, -0.1, 0.1], [-0.1, 0.0, 0.3, 0.2]], device=device)
    limits = torch.tensor([[[-1.0, 2.0], [-2.0, 1.0], [-0.5, 1.0], [-1.5, 0.5]]] * 3, device=device)
    limits[2, 3, 1] = 0.7
    env = SimpleNamespace(device=device, command_manager=SimpleNamespace(active_terms=[]))
    event = UpperBodyDisturbanceEvent.__new__(UpperBodyDisturbanceEvent)
    event._env = env
    event._joint_ids, event._num_joints, event._group_sizes = [3, 0, 2], 3, [2, 1]
    event.robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=default), joint_pos_limits=SimpleNamespace(torch=limits)
        )
    )
    submitted = []
    env.upper_body_motion = SimpleNamespace(set_goal=lambda ids, goal, joints: submitted.append(goal.clone()))
    groups = [{"joint_noise_frac": 0.2}, {"joint_noise_frac": 0.1}]
    ids = slice(None) if selection is None else selection
    q = default[ids][:, event._joint_ids]
    bounds = limits[ids][:, event._joint_ids]
    rng_state = torch.cuda.get_rng_state if device == "cuda" else torch.get_rng_state
    restore_rng = torch.cuda.set_rng_state if device == "cuda" else torch.set_rng_state
    initial_rng = rng_state()
    try:
        # Check both the initial fraction and a live curriculum update.
        for arm_fraction in (0.2, 0.555, 1.0):
            groups[0]["joint_noise_frac"] = arm_fraction
            frac = torch.tensor([arm_fraction, arm_fraction, 0.1], device=device)
            before = rng_state()
            if q.shape[0]:
                lo, hi = bounds[..., 0], bounds[..., 1]
                noise = sample_uniform(-frac * (q - lo).clamp_min(0), frac * (hi - q).clamp_min(0), q.shape, device)
                expected = (q + noise).clamp(lo, hi)
            after = rng_state()
            restore_rng(before)
            event(env, selection, groups)
            assert torch.equal(rng_state(), after)
            if q.shape[0]:
                torch.testing.assert_close(submitted[-1], expected, rtol=0, atol=0)
            else:
                assert not submitted
    finally:
        restore_rng(initial_rng)
