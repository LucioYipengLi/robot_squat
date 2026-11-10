"""Checkpoint contracts and deterministic inference for the real, selected frozen Lower actor."""

import copy
from pathlib import Path

import pytest
import torch

from g1.tasks.upper_limb_agent.frozen_lower import DEFAULT_LOWER_CHECKPOINT, FrozenLowerPolicy, lower_actor_config

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def interface():
    return {
        "input_dim": 650,
        "action_dim": 12,
        "history_length": 5,
        "history_order": "term",
        "observation_terms": [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "pelvis_height_cmd",
            "pelvis_offset_cmd",
            "joint_pos",
            "joint_vel",
            "actions",
            "upper_body_target",
        ],
        "control_dt": 1 / 60,
        "action_scale": 0.5,
        "action_joint_patterns": [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
        "preserve_order": False,
        "use_default_offset": True,
        "offset": 0.0,
        "clip": None,
    }


@pytest.fixture
def frozen_policy(interface):
    if not (PROJECT / DEFAULT_LOWER_CHECKPOINT).is_file():
        pytest.skip("The selected private Lower checkpoint is not installed")
    return FrozenLowerPolicy(DEFAULT_LOWER_CHECKPOINT, lower_actor_config(), interface, "cpu")


def test_real_frozen_actor_matches_export_and_is_deterministic_for_batched_inputs(frozen_policy):
    exported = torch.jit.load(str(frozen_policy.path.parent / "exported/policy.pt")).eval()
    observations = torch.randn(3, 650, generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        expected = exported(observations)
    torch.testing.assert_close(frozen_policy(observations), expected, rtol=0, atol=0)
    torch.testing.assert_close(frozen_policy(observations), frozen_policy(observations), rtol=0, atol=0)
    assert not frozen_policy.model.training
    assert all(not parameter.requires_grad for parameter in frozen_policy.model.parameters())
    assert not frozen_policy(observations).requires_grad
    assert frozen_policy.metadata["iteration"] == 9999
    assert len(frozen_policy.metadata["sha256"]) == 64


def test_loading_frozen_actor_preserves_upper_torch_rng(interface):
    if not (PROJECT / DEFAULT_LOWER_CHECKPOINT).is_file():
        pytest.skip("The selected private Lower checkpoint is not installed")
    actor_cfg = lower_actor_config()
    before = torch.random.get_rng_state().clone()
    FrozenLowerPolicy(DEFAULT_LOWER_CHECKPOINT, actor_cfg, interface, "cpu")
    torch.testing.assert_close(torch.random.get_rng_state(), before)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("history_length", 4, "policy_history"),
        ("history_order", "time", "history order"),
        ("control_dt", 1 / 30, "control period"),
        ("action_scale", 0.25, "action_scale"),
        ("action_joint_patterns", [".*"], "joint selection"),
        ("observation_terms", ["joint_pos"], "term order"),
        ("preserve_order", True, "preserve_order"),
        ("use_default_offset", False, "use_default_offset"),
    ],
)
def test_frozen_actor_rejects_silent_interface_changes(interface, key, value, match):
    if not (PROJECT / DEFAULT_LOWER_CHECKPOINT).is_file():
        pytest.skip("The selected private Lower checkpoint is not installed")
    changed = copy.deepcopy(interface)
    changed[key] = value
    with pytest.raises(ValueError, match=match):
        FrozenLowerPolicy(DEFAULT_LOWER_CHECKPOINT, lower_actor_config(), changed, "cpu")


def test_frozen_actor_checks_runtime_input_shape(frozen_policy):
    with pytest.raises(ValueError, match="observations"):
        frozen_policy(torch.zeros(2, 649))


def test_frozen_actor_rejects_changed_activation_even_with_matching_weight_shapes(interface):
    if not (PROJECT / DEFAULT_LOWER_CHECKPOINT).is_file():
        pytest.skip("The selected private Lower checkpoint is not installed")
    actor_cfg = lower_actor_config()
    actor_cfg["activation"] = "tanh"
    with pytest.raises(ValueError, match="activation"):
        FrozenLowerPolicy(DEFAULT_LOWER_CHECKPOINT, actor_cfg, interface, "cpu")


def test_missing_frozen_checkpoint_fails_with_its_path(tmp_path, interface):
    missing = tmp_path / "model_missing.pt"
    with pytest.raises(FileNotFoundError, match="model_missing.pt"):
        FrozenLowerPolicy(str(missing), lower_actor_config(), interface, "cpu")
