"""Reject equal-sized but semantically different policy inputs and action mappings."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml

from g1.rl.evaluation.policy import (
    _load_saved_env_config,
    _validate_saved_policy_semantics,
    validate_policy_compatibility,
)
from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg
from g1.tasks.lower_limb_agent.curriculum_state import _compatibility


def _raw_env():
    return SimpleNamespace(cfg=LowerLimbAgentEnvCfg(), step_dt=1 / 60)


def _infos(env):
    return {
        "g1_curriculum": {
            "schema_version": 1,
            "family": "g1.lower_limb_agent",
            "phase": 3,
            "control_dt": env.step_dt,
            "compatibility": _compatibility(env),
        }
    }


def _save_env(tmp_path, env):
    path = tmp_path / "env.yaml"
    dump_yaml(str(path), env.cfg)
    return {"env_config": str(path)}


def test_actual_env_config_roundtrip_supports_callables_and_inert_python_slice_tags(tmp_path):
    env = _raw_env()
    checkpoint = _save_env(tmp_path, env)
    saved = _load_saved_env_config(checkpoint["env_config"])
    assert isinstance(saved["observations"]["policy"]["upper_body_target"]["params"]["asset_cfg"]["joint_ids"], slice)
    assert isinstance(saved["observations"]["policy"]["joint_pos"]["func"], str)
    assert validate_policy_compatibility(env, checkpoint, _infos(env)) == "checkpoint_metadata+saved_env_config"
    assert validate_policy_compatibility(env, checkpoint, None) == "saved_env_config"


def test_native_metadata_does_not_skip_adjacent_observation_semantics(tmp_path):
    env = _raw_env()
    checkpoint = _save_env(tmp_path, env)
    infos = _infos(env)
    env.cfg.observations.policy.base_lin_vel.scale = 2.0
    with pytest.raises(ValueError, match="observation layout.*scale"):
        validate_policy_compatibility(env, checkpoint, infos)


def test_reordered_equal_sized_observation_terms_are_rejected():
    env = _raw_env()
    saved = to_dict(env.cfg)
    policy = saved["observations"]["policy"]
    # Both terms have three values; input dimension and actor weight shapes stay identical.
    names = list(policy)
    left, right = names.index("base_lin_vel"), names.index("base_ang_vel")
    names[left], names[right] = names[right], names[left]
    saved["observations"]["policy"] = {name: policy[name] for name in names}
    with pytest.raises(ValueError, match="observation layout.*name"):
        _validate_saved_policy_semantics(env.cfg, saved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", 2.0),
        ("clip", (-2.0, 2.0)),
        ("func", "isaaclab.envs.mdp.observations:base_ang_vel"),
        ("params", {"unexpected_argument": True}),
        ("delay_min_lag", 1),
        ("delay_max_lag", 2),
        ("delay_hold_prob", 0.5),
        ("modifiers", [{"func": "example.module:modifier", "params": {}}]),
    ],
)
def test_same_shape_processing_changes_are_rejected(field, value):
    env = _raw_env()
    saved = to_dict(env.cfg)
    saved["observations"]["policy"]["base_lin_vel"][field] = value
    with pytest.raises(ValueError, match="observation layout"):
        _validate_saved_policy_semantics(env.cfg, saved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("history_order", "time"),
        ("flatten_history_dim", False),
        ("concatenate_dim", 0),
        ("concatenate_terms", False),
    ],
)
def test_group_history_and_concatenation_semantics_are_verified(field, value):
    env = _raw_env()
    saved = to_dict(env.cfg)
    saved["observations"]["policy"][field] = value
    with pytest.raises(ValueError, match="observation layout"):
        _validate_saved_policy_semantics(env.cfg, saved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_default_offset", False),
        ("preserve_order", True),
        ("offset", 0.3),
        ("clip", {".*": (-1.0, 1.0)}),
        ("class_type", "isaaclab.envs.mdp.actions.joint_actions:RelativeJointPositionAction"),
    ],
)
def test_additional_action_mapping_semantics_are_verified(field, value):
    env = _raw_env()
    saved = to_dict(env.cfg)
    saved["actions"]["joint_pos"][field] = value
    with pytest.raises(ValueError, match="action layout"):
        _validate_saved_policy_semantics(env.cfg, saved)


def test_suite_owned_noise_and_overridden_term_history_are_ignored():
    env = _raw_env()
    saved = to_dict(env.cfg)
    env.cfg.observations.policy.enable_corruption = False
    env.cfg.observations.policy.base_lin_vel.noise = None
    saved["observations"]["policy"]["base_lin_vel"]["history_length"] = 17
    # Group history overrides the term's raw setting in ObservationManager.
    _validate_saved_policy_semantics(env.cfg, saved)


def test_metadata_only_and_explicit_unverified_paths_keep_their_existing_contract():
    env = _raw_env()
    assert validate_policy_compatibility(env, {}, _infos(env)) == "checkpoint_metadata"
    assert (
        validate_policy_compatibility(env, {}, None, allow_unverified=True)
        == "unverified_task_semantics_strict_actor_shapes"
    )
    with pytest.raises(ValueError, match="no task metadata"):
        validate_policy_compatibility(env, {}, None)


def test_unknown_python_yaml_objects_are_rejected_and_global_loader_is_unchanged(tmp_path):
    path = tmp_path / "env.yaml"
    path.write_text("object: !!python/object:builtins.dict {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported saved environment YAML"):
        _load_saved_env_config(str(path))
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("selector: !!python/object/apply:builtins.slice [null, null, null]")


def test_existing_training_config_if_present_is_portable_and_semantically_compatible():
    path = Path(__file__).resolve().parents[1] / "logs/rsl_rl/g1_flat/4090-10-16_v1/params/env.yaml"
    if not path.is_file():
        pytest.skip("Historical local training run is not part of the repository")
    env = _raw_env()
    saved = _load_saved_env_config(str(path))
    _validate_saved_policy_semantics(env.cfg, copy.deepcopy(saved))


@pytest.mark.parametrize("profile", [None, "step_v1", "quintic_v1"])
def test_historical_target_source_is_inferred_without_changing_saved_config(profile):
    env = _raw_env()
    saved = to_dict(env.cfg)
    saved["observations"]["policy"]["upper_body_target"]["params"].pop("target_source")
    if profile is None:
        saved.pop("upper_body_motion")
    else:
        saved["upper_body_motion"]["profile"] = profile
    before = copy.deepcopy(saved)
    if profile == "quintic_v1":
        with pytest.raises(ValueError, match="observation layout.*target_source"):
            _validate_saved_policy_semantics(env.cfg, saved)
        env.cfg.observations.policy.upper_body_target.params["target_source"] = "reference"
    _validate_saved_policy_semantics(env.cfg, saved)
    assert saved == before


@pytest.mark.parametrize("source", ["final_goal", "reference"])
def test_explicit_target_source_overrides_profile_and_same_shape_mismatch_is_rejected(source):
    env = _raw_env()
    env.cfg.observations.policy.upper_body_target.params["target_source"] = source
    saved = to_dict(env.cfg)
    _validate_saved_policy_semantics(env.cfg, saved)
    saved["observations"]["policy"]["upper_body_target"]["params"]["target_source"] = (
        "reference" if source == "final_goal" else "final_goal"
    )
    with pytest.raises(ValueError, match="observation layout.*target_source"):
        _validate_saved_policy_semantics(env.cfg, saved)


@pytest.mark.parametrize("explicit", [False, True])
def test_metadata_only_checks_target_source_with_legacy_fallback(explicit):
    env = _raw_env()
    infos = _infos(env)
    state = infos["g1_curriculum"]
    state["upper_body_motion"] = {"profile": "quintic_v1"}
    if explicit:
        state["upper_body_target_source"] = "reference"
    with pytest.raises(ValueError, match="upper-body target source"):
        validate_policy_compatibility(env, {}, infos)
    env.cfg.observations.policy.upper_body_target.params["target_source"] = "reference"
    assert validate_policy_compatibility(env, {}, infos) == "checkpoint_metadata"
    state["upper_body_target_source"] = "final_goal"
    env.cfg.observations.policy.upper_body_target.params["target_source"] = "final_goal"
    assert validate_policy_compatibility(env, {}, infos) == "checkpoint_metadata"
