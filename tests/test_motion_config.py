"""Versioned benchmark profiles and narrow legacy action compatibility."""

import copy

import pytest

from isaaclab.utils import to_dict

from g1.rl.evaluation.config import load_suite, suite_config_digest, validate_suite
from g1.rl.evaluation.policy import _validate_saved_policy_semantics
from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg


def test_v1_preserves_step_baseline_and_v2_records_quintic_settings():
    for name in ("g1_core", "g1_robustness"):
        old, new = (load_suite(f"{name}_v{version}") for version in (1, 2))
        assert old["benchmark"]["upper_body_motion"]["profile"] == "step_v1"
        assert new["benchmark"]["upper_body_motion"]["profile"] == "quintic_v1"
        assert old["modules"] == new["modules"]
        changed = copy.deepcopy(new)
        changed["benchmark"]["upper_body_motion"]["max_velocity"] *= 2
        assert suite_config_digest(changed) != suite_config_digest(new)


@pytest.mark.parametrize("name", ["max_velocity", "max_acceleration", "min_duration_s"])
@pytest.mark.parametrize("value", [0, -1, True, float("nan")])
def test_invalid_motion_limits_are_rejected_during_planning(name, value):
    suite = load_suite("g1_core_v2")
    suite["benchmark"]["upper_body_motion"][name] = value
    with pytest.raises(ValueError, match="upper_body_motion"):
        validate_suite(suite)


def test_known_legacy_action_class_is_compatible_but_scale_remains_strict():
    cfg = LowerLimbAgentEnvCfg()
    saved = to_dict(cfg)
    saved["actions"]["joint_pos"]["class_type"] = "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction"
    _validate_saved_policy_semantics(cfg, saved)
    saved["actions"]["joint_pos"]["scale"] = 0.7
    with pytest.raises(ValueError, match="action layout"):
        _validate_saved_policy_semantics(cfg, saved)


def test_target_source_is_explicit_in_benchmark_digest_and_reaches_environment():
    from g1.tasks.lower_limb_agent.evaluation import prepare_env_cfg

    suite = load_suite("g1_core_v2")
    assert suite["benchmark"]["upper_body_target_source"] == "final_goal"
    reference = copy.deepcopy(suite)
    reference["benchmark"]["upper_body_target_source"] = "reference"
    assert suite_config_digest(validate_suite(reference)) != suite_config_digest(suite)
    cfg = LowerLimbAgentEnvCfg()
    prepare_env_cfg(cfg, reference["benchmark"], {"arm_fraction": 0.2, "waist_fraction": 0.1}, 4, 42)
    assert cfg.observations.policy.upper_body_target.params["target_source"] == "reference"


@pytest.mark.parametrize("source", [None, True, "goal", ["final_goal"]])
def test_invalid_observation_source_is_rejected_during_planning(source):
    suite = load_suite("g1_core_v2")
    suite["benchmark"]["upper_body_target_source"] = source
    with pytest.raises(ValueError, match="upper_body_target_source"):
        validate_suite(suite)
