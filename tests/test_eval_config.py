"""Validate reproducible benchmark planning without importing the simulator."""

import copy
import json

import pytest
import yaml

from g1.rl.evaluation.config import (
    fraction_axis_step,
    load_suite,
    mode_episode_quotas,
    suite_config_digest,
    validate_suite,
)
from g1.rl.evaluation.modules import conditions


def test_builtin_suites_use_same_schema_and_independent_exact_mode_quotas():
    core = load_suite("g1_core")
    robustness = load_suite("g1_robustness")
    assert set(core["benchmark"]) == set(robustness["benchmark"])
    assert core["benchmark"]["control_dt"] == 1 / 60
    assert core["benchmark"]["envelope"] == {"depth_frac": 0.8, "vx_floor": 0.2, "wz_floor": 0.3}
    assert core["policy_outputs"] == ["mean"]
    # Mode proportions belong to each benchmark and may be customized independently.
    for config in (core, robustness):
        benchmark = config["benchmark"]
        quotas = mode_episode_quotas(benchmark["episodes_per_seed"], benchmark["mode_weights"])
        assert sum(quotas) == benchmark["episodes_per_seed"]
        assert all(
            (quota == 0) == (weight == 0) for quota, weight in zip(quotas, benchmark["mode_weights"], strict=True)
        )
    core_conditions = conditions(core)
    assert core_conditions == [
        {
            "module": "tracking",
            "condition_id": "arm_0000_waist_0000",
            "axis_step": 0,
            "arm_fraction": 0.0,
            "waist_fraction": 0.0,
        }
    ]
    sweep = conditions(robustness)[1:]
    assert [condition["axis_step"] for condition in sweep] == [200, 300, 400, 555]
    assert all(condition["module"] == "upper_body_sweep" and condition["waist_fraction"] == 0.1 for condition in sweep)
    assert len({(condition["module"], condition["condition_id"]) for condition in conditions(robustness)}) == 5


@pytest.mark.parametrize(
    "weights,expected",
    [
        ([0.25] * 4, [3, 3, 2, 2]),
        ([4, 4, 2, 0], [4, 4, 2, 0]),
        ([0, 1, 0, 0], [0, 10, 0, 0]),
        ([1e308, 1e308, 1e308, 1e308], [3, 3, 2, 2]),
    ],
)
def test_largest_remainder_quotas_preserve_exact_budget_and_zero_modes(weights, expected):
    assert mode_episode_quotas(10, weights) == expected


@pytest.mark.parametrize(
    "episodes,weights",
    [
        (2, [1, 1, 1, 1]),
        (10, [0.999, 0.001, 0, 0]),
        (10, [0, 0, 0, 0]),
        (10, [0.4, -0.1, 0.5, 0.2]),
        (10, [float("nan"), 1, 1, 1]),
        (10, [True, 1, 1, 1]),
        (True, [1, 1, 1, 1]),
        (1.5, [1, 1, 1, 1]),
    ],
)
def test_invalid_or_insufficient_mode_budgets_fail_early(episodes, weights):
    with pytest.raises(ValueError):
        mode_episode_quotas(episodes, weights)


def test_external_yaml_normalization_detaches_values_and_digest_is_order_independent(tmp_path):
    source = load_suite("g1_robustness")
    source["policy_outputs"] = ["mean", "sample"]
    source["modules"][1]["arm_fractions"] = [0.555, 0.2, 0.4, 0.3]
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    result = load_suite(path)
    assert result["modules"][1]["arm_fractions"] == [0.2, 0.3, 0.4, 0.555]
    assert json.loads(json.dumps(result)) == result
    reversed_keys = dict(reversed(list(result.items())))
    assert suite_config_digest(result) == suite_config_digest(reversed_keys)
    assert suite_config_digest(result) == suite_config_digest(validate_suite(result))
    assert len(suite_config_digest(result)) == 64
    source["benchmark"]["seeds"].append(909)
    assert result["benchmark"]["seeds"] == [101, 202, 303]
    changed = copy.deepcopy(result)
    changed["benchmark"]["ranges"]["height_offset"][0] = -0.345
    assert suite_config_digest(changed) != suite_config_digest(result)


@pytest.mark.parametrize(
    "path,value",
    [
        (("typo",), 1),
        (("schema_version",), True),
        (("schema_version",), 2),
        (("suite",), "../outside"),
        (("task",), ""),
        (("policy_outputs",), ["both"]),
        (("policy_outputs",), ["mean", "mean"]),
        (("benchmark", "typo"), 1),
        (("benchmark", "episode_length_s"), 0),
        (("benchmark", "episode_length_s"), 6.001),
        (("benchmark", "control_dt"), 0),
        (("benchmark", "episodes_per_seed"), 0),
        (("benchmark", "mode_weights"), [1, 1, 1]),
        (("benchmark", "seeds"), [101, 101]),
        (("benchmark", "seeds"), [-1]),
        (("benchmark", "seeds"), [True]),
        (("benchmark", "seeds"), [2**31]),
        (("benchmark", "observation_noise"), "false"),
        (("benchmark", "observation_noise"), True),
        (("benchmark", "reset_mode"), "training"),
        (("benchmark", "pelvis_zero_probability"), 1.1),
        (("benchmark", "pelvis_target_speed_mps"), 0),
        (("benchmark", "envelope", "depth_frac"), 0),
        (("benchmark", "envelope", "depth_frac"), 1.1),
        (("benchmark", "envelope", "vx_floor"), -0.1),
        (("benchmark", "envelope", "unexpected"), 1),
        (("benchmark", "ranges", "height_offset"), [0, 0.1]),
        (("benchmark", "ranges", "height_offset"), [0.02, -0.15]),
        (("benchmark", "ranges", "pelvis_offset_x"), [0.01, 0.03]),
        (("benchmark", "ranges", "lin_vel_x"), [-0.3, float("inf")]),
        (("benchmark", "ranges", "extra"), [0, 1]),
        (("benchmark", "thresholds", "height_m"), 0),
        (("benchmark", "thresholds", "extra"), 0.1),
        (("benchmark", "resampling_time_range"), [0, 4]),
        (("benchmark", "disturbance_interval_s"), [3, 1.5]),
        (("benchmark", "disturbance_interval_s"), [float("nan"), 1.5]),
    ],
)
def test_strict_schema_rejects_invalid_or_unknown_values(path, value):
    source = load_suite("g1_core")
    target = source
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_suite(source)


@pytest.mark.parametrize(
    "module",
    [
        {"name": "unknown"},
        {"name": "tracking", "arm_fraction": 0.2, "waist_fraction": 0.1, "extra": True},
        {"name": "tracking", "arm_fraction": -0.2, "waist_fraction": 0.1},
        {"name": "tracking", "arm_fraction": 0.2, "waist_fraction": 1.1},
        {"name": "upper_body_sweep", "arm_fractions": [], "waist_fraction": 0.1},
        {"name": "upper_body_sweep", "arm_fractions": [0.2, 0.2], "waist_fraction": 0.1},
        {"name": "upper_body_sweep", "arm_fractions": [0.2001], "waist_fraction": 0.1},
    ],
)
def test_invalid_module_parameters_fail_before_expansion(module):
    source = load_suite("g1_core")
    source["modules"] = [module]
    with pytest.raises(ValueError):
        conditions(source)


def test_duplicate_modules_missing_fields_and_absent_files_are_actionable(tmp_path):
    source = load_suite("g1_core")
    source["modules"] *= 2
    with pytest.raises(ValueError, match="Duplicate evaluation module"):
        validate_suite(source)
    source = load_suite("g1_core")
    del source["benchmark"]["ranges"]["lin_vel_x"]
    with pytest.raises(ValueError, match="Missing fields"):
        validate_suite(source)
    with pytest.raises(FileNotFoundError, match="suite not found"):
        load_suite(tmp_path / "missing.yaml")


def test_fraction_axis_steps_are_exact_and_policy_output_default_is_explicit():
    assert fraction_axis_step(0) == 0
    assert fraction_axis_step(0.555) == 555
    assert fraction_axis_step(1) == 1000
    source = load_suite("g1_core")
    del source["policy_outputs"]
    assert validate_suite(source)["policy_outputs"] == ["mean"]
    with pytest.raises(ValueError, match="0.001"):
        fraction_axis_step(0.5555)
