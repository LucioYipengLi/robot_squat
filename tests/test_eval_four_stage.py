"""Four-stage budgets, paired inputs, terminal exposure and exact RMSE reductions."""

import copy
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from g1.rl.evaluation import cli
from g1.rl.evaluation.config import load_suite, suite_config_digest, validate_suite
from g1.rl.evaluation.four_stage import STAGES, benchmark_for_job, protocol_plan, protocol_specs
from g1.rl.evaluation.modules import conditions
from g1.rl.evaluation.reporting import summarize_results, write_report
from g1.tasks.lower_limb_agent.evaluation_four_stage import FourStageScenarioController


def _config():
    config = load_suite("g1_lower_four_stage")
    config["benchmark"]["repetitions"] = 1
    return config


def _benchmark(config, stage, count=3):
    return benchmark_for_job(config, {"module": stage}, {"num_envs": count})


def test_budget_contiguous_groups_and_revalidation():
    config = _config()
    config["benchmark"]["repetitions"] = 2
    assert validate_suite(config) == config
    assert suite_config_digest(config) == suite_config_digest(copy.deepcopy(config))
    assert [x["module"] for x in conditions(config)] == list(STAGES)
    for stage in STAGES:
        benchmark = _benchmark(config, stage, 7)
        specs = protocol_specs(benchmark)
        assert len(specs) == benchmark["episodes_per_seed"] == 14
        assert [x["repetition"] for x in specs] == [0] * 7 + [1] * 7
        if stage == "walk_axes":
            assert [x["subtask"] for x in specs[:7]] == ["straight"] * 3 + ["turn"] * 2 + ["lateral"] * 2
        if stage == "squat_push":
            assert {x["height_offset_m"] for x in specs[:7]} == set(benchmark["push_height_offsets"])
    with pytest.raises(ValueError, match="num_envs"):
        _benchmark(config, STAGES[0], 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("repetitions", 0),
        ("episode_length_s", 0.01),
        ("push_height_offsets", [-9]),
        ("push_windows_fraction", [[0.1, 0.9], [0.2, 0.3]]),
        ("squat_walk_use_envelope", "yes"),
    ],
)
def test_invalid_protocol_fails_before_launch(field, value):
    config = _config()
    config["benchmark"][field] = value
    with pytest.raises(ValueError):
        validate_suite(config)


def test_zero_normalization_scale_rejected():
    config = _config()
    config["benchmark"]["ranges"]["lin_vel_y"] = [0, 0]
    with pytest.raises(ValueError, match="normalization"):
        validate_suite(config)


def test_fixed_height_default_and_nonfinite_step_count():
    config = _config()
    config["benchmark"]["ranges"]["height_offset"] = [-0.2, -0.2]
    config["benchmark"].pop("push_height_offsets")
    assert validate_suite(config)["benchmark"]["push_height_offsets"] == [-0.2]
    config["benchmark"].update(control_dt=1e-308, episode_length_s=1e308)
    with pytest.raises(ValueError, match="control steps"):
        validate_suite(config)


def test_two_walk_samples_axis_isolation_and_rng_independence():
    benchmark = _benchmark(_config(), "walk_axes")
    state = random.getstate()
    for spec in protocol_specs(benchmark):
        commands, pushes = protocol_plan(benchmark, 17, spec)
        assert not pushes and list(commands) == [0, 180]
        assert commands[0] != commands[180]
        for command in commands.values():
            assert command[0] == 0 and command[4:] == (0, 0)
            if spec["subtask"] == "straight":
                assert command[2:4] == (0, 0)
            elif spec["subtask"] == "turn":
                assert command[2] == 0
            else:
                assert command[1] == command[3] == 0
        assert (commands, pushes) == protocol_plan(benchmark, 17, spec)
    assert random.getstate() == state


def test_squat_cycle_exact_squat_walk_range_and_paired_pushes():
    config = _config()
    squat = _benchmark(config, "squat_tracking")
    commands, _ = protocol_plan(squat, 17, protocol_specs(squat)[0])
    assert list(commands) == [0, 120, 240]
    assert commands[0] == commands[240] == (0,) * 6
    assert commands[120][1:] == (0,) * 5
    config["benchmark"]["ranges"]["height_offset"] = [-0.3, -0.3]
    config["benchmark"]["push_height_offsets"] = [-0.3]
    walk = _benchmark(validate_suite(config), "squat_walk_tracking")
    commands, _ = protocol_plan(walk, 17, protocol_specs(walk)[0])
    assert list(commands) == [0] and commands[0][0] == -0.3
    push = _benchmark(_config(), "squat_push", 6)
    plans = [protocol_plan(push, 17, spec) for spec in protocol_specs(push)]
    assert all(plan[1] == plans[0][1] for plan in plans)
    for commands, pushes in plans:
        assert len(pushes) == 2
        assert commands[0][1:] == (0,) * 5
        assert all(-0.1 <= value <= 0.1 for delta in pushes.values() for value in delta)


def _case(config, stage, seed=101):
    benchmark = _benchmark(config, stage)
    records = []
    for spec in protocol_specs(benchmark):
        record = {
            **spec,
            "mode": ("stand", "squat", "walk", "squat_walk")[spec["mode"]],
            "metric_steps": 360,
            "steps": 360,
            "length_s": 6,
            "terminated": False,
            "time_out": True,
            "reasons": {"bad_ori": False, "time_out": True},
            "max_height_error_m": 0.02,
            "attitude_quaternion_order": "xyzw",
            "max_roll_rad": 0.01,
            "max_pitch_rad": 0.01,
            "height_squared_error_sum": 360 * 0.02**2,
            "vx_squared_error_sum": 0,
            "vy_squared_error_sum": 0,
            "wz_squared_error_sum": 0,
            "planned_push_count": 2 if stage == "squat_push" else 0,
            "push_count": 0,
            "pushes": [],
        }
        records.append(record)
    return {
        "schema_version": 1,
        "checkpoint": {"label": "model", "path": "/model.pt", "sha256": "abc"},
        "module": stage,
        "condition_id": stage,
        "axis_step": 0,
        "arm_fraction": 0,
        "waist_fraction": 0,
        "seed": seed,
        "policy_output": "mean",
        "metadata": {"num_envs": 3},
        "episodes": records,
    }


def test_walk_score_uses_only_its_own_group_and_pools_squares():
    config = _config()
    case = _case(config, "walk_axes")
    for record in case["episodes"]:
        # Extremely bad off-task channels must not enter the three selected RMSEs.
        for axis in ("vx", "vy", "wz"):
            record[axis + "_squared_error_sum"] = 360 * 999**2
        axis, error = {"straight": ("vx", 0.06), "turn": ("wz", 0.08), "lateral": ("vy", 0.015)}[record["subtask"]]
        record[axis + "_squared_error_sum"] = 360 * error**2
    rows = summarize_results(config, [case])
    aggregate = next(x for x in rows if x["mode"] == "aggregate")
    assert aggregate["e_walk_percent"] == pytest.approx(10)
    assert aggregate["group_rmse_vx_mps"] == pytest.approx(0.06)
    case = _case(config, "squat_tracking")
    # One early failure, two long episodes: sqrt(sum SSE / sum N), not mean of RMSE.
    case["episodes"][0].update(
        metric_steps=60, steps=60, length_s=1, terminated=True, time_out=False, height_squared_error_sum=60 * 0.2**2
    )
    row = summarize_results(config, [case])[0]
    assert row["e_squat_m"] == pytest.approx(math.sqrt((60 * 0.2**2 + 720 * 0.02**2) / 780))
    assert row["failure_rate"] == pytest.approx(1 / 3)


def test_height_groups_incomplete_seeds_and_real_report_events(tmp_path):
    config = _config()
    results = [_case(config, stage) for stage in STAGES]
    report = write_report(tmp_path, config, results)
    document = json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    heights = [row for row in document["rows"] if row["module"] == "squat_push" and row["mode"] == "height"]
    assert len(heights) == 3 and all(row["episode_count"] == 1 for row in heights)
    events = EventAccumulator(str(Path(report["tensorboard_dir"]) / "walk_axes/mean/model"))
    events.Reload()
    assert events.Scalars("00 四阶段指标/aggregate/e_walk_percent")[0].value == 0
    config["benchmark"]["seeds"].append(202)
    assert summarize_results(config, results) == []
    with pytest.raises(ValueError, match="budget"):
        summarize_results(_config(), [results[0] | {"episodes": results[0]["episodes"][:-1]}])


def _controller(stage="squat_push"):
    from test_eval_scenarios import _controller as legacy_controller

    config = _config()
    benchmark = _benchmark(config, stage)
    _, env, term = legacy_controller(benchmark, n=3)
    data = env.scene["robot"].data
    data.root_vel_w = SimpleNamespace(torch=torch.zeros(3, 6))
    data.root_lin_vel_b = SimpleNamespace(torch=torch.zeros(3, 3))
    data.root_ang_vel_b = SimpleNamespace(torch=torch.zeros(3, 3))
    data.root_quat_w = SimpleNamespace(torch=torch.tensor([[0, 0, math.sqrt(0.5), math.sqrt(0.5)]]).repeat(3, 1))
    writes = []

    def write_velocity(*, root_velocity, env_ids):
        data.root_vel_w.torch[env_ids] = root_velocity
        writes.append((env_ids.clone(), root_velocity.clone()))

    env.scene["robot"].write_root_velocity_to_sim_index = write_velocity
    controller = FourStageScenarioController(env, benchmark, conditions(config)[STAGES.index(stage)], 101)
    specs = protocol_specs(benchmark)
    controller.start_batch(specs)
    controller.assign_modes(torch.arange(3))
    controller.resample_command(torch.arange(3))
    return controller, env, term, writes


def test_push_addition_is_idempotent_and_failure_stops_future_events():
    controller, env, term, writes = _controller()
    env.scene["robot"].data.root_vel_w.torch[:] = 0.5
    first, second = controller.plans[0].pushes
    env.episode_length_buf[:] = first
    controller.update_commands()
    controller.update_commands()
    assert len(writes) == 1
    assert torch.all(writes[0][1][:, 2:] == 0.5)
    delta = torch.tensor(controller.plans[0].pushes[first])
    assert torch.allclose(writes[0][1][0, :2], 0.5 + delta)
    controller.before_step()
    controller.complete(torch.tensor([0]))
    env.episode_length_buf[:] = second
    controller.update_commands()
    controller.before_step()
    assert controller.context(0)["push_count"] == 1
    assert controller.context(1)["push_count"] == 2
    assert controller.context(0)["planned_push_count"] == 2
    assert writes[-1][0].tolist() == [1, 2]


def test_squared_metrics_include_terminal_sample_and_recovery_requires_hold():
    controller, env, term, _ = _controller()
    step = next(iter(controller.plans[0].pushes))
    env.episode_length_buf[:] = step
    controller.update_commands()
    for offset in range(1, 13):
        env.episode_length_buf[:] = step + offset
        controller.sample_metrics(torch.ones(3, dtype=torch.bool), torch.full((3,), 0.01))
    context = controller.context(0)
    assert context["metric_steps"] == 12
    assert context["height_squared_error_sum"] == pytest.approx(12 * 0.01**2)
    assert context["pushes"][0]["recovery_s"] == pytest.approx(0.2)
    controller.complete(torch.tensor([0]))
    controller.sample_metrics(torch.ones(3, dtype=torch.bool), torch.ones(3))
    assert controller.context(0)["height_squared_error_sum"] == context["height_squared_error_sum"]
    controller.start_batch(protocol_specs(controller.benchmark))
    assert controller.context(0)["metric_steps"] == controller.context(0)["push_count"] == 0


def test_xyzw_angles_match_known_tilt_and_yaw_does_not_count_as_roll():
    from isaaclab.utils.math import quat_from_euler_xyz

    controller, env, _, _ = _controller()
    controller.sample_metrics(torch.ones(3, dtype=torch.bool), torch.zeros(3))
    assert controller.context(0)["max_roll_rad"] == pytest.approx(0, abs=1e-6)
    assert controller.context(0)["max_pitch_rad"] == pytest.approx(0, abs=1e-6)
    roll, pitch = torch.tensor([0.2, -0.3, 0.05]), torch.tensor([-0.1, 0.15, 0.02])
    env.scene["robot"].data.root_quat_w.torch[:] = quat_from_euler_xyz(roll, pitch, torch.tensor([1.57, -2.0, 0.0]))
    controller.sample_metrics(torch.ones(3, dtype=torch.bool), torch.zeros(3))
    for slot in range(3):
        assert controller.context(slot)["max_roll_rad"] == pytest.approx(abs(roll[slot].item()), abs=1e-6)
        assert controller.context(slot)["max_pitch_rad"] == pytest.approx(abs(pitch[slot].item()), abs=1e-6)
        assert controller.context(slot)["attitude_quaternion_order"] == "xyzw"


def test_legacy_recordings_do_not_report_invalid_roll_or_recovery():
    case = _case(_config(), "squat_push")
    for episode in case["episodes"]:
        episode.pop("attitude_quaternion_order")
    rows = summarize_results(_config(), [case])
    assert all(row["attitude_metrics_valid"] == 0 for row in rows)
    for row in rows:
        assert "max_roll_rad" not in row
        assert "push_recovery_rate" not in row
        assert "unrecovered_push_count" not in row
        assert row["rmse_height_m"] == pytest.approx(0.02)


def test_cli_overrides_make_four_cells_and_exact_budget(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"plan only")
    output = tmp_path / "plan"
    assert (
        cli.run(
            [
                "--suite",
                "g1_lower_four_stage",
                "--checkpoints",
                str(checkpoint),
                "--num_envs",
                "7",
                "--repetitions",
                "2",
                "--height-range",
                "-0.2",
                "0",
                "--vx-range",
                "-0.4",
                "0.8",
                "--output",
                str(output),
                "--plan-only",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["jobs"]) == 4
    assert manifest["config"]["benchmark"]["push_height_offsets"] == [-0.2, -0.1, 0]
    assert all(
        benchmark_for_job(manifest["config"], job, manifest["runtime"])["episodes_per_seed"] == 14
        for job in manifest["jobs"]
    )
