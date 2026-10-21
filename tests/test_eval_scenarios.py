"""Evaluation-case replay, isolated amplitude scans and task lifecycle regressions."""

import copy
import random
from types import SimpleNamespace

import pytest
import torch

from g1.tasks.lower_limb_agent.evaluation import (
    EvaluationScenarioController,
    make_case_plan,
    prepare_env_cfg,
)


@pytest.fixture
def benchmark():
    return {
        "control_dt": 1 / 60,
        "upper_body_motion": {"max_velocity": 2.0, "max_acceleration": 8.0, "min_duration_s": 0.5},
        "episode_length_s": 6.0,
        "mode_weights": [0.25] * 4,
        "observation_noise": False,
        "reset_mode": "nominal",
        "ranges": {
            "height_offset": [-0.345, 0.05],
            "lin_vel_x": [-0.3, 0.6],
            "lin_vel_y": [-0.15, 0.15],
            "ang_vel_z": [-0.8, 0.8],
            "pelvis_offset_x": [-0.03, 0.03],
            "pelvis_offset_y": [-0.03, 0.03],
        },
        "resampling_time_range": [0.2, 0.2],
        "disturbance_interval_s": [0.15, 0.15],
        "thresholds": {"height_m": 0.03, "velocity_xy_mps": 0.5, "yaw_radps": 0.4, "pelvis_xy_m": 0.02},
        "envelope": {"depth_frac": 0.8, "vx_floor": 0.2, "wz_floor": 0.3},
        "pelvis_target_speed_mps": 0.02,
        "pelvis_zero_probability": 0.2,
    }


@pytest.mark.parametrize("mode", range(4))
def test_cases_are_independent_of_global_rng_and_preserve_training_semantics(benchmark, mode):
    original_rng = random.getstate()
    first = make_case_plan(benchmark, 42, mode, 77, 17)
    assert original_rng == random.getstate()
    torch_state = torch.get_rng_state()
    random.seed(17)
    torch.manual_seed(18)
    second = make_case_plan(benchmark, 42, mode, 77, 17)
    assert first == second
    assert first != make_case_plan(benchmark, 42, mode, 78, 17)
    torch.set_rng_state(torch_state)
    random.setstate(original_rng)
    commands = list(first.commands.values())
    assert len(commands) > 1
    if mode in (0, 2):
        assert all(command[0] == 0 for command in commands)
    elif mode == 1:
        assert len({command[0] for command in commands}) > 1
    else:
        assert len({command[0] for command in commands}) == 1
        assert benchmark["envelope"]["depth_frac"] * benchmark["ranges"]["height_offset"][0] <= commands[0][0]
    assert len({command[1:4] for command in commands}) == 1
    if mode < 2:
        assert all(command[1:4] == (0.0, 0.0, 0.0) for command in commands)
    else:
        assert all(command[4:] == (0.0, 0.0) for command in commands)


def test_squat_walk_velocity_envelope_at_deepest_height(benchmark):
    benchmark["ranges"]["height_offset"] = [-0.4, -0.32]
    plan = make_case_plan(benchmark, 7, 3, 0, 2)
    command = plan.commands[0]
    assert command[0] == pytest.approx(-0.32)
    assert -0.1 <= command[1] <= 0.2
    assert -0.15 <= command[2] <= 0.15
    assert -0.3 <= command[3] <= 0.3
    assert all(next_command == command for next_command in plan.commands.values())


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_prepare_disables_all_training_randomization_but_preserves_resets(benchmark, phase):
    from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg
    from g1.tasks.lower_limb_agent.config.g1.env_cfg_phase import (
        LowerLimbAgentEnvCfg_Phase2,
        LowerLimbAgentEnvCfg_Phase3,
    )

    cfg_type = (LowerLimbAgentEnvCfg, LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3)[phase - 1]
    cfg = cfg_type()
    training_cfg = cfg_type()
    condition = {"arm_fraction": 0.555, "waist_fraction": 0.1}
    prepare_env_cfg(cfg, benchmark, condition, 17, 99)
    assert cfg.scene.num_envs == 17
    assert cfg.seed == 99
    assert cfg.episode_length_s == 6
    assert cfg.curriculum is None
    assert not cfg.observations.policy.enable_corruption
    assert cfg.terminations.time_out is not None
    assert len(cfg._g1_eval_disturbance_groups) == 2
    for name, event in vars(cfg.events).items():
        if event is not None and hasattr(event, "mode"):
            assert name in ("reset_base", "reset_robot_joints")
            assert event.mode == "reset"
    assert cfg.events.reset_base.params["pose_range"] == {}
    assert cfg.events.reset_base.params["velocity_range"] == {}
    assert cfg.events.reset_robot_joints.params["position_range"] == (0, 0)
    assert cfg.events.reset_robot_joints.params["velocity_range"] == (0, 0)
    command = cfg.commands.task_command
    for name, bounds in benchmark["ranges"].items():
        assert getattr(command.ranges, name) == tuple(bounds)
    assert command.resampling_time_range == (1e9, 1e9)
    assert not command.debug_vis
    assert not command.pelvis_debug_vis
    assert training_cfg.curriculum is not None
    assert training_cfg.events.physics_material is not None
    assert training_cfg.events.upper_body_disturbance is not None


def test_prepare_rejects_unreplayed_observation_noise_and_wrong_control_dt(benchmark):
    cfg = SimpleNamespace(scene=SimpleNamespace(), sim=SimpleNamespace(dt=1 / 120), decimation=2)
    benchmark["observation_noise"] = True
    with pytest.raises(ValueError, match="observation_noise=false"):
        prepare_env_cfg(cfg, benchmark, {}, 1, 42)
    benchmark["observation_noise"] = False
    benchmark["control_dt"] = 0.02
    with pytest.raises(ValueError, match="control_dt"):
        prepare_env_cfg(cfg, benchmark, {}, 1, 42)


class _Entity:
    def __init__(self, joint_ids):
        self.joint_ids = joint_ids

    def resolve(self, scene):
        pass


class _Robot:
    def __init__(self, n):
        root = torch.zeros(n, 7)
        root[:, 2] = 0.75
        self.data = SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=torch.tensor([[0.25, 0.1]]).repeat(n, 1)),
            joint_pos_target=SimpleNamespace(torch=torch.tensor([[0.25, 0.1]]).repeat(n, 1)),
            joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-1.0, 2.0], [-0.5, 0.5]]]).repeat(n, 1, 1)),
            default_root_pose=SimpleNamespace(torch=root),
        )
        self.data.joint_pos = SimpleNamespace(torch=self.data.default_joint_pos.torch.clone())
        self.writes = []
        self.actuators = SimpleNamespace(
            target_command=SimpleNamespace(
                set_position_index=lambda *, value, joint_ids, env_ids=None: self.set_joint_position_target_index(
                    target=value, joint_ids=joint_ids, env_ids=torch.arange(n) if env_ids is None else env_ids
                ),
                set_velocity_index=lambda **kwargs: None,
            )
        )

    def set_joint_position_target_index(self, *, target, joint_ids, env_ids):
        self.writes.append((target.clone(), list(joint_ids), env_ids.clone()))
        self.data.joint_pos_target.torch[env_ids[:, None], torch.tensor(joint_ids)[None, :]] = target


class _Scene(dict):
    def __init__(self, n):
        super().__init__(robot=_Robot(n))
        self.env_origins = torch.zeros(n, 3)


def _controller(benchmark, arm_fraction=0.2, n=2):
    term = SimpleNamespace(
        cfg=SimpleNamespace(asset_name="robot"),
        task_command=torch.zeros(n, 6),
        pelvis_target_offset=torch.zeros(n, 2),
        mode=torch.zeros(n, dtype=torch.long),
        is_default_env=torch.zeros(n, dtype=torch.bool),
        height_command_world=torch.zeros(n, 1),
    )
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        scene=_Scene(n),
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda _: term),
        cfg=SimpleNamespace(_g1_eval_disturbance_groups=[{"asset_cfg": _Entity([0])}, {"asset_cfg": _Entity([1])}]),
    )
    controller = EvaluationScenarioController(env, benchmark, {"arm_fraction": arm_fraction, "waist_fraction": 0.1}, 42)
    from g1.tasks.lower_limb_agent.motion import UpperBodyMotion, UpperBodyMotionCfg

    env.upper_body_motion = UpperBodyMotion(env.scene["robot"], torch.tensor([0, 1]), UpperBodyMotionCfg())
    return controller, env, term


def test_controller_commands_observed_before_action_and_completed_slots_stop_contributing(benchmark):
    controller, env, term = _controller(benchmark)
    controller.start_batch([{"mode": 1, "episode_id": 3}, {"mode": 3, "episode_id": 7}])
    controller.assign_modes(torch.arange(2))
    controller.resample_command(torch.arange(2))
    assert term.mode.tolist() == [1, 3]
    original_squat = term.task_command[0, 0].item()
    for step in range(15):
        env.episode_length_buf[:] = step
        controller.update_commands()  # The command term invokes this before constructing next obs.
        controller.before_step()
        if step == 0:
            controller.complete(torch.tensor([1]))
    assert term.task_command[0, 0] != original_squat
    assert controller.context(1)["command_height_sum"] == pytest.approx(controller.plans[1].commands[0][0])
    assert controller.context(1)["disturbance_count"] == 0
    assert controller.context(0)["disturbance_count"] == 1
    expected = sum(controller.plans[0].commands[0 if step < 12 else 12][0] for step in range(15))
    assert controller.context(0)["command_height_sum"] == pytest.approx(expected)
    assert term.height_command_world[0, 0] == pytest.approx(0.75 + term.task_command[0, 0].item())
    controller.complete(torch.tensor([0]))
    context = copy.deepcopy(controller.context(0))
    env.episode_length_buf.zero_()  # Automatic resets of finished slots are ignored.
    controller.resample_command(torch.arange(2))
    controller.before_step()
    assert controller.context(0) == context


def test_amplitude_scan_reuses_normalized_targets_and_held_target_resets(benchmark):
    low, low_env, _ = _controller(benchmark, 0.2)
    high, high_env, _ = _controller(benchmark, 0.555)
    specs = [{"mode": 2, "episode_id": 0}]
    for controller, env in ((low, low_env), (high, high_env)):
        controller.start_batch(specs)
        controller.resample_command(torch.arange(2))
        step = min(controller.plans[0].disturbances)
        env.episode_length_buf[0] = step
        controller.update_commands()
        controller.before_step()
    assert low.plans == high.plans
    low_target = low_env.upper_body_motion.get_observation_goal()[0]
    high_target = high_env.upper_body_motion.get_observation_goal()[0]
    assert (high_target[0] - 0.25) / (low_target[0] - 0.25) == pytest.approx(0.555 / 0.2)
    assert high_target[1] == low_target[1]
    assert -1 <= high_target[0] <= 2
    low.start_batch(specs)
    assert not low_env.scene["robot"].writes  # Batch assignment leaves reset/execution to the environment.
    assert low.context(0)["disturbance_count"] == 0


@pytest.mark.parametrize("arm_fraction", [0.0, 0.2, 1.0])
def test_training_and_eval_submit_identical_goals_for_identical_samples(benchmark, arm_fraction):
    from g1.tasks.lower_limb_agent.mdp.events import UpperBodyDisturbanceEvent

    controller, env, _ = _controller(benchmark, arm_fraction)
    controller.start_batch([{"mode": 1, "episode_id": 9}, {"mode": 2, "episode_id": 17}])
    groups = [
        {"asset_cfg": SimpleNamespace(name="robot", joint_ids=[0]), "joint_noise_frac": arm_fraction},
        {"asset_cfg": SimpleNamespace(name="robot", joint_ids=[1]), "joint_noise_frac": 0.1},
    ]
    env.command_manager.active_terms = []
    event = UpperBodyDisturbanceEvent(SimpleNamespace(params={"disturb_groups": groups}), env)
    with torch.random.fork_rng(devices=[]):
        initial_rng = torch.get_rng_state()
        samples = torch.rand(2, 2)
        expected_rng = torch.get_rng_state()
        # Reverse slot order to also check case/slot alignment.
        for row, slot in enumerate((1, 0)):
            controller.plans[slot].disturbances = {1: tuple(samples[row].tolist())}
        torch.set_rng_state(initial_rng)
        controller._publish_disturbances([(1, 1), (0, 1)])
        assert torch.equal(torch.get_rng_state(), initial_rng)
        expected_goals = env.upper_body_motion.get_observation_goal().clone()
        env.upper_body_motion.reset(torch.arange(2))
        event(env, torch.tensor([1, 0]), groups)
        torch.testing.assert_close(env.upper_body_motion.get_observation_goal(), expected_goals, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)


def test_case_is_independent_of_slot_batch_and_other_env_failure(benchmark):
    first, env, term = _controller(benchmark)
    first.start_batch([{"mode": 1, "episode_id": 9}, {"mode": 2, "episode_id": 17}])
    first.resample_command(torch.arange(2))
    saved = first.plans[1]
    first.complete(torch.tensor([0]))
    env.episode_length_buf[:] = 12
    first.update_commands()
    first.before_step()
    second, _, _ = _controller(benchmark)
    second.start_batch([{"mode": 2, "episode_id": 17}])
    assert second.plans[0] == saved
    with pytest.raises(RuntimeError, match="exactly once"):
        first.before_step()
    assert term.task_command[1, 4:].tolist() == [0, 0]


def test_upper_body_feedforward_target_is_published_before_policy_observation(benchmark):
    controller, env, _ = _controller(benchmark)
    controller.start_batch([{"mode": 2, "episode_id": 11}])
    controller.resample_command(torch.arange(2))
    event_step = min(controller.plans[0].disturbances)
    env.episode_length_buf[0] = event_step - 1
    controller.update_commands()
    default_target = env.upper_body_motion.get_observation_goal().clone()
    env.episode_length_buf[0] = event_step
    controller.update_commands()  # Command-manager compute precedes observation-manager compute.
    observed_target = env.upper_body_motion.get_observation_goal().clone()
    assert not torch.equal(default_target[0], observed_target[0])
    torch.testing.assert_close(default_target[1], observed_target[1])
    assert controller.context(0)["disturbance_count"] == 0
    controller.update_commands()  # Publishing the same observed state again is idempotent.
    controller.before_step()  # The policy now acts on the already-published cue.
    torch.testing.assert_close(env.upper_body_motion.get_observation_goal(), observed_target)
    assert controller.context(0)["disturbance_count"] == 1
    env.episode_length_buf[0] += 1
    controller.update_commands()
    controller.before_step()
    assert controller.context(0)["disturbance_count"] == 1
