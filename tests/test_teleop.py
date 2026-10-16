"""Teleop inputs, task lifecycle and current-runner integration without opening Kit."""

import copy
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply

from g1.rl import __main__ as cli
from g1.rl import teleop
from g1.tasks.lower_limb_agent.config.g1.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg
from g1.tasks.lower_limb_agent.config.g1.env_cfg_phase import LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3
from g1.tasks.lower_limb_agent.visualization import PelvisVisualizer
from g1.teleop.controls import CommandController, CommandLimits
from g1.teleop.kit import KitInput


def _controller():
    # Deliberately differ from the old script's hard-coded final curriculum limits.
    ranges = SimpleNamespace(
        height_offset=(-0.1, 0.02),
        lin_vel_x=(-0.2, 0.4),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_z=(-0.3, 0.5),
        pelvis_offset_x=(-0.01, 0.01),
        pelvis_offset_y=(-0.02, 0.02),
    )
    return CommandController(CommandLimits.from_ranges(ranges))


def test_keyboard_uses_restored_ranges_and_edge_triggered_toggles():
    controls = _controller()
    for _ in range(30):
        for key in ("X", "W", "A", "Q", "I", "J"):
            controls.key(key)
    assert controls.sample(0.01) == pytest.approx((-0.1, 0.4, 0.1, 0.5, 0.0, 0.0))
    controls.key("C")
    controls.key("C", repeat=True)
    assert controls.show_pelvis
    assert controls.key("H") == "help"
    controls.key("SPACE")
    controls.key("I")
    controls.key("J")
    assert controls.sample(0.01)[4:] == pytest.approx((0.005, 0.005))
    controls.key("V")
    assert controls.sample(0.01) == (0.0,) * 6


def test_gamepad_is_proportional_integrated_and_debounces_dpad():
    controls = _controller()
    controls.gamepad_event("LSTICK_DOWN", 1.0)
    controls.gamepad_event("LSTICK_RIGHT", 1.0)
    controls.gamepad_event("RSTICK_DOWN", 1.0)
    assert controls.sample(0.1)[:4] == pytest.approx((-0.03, -0.2, 0.0, -0.3))
    controls.gamepad_event("RSTICK_DOWN", 0.0)
    controls.gamepad_event("LSTICK_DOWN", 0.01)  # dead zone
    controls.gamepad_event("LSTICK_RIGHT", 0.0)
    assert controls.sample(0.1)[:4] == pytest.approx((-0.03, 0.0, 0.0, 0.0))
    controls.gamepad_event("DPAD_UP", 1.0)
    controls.gamepad_event("DPAD_UP", 1.0)
    assert controls.sample(0.1)[1] == pytest.approx(0.1)
    controls.gamepad_event("DPAD_UP", 0.0)
    controls.gamepad_event("DPAD_UP", 1.0)
    assert controls.sample(0.1)[1] == pytest.approx(0.2)
    controls.key("X")
    controls.gamepad_event("FACE_SOUTH", 1.0)
    assert controls.sample(0.1) == (0.0,) * 6


def test_walk_clears_pelvis_input_without_replaying_it_after_stopping():
    controls = _controller()
    controls.key("I")
    controls.gamepad_event("LSTICK_UP", 1.0)
    assert controls.sample(0.1)[4:] == (0.0, 0.0)
    controls.clear_gamepad()
    assert controls.sample(0.1) == (0.0,) * 6
    controls.key("W")
    controls.gamepad_event("LSTICK_UP", 1.0)
    controls.key("SPACE")
    assert controls.sample(0.1) == (0.0,) * 6


@pytest.mark.parametrize(
    "config_type", [LowerLimbAgentEnvCfg, LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3]
)
def test_teleop_specializes_evaluation_without_breaking_checkpoint_compatibility(config_type):
    cfg = config_type()
    duration, dt, history = cfg.episode_length_s, cfg.sim.dt, cfg.observations.policy.history_length
    terminations = copy.copy(vars(cfg.terminations))
    cfg.teleop_mode()
    cfg.teleop_mode()  # Idempotent and dynamically dispatches each phase's play_mode().
    assert cfg.scene.num_envs == 1
    assert cfg.terminations.time_out is None
    assert all(getattr(cfg.terminations, name) is value for name, value in terminations.items() if name != "time_out")
    assert cfg.episode_length_s == duration
    assert cfg.sim.dt == dt
    assert cfg.observations.policy.history_length == history
    assert cfg.curriculum is None
    assert cfg.events.upper_body_disturbance is None
    for event in vars(cfg.events).values():
        if event is not None:
            assert event.mode == "reset"
    assert cfg.commands.task_command.debug_vis
    assert not cfg.commands.task_command.pelvis_debug_vis


def _playback_env():
    command = torch.zeros(1, 6)
    calls = []
    term = SimpleNamespace(task_command=command)

    def submit(value):
        command.copy_(value)
        calls.append("submit")

    term.set_manual_command = submit
    inner = SimpleNamespace(device="cpu", step_dt=0.1, command_manager=SimpleNamespace(get_term=lambda _: term))
    policy_inputs, resets = [], []

    class Policy:
        def __call__(self, obs):
            policy_inputs.append(obs.clone())
            return torch.zeros(1, 12)

        def reset(self, dones):
            resets.append(dones.clone())

    def reset():
        calls.append("reset")
        return command.clone(), {}

    def step(actions):
        calls.append("step")
        return command.clone(), None, torch.tensor([1]), {}

    env = SimpleNamespace(unwrapped=inner, reset=reset, step=step)
    return env, Policy(), calls, policy_inputs, resets


def test_manual_playback_resets_to_zero_and_advances_observations_only_with_env_steps():
    env, policy, calls, policy_inputs, resets = _playback_env()
    controls = _controller()
    playback = teleop.ManualPolicyPlayback(env, policy, controls)
    assert calls == ["submit", "reset"]
    controls.key("W")
    playback.step()
    playback.step()
    assert calls == ["submit", "reset", "submit", "step", "submit", "step"]
    assert policy_inputs[0][0, 1] == 0.0
    assert policy_inputs[1][0, 1] == pytest.approx(0.1)
    assert len(resets) == 2
    assert resets[0].tolist() == [1]


def test_geometry_hull_and_foot_corners_use_current_xyzw_convention():
    hull = PelvisVisualizer._convex_hull_2d([[1, 1], [0, 0], [1, 0], [0, 1], [0.5, 0.5], [0, 0]])
    assert hull == [[0, 0], [1, 0], [1, 1], [0, 1]]
    vis = PelvisVisualizer.__new__(PelvisVisualizer)
    vis._foot_body_ids = [0, 1]
    vis._quat_apply = quat_apply
    vis._corner_offsets = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    positions = torch.tensor([[[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]]])
    quats = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
    vis.robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=SimpleNamespace(torch=positions), body_quat_w=SimpleNamespace(torch=quats))
    )
    corners = vis._foot_corners_world()
    torch.testing.assert_close(corners[:4], vis._corner_offsets + positions[0, 0])
    torch.testing.assert_close(corners[4:], vis._corner_offsets + positions[0, 1])


def test_current_cli_defaults_require_no_simulation_launch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["teleop"])
    args = teleop._parse_args(["--task", "Template-G1-v0"])
    assert args.num_envs == 1
    assert args.real_time
    assert args.checkpoint == "latest"
    assert args.visualizer in ("kit", ["kit"])


@pytest.mark.parametrize("extra", [["--num_envs", "2"], ["--viz", "rerun"], ["--train_env_cfg"], ["--max_steps", "0"]])
def test_invalid_teleop_options_fail_before_launch(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["teleop"])
    with pytest.raises(SystemExit) as error:
        teleop._parse_args(["--task", "Template-G1-v0", *extra])
    assert error.value.code == 2


@pytest.mark.parametrize("fail", [False, True])
def test_teleop_entry_delegates_snapshot_and_restores_argv(monkeypatch, fail):
    import isaaclab.utils

    original = sys.argv

    def run(argv, *, snapshot_path):
        assert argv == ["--task", "Template-G1-v0", "--checkpoint", "old.pt"]
        assert sys.argv[1:] == argv
        assert snapshot_path == "legacy.json"
        if fail:
            raise RuntimeError("test failure")
        return 0

    monkeypatch.setattr(teleop, "run", run)
    monkeypatch.setattr(isaaclab.utils, "configure_logging", lambda **kwargs: None)
    monkeypatch.setenv("WORLD_SIZE", "1")
    argv = ["teleop", "--task", "Template-G1-v0", "--checkpoint", "old.pt", "--curriculum-snapshot", "legacy.json"]
    if fail:
        with pytest.raises(RuntimeError, match="test failure"):
            cli.main(argv)
    else:
        assert cli.main(argv) == 0
    assert sys.argv is original


def test_legacy_script_is_import_safe_and_only_dispatches_when_executed(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["play_teleop.py", "--help"])
    script = Path(__file__).parents[1] / "scripts/play_teleop.py"
    runpy.run_path(str(script), run_name="import_test")
    assert calls == []
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    assert calls == [["teleop", "--help"]]


def test_kit_keyboard_subscription_is_released_and_missing_input_is_an_error(monkeypatch):
    released = []
    interface = SimpleNamespace(
        subscribe_to_keyboard_events=lambda keyboard, callback: 7,
        unsubscribe_to_keyboard_events=lambda keyboard, sub: released.append((keyboard, sub)),
    )
    window = SimpleNamespace(get_keyboard=lambda: "keyboard", get_gamepad=lambda _: None)
    carb = ModuleType("carb")
    carb.input = ModuleType("carb.input")
    carb.input.acquire_input_interface = lambda: interface
    omni = ModuleType("omni")
    omni.appwindow = ModuleType("omni.appwindow")
    omni.appwindow.get_default_app_window = lambda: window
    for name, module in {
        "carb": carb,
        "carb.input": carb.input,
        "omni": omni,
        "omni.appwindow": omni.appwindow,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    inputs = KitInput(_controller())
    inputs.close()
    inputs.close()
    assert released == [("keyboard", 7)]
    with pytest.raises(RuntimeError, match="No usable"):
        KitInput(_controller(), devices="gamepad")


@pytest.mark.parametrize("fail,window_closed", [(False, False), (True, False), (False, True)])
def test_complete_teleop_flow_loads_before_limits_and_cleans_up(monkeypatch, fail, window_closed):
    import isaaclab.app

    import isaaclab_rl.entrypoints.backends.play_rsl_rl as upstream_play
    import isaaclab_rl.entrypoints.common as common
    import isaaclab_rl.rsl_rl

    import isaaclab_tasks.utils

    import g1.rl.runner
    import g1.teleop.kit

    events = []
    args = teleop._parse_args(["--task", "Template-G1-v0", "--max_steps", "1"])
    monkeypatch.setattr(teleop, "_parse_args", lambda argv: args)
    cfg = LowerLimbAgentEnvCfg()
    cfg.play_mode()
    agent = PPORunnerCfg()
    agent.seed = 1
    agent.device = "cpu"
    agent.experiment_name = "test"
    monkeypatch.setattr(isaaclab_tasks.utils, "resolve_task_config", lambda *a, **kw: (cfg, agent))
    monkeypatch.setattr(isaaclab.app, "launch_simulation", lambda *a: nullcontext())
    monkeypatch.setattr(common, "pre_launch_video_config", lambda *a: None)
    monkeypatch.setattr(common, "apply_video_recording", lambda *a, **kw: None)
    monkeypatch.setattr(upstream_play, "_resolve_checkpoint", lambda *a: "logs/model.pt")
    monkeypatch.setattr(upstream_play.cli_args, "update_rsl_rl_cfg", lambda *a: agent)
    env, policy, calls, _, _ = _playback_env()
    env.unwrapped.sim = SimpleNamespace(is_running=lambda: not window_closed)
    term = env.unwrapped.command_manager.get_term("task_command")
    term.cfg = copy.deepcopy(cfg.commands.task_command)

    def create(task, config, options, *, convert_marl_to_single_agent):
        assert not convert_marl_to_single_agent
        assert config.terminations.time_out is None
        events.append("create")
        return env

    monkeypatch.setattr(common, "create_isaaclab_env", create)
    monkeypatch.setattr(common, "close_env", lambda env: events.append("env_close"))
    monkeypatch.setattr(isaaclab_rl.rsl_rl, "RslRlVecEnvWrapper", lambda env, **kw: env)

    class Runner:
        def __init__(self, env, config, **kw):
            assert kw["evaluation"] and kw["snapshot_path"] == "legacy.json"

        def load(self, checkpoint):
            assert checkpoint == "logs/model.pt"
            events.append("load")
            term.cfg.ranges.height_offset = (-0.33, 0.04)

        def get_inference_policy(self, **kwargs):
            return policy

    class Input:
        def __init__(self, controller, **kwargs):
            assert controller.limits.bounds[0] == (-0.33, 0.04)
            events.append("input")

        def poll(self):
            if fail:
                raise RuntimeError("step failed")

        def close(self):
            events.append("input_close")

    monkeypatch.setattr(g1.rl.runner, "CurriculumOnPolicyRunner", Runner)
    monkeypatch.setattr(g1.teleop.kit, "KitInput", Input)
    monkeypatch.setattr(teleop.time, "sleep", lambda dt: None)
    if fail:
        with pytest.raises(RuntimeError, match="step failed"):
            teleop.run([], snapshot_path="legacy.json")
    else:
        assert teleop.run([], snapshot_path="legacy.json") == 0
        assert calls.count("step") == (0 if window_closed else 1)
    assert events == ["create", "load", "input", "input_close", "env_close"]


@pytest.mark.parametrize("unsubscribe_fails", [False, True])
def test_gamepad_disconnect_and_shutdown_restore_settings(monkeypatch, unsubscribe_fails):
    import isaaclab.app.settings_manager

    controls = _controller()
    released = []
    window = SimpleNamespace(get_keyboard=lambda: "keyboard", get_gamepad=lambda _: "gamepad")

    def unsubscribe(gamepad, sub):
        released.append((gamepad, sub))
        if unsubscribe_fails:
            raise RuntimeError("native teardown")

    interface = SimpleNamespace(
        subscribe_to_keyboard_events=lambda *args: 1,
        subscribe_to_gamepad_events=lambda *args: 2,
        unsubscribe_to_keyboard_events=lambda *args: released.append(args),
        unsubscribe_to_gamepad_events=unsubscribe,
    )
    carb = ModuleType("carb")
    carb.input = ModuleType("carb.input")
    carb.input.acquire_input_interface = lambda: interface
    omni = ModuleType("omni")
    omni.appwindow = ModuleType("omni.appwindow")
    omni.appwindow.get_default_app_window = lambda: window
    for name, module in {
        "carb": carb,
        "carb.input": carb.input,
        "omni": omni,
        "omni.appwindow": omni.appwindow,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    settings = {"/persistent/app/omniverse/gamepadCameraControl": True}
    adapter = SimpleNamespace(get=settings.get, set=lambda path, value: settings.update({path: value}))
    monkeypatch.setattr(isaaclab.app.settings_manager, "get_settings_manager", lambda: adapter)
    inputs = KitInput(controls)
    assert not settings["/persistent/app/omniverse/gamepadCameraControl"]
    controls.gamepad_event("LSTICK_UP", 1.0)
    window.get_gamepad = lambda _: None
    inputs.poll()
    assert controls.sample(0.1) == (0.0,) * 6
    if unsubscribe_fails:
        with pytest.raises(RuntimeError, match="native teardown"):
            inputs.close()
    else:
        inputs.close()
    assert released == [("gamepad", 2), ("keyboard", 1)]
    assert settings["/persistent/app/omniverse/gamepadCameraControl"]
    inputs.close()
    assert len(released) == 2
