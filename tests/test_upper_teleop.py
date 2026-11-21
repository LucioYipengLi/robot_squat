"""Pose editing, native input lifecycle and inference ordering for Upper teleop."""

import json
from types import SimpleNamespace

import pytest
import torch
from test_planar_timing import make_command

from g1.rl.teleop import _parse_args
from g1.tasks.upper_limb_agent.config.g1.env_cfg import UpperLimbAgentEnvCfg
from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import UpperLimbMobileEnvCfg
from g1.tasks.upper_limb_agent.teleop import UpperPolicyPlayback
from g1.teleop.newton import NewtonInput
from g1.teleop.wrist_controls import WristController


def test_pair_translation_rotation_and_fine_edits_preserve_geometry():
    c = WristController()
    c.poses[0, 1], c.poses[1, 1] = 0.15, -0.15
    original = c.poses.clone()
    c.nudge(0, 1)
    torch.testing.assert_close(c.poses[:, 0], torch.full((2,), 0.01))
    center = c.poses[:, :3].mean(0).clone()
    c.nudge(2, 1, rotate=True)
    torch.testing.assert_close(c.poses[:, :3].mean(0), center)
    assert (c.poses[0, :3] - c.poses[1, :3]).norm() == pytest.approx(0.3)
    assert c.angles_deg(0)[2] == pytest.approx(2.0, abs=1e-4)
    c.arm_mode = 0
    c.fine = True
    before = c.poses.clone()
    c.nudge(2, 1)
    assert c.poses[0, 2] - before[0, 2] == pytest.approx(0.001)
    torch.testing.assert_close(c.poses[1], before[1])
    assert c.set_pose(0, [0.1, 0.2, 0.3], [10, 20, 30])
    assert c.angles_deg(0) == pytest.approx([10, 20, 30], abs=1e-4)
    assert not c.set_pose(0, [float("nan"), 0, 0], [0, 0, 0])
    c.source = "auto"
    before = c.poses.clone()
    c.nudge(0, 1)
    torch.testing.assert_close(c.poses, before)
    assert original[0, 0] == 0
    c.arm_mode = None
    c.key("M")
    assert c.source == "manual" and c.arm_mode == 2


def test_panel_cleanup_still_runs_if_native_handler_removal_fails():
    inputs, _, viewer, _, _ = _input()

    def fail(**kwargs):
        raise RuntimeError("native window closed")

    inputs.window.remove_handlers = fail
    with pytest.raises(RuntimeError, match="native window"):
        inputs.close()
    assert not viewer.gui._ui_callbacks["side"]


def _input():
    keys, handlers = set(), {}
    callbacks = {"side": []}
    gui = SimpleNamespace(_ui_callbacks=callbacks, capturing=False)
    gui.is_keyboard_capturing = lambda: gui.capturing
    window = SimpleNamespace(
        push_handlers=lambda **kw: handlers.update(kw),
        remove_handlers=lambda **kw: [handlers.pop(k) for k in kw],
    )
    viewer = SimpleNamespace(
        gui=gui,
        renderer=SimpleNamespace(window=window),
        paused=False,
        is_key_down=lambda key: key in keys,
        register_ui_callback=lambda callback, position: callbacks[position].append(callback),
    )
    viewer.is_paused = lambda: viewer.paused
    sim = SimpleNamespace(
        visualizers=[SimpleNamespace(cfg=SimpleNamespace(visualizer_type="newton_gl"), _viewer=viewer)]
    )
    controls = WristController()
    return NewtonInput(sim, controls), controls, viewer, keys, handlers


def test_native_input_debounces_modes_repeats_edits_and_blocks_focus_and_ui_capture():
    inputs, c, viewer, keys, handlers = _input()
    keys.add("L")
    inputs.poll(0.01)
    assert c.poses[0, 0] == pytest.approx(0.01)
    inputs.poll(0.01)
    assert c.poses[0, 0] == pytest.approx(0.01)
    inputs.poll(0.2)
    assert c.poses[0, 0] == pytest.approx(0.02)
    handlers["on_deactivate"]()
    inputs.poll(1)
    handlers["on_activate"]()
    inputs.poll(1)
    assert c.poses[0, 0] == pytest.approx(0.02)
    keys.clear()
    inputs.poll(0.01)
    keys.add("L")
    viewer.gui.capturing = True
    inputs.poll(1)
    viewer.gui.capturing = False
    inputs.poll(1)
    assert c.poses[0, 0] == pytest.approx(0.02)
    keys.clear()
    inputs.poll(0.01)
    keys.add("M")
    inputs.poll(0.01)
    inputs.poll(1)
    assert c.source == "auto" and list(c.requests) == ["configure"]
    inputs.close()
    inputs.close()
    assert not handlers and not viewer.gui._ui_callbacks["side"]


@pytest.mark.parametrize("task", ["Template-G1-Upper-v0", "Template-G1-Upper-Mobile-v0"])
def test_upper_cli_defaults_and_eval_configuration(task):
    args = _parse_args(["--task", task])
    assert args.visualizer in ("newton_gl", ["newton_gl"])
    assert args.input == "keyboard" and args.num_envs == 1
    cfg = UpperLimbMobileEnvCfg() if "Mobile" in task else UpperLimbAgentEnvCfg()
    seconds = cfg.episode_length_s
    cfg.teleop_mode()
    assert cfg.scene.num_envs == 1 and cfg.episode_length_s == seconds
    assert cfg.terminations.time_out.func.__name__ == "task_time_out"
    if "Mobile" in task:
        assert cfg.terminations.fallen is not None and cfg.terminations.base_contact is not None


@pytest.mark.parametrize("extra", [["--viz", "kit"], ["--input", "gamepad"], ["--show_pelvis"]])
def test_upper_unsupported_input_fails_before_launch(extra):
    with pytest.raises(SystemExit):
        _parse_args(["--task", "Template-G1-Upper-Mobile-v0", *extra])


def _session(tmp_path):
    def one_env(cfg, env):
        env.num_envs = 1
        env.scene.env_origins = env.scene.env_origins[:1]
        for name in ("body_link_pose_w", "joint_pos", "default_joint_pos"):
            data = getattr(env.scene["robot"].data, name)
            data.torch = data.torch[:1]
        for name in ("episode_length_buf", "reset_terminated", "reset_time_outs"):
            setattr(env, name, getattr(env, name)[:1])

    term = make_command(probability=0.2, configure=one_env)
    raw = term._env
    raw.action_manager = SimpleNamespace(active_terms=[])
    calls, inputs = [], []

    class Policy:
        def __call__(self, obs, **kwargs):
            inputs.append((obs.clone(), kwargs))
            return torch.zeros(1, 23)

        def reset(self, dones=None):
            calls.append("policy_reset")

    def reset():
        calls.append("reset")
        raw.episode_length_buf.zero_()
        term.reset()
        return term.goal_pose_w.clone(), {}

    def step(actions):
        calls.append("step")
        raw.common_step_counter += 1
        raw.episode_length_buf += 1
        term._update_command()
        return term.goal_pose_w.clone(), None, torch.tensor([False]), {}

    env = SimpleNamespace(unwrapped=raw, reset=reset, step=step)
    c = WristController()
    session = UpperPolicyPlayback(env, Policy(), c, checkpoint="model.pt", seed=42, output_dir=tmp_path)
    return session, c, term, calls, inputs


def test_playback_preserves_one_step_observation_timing_and_can_retry_and_switch_categories(tmp_path):
    session, c, term, calls, inputs = _session(tmp_path)
    initial = session.obs.clone()
    c.nudge(0, 1)
    session.step()
    torch.testing.assert_close(inputs[-1][0], initial)
    session.step()
    assert inputs[-1][0][0, 0, 0] == pytest.approx(float(initial[0, 0, 0]) + 0.01)
    c.stochastic = True
    c.source, c.category = "auto", "planar"
    c.request("configure")
    session.step()
    assert bool(term.planar_task_mask[0]) and inputs[-1][1] == {"stochastic_output": True}
    goal, deadline = term.goal_pose_w.clone(), term.episode_deadline_steps.clone()
    c.request("retry")
    session.step()
    torch.testing.assert_close(term.goal_pose_w, goal)
    torch.testing.assert_close(term.episode_deadline_steps, deadline)
    assert term.range_curriculum.evaluation and term.range_curriculum.level == 0
    session.close()
    rows = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()]
    assert {row["event"] for row in rows} >= {"session", "target", "edit", "interrupted", "closed"}
    assert len([row for row in rows if row["event"] == "target"]) == 3


def test_manual_targets_survive_physical_reset_and_can_be_applied_together(tmp_path):
    session, c, term, _, _ = _session(tmp_path)
    reset_goals = term.goal_pose_w.clone()
    c.set_pose(0, [0.2, 1.2, 0.8], [10, 0, 90])
    session.step()
    edited = c.poses.clone()
    normal_step = session.env.step

    def contact_step(actions):
        normal_step(actions)
        obs, _ = session.env.reset()
        session.env.step = normal_step
        return obs, None, torch.tensor([True]), {"log": {"Episode_Termination/base_contact": 1.0}}

    session.env.step = contact_step
    session.step()
    torch.testing.assert_close(c.poses, edited)
    assert "targets kept" in c.status["Stage"]
    session.step()
    torch.testing.assert_close(term.goal_pose_w, reset_goals)

    c.set_pose(1, [-0.2, 1.1, 0.8], [0, 0, 90])
    session.step()
    torch.testing.assert_close(term.goal_pose_w[0, 0], edited[0])
    torch.testing.assert_close(term.goal_pose_w[0, 1], c.poses[1])
    edited = c.poses.clone()
    session.env.step = contact_step
    session.step()
    c.request("retry")
    session.step()
    torch.testing.assert_close(term.goal_pose_w[0], edited)
    session.close()

    rows = [json.loads(line) for line in (tmp_path / "session.jsonl").read_text().splitlines()]
    reset = next(row for row in rows if "retained_manual_goal_pose_w" in row)
    torch.testing.assert_close(torch.tensor(reset["retained_manual_goal_pose_w"][0][0]), edited[0])
    torch.testing.assert_close(torch.tensor(reset["target"]["goal_pose_w"]), reset_goals)
