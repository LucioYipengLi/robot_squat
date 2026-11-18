"""Drive the public teleop entrypoint with scripted native-window input."""

import json
import sys
from pathlib import Path

import torch

from g1.rl import teleop
from g1.teleop.newton import NewtonInput

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
project = Path(__file__).resolve().parents[1]
checkpoint = project / "logs/rsl_rl/g1_upper_mobile/2026-11-16_00-47-24/model_9999.pt"
checks = []
saved = {}
original_parse = teleop._parse_args
original_poll = NewtonInput.poll
original_close = NewtonInput.close


def parse(argv):
    args = original_parse(argv)
    args.real_time = False
    return args


def poll(self, dt):
    original_poll(self, dt)

    playback = active[0]
    c, term = self.controller, playback.term
    n = len(checks)
    if n == 0:
        import isaaclab.sim as sim_utils

        from g1.tasks.upper_limb_agent.config.g1.utils.collisions import ARM_COLLISION_LINK_NAMES

        stage = sim_utils.get_current_stage()
        for name in ARM_COLLISION_LINK_NAMES:
            collision = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}/collisions")
            assert collision and collision.IsActive()
        saved["arm_collisions_active"] = True
        saved["actor"] = {k: v.detach().cpu().clone() for k, v in playback.policy.state_dict().items()}
        saved["contract"] = term.range_curriculum.contract()
        saved["level"] = term.range_curriculum.level
        assert saved["level"] == 1
    if n == 2:
        from pyglet.window import key

        before = c.poses.clone()
        self.window.activate()
        self.window.dispatch_events()
        self.window.dispatch_event("on_key_press", key.L, 0)
        self.window.dispatch_pending_events()
        assert self.viewer.is_key_down("l")
        original_poll(self, dt)
        torch.testing.assert_close(c.poses[:, 0], before[:, 0] + c.cfg.position_step_m)
        self.window.dispatch_event("on_key_release", key.L, 0)
        self.window.dispatch_pending_events()
        saved["native_key"] = True
    if n == 4:
        c.arm_mode = 0
        c.request("configure")
    if n == 6:
        assert term.mode_id[0] == 0
        playback.update_status()
        assert c.status["Position L/R [m]"].endswith("idle")
    if n in (10, 20, 30, 42):
        c.source = "auto"
        c.category = {10: "regular", 20: "height_shifted", 30: "planar", 42: "mixed"}[n]
        c.arm_mode = 2
        c.request("configure")
    if n in (12, 22, 32):
        assert int(term.task_ids[0]) == {12: 0, 22: 1, 32: 2}[n]
        assert torch.isinf(term.time_left).all()
    if n == 34:
        saved["goal"] = term.goal_pose_w.clone()
        saved["budget"] = term.episode_deadline_steps.clone()
        c.request("retry")
    if n == 35:
        torch.testing.assert_close(term.goal_pose_w, saved["goal"], atol=0, rtol=0)
        torch.testing.assert_close(term.episode_deadline_steps, saved["budget"], atol=0, rtol=0)
    if n == 38:
        term._episode_start_step[:] = playback.raw.common_step_counter - term.episode_deadline_steps
    if n == 39:
        assert int(term.episode_age_steps()[0]) == 0
    if n == 47:
        c.stochastic = True
        c.request("policy")
    if n == 52:
        c.stochastic = False
        c.request("policy")
    if n == 56:
        c.source = "manual"
        c.request("configure")
    if n == 60:
        # Render a real panel into a screenshot; signature/layout failures are
        # detected by the native callback and the resulting image is reviewable.
        from PIL import Image

        frame = self.viewer.get_frame(render_ui=True).numpy()
        Image.fromarray(frame).save(out / "newton_panel.png")
    if n == 64:
        self.viewer.renderer.close()
        saved["window_closed"] = True
    assert term.range_curriculum.contract() == saved["contract"]
    assert term.range_curriculum.level == saved["level"]
    assert term.range_curriculum.evaluation
    assert all(torch.isfinite(v).all() for v in playback.policy.state_dict().values())
    checks.append(n)


def close(self):
    original_close(self)
    assert all(self._panel not in values for values in self.gui._ui_callbacks.values())
    saved["input_closed"] = True
    verify()


from g1.tasks.upper_limb_agent.teleop import UpperPolicyPlayback  # noqa: E402

active = []
original_init = UpperPolicyPlayback.__init__


def init(self, *args, **kwargs):
    kwargs["output_dir"] = out / "session"
    original_init(self, *args, **kwargs)
    active.append(self)


def verify():
    # PhysX teardown may terminate the process directly: verify while the
    # ExitStack is still inside launch_simulation, not after its __exit__.
    assert len(checks) == 65 and saved["input_closed"] and saved["native_key"] and saved["window_closed"]
    for name, value in active[0].policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved["actor"][name], atol=0, rtol=0)
    rows = [json.loads(line) for line in (out / "session/session.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "ended" and "time_out" in row["reasons"] for row in rows)
    assert {row["category_selection"] for row in rows} >= {"regular", "height_shifted", "planar", "mixed"}
    (out / "summary.json").write_text(
        json.dumps(
            {
                "checks": len(checks),
                "level": saved["level"],
                "actor_unchanged": True,
                "arm_collisions_active": saved["arm_collisions_active"],
                "contract_unchanged": True,
                "input_closed": True,
                "native_key": True,
                "window_closed": True,
                "events": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


UpperPolicyPlayback.__init__ = init
NewtonInput.poll = poll
NewtonInput.close = close
teleop._parse_args = parse
try:
    result = teleop.run(
        [
            "--task",
            "Template-G1-Upper-Mobile-v0",
            "--checkpoint",
            str(checkpoint),
            "--max_steps",
            "65",
            "--seed",
            "42",
            "physics=isaacsim_physx",
        ]
    )
    assert result == 0
finally:
    UpperPolicyPlayback.__init__ = original_init
    NewtonInput.poll = original_poll
    NewtonInput.close = original_close
    teleop._parse_args = original_parse
