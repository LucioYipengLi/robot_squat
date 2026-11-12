"""Bounded PhysX oracle for WALK rewards, timer reads and terminal-frame diagnostics."""

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, _ = resolve_task_config("Template-G1-v0", "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    cfg.scene.num_envs = 8
    cfg.seed = 42
    cfg.episode_length_s = 0.2
    cfg.commands.task_command.rel_mode_envs = (0.0, 0.0, 1.0, 0.0)
    cfg.commands.task_command.ranges.lin_vel_x = (0.6, 0.6)
    cfg.commands.task_command.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.task_command.ranges.ang_vel_z = (0.0, 0.0)
    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make("Template-G1-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            term, robot = env.command_manager.get_term("task_command"), env.scene["robot"]
            sensor = env.scene.sensors["contact_forces"]
            assert sensor.cfg.force_threshold == 1.0
            reference_cfg = env.reward_manager.get_term_cfg("walk_joint_reference")
            support_cfg = env.reward_manager.get_term_cfg("walk_support_timing")
            ids = reference_cfg.params["asset_cfg"].joint_ids
            assert [robot.joint_names[i] for i in ids] == [
                "left_hip_pitch_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "right_hip_pitch_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
            ]
            foot_ids = reference_cfg.params["sensor_cfg"].body_ids
            assert [sensor.body_names[i] for i in foot_ids] == ["left_ankle_roll_link", "right_ankle_roll_link"]
            assert env.reward_manager.get_term_cfg("track_pelvis_height").weight == 2.0
            assert env.reward_manager.get_term_cfg("vertical_bounce").weight == -0.5
            assert env.reward_manager.get_term_cfg("feet_gait_phase").params["walk_modes"] == (3,)
            original = env.reward_manager.compute
            checks, positive_frames, resets = 0, 0, 0

            def checked_reward(dt):
                nonlocal checks, positive_frames
                air = sensor.data.current_air_time.torch[:, foot_ids]
                contact_time = sensor.data.current_contact_time.torch[:, foot_ids]
                contact = contact_time > 0
                q = robot.data.joint_pos.torch[:, ids]
                q0 = robot.data.default_joint_pos.torch[:, ids]
                target = q0.clone()
                lift = torch.zeros_like(air)
                for leg in range(2):
                    live = (air[:, leg] < 0.4) & ~contact[:, leg]
                    lift[:, leg] = torch.where(
                        live, torch.sin(torch.pi * (air[:, leg] / 0.4).clamp(0, 1)), 0
                    ).clamp_min(0)
                for leg in range(2):
                    signal = lift[:, leg] - lift[:, 1 - leg]
                    target[:, 3 * leg] -= 0.3 * signal
                    target[:, 3 * leg + 1] += 0.5 * lift[:, leg]
                    target[:, 3 * leg + 2] -= 0.12 * signal
                expected_ref = torch.exp(-(q - target).square().mean(-1) / 0.2**2)
                expected_ref *= lift.amax(-1) * (contact.sum(-1) == 1)
                expected_cost = (contact.sum(-1) != 1).float()
                expected_cost += torch.maximum((air / 0.4 - 1).clamp(0, 1), (contact_time / 0.6 - 1).clamp(0, 1)).mean(
                    -1
                )
                buffers = [
                    robot.data.joint_pos.torch,
                    robot.data.default_joint_pos.torch,
                    robot.actuators.target_command.position.torch,
                    env.action_manager.action,
                    env.action_manager.prev_action,
                    sensor.data.current_air_time.torch,
                    sensor.data.current_contact_time.torch,
                ]
                before = [value.clone() for value in buffers]
                for _ in range(2):
                    torch.testing.assert_close(
                        reference_cfg.func(env, **reference_cfg.params), expected_ref, atol=1e-6, rtol=1e-5
                    )
                    torch.testing.assert_close(
                        support_cfg.func(env, **support_cfg.params), expected_cost, atol=1e-6, rtol=1e-5
                    )
                for current, saved in zip(buffers, before, strict=True):
                    torch.testing.assert_close(current, saved, rtol=0, atol=0)
                result = original(dt)
                for name, value, weight in (
                    ("walk_joint_reference", expected_ref, 2.0),
                    ("walk_support_timing", expected_cost, -0.75),
                ):
                    column = env.reward_manager.active_terms.index(name)
                    torch.testing.assert_close(
                        env.reward_manager._step_reward[:, column], value * weight, atol=1e-6, rtol=1e-5
                    )
                positive_frames += int((expected_ref > 0).sum())
                checks += 1
                return result

            env.reward_manager.compute = checked_reward
            action = env.action_manager.get_term("joint_pos")
            with torch.inference_mode():
                for step in range(48):
                    values = torch.zeros(8, 12, device=env.device)
                    side = "left" if (step // 8) % 2 == 0 else "right"
                    values[:, action._joint_names.index(f"{side}_hip_pitch_joint")] = -0.5
                    values[:, action._joint_names.index(f"{side}_knee_joint")] = 1.0
                    obs, _, terminated, truncated, _ = env.step(values)
                    assert obs["policy"].shape == (8, 650) and torch.isfinite(obs["policy"]).all()
                    resets += int((terminated | truncated).sum())
                assert term.performance.walk[0] == 48 * 8
                previous = term.performance.walk.clone()
                saved_air = sensor.data.current_air_time.torch[2:].clone()
                saved_contact = sensor.data.current_contact_time.torch[2:].clone()
                env.reset(env_ids=torch.tensor([0, 1], device=env.device))
                torch.testing.assert_close(term.performance.walk, previous, rtol=0, atol=0)
                torch.testing.assert_close(sensor.data.current_air_time.torch[2:], saved_air, rtol=0, atol=0)
                torch.testing.assert_close(sensor.data.current_contact_time.torch[2:], saved_contact, rtol=0, atol=0)
                assert not reference_cfg.func(env, **reference_cfg.params)[:2].any()
                metrics = term.performance.drain(term._mode_probs)
                assert "Performance/walk/single_stance_fraction" in metrics
                assert "Performance/walk/hip_pitch_rms_rad" in metrics
                assert positive_frames > 0 and resets > 0
            report = {
                "reward_checks": checks,
                "positive_reference_frames": positive_frames,
                "physical_resets": resets,
                "observations": 650,
                "actions": 12,
                "terminal_and_partial_reset_metrics_checked": True,
                "physical_state_targets_and_timers_preserved": True,
                "metrics": metrics,
            }
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        finally:
            env.close()


if __name__ == "__main__":
    main()
