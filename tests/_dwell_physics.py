"""Real PhysX oracle for every-substep vertical-speed RMS and per-environment warmup."""

import argparse
import json
from collections import deque
from pathlib import Path

import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, _ = resolve_task_config(
        "Template-G1-Upper-Mobile-v0", "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"]
    )
    cfg.scene.num_envs, cfg.seed, cfg.episode_length_s = 4, 42, 10.0
    cfg.commands.wrist_pose.resampling_time_range = (100.0, 100.0)
    cfg.commands.wrist_pose.height_task.probability = 0.0
    cfg.commands.wrist_pose.debug_vis = False
    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make("Template-G1-Upper-Mobile-v0", cfg=cfg).unwrapped
        try:
            env.reset()
            action = env.action_manager.get_term("joint_pos")
            command = env.command_manager.get_term("wrist_pose")
            dwell = env.reward_manager.get_term_cfg("dwell").func
            window = action._vertical_speed_window_steps * cfg.decimation
            assert window == 12
            history = [deque(maxlen=window) for _ in range(4)]
            period = []
            recording = False
            original_update = env.scene.update

            def measured_update(dt):
                original_update(dt)
                if recording and len(period) < cfg.decimation:
                    period.append(action._asset.data.body_link_vel_w.torch[:, action._base_body_id, 2].cpu().clone())

            env.scene.update = measured_update
            before = {key: value.clone() for key, value in action.frozen_policy.model.state_dict().items()}
            max_error, full_windows, positive, failures = 0.0, 0, 0, 0
            partial_preserved = False
            for step in range(90):
                if step == 45:
                    previous_rms, previous_full = action.base_vertical_speed_rms()
                    previous_rms, previous_full = previous_rms.clone(), previous_full.clone()
                    env.reset(env_ids=torch.tensor([0], device=env.device))
                    history[0].clear()
                    current_rms, current_full = action.base_vertical_speed_rms()
                    assert not current_full[0] and current_rms[0] == 0
                    torch.testing.assert_close(current_rms[1:], previous_rms[1:], atol=0, rtol=0)
                    torch.testing.assert_close(current_full[1:], previous_full[1:])
                    partial_preserved = True
                command.goal_pose_w.copy_(action._asset.data.body_link_pose_w.torch[:, command.body_indices])
                command.time_left.fill_(torch.inf)
                period.clear()
                recording = True
                _, _, terminated, truncated, _ = env.step(torch.zeros(4, 23, device=env.device))
                recording = False
                assert len(period) == cfg.decimation
                done = (terminated | truncated).cpu()
                failures += int(terminated.sum())
                for i in range(4):
                    history[i].extend(float(sample[i]) for sample in period)
                    if done[i]:
                        history[i].clear()
                rms, full = action.base_vertical_speed_rms()
                expected = torch.tensor(
                    [(sum(v * v for v in values) / len(values)) ** 0.5 if values else 0.0 for values in history],
                    device=env.device,
                )
                expected_full = torch.tensor([len(values) == window for values in history], device=env.device)
                torch.testing.assert_close(rms, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(full, expected_full)
                max_error = max(max_error, float((rms - expected).abs().max()))
                full_windows += int(full.sum())
                position, orientation = command.pose_error()
                pose = (((position <= 0.01) & (orientation <= 0.04)) | ~command.active_mask).all(-1)
                expected_bonus = (
                    pose
                    & full
                    & (expected <= 0.02)
                    & (env.episode_length_buf > 0)
                    & ~env.termination_manager.terminated
                ).float()
                torch.testing.assert_close(dwell(env), expected_bonus)
                positive += int(expected_bonus.sum())
                assert action.mean_base_motion_l2().shape == (4, 4)
            for key, value in action.frozen_policy.model.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
            assert full_windows > 0 and positive > 0
            report = {
                "control_steps": 90,
                "physics_samples_per_window": window,
                "full_window_checks": full_windows,
                "maximum_rms_oracle_error_m_s": max_error,
                "positive_dwell_samples": positive,
                "failures": failures,
                "partial_reset_preserved": partial_preserved,
                "frozen_weights_unchanged": True,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
