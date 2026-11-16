"""Independent per-step integration of window scores and reset reward rates."""

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import launch_simulation
from isaaclab.managers import TerminationTermCfg

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401


def fail_once(env):
    return (torch.arange(env.num_envs, device=env.device) == 2) & (env.common_step_counter == 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, _ = resolve_task_config(
        "Template-G1-Upper-Mobile-v0", "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"]
    )
    cfg.scene.num_envs, cfg.seed, cfg.episode_length_s = 4, 42, 35
    wrist = cfg.commands.wrist_pose
    wrist.mode_probabilities = (0, 0, 1)
    wrist.debug_vis = False
    wrist.planar_task.probability = 0.5
    wrist.planar_task.initial_half_extent_m = wrist.planar_task.half_extent_m = 1e-6
    wrist.planar_task.initial_bearing_rad = wrist.planar_task.bearing_rad = 0
    wrist.planar_task.timing.settle_s = 0.01
    wrist.planar_task.timing.evaluation_s = 0.05
    wrist.planar_task.timing.near_episode_s = 0.1
    wrist.range_curriculum.position_tolerance_m = 1.0
    wrist.range_curriculum.orientation_tolerance_rad = 3.1
    wrist.range_curriculum.min_episodes_per_class = 1
    wrist.range_curriculum.check_period_steps = 12
    cfg.terminations.test_failure = TerminationTermCfg(func=fail_once)
    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make("Template-G1-Upper-Mobile-v0", cfg=cfg).unwrapped
        try:
            command = env.command_manager.get_term("wrist_pose")
            original_sample = command._sample_task_kind

            def forced_classes(ids):
                original_sample(ids)
                command.planar_task_mask[ids] = ids >= 2
                command.height_task_mask[ids] = ids == 1
                command.height_shift[ids] = torch.where(ids == 1, -0.02, 0.0)

            command._sample_task_kind = forced_classes
            env.reset()
            device = env.device
            names = env.reward_manager.active_terms
            reward_sums = torch.zeros(4, len(names), device=device)
            observed = torch.zeros(4, device=device)
            ready_sums = torch.zeros_like(observed)
            start_step = torch.zeros(4, dtype=torch.long, device=device)
            reset_this_step = torch.zeros(4, dtype=torch.bool, device=device)
            original_reset = env._reset_idx
            maximum_window_error = maximum_reward_error = 0.0
            window_checks = reward_checks = early_checks = 0

            def record_window(ids):
                age = env.common_step_counter - start_step[ids]
                selected = command.planar_task_mask[ids] & (age >= command.evaluation_start_steps[ids])
                selected &= (age < command.episode_deadline_steps[ids]) & ~env.reset_terminated[ids]
                position, orientation = command.pose_error()
                ready = (((position[ids] <= 1.0) & (orientation[ids] <= 3.1)) | ~command.active_mask[ids]).all(-1)
                observed[ids] += selected
                ready_sums[ids] += selected & ready

            def checked_reset(ids):
                nonlocal maximum_window_error, maximum_reward_error, window_checks, reward_checks, early_checks
                record_window(ids)
                old_planar = command.planar_task_mask[ids].clone()
                failed = env.reset_terminated[ids].clone()
                age = env.common_step_counter - start_step[ids]
                expected_rates = (
                    (reward_sums[ids] + env.reward_manager._step_reward[ids] * env.step_dt)
                    / (age[:, None] * env.step_dt)
                ).mean(0)
                total = command.episode_deadline_steps[ids] - command.evaluation_start_steps[ids]
                expected_score = (ready_sums[ids][old_planar] / total[old_planar]).mean() if old_planar.any() else None
                expected_coverage = (observed[ids][old_planar] / total[old_planar]).mean() if old_planar.any() else None
                original_reset(ids)
                logs = env.extras["log"]
                for index, name in enumerate(names):
                    difference = (logs[f"Episode_Reward/{name}"] - expected_rates[index]).abs().item()
                    maximum_reward_error = max(maximum_reward_error, difference)
                    torch.testing.assert_close(
                        logs[f"Episode_Reward/{name}"], expected_rates[index], atol=1e-4, rtol=1e-5
                    )
                reward_checks += 1
                if expected_score is not None:
                    actual = logs["Metrics/wrist_pose/task/planar/eval_pose_ready_fraction"]
                    coverage = logs["Metrics/wrist_pose/task/planar/eval_coverage_fraction"]
                    maximum_window_error = max(maximum_window_error, abs(float(actual - expected_score)))
                    torch.testing.assert_close(actual, expected_score, atol=1e-6, rtol=0)
                    torch.testing.assert_close(coverage, expected_coverage, atol=1e-6, rtol=0)
                    window_checks += 1
                    if failed.any():
                        assert expected_score < 1
                        early_checks += int(failed.sum())
                reward_sums[ids] = 0
                observed[ids] = ready_sums[ids] = 0
                start_step[ids] = env.common_step_counter
                reset_this_step[ids] = True

            env._reset_idx = checked_reset
            for _ in range(48):
                reset_this_step.zero_()
                env.step(torch.zeros(4, 23, device=device))
                ongoing = (~reset_this_step).nonzero().flatten()
                reward_sums[ongoing] += env.reward_manager._step_reward[ongoing] * env.step_dt
                record_window(ongoing)
            assert command.range_curriculum.level >= 1
            report = {
                "window_checks": window_checks,
                "reward_rate_checks": reward_checks,
                "early_failures_checked": early_checks,
                "final_level": command.range_curriculum.level,
                "maximum_window_error": maximum_window_error,
                "maximum_reward_rate_error": maximum_reward_error,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
