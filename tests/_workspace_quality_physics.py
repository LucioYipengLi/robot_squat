"""Bounded PhysX checks for Q gates, reward accounting, diagnostics and frozen Lower timing."""

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.episode_length_s = 0.12
    cfg.commands.wrist_pose.resampling_time_range = (100.0, 100.0)
    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            env.reset()
            term = env.command_manager.get_term("wrist_pose")
            robot = env.scene["robot"]
            action = env.action_manager.get_term("joint_pos")
            mobile = hasattr(action, "frozen_policy")
            lower_calls = 0
            checks = 0
            positive_steps = 0
            disabled_steps = 0
            log_checks = 0
            expected_episode = torch.zeros(4, device=env.device)
            weight = 0.0
            original_reward = env.reward_manager.compute

            def count_lower(*_):
                nonlocal lower_calls
                lower_calls += 1

            if mobile:
                before_weights = {key: value.clone() for key, value in action.frozen_policy.model.state_dict().items()}
                action.frozen_policy.model.register_forward_hook(count_lower)

            def checked_reward(dt):
                nonlocal checks, positive_steps, disabled_steps
                # Deliberately publish known test goals after physics. The fourth
                # environment has one active wrist outside position tolerance.
                term.mode_id[:] = torch.tensor([0, 1, 2, 2], device=env.device)
                term.goal_pose_w.copy_(robot.data.body_link_pose_w.torch[:, term.body_indices])
                term.goal_pose_w[0, 1, :3] += 100  # Idle target is excluded.
                term.goal_pose_w[3, 1, 0] += 0.02
                term.goal_pose_w[..., 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
                raw_state = [
                    robot.data.joint_pos.torch,
                    robot.data.joint_vel.torch,
                    robot.data.body_link_pose_w.torch,
                    robot.actuators.target_command.position.torch,
                    env.action_manager.action,
                    env.action_manager.prev_action,
                ]
                saved = [value.clone() for value in raw_state]
                terms = {name: env.reward_manager.get_term_cfg(name) for name in env.reward_manager.active_terms}
                qstate = term.workspace_quality_state()
                assert qstate.position_ready.tolist() == [True, True, True, False]
                assert qstate.map_covered.all()
                value = terms["workspace_quality"].func(env, **terms["workspace_quality"].params)
                assert (value[:3] > 0).all() and value[3] == 0
                torch.testing.assert_close(value, qstate.reward, rtol=0, atol=0)
                torch.testing.assert_close(terms["workspace_quality"].func(env), value, rtol=0, atol=0)
                expected = (
                    sum(term_cfg.func(env, **term_cfg.params) * term_cfg.weight for term_cfg in terms.values()) * dt
                )
                result = original_reward(dt)
                torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
                for current, previous in zip(raw_state, saved, strict=True):
                    torch.testing.assert_close(current, previous, rtol=0, atol=0)
                expected_episode.add_(value, alpha=weight * dt)
                checks += 1
                positive_steps += int(weight > 0)
                disabled_steps += int(weight == 0)
                return result

            env.reward_manager.compute = checked_reward
            with torch.inference_mode():
                for step in range(12):
                    if step == 6:
                        weight = 0.02
                        reward_cfg = env.reward_manager.get_term_cfg("workspace_quality")
                        reward_cfg.weight = weight
                        env.reward_manager.set_term_cfg("workspace_quality", reward_cfg)
                    obs, _, terminated, truncated, info = env.step(
                        torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
                    )
                    assert obs["policy"].shape == (4, 107 if mobile else 93)
                    assert torch.isfinite(obs["policy"]).all()
                    if step == 5:
                        # Metrics remain active throughout the zero-weight stage.
                        assert (term._workspace_sums[:, 1] > 0).all()
                        assert (term._workspace_sums[:, 4] > 0).any()
                    ended = terminated | truncated
                    if ended.any():
                        torch.testing.assert_close(
                            info["log"]["Episode_Reward/workspace_quality"],
                            expected_episode[ended].mean() / env.max_episode_length_s,
                            rtol=1e-5,
                            atol=1e-6,
                        )
                        expected_episode[ended] = 0
                        for metric in (
                            "target_score",
                            "actual_score",
                            "position_ready_fraction",
                            "map_coverage_fraction",
                            "reward_available_fraction",
                        ):
                            key = f"Metrics/wrist_pose/workspace_quality/{metric}"
                            assert key in info["log"] and torch.isfinite(info["log"][key])
                        log_checks += 1
                previous = term._workspace_sums[1:].clone()
                env.reset(env_ids=torch.tensor([0], device=env.device))
                torch.testing.assert_close(term._workspace_sums[1:], previous, rtol=0, atol=0)
                assert not term._workspace_sums[0].any()
                if mobile:
                    assert lower_calls == 12
                    assert not action.lower_command.any()
                    for key, value in action.frozen_policy.model.state_dict().items():
                        torch.testing.assert_close(value, before_weights[key], rtol=0, atol=0)
            result = {
                "task": args.task,
                "reward_checks": checks,
                "zero_weight_steps": disabled_steps,
                "positive_weight_steps": positive_steps,
                "episode_log_checks": log_checks,
                "lower_calls": lower_calls,
                "partial_reset_checked": True,
                "physical_state_targets_and_actions_preserved": True,
                "map_sha256": term.workspace_quality_map.sha256,
            }
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
