"""Real frozen Lower: current-command observations, transitions, costs and partial resets."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from isaaclab.app import launch_simulation
from isaaclab.managers import ObservationManager

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, _ = resolve_task_config(
        "Template-G1-Upper-Mobile-v0", "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"]
    )
    cfg.scene.num_envs, cfg.seed, cfg.episode_length_s = 4, 42, 20.0
    cfg.commands.wrist_pose.resampling_time_range = (100.0, 100.0)
    optional_names = (
        "lower_command_overflow",
        "lower_command_magnitude",
        "base_lin_vel",
        "base_ang_vel",
        "base_lin_acc",
        "base_ang_acc",
    )
    assert all(getattr(cfg.rewards, name).weight == 0.0 for name in optional_names[2:])
    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make("Template-G1-Upper-Mobile-v0", cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            manager = env.action_manager
            command, bridge = manager.get_term("lower_command"), manager.get_term("joint_pos")
            height = manager.get_term("height_offset")
            assert manager.active_terms == ["lower_command", "height_offset", "joint_pos"]
            assert manager.action_term_dim == [3, 1, 17] and obs["policy"].shape == (4, 107)
            robot = env.scene["robot"]
            scale = torch.tensor(command.cfg.command_scale, device=env.device)
            base_ids, _ = robot.find_bodies("pelvis")
            assert len(base_ids) == 1
            base_id = base_ids[0]
            base_samples = []
            vertical_samples = []
            recording_base = False
            maximum_base_oracle_error = 0.0
            maximum_vertical_oracle_error = 0.0
            vertical_reward_checks = 0
            maximum_base_com_acc_derivative_error = 0.0
            previous_base_com_velocity = None
            optional_reward_checks = 0
            original_process, original_update = bridge.process_actions, env.scene.update

            def checked_process(actions):
                nonlocal recording_base, previous_base_com_velocity
                base_samples.clear()
                vertical_samples.clear()
                recording_base = True
                previous_base_com_velocity = robot.data.body_com_vel_w.torch[:, base_id].clone()
                original_process(actions)

            def measured_update(dt):
                nonlocal previous_base_com_velocity, maximum_base_com_acc_derivative_error
                original_update(dt)
                if recording_base:
                    data = robot.data
                    velocity = data.body_link_vel_w.torch[:, base_id]
                    acceleration = data.body_com_acc_w.torch[:, base_id]
                    com_velocity = data.body_com_vel_w.torch[:, base_id]
                    derivative = (com_velocity - previous_base_com_velocity) / dt
                    maximum_base_com_acc_derivative_error = max(
                        maximum_base_com_acc_derivative_error, (acceleration - derivative).abs().max().item()
                    )
                    previous_base_com_velocity = com_velocity.clone()
                    omega = data.body_com_vel_w.torch[:, base_id, 3:]
                    offset = data.body_link_pose_w.torch[:, base_id, :3] - data.body_com_pose_w.torch[:, base_id, :3]
                    # Independent COM-to-link expansion at every completed physical step.
                    origin_acc = (
                        acceleration[:, :3]
                        + torch.cross(acceleration[:, 3:], offset, dim=-1)
                        + torch.cross(omega, torch.cross(omega, offset, dim=-1), dim=-1)
                    )
                    vertical_samples.append(origin_acc[:, 2].clone())
                    base_samples.append(
                        torch.stack(
                            (
                                velocity[:, :3].square().sum(-1),
                                velocity[:, 3:].square().sum(-1),
                                origin_acc.square().sum(-1),
                                acceleration[:, 3:].square().sum(-1),
                            ),
                            dim=-1,
                        )
                    )

            bridge.process_actions, env.scene.update = checked_process, measured_update

            def expected_command():
                value = torch.zeros(4, 6, device=env.device)
                value[:, 1:4] = manager.action[:, :3].clamp(-1, 1) * scale
                u = manager.action[:, 3].clamp(-1, 1)
                low, high = height.cfg.height_offset_range
                value[:, 0] = u.clamp_max(0) * -low + u.clamp_min(0) * high
                return value

            # Use original Lower observation functions and an independent history
            # manager. Commands come from the current raw policy actions, not the
            # bridge buffer, so a one-control-step delay cannot pass this oracle.
            from g1.tasks.lower_limb_agent.config.g1.env_cfg import ObservationsCfg

            group = ObservationsCfg.PolicyCfg()
            group.enable_corruption = False
            reference_env = SimpleNamespace(
                num_envs=4,
                device=env.device,
                scene=env.scene,
                sim=env.sim,
                action_manager=SimpleNamespace(action=bridge.lower_actions),
                command_manager=SimpleNamespace(get_command=lambda _: expected_command()),
                upper_body_motion=SimpleNamespace(
                    get_observation_goal=lambda: bridge.processed_actions[:, bridge._preview_order]
                ),
            )
            reference = ObservationManager({"policy": group}, reference_env)
            inference_calls, reward_checks = 0, 0

            def compare_inputs(_model, inputs):
                nonlocal inference_calls
                torch.testing.assert_close(bridge.lower_command, expected_command(), rtol=0, atol=0)
                expected = reference.compute_group("policy", update_history=True)
                torch.testing.assert_close(inputs[0], expected, rtol=0, atol=0)
                inference_calls += 1

            bridge.frozen_policy.model.register_forward_pre_hook(compare_inputs)
            weights = {key: value.clone() for key, value in bridge.frozen_policy.model.state_dict().items()}
            original_reward = env.reward_manager.compute

            def checked_reward(dt):
                nonlocal reward_checks, optional_reward_checks, recording_base, maximum_base_oracle_error
                nonlocal vertical_reward_checks, maximum_vertical_oracle_error
                current, previous = manager.action, manager.prev_action
                expected_cost = (current[:, :4].clamp(-1, 1) - previous[:, :4].clamp(-1, 1)).square().sum(-1)
                cost_cfg = env.reward_manager.get_term_cfg("lower_command_rate")
                torch.testing.assert_close(cost_cfg.func(env, **cost_cfg.params), expected_cost)
                upper_cfg = env.reward_manager.get_term_cfg("action_rate")
                upper_cost = (current[:, 4:] - previous[:, 4:]).square().sum(-1)
                torch.testing.assert_close(upper_cfg.func(env, **upper_cfg.params), upper_cost)
                assert len(base_samples) == cfg.decimation
                base_oracle = torch.stack(base_samples).mean(0)
                vertical_cfg = env.reward_manager.get_term_cfg("base_vertical_acc")
                assert len(vertical_samples) == cfg.decimation
                vertical_oracle = (
                    (torch.stack(vertical_samples).abs() / vertical_cfg.params["soft_limit"] - 1)
                    .clamp_min(0)
                    .square()
                    .mean(0)
                )
                vertical_measured = vertical_cfg.func(env, **vertical_cfg.params)
                maximum_vertical_oracle_error = max(
                    maximum_vertical_oracle_error, (vertical_measured - vertical_oracle).abs().max().item()
                )
                torch.testing.assert_close(vertical_measured, vertical_oracle, rtol=1e-5, atol=1e-5)
                vertical_reward_checks += 1
                for index, name in enumerate(optional_names):
                    term = env.reward_manager.get_term_cfg(name)
                    measured = term.func(env, **term.params)
                    if index == 0:
                        oracle = (current[:, :3].abs() - 1).clamp_min(0).square().sum(-1)
                    elif index == 1:
                        oracle = current[:, :3].clamp(-1, 1).square().sum(-1)
                    else:
                        oracle = base_oracle[:, index - 2]
                        maximum_base_oracle_error = max(
                            maximum_base_oracle_error, (measured - oracle).abs().max().item()
                        )
                    torch.testing.assert_close(measured, oracle, rtol=1e-5, atol=1e-4)
                    optional_reward_checks += 1
                expected = torch.zeros(4, device=env.device)
                for name in env.reward_manager.active_terms:
                    term = env.reward_manager.get_term_cfg(name)
                    if term.weight:
                        expected += term.func(env, **term.params) * term.weight * dt
                result = original_reward(dt)
                torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
                recording_base = False  # Exclude updates caused by auto-reset writes.
                reward_checks += 1
                return result

            env.reward_manager.compute = checked_reward
            target = torch.tensor([[0.2, 0, 0], [0, 0.08, 0], [0, 0, 0.25], [0.15, 0.05, 0.15]], device=env.device)
            phases = [
                ("settle", 0, 60, 0.0),
                ("positive", 1, 180, 0.0),
                ("stop", 0, 60, 0.0),
                ("negative", -1, 180, 0.0),
                ("final_stop", 0, 60, 0.0),
                ("height_down", 0, 180, -0.15),
                ("height_up", 0, 180, 0.02),
                ("height_return", 0, 120, 0.0),
            ]
            goals = env.command_manager.get_term("wrist_pose").goal_pose_w.clone()
            rows, total_steps = [], 0
            with torch.inference_mode():
                for name, sign, steps, height_goal in phases:
                    actual_samples, error_samples, tilts, heights = [], [], [], []
                    failures = 0
                    goal = target * sign
                    for step in range(steps):
                        values = torch.zeros(4, 21, device=env.device)
                        values[:, :3] = goal / scale
                        low, high = height.cfg.height_offset_range
                        values[:, 3] = height_goal / (-low if height_goal < 0 else high)
                        # One environment also exercises current Upper goal preview.
                        index = bridge._joint_names.index("left_elbow_joint")
                        values[3, 4 + index] = 0.08 * math.sin(total_steps * env.step_dt)
                        obs, reward, ended, timeout, _ = env.step(values)
                        total_steps += 1
                        assert inference_calls == reward_checks == total_steps
                        assert not timeout.any()
                        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                        torch.testing.assert_close(obs["policy"][:, -4:-1], goal, rtol=0, atol=1e-7)
                        torch.testing.assert_close(
                            obs["policy"][:, -1], torch.full((4,), height_goal, device=env.device)
                        )
                        failures += int(ended.sum().item())
                        if not ended.any():
                            torch.testing.assert_close(env.command_manager.get_term("wrist_pose").goal_pose_w, goals)
                        else:
                            # Keep the independent history oracle synchronized with auto resets.
                            reference.reset(torch.where(ended)[0])
                            goals.copy_(env.command_manager.get_term("wrist_pose").goal_pose_w)
                        actual = torch.cat(
                            (robot.data.root_lin_vel_b.torch[:, :2], robot.data.root_ang_vel_b.torch[:, 2:3]), -1
                        )
                        if step >= min(60, steps // 2):
                            actual_samples.append(actual.clone())
                            error_samples.append((actual - goal).clone())
                            heights.append((robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).clone())
                        tilts.append(torch.acos((-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1, 1)))
                    actual_mean = torch.stack(actual_samples).mean(0)
                    error = torch.stack(error_samples)
                    rows.append(
                        {
                            "phase": name,
                            "duration_s": steps * env.step_dt,
                            "failures": failures,
                            "commands": goal.tolist(),
                            "height_offset_m": height_goal,
                            "steady_height_mean_m": torch.stack(heights).mean(0).tolist(),
                            "steady_height_error_m": (
                                torch.stack(heights) - robot.data.default_root_pose.torch[:, 2] - height_goal
                            )
                            .abs()
                            .mean(0)
                            .tolist(),
                            "steady_actual_mean": actual_mean.tolist(),
                            "steady_planar_error_m_s": error[..., :2].norm(dim=-1).mean(0).tolist(),
                            "steady_yaw_error_rad_s": error[..., 2].abs().mean(0).tolist(),
                            "maximum_tilt_rad": torch.stack(tilts).max().item(),
                        }
                    )
                command_before = [
                    value[1:].clone()
                    for value in (
                        command.raw_actions,
                        command.processed_actions,
                        command.change_l2,
                        height.raw_actions,
                        height.processed_actions,
                        height.change_l2,
                    )
                ]
                lower_before = bridge.lower_actions[1:].clone()
                vertical_before = bridge._base_vertical_acc_samples[1:].clone()
                history_before = bridge.lower_observations.compute_group("policy")[1:].clone()
                env.reset(env_ids=torch.tensor([0], device=env.device))
                reference.reset(torch.tensor([0], device=env.device))
                for value, before in zip(
                    (
                        command.raw_actions,
                        command.processed_actions,
                        command.change_l2,
                        height.raw_actions,
                        height.processed_actions,
                        height.change_l2,
                    ),
                    command_before,
                    strict=True,
                ):
                    torch.testing.assert_close(value[1:], before, rtol=0, atol=0)
                    assert not value[0].any()
                torch.testing.assert_close(bridge.lower_actions[1:], lower_before, rtol=0, atol=0)
                torch.testing.assert_close(bridge._base_vertical_acc_samples[1:], vertical_before, rtol=0, atol=0)
                assert not bridge._base_vertical_acc_samples[0].any()
                assert not bridge.mean_base_vertical_acc_excess_l2()[0].any()
                torch.testing.assert_close(
                    bridge.lower_observations.compute_group("policy")[1:], history_before, rtol=0, atol=0
                )
                assert not bridge.lower_command[0].any()
                assert not bridge.mean_base_motion_l2()[0].any()
                # First command after a partial reset also advances each history exactly once.
                env.step(torch.zeros(4, 21, device=env.device))
                total_steps += 1
                assert inference_calls == reward_checks == total_steps
                for key, value in bridge.frozen_policy.model.state_dict().items():
                    torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
            report = {
                "phases": rows,
                "control_steps": total_steps,
                "inference_calls": inference_calls,
                "reward_checks": reward_checks,
                "vertical_reward_checks": vertical_reward_checks,
                "maximum_vertical_oracle_error": maximum_vertical_oracle_error,
                "base_vertical_acc_weight": env.reward_manager.get_term_cfg("base_vertical_acc").weight,
                "base_vertical_acc_soft_limit": env.reward_manager.get_term_cfg("base_vertical_acc").params[
                    "soft_limit"
                ],
                "rate_command_dimensions": 4,
                "partial_reset_preserved": True,
                "native_lower_observations_match": True,
                "frozen_weights_unchanged": True,
                "control_dt": env.step_dt,
                "upper_actions": 21,
                "upper_observations": 107,
                "base_rewards_default_zero": True,
                "optional_reward_checks": optional_reward_checks,
                "maximum_base_substep_oracle_error": maximum_base_oracle_error,
                "maximum_base_com_acc_derivative_error": maximum_base_com_acc_derivative_error,
                "base_motion_reference": "pelvis_link_origin_world",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
