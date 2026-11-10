"""Independent target-speed/substep oracles and action/reward side-effect checks."""

import argparse
import copy
import json
from pathlib import Path

import torch

from isaaclab.app import launch_simulation
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401


def relative_acceleration_cost(robot, wrist_ids, pelvis_id):
    """Independent expanded moving-frame formula from raw physics telemetry.

    Squared norm is invariant under the final pelvis rotation. Compare this
    world-axis representation with the action's pelvis-axis implementation.
    """
    data = robot.data
    origin = data.body_link_pose_w.torch[..., :3]
    com = data.body_com_pose_w.torch[..., :3]
    velocity = data.body_link_vel_w.torch
    acceleration = data.body_com_acc_w.torch
    omega = data.body_com_vel_w.torch[..., 3:]
    r_com = origin - com
    origin_acc = (
        acceleration[..., :3]
        + torch.cross(acceleration[..., 3:], r_com, dim=-1)
        + torch.cross(omega, torch.cross(omega, r_com, dim=-1), dim=-1)
    )
    r = origin[:, wrist_ids] - origin[:, pelvis_id : pelvis_id + 1]
    dv = velocity[:, wrist_ids, :3] - velocity[:, pelvis_id : pelvis_id + 1, :3]
    w = omega[:, pelvis_id : pelvis_id + 1]
    alpha = acceleration[:, pelvis_id : pelvis_id + 1, 3:]
    relative_acc = (
        origin_acc[:, wrist_ids]
        - origin_acc[:, pelvis_id : pelvis_id + 1]
        - torch.cross(alpha, r, dim=-1)
        - 2 * torch.cross(w, dv, dim=-1)
        + torch.cross(w, torch.cross(w, r, dim=-1), dim=-1)
    )
    return relative_acc.square().sum((1, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.episode_length_s = 0.12
    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            env.reset()
            action = env.action_manager.get_term("joint_pos")
            robot = env.scene["robot"]
            wrist_ids, _ = robot.find_bodies(list(cfg.commands.wrist_pose.body_names), preserve_order=True)
            pelvis_ids, _ = robot.find_bodies(cfg.commands.wrist_pose.reference_body_name)
            reference_cfg = JointPositionActionCfg(
                **{
                    key: copy.deepcopy(getattr(action.cfg, key))
                    for key in (
                        "asset_name",
                        "joint_names",
                        "preserve_order",
                        "scale",
                        "offset",
                        "use_default_offset",
                        "clip",
                    )
                }
            )
            reference = JointPositionAction(reference_cfg, env)
            samples = []
            end_effector_samples = []
            reward_checks = 0
            max_cost_error = 0.0
            max_first_substep_cost = 0.0
            max_end_effector_cost_error = 0.0
            max_first_end_effector_cost = 0.0
            mobile = hasattr(action, "frozen_policy")
            lower_calls = 0
            disabled_reward = False
            oracle_episode_sum = torch.zeros(env.num_envs, device=env.device)
            target_episode_sum = torch.zeros_like(oracle_episode_sum)
            end_effector_episode_sum = torch.zeros_like(oracle_episode_sum)
            target_velocity_oracle = torch.zeros_like(action.processed_actions)
            max_target_velocity_error = 0.0
            episode_log_checks = 0
            weights = None
            if mobile:
                weights = {key: value.clone() for key, value in action.frozen_policy.model.state_dict().items()}

                def count_lower(_model, _inputs, _outputs):
                    nonlocal lower_calls
                    lower_calls += 1

                action.frozen_policy.model.register_forward_hook(count_lower)

            original_process = action.process_actions
            original_apply = action.apply_actions
            original_update = env.scene.update
            original_reward = env.reward_manager.compute

            def checked_process(values):
                nonlocal max_target_velocity_error
                previous_target = robot.actuators.target_command.position.torch[:, action._joint_ids].clone()
                reference.process_actions(values)
                before = lower_calls
                original_process(values)
                torch.testing.assert_close(action.raw_actions, reference.raw_actions, rtol=0, atol=0)
                torch.testing.assert_close(action.processed_actions, reference.processed_actions, rtol=0, atol=0)
                target_velocity_oracle.copy_((reference.processed_actions - previous_target) / env.step_dt)
                torch.testing.assert_close(action.joint_target_velocity, target_velocity_oracle, rtol=1e-6, atol=1e-4)
                max_target_velocity_error = max(
                    max_target_velocity_error,
                    (action.joint_target_velocity - target_velocity_oracle).abs().max().item(),
                )
                if mobile:
                    assert lower_calls == before + 1

            def checked_apply():
                before = robot.data.joint_acc.torch.clone()
                velocity_before = action.joint_target_velocity.clone()
                calls = lower_calls
                original_apply()
                torch.testing.assert_close(robot.data.joint_acc.torch, before, rtol=0, atol=0)
                torch.testing.assert_close(action.joint_target_velocity, velocity_before, rtol=0, atol=0)
                target = robot.actuators.target_command.position.torch
                torch.testing.assert_close(target[:, action._joint_ids], reference.processed_actions, rtol=0, atol=0)
                if mobile:
                    assert lower_calls == calls  # Lower infers once per control step, not per physical substep.
                    torch.testing.assert_close(
                        target[:, action.lower_action._joint_ids], action.lower_action.processed_actions, rtol=0, atol=0
                    )

            def measured_update(dt):
                original_update(dt)
                samples.append(robot.data.joint_acc.torch[:, action._joint_ids].clone())
                end_effector_samples.append(relative_acceleration_cost(robot, wrist_ids, pelvis_ids[0]))

            def checked_reward(dt):
                nonlocal reward_checks, max_cost_error, max_first_substep_cost
                nonlocal max_end_effector_cost_error, max_first_end_effector_cost
                assert len(samples) == cfg.decimation
                assert len(end_effector_samples) == cfg.decimation
                oracle = torch.stack(samples).square().sum(dim=-1).mean(dim=0)
                end_effector_oracle = torch.stack(end_effector_samples).mean(dim=0)
                max_first_substep_cost = max(max_first_substep_cost, samples[0].square().sum(dim=-1).max().item())
                max_first_end_effector_cost = max(max_first_end_effector_cost, end_effector_samples[0].max().item())
                terms = {name: env.reward_manager.get_term_cfg(name) for name in env.reward_manager.active_terms}
                other_values = {
                    name: term.func(env, **term.params).clone()
                    for name, term in terms.items()
                    if name not in ("joint_acc", "target_speed", "end_effector_acc")
                }
                limit = terms["target_speed"].params["soft_limit"]
                target_cost = (target_velocity_oracle.abs() / limit - 1).clamp_min(0).square().sum(dim=-1)
                torch.testing.assert_close(
                    terms["target_speed"].func(env, **terms["target_speed"].params), target_cost, rtol=1e-5, atol=1e-4
                )
                state = [
                    robot.data.joint_pos.torch,
                    robot.data.joint_vel.torch,
                    robot.data.joint_acc.torch,
                    robot.actuators.target_command.position.torch,
                    env.action_manager.action,
                    env.action_manager.prev_action,
                    action.processed_actions,
                    action.joint_target_velocity,
                    robot.data.body_com_acc_w.torch,
                    robot.data.body_link_vel_w.torch,
                    robot.data.body_link_pose_w.torch,
                ]
                before = [value.clone() for value in state]
                if not disabled_reward:
                    measured = action.mean_joint_acc_l2()
                    torch.testing.assert_close(measured, oracle, rtol=1e-6, atol=1e-3)
                    torch.testing.assert_close(action.mean_joint_acc_l2(), oracle, rtol=1e-6, atol=1e-3)
                    max_cost_error = max(max_cost_error, (measured - oracle).abs().max().item())
                    measured_ee = action.mean_end_effector_acc_l2()
                    torch.testing.assert_close(measured_ee, end_effector_oracle, rtol=2e-5, atol=1e-3)
                    torch.testing.assert_close(
                        action.mean_end_effector_acc_l2(), end_effector_oracle, rtol=2e-5, atol=1e-3
                    )
                    max_end_effector_cost_error = max(
                        max_end_effector_cost_error, (measured_ee - end_effector_oracle).abs().max().item()
                    )
                    torch.testing.assert_close(
                        terms["end_effector_acc"].func(env, **terms["end_effector_acc"].params),
                        end_effector_oracle,
                        rtol=2e-5,
                        atol=1e-3,
                    )
                for current, saved in zip(state, before, strict=True):
                    torch.testing.assert_close(current, saved, rtol=0, atol=0)
                expected = torch.zeros(env.num_envs, device=env.device)
                for name, term in terms.items():
                    if name == "joint_acc":
                        value = oracle
                    elif name == "target_speed":
                        value = target_cost
                    elif name == "end_effector_acc":
                        value = end_effector_oracle
                    else:
                        value = term.func(env, **term.params)
                        torch.testing.assert_close(value, other_values[name], rtol=0, atol=0)
                    expected += value * term.weight
                result = original_reward(dt)
                torch.testing.assert_close(result, expected * dt, rtol=1e-5, atol=1e-5)
                oracle_episode_sum.add_(oracle, alpha=terms["joint_acc"].weight * dt)
                target_episode_sum.add_(target_cost, alpha=terms["target_speed"].weight * dt)
                end_effector_episode_sum.add_(end_effector_oracle, alpha=terms["end_effector_acc"].weight * dt)
                reward_checks += 1
                return result

            action.process_actions = checked_process
            action.apply_actions = checked_apply
            env.scene.update = measured_update
            env.reward_manager.compute = checked_reward
            resets = 0
            generator = torch.Generator(device=env.device).manual_seed(42)
            with torch.inference_mode():
                for step in range(48):
                    if step == 24:
                        target_cfg = env.reward_manager.get_term_cfg("target_speed")
                        target_cfg.weight = -1e-4
                        env.reward_manager.set_term_cfg("target_speed", target_cfg)
                    samples.clear()
                    end_effector_samples.clear()
                    values = (
                        torch.randn(4, env.action_manager.total_action_dim, device=env.device, generator=generator)
                        * 0.08
                    )
                    values[:, action._joint_names.index("left_wrist_pitch_joint")] = 0.9 if step % 4 < 2 else -0.9
                    observations, rewards, terminated, truncated, info = env.step(values)
                    assert observations["policy"].shape == (4, 107 if mobile else 93)
                    assert torch.isfinite(observations["policy"]).all() and torch.isfinite(rewards).all()
                    resets += int(truncated.sum())
                    ended = terminated | truncated
                    if ended.any():
                        expected_log = oracle_episode_sum[ended].mean() / env.max_episode_length_s
                        torch.testing.assert_close(
                            info["log"]["Episode_Reward/joint_acc"], expected_log, rtol=1e-5, atol=1e-6
                        )
                        oracle_episode_sum[ended] = 0
                        torch.testing.assert_close(
                            info["log"]["Episode_Reward/target_speed"],
                            target_episode_sum[ended].mean() / env.max_episode_length_s,
                            rtol=1e-5,
                            atol=1e-6,
                        )
                        target_episode_sum[ended] = 0
                        torch.testing.assert_close(
                            info["log"]["Episode_Reward/end_effector_acc"],
                            end_effector_episode_sum[ended].mean() / env.max_episode_length_s,
                            rtol=2e-5,
                            atol=1e-6,
                        )
                        end_effector_episode_sum[ended] = 0
                        episode_log_checks += 1
                # A zero-weight reward skips its getter. Partial reset still preserves other slots.
                env.reset()
                oracle_episode_sum.zero_()
                target_episode_sum.zero_()
                end_effector_episode_sum.zero_()
                joint_cfg = env.reward_manager.get_term_cfg("joint_acc")
                joint_cfg.weight = 0.0
                env.reward_manager.set_term_cfg("joint_acc", joint_cfg)
                end_effector_cfg = env.reward_manager.get_term_cfg("end_effector_acc")
                end_effector_cfg.weight = 0.0
                env.reward_manager.set_term_cfg("end_effector_acc", end_effector_cfg)
                disabled_reward = True
                samples.clear()
                end_effector_samples.clear()
                env.step(torch.full((4, env.action_manager.total_action_dim), 0.02, device=env.device))
                assert action._joint_acc_mean is None
                assert action._end_effector_acc_mean is None
                expected = torch.stack(samples).square().sum(dim=-1).mean(dim=0)
                expected_ee = torch.stack(end_effector_samples).mean(dim=0)
                preserved_lower = action.lower_actions[1:].clone() if mobile else None
                preserved_target_velocity = action.joint_target_velocity[1:].clone()
                reset_obs, _ = env.reset(env_ids=torch.tensor([0], device=env.device))
                if mobile:
                    assert not reset_obs["policy"][0, 76:93].any()
                torch.testing.assert_close(action.joint_target_velocity[1:], preserved_target_velocity, rtol=0, atol=0)
                assert not action.joint_target_velocity[0].any()
                torch.testing.assert_close(action.mean_joint_acc_l2()[1:], expected[1:], rtol=1e-6, atol=1e-3)
                assert action.mean_joint_acc_l2()[0] == 0
                torch.testing.assert_close(action.mean_end_effector_acc_l2()[1:], expected_ee[1:], rtol=2e-5, atol=1e-3)
                assert action.mean_end_effector_acc_l2()[0] == 0
                if mobile:
                    torch.testing.assert_close(action.lower_actions[1:], preserved_lower, rtol=0, atol=0)
                    for key, value in action.frozen_policy.model.state_dict().items():
                        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
            summary = {
                "task": args.task,
                "reward_checks": reward_checks,
                "physical_substeps_per_control": cfg.decimation,
                "maximum_oracle_absolute_error": max_cost_error,
                "maximum_target_velocity_oracle_error": max_target_velocity_error,
                "maximum_end_effector_oracle_absolute_error": max_end_effector_cost_error,
                "maximum_first_end_effector_squared_cost": max_first_end_effector_cost,
                "end_effector_reference_body": cfg.actions.joint_pos.acceleration_reference_body_name,
                "end_effector_substeps_and_episode_logs_checked": True,
                "target_speed_disabled_control_steps": 24,
                "target_speed_enabled_control_steps": 25,
                "maximum_first_substep_squared_cost": max_first_substep_cost,
                "timeout_resets": resets,
                "lower_inference_calls": lower_calls,
                "episode_log_checks": episode_log_checks,
                "other_rewards_and_targets_unchanged": True,
                "zero_weight_partial_reset_checked": True,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
