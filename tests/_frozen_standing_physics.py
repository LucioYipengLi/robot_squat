"""Bounded, real free-base standing and Upper-motion validation; invoked by pytest."""

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
    task = "Template-G1-Upper-Mobile-v0"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    cfg.scene.num_envs = 4
    cfg.actions.lower_command.enabled = False
    cfg.actions.height_offset.enabled = False
    cfg.seed = 42
    cfg.episode_length_s = 30.0
    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make(task, cfg=cfg).unwrapped
        try:
            observations, _ = env.reset()
            action = env.action_manager.get_term("joint_pos")
            robot = env.scene["robot"]
            assert not robot.is_fixed_base
            assert env.action_manager.total_action_dim == 21
            assert observations["policy"].shape == (4, 107)
            assert action.last_lower_observations.shape == (4, 650)
            assert action.lower_actions.shape == (4, 12)
            assert not hasattr(env, "upper_body_motion")
            assert set(action._joint_ids.tolist()).isdisjoint(action.lower_action._joint_ids.tolist())
            # Compare against the original Lower observation functions, on the same live state.
            from g1.tasks.lower_limb_agent.config.g1.env_cfg import ObservationsCfg

            reference_cfg = ObservationsCfg.PolicyCfg()
            reference_cfg.enable_corruption = False
            reference_env = SimpleNamespace(
                num_envs=env.num_envs,
                device=env.device,
                scene=env.scene,
                sim=env.sim,
                action_manager=SimpleNamespace(action=action.lower_actions),
                command_manager=SimpleNamespace(get_command=lambda _name: action.lower_command),
                upper_body_motion=SimpleNamespace(
                    get_observation_goal=lambda: action.processed_actions[:, action._preview_order]
                ),
            )
            reference = ObservationManager({"policy": reference_cfg}, reference_env)

            def compare_observations(_model, inputs):
                expected = reference.compute_group("policy", update_history=True)
                torch.testing.assert_close(inputs[0], expected, rtol=0, atol=0)

            action.frozen_policy.model.register_forward_pre_hook(compare_observations)
            original_weights = {name: value.clone() for name, value in action.frozen_policy.model.state_dict().items()}
            actions = torch.zeros(4, 21, device=env.device)
            upper_actions = actions[:, 4:]
            summary = {}
            with torch.inference_mode():
                for phase in ("standing", "upper_motion"):
                    env.reset()
                    reference.reset()
                    heights, tilts, speeds, lower_motion = [], [], [], []
                    failures = 0
                    previous = robot.data.joint_pos.torch[:, action.lower_action._joint_ids].clone()
                    for step in range(600):  # Ten physical seconds per phase.
                        actions.zero_()
                        actions[:, :3] = 1.0  # Force-zero mode ignores even nonzero policy commands.
                        if phase == "upper_motion":
                            # Different sub-environments exercise unilateral and bilateral motions.
                            t = step * env.step_dt
                            ramp = min(1.0, t / 2.0)
                            angles = 0.35 * ramp * math.sin(2 * math.pi * 0.35 * t)
                            for name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
                                index = action._joint_names.index(name)
                                upper_actions[:, index] = angles / cfg.actions.joint_pos.scale
                            upper_actions[0, action._joint_names.index("right_shoulder_pitch_joint")] = 0
                            upper_actions[1, action._joint_names.index("left_shoulder_pitch_joint")] = 0
                            upper_actions[:, action._joint_names.index("left_elbow_joint")] = 0.2 * ramp / 0.5
                            upper_actions[:, action._joint_names.index("right_elbow_joint")] = 0.2 * ramp / 0.5
                            for name, amplitude in (
                                ("waist_yaw_joint", 0.12),
                                ("waist_roll_joint", 0.04),
                                ("waist_pitch_joint", 0.06),
                                ("left_wrist_roll_joint", 0.15),
                                ("right_wrist_roll_joint", 0.15),
                            ):
                                upper_actions[:, action._joint_names.index(name)] = (
                                    amplitude * ramp * math.sin(2 * math.pi * 0.25 * t) / 0.5
                                )
                        obs, rewards, terminated, truncated, _ = env.step(actions)
                        failures += int(terminated.sum().item())
                        assert not truncated.any()
                        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
                        assert not action.lower_command.any()
                        assert not env.action_manager.get_term("lower_command").change_l2.any()
                        assert torch.isfinite(action.lower_actions).all()
                        heights.append((robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).clone())
                        tilts.append(torch.acos((-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1, 1)))
                        speeds.append(robot.data.root_lin_vel_b.torch[:, :2].norm(dim=-1).clone())
                        current = robot.data.joint_pos.torch[:, action.lower_action._joint_ids]
                        lower_motion.append((current - previous).abs().amax().item())
                        previous = current.clone()
                    height, tilt, speed = torch.stack(heights), torch.stack(tilts), torch.stack(speeds)
                    summary[phase] = {
                        "duration_s": 10,
                        "envs": 4,
                        "failures": failures,
                        "minimum_height_m": height.min().item(),
                        "mean_height_m": height.mean().item(),
                        "maximum_tilt_rad": tilt.max().item(),
                        "mean_planar_speed_m_s": speed.mean().item(),
                        "maximum_lower_joint_step_rad": max(lower_motion),
                    }
                # One-row reset must preserve ongoing histories and lower action buffers.
                untouched = action.last_lower_observations[1:].clone()
                history = action.lower_observations.compute_group("policy").clone()
                other_actions = action.lower_actions[1:].clone()
                env.reset(env_ids=torch.tensor([0], device=env.device))
                torch.testing.assert_close(action.last_lower_observations[1:], untouched)
                torch.testing.assert_close(action.lower_actions[1:], other_actions)
                torch.testing.assert_close(action.lower_observations.compute_group("policy")[1:], history[1:])
                assert not action.lower_actions[0].any()
                for name, value in action.frozen_policy.model.state_dict().items():
                    torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)
            assert all(row["failures"] == 0 for row in summary.values()), summary
        finally:
            env.close()


if __name__ == "__main__":
    main()
