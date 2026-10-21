"""Opt-in headless verification of real reference writes, stepping and reset.

Run with the project Python; writes outputs/verification/goal_observation_physics.json.
"""

import json
from pathlib import Path

import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401
from g1.rl.evaluation.config import load_suite
from g1.tasks.lower_limb_agent.evaluation import prepare_env_cfg
from g1.tasks.lower_limb_agent.mdp.observations import upper_body_joint_pos_target_rel


def main():
    suite = load_suite("g1_core")
    cfg, _ = resolve_task_config(suite["task"], "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    prepare_env_cfg(cfg, suite["benchmark"], {"arm_fraction": 0, "waist_fraction": 0}, 4, 17)
    # Isolate reference execution from policy falls and automatic terminations.
    for name, term in vars(cfg.terminations).items():
        if term is not None and name != "time_out":
            setattr(cfg.terminations, name, None)
    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make(suite["task"], cfg=cfg).unwrapped
        try:
            env.reset()
            motion = env.upper_body_motion
            curve = motion.trajectory
            ids = motion.joint_ids
            goals = curve.position[:2].clone()
            goals[:, 0] += 0.3
            observation_params = env.observation_manager.cfg.policy.upper_body_target.params
            default = env.scene["robot"].data.default_joint_pos.torch[:, ids]
            motion.set_goal(torch.tensor([0, 1], device=env.device), goals, ids.tolist())
            torch.testing.assert_close(
                upper_body_joint_pos_target_rel(env, **observation_params)[:2], goals - default[:2]
            )
            assert not curve.active.any()
            previous, previous_v = curve.position.clone(), curve.velocity.clone()
            peak_speed, peak_acceleration = 0.0, 0.0
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            with torch.inference_mode():
                for step in range(90):
                    env.step(actions)
                    peak_speed = max(peak_speed, curve.velocity.abs().max().item())
                    peak_acceleration = max(
                        peak_acceleration, ((curve.velocity - previous_v) / env.step_dt).abs().max().item()
                    )
                    max_step = cfg.upper_body_motion.max_velocity * env.step_dt
                    assert (curve.position - previous).abs().max() <= max_step + 1e-5
                    torch.testing.assert_close(env.scene["robot"].data.joint_pos_target.torch[:, ids], curve.position)
                    previous, previous_v = curve.position.clone(), curve.velocity.clone()
                    if step == 1:
                        pending = goals.clone()
                        pending[:, 0] -= 0.4
                        motion.set_goal(torch.tensor([0, 1], device=env.device), pending, ids.tolist())
                        torch.testing.assert_close(
                            upper_body_joint_pos_target_rel(env, **observation_params)[:2], goals - default[:2]
                        )
                assert peak_speed <= cfg.upper_body_motion.max_velocity + 1e-4
                assert peak_acceleration <= cfg.upper_body_motion.max_acceleration + 1e-3
                assert curve.statistics[0, 1] == 2
                torch.testing.assert_close(
                    upper_body_joint_pos_target_rel(env, **observation_params)[:2], pending - default[:2]
                )
                untouched = curve.position[1:].clone()
                env.reset(env_ids=torch.tensor([0], device=env.device))
                torch.testing.assert_close(curve.position[1:], untouched)
                torch.testing.assert_close(curve.position[0], env.scene["robot"].data.joint_pos.torch[0, ids])
                assert not curve.active[0] and not curve.pending[0]
                assert not env.scene["robot"].data.joint_vel.torch[0, ids].any()
                torch.testing.assert_close(
                    upper_body_joint_pos_target_rel(env, **observation_params)[0],
                    env.scene["robot"].data.joint_pos.torch[0, ids] - default[0],
                )
            result = {
                "status": "passed",
                "num_envs": 4,
                "steps": 90,
                "peak_reference_velocity_radps": peak_speed,
                "peak_reference_acceleration_radps2": peak_acceleration,
                "motion": "quintic",
                "upper_body_target_source": observation_params["target_source"],
                "device": torch.cuda.get_device_name(),
            }
            path = Path("outputs/verification/goal_observation_physics.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
