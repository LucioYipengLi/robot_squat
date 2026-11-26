"""Verify the applied target at each real physics substep before shutdown."""

import math

import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401
from g1.rl.evaluation.config import load_suite
from g1.rl.evaluation.modules import conditions
from g1.tasks.lower_limb_agent.evaluation import prepare_env_cfg


def main():
    suite = load_suite("g1_lower_delay")
    cfg, _ = resolve_task_config(suite["task"], "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    prepare_env_cfg(cfg, suite["benchmark"], conditions(suite)[-1], 2, 17)
    for name in vars(cfg.terminations):
        if name != "time_out":
            setattr(cfg.terminations, name, None)

    with launch_simulation(cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        env = gym.make(suite["task"], cfg=cfg).unwrapped
        try:
            env.reset()
            term = env.action_manager.get_term("joint_pos")
            data = env.scene["robot"].data
            joint_ids = term._joint_ids
            positions = data.joint_pos.torch[:, joint_ids].clone()
            writes, goals = [], []
            original_write = env.scene.write_data_to_sim

            def trace_write():
                writes.append(data.joint_pos_target.torch[:, joint_ids].clone())
                original_write()

            env.scene.write_data_to_sim = trace_write
            for value in (0.10, -0.10, 0.22, 0.12):
                env.step(torch.full_like(term.raw_actions, value))
                goals.append(term.processed_actions.clone())
            assert len(writes) == 8
            for tick, target in enumerate(writes):
                expected = positions if tick < 5 else goals[(tick - 5) // 2]
                torch.testing.assert_close(target, expected)

            _, extras = env.reset(env_ids=torch.tensor([0], device=env.device))
            delay_ms = extras["log"]["ActionDelay/episode_ms"].item()
            assert math.isclose(delay_ms, 1000 * 5 / 120, rel_tol=1e-6)
            reset_position = data.joint_pos.torch[0, joint_ids].clone()
            torch.testing.assert_close(data.joint_pos_target.torch[0, joint_ids], reset_position)
            writes.clear()
            env.step(torch.full_like(term.raw_actions, -0.22))
            assert len(writes) == 2
            for offset, target in enumerate(writes):
                torch.testing.assert_close(target[0], reset_position)
                torch.testing.assert_close(target[1], goals[(8 + offset - 5) // 2][1])
            print("LOWER_ACTION_DELAY_PHYSX_VERIFIED", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
