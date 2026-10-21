"""Create a synthetic actor with dimensions inferred from a real headless scene."""

import sys
from pathlib import Path

import torch
from rsl_rl.utils import resolve_class, resolve_obs_groups

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import resolve_task_config

import g1.tasks  # noqa: F401
from g1.rl.evaluation.config import load_suite
from g1.tasks.lower_limb_agent.evaluation import prepare_env_cfg


def main():
    checkpoint = Path(sys.argv[1])
    checkpoint.parent.mkdir(parents=True)
    suite = load_suite("g1_core_v1")
    env_cfg, agent = resolve_task_config(suite["task"], "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    dump_yaml(str(checkpoint.parent / "params/env.yaml"), env_cfg)
    agent.actor.hidden_dims = [16]
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    dump_yaml(str(checkpoint.parent / "params/agent.yaml"), agent)
    condition = {"arm_fraction": 0.0, "waist_fraction": 0.0}
    prepare_env_cfg(env_cfg, suite["benchmark"], condition, 4, 17)
    with launch_simulation(env_cfg, {"headless": True, "device": "cuda:0", "visualizer": None}):
        import gymnasium as gym

        raw = gym.make(suite["task"], cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(raw)
            observations = env.get_observations()
            config = to_dict(agent)
            actor_class, actor_cfg = resolve_class(config["actor"])
            groups = resolve_obs_groups(observations, config["obs_groups"], ["actor"])
            actor = actor_class(observations, groups, "actor", env.num_actions, **actor_cfg)
            torch.save({"actor_state_dict": actor.state_dict(), "iter": 0, "infos": {}}, checkpoint)
        finally:
            raw.close()


if __name__ == "__main__":
    main()
