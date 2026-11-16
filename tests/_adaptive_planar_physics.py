"""Real PhysX command sampling, old-checkpoint migration, PPO logging and schedule resumption."""

import json
import sys
from pathlib import Path

import torch

from isaaclab.app import launch_simulation

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import resolve_task_config

from g1.rl.runner import create_task_runner
from g1.rl.teleop import _parse_args

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
args = _parse_args(["--task", "Template-G1-Upper-Mobile-v0", "--seed", "42"])
args.visualizer = "none"
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
env_cfg.scene.num_envs = 64
env_cfg.seed = agent_cfg.seed = 42
checkpoint = Path("logs/rsl_rl/g1_upper_mobile/2026-11-16_00-47-24/model_9999.pt")

with launch_simulation(env_cfg, args):
    import gymnasium as gym

    import g1.tasks  # noqa: F401

    raw = gym.make(args.task, cfg=env_cfg).unwrapped
    try:
        env = RslRlVecEnvWrapper(raw)
        runner = create_task_runner(env, agent_cfg, log_dir=str(out / "train"))
        runner.load(str(checkpoint))
        command = raw.command_manager.get_term("wrist_pose")
        course = command.range_curriculum
        assert course.level == 1 and course.planar_probability == 0.2
        assert not course.planar_sampling.history  # Older checkpoint preserves level, starts new schedule.
        # Supply controlled completed-episode statistics. Physics, action
        # inference and target generation remain real; this is not a performance evaluation.
        course.cfg.check_period_steps = 24
        course.cfg.min_episodes_per_class = 1
        course.restart(raw.common_step_counter)
        course.level = 1
        ids = torch.arange(64, device=raw.device)
        course.begin(ids)
        categories = torch.full((64,), 2, device=raw.device)
        failed = torch.zeros(64, dtype=torch.bool, device=raw.device)
        failed[:3] = True
        for index in range(1, 5):
            course.complete(
                ids,
                categories,
                torch.ones(64, device=raw.device),
                failed,
                raw.common_step_counter + index * 24,
                torch.full((64,), 15.0, device=raw.device),
            )
        course.clock_offset = course.last_check - raw.common_step_counter
        assert course.planar_probability == 0.25 and course.planar_sampling.cooldown == 1
        torch.cuda.manual_seed(781)
        expected = torch.rand(64, device=raw.device) < 0.25
        torch.cuda.manual_seed(781)
        command._sample_task_kind(ids)
        torch.testing.assert_close(command.planar_task_mask, expected)
        raw.reset()
        saved_sampling = course.planar_sampling.state_dict()
        runner.save(str(out / "adaptive.pt"))
        runner.load(str(out / "adaptive.pt"))
        assert course.level == 1 and course.planar_sampling.state_dict() == saved_sampling
        assert raw.extras["log"]["Curriculum/upper_sampling/planar_probability"] == 0.25
        # One genuine PPO iteration exercises the new scalar group in the standard logger.
        runner.learn(1, init_at_random_ep_len=False)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        runner.logger.writer.flush()
        events = EventAccumulator(str(out / "train")).Reload()
        values = events.Scalars("Curriculum/upper_sampling/planar_probability")
        assert values and values[-1].value >= 0.25
        assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "legacy_level_preserved": True,
                    "probability": course.planar_probability,
                    "gpu_draws_match": True,
                    "schedule_restored": True,
                    "ppo_iterations": 1,
                    "tensorboard_tag": values[-1].value,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        raw.close()
