"""Interactive checkpoint playback using the project's curriculum-aware runner."""

from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import ExitStack, suppress

import torch

from g1.teleop.controls import HELP_TEXT, CommandController, CommandLimits

logger = logging.getLogger(__name__)


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parse_args(argv):
    from isaaclab.app import add_launcher_args

    from isaaclab_rl.entrypoints import common
    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl

    from isaaclab_tasks.utils import setup_preset_cli

    import g1.tasks  # noqa: F401 - register the downstream tasks before preset parsing

    parser = argparse.ArgumentParser(description="Control a G1 policy with a Kit keyboard or Xbox gamepad.")
    common.add_common_play_args(
        parser, agent_default="rsl_rl_cfg_entry_point", agent_help="Registered RSL-RL agent configuration"
    )
    cli_args_rsl_rl.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    parser.add_argument("--input", choices=("keyboard", "gamepad", "both"), default="both")
    parser.add_argument("--show_pelvis", action="store_true", help="Show pelvis and projected foot geometry initially")
    parser.add_argument(
        "--max_steps", type=_positive_int, help="Stop after this many control steps (default: unbounded)"
    )
    parser.set_defaults(num_envs=1, visualizer="kit", real_time=True, checkpoint="latest")
    args, remaining = setup_preset_cli(parser, argv)
    if args.task is None:
        parser.error("--task is required")
    if args.num_envs != 1:
        parser.error("Teleop supports exactly one environment (--num_envs 1)")
    if args.frontend != "torch" or args.train_env_cfg:
        parser.error("Teleop requires the Torch frontend and play_mode() evaluation settings")
    if args.visualizer not in ("kit", ["kit"]):
        parser.error("Keyboard/gamepad teleop requires --viz kit")
    common.set_hydra_args(remaining)
    return args


class ManualPolicyPlayback:
    """Submit input once per control step and keep the normal observation-history lifecycle."""

    def __init__(self, env, policy, controller):
        self.env = env
        self.policy = policy
        self.controller = controller
        self.term = env.unwrapped.command_manager.get_term("task_command")
        # Curriculum load has already reset the environment. Reset once more with a zero
        # manual command so the first policy input/history contains no randomly sampled task.
        self.term.set_manual_command(torch.zeros_like(self.term.task_command))
        self.obs, _ = env.reset()
        self.last_command = (0.0,) * 6

    def step(self):
        with torch.inference_mode():
            # This observation describes the previous completed step. New input is submitted
            # for the upcoming command-manager update and appears in the next observation.
            # Do not recompute/push history or call _update_command() a second time here.
            actions = self.policy(self.obs)
            self.last_command = self.controller.sample(self.env.unwrapped.step_dt)
            command = torch.tensor(
                [self.last_command], device=self.term.task_command.device, dtype=self.term.task_command.dtype
            )
            self.term.set_manual_command(command)
            self.obs, _, dones, _ = self.env.step(actions)
            self.policy.reset(dones)


def run(argv, *, snapshot_path=None):
    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict
    from isaaclab.utils.seed import configure_seed

    from isaaclab_rl.entrypoints import common
    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl, play_rsl_rl
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import resolve_task_config

    from g1.tasks.lower_limb_agent.visualization import PelvisVisualizer
    from g1.teleop.kit import KitInput

    from .runner import CurriculumOnPolicyRunner

    args = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=True)
    common.apply_env_overrides(args, env_cfg)
    if not callable(getattr(env_cfg, "teleop_mode", None)):
        raise ValueError("This task does not define teleop_mode()")
    env_cfg.teleop_mode()
    common.pre_launch_video_config(env_cfg, args)
    with launch_simulation(env_cfg, args), ExitStack() as cleanup:
        agent_cfg = cli_args_rsl_rl.update_rsl_rl_cfg(agent_cfg, args)
        if agent_cfg.class_name != "OnPolicyRunner":
            raise ValueError("Teleop supports OnPolicyRunner only")
        env_cfg.seed = agent_cfg.seed
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        # Share the current upstream path/latest/best/W&B resolver with ordinary playback.
        checkpoint = play_rsl_rl._resolve_checkpoint(args, agent_cfg, env_cfg, log_root)
        if checkpoint is None:
            raise ValueError("No checkpoint resolved for teleop; provide --checkpoint")
        env_cfg.log_dir = os.path.dirname(checkpoint)
        common.apply_video_recording(env_cfg, env_cfg.log_dir, args, subdir="teleop", checkpoint_path=checkpoint)
        raw_env = common.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(common.close_env, raw_env)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = CurriculumOnPolicyRunner(
            env,
            to_dict(agent_cfg),
            log_dir=None,
            device=agent_cfg.device,
            snapshot_path=snapshot_path,
            evaluation=True,
        )
        if args.deterministic:
            configure_seed(env_cfg.seed, torch_deterministic=True)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        term = env.unwrapped.command_manager.get_term("task_command")
        controller = CommandController(CommandLimits.from_ranges(term.cfg.ranges), show_pelvis=args.show_pelvis)
        inputs = KitInput(controller, devices=args.input)
        cleanup.callback(inputs.close)
        playback = ManualPolicyPlayback(env, policy, controller)
        print(HELP_TEXT)
        logger.info("Loaded %s; teleop ranges: %s", checkpoint, controller.limits.bounds)
        budget = common.video_playback_steps(args, env_cfg)
        if args.max_steps is not None:
            budget = args.max_steps if budget is None else min(budget, args.max_steps)
        visualizer = None
        last_print = None
        next_print_time = 0.0
        step = 0
        with suppress(KeyboardInterrupt):
            while env.unwrapped.sim.is_running() and (budget is None or step < budget):
                start = time.perf_counter()
                inputs.poll()
                playback.step()
                if controller.show_pelvis:
                    if visualizer is None:
                        visualizer = PelvisVisualizer(env)
                        cleanup.callback(visualizer.clear)
                    visualizer.update()
                elif visualizer is not None:
                    visualizer.clear()
                if playback.last_command != last_print and start >= next_print_time:
                    print(
                        "[CMD] h={:+.3f} vx={:+.3f} vy={:+.3f} wz={:+.3f} pelvis=({:+.3f}, {:+.3f})".format(
                            *playback.last_command
                        )
                    )
                    last_print = playback.last_command
                    next_print_time = start + 0.25
                step += 1
                delay = env.unwrapped.step_dt - (time.perf_counter() - start)
                if args.real_time and delay > 0:
                    time.sleep(delay)
    return 0
