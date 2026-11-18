"""Restore an Upper checkpoint, freeze its curriculum and collect bounded failure clips."""

import hashlib
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .storage import FailureArchive, write_json


def run_mining(args):
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.utils import configure_logging
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.seed import configure_seed

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.utils import resolve_task_config

    import g1.tasks  # noqa: F401

    from ..runner import create_task_runner
    from .geometry import contact_filters, export_geometry
    from .recording import FailureMiner, force_category, prepare_recording

    checkpoint = args.checkpoint.resolve(strict=True)
    directory = args.output or Path("outputs/failures") / (
        datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    if directory.exists():
        raise FileExistsError(f"Choose a new output directory: {directory}")
    configure_logging(save_logs_to_file=False)
    configure_seed(args.seed)
    version = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(
        args.task, "rsl_rl_cfg_entry_point", play_mode=True, overrides=["physics=isaacsim_physx"]
    )
    num_levels = env_cfg.commands.wrist_pose.range_curriculum.num_levels
    if args.level is not None and not 0 <= args.level < num_levels:
        raise ValueError(f"--level must be between 0 and {num_levels - 1}, got {args.level}")
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version)
    prepare_recording(env_cfg, args.num_envs)
    launch_args = {"headless": True, "visualizer": None, "device": args.device}
    with launch_simulation(env_cfg, launch_args):
        import isaaclab.sim as sim_utils

        from isaaclab_rl.entrypoints.common import close_env, create_isaaclab_env

        partners, filters = contact_filters(env_cfg)
        env_cfg.scene.torso_contacts.filter_prim_paths_expr = filters
        raw_env = create_isaaclab_env(
            args.task, env_cfg, SimpleNamespace(frontend="torch"), convert_marl_to_single_agent=False
        )
        archive = None
        try:
            raw = raw_env.unwrapped
            raw.render_enabled = False
            env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
            runner = create_task_runner(env, agent_cfg, evaluation=True)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=raw.device)
            term = raw.command_manager.get_term("wrist_pose")
            checkpoint_level = term.range_curriculum.level
            if args.level is not None:
                # Override after checkpoint restoration, before reset assigns
                # episode-owned target and Lower command ranges. Evaluation
                # mode keeps both the course and sampling scheduler frozen.
                term.range_curriculum.level = args.level
            force_category(term, args.category)
            obs, _ = env.reset()
            policy.reset()
            print(
                f"[failures] checkpoint_level={checkpoint_level} collection_level={term.range_curriculum.level} "
                f"category={args.category} policy_output={args.policy_output}",
                flush=True,
            )
            body_names = list(raw.scene["robot"].body_names)
            archive = FailureArchive(
                directory,
                {
                    "task": args.task,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "seed": args.seed,
                    "policy_output": args.policy_output,
                    "category": args.category,
                    "num_envs": raw.num_envs,
                    "control_dt": raw.step_dt,
                    "physics_dt": env_cfg.sim.dt,
                    "physics_substeps": env_cfg.decimation,
                    "level": term.range_curriculum.level,
                    "checkpoint_level": checkpoint_level,
                    "requested_level": args.level,
                    "history_s": args.history_s,
                    "limits": {k: getattr(args, k) for k in ("episodes", "max_failures", "max_steps")},
                    "partners": partners,
                    "filter_paths": filters,
                    "body_names": body_names,
                    "joint_names": list(raw.scene["robot"].joint_names),
                    "coordinate_frame": "world axes minus env origin",
                    "force_kind": "normal contact force in N; per-physics-step samples, oldest first",
                    "clip_semantics": "recorded states through the failure step, before automatic reset",
                    "source_usd": env_cfg.scene.robot.spawn.usd_path,
                    "variants": env_cfg.scene.robot.spawn.variants,
                    "terminations": {
                        name: raw.termination_manager.get_term_cfg(name).params["threshold"]
                        for name in ("base_contact", "knee_contact")
                    },
                    "torso_contact_policy": {
                        "force_measure": "sum_filtered_arm_normal_force_magnitudes",
                        "other_threshold_n": raw.termination_manager.get_term_cfg("base_contact").params.get(
                            "other_threshold", 1.0
                        ),
                    },
                },
            )
            robot_prim = sim_utils.get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")
            write_json(archive.directory / "geometry.json", export_geometry(robot_prim, body_names), indent=None)
            dump_yaml(str(archive.directory / "effective_env.yaml"), env_cfg)
            dump_yaml(str(archive.directory / "effective_agent.yaml"), agent_cfg)
            miner = FailureMiner(
                raw, archive, partners, history_s=args.history_s, episodes=args.episodes, max_failures=args.max_failures
            )
            raw.failure_miner = miner
            started = time.perf_counter()
            steps, status = 0, "max_steps"
            try:
                with torch.inference_mode():
                    for step in range(args.max_steps):
                        actions = policy(obs, stochastic_output=True) if args.policy_output == "sample" else policy(obs)
                        if not torch.isfinite(actions).all():
                            raise RuntimeError("Policy produced non-finite actions")
                        obs, _, dones, _ = env.step(actions)
                        policy.reset(dones)
                        steps = step + 1
                        if step % 100 == 0 or miner.done:
                            archive.progress("mining", control_steps=steps)
                            print(
                                f"[failures] steps={steps} episodes={archive.completed}/{args.episodes} "
                                f"failures={archive.saved}/{args.max_failures}",
                                flush=True,
                            )
                        if miner.done:
                            status = "max_failures" if archive.saved >= args.max_failures else "episodes"
                            break
            except KeyboardInterrupt:
                status = "interrupted"
            finally:
                raw.failure_miner = None
                archive.progress(
                    status,
                    control_steps=steps,
                    wall_seconds=time.perf_counter() - started,
                    sampled_robot_seconds=miner.elapsed_robot_s,
                    unrecorded_final_step_completions=miner.unrecorded_completions,
                    unfinished_episodes=int((term.episode_age_steps() > 0).sum()),
                    unfinished_robot_seconds=float(term.elapsed_seconds(slice(None)).sum()),
                )
            print(f"[failures] Saved {archive.saved} clips to {archive.directory.resolve()}")
            return 0
        except BaseException as error:
            if archive is not None:
                archive.progress("error", error=str(error))
            raise
        finally:
            if archive is not None:
                archive.close()
            close_env(raw_env)
