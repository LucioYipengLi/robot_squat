"""Restore an Upper checkpoint, freeze its curriculum and collect bounded failure clips."""

import hashlib
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .storage import FailureArchive, write_json


def configure_checkpoint(env_cfg, agent_cfg, checkpoint):
    """Match the saved network and restore its command timing contract before loading."""
    import torch

    from isaaclab.utils.io import load_yaml

    from g1.tasks.upper_limb_agent.checkpoint_observations import configure_checkpoint_observations

    agent_path = checkpoint.parent / "params/agent.yaml"
    if agent_path.is_file():
        agent_cfg.from_dict(load_yaml(str(agent_path)))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configure_checkpoint_observations(env_cfg, saved=saved)
    course = (saved.get("infos") or {}).get("g1_upper_ranges") or {}
    contract = course.get("contract", {})
    target = contract.get("target") or {}
    wrist = env_cfg.commands.wrist_pose
    initial_level = contract.get("cfg", {}).get("initial_level")
    if initial_level is not None:
        wrist.range_curriculum.initial_level = initial_level
    saved_sampling = course.get("planar_sampling")
    if saved_sampling is not None and wrist.range_curriculum.planar_sampling is not None:
        wrist.range_curriculum.planar_sampling.from_dict(saved_sampling["cfg"])
    if "resampling_time_range" in target:
        wrist.resampling_time_range = tuple(target["resampling_time_range"])
    wrist.regular_retract_sampling.probability = 0.0
    wrist.regular_retract_sampling.dual_probability = 0.0
    if "regular_retract_sampling" in target:
        wrist.regular_retract_sampling.from_dict(target["regular_retract_sampling"])
    planar = wrist.planar_task
    if planar is not None:
        transition = target.get("planar_task", {}).get("transition_sampling")
        planar.transition_sampling = type(planar.transition_sampling)()
        if transition is not None:
            planar.transition_sampling.from_dict(transition)


def limit_near_planar_actions(actions, entered_near, velocity_offset, scale):
    """Cap the normalized lateral and yaw actions before their usual physical decoding."""
    if scale == 1.0:
        return actions
    actions = actions.clone()
    lateral_yaw = actions[:, velocity_offset + 1 : velocity_offset + 3]
    lateral_yaw.copy_(lateral_yaw.clamp(-scale, scale).where(entered_near[:, None], lateral_yaw))
    return actions


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
    from .paired_recording import FixedSwitchPairs
    from .pairs import load_switch_pairs
    from .recording import FailureMiner, force_category, prepare_recording
    from .switches import RegularSwitchTracker

    checkpoint = args.checkpoint.resolve(strict=True)
    directory = args.output or Path("outputs/failures") / (
        datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    if directory.exists():
        raise FileExistsError(f"Choose a new output directory: {directory}")
    suite = load_switch_pairs(args.switch_pairs) if args.switch_pairs is not None else None
    if args.switch_cases:
        requested = set(args.switch_cases)
        pairs = [pair for pair in suite["pairs"] if pair["id"] in requested]
        if missing := requested - {pair["id"] for pair in pairs}:
            raise ValueError(f"Unknown switch case IDs: {', '.join(sorted(missing))}")
        suite["pairs"] = pairs
    level = args.level
    episodes, max_failures, num_envs = args.episodes, args.max_failures, args.num_envs
    near_scale = args.planar_near_lower_scale
    startup_s = 0.2 if args.planar_startup else None
    planar_hold = None
    hold_source = args.planar_hold_from or args.planar_hold_waist_from
    if hold_source is not None:
        from .planar_hold import PlanarArmHold

        planar_hold = PlanarArmHold(hold_source, hold_waist=args.planar_hold_waist_from is not None)
        startup_s = planar_hold.hold_s
        if level is None:
            level = planar_hold.baseline["level"]
    first_episode_only = near_scale is not None or startup_s is not None
    if first_episode_only:
        episodes, max_failures = num_envs, num_envs + 1
    elif suite is not None:
        if level is not None and level != suite["level"]:
            raise ValueError("--level must match the switch-pair suite level")
        level = suite["level"]
        episodes = max_failures = (1 if suite["schema_version"] == 2 else 2) * len(suite["pairs"])
        num_envs = min(num_envs, episodes)
    configure_logging(save_logs_to_file=False)
    configure_seed(args.seed)
    version = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(
        args.task, "rsl_rl_cfg_entry_point", play_mode=True, overrides=["physics=isaacsim_physx"]
    )
    num_levels = env_cfg.commands.wrist_pose.range_curriculum.num_levels
    if level is not None and not 0 <= level < num_levels:
        raise ValueError(f"--level must be between 0 and {num_levels - 1}, got {args.level}")
    configure_checkpoint(env_cfg, agent_cfg, checkpoint)
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version)
    prepare_recording(env_cfg, num_envs)
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
            checkpoint_refresh_s = env_cfg.commands.wrist_pose.resampling_time_range
            checkpoint_retract_probability = env_cfg.commands.wrist_pose.regular_retract_sampling.probability
            checkpoint_dual_retract_probability = env_cfg.commands.wrist_pose.regular_retract_sampling.dual_probability
            if args.regular_refresh_s is not None:
                term.cfg.resampling_time_range = tuple(args.regular_refresh_s)
                env_cfg.commands.wrist_pose.resampling_time_range = term.cfg.resampling_time_range
            if args.regular_sampling == "native":
                for cfg in (term.cfg, env_cfg.commands.wrist_pose):
                    cfg.regular_retract_sampling.probability = 0.0
                    cfg.regular_retract_sampling.dual_probability = 0.0
                configure_seed(args.seed)
            if level is not None:
                # Override after checkpoint restoration, before reset assigns
                # episode-owned target and Lower command ranges. Evaluation
                # mode keeps both the course and sampling scheduler frozen.
                term.range_curriculum.level = level
            fixed_pairs = None
            if args.planar_sampling is not None:
                # Use the same geometry, task mix and initial random stream on
                # both sides of a comparison, after restoring each checkpoint.
                enabled = args.planar_sampling == "transition"
                cfg = type(term.cfg.planar_task.transition_sampling)(enabled=enabled)
                term.cfg.planar_task.transition_sampling = cfg
                env_cfg.commands.wrist_pose.planar_task.transition_sampling = cfg
                term.cfg.planar_task.validate_parameters()
                term.set_teleop("auto", "planar", None)
                configure_seed(args.seed)
            elif suite is None:
                force_category(term, args.category)
            else:
                if raw.termination_manager.get_term_cfg("base_contact").params["threshold"] != 20.0:
                    raise ValueError("Fixed switch-pair evaluation requires the 20 N arm/torso termination threshold")
                fixed_pairs = FixedSwitchPairs(term, suite, diagnostic=args.switch_diagnostic)
                term._teleop = fixed_pairs
            obs, _ = env.reset()
            policy.reset()
            if near_scale is not None:
                action_manager = raw.action_manager
                velocity_index = action_manager.active_terms.index("lower_command")
                velocity_offset = sum(action_manager.action_term_dim[:velocity_index])
                if action_manager.action_term_dim[velocity_index] != 3:
                    raise ValueError("Planar Lower diagnostic requires the [vx, vy, yaw] action term")
            switch_tracker = None
            if args.category in ("regular", "mixed"):
                hard_threshold = raw.termination_manager.get_term_cfg("base_contact").params["threshold"]
                switch_tracker = RegularSwitchTracker(raw, term, hard_threshold)
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
                    "planar_sampling": "transition" if term.cfg.planar_task.transition_sampling.enabled else "legacy",
                    "planar_near_lower_scale": near_scale,
                    **(
                        {"planar_startup": {"window_s": startup_s, "initial_contact_sampled": False}}
                        if startup_s is not None
                        else {}
                    ),
                    **({"planar_hold": planar_hold.settings()} if planar_hold is not None else {}),
                    "checkpoint_regular_refresh_s": checkpoint_refresh_s,
                    "regular_refresh_s": env_cfg.commands.wrist_pose.resampling_time_range if suite is None else None,
                    "checkpoint_regular_retract_probability": checkpoint_retract_probability,
                    "regular_retract_probability": (
                        term.cfg.regular_retract_sampling.probability if suite is None else None
                    ),
                    "checkpoint_regular_dual_retract_probability": checkpoint_dual_retract_probability,
                    "regular_dual_retract_probability": (
                        term.cfg.regular_retract_sampling.dual_probability if suite is None else None
                    ),
                    "regular_switch_evaluation": switch_tracker.settings() if switch_tracker is not None else None,
                    "switch_pair_evaluation": {
                        "cases": [{"id": pair["id"], "group": pair["group"]} for pair in fixed_pairs.pairs],
                        "trial_mode": "dual_arm" if fixed_pairs.dual else "mirrored_single_arm",
                        "switch_time_s": fixed_pairs.switch_step * raw.step_dt,
                        "trial_duration_s": fixed_pairs.end_step * raw.step_dt,
                    }
                    if fixed_pairs is not None
                    else None,
                    "history_s": args.history_s,
                    "limits": {"episodes": episodes, "max_failures": max_failures, "max_steps": args.max_steps},
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
                    **(
                        {"switch_diagnostic": fixed_pairs.diagnostic.settings()}
                        if fixed_pairs is not None and fixed_pairs.diagnostic is not None
                        else {}
                    ),
                },
                planar_hold=planar_hold,
            )
            if suite is not None:
                write_json(archive.directory / "switch_pairs.json", suite)
            robot_prim = sim_utils.get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")
            write_json(archive.directory / "geometry.json", export_geometry(robot_prim, body_names), indent=None)
            dump_yaml(str(archive.directory / "effective_env.yaml"), env_cfg)
            dump_yaml(str(archive.directory / "effective_agent.yaml"), agent_cfg)
            miner = FailureMiner(
                raw,
                archive,
                partners,
                history_s=args.history_s,
                episodes=episodes,
                max_failures=max_failures,
                switch_tracker=switch_tracker,
                fixed_pairs=fixed_pairs,
                first_episode_only=first_episode_only,
                startup_s=startup_s,
            )
            raw.failure_miner = miner
            started = time.perf_counter()
            steps, status = 0, "max_steps"
            try:
                with torch.inference_mode():
                    if startup_s is not None:
                        initial = miner.capture_startup_initial()
                        if planar_hold is not None:
                            planar_hold.prepare(raw, term, initial)
                    for step in range(args.max_steps):
                        actions = policy(obs, stochastic_output=True) if args.policy_output == "sample" else policy(obs)
                        if not torch.isfinite(actions).all():
                            raise RuntimeError("Policy produced non-finite actions")
                        if planar_hold is not None:
                            actions = planar_hold.intervene(actions, step, miner.pending_first_episodes)
                        if near_scale is not None:
                            actions = limit_near_planar_actions(
                                actions, term.planar_first_near_step >= 0, velocity_offset, near_scale
                            )
                        if fixed_pairs is not None and fixed_pairs.diagnostic is not None:
                            actions = fixed_pairs.diagnostic.intervene(actions)
                        obs, _, dones, _ = env.step(actions)
                        policy.reset(dones)
                        steps = step + 1
                        if step % 100 == 0 or miner.done:
                            archive.progress("mining", control_steps=steps)
                            print(
                                f"[failures] steps={steps} episodes={archive.completed}/{episodes} "
                                f"failures={archive.saved}/{max_failures}",
                                flush=True,
                            )
                        if miner.done:
                            status = (
                                "max_failures" if fixed_pairs is None and archive.saved >= max_failures else "episodes"
                            )
                            break
            except KeyboardInterrupt:
                status = "interrupted"
            finally:
                raw.failure_miner = None
                ages = term.episode_age_steps()
                unfinished = ages > 0 if fixed_pairs is None else (fixed_pairs.trial_ids >= 0) & (ages > 0)
                if miner.pending_first_episodes is not None:
                    unfinished &= miner.pending_first_episodes
                archive.progress(
                    status,
                    control_steps=steps,
                    wall_seconds=time.perf_counter() - started,
                    sampled_robot_seconds=miner.elapsed_robot_s,
                    unrecorded_final_step_completions=miner.unrecorded_completions,
                    unfinished_episodes=int(unfinished.sum()),
                    unfinished_robot_seconds=float((ages * unfinished).sum()) * raw.step_dt,
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
