"""Simulator-free controller for versioned, headless evaluation suites."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .jobs import build_manifest, checkpoint_records, execute_jobs, validate_resume, write_json
from .upper_fixed_targets import SCHEMA_VERSION as UPPER_SCHEMA_VERSION
from .upper_fixed_targets import upper_height_bounds
from .upper_q import MODES as Q_MODES
from .upper_q import PROTOCOL as Q_PROTOCOL


def _check_task_family(config, expected):
    actual = "upper_limb_agent" if config["schema_version"] == UPPER_SCHEMA_VERSION else "lower_limb_agent"
    if expected is not None and expected != actual:
        raise ValueError(f"Suite {config['suite']} requires --task-family {actual}")
    return actual


def run(argv: list[str], *, task_family=None) -> int:
    parser = argparse.ArgumentParser(description="Compare G1 checkpoints using fixed, versioned evaluation suites.")
    parser.add_argument("--suite", "--eval-config", default=None, help="Built-in suite name or YAML/JSON path")
    parser.add_argument("--checkpoints", nargs="+", help="Files/globs, optionally LABEL=PATH")
    parser.add_argument("--task", help="Override the suite's evaluation task; creates a different benchmark identity")
    parser.add_argument("--num_envs", "--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--physics", default=None, help="Physics preset (default: isaacsim_physx)")
    parser.add_argument("--output", type=Path, help="New output directory")
    parser.add_argument("--resume", type=Path, help="Continue an existing manifest without changing conditions")
    parser.add_argument("--policy-output", choices=("mean", "sample", "both"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--episodes-per-seed", type=int, help="Upper: episodes per target group per seed")
    parser.add_argument("--q-mode", choices=(*Q_MODES, "both"), help="Workspace Q target mode (default: both)")
    for side in ("left", "right"):
        parser.add_argument(
            f"--{side}-wrist-pose",
            type=float,
            nargs=6,
            metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
            help=f"Workspace Q: fixed {side} wrist position (m) and roll/pitch/yaw (deg)",
        )
    parser.add_argument(
        "--frozen-lower-checkpoint", type=Path, help="Upper: evaluate with a different frozen Lower model_*.pt"
    )
    parser.add_argument(
        "--upper-height-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Upper: fix the height action range in m inside both policies' bounds",
    )
    parser.add_argument("--repetitions", type=int, help="Lower suite (schema 2): repeated full batches per stage")
    parser.add_argument("--episode-length-s", type=float, help="Lower or workspace Q suite: seconds per episode")
    parser.add_argument("--height-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--vx-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--vy-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--wz-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument(
        "--push-heights", type=float, nargs="+", help="Lower suite (schema 2): fixed height offsets in m"
    )
    parser.add_argument(
        "--agent-config", type=Path, help="Explicit shared agent YAML; otherwise use adjacent params/agent.yaml"
    )
    parser.add_argument(
        "--allow-unverified-checkpoint", action="store_true", help="Allow legacy weights without task metadata"
    )
    parser.add_argument("--headless", action="store_true", help="Accepted for clarity; eval always runs headless")
    parser.add_argument(
        "--progress",
        choices=("auto", "plain", "off"),
        default="auto",
        help="Progress display: live terminal bars, plain log lines, or off",
    )
    parser.add_argument("--plan-only", action="store_true", help="Write the manifest without launching physics")
    parser.add_argument(
        "--report-only", action="store_true", help="Regenerate a report from an existing --resume directory"
    )
    parser.add_argument("--worker-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-job", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_manifest:
        if not args.worker_job:
            parser.error("--worker-manifest requires --worker-job")
        from .worker import run_worker

        return run_worker(args.worker_manifest, args.worker_job)
    if args.worker_job:
        parser.error("--worker-job requires --worker-manifest")
    if args.report_only and not args.resume:
        parser.error("--report-only requires --resume")
    try:
        if args.resume:
            changes = (
                args.suite,
                args.checkpoints,
                args.task,
                args.num_envs,
                args.device,
                args.physics,
                args.output,
                args.policy_output,
                args.seeds,
                args.episodes_per_seed,
                args.q_mode,
                args.left_wrist_pose,
                args.right_wrist_pose,
                args.frozen_lower_checkpoint,
                args.upper_height_range,
                args.agent_config,
                args.allow_unverified_checkpoint,
                args.repetitions,
                args.episode_length_s,
                args.height_range,
                args.vx_range,
                args.vy_range,
                args.wz_range,
                args.push_heights,
            )
            if any(value is not None and value is not False for value in changes):
                raise ValueError("--resume reuses its saved configuration; omit configuration overrides")
            directory = args.resume.resolve()
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            _check_task_family(manifest["config"], task_family)
            validate_resume(manifest, require_source=not args.report_only)
        else:
            from .config import load_suite, validate_suite

            config = load_suite(args.suite or "g1_core")
            four_stage_options = (
                args.repetitions,
                args.height_range,
                args.vx_range,
                args.vy_range,
                args.wz_range,
                args.push_heights,
            )
            if config["schema_version"] == 2:
                if args.episodes_per_seed is not None:
                    raise ValueError("Lower suites (schema 2) use --repetitions instead of --episodes-per-seed")
                benchmark = config["benchmark"]
                if args.repetitions is not None:
                    benchmark["repetitions"] = args.repetitions
                if args.episode_length_s is not None:
                    benchmark["episode_length_s"] = args.episode_length_s
                for key, value in (
                    ("height_offset", args.height_range),
                    ("lin_vel_x", args.vx_range),
                    ("lin_vel_y", args.vy_range),
                    ("ang_vel_z", args.wz_range),
                ):
                    if value is not None:
                        benchmark["ranges"][key] = value
                if args.push_heights is not None:
                    benchmark["push_height_offsets"] = args.push_heights
                elif args.height_range is not None:
                    # Regenerate height buckets inside the new range during validation.
                    benchmark.pop("push_height_offsets", None)
            elif any(value is not None for value in four_stage_options):
                raise ValueError("These options require a schema 2 suite: g1_lower_four_stage or g1_lower_push")
            elif args.episode_length_s is not None:
                if config.get("protocol") != Q_PROTOCOL:
                    raise ValueError("--episode-length-s requires a schema 2 or workspace Q suite")
                config["benchmark"]["episode_length_s"] = args.episode_length_s
            if args.task:
                config["task"] = args.task
            if args.policy_output:
                config["policy_outputs"] = ["mean", "sample"] if args.policy_output == "both" else [args.policy_output]
            if args.seeds is not None:
                config["benchmark"]["seeds"] = args.seeds
            if args.episodes_per_seed is not None:
                config["benchmark"]["episodes_per_seed"] = args.episodes_per_seed
            if any(value is not None for value in (args.q_mode, args.left_wrist_pose, args.right_wrist_pose)):
                if config.get("protocol") != Q_PROTOCOL:
                    raise ValueError("--q-mode and --left/right-wrist-pose require a workspace Q suite")
                if args.q_mode is not None:
                    config["benchmark"]["modes"] = list(Q_MODES) if args.q_mode == "both" else [args.q_mode]
                for side, pose in (("left", args.left_wrist_pose), ("right", args.right_wrist_pose)):
                    if pose is not None:
                        config["benchmark"]["fixed_wrist_poses"][side] = {"position_m": pose[:3], "rpy_deg": pose[3:]}
            config = validate_suite(config)
            family = _check_task_family(config, task_family)
            if family == "upper_limb_agent" and args.allow_unverified_checkpoint:
                raise ValueError("Upper eval requires verified Mobile control and frozen Lower metadata")
            if family != "upper_limb_agent" and (args.frozen_lower_checkpoint or args.upper_height_range is not None):
                raise ValueError("--frozen-lower-checkpoint and --upper-height-range require an Upper Eval suite")
            if args.num_envs is not None and args.num_envs <= 0:
                raise ValueError("--num_envs must be positive")
            if not args.checkpoints:
                raise ValueError("Provide --checkpoints or --resume")
            checkpoints = checkpoint_records(args.checkpoints)
            agent_path = args.agent_config.resolve() if args.agent_config else None
            if agent_path and not agent_path.is_file():
                raise ValueError(f"Agent config does not exist: {agent_path}")
            runtime = {
                "num_envs": args.num_envs or 128,
                "device": args.device or "cuda:0",
                "physics": args.physics or "isaacsim_physx",
                "agent_config": str(agent_path) if agent_path else None,
                "agent_config_sha256": None,
                "allow_unverified_checkpoint": args.allow_unverified_checkpoint,
            }
            from .jobs import digest_file, frozen_lower_record

            lower_override = frozen_lower_record(args.frozen_lower_checkpoint) if args.frozen_lower_checkpoint else None
            upper_height = (
                upper_height_bounds(args.upper_height_range, "--upper-height-range")
                if args.upper_height_range is not None
                else None
            )
            for checkpoint in checkpoints:
                adjacent = Path(checkpoint["path"]).parent / "params" / "agent.yaml"
                resolved_agent = agent_path or (adjacent if adjacent.is_file() else None)
                checkpoint["agent_config"] = str(resolved_agent) if resolved_agent else None
                checkpoint["agent_config_sha256"] = digest_file(resolved_agent) if resolved_agent else None
                adjacent_env = Path(checkpoint["path"]).parent / "params" / "env.yaml"
                checkpoint["env_config"] = str(adjacent_env) if adjacent_env.is_file() else None
                checkpoint["env_config_sha256"] = digest_file(adjacent_env) if adjacent_env.is_file() else None
                if lower_override is not None:
                    checkpoint["frozen_lower_override"] = lower_override
                if upper_height is not None:
                    checkpoint["upper_height_range"] = list(upper_height)
            if agent_path:
                runtime["agent_config_sha256"] = digest_file(agent_path)
            manifest = build_manifest(config, checkpoints, runtime)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            task_dir = "g1_upper_mobile" if family == "upper_limb_agent" else "g1_flat"
            directory = (args.output or Path("logs/eval") / task_dir / timestamp).resolve()
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "manifest.json", manifest)
        print(
            f"Evaluation: {directory}\nSuite: {manifest['config']['suite']}; cells: {len(manifest['jobs'])}", flush=True
        )
        if args.plan_only:
            return 0
        from .progress import EvaluationProgress

        report = None
        with EvaluationProgress(len(manifest["jobs"]), mode=args.progress) as progress:
            if args.report_only:
                from .jobs import read_result

                results = [
                    result for job in manifest["jobs"] if (result := read_result(directory, job, manifest)) is not None
                ]
                complete = len(results) == len(manifest["jobs"])
                progress.set_cached(len(results))
            else:
                results, complete = execute_jobs(directory, manifest, progress=progress)
            if results:
                from .reporting import write_report

                progress.reporting()
                report = write_report(directory, manifest["config"], results)
                write_json(directory / "report_latest.json", report)
            progress.finish(success=complete)
        if report:
            print(
                f'Report: {report["report_dir"]}\nTensorBoard: tensorboard --logdir "{report["tensorboard_dir"]}"',
                flush=True,
            )
            if "replay_dir" in report:
                print(f'Viewer: python -m g1.rl failures view --run "{report["replay_dir"]}"', flush=True)
        return 0 if complete else 1
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 2
