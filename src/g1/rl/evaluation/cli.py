"""Simulator-free controller for versioned, headless evaluation suites."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .jobs import build_manifest, checkpoint_records, execute_jobs, validate_resume, write_json


def run(argv: list[str]) -> int:
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
    parser.add_argument("--episodes-per-seed", type=int)
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
                args.agent_config,
                args.allow_unverified_checkpoint,
            )
            if any(value is not None and value is not False for value in changes):
                raise ValueError("--resume reuses its saved configuration; omit configuration overrides")
            directory = args.resume.resolve()
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            validate_resume(manifest, require_source=not args.report_only)
        else:
            from .config import load_suite, validate_suite

            config = load_suite(args.suite or "g1_core")
            if args.task:
                config["task"] = args.task
            if args.policy_output:
                config["policy_outputs"] = ["mean", "sample"] if args.policy_output == "both" else [args.policy_output]
            if args.seeds is not None:
                config["benchmark"]["seeds"] = args.seeds
            if args.episodes_per_seed is not None:
                config["benchmark"]["episodes_per_seed"] = args.episodes_per_seed
            config = validate_suite(config)
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
            from .jobs import digest_file

            for checkpoint in checkpoints:
                adjacent = Path(checkpoint["path"]).parent / "params" / "agent.yaml"
                resolved_agent = agent_path or (adjacent if adjacent.is_file() else None)
                checkpoint["agent_config"] = str(resolved_agent) if resolved_agent else None
                checkpoint["agent_config_sha256"] = digest_file(resolved_agent) if resolved_agent else None
                adjacent_env = Path(checkpoint["path"]).parent / "params" / "env.yaml"
                checkpoint["env_config"] = str(adjacent_env) if adjacent_env.is_file() else None
                checkpoint["env_config_sha256"] = digest_file(adjacent_env) if adjacent_env.is_file() else None
            if agent_path:
                runtime["agent_config_sha256"] = digest_file(agent_path)
            manifest = build_manifest(config, checkpoints, runtime)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            directory = (args.output or Path("logs/eval/g1_flat") / timestamp).resolve()
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
        return 0 if complete else 1
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 2
