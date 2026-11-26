"""Run the v10 Upper Mobile domain-randomization phases in sequence."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
PHASE_BUDGETS = ((2, 4000), (3, 10000), (4, 6000))
V10_OVERRIDES = (
    "physics=isaacsim_physx",
    "env.commands.wrist_pose.range_curriculum.advance=false",
    "env.commands.wrist_pose.range_curriculum.freeze_planar_sampling=true",
    "agent.mirror_augmentation=true",
    "env.commands.wrist_pose.planar_approach.enabled=true",
    "env.commands.wrist_pose.planar_approach.geometry_enabled=true",
    "env.commands.wrist_pose.planar_approach.arm_posture_weight=-0.35",
    "env.commands.wrist_pose.planar_approach.wrist_posture_weight=-0.05",
    "env.commands.wrist_pose.planar_approach.waist_distance_axis_scale=[0.25,0.50,0.75]",
    "env.commands.wrist_pose.planar_approach.waist_large_heading_yaw_gain=0.50",
    "env.rewards.lower_command_magnitude.params.near_planar_gain=4.0",
)


@dataclass(frozen=True)
class CheckpointProgress:
    iteration: int
    phase: int
    level: int
    planar_probability: float
    archived_lower: Path | None


def read_checkpoint(path: Path) -> CheckpointProgress:
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    infos = saved.get("infos") or {}
    control = infos.get("g1_control_interface") or {}
    course = infos.get("g1_upper_ranges") or {}
    if control.get("policy_observation_dim") != 660 or "level" not in course:
        raise ValueError(f"Expected a 660-observation Upper Mobile checkpoint with curriculum state: {path}")

    iteration = saved.get("iter")
    if type(iteration) is not int or iteration < 0:
        raise ValueError(f"Invalid saved training iteration in {path}")
    dr = infos.get("g1_upper_domain_randomization") or {}
    sampling = course.get("planar_sampling")
    if sampling is None:
        # A fixed planar probability is part of the command contract.
        planar_probability = course["contract"]["target"].get("planar_task", {}).get("probability", 0.0)
    else:
        planar_probability = sampling["probability"]
    lower_artifact = (infos.get("g1_frozen_lower") or {}).get("artifact")
    archived_lower = path.parent / lower_artifact if lower_artifact else None
    if archived_lower is not None and not archived_lower.is_file():
        archived_lower = None
    return CheckpointProgress(
        iteration, dr.get("settings", {}).get("phase", 1), course["level"], planar_probability, archived_lower
    )


def run_training(command: list[str], console_log: Path) -> None:
    with (
        console_log.open("x", encoding="utf-8", buffering=1) as log,
        subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace"
        ) as process,
    ):
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        print(f"Training console log: {console_log}", file=sys.stderr)
        raise subprocess.CalledProcessError(status, command)


def train_phases(checkpoint: Path) -> Path:
    progress = read_checkpoint(checkpoint)
    if progress.phase not in (1, 2):
        raise ValueError("The Phase 2 start checkpoint must come from Phase 1 or Phase 2")

    lower_override = []
    if progress.archived_lower is not None:
        quoted_path = json.dumps(str(progress.archived_lower), ensure_ascii=False)
        lower_override.append(f"env.actions.joint_pos.checkpoint_path={quoted_path}")
    chain_id = uuid4().hex[:8]
    for phase, budget in PHASE_BUDGETS:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"upper_dr_{chain_id}_phase{phase}"
        run_dir = ROOT / "logs" / "rsl_rl" / "g1_upper_mobile" / f"{timestamp}_{run_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        console_log = run_dir / "console.log"

        # RSL-RL starts at saved['iter'] and saves the last loop iteration.
        final_iteration = progress.iteration + budget - 1
        output = run_dir / f"model_{final_iteration}.pt"
        print(
            f"Phase {phase} | {budget:,} iterations | L{progress.level}, Planar {progress.planar_probability:.0%}",
            flush=True,
        )
        print(
            f"Input checkpoint:  {checkpoint}\nExpected output:   {output}\nConsole log:       {console_log}",
            flush=True,
        )
        run_training(
            [
                sys.executable,
                "-u",
                "-m",
                "g1.rl",
                "train",
                "--task",
                "Template-G1-Upper-Mobile-v0",
                "--checkpoint",
                str(checkpoint),
                "--resume_optimizer_state",
                "--num_envs",
                "4096",
                "--seed",
                "1217707953",
                "--max_iterations",
                str(budget),
                "--viz",
                "none",
                "--run_timestamp",
                timestamp,
                "--run_name",
                run_name,
                f"env.domain_randomization.phase={phase}",
                *V10_OVERRIDES,
                *lower_override,
            ],
            console_log,
        )
        # The trainer can exit normally after Ctrl+C. Require the full-budget checkpoint.
        if not output.is_file():
            raise FileNotFoundError(f"Phase {phase} did not write its final checkpoint: {output}")
        completed = read_checkpoint(output)
        if (completed.phase, completed.iteration) != (phase, final_iteration):
            raise ValueError(f"Unexpected phase or iteration in output checkpoint: {output}")
        if (completed.level, completed.planar_probability) != (progress.level, progress.planar_probability):
            raise ValueError(f"Level or Planar probability changed in Phase {phase}: {output}")
        checkpoint, progress = output, completed

    return checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, help="Explicit starting Upper Mobile checkpoint")
    args = parser.parse_args(argv)
    if not args.checkpoint.strip():
        parser.error("--checkpoint requires a non-empty file path")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"--checkpoint must name an existing file: {checkpoint}")

    try:
        final = train_phases(checkpoint)
    except KeyboardInterrupt:
        print("Upper DR sequence interrupted.", file=sys.stderr)
        return 130
    except subprocess.CalledProcessError as error:
        print(f"Upper training exited with status {error.returncode}.", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 128 - error.returncode
    except (OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as error:
        print(f"Upper DR sequence stopped: {error}", file=sys.stderr)
        return 1

    print(f"Final Phase 4 checkpoint: {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
