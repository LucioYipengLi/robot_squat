"""One simulator process per checkpoint/condition/seed/action-output cell."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import uuid
from pathlib import Path

from .jobs import canonical_digest, result_path, validate_resume, write_json


def episode_specs(benchmark: dict) -> list[dict]:
    from .config import mode_episode_quotas

    quotas = mode_episode_quotas(benchmark["episodes_per_seed"], benchmark["mode_weights"])
    # Interleave modes so every batch has a representative distribution, while preserving IDs.
    return [
        {"episode_id": number, "mode": mode}
        for number in range(max(quotas))
        for mode, count in enumerate(quotas)
        if number < count
    ]


class EpisodeCollector:
    """Accept exactly the assigned first episode from every active batch slot."""

    def __init__(self, controller):
        self.controller = controller
        self.specs = []
        self.completed = set()
        self.records = []

    def start_batch(self, specs):
        self.specs = specs
        self.completed = set()

    def __call__(self, env_ids, values):
        from .config import MODE_NAMES

        ids = env_ids.cpu().tolist()
        plain = {key: tensor.cpu().tolist() for key, tensor in values.items() if key != "reasons"}
        reasons = {name: tensor.cpu().tolist() for name, tensor in values["reasons"].items()}
        accepted = []
        for index, slot in enumerate(ids):
            if slot >= len(self.specs) or slot in self.completed:
                continue
            spec = self.specs[slot]
            if int(plain["mode"][index]) != spec["mode"]:
                raise RuntimeError("Evaluation episode changed its assigned mode")
            record = {key: vector[index] for key, vector in plain.items() if key != "mode"}
            record.update(
                episode_id=spec["episode_id"],
                mode=MODE_NAMES[spec["mode"]],
                steps=int(record["steps"]),
                reasons={name: vector[index] for name, vector in reasons.items()},
                **self.controller.context(slot),
            )
            if spec["mode"] >= 2:
                record["error_pelvis_xy_m"] = None
            self.records.append(record)
            self.completed.add(slot)
            accepted.append(slot)
        if accepted:
            self.controller.complete(accepted)


def run_rollouts(
    env, policy, controller, benchmark: dict, *, seed: int, policy_output: str, progress=None
) -> list[dict]:
    import torch

    from .policy import policy_action

    collector = EpisodeCollector(controller)
    monitor = env.unwrapped.command_manager.get_term("task_command").performance
    previous_callback = monitor.episode_callback
    monitor.episode_callback = collector
    specs = episode_specs(benchmark)
    batches = (len(specs) + env.num_envs - 1) // env.num_envs

    def update_progress(batch_index, local_step=0, *, completed=False, force=False):
        if progress is not None:
            progress.update(
                "evaluating",
                force=force,
                episodes_completed=len(collector.records),
                episodes_total=len(specs),
                batches_completed=batch_index + int(completed),
                batches_total=batches,
                batch_index=batch_index + 1,
                batch_episodes_completed=len(collector.completed),
                batch_episodes_total=len(collector.specs),
                batch_step=min(local_step, env.unwrapped.max_episode_length),
                batch_steps_total=env.unwrapped.max_episode_length,
            )

    try:
        with torch.inference_mode():
            for batch_index, offset in enumerate(range(0, len(specs), env.num_envs)):
                batch = specs[offset : offset + env.num_envs]
                collector.start_batch(batch)
                controller.start_batch(batch)
                obs, _ = env.reset()
                policy.reset()
                update_progress(batch_index, force=True)
                for local_step in range(env.unwrapped.max_episode_length + 1):
                    controller.before_step()
                    action_seed = int.from_bytes(
                        hashlib.sha256(f"{seed}:{batch_index}:{local_step}:policy".encode()).digest()[:8], "little"
                    ) % (2**63)
                    actions = policy_action(policy, obs, stochastic=policy_output == "sample", seed=action_seed)
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Policy produced non-finite actions")
                    obs, _, dones, _ = env.step(actions)
                    policy.reset(dones)
                    update_progress(batch_index, local_step + 1)
                    if len(collector.completed) == len(batch):
                        break
                if len(collector.completed) != len(batch):
                    raise RuntimeError("Evaluation horizon ended before every assigned episode was finalized")
                update_progress(batch_index, local_step + 1, completed=True, force=True)
                print(f"Completed {len(collector.records)}/{len(specs)} assigned episodes", flush=True)
        if len(collector.records) != len(specs):
            raise RuntimeError("Evaluation episode quota mismatch")
        return collector.records
    finally:
        monitor.episode_callback = previous_callback


def _versions() -> dict:
    versions = {"python": sys.version, "platform": platform.platform()}
    for package in ("torch", "rsl-rl", "isaaclab", "isaacsim", "tensorboard", "isaaclab-physx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed-as-distribution"
    return versions


def run_worker(manifest_path: Path, job_id: str) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = next((job for job in manifest["jobs"] if job["job_id"] == job_id), None)
    if job is None:
        raise ValueError(f"Unknown evaluation job {job_id}")
    directory = manifest_path.parent
    cell = result_path(directory, job).parent
    cell.mkdir(parents=True, exist_ok=True)
    config, runtime = manifest["config"], manifest["runtime"]
    benchmark = config["benchmark"]
    from .progress import PROGRESS_ID_ENV, ProgressWriter

    progress = ProgressWriter(cell / "progress.json", job_id, os.environ.get(PROGRESS_ID_ENV) or uuid.uuid4().hex)
    progress.update("validating", episodes_total=benchmark["episodes_per_seed"])
    try:
        validate_resume(manifest)
        progress.update("importing")
        # Import simulation APIs only in this child process.
        import torch

        from isaaclab.app import launch_simulation
        from isaaclab.utils import configure_logging
        from isaaclab.utils.io import dump_yaml
        from isaaclab.utils.seed import configure_seed

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from isaaclab_tasks.utils import resolve_task_config

        import g1.tasks  # noqa: F401
        from g1.tasks.lower_limb_agent.evaluation import EvaluationScenarioController, prepare_env_cfg

        from .policy import load_policy

        configure_logging(save_logs_to_file=False)
        progress.update("configuring")
        env_cfg, agent_cfg = resolve_task_config(
            config["task"],
            "rsl_rl_cfg_entry_point",
            play_mode=False,
            overrides=[f"physics={runtime['physics']}"],
        )
        prepare_env_cfg(env_cfg, benchmark, job, runtime["num_envs"], job["seed"])
        env_cfg.sim.device = runtime["device"]
        launch_args = {"headless": True, "device": runtime["device"], "visualizer": None}
        configure_seed(job["seed"])
        progress.update("starting_simulation")
        with launch_simulation(env_cfg, launch_args):
            import gymnasium as gym

            progress.update("creating_environment")
            raw_env = gym.make(config["task"], cfg=env_cfg)
            try:
                raw_env.unwrapped.render_enabled = False
                # Agent-specific clipping is applied after its saved configuration is resolved.
                env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
                progress.update("loading_policy")
                policy, effective_agent, policy_metadata = load_policy(
                    env,
                    agent_cfg,
                    job["checkpoint"],
                    device=runtime["device"],
                    allow_unverified=runtime["allow_unverified_checkpoint"],
                )
                env.clip_actions = effective_agent.get("clip_actions")
                controller = EvaluationScenarioController(raw_env.unwrapped, benchmark, job, job["seed"])
                dump_yaml(str(cell / "effective_env.yaml"), env_cfg)
                dump_yaml(str(cell / "effective_agent.yaml"), effective_agent)
                records = run_rollouts(
                    env,
                    policy,
                    controller,
                    benchmark,
                    seed=job["seed"],
                    policy_output=job["policy_output"],
                    progress=progress,
                )
                progress.update("saving_results")
                device_name = torch.cuda.get_device_name(env_cfg.sim.device) if "cuda" in env_cfg.sim.device else "cpu"
                result = {
                    "schema_version": 1,
                    "status": "complete",
                    **job,
                    "episodes": records,
                    "metadata": {
                        **policy_metadata,
                        "versions": _versions(),
                        "device_name": device_name,
                        "config_sha256": manifest["config_sha256"],
                        "source_sha256": manifest["source_sha256"],
                        "environment": manifest["environment"],
                        "control_dt": raw_env.unwrapped.step_dt,
                        "upper_body_motion": benchmark["upper_body_motion"],
                        "upper_body_target_source": env_cfg.observations.policy.upper_body_target.params[
                            "target_source"
                        ],
                        "actual_episode_length_s": raw_env.unwrapped.max_episode_length * raw_env.unwrapped.step_dt,
                        "case_specs_sha256": canonical_digest(episode_specs(benchmark)),
                    },
                }
                write_json(result_path(directory, job), result)
                print(f"Saved {result_path(directory, job)}", flush=True)
            finally:
                progress.update("closing_simulation")
                raw_env.close()
        progress.update("finished")
        return 0
    except KeyboardInterrupt:
        progress.update("interrupted")
        raise
    except (Exception, SystemExit) as error:
        import traceback

        write_json(
            cell / "error.json",
            {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        progress.update("failed")
        traceback.print_exc()
        return 1
