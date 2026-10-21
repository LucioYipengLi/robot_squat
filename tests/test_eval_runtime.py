"""Evaluation manifests, actor isolation and first-episode scheduling without physics."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.utils import resolve_class
from tensordict import TensorDict

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

from g1.rl import __main__ as main_cli
from g1.rl.evaluation import cli, jobs
from g1.rl.evaluation.config import load_suite
from g1.rl.evaluation.policy import load_policy, policy_action
from g1.rl.evaluation.worker import EpisodeCollector, episode_specs, run_rollouts
from g1.tasks.lower_limb_agent.config.g1.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg
from g1.tasks.lower_limb_agent.curriculum_state import _compatibility


def test_eval_dispatch_does_not_import_simulator_for_help(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(SystemExit) as error:
        main_cli.main(["eval", "--help"])
    assert error.value.code == 0


def test_manifest_plan_resume_hashes_and_globs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "source_digest", lambda: "project-source")
    model = tmp_path / "run" / "model_100.pt"
    model.parent.mkdir()
    model.write_bytes(b"test checkpoint")
    records = jobs.checkpoint_records([str(model.parent / "*.pt"), str(model)])
    assert len(records) == 1
    manifest = jobs.build_manifest(load_suite("g1_robustness"), records, {"num_envs": 8})
    assert len(manifest["jobs"]) == 15
    assert len({job["job_id"] for job in manifest["jobs"]}) == 15
    jobs.validate_resume(manifest)
    changed = copy.deepcopy(manifest)
    changed["config"]["benchmark"]["episode_length_s"] = 9
    with pytest.raises(ValueError, match="modified"):
        jobs.validate_resume(changed)
    model.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="changed"):
        jobs.validate_resume(manifest)


def test_cli_plan_only_writes_complete_manifest(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"shape validation intentionally deferred to physics worker")
    output = tmp_path / "plan"
    assert (
        cli.run(
            [
                "--checkpoints",
                f"early={model}",
                "--output",
                str(output),
                "--plan-only",
                "--episodes-per-seed",
                "8",
                "--seeds",
                "17",
                "--policy-output",
                "both",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["jobs"]) == 2
    assert manifest["config"]["benchmark"]["episodes_per_seed"] == 8
    assert manifest["runtime"]["num_envs"] == 128
    assert not (output / "cases").exists()
    with pytest.raises(SystemExit):
        cli.run(["--resume", str(output), "--num_envs", "2"])


def test_checkpoint_paths_with_equals_and_portable_case_insensitive_labels(tmp_path):
    first = tmp_path / "phase=1" / "model.pt"
    first.parent.mkdir()
    first.write_bytes(b"first")
    second = tmp_path / "model_2.pt"
    second.write_bytes(b"second")
    assert jobs.checkpoint_records([str(first)])[0]["path"] == str(first.resolve())
    assert jobs.checkpoint_records([str(first.parent / "*.pt")])[0]["path"] == str(first.resolve())
    records = jobs.checkpoint_records([f"Early={first}", f"early={second}"])
    assert len({record["label"].casefold() for record in records}) == 2
    assert records[0]["label"] == "Early"


def test_episode_specs_exact_quotas_and_stable_ids():
    benchmark = load_suite("g1_core")["benchmark"]
    benchmark.update(episodes_per_seed=10, mode_weights=[4, 4, 2, 0])
    specs = episode_specs(benchmark)
    assert [sum(spec["mode"] == mode for spec in specs) for mode in range(4)] == [4, 4, 2, 0]
    assert len({(spec["mode"], spec["episode_id"]) for spec in specs}) == 10


def _record_values(modes):
    count = len(modes)
    return {
        "mode": torch.tensor(modes),
        "steps": torch.ones(count),
        "length_s": torch.ones(count),
        "terminated": torch.zeros(count, dtype=torch.bool),
        "time_out": torch.ones(count, dtype=torch.bool),
        "error_height_m": torch.zeros(count),
        "error_velocity_xy_mps": torch.zeros(count),
        "error_yaw_radps": torch.zeros(count),
        "error_pelvis_xy_m": torch.zeros(count),
        "pelvis_steps": torch.ones(count),
        "pelvis_error_sum": torch.zeros(count),
        "pelvis_success_steps": torch.ones(count),
        "reasons": {"time_out": torch.ones(count, dtype=torch.bool)},
    }


def test_collector_ignores_autoresets_and_unassigned_slots():
    stopped = []
    controller = SimpleNamespace(context=lambda slot: {"disturbance_count": 0}, complete=stopped.extend)
    collector = EpisodeCollector(controller)
    collector.start_batch([{"episode_id": 0, "mode": 0}, {"episode_id": 0, "mode": 2}])
    collector(torch.tensor([0, 1, 2]), _record_values([0, 2, 0]))
    collector(torch.tensor([0, 1]), _record_values([0, 2]))
    assert len(collector.records) == 2
    assert stopped == [0, 1]
    assert collector.records[1]["error_pelvis_xy_m"] is None


def test_rollouts_wait_for_slow_episodes_without_recounting_early_failures():
    benchmark = load_suite("g1_core")["benchmark"]
    benchmark.update(episodes_per_seed=4, mode_weights=[1, 1, 0, 0])
    monitor = SimpleNamespace(episode_callback=None)
    raw = SimpleNamespace(
        max_episode_length=3,
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(performance=monitor)),
    )
    controller = SimpleNamespace(context=lambda slot: {"disturbance_count": 0})
    state = {}
    updates = []

    class Telemetry:
        def update(self, stage=None, **values):
            updates.append({"stage": stage, **values})

    def start(specs):
        state["specs"] = specs

    controller.start_batch = start
    controller.before_step = lambda: None
    controller.complete = lambda ids: None

    def reset():
        state["step"] = 0
        return torch.zeros(2, 1), {}

    def step(actions):
        state["step"] += 1
        slots = [0] if state["step"] < 3 else [0, 1]
        values = _record_values([state["specs"][slot]["mode"] for slot in slots])
        values["terminated"][0] = True
        values["time_out"][0] = False
        monitor.episode_callback(torch.tensor(slots), values)
        return torch.zeros(2, 1), None, torch.ones(2), {}

    env = SimpleNamespace(unwrapped=raw, num_envs=2, reset=reset, step=step)

    class Policy(torch.nn.Module):
        def forward(self, obs, **kwargs):
            return torch.zeros(2, 1)

        def reset(self, dones=None):
            pass

    records = run_rollouts(env, Policy(), controller, benchmark, seed=7, policy_output="mean", progress=Telemetry())
    assert len(records) == 4
    assert sum(record["terminated"] for record in records) == 2
    assert monitor.episode_callback is None
    assert updates[-1]["episodes_completed"] == 4
    assert updates[-1]["batches_completed"] == 2
    assert any(update["batch_episodes_completed"] == 1 for update in updates)
    assert all(update["episodes_completed"] <= update["episodes_total"] for update in updates)


def test_real_actor_loading_preserves_normalizer_and_isolates_sampling_rng(tmp_path):
    obs = TensorDict({"policy": torch.randn(3, 9)}, batch_size=[3])
    agent = PPORunnerCfg()
    agent.device = "cpu"
    agent.actor.hidden_dims = [12]
    agent.actor.obs_normalization = True
    agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
    actor_class, actor_cfg = resolve_class(to_dict(agent)["actor"])
    original = actor_class(obs, {"actor": ["policy"]}, "actor", 2, **actor_cfg)
    original.update_normalization(obs)
    task_cfg = LowerLimbAgentEnvCfg()
    raw = SimpleNamespace(cfg=task_cfg, step_dt=1 / 60)
    env = SimpleNamespace(unwrapped=raw, num_actions=2, get_observations=lambda: obs)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "actor_state_dict": original.state_dict(),
            "iter": 5500,
            "infos": {
                "g1_curriculum": {
                    "schema_version": 1,
                    "family": "g1.lower_limb_agent",
                    "phase": 3,
                    "control_dt": 1 / 60,
                    "compatibility": _compatibility(raw),
                }
            },
        },
        path,
    )
    checkpoint = {"path": str(path), "sha256": jobs.digest_file(path)}
    policy, _, metadata = load_policy(env, agent, checkpoint, device="cpu")
    assert metadata["training_phase"] == 3  # Cross-phase fixed benchmarks use compatible actors.
    assert not policy.training
    assert not any(parameter.requires_grad for parameter in policy.parameters())
    original.eval()
    before = copy.deepcopy(policy.state_dict())
    random_state = torch.random.get_rng_state()
    with torch.inference_mode():
        torch.testing.assert_close(policy(obs), original(obs))
        first = policy_action(policy, obs, stochastic=True, seed=123)
        second = policy_action(policy, obs, stochastic=True, seed=123)
    torch.testing.assert_close(first, second)
    assert torch.equal(random_state, torch.random.get_rng_state())
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
    wrong_raw = SimpleNamespace(cfg=task_cfg.copy(), step_dt=1 / 60)
    wrong_raw.cfg.observations.policy.history_length = 99
    env.unwrapped = wrong_raw
    with pytest.raises(ValueError, match="policy_history"):
        load_policy(env, agent, checkpoint, device="cpu")
