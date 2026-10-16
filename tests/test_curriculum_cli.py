"""Verify the project entrypoint delegates arguments and always restores upstream globals."""

import logging
import sys
from types import SimpleNamespace

import pytest

import isaaclab.app.sim_launcher
import isaaclab.utils

from isaaclab_rl.rsl_rl import create_rsl_rl_runner

from g1.rl import __main__ as cli
from g1.rl import runner as project_runner
from g1.rl.runner import CurriculumOnPolicyRunner, create_curriculum_runner
from g1.tasks.lower_limb_agent.config.g1.agents.rsl_rl_ppo_cfg import PPORunnerCfg


@pytest.mark.parametrize("action", ["train", "play"])
@pytest.mark.parametrize("fail", [False, True])
def test_backend_adapter_is_scoped_and_forwards_curriculum_options(monkeypatch, action, fail):
    original_argv = sys.argv
    expected = ["--task", "Template-G1-v0", "--checkpoint=old.pt", "physics=isaacsim_physx"]
    backend = SimpleNamespace(create_rsl_rl_runner=create_rsl_rl_runner)

    def run(arguments):
        assert arguments == expected
        assert sys.argv[1:] == expected
        factory = backend.create_rsl_rl_runner
        assert factory.func is create_curriculum_runner
        assert factory.keywords == {"snapshot_path": "legacy.json", "evaluation": action == "play"}
        with pytest.raises(ValueError, match="OnPolicyRunner only"):
            factory(None, SimpleNamespace(class_name="DistillationRunner"))
        if fail:
            raise RuntimeError("backend failed")

    backend.run = run

    def import_backend(name):
        assert name == f"isaaclab_rl.entrypoints.backends.{action}_rsl_rl"
        return backend

    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=import_backend))
    monkeypatch.setattr(isaaclab.utils, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(isaaclab.app.sim_launcher, "fuse_kit_args", lambda args: args)
    monkeypatch.setenv("WORLD_SIZE", "1")
    arguments = [action, "--curriculum-snapshot", "legacy.json", *expected]
    if fail:
        with pytest.raises(RuntimeError, match="backend failed"):
            cli.main(arguments)
    else:
        assert cli.main(arguments) == 0
    assert sys.argv is original_argv
    assert backend.create_rsl_rl_runner is create_rsl_rl_runner


@pytest.mark.parametrize("action", ["train", "play"])
def test_adapter_matches_installed_backend_and_restores_factory(action):
    backend = cli.importlib.import_module(f"isaaclab_rl.entrypoints.backends.{action}_rsl_rl")
    factory = object()
    with cli._runner_adapter(backend, factory):
        assert backend.create_rsl_rl_runner is factory
    assert backend.create_rsl_rl_runner is create_rsl_rl_runner


def test_adapter_rejects_an_unknown_factory_without_replacing_it():
    factory = object()
    backend = SimpleNamespace(create_rsl_rl_runner=factory)
    with pytest.raises(RuntimeError, match="constructor seam"), cli._runner_adapter(backend, object()):
        pytest.fail("Unexpected adapter entry")
    assert backend.create_rsl_rl_runner is factory


@pytest.mark.parametrize("evaluation", [False, True])
def test_curriculum_factory_converts_current_agent_config(monkeypatch, evaluation):
    agent = PPORunnerCfg()
    agent.device = "cpu"
    env = object()
    calls = []
    result = object()

    def construct(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(project_runner, "CurriculumOnPolicyRunner", construct)
    assert (
        create_curriculum_runner(env, agent, "logs/test", snapshot_path="legacy.json", evaluation=evaluation) is result
    )
    (actual_env, config), options = calls[0]
    assert actual_env is env
    assert isinstance(config, dict)
    assert config["algorithm"]["class_name"] == agent.algorithm.class_name
    assert options == {
        "log_dir": "logs/test",
        "device": "cpu",
        "snapshot_path": "legacy.json",
        "evaluation": evaluation,
    }


@pytest.mark.parametrize("runner_class", ["DistillationRunner", "UnknownRunner"])
def test_curriculum_factory_rejects_unsupported_runners(runner_class):
    with pytest.raises(ValueError, match="OnPolicyRunner only"):
        create_curriculum_runner(None, SimpleNamespace(class_name=runner_class))


@pytest.mark.parametrize("action", ["train", "play", "teleop"])
def test_real_entrypoint_help_initializes_console_logging_without_launch(monkeypatch, capsys, action):
    # Isolate pytest's handlers while exercising the actual installed logging API.
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.WARNING)
    monkeypatch.setenv("WORLD_SIZE", "1")
    original_argv = sys.argv
    try:
        for _ in range(2):
            with pytest.raises(SystemExit) as error:
                cli.main([action, "--help"])
            assert error.value.code == 0
            assert sys.argv is original_argv
            assert len(root.handlers) == 1
            assert not isinstance(root.handlers[0], logging.FileHandler)
        capsys.readouterr()
        logging.getLogger("g1").warning("entrypoint logging works")
        assert capsys.readouterr().out.count("entrypoint logging works") == 1
    finally:
        for handler in root.handlers:
            handler.close()
    if action != "teleop":
        backend = cli.importlib.import_module(f"isaaclab_rl.entrypoints.backends.{action}_rsl_rl")
        assert backend.create_rsl_rl_runner is create_rsl_rl_runner


@pytest.mark.parametrize(
    "arguments,world_size",
    [
        (["train", "--distributed"], "1"),
        (["train"], "2"),
        (["train", "--curriculum-snapshot", "old.json"], "1"),
        (["play", "--train_env_cfg"], "1"),
        (["train", "--frontend", "warp"], "1"),
    ],
)
def test_unsupported_invocations_fail_before_launch(monkeypatch, arguments, world_size):
    monkeypatch.setenv("WORLD_SIZE", world_size)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


def test_runner_constructor_also_rejects_distributed_use(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single-process"):
        CurriculumOnPolicyRunner()
