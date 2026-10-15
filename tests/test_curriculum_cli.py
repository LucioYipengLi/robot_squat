"""Verify the project entrypoint delegates arguments and always restores upstream globals."""

import sys
from types import SimpleNamespace

import pytest
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.app.logging_utils
import isaaclab.app.sim_launcher

from g1.rl import __main__ as cli
from g1.rl.runner import CurriculumOnPolicyRunner


@pytest.mark.parametrize("action", ["train", "play"])
@pytest.mark.parametrize("fail", [False, True])
def test_backend_adapter_is_scoped_and_forwards_curriculum_options(monkeypatch, action, fail):
    original_argv = sys.argv
    expected = ["--task", "Template-G1-v0", "--checkpoint=old.pt", "physics=isaacsim_physx"]
    backend = SimpleNamespace(OnPolicyRunner=OnPolicyRunner, DistillationRunner=DistillationRunner)

    def run(arguments):
        assert arguments == expected
        assert sys.argv[1:] == expected
        factory = backend.OnPolicyRunner
        assert factory.func is CurriculumOnPolicyRunner
        assert factory.keywords == {"snapshot_path": "legacy.json", "evaluation": action == "play"}
        with pytest.raises(ValueError, match="OnPolicyRunner only"):
            backend.DistillationRunner()
        if fail:
            raise RuntimeError("backend failed")

    backend.run = run

    def import_backend(name):
        assert name == f"isaaclab_rl.entrypoints.backends.{action}_rsl_rl"
        return backend

    monkeypatch.setattr(cli.importlib, "import_module", import_backend)
    monkeypatch.setattr(isaaclab.app.logging_utils, "configure_console_logging", lambda: None)
    monkeypatch.setattr(isaaclab.app.sim_launcher, "fuse_kit_args", lambda args: args)
    monkeypatch.setenv("WORLD_SIZE", "1")
    arguments = [action, "--curriculum-snapshot", "legacy.json", *expected]
    if fail:
        with pytest.raises(RuntimeError, match="backend failed"):
            cli.main(arguments)
    else:
        assert cli.main(arguments) == 0
    assert sys.argv is original_argv
    assert backend.OnPolicyRunner is OnPolicyRunner
    assert backend.DistillationRunner is DistillationRunner


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
