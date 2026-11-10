"""Dispatch runner construction to the selected task's explicit implementation."""

from importlib import import_module

from .task_registry import get_task_family


def create_task_runner(env, agent_cfg, log_dir=None, *, task_family=None, snapshot_path=None, evaluation=False):
    actual = getattr(env.unwrapped.cfg, "task_family", None)
    if not isinstance(actual, str) or not actual:
        raise ValueError(f"Invalid environment task_family={actual!r}; declare task_family on its configuration")
    if task_family is None:
        task_family = actual
    family = get_task_family(task_family)
    if actual != task_family:
        raise ValueError(
            f"Selected task family {task_family!r} differs from environment task_family={actual!r}; "
            "declare task_family on the environment configuration and select the matching family"
        )
    if snapshot_path is not None and not family.curriculum_snapshot:
        raise ValueError(f"{task_family} does not support --curriculum-snapshot")
    module, name = family.runner_factory.split(":", 1)
    factory = getattr(import_module(module), name)
    return factory(env, agent_cfg, log_dir, snapshot_path=snapshot_path, evaluation=evaluation)
