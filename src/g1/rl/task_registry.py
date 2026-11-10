"""Explicit task-family integration contracts; importing this module never loads a simulator."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskFamily:
    runner_factory: str
    curriculum_snapshot: bool = False
    requires_play_mode: bool = False


# Every family owns its factory and runner. There is deliberately no generic fallback.
TASK_FAMILIES = {
    "lower_limb_agent": TaskFamily(
        "g1.tasks.lower_limb_agent.runner:create_curriculum_runner",
        curriculum_snapshot=True,
        requires_play_mode=True,
    ),
    "upper_limb_agent": TaskFamily("g1.tasks.upper_limb_agent.runner:create_upper_limb_runner"),
}

LOWER_LIMB_TASK_IDS = frozenset(("Template-G1-v0", "Template-G1-Phase2-v0", "Template-G1-Phase3-v0"))


def get_task_family(name: str) -> TaskFamily:
    try:
        family = TASK_FAMILIES[name]
    except KeyError:
        raise ValueError(f"Unknown task family {name!r}; choose from {', '.join(TASK_FAMILIES)}") from None
    module, separator, factory = family.runner_factory.partition(":")
    if not separator or not factory or not module.startswith(f"g1.tasks.{name}."):
        raise ValueError(f"Task family {name!r} must implement its runner factory inside g1.tasks.{name}")
    return family


def resolve_task_family(task_id: str, expected: str | None = None) -> str:
    """Read the registered configuration's owning task family before physics starts.

    Registration imports package initializers only. Environment and runner modules
    remain lazy; Hydra still owns all actual configuration parsing and overrides.
    """
    import gymnasium as gym

    import g1.tasks  # noqa: F401 - register project environments

    if expected is not None:
        get_task_family(expected)
    try:
        spec = gym.spec(task_id)
    except gym.error.Error as error:
        raise ValueError(
            f"Environment {task_id!r} is not registered; register its environment and agent configuration first."
        ) from error
    entry = spec.kwargs.get("env_cfg_entry_point")
    module = entry.split(":", 1)[0] if isinstance(entry, str) else getattr(entry, "__module__", "")
    if not module.startswith("g1.tasks."):
        raise ValueError(f"Environment {task_id!r} has no recognized project task family in env_cfg_entry_point")
    family = module.removeprefix("g1.tasks.").split(".", 1)[0]
    get_task_family(family)
    if expected is not None and family != expected:
        raise ValueError(f"Environment {task_id!r} belongs to {family}, not --task-family {expected}")
    return family


def require_lower_limb_task(action: str, task_id: str) -> None:
    """Keep the current teleop/eval boundary explicit, including simulator-free planning."""
    if task_id not in LOWER_LIMB_TASK_IDS:
        raise ValueError(f"{action} currently supports only the registered lower-limb Phase 1/2/3 tasks")
