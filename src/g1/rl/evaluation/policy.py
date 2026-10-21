"""Actor-only checkpoint loading; evaluation never restores a training curriculum."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path

from .jobs import digest_file


def verify_checkpoint_files(checkpoint: dict) -> None:
    for key, hash_key in (
        ("path", "sha256"),
        ("agent_config", "agent_config_sha256"),
        ("env_config", "env_config_sha256"),
    ):
        if checkpoint.get(key) and digest_file(checkpoint[key]) != checkpoint[hash_key]:
            raise ValueError(f"Checkpoint input changed: {checkpoint[key]}")


def _plain(value):
    return json.loads(json.dumps(value))


def _load_saved_env_config(filename: str) -> dict:
    """Read portable IsaacLab YAML, admitting only inert tuple and slice tags.

    IsaacLab's YAML dumper writes SceneEntityCfg selector slices with a Python
    tag. Register that specific data constructor locally instead of enabling a
    general Python-object loader or modifying global YAML loader behavior.
    """
    import yaml

    class EnvConfigLoader(yaml.SafeLoader):
        pass

    def construct_slice(loader, node):
        values = loader.construct_sequence(node)
        if len(values) != 3 or any(value is not None and type(value) is not int for value in values):
            raise ValueError("Invalid slice selector in saved environment configuration")
        return slice(*values)

    EnvConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
    )
    EnvConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", construct_slice)
    try:
        with Path(filename).open(encoding="utf-8-sig") as stream:
            saved = yaml.load(stream, Loader=EnvConfigLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Unsupported saved environment YAML: {filename}: {error}") from error
    if not isinstance(saved, dict):
        raise ValueError("Saved environment configuration must be a mapping")
    return saved


def _portable(value):
    """Normalize serialized selectors/sequences without losing ordered term lists."""
    if isinstance(value, Mapping):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Runtime config serialization can retain tensor-valued per-axis scales.
    if hasattr(value, "tolist"):
        return _portable(value.tolist())
    raise ValueError(f"Unsupported policy-layout configuration value: {type(value).__name__}")


def _upper_body_target_source(record: dict) -> str:
    """Read explicit semantics, inferring only the two historical implementations."""
    params = record.get("observations", {}).get("policy", {}).get("upper_body_target", {}).get("params", {})
    source = record.get("upper_body_target_source", params.get("target_source"))
    if source is None:
        # Before the explicit marker, quintic observed q_ref; step targets were
        # already endpoints. Pre-motion records have no upper_body_motion field.
        source = "reference" if record.get("upper_body_motion", {}).get("profile") == "quintic_v1" else "final_goal"
    if source not in ("final_goal", "reference"):
        raise ValueError(f"Unsupported checkpoint upper-body target source: {source}")
    return source


def _observation_layout(group: dict, *, upper_body_target_source="final_goal") -> dict:
    """Describe active term order and effective processing, excluding benchmark noise."""
    if not isinstance(group, dict):
        raise ValueError("Policy observation configuration must be a mapping")
    settings, terms = {}, []
    for name, value in group.items():
        if name == "enable_corruption":
            continue
        if isinstance(value, dict) and "func" in value:
            term = {key: item for key, item in value.items() if key != "noise"}
            if (
                name == "upper_body_target"
                and term["func"] == "g1.tasks.lower_limb_agent.mdp.observations:upper_body_joint_pos_target_rel"
            ):
                term["params"] = {"target_source": upper_body_target_source, **term.get("params", {})}
            # ObservationManager applies group-level history overrides to every term.
            if group.get("history_length") is not None:
                term["history_length"] = group["history_length"]
                term["flatten_history_dim"] = group["flatten_history_dim"]
            terms.append({"name": name, "definition": _portable(term)})
        elif value is not None:
            settings[name] = _portable(value)
    return {"settings": settings, "terms": terms}


def _action_layout(actions: dict) -> list[dict]:
    if not isinstance(actions, dict):
        raise ValueError("Action configuration must be a mapping")
    result = [
        {"name": name, "definition": _portable({key: item for key, item in value.items() if key != "debug_vis"})}
        for name, value in actions.items()
        if value is not None
    ]
    # This extension only adds external upper-body execution. Its inherited
    # lower-body action mapping is identical; every other field stays strict.
    equivalent = {
        "g1.tasks.lower_limb_agent.mdp.actions:LowerBodyPositionAction",
        "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction",
        "{DIR}.joint_actions:JointPositionAction",
    }
    for term in result:
        if term["definition"].get("class_type") in equivalent:
            term["definition"]["class_type"] = "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction"
    return result


def _first_difference(expected, actual, path: str) -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return path + " fields"
        for key in expected:
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return path + " length"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if expected == actual else path


def _validate_saved_policy_semantics(env_cfg, saved: dict) -> None:
    """Check equal-sized observation/action layouts using official callable serialization."""
    from isaaclab.utils import to_dict

    try:
        expected_obs = _observation_layout(
            saved["observations"]["policy"], upper_body_target_source=_upper_body_target_source(saved)
        )
        actual_obs = _observation_layout(to_dict(env_cfg.observations.policy))
        expected_actions = _action_layout(saved["actions"])
        actual_actions = _action_layout(to_dict(env_cfg.actions))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Incomplete saved policy observation/action configuration: {error}") from error
    difference = _first_difference(expected_obs, actual_obs, "observations.policy")
    if difference:
        raise ValueError(f"Checkpoint observation layout differs from evaluation: {difference}")
    difference = _first_difference(expected_actions, actual_actions, "actions")
    if difference:
        raise ValueError(f"Checkpoint action layout differs from evaluation: {difference}")


def validate_policy_compatibility(env, checkpoint: dict, infos: dict | None, *, allow_unverified=False) -> str:
    """Validate task semantics separately from training phase, horizon and curriculum."""
    from g1.tasks.lower_limb_agent.curriculum_state import STATE_KEY, _compatibility

    actual = _plain(_compatibility(env))
    saved = _load_saved_env_config(checkpoint["env_config"]) if checkpoint.get("env_config") else None
    state = infos.get(STATE_KEY) if isinstance(infos, dict) else None
    if state is not None:
        if state.get("family") != "g1.lower_limb_agent" or state.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint task metadata")
        expected = _plain(state["compatibility"])
        expected_source = _upper_body_target_source(state)
        actual_source = env.cfg.observations.policy.upper_body_target.params.get("target_source", "final_goal")
        if expected_source != actual_source:
            raise ValueError(
                f"Checkpoint upper-body target source differs from evaluation: {expected_source} vs {actual_source}"
            )
        dt = state["control_dt"]
        source = "checkpoint_metadata"
    elif saved is not None:
        event = saved["events"]["upper_body_disturbance"]
        groups = event["params"]["disturb_groups"]
        expected = {
            "robot_usd": saved["scene"]["robot"]["spawn"]["usd_path"],
            "action_joints": saved["actions"]["joint_pos"]["joint_names"],
            "action_scale": saved["actions"]["joint_pos"]["scale"],
            "policy_history": saved["observations"]["policy"]["history_length"],
            "disturbance_groups": [
                {"asset": group["asset_cfg"]["name"], "joints": group["asset_cfg"]["joint_names"]} for group in groups
            ],
        }
        expected = _plain(expected)
        dt = saved["sim"]["dt"] * saved["decimation"]
        source = "saved_env_config"
    elif allow_unverified:
        return "unverified_task_semantics_strict_actor_shapes"
    else:
        raise ValueError(
            "Checkpoint has no task metadata or adjacent params/env.yaml. Provide the original run metadata, "
            "or explicitly use --allow-unverified-checkpoint to accept task semantics from the suite."
        )
    differing = [key for key, value in actual.items() if expected.get(key) != value]
    if differing:
        raise ValueError(f"Checkpoint task layout differs from evaluation: {', '.join(differing)}")
    if not math.isclose(dt, env.step_dt, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Checkpoint control dt differs from evaluation")
    if saved is not None:
        _validate_saved_policy_semantics(env.cfg, saved)
        if state is not None:
            source = "checkpoint_metadata+saved_env_config"
    return source


def load_policy(env, agent_cfg, checkpoint: dict, *, device: str, allow_unverified=False):
    """Build only the actor, including its normalizer and output distribution."""
    import torch
    from rsl_rl.utils import resolve_class, resolve_obs_groups

    from isaaclab.utils import to_dict
    from isaaclab.utils.io import load_yaml

    from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

    verify_checkpoint_files(checkpoint)
    agent_cfg = copy.deepcopy(agent_cfg)
    if checkpoint.get("agent_config"):
        agent_cfg.from_dict(load_yaml(checkpoint["agent_config"]))
    agent_cfg.device = device
    # The upstream migration deletes legacy attributes and is not idempotent on
    # already-normalized configs. An absent flag means a modern configuration.
    for model_name in ("actor", "critic"):
        model = getattr(agent_cfg, model_name)
        if not hasattr(model, "stochastic"):
            model.stochastic = False
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    cfg = to_dict(agent_cfg)
    if cfg["class_name"] != "OnPolicyRunner":
        raise ValueError("Evaluation currently supports RSL-RL OnPolicyRunner actors only")
    saved = torch.load(checkpoint["path"], weights_only=False, map_location=device)
    compatibility = validate_policy_compatibility(
        env.unwrapped, checkpoint, saved.get("infos"), allow_unverified=allow_unverified
    )
    if "actor_state_dict" not in saved:
        raise ValueError("Expected the current RSL-RL actor_state_dict checkpoint format")
    observations = env.get_observations().to(device)
    groups = resolve_obs_groups(observations, cfg["obs_groups"], ["actor"])
    actor_class, actor_cfg = resolve_class(cfg["actor"])
    actor = actor_class(observations, groups, "actor", env.num_actions, **actor_cfg).to(device)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    actor.eval()
    actor.requires_grad_(False)
    metadata = {
        "compatibility_source": compatibility,
        "compatibility_limitations": (
            ["Observation processing/order and extended action semantics were not verified without params/env.yaml"]
            if compatibility == "checkpoint_metadata"
            else ["Task and observation/action semantics were not verified"]
            if compatibility.startswith("unverified_")
            else []
        ),
        "training_iteration": saved.get("iter"),
        "training_phase": (saved.get("infos") or {}).get("g1_curriculum", {}).get("phase"),
        "actor_class": str(cfg["actor"]["class_name"]),
        "agent_config_source": checkpoint.get("agent_config") or "registered_task_default",
    }
    return actor, cfg, metadata


def policy_action(policy, observations, *, stochastic: bool, seed: int):
    """Sample in an isolated stream so policy noise never consumes environment randomness."""
    import torch

    if not stochastic:
        return policy(observations, stochastic_output=False)
    device = next(policy.parameters()).device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        return policy(observations, stochastic_output=True)
