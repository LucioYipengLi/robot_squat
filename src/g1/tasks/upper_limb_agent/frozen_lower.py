"""Load only a frozen Lower actor; keep its optimizer and curriculum outside Upper training."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import torch
import yaml
from rsl_rl.utils import resolve_class
from tensordict import TensorDict

DEFAULT_LOWER_CHECKPOINT = "logs/rsl_rl/g1_flat/2026-10-21_22-59-54/model_9999.pt"


def lower_actor_config() -> dict:
    """Use the same public RSL-RL config migration as the train/play entrypoints."""
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

    from g1.tasks.lower_limb_agent.config.g1.agents.rsl_rl_ppo_cfg import PPORunnerCfg

    cfg = handle_deprecated_rsl_rl_cfg(PPORunnerCfg(), check_rsl_rl_version())
    return to_dict(cfg.actor)


def _yaml_value(path: Path, *keys):
    """Read a YAML node without constructing IsaacLab's Python-tagged objects."""
    node = yaml.compose(path.read_text(encoding="utf-8"))
    for key in keys:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError(f"Missing {'.'.join(keys)} in {path}")
        node = next((value for name, value in node.value if name.value == key), None)
    return node


class FrozenLowerPolicy:
    """Deterministic batched actor with an explicit observation/action contract.

    Build from the latest Lower actor configuration and load the specified
    checkpoint strictly. The exported inference module includes normalization;
    it is frozen, stays in eval mode, and is never registered in Upper's PPO.
    """

    def __init__(self, checkpoint_path: str, actor_cfg: dict, interface: dict, device: str):
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Frozen Lower checkpoint not found: {self.path}")
        saved = torch.load(self.path, map_location="cpu", weights_only=False)
        if "actor_state_dict" not in saved:
            raise ValueError("Expected a Lower model_*.pt checkpoint containing actor_state_dict")
        self._validate_contract(saved, interface)
        cfg = copy.deepcopy(actor_cfg)
        cfg.pop("stochastic", None)  # Legacy config-only flag; inference is always deterministic.
        if cfg.get("class_name") != "MLPModel":
            raise ValueError("Frozen Lower currently supports non-recurrent MLPModel actors")
        agent_path = self.path.parent / "params/agent.yaml"
        if agent_path.is_file():
            actor_fields = {key.value: value for key, value in _yaml_value(agent_path, "actor").value}
            for key in ("class_name", "hidden_dims", "activation", "obs_normalization"):
                if yaml.safe_load(yaml.serialize(actor_fields[key])) != cfg[key]:
                    raise ValueError(f"Frozen Lower actor {key} differs from its saved configuration")
        actor_class, kwargs = resolve_class(cfg)
        # Loading a frozen actor must not consume the Upper policy's initialization RNG stream.
        with torch.random.fork_rng(devices=[]):
            dummy = TensorDict({"policy": torch.zeros(1, interface["input_dim"])}, batch_size=[1])
            actor = actor_class(dummy, {"actor": ["policy"]}, "actor", interface["action_dim"], **kwargs)
            actor.load_state_dict(saved["actor_state_dict"], strict=True)
            self.model = actor.as_jit().to(device).eval().requires_grad_(False)
        self.interface = copy.deepcopy(interface)
        self.interface["actor_config"] = cfg
        with self.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.metadata = {
            "checkpoint": str(self.path),
            "sha256": digest,
            "iteration": saved.get("iter"),
            "interface": self.interface,
        }

    def _validate_contract(self, saved: dict, interface: dict) -> None:
        state = (saved.get("infos") or {}).get("g1_curriculum")
        if state is None or state.get("family") != "g1.lower_limb_agent":
            raise ValueError("Frozen Lower requires a checkpoint with G1 Lower interface metadata")
        compatibility = state["compatibility"]
        for key, actual in (
            ("policy_history", interface["history_length"]),
            ("action_scale", interface["action_scale"]),
        ):
            if compatibility[key] != actual:
                raise ValueError(f"Frozen Lower {key} differs from the latest Lower configuration")
        if not math.isclose(state["control_dt"], interface["control_dt"], rel_tol=0, abs_tol=1e-10):
            raise ValueError("Frozen Lower control period differs from the latest Lower configuration")
        if compatibility["action_joints"] != interface["action_joint_patterns"]:
            raise ValueError("Frozen Lower action joint selection differs from the latest Lower configuration")
        if state.get("upper_body_target_source") != "final_goal":
            raise ValueError("Frozen Lower must have been trained with final_goal upper-body previews")
        config_path = self.path.parent / "params/env.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Frozen Lower observation contract requires {config_path}")
        policy = _yaml_value(config_path, "observations", "policy")
        fields = {key.value: value for key, value in policy.value}
        saved_names = [
            key
            for key, value in fields.items()
            if isinstance(value, yaml.MappingNode) and any(name.value == "func" for name, _ in value.value)
        ]
        if saved_names != interface["observation_terms"]:
            raise ValueError("Frozen Lower observation term order differs from the latest Lower configuration")
        order = fields.get("history_order")
        if (order.value if order is not None else "term") != interface["history_order"]:
            raise ValueError("Frozen Lower history order differs from the latest Lower configuration")
        action_fields = {key.value: value for key, value in _yaml_value(config_path, "actions", "joint_pos").value}
        for key in ("preserve_order", "use_default_offset", "offset", "clip"):
            if key in interface and yaml.safe_load(yaml.serialize(action_fields[key])) != interface[key]:
                raise ValueError(f"Frozen Lower action {key} differs from its saved configuration")

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.interface["input_dim"]:
            raise ValueError(f"Frozen Lower expects [N,{self.interface['input_dim']}] observations")
        with torch.no_grad():
            return self.model(observations)
