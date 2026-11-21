"""Select the original observation layout when evaluating saved Upper policies."""

import torch

from .symmetry import FULL_OBSERVATION_DIM, LEGACY_OBSERVATION_DIM


def configure_checkpoint_observations(env_cfg, checkpoint=None, *, saved=None):
    if not hasattr(env_cfg, "legacy_observations"):
        return
    if saved is None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    control = (saved.get("infos") or {}).get("g1_control_interface") or {}
    width = control.get("policy_observation_dim")
    if control.get("schema_version") == 4 and width in (LEGACY_OBSERVATION_DIM, FULL_OBSERVATION_DIM):
        env_cfg.legacy_observations = width == LEGACY_OBSERVATION_DIM
