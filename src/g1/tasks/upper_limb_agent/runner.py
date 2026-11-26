"""Standard Upper PPO with workspace-prior and optional frozen-Lower artifact records."""

import copy
import hashlib
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import to_dict

from .symmetry import (
    FULL_OBSERVATION_DIM,
    LEGACY_OBSERVATION_DIM,
    expand_legacy_observation_columns,
    validate_mobile_observation_layout,
)


class UpperLimbOnPolicyRunner(OnPolicyRunner):
    """Keep PPO updates unchanged and archive the configured prior and frozen Lower model."""

    _evaluation = False
    _checkpoint_loaded = False

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        dr = self._domain_randomization()
        if dr is not None and dr.phase > 1 and not self._checkpoint_loaded:
            raise ValueError(
                "Upper domain randomization training requires --checkpoint from the same or preceding phase"
            )
        manager = getattr(self.env.unwrapped, "command_manager", None)
        if manager is not None and "wrist_pose" in manager.active_terms:
            command = manager.get_term("wrist_pose")
            if getattr(command, "planar_enabled", False):
                # Preserve the legacy RNG path when planar tasks are off. A
                # distance-budget episode must start at its real reset time.
                init_at_random_ep_len = False
        return super().learn(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)

    def _domain_randomization(self):
        return getattr(getattr(self.env.unwrapped, "cfg", None), "domain_randomization", None)

    def _range_curriculum(self):
        manager = getattr(self.env.unwrapped, "command_manager", None)
        if manager is not None and "wrist_pose" in manager.active_terms:
            return getattr(manager.get_term("wrist_pose"), "range_curriculum", None)
        return None

    def _frozen_lower(self):
        manager = getattr(self.env.unwrapped, "action_manager", None)
        if manager is None or "joint_pos" not in manager.active_terms:
            return None
        return getattr(manager.get_term("joint_pos"), "frozen_policy", None)

    @staticmethod
    def mobile_control_interface(raw):
        manager = getattr(raw, "action_manager", None)
        if manager is None or "lower_command" not in manager.active_terms:
            return None
        command = manager.get_term("lower_command")
        height = manager.get_term("height_offset")
        pelvis = manager.get_term("pelvis_offset")
        joint_cfg = manager.get_term("joint_pos").cfg
        observation_dim = raw.observation_manager.group_obs_dim["policy"][0]
        if observation_dim == FULL_OBSERVATION_DIM:
            validate_mobile_observation_layout(raw)
        control = {
            "schema_version": 4,
            "action_mapping": "episode_range",
            "range_command_names": [
                command.cfg.range_command_name,
                height.cfg.range_command_name,
                pelvis.cfg.range_command_name,
            ],
            "action_terms": list(manager.active_terms),
            "action_dimensions": list(manager.action_term_dim),
            "policy_observation_dim": observation_dim,
            "upper_joint_names": list(manager.get_term("joint_pos")._joint_names),
            "command_ranges": command.cfg.command_ranges,
            "velocity_reward_reference": command.cfg.reward_reference_scale,
            "velocity_magnitude_normalization": "episode_range",
            "commands_enabled": command.cfg.enabled,
            "height_offset_range": list(height.cfg.height_offset_range),
            "height_enabled": height.cfg.enabled,
            "height_reward_reference": height.cfg.reward_reference_range,
            "pelvis_ranges": pelvis.cfg.offset_ranges,
            "pelvis_reward_reference": pelvis.cfg.reward_reference_scale,
            "pelvis_target_speed": pelvis.cfg.target_speed,
            "pelvis_enabled": pelvis.cfg.enabled,
            "use_training_envelope": joint_cfg.use_training_envelope,
            "standing_thresholds": (joint_cfg.standing_linear_threshold, joint_cfg.standing_yaw_threshold),
            "control_dt": raw.step_dt,
        }
        return json.loads(json.dumps(control))

    def save(self, path, infos=None):
        curriculum = self._range_curriculum()
        if curriculum is not None:
            infos = copy.deepcopy(infos) if infos is not None else {}
            infos["g1_upper_ranges"] = curriculum.state_dict(self.env.unwrapped.common_step_counter)
        control = self.mobile_control_interface(self.env.unwrapped)
        if control is not None:
            infos = copy.deepcopy(infos) if infos is not None else {}
            infos["g1_control_interface"] = control
        manager = getattr(self.env.unwrapped, "command_manager", None)
        if manager is not None and "wrist_pose" in manager.active_terms:
            command = manager.get_term("wrist_pose")
            dr = self._domain_randomization()
            if dr is not None:
                infos = copy.deepcopy(infos) if infos is not None else {}
                infos["g1_upper_domain_randomization"] = {
                    "settings": dr.settings(),
                    "seed": self.env.unwrapped.cfg.seed,
                    "planar_turn_focus": asdict(command.cfg.planar_turn_focus),
                }
            quality = getattr(command, "workspace_quality_map", None)
            if quality is not None:
                artifact = Path(path).parent / "artifacts" / "workspace_quality.npz"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                if artifact.exists():
                    if hashlib.sha256(artifact.read_bytes()).hexdigest() != quality.sha256:
                        raise ValueError("Archived workspace Q map differs from the active map")
                else:
                    artifact.write_bytes(quality.archive_bytes)
                    artifact.with_suffix(".json").write_text(
                        json.dumps(quality.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                infos = copy.deepcopy(infos) if infos is not None else {}
                infos["g1_workspace_quality"] = {
                    "map_sha256": quality.sha256,
                    "source_metrics_sha256": quality.metadata["source_metrics"]["sha256"],
                    "q_reference": quality.q_reference,
                    "payload_mass_kg": 0.0,
                    "coordinate_frame": "pelvis",
                    "right_mirror_axis": "y",
                    "position_tolerance_m": command.cfg.workspace_quality.position_tolerance_m,
                    "artifact": "artifacts/workspace_quality.npz",
                }
        lower = self._frozen_lower()
        if lower is not None:
            artifact = Path(path).parent / "artifacts" / "frozen_lower_model.pt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if not artifact.exists():
                shutil.copy2(lower.path, artifact)
                config_dir = artifact.parent / "params"
                config_dir.mkdir(exist_ok=True)
                shutil.copy2(lower.path.parent / "params/env.yaml", config_dir / "env.yaml")
                actor_config = lower.path.parent / "params/agent.yaml"
                if actor_config.is_file():
                    shutil.copy2(actor_config, config_dir / "agent.yaml")
            with artifact.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != lower.metadata["sha256"]:
                raise ValueError("Archived frozen Lower checkpoint differs from the active actor")
            infos = copy.deepcopy(infos) if infos is not None else {}
            infos["g1_frozen_lower"] = copy.deepcopy(lower.metadata)
            infos["g1_frozen_lower"]["artifact"] = "artifacts/frozen_lower_model.pt"
        super().save(path, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, weights_only=False, map_location="cpu")
        dr = self._domain_randomization()
        new_distribution = False
        if dr is not None and not self._evaluation:
            metadata = (saved.get("infos") or {}).get("g1_upper_domain_randomization") or {}
            source = metadata.get("settings", {"phase": 1})
            if source["phase"] not in (dr.phase, dr.phase - 1):
                raise ValueError(
                    "Resume Upper training from a checkpoint of the same or immediately preceding DR phase"
                )
            new_distribution = json.dumps(source, sort_keys=True) != json.dumps(dr.settings(), sort_keys=True)
        control = self.mobile_control_interface(self.env.unwrapped)
        saved_control = (saved.get("infos") or {}).get("g1_control_interface")
        migrate_observations = False
        if control is not None:
            # Enabled flags permit playback ablations; layout, mapping and cost scales are structural.
            # Magnitude normalization is recorded for audits, allowing reward changes on fine-tuning.
            keys = (
                "schema_version",
                "action_mapping",
                "range_command_names",
                "action_terms",
                "action_dimensions",
                "upper_joint_names",
                "control_dt",
                "command_ranges",
                "velocity_reward_reference",
                "height_offset_range",
                "height_reward_reference",
                "pelvis_ranges",
                "pelvis_reward_reference",
                "pelvis_target_speed",
                "use_training_envelope",
                "standing_thresholds",
            )
            actual_width = control["policy_observation_dim"]
            saved_width = (saved_control or {}).get("policy_observation_dim")
            migrate_observations = saved_width == LEGACY_OBSERVATION_DIM and actual_width == FULL_OBSERVATION_DIM
            if (
                saved_control is None
                or any(saved_control.get(key) != control[key] for key in keys)
                or (saved_width != actual_width and not migrate_observations)
            ):
                raise ValueError(
                    "Mobile Upper control interface mismatch: expected 23 actions "
                    f"[lower_command:3, height_offset:1, pelvis_offset:2, joint_pos:17], {actual_width} observations "
                    "with episode-range action mapping and matching physical/reward scales. "
                    "A1 schema-3 fixed-maximum mappings and older 21/107 or 20/106 checkpoints require retraining."
                )
            if migrate_observations:
                if load_cfg is None or load_cfg.get("optimizer"):
                    raise ValueError(
                        "Upper observation migration requires a fresh optimizer. "
                        "Set env.legacy_observations=true to resume the original optimizer and observation layout."
                    )
                for model in ("actor", "critic"):
                    if load_cfg.get(model):
                        weights = saved[f"{model}_state_dict"]
                        weights["mlp.0.weight"] = expand_legacy_observation_columns(weights["mlp.0.weight"])
                        for stat, fill in (("_mean", 0), ("_var", 1), ("_std", 1)):
                            key = f"obs_normalizer.{stat}"
                            if key in weights:
                                weights[key] = expand_legacy_observation_columns(weights[key], fill)
        elif saved_control is not None:
            raise ValueError("This mobile Upper checkpoint requires the velocity-command environment")
        expected = (saved.get("infos") or {}).get("g1_frozen_lower")
        lower = self._frozen_lower()
        if lower is not None:
            if expected is None:
                raise ValueError("This mobile Upper checkpoint must record its frozen Lower model")
            if expected["sha256"] != lower.metadata["sha256"] or expected["interface"] != lower.interface:
                raise ValueError("Frozen Lower model or interface differs from the Upper checkpoint")
        elif expected is not None:
            raise ValueError("A mobile Upper checkpoint requires the frozen Lower environment")
        curriculum = self._range_curriculum()
        range_state = (saved.get("infos") or {}).get("g1_upper_ranges")
        if (
            dr is not None
            and dr.phase > 1
            and not self._evaluation
            and (curriculum is None or curriculum.cfg.reset_on_load)
        ):
            raise ValueError("Upper domain randomization must resume its saved task curriculum (reset_on_load=false)")
        if curriculum is not None and not curriculum.cfg.reset_on_load:
            if range_state is None:
                raise ValueError("Upper range curriculum state is missing; use range_curriculum.reset_on_load=true")
            # Validate and restore before any simulation reset; PPO owns all optimizer restoration.
            curriculum.load_state_dict(range_state, self.env.unwrapped.common_step_counter)
            if new_distribution:
                curriculum.reset_statistics(self.env.unwrapped.common_step_counter)
                logging.getLogger(__name__).info(
                    "Upper DR phase %s started a fresh evaluation window at level %s, Planar probability %.2f",
                    dr.phase,
                    curriculum.level,
                    curriculum.planar_probability,
                )
        elif curriculum is not None:
            curriculum.restart(self.env.unwrapped.common_step_counter)
        elif range_state is not None and curriculum is None:
            raise ValueError("Checkpoint requires an Upper range curriculum configuration")
        if migrate_observations:
            if self.alg.load(saved, load_cfg, strict):
                self.current_learning_iteration = saved["iter"]
            result = saved["infos"]
        else:
            result = super().load(path, load_cfg, strict, map_location)
        if curriculum is not None:
            raw = self.env.unwrapped
            command = raw.command_manager.get_term("wrist_pose")
            command._restoring_ranges = True
            try:
                raw.reset()  # Fresh physical episodes; checkpoint stores curriculum, not simulator state.
            finally:
                command._restoring_ranges = False
        self._checkpoint_loaded = True
        return result


def create_upper_limb_runner(env, agent_cfg, log_dir=None, *, snapshot_path=None, evaluation=False):
    """Implement the shared train/play factory signature for this task only."""
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError("Upper-limb training currently supports OnPolicyRunner only")
    if snapshot_path is not None:
        raise ValueError("upper_limb_agent does not support --curriculum-snapshot")
    cfg = to_dict(agent_cfg)
    if cfg.pop("mirror_augmentation", False) and not evaluation:
        from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

        from .symmetry import augment_mobile_batch, validate_mobile_mirror_interface

        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("Choose either agent.mirror_augmentation or an explicit PPO symmetry_cfg")
        validate_mobile_mirror_interface(env)
        cfg["algorithm"]["symmetry_cfg"] = to_dict(
            RslRlSymmetryCfg(
                use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=augment_mobile_batch
            )
        )
    runner = UpperLimbOnPolicyRunner(env, cfg, log_dir=log_dir, device=agent_cfg.device)
    runner._evaluation = evaluation
    curriculum = runner._range_curriculum()
    if curriculum is not None:
        curriculum.evaluation = evaluation
    return runner
