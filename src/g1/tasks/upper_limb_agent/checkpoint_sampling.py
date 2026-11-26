"""Restore Upper sampling before managers allocate episode and switch buffers."""

from pathlib import Path

import torch
import yaml


def configure_dr_checkpoint_sampling(env_cfg, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    infos = saved.get("infos") or {}
    course = infos.get("g1_upper_ranges")
    if course is None:
        raise ValueError("Upper domain randomization requires a checkpoint with its task curriculum")

    contract = course["contract"]
    target = contract["target"]
    wrist = env_cfg.commands.wrist_pose
    wrist.range_curriculum.from_dict(contract["cfg"])
    saved_sampling = course.get("planar_sampling")
    if saved_sampling is None:
        wrist.range_curriculum.planar_sampling.enabled = False
    else:
        wrist.range_curriculum.planar_sampling.from_dict(saved_sampling["cfg"])

    for name in ("left_offsets", "right_offsets", "dual_targets"):
        getattr(wrist, name).from_dict(target[name])
    wrist.mode_probabilities = tuple(target["mode_probabilities"])
    wrist.resampling_time_range = tuple(target["resampling_time_range"])
    wrist.distance_reference_speed = target["distance_reference_speed"]
    wrist.height_task.probability = target["class_probability"]
    wrist.height_task.shift_range = tuple(contract["height_shift"])
    wrist.height_task.hold_episode = target["hold_episode"]
    wrist.regular_retract_sampling = type(wrist.regular_retract_sampling)()
    wrist.regular_retract_sampling.from_dict(target.get("regular_retract_sampling", {}))
    wrist.planar_task.transition_sampling = type(wrist.planar_task.transition_sampling)()
    if "planar_task" in target:
        wrist.planar_task.from_dict(target["planar_task"])
    else:
        wrist.planar_task.probability = 0.0

    focus = (infos.get("g1_upper_domain_randomization") or {}).get("planar_turn_focus")
    if focus is None:
        env_path = Path(checkpoint).parent / "params/env.yaml"
        if env_path.is_file():
            # Other parts of archived configs contain Python-specific tags, e.g. slices.
            node = yaml.compose(env_path.read_text(encoding="utf-8"))
            for name in ("commands", "wrist_pose", "planar_turn_focus"):
                node = next((value for key, value in node.value if key.value == name), None)
                if node is None:
                    break
            focus = yaml.safe_load(yaml.serialize(node)) if node is not None else {"enabled": False}
    if focus is not None:
        wrist.planar_turn_focus.from_dict(focus)
