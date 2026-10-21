"""Expand evaluation modules into fixed-condition jobs without starting a simulator."""

from __future__ import annotations

from collections.abc import Mapping

from .config import fraction_axis_step, validate_suite


def conditions(config: Mapping) -> list[dict]:
    """Expand validated modules with deterministic IDs and comparable axis values.

    Tracking uses a single scenario at step zero; sweep steps encode the arm range
    fraction in per mille, so 0.555 is logged at TensorBoard step 555.
    """
    config = validate_suite(config)
    result = []
    for module in config["modules"]:
        name = module["name"]
        fractions = [module["arm_fraction"]] if name == "tracking" else module["arm_fractions"]
        waist = module["waist_fraction"]
        for arm in fractions:
            arm_step = fraction_axis_step(arm)
            waist_step = fraction_axis_step(waist)
            result.append(
                {
                    "module": name,
                    "condition_id": f"arm_{arm_step:04d}_waist_{waist_step:04d}",
                    "axis_step": arm_step if name == "upper_body_sweep" else 0,
                    "arm_fraction": arm,
                    "waist_fraction": waist,
                }
            )
    return result
