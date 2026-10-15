# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import g1.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "Template-G1-v0": {
            "entry_point": "g1.tasks.lower_limb_agent.env:LowerLimbAgentEnv",
            "env_cfg_entry_point": "g1.tasks.lower_limb_agent.config.g1.env_cfg:LowerLimbAgentEnvCfg",
            "default_agent": "rsl_rl",
        },
        "Template-G1-Phase2-v0": {
            "entry_point": "g1.tasks.lower_limb_agent.env:LowerLimbAgentEnv",
            "env_cfg_entry_point": "g1.tasks.lower_limb_agent.config.g1.env_cfg_phase:LowerLimbAgentEnvCfg_Phase2",
            "default_agent": "rsl_rl",
        },
        "Template-G1-Phase3-v0": {
            "entry_point": "g1.tasks.lower_limb_agent.env:LowerLimbAgentEnv",
            "env_cfg_entry_point": "g1.tasks.lower_limb_agent.config.g1.env_cfg_phase:LowerLimbAgentEnvCfg_Phase3",
            "default_agent": "rsl_rl",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
