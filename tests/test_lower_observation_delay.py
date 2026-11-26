"""Keep sensor delay scoped to the selected lower-limb training observations."""

import pytest

from isaaclab.managers import ObservationTermCfg

from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg
from g1.tasks.lower_limb_agent.config.g1.env_cfg_phase import (
    LowerLimbAgentEnvCfg_Phase2,
    LowerLimbAgentEnvCfg_Phase3,
)


@pytest.mark.parametrize("cfg_type", [LowerLimbAgentEnvCfg_Phase2, LowerLimbAgentEnvCfg_Phase3])
def test_observation_delay_only_applies_to_the_five_sensor_terms(cfg_type):
    cfg = cfg_type()
    terms = {name: term for name, term in vars(cfg.observations.policy).items() if isinstance(term, ObservationTermCfg)}
    assert {name for name, term in terms.items() if term.delay_max_lag} == {
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
    }
    assert all(term.delay_min_lag == 0 and term.delay_max_lag <= 1 for term in terms.values())
    assert all(term.delay_hold_prob == 1.0 for term in terms.values())
    assert cfg.observations.policy.history_length == 5
    assert LowerLimbAgentEnvCfg().observations.policy.joint_pos.delay_max_lag == 0


@pytest.mark.parametrize("mode", ["play_mode", "teleop_mode"])
def test_phase3_nontraining_modes_keep_zero_sensor_delay(mode):
    cfg = LowerLimbAgentEnvCfg_Phase3()
    getattr(cfg, mode)()
    assert all(
        term.delay_min_lag == term.delay_max_lag == 0
        for term in vars(cfg.copy().observations.policy).values()
        if isinstance(term, ObservationTermCfg)
    )
    assert LowerLimbAgentEnvCfg_Phase3().observations.policy.joint_pos.delay_max_lag == 1
