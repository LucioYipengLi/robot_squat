"""Check random defaults, explicit seed overrides, and persisted run seeds without a simulator."""

import argparse

import pytest

from isaaclab.utils.io import load_yaml

from isaaclab_rl.entrypoints import common
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl

from g1.tasks.lower_limb_agent.config.g1.agents import rsl_rl_ppo_cfg as lower_cfg
from g1.tasks.upper_limb_agent.config.g1.agents import rsl_rl_ppo_cfg as upper_cfg


@pytest.fixture(params=[lower_cfg, upper_cfg], ids=["lower_limb", "upper_g1"])
def rsl_rl_ppo_cfg(request):
    return request.param


def test_each_new_config_draws_a_seed_but_copies_keep_the_selected_seed(monkeypatch, rsl_rl_ppo_cfg):
    draws = iter((123, 456))

    def draw(limit):
        assert limit == 2**31
        return next(draws)

    monkeypatch.setattr(rsl_rl_ppo_cfg, "randbelow", draw)
    first = rsl_rl_ppo_cfg.PPORunnerCfg()
    second = rsl_rl_ppo_cfg.PPORunnerCfg()
    assert first.seed == 123
    assert second.seed == 456
    assert first.copy().seed == 123
    assert first.replace(run_name="copied").seed == 123
    assert first.to_dict()["seed"] == 123


def test_explicit_config_seed_does_not_draw_a_random_default(monkeypatch, rsl_rl_ppo_cfg):
    def unexpected_draw(limit):
        pytest.fail("An explicit seed must bypass the default seed factory")

    monkeypatch.setattr(rsl_rl_ppo_cfg, "randbelow", unexpected_draw)
    assert rsl_rl_ppo_cfg.PPORunnerCfg(seed=0).seed == 0


@pytest.mark.parametrize("requested,expected", [(None, 2**31 - 1), (0, 0), (123, 123), (-1, 789)])
def test_native_seed_override_and_run_configs_store_the_selected_integer(
    tmp_path, monkeypatch, requested, expected, rsl_rl_ppo_cfg
):
    monkeypatch.setattr(rsl_rl_ppo_cfg, "randbelow", lambda limit: 2**31 - 1)
    calls = []

    def random_seed(low, high):
        calls.append((low, high))
        return 789

    monkeypatch.setattr(common.random, "randint", random_seed)
    args = argparse.Namespace(
        seed=requested,
        checkpoint=None,
        experiment_name=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )
    agent = cli_args_rsl_rl.update_rsl_rl_cfg(rsl_rl_ppo_cfg.PPORunnerCfg(), args)
    assert agent.seed == expected
    assert calls == ([(0, 10000)] if requested == -1 else [])
    # The backend copies the selected agent seed to env_cfg before calling this actual dumper.
    common.dump_train_configs(str(tmp_path), {"seed": agent.seed}, agent)
    for name in ("env", "agent"):
        stored = load_yaml(str(tmp_path / "params" / f"{name}.yaml"))
        assert type(stored["seed"]) is int
        assert stored["seed"] == expected
