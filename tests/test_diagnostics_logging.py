"""Verify rollout diagnostics reach existing writers without reset-batch averaging."""

from types import SimpleNamespace

import pytest
import torch

from g1.rl.diagnostics import install_performance_logging
from g1.tasks.lower_limb_agent.performance import ModePerformanceMonitor


@pytest.mark.parametrize("writer_enabled", [False, True])
def test_logger_adapter_forwards_original_call_and_publishes_one_window(writer_enabled):
    records = {}
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: records.setdefault((tag, step), value))
    logger, monitor, base, calls = _fixture(writer if writer_enabled else None)
    install_performance_logging(logger, SimpleNamespace(unwrapped=base))
    logger.log(it=42, learning_rate=0.001)
    assert calls == [{"it": 42, "learning_rate": 0.001}]
    assert monitor.steps.sum() == 0
    if writer_enabled:
        assert records[("Performance/task_command/stand/step_count", 42)] == 10
        assert records[("TaskDistribution/task_command/curriculum_control_steps", 42)] == 600
        assert records[("TaskDistribution/task_command/disturbance_group_0_fraction", 42)] == 0.3
        assert records[("TaskDistribution/task_command/height_offset/lo", 42)] == -0.2
        assert ("Performance/task_command/stand/error_height_m", 42) not in records
    else:
        assert records == {}


def _fixture(writer):
    calls = []
    logger = SimpleNamespace(writer=writer, log=lambda **kwargs: calls.append(kwargs))
    monitor = ModePerformanceMonitor(1, "cpu", 0.1, (0.03, 0.5, 0.4, 0.02))
    monitor.steps[0, 0] = 10
    term = SimpleNamespace(
        performance=monitor,
        _mode_probs=torch.tensor([1.0, 0, 0, 0]),
        cfg=SimpleNamespace(ranges=SimpleNamespace(height_offset=(-0.2, 0.02))),
    )
    base = SimpleNamespace(
        command_manager=SimpleNamespace(active_terms=["task_command"], get_term=lambda name: term),
        cfg=SimpleNamespace(training_phase=1, events=SimpleNamespace(upper_body_disturbance=True)),
        event_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(params={"disturb_groups": [{"joint_noise_frac": 0.3}]})
        ),
        common_step_counter=100,
        _g1_curriculum_step_offset=500,
    )
    return logger, monitor, base, calls


@pytest.mark.parametrize("backend", ["tensorboard", "wandb"])
def test_real_writer_records_diagnostics_at_training_iteration(tmp_path, monkeypatch, backend):
    import rsl_rl.utils.wandb_log_writer as wandb_writer
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    remote = []
    if backend == "wandb":
        monkeypatch.setattr(
            wandb_writer,
            "wandb",
            SimpleNamespace(
                init=lambda **kwargs: None,
                log=lambda values, step: remote.append((values, step)),
                finish=lambda: None,
            ),
        )
        writer = wandb_writer.WandbLogWriter(str(tmp_path), "diagnostic-test")
    else:
        writer = SummaryWriter(str(tmp_path))
    logger, _, base, _ = _fixture(writer)
    install_performance_logging(logger, SimpleNamespace(unwrapped=base))
    try:
        logger.log(it=42)
        writer.flush()
        logger.stop_logging_writer()
        accumulator = EventAccumulator(str(tmp_path)).Reload()
        events = accumulator.Scalars("Performance/task_command/stand/step_count")
        assert [(event.step, event.value) for event in events] == [(42, 10.0)]
        assert "custom_scalars__config__" in accumulator.Tags()["tensors"]
        view = EventAccumulator(str(tmp_path / "mode_comparison" / "stand")).Reload()
        assert [(event.step, event.value) for event in view.Scalars("Performance/task_command/stand/step_count")] == [
            (42, 10.0)
        ]
        if backend == "wandb":
            assert ({"Performance/task_command/stand/step_count": 10.0}, 42) in remote
            assert sum("Performance/task_command/stand/step_count" in values for values, _ in remote) == 1
    finally:
        logger.stop_logging_writer()
        writer.close()


def test_curriculum_runner_installs_diagnostic_adapter(monkeypatch):
    from rsl_rl.runners import OnPolicyRunner

    from g1.rl.runner import CurriculumOnPolicyRunner

    records = {}
    logger, _, base, calls = _fixture(
        SimpleNamespace(add_scalar=lambda tag, value, step: records.setdefault((tag, step), value))
    )

    def initialize(self, env, cfg):
        self.env, self.logger = env, logger

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(OnPolicyRunner, "__init__", initialize)
    runner = CurriculumOnPolicyRunner(SimpleNamespace(unwrapped=base), {})
    runner.logger.log(it=5)
    assert calls == [{"it": 5}]
    assert records[("Performance/task_command/stand/step_count", 5)] == 10


@pytest.mark.parametrize("fail", [False, True])
def test_legacy_pruning_preserves_source_data_unique_metrics_and_writer_method(fail):
    records = {}
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: records.setdefault((tag, step), value))
    original_scalar = writer.add_scalar
    logger, _, base, _ = _fixture(writer)
    source = {
        "Metrics/success_rate": torch.tensor(0.9),
        "Metrics/task_command/error_height": torch.tensor(0.02),
        "Metrics/task_command/success_rate_vel": torch.tensor(0.8),
        "Episode_Termination/bad_ori": torch.tensor(0.1),
        "Curriculum/command_range/height_offset_lo": -0.3,
        "Curriculum/command_range/lin_vel_x_span": 0.9,
        "Curriculum/upper_body_disturbance_magnitude/joint_noise_frac": 0.3,
        "Curriculum/squat_walk_mode_introduction/squat_walk_weight": 0.25,
        "Curriculum/command_range/height_success": torch.tensor(0.99),
        "Curriculum/command_range/vel_success": torch.tensor(0.95),
        "Curriculum/upper_body_disturbance_magnitude/survival_ema": torch.tensor(0.97),
        "Curriculum/upper_body_disturbance_magnitude/survival_valid": 1,
        "Curriculum/upper_body_disturbance_magnitude/survival_episodes": 500,
        "Curriculum/squat_walk_mode_introduction/introduced": 1,
        "Episode_Reward/track_velocity": 1.5,
        "Metrics/another_command/custom": 7,
        "Curriculum/custom_term/signal": 8,
    }
    original_keys = set(source)
    logger.ep_extras = [source]

    def original_log(it):
        for batch in logger.ep_extras:
            for key, value in batch.items():
                writer.add_scalar(key, value, it)
        for key in (
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Train/mean_reward/time",
            "Train/mean_episode_length/time",
        ):
            writer.add_scalar(key, 1.0, it)
        if fail:
            raise RuntimeError("writer failed")

    logger.log = original_log
    install_performance_logging(logger, SimpleNamespace(unwrapped=base))
    if fail:
        with pytest.raises(RuntimeError, match="writer failed"):
            logger.log(it=9)
    else:
        logger.log(it=9)
    assert set(source) == original_keys  # environment dictionaries are not mutated
    assert writer.add_scalar is original_scalar
    for key in list(source)[:8]:
        assert (key, 9) not in records
    for key in list(source)[8:]:
        assert (key, 9) in records
    assert ("Train/mean_reward", 9) in records
    assert ("Train/mean_episode_length", 9) in records
    assert ("Train/mean_reward/time", 9) not in records
    assert ("Train/mean_episode_length/time", 9) not in records


def test_real_rsl_logger_keeps_training_signals_but_omits_superseded_series(tmp_path, capsys):
    from rsl_rl.utils.logger import Logger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    _, _, base, _ = _fixture(None)
    logger = Logger(
        log_dir=str(tmp_path),
        cfg={"algorithm": {}, "num_steps_per_env": 50},
        env_cfg={},
        num_envs=1,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    logger.writer = SummaryWriter(str(tmp_path))
    logger.rewbuffer.append(2.0)
    logger.lenbuffer.append(360)
    logger.ep_extras.append(
        {
            "Metrics/task_command/error_height": torch.tensor(0.2),
            "Episode_Termination/bad_ori": torch.tensor(1.0),
            "Episode_Reward/track_velocity": torch.tensor(1.2),
            "Curriculum/command_range/height_success": torch.tensor(0.95),
        }
    )
    install_performance_logging(logger, SimpleNamespace(unwrapped=base))
    try:
        logger.log(
            it=0,
            start_it=0,
            total_it=10,
            collect_time=0.1,
            learn_time=0.2,
            loss_dict={"value_function": 0.3},
            learning_rate=0.001,
            action_std=torch.ones(1),
            rnd_weight=None,
        )
        logger.writer.flush()
        tags = EventAccumulator(str(tmp_path)).Reload().Tags()["scalars"]
        for key in (
            "Loss/value_function",
            "Loss/learning_rate",
            "Policy/mean_std",
            "Perf/total_fps",
            "Perf/collection_time",
            "Perf/learning_time",
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Episode_Reward/track_velocity",
            "Curriculum/command_range/height_success",
            "Performance/task_command/stand/step_count",
        ):
            assert key in tags
        for key in (
            "Metrics/task_command/error_height",
            "Episode_Termination/bad_ori",
            "Train/mean_reward/time",
            "Train/mean_episode_length/time",
        ):
            assert key not in tags
        output = capsys.readouterr().out
        assert "Metrics/task_command/error_height" not in output
        assert "Episode_Reward/track_velocity" in output
        assert "Mean reward:" in output
        assert logger.ep_extras == []
        assert "add_scalar" not in vars(logger.writer)  # restore inherited writer method
    finally:
        logger.stop_logging_writer()
        logger.writer.close()


@pytest.mark.parametrize("fail", [False, True])
def test_runner_closes_display_writers_when_learning_finishes_or_raises(tmp_path, monkeypatch, fail):
    from rsl_rl.runners import OnPolicyRunner
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    from g1.rl.runner import CurriculumOnPolicyRunner

    writer = SummaryWriter(str(tmp_path))
    logger, _, base, _ = _fixture(writer)

    def initialize(self, env, cfg):
        self.env, self.logger = env, logger

    def learn(self, *args, **kwargs):
        self.logger.log(it=42)
        if fail:
            raise RuntimeError("training interrupted")

    monkeypatch.setattr(OnPolicyRunner, "__init__", initialize)
    monkeypatch.setattr(OnPolicyRunner, "learn", learn)
    monkeypatch.setenv("WORLD_SIZE", "1")
    runner = CurriculumOnPolicyRunner(SimpleNamespace(unwrapped=base), {})
    try:
        if fail:
            with pytest.raises(RuntimeError, match="training interrupted"):
                runner.learn(1)
        else:
            runner.learn(1)
        # The buffered last iteration must be readable even after an exception.
        accumulator = EventAccumulator(str(tmp_path / "mode_comparison" / "stand")).Reload()
        assert [
            (event.step, event.value) for event in accumulator.Scalars("Performance/task_command/stand/step_count")
        ] == [(42, 10.0)]
    finally:
        runner._close_performance_logging()
        writer.close()


def test_logging_without_per_mode_monitor_preserves_legacy_outputs():
    records = {}
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: records.setdefault((tag, step), value))
    logger, _, base, _ = _fixture(writer)
    term = base.command_manager.get_term("task_command")
    del term.performance
    logger.ep_extras = [{"Metrics/success_rate": 1.0}]
    logger.log = lambda it: writer.add_scalar("Train/mean_reward/time", 1.0, it)
    install_performance_logging(logger, SimpleNamespace(unwrapped=base))
    logger.log(it=3)
    assert logger.ep_extras == [{"Metrics/success_rate": 1.0}]
    assert records == {("Train/mean_reward/time", 3): 1.0}
