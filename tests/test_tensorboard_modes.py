"""Verify per-mode display runs retain source values and work with native layouts."""

import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.plugins.custom_scalar import metadata
from torch.utils.tensorboard import SummaryWriter

from g1.rl.tensorboard_layout import main, performance_layout
from g1.rl.tensorboard_modes import MODE_VIEW_DIR, ModeComparisonWriter, export_mode_comparison

MODES = ("stand", "squat", "walk", "squat_walk")


def _scalars(directory, tag):
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    return [(event.wall_time, event.step, event.value) for event in accumulator.Scalars(tag)]


def test_live_view_gives_each_mode_a_distinct_run_in_the_same_overview_chart(tmp_path):
    values = {f"Performance/task_command/{mode}/joint_success_rate": index / 4 for index, mode in enumerate(MODES)}
    values.update(
        {
            "Performance/task_command/excluded/partial_episodes": 3,
            "TaskDistribution/task_command/height_offset/lo": -0.3,
            "Loss/value_function": 0.5,
            "Performance/another_command/stand/joint_success_rate": 0.8,
        }
    )
    view = ModeComparisonWriter(tmp_path)
    try:
        assert view.add_scalars(values, 42) == 6
        view.flush()
        patterns = performance_layout().category[0].chart[0].multiline.tag
        for mode in MODES:
            accumulator = EventAccumulator(str(view.directory / mode)).Reload()
            tags = accumulator.Tags()["scalars"]
            assert tags == [f"Performance/task_command/{mode}/joint_success_rate"]
            assert sum(bool(re.fullmatch(pattern, tag)) for pattern in patterns for tag in tags) == 1
            assert metadata.CONFIG_SUMMARY_TAG in accumulator.Tags()["tensors"]
        context = EventAccumulator(str(view.directory / "context")).Reload()
        assert set(context.Tags()["scalars"]) == {
            "Performance/task_command/excluded/partial_episodes",
            "TaskDistribution/task_command/height_offset/lo",
        }
        assert len(list(view.directory.iterdir())) == 5
        assert list(tmp_path.glob("events.out.tfevents.*")) == []
    finally:
        view.close()
    assert view._writers == {}


def test_backfill_preserves_history_wall_times_and_sources_without_repeated_points(tmp_path):
    tags = [f"Performance/task_command/{mode}/joint_success_rate" for mode in MODES]
    with SummaryWriter(str(tmp_path)) as writer:
        for step in (4, 17):
            for index, tag in enumerate(tags):
                writer.add_scalar(tag, index / 4, step, walltime=1000 + step)
        writer.add_scalar("TaskDistribution/task_command/phase", 1, 17, walltime=1017)
        writer.add_scalar("Loss/value_function", 0.5, 17)
    originals = {path: path.read_bytes() for path in tmp_path.glob("events.out.tfevents.*")}
    assert export_mode_comparison(tmp_path) == 9
    assert export_mode_comparison(tmp_path) == 0
    for tag, mode in zip(tags, MODES, strict=True):
        assert _scalars(tmp_path / MODE_VIEW_DIR / mode, tag) == _scalars(tmp_path, tag)
    for path, contents in originals.items():
        assert path.read_bytes() == contents
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar(tags[0], 0.9, 20, walltime=1020)
    assert export_mode_comparison(tmp_path) == 1
    assert len(_scalars(tmp_path / MODE_VIEW_DIR / "stand", tags[0])) == 3


def test_backfill_skips_points_already_published_live(tmp_path):
    tag = "Performance/task_command/walk/step_share"
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar(tag, 0.3, 42, walltime=100)
    view = ModeComparisonWriter(tmp_path)
    try:
        view.add_scalars({tag: 0.3}, 42)
    finally:
        view.close()
    assert export_mode_comparison(tmp_path) == 0
    assert len(_scalars(tmp_path / MODE_VIEW_DIR / "walk", tag)) == 1


def test_backfill_supports_tensor_scalars_and_ignores_non_scalar_summaries(tmp_path):
    tag = "Performance/task_command/walk/step_share"
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar(tag, 0.3, 42, walltime=100, new_style=True)
    view = ModeComparisonWriter(tmp_path)
    try:
        ignored = summary_pb2.Summary.Value(tag=tag, image=summary_pb2.Summary.Image())
        assert view.add_event(event_pb2.Event(summary=summary_pb2.Summary(value=[ignored]))) == 0
        assert not view.directory.exists()
    finally:
        view.close()
    assert export_mode_comparison(tmp_path) == 1
    assert export_mode_comparison(tmp_path) == 0
    events = EventAccumulator(str(tmp_path / MODE_VIEW_DIR / "walk")).Reload().Tensors(tag)
    assert [(event.wall_time, event.step) for event in events] == [(100, 42)]


def test_recursive_cli_never_uses_generated_views_as_sources(tmp_path, capsys):
    for run in ("run_a", "run_b"):
        with SummaryWriter(str(tmp_path / run)) as writer:
            writer.add_scalar("Performance/task_command/stand/step_count", 50, 0)
    for expected in (1, 0):
        assert main(["--logdir", str(tmp_path), "--mode-colors"]) == 0
        output = capsys.readouterr().out
        assert output.count(f"Added {expected} mode comparison points:") == 2
        assert output.count("layout:") + output.count("Layout already present:") == 2
    assert all(path.parts.count(MODE_VIEW_DIR) == 1 for path in tmp_path.rglob(MODE_VIEW_DIR))


def test_export_without_per_mode_metrics_creates_no_empty_view(tmp_path):
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar("Loss/value_function", 0.2, 0)
    assert export_mode_comparison(tmp_path) == 0
    assert not (tmp_path / MODE_VIEW_DIR).exists()
