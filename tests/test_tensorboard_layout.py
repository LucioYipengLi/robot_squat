"""Validate native TensorBoard grouping without duplicating or changing scalar data."""

import re

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.custom_scalar import layout_pb2, metadata
from torch.utils.tensorboard import SummaryWriter

from g1.rl.tensorboard_layout import ensure_performance_layout, main, performance_layout


def _read_layout(directory):
    accumulator = EventAccumulator(str(directory)).Reload()
    events = accumulator.Tensors(metadata.CONFIG_SUMMARY_TAG)
    return layout_pb2.Layout.FromString(events[0].tensor_proto.string_val[0]), events, accumulator


def test_overview_combines_matching_metrics_across_modes_and_other_groups_start_closed():
    layout = performance_layout()
    assert len(layout.category) == 6
    assert not layout.category[0].closed
    assert all(category.closed for category in layout.category[1:])
    overview = layout.category[0]
    assert len(overview.chart) == 6
    tags = overview.chart[0].multiline.tag
    assert len(tags) == 4
    for mode in ("stand", "squat", "walk", "squat_walk"):
        expected = f"Performance/task_command/{mode}/joint_success_rate"
        assert sum(re.fullmatch(pattern, expected) is not None for pattern in tags) == 1
    for category in layout.category:
        for chart in category.chart:
            assert chart.WhichOneof("content") == "multiline"
            if "骨盆" in chart.title and category.title.startswith("02"):
                assert len(chart.multiline.tag) == 2


def test_layout_regexes_do_not_capture_other_commands_or_similarly_named_metrics():
    patterns = performance_layout(("task.+[1]",)).category[0].chart[0].multiline.tag
    intended = "Performance/task.+[1]/stand/joint_success_rate"
    assert any(re.fullmatch(pattern, intended) for pattern in patterns)
    assert not any(re.fullmatch(pattern, intended + "_extra") for pattern in patterns)
    assert not any(re.fullmatch(pattern, "Performance/task_x1/stand/joint_success_rate") for pattern in patterns)


def test_existing_run_gains_only_one_metadata_file_and_preserves_scalar_history(tmp_path):
    tag = "Performance/task_command/stand/joint_success_rate"
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar(tag, 0.75, 42)
    original_files = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert ensure_performance_layout(tmp_path)
    assert not ensure_performance_layout(tmp_path)
    assert len(list(tmp_path.glob("events.out.tfevents.*.g1-layout-*"))) == 1
    for path, contents in original_files.items():
        assert path.read_bytes() == contents
    layout, events, accumulator = _read_layout(tmp_path)
    assert len(events) == 1
    assert len(layout.category) == 6
    assert accumulator.Tags()["scalars"] == [tag]
    assert [(event.step, event.value) for event in accumulator.Scalars(tag)] == [(42, 0.75)]
    assert accumulator.SummaryMetadata(metadata.CONFIG_SUMMARY_TAG).plugin_data.plugin_name == "custom_scalars"


def test_cli_registers_each_existing_run_and_can_be_repeated(tmp_path, capsys):
    runs = [tmp_path / "run_a", tmp_path / "run_b"]
    for directory in runs:
        with SummaryWriter(str(directory)) as writer:
            writer.add_scalar("Performance/task_command/stand/step_count", 50, 0)
    assert main(["--logdir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("Added layout:") == 2
    assert main(["--logdir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("Layout already present:") == 2
    for directory in runs:
        assert len(_read_layout(directory)[1]) == 1


def test_cli_rejects_empty_directories_without_creating_dummy_runs(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--logdir", str(tmp_path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
