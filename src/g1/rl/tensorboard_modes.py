# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local display copies that let TensorBoard color each task mode as a separate run."""

from __future__ import annotations

import struct
import time
from collections import defaultdict
from pathlib import Path

from .tensorboard_layout import ensure_performance_layout

MODE_VIEW_DIR = "mode_comparison"
_MODES = ("stand", "squat", "walk", "squat_walk")


def _destination(tag, command_names):
    parts = tag.split("/", 3)
    if len(parts) < 3 or parts[1] not in command_names:
        return None
    if parts[0] == "Performance" and len(parts) == 4:
        if parts[2] in _MODES:
            return parts[2]
        if parts[2] == "excluded":
            return "context"
    if parts[0] == "TaskDistribution":
        return "context"
    return None


def _point_key(step, value):
    """Identify a scalar regardless of wall time or v1/v2 summary encoding."""
    if value.HasField("simple_value"):
        number = value.simple_value
    elif value.HasField("tensor"):
        from tensorboard.util import tensor_util

        array = tensor_util.make_ndarray(value.tensor)
        if array.ndim != 0 or array.dtype.kind not in "biuf":
            return None
        number = array.item()
    else:
        return None
    return value.tag, step, struct.pack("!d", float(number))


class ModeComparisonWriter:
    """Write only diagnostics to four mode runs plus a shared context run.

    Tags and numerical values match the source run, so the existing Custom Scalars
    layout still works. Colors remain a TensorBoard UI setting, never event metadata.
    """

    def __init__(self, log_dir, command_names=("task_command",)):
        self.directory = Path(log_dir) / MODE_VIEW_DIR
        self.command_names = tuple(command_names)
        self._writers = {}

    def add_scalars(self, values, step):
        from tensorboard.compat.proto import event_pb2, summary_pb2

        summary = summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value)) for tag, value in values.items()]
        )
        return self.add_event(event_pb2.Event(wall_time=time.time(), step=step, summary=summary))

    def add_event(self, event, seen=None):
        from tensorboard.compat.proto import event_pb2, summary_pb2
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        grouped = defaultdict(list)
        for value in event.summary.value:
            destination = _destination(value.tag, self.command_names)
            if destination is None:
                continue
            key = _point_key(event.step, value)
            if key is None or (seen is not None and key in seen):
                continue
            grouped[destination].append(value)
            if seen is not None:
                seen.add(key)
        for destination, values in grouped.items():
            if destination not in self._writers:
                directory = self.directory / destination
                ensure_performance_layout(directory, self.command_names)
                self._writers[destination] = EventFileWriter(
                    str(directory), flush_secs=10, filename_suffix=".g1-mode-view"
                )
            self._writers[destination].add_event(
                event_pb2.Event(wall_time=event.wall_time, step=event.step, summary=summary_pb2.Summary(value=values))
            )
        return sum(len(values) for values in grouped.values())

    def flush(self):
        for writer in self._writers.values():
            writer.flush()

    def close(self):
        try:
            for writer in self._writers.values():
                writer.close()
        finally:
            self._writers.clear()


def export_mode_comparison(log_dir, command_names=("task_command",)) -> int:
    """Backfill missing scalar points without rewriting source files or existing views.

    Repeated exports skip matching tag/iteration/value points, including points already
    copied by live logging. Only immediate source files are read; views never feed back
    into themselves. Existing raw histories and timestamps are preserved.
    """
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

    source = Path(log_dir)
    seen = set()
    view = ModeComparisonWriter(source, command_names)
    for directory in (*_MODES, "context"):
        for path in sorted((view.directory / directory).glob("events.out.tfevents.*.g1-mode-view")):
            for event in LegacyEventFileLoader(str(path)).Load():
                for value in event.summary.value:
                    key = _point_key(event.step, value)
                    if key is not None:
                        seen.add(key)
    count = 0
    try:
        for path in sorted(source.glob("events.out.tfevents.*")):
            for event in LegacyEventFileLoader(str(path)).Load():
                count += view.add_event(event, seen)
    finally:
        view.close()
    return count
