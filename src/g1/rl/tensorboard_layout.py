# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TensorBoard-native comparison panels; scalar names and stored values stay unchanged."""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

_MODES = ("stand", "squat", "walk", "squat_walk")
_GROUPS = (
    (
        "01 总览",
        (
            ("联合成功率 ↑", "joint_success_rate"),
            ("失败率 ↓", "failure_rate"),
            ("存活比例 ↑", "survival_fraction"),
            ("高度误差 (m) ↓", "error_height_m"),
            ("平面速度误差 (m/s) ↓", "error_velocity_xy_mps"),
            ("实际训练步数占比", "step_share"),
        ),
    ),
    (
        "02 跟踪详情",
        (
            ("转向误差 (rad/s) ↓", "error_yaw_radps"),
            ("骨盆水平误差 (m) ↓", "error_pelvis_xy_m"),
            ("高度跟踪成功率 ↑", "height_tracking_success_rate"),
            ("速度跟踪成功率 ↑", "velocity_tracking_success_rate"),
            ("骨盆逐步成功率 ↑", "pelvis_step_success_rate"),
        ),
    ),
    (
        "03 样本与有效性",
        (
            ("有效结束回合数", "episode_count"),
            ("有效结束回合占比", "episode_share"),
            ("实际控制步数", "step_count"),
            ("新回合采样概率", "sampling_probability"),
        ),
    ),
    (
        "04 生存与终止",
        (
            ("回合时长 (s) ↑", "episode_length_s"),
            ("正常超时率 ↑", "timeout_rate"),
            ("躯干接触终止 ↓", "termination/base_contact"),
            ("膝部接触终止 ↓", "termination/knee_contact"),
            ("姿态异常终止 ↓", "termination/bad_ori"),
            ("高度不足终止 ↓", "termination/base_height_below_minimum"),
        ),
    ),
    (
        "05 扰动暴露",
        (
            ("实际扰动次数", "disturbance_count"),
            ("实际扰动暴露率 (Hz)", "disturbance_hz"),
            ("实际双臂扰动比例", "disturbance_arm_fraction"),
        ),
    ),
    (
        "06 任务难度",
        (
            ("实际高度目标偏移 (m)", "height_offset_m"),
            ("实际目标速度 (m/s)", "command_speed_xy_mps"),
            ("实际目标转向 (rad/s)", "command_yaw_abs_radps"),
        ),
    ),
)


def performance_layout(command_names=("task_command",)):
    """Build multiline panels with exact tag regexes and only Overview initially expanded."""
    from tensorboard.plugins.custom_scalar.layout_pb2 import Layout

    layout = Layout()
    for name in sorted(set(command_names)):
        prefix = f"Performance/{name}/"
        categories = {}
        for index, (title, charts) in enumerate(_GROUPS):
            category = layout.category.add(title=f"{title} · {name}", closed=index != 0)
            categories[title[:2]] = category
            for chart_title, metric in charts:
                # Pelvis tracking has no definition for walking modes; do not invent curves.
                modes = _MODES[:2] if metric in ("error_pelvis_xy_m", "pelvis_step_success_rate") else _MODES
                tags = [f"^{re.escape(prefix + mode + '/' + metric)}$" for mode in modes]
                category.chart.add(title=chart_title).multiline.tag.extend(tags)
        categories["03"].chart.add(title="排除回合数").multiline.tag.extend(
            f"^{re.escape(prefix + 'excluded/' + reason)}$"
            for reason in ("manual_episodes", "partial_episodes", "mixed_mode_episodes")
        )
        categories["05"].chart.add(title="配置扰动比例：双臂 / 腰部").multiline.tag.extend(
            f"^{re.escape(f'TaskDistribution/{name}/disturbance_group_{index}_fraction')}$" for index in (0, 1)
        )
        for field, title in (
            ("height_offset", "高度采样边界 (m)"),
            ("lin_vel_x", "前后速度采样边界 (m/s)"),
            ("lin_vel_y", "侧向速度采样边界 (m/s)"),
            ("ang_vel_z", "转向采样边界 (rad/s)"),
            ("pelvis_offset_x", "骨盆前后偏移边界 (m)"),
            ("pelvis_offset_y", "骨盆左右偏移边界 (m)"),
        ):
            categories["06"].chart.add(title=title).multiline.tag.extend(
                f"^{re.escape(f'TaskDistribution/{name}/{field}/{bound}')}$" for bound in ("lo", "hi")
            )
    return layout


def ensure_performance_layout(log_dir, command_names=("task_command",)) -> bool:
    """Append a separate metadata-only event file, once per layout content and run directory.

    TensorBoard's Layout proto supports initially collapsed groups, unlike the shorter
    SummaryWriter.add_custom_scalars dictionary API. No extra scalar series or child runs
    are created. This also works for existing runs and the TensorBoard side of W&B writers.
    """
    from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
    from tensorboard.plugins.custom_scalar import metadata
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    serialized = performance_layout(command_names).SerializeToString()
    suffix = ".g1-layout-" + hashlib.sha256(serialized).hexdigest()[:16]
    directory = Path(log_dir)
    if any(directory.glob("events.out.tfevents.*" + suffix)):
        return False
    tensor = tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING, string_val=[serialized])
    summary = summary_pb2.Summary(
        value=[
            summary_pb2.Summary.Value(
                tag=metadata.CONFIG_SUMMARY_TAG,
                metadata=metadata.create_summary_metadata(),
                tensor=tensor,
            )
        ]
    )
    writer = EventFileWriter(str(directory), filename_suffix=suffix)
    try:
        writer.add_event(event_pb2.Event(wall_time=time.time(), summary=summary))
        writer.flush()
    finally:
        writer.close()
    return True


def main(argv=None):
    """Add native comparison panels to an existing run or recursively to an experiment."""
    parser = argparse.ArgumentParser(description="Register grouped TensorBoard panels without changing scalar data.")
    parser.add_argument("--logdir", required=True, type=Path, help="Existing run or experiment directory")
    parser.add_argument("--command-name", default="task_command")
    parser.add_argument(
        "--mode-colors",
        action="store_true",
        help="Also copy diagnostics into per-mode display runs for distinct colors",
    )
    args = parser.parse_args(argv)
    from .tensorboard_modes import MODE_VIEW_DIR, export_mode_comparison

    if not args.logdir.is_dir():
        parser.error("--logdir must be an existing directory")
    if MODE_VIEW_DIR in args.logdir.parts:
        parser.error("--logdir must point to the source run or experiment, not mode_comparison")
    runs = sorted(
        {
            path.parent
            for path in args.logdir.rglob("events.out.tfevents.*")
            if MODE_VIEW_DIR not in path.relative_to(args.logdir).parts
        }
    )
    if not runs:
        parser.error("No TensorBoard event files found")
    for directory in runs:
        created = ensure_performance_layout(directory, (args.command_name,))
        print(f"{'Added layout' if created else 'Layout already present'}: {directory}")
        if args.mode_colors:
            count = export_mode_comparison(directory, (args.command_name,))
            print(f"Added {count} mode comparison points: {directory / MODE_VIEW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
