# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive teleoperation play script for an RSL-RL checkpoint.

与 ``play.py`` 的区别：本脚本用于「人在环」实时控制——

* 只创建 **1 个环境**（``num_envs=1``）。
* **无限时间**：移除 ``time_out`` 终止项，episode 不会因超时结束。
* **摔倒等终止仍生效并自动恢复**：``base_contact`` / ``knee_contact`` / ``bad_ori`` /
  ``base_height_below_minimum`` 保留；触发后 ``ManagerBasedRLEnv`` 自动 reset，机器人
  在原地（无随机姿态/速度扰动）重新站起。
* **键盘 / 手柄实时控制目标指令**：通过 ``set_manual_command`` 提交六维
  ``[h_offset, vx, vy, wz, pelvis_dx, pelvis_dy]``，保留指令项的模式约束和骨盆目标限速。
  高度/速度键盘与手柄叠加；骨盆偏置通过 I/K、J/L 调整，V 清零，仅 STAND/SQUAT 生效。
  当前策略输入为 102 维，旧 100 维 checkpoint 需要重训或显式迁移。

控制映射（Isaac Sim 窗口需处于焦点）::

    [键盘]
    W / ↑        vx +（前进）        S / ↓        vx -（后退）
    A            vy +（左移）        D            vy -（右移）
    Q / ←        wz +（左转）        E / →        wz -（右转）
    Z            h  +（升高/站直）   X            h  -（降低/下蹲）
    SPACE        归零（回到默认站立） H            打印按键帮助
    C            切换骨盆实际/目标投影及双脚几何包络
    I / K        骨盆前 / 后        J / L        骨盆左 / 右
    V            仅骨盆偏置归零（仍跟踪双踝中点）

    [手柄 · Xbox 布局]
    左摇杆 上/下   vx 前进/后退（比例，松杆归零）
    左摇杆 左/右   wz 左转/右转（比例，松杆归零）
    方向键 上/下   vx + / -（逐次步进，累加保持）
    方向键 左/右   vy + / -（逐次步进，累加保持）
    右摇杆 上/下   h  升高/降低（积分，松杆保持当前髋高）
    A 键           归零（回到默认站立）

运行示例::

    ./isaaclab.sh -p scripts/rsl_rl/play_teleop.py --task Template-G1-v0
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import g1.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


##
# Teleop command limits & steps
##
# 钳制到训练时（含课程扩展后）见过的指令范围，避免下发分布外目标导致策略失稳。
LIN_VEL_X_LIMIT = (-0.5, 1.0)
LIN_VEL_Y_LIMIT = (-0.2, 0.2)
ANG_VEL_Z_LIMIT = (-0.8, 0.8)
HEIGHT_OFFSET_LIMIT = (-0.45, 0.05)

LIN_VEL_STEP = 0.1  # m/s，每按一次键 / 方向键的线速度增量
ANG_VEL_STEP = 0.2  # rad/s，每按一次键的偏航角速度增量
HEIGHT_STEP = 0.05  # m，每按一次键的髋高偏移增量
PELVIS_OFFSET_STEP = 0.005  # m，每次按键 5 mm；上下限从当前任务配置读取

# 手柄（Xbox 布局）参数
GAMEPAD_DEADZONE = 0.08  # 摇杆死区，抑制中位漂移
GAMEPAD_HEIGHT_RATE = 0.3  # m/s，右摇杆满偏时的髋高调整速率（积分式，松杆保持）

HELP_TEXT = (
    "\n==== G1 Teleop 控制 (Isaac Sim 窗口需处于焦点) ====\n"
    "[键盘]\n"
    "  W / ↑ : vx +  前进          S / ↓ : vx -  后退\n"
    "  A     : vy +  左移          D     : vy -  右移\n"
    "  Q / ← : wz +  左转          E / → : wz -  右转\n"
    "  Z     : h  +  升高/站直     X     : h  -  降低/下蹲\n"
    "  SPACE : 归零 (默认站立)     H     : 打印本帮助\n"
    "  C     : 切换骨盆实际/目标投影及双脚几何包络\n"
    "  I / K : 骨盆前 / 后       J / L : 骨盆左 / 右（每次 5 mm）\n"
    "  V     : 仅骨盆偏置归零（仍跟踪中点）；行走时偏置指令屏蔽\n"
    "[手柄 · Xbox 布局]\n"
    "  左摇杆 上/下 : vx  前进/后退 (比例，松杆归零)\n"
    "  左摇杆 左/右 : wz  左转/右转 (比例，松杆归零)\n"
    "  方向键 上/下 : vx  + / -  (逐次步进，累加保持)\n"
    "  方向键 左/右 : vy  + / -  (逐次步进，累加保持)\n"
    "  右摇杆 上/下 : h   升高/降低 (积分，松杆保持当前髋高)\n"
    "  A 键         : 归零 (默认站立)\n"
    "  在终端按 Ctrl+C 退出。\n"
    "===================================================\n"
)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


class KeyboardCommandTeleop:
    """Read keyboard events from the Isaac Sim window and maintain a manual task command.

    通过 ``carb.input`` 订阅键盘事件（Isaac Sim 原生输入，无需 tkinter/pynput 等 GUI 依赖）。
    维护 6 维指令状态 ``[h, vx, vy, wz, pelvis_dx, pelvis_dy]``，骨盆偏置范围从任务配置读取。
    若在无窗口（headless）模式下无法获取键盘，则退化为保持零指令并打印警告。
    """

    def __init__(self, pelvis_x_limits=(-0.03, 0.03), pelvis_y_limits=(-0.03, 0.03)) -> None:
        # 保留高度/速度输入语义，追加骨盆原始目标；限速由指令项执行。
        self.h = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.pelvis_dx = 0.0
        self.pelvis_dy = 0.0
        self.pelvis_x_limits = pelvis_x_limits
        self.pelvis_y_limits = pelvis_y_limits
        self.enabled = False
        self.show_pelvis = False  # C 键切换：骨盆投影 + 双脚支撑多边形可视化
        self._subscription = None

        try:
            import carb
            import omni.appwindow

            self._carb = carb
            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            if keyboard is None:
                print("[WARN] 无法获取键盘设备（可能处于 headless 模式），teleop 已禁用。")
                return
            self._input = carb.input.acquire_input_interface()
            self._subscription = self._input.subscribe_to_keyboard_events(keyboard, self._on_keyboard_event)
            self.enabled = True
        except Exception as e:  # noqa: BLE001 - 输入子系统缺失不应中断仿真
            print(f"[WARN] 键盘 teleop 初始化失败：{e}。将保持零指令。")
            self.enabled = False

    def _on_keyboard_event(self, event, *_) -> bool:
        """carb 键盘事件回调：仅在按下/重复时更新指令状态。"""
        kb = self._carb.input.KeyboardInput
        event_type = self._carb.input.KeyboardEventType
        if event.type in (event_type.KEY_PRESS, event_type.KEY_REPEAT):
            self._handle_key(event.input, kb)
        return True

    def _handle_key(self, key, kb) -> None:
        """Map a single key press to a command increment."""
        # 前进 / 后退 (vx)
        if key in (kb.W, kb.UP):
            self.vx = _clamp(self.vx + LIN_VEL_STEP, *LIN_VEL_X_LIMIT)
        elif key in (kb.S, kb.DOWN):
            self.vx = _clamp(self.vx - LIN_VEL_STEP, *LIN_VEL_X_LIMIT)
        # 左移 / 右移 (vy)
        elif key == kb.A:
            self.vy = _clamp(self.vy + LIN_VEL_STEP, *LIN_VEL_Y_LIMIT)
        elif key == kb.D:
            self.vy = _clamp(self.vy - LIN_VEL_STEP, *LIN_VEL_Y_LIMIT)
        # 左转 / 右转 (wz)
        elif key in (kb.Q, kb.LEFT):
            self.wz = _clamp(self.wz + ANG_VEL_STEP, *ANG_VEL_Z_LIMIT)
        elif key in (kb.E, kb.RIGHT):
            self.wz = _clamp(self.wz - ANG_VEL_STEP, *ANG_VEL_Z_LIMIT)
        # 升高 / 降低 (h_offset)
        elif key == kb.Z:
            self.h = _clamp(self.h + HEIGHT_STEP, *HEIGHT_OFFSET_LIMIT)
        elif key == kb.X:
            self.h = _clamp(self.h - HEIGHT_STEP, *HEIGHT_OFFSET_LIMIT)
        elif key == kb.I:
            self.pelvis_dx = _clamp(self.pelvis_dx + PELVIS_OFFSET_STEP, *self.pelvis_x_limits)
        elif key == kb.K:
            self.pelvis_dx = _clamp(self.pelvis_dx - PELVIS_OFFSET_STEP, *self.pelvis_x_limits)
        elif key == kb.J:
            self.pelvis_dy = _clamp(self.pelvis_dy + PELVIS_OFFSET_STEP, *self.pelvis_y_limits)
        elif key == kb.L:
            self.pelvis_dy = _clamp(self.pelvis_dy - PELVIS_OFFSET_STEP, *self.pelvis_y_limits)
        elif key == kb.V:
            self.pelvis_dx = self.pelvis_dy = 0.0
        # 归零
        elif key == kb.SPACE:
            self.reset()
        # 帮助
        elif key == kb.H:
            print(HELP_TEXT)
        # 骨盆 / 支撑多边形可视化开关
        elif key == kb.C:
            self.show_pelvis = not self.show_pelvis
            print(f"\n[VIS] 骨盆可视化：{'ON' if self.show_pelvis else 'OFF'}")

    def reset(self) -> None:
        """Zero the command (回到默认站立)。"""
        self.h = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.pelvis_dx = 0.0
        self.pelvis_dy = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        """返回人工原始目标 (h, vx, vy, wz, pelvis_dx, pelvis_dy)。"""
        return (self.h, self.vx, self.vy, self.wz, self.pelvis_dx, self.pelvis_dy)


class GamepadCommandTeleop:
    """Read Xbox-layout gamepad events and maintain a manual task command.

    通过 ``carb.input`` 订阅手柄事件（Isaac Sim 原生输入，无额外 GUI 依赖）。控制约定：

    * **左摇杆**（比例，松杆归零）：上/下 → vx 前进/后退；左/右 → wz 左转/右转。
    * **方向键 D-pad**（逐次步进，累加保持）：上/下 → vx ±；左/右 → vy ±。
    * **右摇杆**（积分，松杆保持）：上/下 → 髋高偏移 h 升降；需在主循环每帧调用
      :meth:`update` 按 ``dt`` 积分摇杆偏转量（carb 对稳定握住的摇杆不重复上报事件）。
    * **A 键**（FACE_SOUTH）：归零。

    vx 由左摇杆（比例）与方向键（累加）叠加后钳制；其余轴各自独立。若无手柄或处于
    headless 模式，则 ``enabled=False``，``as_tuple()`` 恒返回零指令。
    """

    def __init__(self) -> None:
        # 摇杆合成偏转量（[-1, 1]，已去死区）
        self._lx = 0.0  # 左摇杆水平：+左 / -右
        self._ly = 0.0  # 左摇杆垂直：+上 / -下
        self._ry = 0.0  # 右摇杆垂直：+上 / -下
        # 方向键累加值
        self._vx_dpad = 0.0
        self._vy_dpad = 0.0
        # 右摇杆积分得到的髋高偏移
        self._h = 0.0
        # 各方向按键的原始值缓存（carb 分别上报 UP/DOWN/LEFT/RIGHT）
        self._axis_values: dict = {}
        self.enabled = False
        self._subscription = None

        try:
            import carb
            import omni.appwindow

            self._carb = carb
            app_window = omni.appwindow.get_default_app_window()
            gamepad = app_window.get_gamepad(0)
            if gamepad is None:
                print("[WARN] 未检测到手柄（get_gamepad(0) 返回 None），手柄 teleop 已禁用。")
                return
            self._input = carb.input.acquire_input_interface()
            self._subscription = self._input.subscribe_to_gamepad_events(gamepad, self._on_gamepad_event)
            self.enabled = True
            print("[INFO] 手柄已连接，Xbox 布局 teleop 就绪。")
        except Exception as e:  # noqa: BLE001 - 输入子系统缺失不应中断仿真
            print(f"[WARN] 手柄 teleop 初始化失败：{e}。手柄输入已禁用。")
            self.enabled = False

    def _on_gamepad_event(self, event, *_) -> bool:
        """carb 手柄事件回调：缓存摇杆轴原始值，处理方向键 / 按钮。"""
        gp = self._carb.input.GamepadInput
        value = float(event.value)
        key = event.input

        # 摇杆轴：缓存原始值后重算合成偏转（松杆时 carb 上报 value==0）
        if key in (
            gp.LSTICK_UP,
            gp.LSTICK_DOWN,
            gp.LSTICK_LEFT,
            gp.LSTICK_RIGHT,
            gp.RSTICK_UP,
            gp.RSTICK_DOWN,
        ):
            self._axis_values[key] = value
            self._refresh_axes(gp)
            return True

        # 方向键 / 按钮：仅在按下（value != 0）时逐次步进
        if value != 0.0:
            if key == gp.DPAD_UP:
                self._vx_dpad = _clamp(self._vx_dpad + LIN_VEL_STEP, *LIN_VEL_X_LIMIT)
            elif key == gp.DPAD_DOWN:
                self._vx_dpad = _clamp(self._vx_dpad - LIN_VEL_STEP, *LIN_VEL_X_LIMIT)
            elif key == gp.DPAD_LEFT:
                self._vy_dpad = _clamp(self._vy_dpad + LIN_VEL_STEP, *LIN_VEL_Y_LIMIT)
            elif key == gp.DPAD_RIGHT:
                self._vy_dpad = _clamp(self._vy_dpad - LIN_VEL_STEP, *LIN_VEL_Y_LIMIT)
            elif key == gp.FACE_SOUTH:  # A 键归零
                self.reset()
        return True

    def _refresh_axes(self, gp) -> None:
        """Recompute stick deflections from cached per-direction values (with deadzone)."""

        def _dz(v: float) -> float:
            return 0.0 if abs(v) < GAMEPAD_DEADZONE else v

        up = self._axis_values.get(gp.LSTICK_UP, 0.0)
        down = self._axis_values.get(gp.LSTICK_DOWN, 0.0)
        left = self._axis_values.get(gp.LSTICK_LEFT, 0.0)
        right = self._axis_values.get(gp.LSTICK_RIGHT, 0.0)
        rup = self._axis_values.get(gp.RSTICK_UP, 0.0)
        rdown = self._axis_values.get(gp.RSTICK_DOWN, 0.0)
        self._ly = _dz(_clamp(up - down, -1.0, 1.0))
        self._lx = _dz(_clamp(left - right, -1.0, 1.0))
        self._ry = _dz(_clamp(rup - rdown, -1.0, 1.0))

    def update(self, dt: float) -> None:
        """Integrate the right-stick deflection into the height offset（每帧调用）。"""
        if self._ry != 0.0:
            self._h = _clamp(self._h + self._ry * GAMEPAD_HEIGHT_RATE * dt, *HEIGHT_OFFSET_LIMIT)

    def reset(self) -> None:
        """Zero the whole command（回到默认站立）。"""
        self._vx_dpad = 0.0
        self._vy_dpad = 0.0
        self._h = 0.0
        # 摇杆偏转为瞬时量，无需清零（松杆自动归零）

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the gamepad command as ``(h, vx, vy, wz)``。

        vx = 左摇杆比例项 + 方向键累加项（钳制）；wz = 左摇杆比例项；h = 右摇杆积分项；
        vy = 方向键累加项。
        """
        # 左摇杆垂直 → vx（前进映射到正上限，后退映射到负下限）
        vx_stick = self._ly * LIN_VEL_X_LIMIT[1] if self._ly >= 0 else self._ly * (-LIN_VEL_X_LIMIT[0])
        vx = _clamp(vx_stick + self._vx_dpad, *LIN_VEL_X_LIMIT)
        # 左摇杆水平 → wz（左转为正）
        wz = _clamp(self._lx * ANG_VEL_Z_LIMIT[1], *ANG_VEL_Z_LIMIT)
        vy = _clamp(self._vy_dpad, *LIN_VEL_Y_LIMIT)
        h = _clamp(self._h, *HEIGHT_OFFSET_LIMIT)
        return (h, vx, vy, wz)


def install_manual_command(env, get_command):
    """提交人工原始目标，不覆盖指令更新钩子。

    Args:
        env: 单环境的 RslRlVecEnvWrapper。
        get_command: 返回 (h, vx, vy, wz, pelvis_dx, pelvis_dy) 的零参可调用对象。

    Returns:
        每帧提交输入的闭包；骨盆偏置平滑仍由 CommandManager 每步推进一次。
    """
    term = env.unwrapped.command_manager.get_term("task_command")

    def apply() -> None:
        command = torch.tensor([get_command()], device=term.device, dtype=term.task_command.dtype)
        term.set_manual_command(command)

    apply()
    return apply


class PelvisVisualizer:
    """按键触发的骨盆地面投影 + 双脚支撑多边形可视化。

    骨盆位置直接读取根刚体 ``root_pos_w``，与任务侧 ``pelvis_error_xy`` 同源，
    不含质量加权，不受上肢扰动事件影响。

    支撑多边形取双脚足底 8 个角点 ``(±foot_length/2, ±foot_width/2)`` 投影到地面后的
    2D 凸包，足底角点定义与 :func:`mdp.rewards.feet_ground_parallel_var` 一致。骨盆地面
    投影红球为实际骨盆位置，紫球为有效目标，青球为双踝中点。
    绿线只是双脚几何投影包络，未按接触筛选，不能直接视为实际支撑域或稳定性保证。

    绘制分两层，保证在不同 Isaac Sim 版本下都能工作：

    * ``VisualizationMarkers`` 画点(骨盆投影红球 + 8 个足底角点黄球)——始终可用；
    * Isaac Sim ``debug_draw`` 画凸包边线——可选，import 失败时自动退化为只显示顶点。
    """

    def __init__(self, env, foot_length: float = 0.18, foot_width: float = 0.08):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import SPHERE_MARKER_CFG
        from isaaclab.utils.math import quat_apply

        self._quat_apply = quat_apply
        self.robot = env.unwrapped.scene["robot"]
        self._command_term = env.unwrapped.command_manager.get_term("task_command")
        device = env.unwrapped.device
        # 世界系投影平面高度：取单环境原点 z(平地假设)
        self._ground_z = float(env.unwrapped.scene.env_origins[0, 2].item())

        # 双脚 ankle_roll body 索引(左右顺序无关，凸包统一处理)
        foot_ids, foot_names = self.robot.find_bodies(".*_ankle_roll_link")
        self._foot_body_ids = foot_ids
        print(f"[INFO] 骨盆可视化就绪，支撑足 body: {foot_names}")

        # 足底四角点偏移(脚坐标系)，与 rewards.feet_ground_parallel_var 一致
        half_l, half_w = foot_length / 2.0, foot_width / 2.0
        self._corner_offsets = torch.tensor(
            [[half_l, half_w, 0.0], [half_l, -half_w, 0.0], [-half_l, half_w, 0.0], [-half_l, -half_w, 0.0]],
            device=device,
        )
        # 框架四元数为 (x,y,z,w)；球体旋转不变，仅作 visualize 占位。
        self._identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)

        # --- markers：骨盆投影(红) + 足底角点(黄) ---
        pelvis_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Pelvis/projection")
        pelvis_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        pelvis_cfg.markers["sphere"].radius = 0.008
        self._pelvis_marker = VisualizationMarkers(pelvis_cfg)
        target_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Pelvis/target")
        target_cfg.markers["sphere"].visual_material.diffuse_color = (0.7, 0.1, 1.0)
        target_cfg.markers["sphere"].radius = 0.010
        self._target_marker = VisualizationMarkers(target_cfg)
        origin_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Pelvis/origin")
        origin_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.8, 1.0)
        origin_cfg.markers["sphere"].radius = 0.004
        self._origin_marker = VisualizationMarkers(origin_cfg)

        corner_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Pelvis/foot_corners")
        corner_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.8, 0.0)
        corner_cfg.markers["sphere"].radius = 0.015
        self._corner_marker = VisualizationMarkers(corner_cfg)

        # --- debug_draw：凸包边线(兼容 Isaac Sim 4.x/5.x，失败则退化为仅顶点) ---
        self._draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
            print("[INFO] 骨盆可视化：debug_draw(isaacsim.util) 就绪，绘制支撑多边形边线。")
        except Exception:  # noqa: BLE001
            try:
                from omni.isaac.debug_draw import _debug_draw

                self._draw = _debug_draw.acquire_debug_draw_interface()
                print("[INFO] 骨盆可视化：debug_draw(omni.isaac) 就绪，绘制支撑多边形边线。")
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] debug_draw 不可用({e})，支撑多边形仅显示 8 个顶点(不连边)。")

        # 标记创建时默认可见；在首次按 C 前显式隐藏，避免原点出现占位球。
        for marker in (self._pelvis_marker, self._corner_marker, self._target_marker, self._origin_marker):
            marker.set_visibility(False)
        self._visible = False
        self._line_warned = False

    def _foot_corners_world(self) -> torch.Tensor:
        """双脚足底 8 角点世界坐标，shape (8, 3)。"""
        data = self.robot.data
        body_pos = data.body_pos_w.torch[0, self._foot_body_ids]  # (2, 3)
        body_quat = data.body_quat_w.torch[0, self._foot_body_ids]  # (2, 4)
        corners = self._quat_apply(
            body_quat.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),  # (8, 4)
            self._corner_offsets.repeat(2, 1),  # (8, 3)
        ).reshape(2, 4, 3)  # (2, 4, 3)
        corners += body_pos.unsqueeze(1)  # 平移到世界坐标
        return corners.reshape(8, 3)

    @staticmethod
    def _convex_hull_2d(points: list[list[float]]) -> list[list[float]]:
        """Andrew 单调链求 2D 凸包，返回逆时针顶点(不含重复终点)。"""
        pts = sorted({(round(x, 5), round(y, 5)) for x, y in points})
        if len(pts) <= 2:
            return [list(p) for p in pts]

        def cross(o, a, b) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[tuple] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
                lower.pop()
            lower.append(p)
        upper: list[tuple] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
                upper.pop()
            upper.append(p)
        return [list(p) for p in lower[:-1] + upper[:-1]]

    def _draw_hull_edges(self, hull: list[list[float]]) -> None:
        """用 debug_draw 画凸包闭合边线(绿色)。API 差异时静默退化并只警告一次。"""
        n = len(hull)
        if self._draw is None or n < 2:
            return
        try:
            self._draw.clear_lines()
            z = self._ground_z + 0.005  # 略抬升避免与地面 z-fighting
            # draw_lines 需 4 个等长列表：起点集、终点集、每线 RGBA、每线像素宽度
            # (点与色均以 tuple 提供，pybind 自动转 Float3 / ColorRgba)
            starts: list[tuple] = []
            ends: list[tuple] = []
            for i in range(n):
                x0, y0 = hull[i]
                x1, y1 = hull[(i + 1) % n]
                starts.append((x0, y0, z))
                ends.append((x1, y1, z))
            colors = [(0.1, 1.0, 0.1, 1.0)] * n  # 每线 RGBA
            widths = [3.0] * n  # 每线像素宽度
            self._draw.draw_lines(starts, ends, colors, widths)
        except Exception as e:  # noqa: BLE001
            if not self._line_warned:
                print(f"[WARN] 支撑多边形边线绘制失败({e})，仅显示顶点。")
                self._line_warned = True

    def update(self) -> None:
        """重算并绘制骨盆投影与支撑多边形(每帧调用)。"""
        pelvis_proj = self.robot.data.root_pos_w.torch.clone()  # (1, 3)
        pelvis_proj[:, 2] = self._ground_z + 0.01  # 稍抬升以免标记被地面遮挡
        corners_proj = self._foot_corners_world().clone()  # (8, 3)
        corners_proj[:, 2] = self._ground_z
        # 画点(首次显示时打开可见性；hidden 状态下 visualize 会被后端跳过)
        if not self._visible:
            self._pelvis_marker.set_visibility(True)
            self._corner_marker.set_visibility(True)
            self._visible = True
        self._pelvis_marker.visualize(pelvis_proj, self._identity_quat)
        self._corner_marker.visualize(corners_proj, self._identity_quat.expand(8, -1))
        active = bool(self._command_term.pelvis_tracking_mask[0])
        self._target_marker.set_visibility(active)
        self._origin_marker.set_visibility(active)
        if active:
            target = self._command_term.pelvis_target_pos_w()
            origin, _ = self._command_term.pelvis_support_frame_w()
            target[:, 2] = origin[:, 2] = self._ground_z + 0.01
            self._target_marker.visualize(target, self._identity_quat)
            self._origin_marker.visualize(origin, self._identity_quat)
        # 凸包边线
        self._draw_hull_edges(self._convex_hull_2d(corners_proj[:, :2].tolist()))

    def clear(self) -> None:
        """隐藏 markers 并清除边线(不可见时为 no-op)。"""
        if self._visible:
            self._pelvis_marker.set_visibility(False)
            self._corner_marker.set_visibility(False)
            self._target_marker.set_visibility(False)
            self._origin_marker.set_visibility(False)
            self._visible = False
        if self._draw is not None:
            with contextlib.suppress(Exception):
                self._draw.clear_lines()


# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Interactively teleoperate an RSL-RL policy checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (forced to 1 for teleop).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

# teleop 强制单环境
if args_cli.num_envs != 1:
    print(f"[INFO] Teleop 强制 num_envs=1（忽略 --num_envs={args_cli.num_envs}）。")
    args_cli.num_envs = 1

# Call an external callback if requested (register environments before parsing).
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Teleoperate a loaded RSL-RL policy with the keyboard."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # --- teleop 专属环境配置改写（须在 gym.make 之前）---
        env_cfg.scene.num_envs = 1
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        # 无限时间：移除超时终止（episode 不因 time_out 结束）
        env_cfg.terminations.time_out = None
        # 关闭观测噪声，确定性推理
        env_cfg.observations.policy.enable_corruption = False
        # 运行时由人工接口接管重采样目标，保留内部更新钩子及高度/速度可视化。
        env_cfg.commands.task_command.debug_vis = True
        # 骨盆偏置由 C 键控制的本地可视化显示，避免与指令项标记重复。
        env_cfg.commands.task_command.pelvis_debug_vis = False
        # 摔倒后原地无扰动恢复：清零 reset 事件的位姿/速度随机化
        if getattr(env_cfg.events, "reset_base", None) is not None:
            env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
            env_cfg.events.reset_base.params["velocity_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        if getattr(env_cfg.events, "reset_robot_joints", None) is not None:
            env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
            env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # --- 解析 checkpoint ---
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        # --- 创建环境 ---
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        try:
            runner.load(resume_path)
        except RuntimeError as exc:
            if "size mismatch" in str(exc):
                raise RuntimeError(
                    "策略参数尺寸不匹配：当前观测末尾增加骨盆偏置两维（默认共 102 维）。"
                    "旧 100 维 checkpoint 不能直接加载，请使用重训或显式迁移后的模型。"
                ) from exc
            raise

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        # 旧版 rsl-rl (<4.0.0) 需要 policy_nn 来 reset 循环状态
        if version.parse(installed_version) >= version.parse("4.0.0"):
            policy_nn = None
        elif version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        dt = env.unwrapped.step_dt

        # --- 键盘 / 手柄 teleop + 指令接管 ---
        pelvis_ranges = env.unwrapped.command_manager.get_term("task_command").cfg.ranges
        kb_teleop = KeyboardCommandTeleop(pelvis_ranges.pelvis_offset_x, pelvis_ranges.pelvis_offset_y)
        gp_teleop = GamepadCommandTeleop()
        pelvis_visualizer = PelvisVisualizer(env)

        def merged_command() -> tuple[float, float, float, float, float, float]:
            """高度/速度维度叠加；骨盆偏置使用键盘目标，行走时清零以防停步后重放旧偏置。"""
            kh, kvx, kvy, kwz, cx, cy = kb_teleop.as_tuple()
            gh, gvx, gvy, gwz = gp_teleop.as_tuple()
            if any(abs(v) > 1e-3 for v in (kvx + gvx, kvy + gvy, kwz + gwz)):
                kb_teleop.pelvis_dx = kb_teleop.pelvis_dy = 0.0
                cx = cy = 0.0
            return (
                _clamp(kh + gh, *HEIGHT_OFFSET_LIMIT),
                _clamp(kvx + gvx, *LIN_VEL_X_LIMIT),
                _clamp(kvy + gvy, *LIN_VEL_Y_LIMIT),
                _clamp(kwz + gwz, *ANG_VEL_Z_LIMIT),
                _clamp(cx, *pelvis_ranges.pelvis_offset_x),
                _clamp(cy, *pelvis_ranges.pelvis_offset_y),
            )

        apply_command = install_manual_command(env, merged_command)
        print(HELP_TEXT)

        # reset environment
        obs = env.get_observations()
        last_print = None

        # simulate environment（实时步进，便于交互控制）
        try:
            while True:
                start_time = time.time()
                # 手柄右摇杆按 dt 积分髋高，再把合并后的指令写入缓冲
                gp_teleop.update(dt)
                apply_command()
                with torch.inference_mode():
                    # 人工输入更新后重算观测，不推进历史/骨盆平滑时钟，避免旧观测对新目标动作。
                    obs = env.get_observations()
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                # 骨盆 / 支撑多边形可视化：C 键切换
                if kb_teleop.show_pelvis:
                    pelvis_visualizer.update()
                else:
                    pelvis_visualizer.clear()

                # 指令反馈：仅在目标变化时刷新终端显示
                cmd = merged_command()
                if cmd != last_print:
                    print(
                        f"\r[CMD] h={cmd[0]:+.2f} m | vx={cmd[1]:+.2f} vy={cmd[2]:+.2f} m/s "
                        f"| wz={cmd[3]:+.2f} rad/s | 骨盆目标=({cmd[4]:+.3f}, {cmd[5]:+.3f}) m   ",
                        end="",
                        flush=True,
                    )
                    last_print = cmd

                # 实时节流
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            env.close()
        except KeyboardInterrupt:
            print("\n[INFO] 退出 teleop。")


if __name__ == "__main__":
    main()
