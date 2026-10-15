"""Task-specific pelvis and projected foot geometry for interactive playback."""

import contextlib

import torch


class PelvisVisualizer:
    """按键触发的骨盆地面投影 + 双脚支撑多边形可视化。

    骨盆位置直接读取根刚体 ``root_pos_w``，与任务侧 ``pelvis_error_xy`` 同源，
    不含质量加权，不受上肢扰动事件影响。

    支撑多边形取双脚足底 8 个角点 ``(±foot_length/2, ±foot_width/2)`` 投影到地面后的
    2D 凸包，足底角点定义与 :func:`mdp.rewards.feet_ground_parallel_var` 一致。骨盆地面
    投影红球为实际骨盆位置，紫球为有效目标，青球为双踝中点。
    绿线只是双脚几何投影包络，未按接触筛选，不能直接视为实际支撑域或稳定性保证。

    绘制分两层：

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
        foot_ids, foot_names = self.robot.find_bodies(
            list(self._command_term.cfg.support_foot_body_names), preserve_order=True
        )
        if len(foot_ids) != 2 or len(set(foot_ids)) != 2:
            raise ValueError("Pelvis visualization requires two distinct configured feet")
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
        pelvis_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/G1Teleop/projection")
        pelvis_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        pelvis_cfg.markers["sphere"].radius = 0.008
        self._pelvis_marker = VisualizationMarkers(pelvis_cfg)
        target_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/G1Teleop/target")
        target_cfg.markers["sphere"].visual_material.diffuse_color = (0.7, 0.1, 1.0)
        target_cfg.markers["sphere"].radius = 0.010
        self._target_marker = VisualizationMarkers(target_cfg)
        origin_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/G1Teleop/origin")
        origin_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.8, 1.0)
        origin_cfg.markers["sphere"].radius = 0.004
        self._origin_marker = VisualizationMarkers(origin_cfg)

        corner_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/G1Teleop/foot_corners")
        corner_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.8, 0.0)
        corner_cfg.markers["sphere"].radius = 0.015
        self._corner_marker = VisualizationMarkers(corner_cfg)

        # Current Isaac Sim debug-draw extension is optional; markers remain available.
        self._draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
            print("[INFO] 骨盆可视化：debug_draw(isaacsim.util) 就绪，绘制支撑多边形边线。")
        except (ImportError, RuntimeError) as exc:
            print(f"[WARN] debug_draw 不可用({exc})，双脚几何包络仅显示 8 个顶点。")

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
        if not self._visible:
            return
        self._pelvis_marker.set_visibility(False)
        self._corner_marker.set_visibility(False)
        self._target_marker.set_visibility(False)
        self._origin_marker.set_visibility(False)
        self._visible = False
        if self._draw is not None:
            with contextlib.suppress(Exception):
                self._draw.clear_lines()
