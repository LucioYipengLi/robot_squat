from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .commands import SquatWalkCommand, UniformHeightCommand

@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for the uniform pelvis height command generator."""

    class_type: type["UniformHeightCommand"] | str = "{DIR}.commands:UniformHeightCommand"

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    resampling_time_range: tuple[float, float] = (5.0, 10.0)
    """Time before commands are changed [s]. Defaults to (5.0, 10.0).

    Mid-episode resampling forces the robot to adjust its pelvis height while standing,
    which is itself part of the balance skill. Set to ``(math.inf, math.inf)`` to keep a
    constant height per episode.
    """

    rel_default_envs: float = 0.0
    """The sampled probability of environments that should stand at the nominal pelvis height. Defaults to 0.0.

    For these environments, the height offset is forced to zero at resample time, so the
    nominal standing pose is always covered in the training distribution.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pelvis height command."""

        height_offset: tuple[float, float] = MISSING
        """Range for the pelvis height offset relative to the default standing pelvis height (in m).

        Positive values command standing higher than the default pose, negative values
        standing lower, e.g. ``(-0.1, 0.05)`` samples offsets between 10 cm below and
        5 cm above the nominal pelvis height.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the pelvis height command."""

    height_success_threshold: float = 0.03
    """Threshold on the per-episode mean pelvis height tracking error [m]. Defaults to 0.03.

    An episode is flagged as successful if its mean tracking error stays below this value.
    The episode success rate is logged under the unified ``Metrics/success_rate`` path.
    """

    goal_height_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/height_goal"
    )
    """The configuration for the goal pelvis height visualization marker. Defaults to a green
    :obj:`SPHERE_MARKER_CFG` placed at the commanded height."""

    current_height_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/height_current"
    )
    """The configuration for the current pelvis height visualization marker. Defaults to a blue
    :obj:`SPHERE_MARKER_CFG` placed at the actual pelvis height."""

    # Color and size of the visualization markers: green sphere for the commanded height,
    # blue sphere for the actual pelvis height.
    goal_height_visualizer_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
    current_height_visualizer_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
    goal_height_visualizer_cfg.markers["sphere"].radius = 0.1
    current_height_visualizer_cfg.markers["sphere"].radius = 0.1


@configclass
class SquatWalkCommandCfg(CommandTermCfg):
    """统一髋高、速度及静止模式全身 COM 偏置指令配置。"""

    class_type: type["SquatWalkCommand"] | str = "{DIR}.commands:SquatWalkCommand"

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    resampling_time_range: tuple[float, float] = (2.0, 4.0)
    """Time before commands are changed [s]. Defaults to (2.0, 4.0).

    A single resampling clock drives both command dimensions at different cadences: the
    height offset is redrawn at every resample (mid-episode height changes are part of the
    balance skill), while the velocity target is redrawn only at the episode-start resample
    (``command_counter == 0``), keeping the gait target constant within an episode.
    """

    rel_mode_envs: tuple[float, float, float, float] = (0.4, 0.3, 0.3, 0.0)
    """Sampling weights of the four per-episode task modes. Defaults to (0.4, 0.3, 0.3, 0.0).

    Order: ``(STAND, SQUAT, WALK, SQUAT_WALK)``. The weights are normalized by the sampler,
    so they need not sum to one. ``SQUAT_WALK`` (walking at a commanded non-default height)
    is a reserved slot for later curricula; keep its weight at 0.0 until the policy tracks
    both dimensions reliably.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the height and velocity command dimensions."""

        height_offset: tuple[float, float] = MISSING
        """Range for the pelvis height offset relative to the default standing pelvis height (in m).

        Sampled in SQUAT / SQUAT_WALK modes; STAND / WALK modes force a zero offset.
        """

        lin_vel_x: tuple[float, float] = (0.0, 0.0)
        """Range for the linear-x velocity command in the base frame (in m/s).

        Sampled in WALK / SQUAT_WALK modes; STAND / SQUAT modes force a zero velocity.
        """

        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        """Range for the linear-y velocity command in the base frame (in m/s)."""

        ang_vel_z: tuple[float, float] = (0.0, 0.0)
        """Range for the angular-z (yaw) velocity command in the base frame (in rad/s)."""

        com_offset_x: tuple[float, float] = (-0.01, 0.01)
        """双足水平参考系前后 COM 偏置范围 [m]；仅 STAND/SQUAT 采样。"""

        com_offset_y: tuple[float, float] = (-0.01, 0.01)
        """双足水平参考系左右 COM 偏置范围 [m]；±1 cm 为待仿真验证的保守起点。"""

    ranges: Ranges = MISSING
    """Distribution ranges for the height and velocity command dimensions."""

    com_foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    """参考点使用双踝 link 原点；不假定它们等于真实足底几何中心。"""

    com_debug_vis: bool = True
    """总 debug_vis 开启时是否显示 COM 标记；遥操作可改用 C 键控制的独立标记。"""

    com_target_speed: float = 0.02
    """COM 偏置目标的二维最大变化速率 [m/s]，不限制机器人实际 COM 速度。"""

    com_zero_probability: float = 0.2
    """静止模式采样零偏置的概率；零偏置仍要求 COM 跟踪双踝中点。"""

    com_success_threshold: float = 0.005
    """静止模式 COM 二维误差的成功阈值 [m]，用于统计而非安全保证。"""

    envelope_depth_frac: float = 0.8
    """Knee point of the height--velocity trapezoid, as a fraction of the deepest squat [0, 1].

    The conditional velocity envelope scales the walkable speed down as the commanded squat
    deepens. With squat-depth ratio ``r = h_offset / ranges.height_offset[0]`` (0 at the default
    height, 1 at the deepest sampled offset), the scale factor ``s(r)`` falls linearly from 1 at
    ``r = 0`` to a per-axis floor at ``r = envelope_depth_frac``, then is cut to 0 beyond it. The
    same fraction also caps SQUAT_WALK's height sampling to ``[envelope_depth_frac *
    height_offset[0], height_offset[1]]`` so SQUAT_WALK never enters the zero-speed cutoff band
    (its target speed stays at or above the floor, keeping it consistent with the gait-phase
    reward). SQUAT still samples the full ``height_offset`` range. Defaults to 0.8.
    """

    envelope_vx_floor: float = 0.2
    """Forward-x velocity ceiling [m/s] at the deepest walkable squat (``r = envelope_depth_frac``).

    The vx band ``ranges.lin_vel_x`` is scaled toward this absolute floor as the squat deepens; at
    ``r = envelope_depth_frac`` the sampled forward ceiling equals this value, keeping a slow but
    nonzero gait instead of shrinking into a near-immobile crawl (a pure triangle would). The floor
    is an absolute anchor: the implied scale ``s_floor = envelope_vx_floor / ranges.lin_vel_x[1]``
    is recomputed from the live ``ranges`` each resample, so widening ``lin_vel_x`` via the
    command-range curriculum keeps this floor fixed. The lower (backward) bound is scaled by the
    same ``s(r)`` to preserve the band's asymmetry. vy is intentionally NOT enveloped. Defaults to 0.2.
    """

    envelope_wz_floor: float = 0.3
    """Yaw-rate ceiling [rad/s] at the deepest walkable squat (``r = envelope_depth_frac``).

    Same trapezoid as :attr:`envelope_vx_floor` but for the yaw band ``ranges.ang_vel_z``; at
    ``r = envelope_depth_frac`` the sampled ``|yaw|`` ceiling equals this value. Both yaw bounds
    are scaled by the same ``s(r)`` (the band is symmetric). Defaults to 0.3.
    """

    height_success_threshold: float = 0.03
    """Threshold on the per-episode mean pelvis height tracking error [m]. Defaults to 0.03.

    The height-based episode success rate is logged under the unified ``Metrics/success_rate``
    path, keeping comparability with the height-only task.
    """

    vel_xy_success_threshold: float = 0.5
    """Threshold on the per-episode mean XY velocity tracking error norm [m/s]. Defaults to 0.5.

    Note: the velocity error is accumulated over all modes (zero-command modes trivially
    succeed), so the logged mean is diluted by non-walking environments. Per-mode bucketed
    metrics are a later refinement.
    """

    vel_yaw_success_threshold: float = 0.4
    """Threshold on the per-episode mean yaw velocity tracking error [rad/s]. Defaults to 0.4."""

    goal_height_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/task_height_goal"
    )
    """Green sphere marker placed at the commanded absolute pelvis height."""

    current_height_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/task_height_current"
    )
    """Blue sphere marker placed at the actual pelvis position."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/task_velocity_goal"
    )
    """Green arrow marker for the commanded XY base velocity."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/task_velocity_current"
    )
    """Blue arrow marker for the actual XY base velocity."""

    # Color and size of the height markers: green sphere for the commanded height,
    # blue sphere for the actual pelvis height.
    goal_height_visualizer_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
    current_height_visualizer_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
    goal_height_visualizer_cfg.markers["sphere"].radius = 0.1
    current_height_visualizer_cfg.markers["sphere"].radius = 0.1
    goal_com_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/task_com_goal"
    )
    """COM 有效目标地面投影（紫色），仅静止模式显示。"""

    current_com_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/task_com_current"
    )
    """整机 COM 地面投影（红色）；不是根刚体的 COM。"""

    goal_com_visualizer_cfg.markers["sphere"].visual_material.diffuse_color = (0.7, 0.1, 1.0)
    current_com_visualizer_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.1, 0.1)
    goal_com_visualizer_cfg.markers["sphere"].radius = 0.008
    current_com_visualizer_cfg.markers["sphere"].radius = 0.006

    # Scale of the velocity arrow markers, matching the built-in velocity command style.
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
