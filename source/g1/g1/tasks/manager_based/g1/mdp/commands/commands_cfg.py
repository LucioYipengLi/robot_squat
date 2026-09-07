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
    """Configuration for the unified squat-walk task command generator."""

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

    ranges: Ranges = MISSING
    """Distribution ranges for the height and velocity command dimensions."""

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
    # Scale of the velocity arrow markers, matching the built-in velocity command style.
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
