from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .commands import UniformHeightCommand

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
