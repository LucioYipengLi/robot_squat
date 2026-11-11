"""Policy velocity/height commands followed by Upper targets and one frozen Lower inference."""

from __future__ import annotations

import copy
import math

import torch

from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationManager
from isaaclab.utils import configclass

from ..frozen_lower import DEFAULT_LOWER_CHECKPOINT, FrozenLowerPolicy, lower_actor_config
from .actions import UpperJointPositionAction, UpperJointPositionActionCfg
from .kinematics import link_origin_linear_acceleration_w

__all__ = [
    "FrozenLowerAction",
    "FrozenLowerActionCfg",
    "LowerHeightCommandAction",
    "LowerHeightCommandActionCfg",
    "LowerVelocityCommandAction",
    "LowerVelocityCommandActionCfg",
]


class LowerVelocityCommandAction(ActionTerm):
    """Decode [vx, vy, yaw rate] in the Lower policy's base frame.

    This term writes no joint targets. It must precede FrozenLowerAction, which
    consumes its current command during process_actions(). Difference costs use
    clipped normalized commands, before physical-unit scaling, once per control step.
    """

    def __init__(self, cfg: LowerVelocityCommandActionCfg, env):
        if len(cfg.command_scale) != 3 or any(not math.isfinite(v) or v <= 0 for v in cfg.command_scale):
            raise ValueError("command_scale must contain three positive finite velocity scales")
        super().__init__(cfg, env)
        self._scale = torch.tensor(cfg.command_scale, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._normalized = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.change_l2 = torch.zeros(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Effective [vx, vy, yaw rate], units m/s, m/s, rad/s."""
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions.copy_(actions)
        normalized = actions.clamp(-1, 1) if self.cfg.enabled else torch.zeros_like(actions)
        if self.cfg.enabled:
            self.change_l2.copy_((normalized - self._normalized).square().sum(-1))
        else:
            self.change_l2.zero_()
        self._normalized.copy_(normalized)
        self._processed_actions.copy_(normalized * self._scale)

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self._raw_actions, self._normalized, self._processed_actions, self.change_l2):
            value[ids] = 0


@configclass
class LowerVelocityCommandActionCfg(ActionTermCfg):
    class_type: type = LowerVelocityCommandAction
    command_scale: tuple[float, float, float] = (0.3, 0.1, 0.3)
    """Symmetric bounds for base-frame vx [m/s], vy [m/s], yaw rate [rad/s]."""
    enabled: bool = True
    """False forces effective commands and their change cost to zero, keeping three actions."""


class LowerHeightCommandAction(ActionTerm):
    """Decode a target height offset [m] and record its normalized step difference.

    Zero means the default standing height. The negative and positive sides of
    the clipped action scale to the corresponding range endpoints. Change costs
    use this normalized command, matching the velocity term. This term writes
    no joints; the shared lower_command_rate reward reads its cached change.
    """

    def __init__(self, cfg: LowerHeightCommandActionCfg, env):
        bounds = cfg.height_offset_range
        if (
            len(bounds) != 2
            or not all(math.isfinite(value) for value in bounds)
            or not bounds[0] <= 0 <= bounds[1]
            or bounds[0] >= bounds[1]
        ):
            raise ValueError("height_offset_range must be finite, ordered and contain zero")
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._normalized = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.change_l2 = torch.zeros(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Effective height offset from the robot's default root height [m]."""
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions.copy_(actions)
        low, high = self.cfg.height_offset_range
        if self.cfg.enabled:
            # A zero range endpoint disables that side: its effective command
            # stays zero, so raw changes on that side must have no change cost.
            normalized = actions.clamp(-1.0 if low < 0 else 0.0, 1.0 if high > 0 else 0.0)
            self.change_l2.copy_((normalized - self._normalized).square().sum(-1))
        else:
            normalized = torch.zeros_like(actions)
            self.change_l2.zero_()
        self._normalized.copy_(normalized)
        self._processed_actions.copy_(torch.where(normalized < 0, normalized * -low, normalized * high))

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self._raw_actions, self._normalized, self._processed_actions, self.change_l2):
            value[ids] = 0


@configclass
class LowerHeightCommandActionCfg(ActionTermCfg):
    class_type: type = LowerHeightCommandAction
    height_offset_range: tuple[float, float] = (-0.25, 0.02)
    """Target offsets from default standing height [m]; zero action maps to zero offset."""
    enabled: bool = True
    """False fixes the target offset at zero while retaining the one-dimensional action."""


class FrozenLowerAction(UpperJointPositionAction):
    """Run Lower once per environment step, apply both controllers every physics step.

    This term receives 17 Upper joint actions. A preceding command action supplies
    three velocity entries and a separate height action; pelvis xy offsets stay zero. Lower keeps
    its own previous 12 actions/history and Upper's current joint target preview.
    Base motion costs sample every physical substep without modifying either controller.
    """

    def __init__(self, cfg: FrozenLowerActionCfg, env):
        # ActionManager constructs/processes terms in configuration order. Its
        # env.action_manager attribute is not assigned until construction ends.
        configured = env.cfg.actions if isinstance(env.cfg.actions, dict) else vars(env.cfg.actions)
        names = [name for name, value in configured.items() if isinstance(value, ActionTermCfg)]
        own_names = [name for name in names if isinstance(configured[name], FrozenLowerActionCfg)]
        own_name = own_names[0] if len(own_names) == 1 else None
        command_cfg = configured.get(cfg.command_action_name)
        height_cfg = configured.get(cfg.height_action_name)
        if (
            own_name is None
            or not isinstance(command_cfg, LowerVelocityCommandActionCfg)
            or not isinstance(height_cfg, LowerHeightCommandActionCfg)
            or names.index(cfg.command_action_name) >= names.index(own_name)
            or names.index(cfg.height_action_name) >= names.index(own_name)
        ):
            raise ValueError("lower_command and height_offset must be configured before joint_pos (FrozenLowerAction)")
        super().__init__(cfg, env)
        base_ids, _ = self._asset.find_bodies(cfg.base_body_name)
        if len(base_ids) != 1:
            raise ValueError("base_body_name must match exactly one base link")
        self._base_body_id = base_ids[0]
        self._base_motion_sum = torch.zeros(self.num_envs, 4, device=self.device)
        self._base_motion_mean: torch.Tensor | None = None
        self._base_vertical_acc_samples = torch.zeros(self.num_envs, self._substep_count, device=self.device)
        from g1.tasks.lower_limb_agent.config.g1.env_cfg import LowerLimbAgentEnvCfg

        lower = LowerLimbAgentEnvCfg()
        control_dt = lower.sim.dt * lower.decimation
        if not math.isclose(env.step_dt, control_dt, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Upper and frozen Lower must use the same control period in this version")
        source = lower.actions.joint_pos
        self.lower_action = JointPositionAction(
            JointPositionActionCfg(
                asset_name=cfg.asset_name,
                joint_names=copy.deepcopy(source.joint_names),
                preserve_order=source.preserve_order,
                scale=copy.deepcopy(source.scale),
                offset=copy.deepcopy(source.offset),
                use_default_offset=source.use_default_offset,
                clip=copy.deepcopy(source.clip),
            ),
            env,
        )
        if set(self._joint_ids.tolist()).intersection(self.lower_action._joint_ids.tolist()):
            raise ValueError("Frozen Lower and Upper must control disjoint joints")
        self.lower_actions = torch.zeros(self.num_envs, self.lower_action.action_dim, device=self.device)
        self.lower_command = torch.zeros(self.num_envs, 6, device=self.device)
        group = copy.deepcopy(lower.observations.policy)
        group.enable_corruption = False
        if group.upper_body_target.params.get("target_source") != "final_goal":
            raise ValueError("Frozen Lower requires final_goal previews in the latest Lower configuration")
        entity = copy.deepcopy(group.upper_body_target.params["asset_cfg"])
        entity.resolve(env.scene)
        preview_ids = list(entity.joint_ids)
        self._preview_order = [self._joint_ids.tolist().index(joint) for joint in preview_ids]
        self._preview_default = self._asset.data.default_joint_pos.torch[:, preview_ids].clone()
        # Callbacks use controller-owned buffers, independent of env.action_manager initialization.
        group.actions.func, group.actions.params = lambda _env: self.lower_actions, {}
        group.velocity_commands.func, group.velocity_commands.params = lambda _env: self.lower_command[:, 1:4], {}
        group.pelvis_height_cmd.func, group.pelvis_height_cmd.params = lambda _env: self.lower_command[:, :1], {}
        group.pelvis_offset_cmd.func, group.pelvis_offset_cmd.params = lambda _env: self.lower_command[:, 4:6], {}
        group.upper_body_target.func = lambda _env: (
            self.processed_actions[:, self._preview_order] - self._preview_default
        )
        group.upper_body_target.params = {}
        self._processed_actions.copy_(self._offset)
        self.lower_observations = ObservationManager({"policy": group}, env)
        interface = {
            "input_dim": self.lower_observations.group_obs_dim["policy"][0],
            "action_dim": self.lower_action.action_dim,
            "history_length": group.history_length,
            "history_order": group.history_order,
            "observation_terms": self.lower_observations.active_terms["policy"],
            "control_dt": control_dt,
            "action_scale": source.scale,
            "action_joint_patterns": list(source.joint_names),
            "preserve_order": source.preserve_order,
            "use_default_offset": source.use_default_offset,
            "offset": source.offset,
            "clip": source.clip,
            "action_joint_names": self.lower_action._joint_names,
            "preview_joint_names": [self._asset.joint_names[joint] for joint in preview_ids],
            "robot_joint_names": list(self._asset.joint_names),
        }
        self.frozen_policy = FrozenLowerPolicy(cfg.checkpoint_path, lower_actor_config(), interface, self.device)
        self.last_lower_observations = torch.zeros(self.num_envs, interface["input_dim"], device=self.device)
        self._motion_sum = torch.zeros(self.num_envs, 9, device=self.device)
        self._motion_steps = torch.zeros(self.num_envs, device=self.device)

    def process_actions(self, actions):
        # Pair the current physical state with the command that produced it,
        # before publishing the next command. Reset and terminal frames are
        # excluded, matching the previous standing diagnostics convention.
        self._update_motion_metrics()
        super().process_actions(actions)
        self._base_motion_sum.zero_()
        self._base_motion_mean = None
        self._base_vertical_acc_samples.zero_()
        self.lower_command[:, :1] = self._env.action_manager.get_term(self.cfg.height_action_name).processed_actions
        self.lower_command[:, 1:4] = self._env.action_manager.get_term(self.cfg.command_action_name).processed_actions
        observations = self.lower_observations.compute_group("policy", update_history=True)
        self.last_lower_observations.copy_(observations)
        self.lower_actions.copy_(self.frozen_policy(observations))
        self.lower_action.process_actions(self.lower_actions)

    def _update_motion_metrics(self):
        data = self._asset.data
        valid = (self._env.episode_length_buf > 0).float()
        height = data.root_pos_w.torch[:, 2] - self._env.scene.env_origins[:, 2]
        target_height = data.default_root_pose.torch[:, 2] + self.lower_command[:, 0]
        height_error = (height - target_height).abs()
        speed = data.root_lin_vel_b.torch[:, :2].norm(dim=-1)
        tilt = torch.acos((-data.projected_gravity_b.torch[:, 2]).clamp(-1.0, 1.0))
        command = self.lower_command[:, 1:4]
        velocity_error = (data.root_lin_vel_b.torch[:, :2] - command[:, :2]).norm(dim=-1)
        yaw_error = (data.root_ang_vel_b.torch[:, 2] - command[:, 2]).abs()
        self._motion_sum += (
            torch.stack(
                (
                    height_error,
                    speed,
                    tilt,
                    command[:, :2].norm(dim=-1),
                    command[:, 2].abs(),
                    velocity_error,
                    yaw_error,
                    height,
                    self.lower_command[:, 0],
                ),
                dim=-1,
            )
            * valid[:, None]
        )
        self._motion_steps += valid

    def apply_actions(self):
        # Before substep k > 1, physics data contains the completed substep k - 1.
        # Shared finalization records the last substep for both base-cost families.
        if 0 < self._apply_count < self._substep_count:
            motion, vertical_acc = self._current_base_motion()
            self._base_motion_sum.add_(motion)
            self._base_vertical_acc_samples[:, self._apply_count - 1] = vertical_acc
        super().apply_actions()
        self.lower_action.apply_actions()

    def _current_base_motion_l2(self) -> torch.Tensor:
        """Read the original four squared norms without advancing sampling state."""
        return self._current_base_motion()[0]

    def _current_base_motion(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read four squared norms and signed world-Z acceleration at the base link origin."""
        data = self._asset.data
        velocity = data.body_link_vel_w.torch[:, self._base_body_id]
        linear_acc = link_origin_linear_acceleration_w(data, [self._base_body_id])[:, 0]
        angular_acc = data.body_com_acc_w.torch[:, self._base_body_id, 3:]
        motion = torch.stack(
            (
                velocity[:, :3].square().sum(-1),
                velocity[:, 3:].square().sum(-1),
                linear_acc.square().sum(-1),
                angular_acc.square().sum(-1),
            ),
            dim=-1,
        )
        return motion, linear_acc[:, 2]

    def mean_base_motion_l2(self) -> torch.Tensor:
        """Substep means of [linear velocity, angular velocity, linear acc, angular acc] squared norms.

        All vectors have three world axes and refer to base_body_name's link
        origin. Units per column: m²/s², rad²/s², m²/s⁴, rad²/s⁴. Acceleration
        is physical kinematics, without a control-step finite difference.
        RewardManager applies weight and control dt once. Repeated reads reuse
        the final sample; the initial, unstepped environment returns zeros.
        """
        if self._apply_count == 0:
            return torch.zeros_like(self._base_motion_sum)
        if self._apply_count != self._substep_count:
            raise RuntimeError(
                f"Base motion reward expected {self._substep_count} physical substeps, received {self._apply_count}"
            )
        if self._base_motion_mean is None:
            motion, vertical_acc = self._current_base_motion()
            self._base_vertical_acc_samples[:, -1] = vertical_acc
            self._base_motion_mean = (self._base_motion_sum + motion) / self._substep_count
        return self._base_motion_mean

    def mean_base_vertical_acc_excess_l2(self, soft_limit: float = 4.0) -> torch.Tensor:
        """Mean substep max(abs(world_az) / soft_limit - 1, 0)^2, dimensionless.

        Threshold each physical sample before averaging, so an early impulse or
        a signed reversal cannot cancel. Shared finalization captures the last
        substep once and preserves the original four-column base-motion API.
        """
        if not math.isfinite(soft_limit) or soft_limit <= 0:
            raise ValueError("soft_limit must be positive and finite [m/s^2]")
        if self._apply_count == 0:
            return torch.zeros_like(self._base_motion_sum[:, 0])
        self.mean_base_motion_l2()
        return (self._base_vertical_acc_samples.abs() / soft_limit - 1.0).clamp_min(0.0).square().mean(dim=-1)

    def reset(self, env_ids=None):
        # Preserve unreset environments even when motion rewards have weight zero
        # and therefore did not finalize the last sample through RewardManager.
        if env_ids is not None and self._apply_count == self._substep_count and self._base_motion_mean is None:
            self.mean_base_motion_l2()
        ids = slice(None) if env_ids is None else env_ids
        steps = self._motion_steps[ids]
        valid = steps > 0
        if valid.any():
            means = self._motion_sum[ids][valid] / steps[valid, None]
            for index, name in enumerate(
                (
                    "height_error_m",
                    "planar_speed_m_s",
                    "tilt_rad",
                    "command_speed_m_s",
                    "command_yaw_rad_s",
                    "velocity_error_m_s",
                    "yaw_error_rad_s",
                    "height_m",
                    "command_height_offset_m",
                )
            ):
                self._env.extras["log"][f"FrozenLower/{name}"] = means[:, index].mean()
        super().reset(env_ids)
        self.lower_action.reset(env_ids)
        self.lower_observations.reset(env_ids)
        self.lower_actions[ids] = 0
        self.lower_command[ids] = 0
        self.last_lower_observations[ids] = 0
        self._processed_actions[ids] = self._asset.data.default_joint_pos.torch[ids][:, self._joint_ids]
        self.lower_action._processed_actions[ids] = self._asset.data.default_joint_pos.torch[ids][
            :, self.lower_action._joint_ids
        ]
        self._motion_sum[ids] = 0
        self._motion_steps[ids] = 0
        self._base_motion_sum[ids] = 0
        self._base_vertical_acc_samples[ids] = 0
        if self._base_motion_mean is not None:
            self._base_motion_mean[ids] = 0
        if env_ids is None:
            self._base_motion_mean = None


@configclass
class FrozenLowerActionCfg(UpperJointPositionActionCfg):
    class_type: type = FrozenLowerAction
    checkpoint_path: str = DEFAULT_LOWER_CHECKPOINT
    command_action_name: str = "lower_command"
    height_action_name: str = "height_offset"
    base_body_name: str = "pelvis"
    """Link origin whose world linear/angular motion costs cover every physics substep."""
