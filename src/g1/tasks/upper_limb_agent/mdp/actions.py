"""Joint-position control ordered from the waist through each arm, excluding fingers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .kinematics import relative_link_linear_acceleration

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

UPPER_BODY_JOINT_NAMES = WAIST_JOINT_NAMES + LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES

__all__ = [
    "LEFT_ARM_JOINT_NAMES",
    "RIGHT_ARM_JOINT_NAMES",
    "UPPER_BODY_JOINT_NAMES",
    "UpperJointPositionAction",
    "UpperJointPositionActionCfg",
    "WAIST_JOINT_NAMES",
]


class UpperJointPositionAction(JointPositionAction):
    """Keep standard position control and record target speeds and substep acceleration costs.

    Before applying substep k > 1, the previous substep's data is already fresh.
    The reward reader adds the final substep after physics, then caches the mean.
    Sampling reads controlled joints and configured link kinematics; it never
    changes articulation data, position targets or the frozen Lower clock.
    """

    def __init__(self, cfg: UpperJointPositionActionCfg, env: ManagerBasedEnv):
        self._substep_count = env.cfg.decimation
        if self._substep_count > 1 and env.sim.physics_manager.handles_decimation():
            raise ValueError(
                "Upper acceleration sampling requires one ActionTerm call per physical substep. "
                "This backend batches decimation internally; use physics=isaacsim_physx. "
                "The --viz option only selects visualization."
            )
        super().__init__(cfg, env)
        if not cfg.acceleration_body_names or len(set(cfg.acceleration_body_names)) != len(cfg.acceleration_body_names):
            raise ValueError("acceleration_body_names must contain distinct link names")
        reference_ids, _ = self._asset.find_bodies(cfg.acceleration_reference_body_name)
        self._acceleration_body_ids, _ = self._asset.find_bodies(list(cfg.acceleration_body_names), preserve_order=True)
        if len(reference_ids) != 1 or len(self._acceleration_body_ids) != len(cfg.acceleration_body_names):
            raise ValueError("Acceleration reference and body names must each match exactly one link")
        self._acceleration_reference_id = reference_ids[0]
        if self._acceleration_reference_id in self._acceleration_body_ids:
            raise ValueError("Acceleration bodies must be distinct from the reference link")
        self._joint_acc_sum = torch.zeros(self.num_envs, device=self.device)
        self._joint_acc_mean: torch.Tensor | None = None
        self._end_effector_acc_sum = torch.zeros_like(self._joint_acc_sum)
        self._end_effector_acc_mean: torch.Tensor | None = None
        self._apply_count = 0
        self._last_joint_target = self._asset.actuators.target_command.position.torch[:, self._joint_ids].clone()
        self._joint_target_velocity = torch.zeros_like(self._last_joint_target)

    @property
    def joint_target_velocity(self) -> torch.Tensor:
        """Signed processed-target change per control step [rad/s], in action joint order.

        This read-only diagnostic buffer does not limit or modify the position target.
        """
        return self._joint_target_velocity

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._joint_target_velocity.copy_((self.processed_actions - self._last_joint_target) / self._env.step_dt)
        self._last_joint_target.copy_(self.processed_actions)
        self._joint_acc_sum.zero_()
        self._joint_acc_mean = None
        self._end_effector_acc_sum.zero_()
        self._end_effector_acc_mean = None
        self._apply_count = 0

    def apply_actions(self):
        if self._apply_count >= self._substep_count:
            raise RuntimeError("Upper action applied more times than the configured decimation")
        if self._apply_count > 0:
            self._joint_acc_sum.add_(self._current_joint_acc_l2())
            self._end_effector_acc_sum.add_(self._current_end_effector_acc_l2())
        super().apply_actions()
        self._apply_count += 1

    def _current_joint_acc_l2(self) -> torch.Tensor:
        return self._asset.data.joint_acc.torch[:, self._joint_ids].square().sum(dim=-1)

    def _current_end_effector_acc_l2(self) -> torch.Tensor:
        acceleration = relative_link_linear_acceleration(
            self._asset.data, self._acceleration_body_ids, self._acceleration_reference_id
        )
        return acceleration.square().sum(dim=(1, 2))

    def mean_joint_acc_l2(self) -> torch.Tensor:
        """Mean substep sum of squared joint accelerations [rad²/s⁴], before weight and dt.

        Repeated reads within the same period reuse the final sample. Before the
        first action, the zero result allows diagnostics to inspect a fresh environment.
        """
        if self._apply_count == 0:
            return torch.zeros_like(self._joint_acc_sum)
        if self._apply_count != self._substep_count:
            raise RuntimeError(
                f"Upper acceleration reward expected {self._substep_count} physical substeps, "
                f"received {self._apply_count}"
            )
        if self._joint_acc_mean is None:
            self._joint_acc_mean = (self._joint_acc_sum + self._current_joint_acc_l2()) / self._substep_count
        return self._joint_acc_mean

    def mean_end_effector_acc_l2(self) -> torch.Tensor:
        """Mean substep sum of squared reference-frame link accelerations [m²/s⁴].

        The moving-frame derivative removes common pelvis translation/rotation.
        Both wrists contribute, including an idle wrist. This cache is separate
        from joint_acc so reading either reward does not finalize the other.
        """
        if self._apply_count == 0:
            return torch.zeros_like(self._end_effector_acc_sum)
        if self._apply_count != self._substep_count:
            raise RuntimeError(
                f"Upper acceleration reward expected {self._substep_count} physical substeps, "
                f"received {self._apply_count}"
            )
        if self._end_effector_acc_mean is None:
            self._end_effector_acc_mean = (
                self._end_effector_acc_sum + self._current_end_effector_acc_l2()
            ) / self._substep_count
        return self._end_effector_acc_mean

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # A disabled reward may not have finalized this period. Preserve the
        # other environments' last samples before clearing selected slots.
        if env_ids is not None and self._apply_count == self._substep_count and self._joint_acc_mean is None:
            self.mean_joint_acc_l2()
        if env_ids is not None and self._apply_count == self._substep_count and self._end_effector_acc_mean is None:
            self.mean_end_effector_acc_l2()
        ids = slice(None) if env_ids is None else env_ids
        self._joint_acc_sum[ids] = 0
        if self._joint_acc_mean is not None:
            self._joint_acc_mean[ids] = 0
        self._end_effector_acc_sum[ids] = 0
        if self._end_effector_acc_mean is not None:
            self._end_effector_acc_mean[ids] = 0
        # Reset events have already synchronized PD targets. The first policy
        # target is measured from that target, not from the previous episode.
        self._last_joint_target[ids] = self._asset.actuators.target_command.position.torch[ids][:, self._joint_ids]
        self._joint_target_velocity[ids] = 0
        if env_ids is None:
            self._apply_count = 0
            self._joint_acc_mean = None
            self._end_effector_acc_mean = None


@configclass
class UpperJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = UpperJointPositionAction
    acceleration_reference_body_name: str = "pelvis"
    """Moving reference frame for the end-effector acceleration reward."""
    acceleration_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    """Link origins whose relative accelerations are sampled every physics substep."""
