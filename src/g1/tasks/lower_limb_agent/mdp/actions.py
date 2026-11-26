"""Lower-body target dispatch and the shared upper-body motion hook."""

import copy

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

from ..motion import UpperBodyMotion


class LowerBodyPositionAction(JointPositionAction):
    """Keep the policy action space unchanged; upper-body goals remain external."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        entity = copy.deepcopy(env.cfg.observations.policy.upper_body_target.params["asset_cfg"])
        entity.resolve(env.scene)
        upper_ids = list(entity.joint_ids)
        lower_ids = self._joint_ids
        if isinstance(lower_ids, slice) or set(upper_ids).intersection(lower_ids.tolist()):
            raise ValueError("Upper-body disturbance joints must be disjoint from policy actions")
        entity.finalize(env.device)
        env.upper_body_motion = UpperBodyMotion(env.scene[entity.name], entity.joint_ids, env.cfg.upper_body_motion)
        # Some backends integrate all decimation substeps within one apply call.
        self._motion_dt = env.step_dt if env._physics_handles_decimation else env.physics_dt
        self._delay = self._make_delay(env)

    def _make_delay(self, env):
        bounds = env.cfg.lower_action_delay_steps
        if (
            not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not 0 <= bounds[0] <= bounds[1]
        ):
            raise ValueError("lower_action_delay_steps must contain two ordered non-negative integers")
        low, high = bounds
        if high == 0:
            return None
        if env._physics_handles_decimation and env.cfg.decimation > 1:
            raise ValueError(
                "Lower action delay requires a target update at every physics substep. "
                "Use physics=isaacsim_physx or env.lower_action_delay_steps=[0,0]."
            )
        delay = DelayBuffer(high, env.num_envs, env.device)
        self._reset_target = torch.zeros_like(self.processed_actions)
        self._delay_rng = None
        self._delay_bounds = low, high
        if low == high:
            delay.set_time_lag(low)
        else:
            seed = env.cfg.seed if env.cfg.seed is not None else torch.initial_seed()
            self._delay_rng = torch.Generator(device=env.device).manual_seed(seed)
        return delay

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self._delay is None:
            return
        rows = slice(None) if env_ids is None else env_ids
        position = self._asset.data.joint_pos.torch[rows][:, self._joint_ids]
        self._delay.reset(env_ids)
        if self._delay_rng is not None:
            low, high = self._delay_bounds
            lags = self._delay.time_lags
            lags[rows] = torch.randint(
                low, high + 1, (position.shape[0],), dtype=lags.dtype, device=lags.device, generator=self._delay_rng
            )
        self._reset_target[rows] = position
        self._asset.set_joint_position_target_index(target=position, joint_ids=self._joint_ids, env_ids=env_ids)

    def completed_delay_ms(self, env_ids):
        """Capture episode samples before resetting the corresponding histories."""
        if self._delay is None:
            return None
        completed = self._delay.num_pushes[env_ids] > 0
        return self._delay.time_lags[env_ids][completed] * (self._env.physics_dt * 1000)

    def apply_actions(self):
        if self._delay is None:
            super().apply_actions()
        else:
            target = self._delay.compute(self.processed_actions)
            warming_up = self._delay.num_pushes <= self._delay.time_lags
            target = torch.where(warming_up[:, None], self._reset_target, target)
            self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)
        self._env.upper_body_motion.advance(self._motion_dt)


@configclass
class LowerBodyPositionActionCfg(JointPositionActionCfg):
    class_type: type = LowerBodyPositionAction
