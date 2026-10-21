"""Unchanged lower-body action mapping plus one shared upper-body update hook."""

import copy

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

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

    def apply_actions(self):
        super().apply_actions()
        self._env.upper_body_motion.advance(self._motion_dt)


@configclass
class LowerBodyPositionActionCfg(JointPositionActionCfg):
    class_type: type = LowerBodyPositionAction
