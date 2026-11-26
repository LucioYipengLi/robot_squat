"""Mobile reset-boundary metrics; the upstream simulation step stays intact."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .config.g1.domain_randomization import configure_domain_randomization


class UpperLimbMobileEnv(ManagerBasedRLEnv):
    _target_pose_noise = None

    def __init__(self, cfg, **kwargs):
        configure_domain_randomization(cfg)
        super().__init__(cfg, **kwargs)

    def load_managers(self):
        dr = self.cfg.domain_randomization
        self._target_pose_noise = None
        if dr.phase >= 2 and (dr.target_position_noise_m > 0 or dr.target_rotation_noise_rad > 0):
            from .mdp.observations import TargetPoseNoise

            self._target_pose_noise = TargetPoseNoise(self, dr)
        if self.cfg.legacy_observations:
            from .config.g1.mobile_env_cfg import LegacyMobileObservationsCfg

            self.cfg.observations = LegacyMobileObservationsCfg()
        super().load_managers()

    def reset(self, env_ids=slice(None), *, seed=None, options=None):
        return self._reset_with_history(super().reset, env_ids, seed=seed, options=options)

    def reset_to(self, state, env_ids=slice(None), seed=None, is_relative=False):
        return self._reset_with_history(super().reset_to, env_ids, state, seed=seed, is_relative=is_relative)

    def _reset_with_history(self, reset, env_ids, *args, **kwargs):
        if self.cfg.legacy_observations or env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return reset(*args, env_ids=env_ids, **kwargs)
        buffers = self.observation_manager._group_obs_term_history_buffer["policy"].values()
        if not buffers or any(buf._buffer is None for buf in buffers):
            return reset(*args, env_ids=env_ids, **kwargs)

        # The upstream explicit reset appends observations to the whole batch.
        # Preserve history for environments that have not taken a control step.
        keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        keep[env_ids] = False
        previous_policy = self.obs_buf["policy"][keep]
        history = [(buf, buf.buffer[keep], buf._num_pushes[keep]) for buf in buffers]
        try:
            obs, extras = reset(*args, env_ids=env_ids, **kwargs)
        finally:
            for buf, frames, count in history:
                buf.buffer[keep] = frames
                buf._num_pushes[keep] = count
        obs["policy"][keep] = previous_policy
        return obs, extras

    def _reset_idx(self, env_ids):
        command = self.command_manager.get_term("wrist_pose")
        command.collect_window_metrics(env_ids)
        # RewardManager.reset clears these already weighted, dt-integrated
        # sums. Snapshot per-episode rates first, before physical/manager reset.
        seconds = command.elapsed_seconds(env_ids)
        valid = seconds > 0
        rates = {}
        if valid.any():
            means = (self.reward_manager._episode_sum_buf[env_ids][valid] / seconds[valid, None]).mean(0)
            rates = {
                f"Episode_Reward/{name}": means[index] for index, name in enumerate(self.reward_manager.active_terms)
            }
        super()._reset_idx(env_ids)
        if self._target_pose_noise is not None:
            self._target_pose_noise.reset(env_ids)
        self.extras["log"].update(rates)
