"""Mobile reset-boundary metrics; the upstream simulation step stays intact."""

from isaaclab.envs import ManagerBasedRLEnv


class UpperLimbMobileEnv(ManagerBasedRLEnv):
    def _reset_idx(self, env_ids):
        command = self.command_manager.get_term("wrist_pose")
        command.collect_window_metrics(env_ids)
        # RewardManager.reset clears these already weighted, dt-integrated
        # sums. Snapshot per-episode rates first, before physical/manager reset.
        # This is the only upstream-private buffer used by this adapter.
        seconds = command.elapsed_seconds(env_ids)
        valid = seconds > 0
        rates = {}
        if valid.any():
            means = (self.reward_manager._episode_sum_buf[env_ids][valid] / seconds[valid, None]).mean(0)
            rates = {
                f"Episode_Reward/{name}": means[index] for index, name in enumerate(self.reward_manager.active_terms)
            }
        super()._reset_idx(env_ids)
        self.extras["log"].update(rates)
