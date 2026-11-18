"""Torso self-contact costs with separate thresholds for arms and other partners."""

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

__all__ = ["ARM_CONTACT_BODY_PATHS", "TorsoContactPenalty", "torso_contact_termination"]

# Explicit body paths give one channel per arm/hand link. Summing force magnitudes
# keeps opposing left/right contacts from cancelling. G1 uses ThreeFinger hands.
ARM_CONTACT_BODY_PATHS = tuple(
    path
    for side in ("left", "right")
    for path in (
        *(
            f"{side}_{part}_link"
            for part in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ),
        *(
            f"{side}_hand/{side}_hand_{part}_link"
            for part in ("palm", "index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
        ),
    )
)


def torso_force_samples(env, sensor_cfg):
    """Return [environment, physics history] normal loads, newest sample first.

    Arm load sums the norm of each filtered force. The remaining vector load
    includes ground and non-arm bodies, which retain their strict threshold.
    """
    data = env.scene[sensor_cfg.name if sensor_cfg is not None else "torso_contact_pairs"].data
    pairs = data.normal_force_matrix_w_history
    if pairs is None:
        raise ValueError("Torso contact policy requires per-arm filtered force history")
    pair = pairs.torch
    net = data.net_normal_forces_w_history.torch
    if pair.shape[2] != 1 or pair.shape[-2] != len(ARM_CONTACT_BODY_PATHS):
        raise ValueError("Torso contact sensor must report one torso and one channel per configured arm/hand body")
    arm = pair.norm(dim=-1).sum(dim=(-1, -2))
    other = (net - pair.sum(dim=-2)).norm(dim=-1).amax(dim=-1)
    return arm, other


def torso_contact_termination(
    env,
    threshold: float = 20.0,
    other_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg | None = None,
):
    """Arm/torso load >20 N terminates; ground and other contacts retain >1 N."""
    if not all(math.isfinite(v) and v > 0 for v in (threshold, other_threshold)):
        raise ValueError("Torso contact thresholds must be positive and finite")
    arm, other = torso_force_samples(env, sensor_cfg)
    return (arm.amax(dim=1) > threshold) | (other.amax(dim=1) > other_threshold)


class TorsoContactPenalty(ManagerTermBase):
    """Persistent contact cost averaged over this control step's physics samples.

    At >1 N a small occupancy cost applies; it grows linearly to one at 20 N.
    RewardManager applies weight and control dt once. A 3-frame sensor history
    still uses only the latest decimation samples, so no physics frame is reused.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        soft, hard = cfg.params.get("soft_threshold", 1.0), cfg.params.get("hard_threshold", 20.0)
        occupancy = cfg.params.get("contact_cost", 0.25)
        if not (math.isfinite(soft) and math.isfinite(hard) and 0 <= soft < hard):
            raise ValueError("Torso contact reward requires 0 <= soft_threshold < hard_threshold")
        if not math.isfinite(occupancy) or not 0 < occupancy <= 1:
            raise ValueError("Torso contact reward contact_cost must be in (0,1]")
        self._sums = torch.zeros(self.num_envs, 4, device=self.device)
        self._onsets = torch.zeros(self.num_envs, device=self.device)
        self._peaks = torch.zeros(self.num_envs, 2, device=self.device)
        self._previous_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_step = -1

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg | None = None,
        soft_threshold: float = 1.0,
        hard_threshold: float = 20.0,
        contact_cost: float = 0.25,
    ):
        arm, other = torso_force_samples(env, sensor_cfg)
        substeps = env.cfg.decimation
        if arm.shape[1] < substeps:
            raise ValueError("Torso sensor history must cover every physics substep")
        arm, other = arm[:, :substeps].flip(1), other[:, :substeps].flip(1)
        contact = arm > soft_threshold
        ramp = ((arm - soft_threshold) / (hard_threshold - soft_threshold)).clamp(0, 1)
        cost = torch.where(contact, contact_cost + (1 - contact_cost) * ramp, 0.0).mean(dim=1)
        if self._last_step != env.common_step_counter:
            dt = env.cfg.sim.dt
            previous = torch.cat((self._previous_contact[:, None], contact[:, :-1]), dim=1)
            self._onsets += (contact & ~previous).sum(dim=1)
            self._previous_contact.copy_(contact[:, -1])
            self._sums[:, 0] += substeps * dt
            self._sums[:, 1] += contact.sum(dim=1) * dt
            self._sums[:, 2] += (arm > hard_threshold).sum(dim=1) * dt
            self._sums[:, 3] += (other > soft_threshold).sum(dim=1) * dt
            self._peaks[:, 0] = torch.maximum(self._peaks[:, 0], arm.amax(dim=1))
            self._peaks[:, 1] = torch.maximum(self._peaks[:, 1], other.amax(dim=1))
            self._last_step = env.common_step_counter
        return cost

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        sums = self._sums[ids]
        valid = sums[:, 0] > 0
        if valid.any():
            log = self._env.extras.setdefault("log", {})
            prefix = "Contact/torso/"
            for column, name in (
                (1, "self_contact_fraction"),
                (2, "self_force_over_20n_fraction"),
                (3, "other_contact_fraction"),
            ):
                log[prefix + name] = (sums[valid, column] / sums[valid, 0]).mean()
            log[prefix + "self_contact_onsets_per_10s"] = 10 * self._onsets[ids][valid].sum() / sums[valid, 0].sum()
            for column, name in ((0, "episode_peak_self_force_n"), (1, "episode_peak_other_force_n")):
                log[prefix + name] = self._peaks[ids][valid, column].mean()
        self._sums[ids] = 0
        self._onsets[ids] = 0
        self._peaks[ids] = 0
        self._previous_contact[ids] = False
        if env_ids is None:
            self._last_step = -1
        return {}
