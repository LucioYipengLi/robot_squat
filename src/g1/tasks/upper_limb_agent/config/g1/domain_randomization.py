"""Cumulative Upper Mobile training phases, applied before managers are created."""

import math
from dataclasses import asdict

from isaaclab.utils import configclass


@configclass
class UpperDomainRandomizationCfg:
    phase: int = 1
    joint_pos_noise_rad: float = 0.01
    joint_vel_noise_rad_s: float = 1.5
    base_lin_vel_noise_m_s: float = 0.10
    base_ang_vel_noise_rad_s: float = 0.20
    projected_gravity_noise: float = 0.05
    target_position_noise_m: float = 0.003
    target_rotation_noise_rad: float = 0.01
    target_noise_period_s: float = 0.10
    gain_scale: tuple[float, float] = (0.90, 1.10)
    mass_scale: tuple[float, float] = (0.90, 1.10)
    hand_payload_kg: tuple[float, float] = (0.0, 0.30)
    push_delta_v_m_s: float = 0.30
    push_interval_s: tuple[float, float] = (4.0, 6.0)

    def settings(self):
        """Record only settings that affect the selected phase's distribution."""
        if type(self.phase) is not int or not 1 <= self.phase <= 4:
            raise ValueError("Upper domain_randomization.phase must be an integer from 1 to 4")
        settings = asdict(self)
        if self.phase < 4:
            settings.pop("push_delta_v_m_s")
            settings.pop("push_interval_s")
        if self.phase < 3:
            settings.pop("gain_scale")
            settings.pop("mass_scale")
            settings.pop("hand_payload_kg")
        if self.phase < 2:
            return {"phase": 1}
        return settings


def configure_domain_randomization(cfg):
    """Resolve phase events after Hydra overrides, before EventManager initialization."""
    dr = cfg.domain_randomization
    settings = dr.settings()
    if dr.phase == 1:
        return
    if cfg.legacy_observations:
        raise ValueError("Upper domain randomization requires the 660-feature Mobile observation layout")
    for name, value in settings.items():
        if name == "phase":
            continue
        if isinstance(value, (tuple, list)):
            low, high = value
            strictly_positive = name != "hand_payload_kg"
            valid = math.isfinite(low) and math.isfinite(high) and 0 <= low <= high
            valid = valid and (not strictly_positive or low > 0)
        else:
            valid = math.isfinite(value) and value >= 0 and (name != "target_noise_period_s" or value > 0)
        if not valid:
            raise ValueError(f"Invalid Upper domain randomization setting {name}={value!r}")

    from isaaclab.envs import mdp as isaac_mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.utils.noise import UniformNoiseCfg

    from ...mdp.actions import UPPER_BODY_JOINT_NAMES
    from ...mdp.observations import observed_wrist_targets

    policy = cfg.observations.policy
    policy.enable_corruption = True
    for name, half_width in (
        ("joint_pos", dr.joint_pos_noise_rad),
        ("joint_vel", dr.joint_vel_noise_rad_s),
        ("base_lin_vel", dr.base_lin_vel_noise_m_s),
        ("base_ang_vel", dr.base_ang_vel_noise_rad_s),
        ("projected_gravity", dr.projected_gravity_noise),
    ):
        if half_width > 0:
            getattr(policy, name).noise = UniformNoiseCfg(n_min=-half_width, n_max=half_width)
    if dr.target_position_noise_m > 0 or dr.target_rotation_noise_rad > 0:
        policy.targets.func = observed_wrist_targets
        policy.wrist_pose_error.params["noisy_target"] = True

    if dr.phase >= 3:
        cfg.events.upper_gains = EventTermCfg(
            func=isaac_mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(UPPER_BODY_JOINT_NAMES)),
                "stiffness_distribution_params": dr.gain_scale,
                "damping_distribution_params": dr.gain_scale,
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        cfg.events.upper_mass = EventTermCfg(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["torso_link", ".*_(shoulder_.*|elbow|wrist_.*|hand_palm)_link"]
                ),
                "mass_distribution_params": dr.mass_scale,
                "operation": "scale",
                "distribution": "log_uniform",
                "recompute_inertia": True,
            },
        )
        cfg.events.hand_payload = EventTermCfg(
            func=isaac_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link", "right_hand_palm_link"]),
                "mass_distribution_params": dr.hand_payload_kg,
                "operation": "add",
                "recompute_inertia": True,
            },
        )
    if dr.phase == 4:
        delta_v = (-dr.push_delta_v_m_s, dr.push_delta_v_m_s)
        cfg.events.push_robot = EventTermCfg(
            func=isaac_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=dr.push_interval_s,
            params={"velocity_range": {"x": delta_v, "y": delta_v}},
        )
