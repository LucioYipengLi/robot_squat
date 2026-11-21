"""Check mirror semantics against pose errors, G1 USD kinematics and real PPO updates."""

import os
from types import SimpleNamespace

import pytest
import torch
from test_task_runners import _agent, _UpperEnv

from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz

from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import MobileObservationsCfg
from g1.tasks.upper_limb_agent.mdp.actions import MOBILE_OBSERVATION_JOINT_NAMES, UPPER_BODY_JOINT_NAMES
from g1.tasks.upper_limb_agent.runner import create_upper_limb_runner
from g1.tasks.upper_limb_agent.symmetry import (
    ACTION_TERMS,
    FULL_OBS_TERMS,
    FULL_OBSERVATION_DIM,
    OBS_HISTORY,
    OBS_TERMS,
    augment_mobile_batch,
    mirror_mobile_actions,
    mirror_mobile_policy,
    validate_mobile_mirror_interface,
)


def test_upper_history_keeps_untouched_rows_across_a_partial_reset():
    from isaaclab.managers import ObservationManager

    from g1.tasks.upper_limb_agent.env import UpperLimbMobileEnv

    cfg = MobileObservationsCfg()
    assert cfg.policy.base_height is None
    assert cfg.policy.history_length == OBS_HISTORY and cfg.policy.history_order == "term"
    for name in ("joint_pos", "joint_vel"):
        assert tuple(getattr(cfg.policy, name).params["asset_cfg"].joint_names) == MOBILE_OBSERVATION_JOINT_NAMES

    def read(env, feature):
        return env.features[feature]

    features = {name: torch.zeros(2, width) for name, width in FULL_OBS_TERMS}
    for name in features:
        term = getattr(cfg.policy, name)
        term.func, term.params, term.scale = read, {"feature": name}, None
    env = SimpleNamespace(
        cfg=SimpleNamespace(legacy_observations=False),
        num_envs=2,
        device="cpu",
        features=features,
        sim=SimpleNamespace(is_playing=lambda: True),
    )
    manager = ObservationManager(cfg, env)
    env.observation_manager = manager
    for step in range(1, 6):
        for value in features.values():
            value[:] = torch.tensor([[step], [-step]])
        env.obs_buf = manager.compute(update_history=True)
    before = env.obs_buf["policy"].clone()
    assert before.shape == (2, 660)

    def upstream_reset(*, env_ids):
        manager.reset(env_ids)
        for value in features.values():
            value[env_ids] = 100
        env.obs_buf = manager.compute(update_history=True)
        return env.obs_buf, {}

    UpperLimbMobileEnv._reset_with_history(env, upstream_reset, torch.tensor([0]))
    torch.testing.assert_close(env.obs_buf["policy"][1], before[1])
    torch.testing.assert_close(manager.compute()["policy"][1], before[1])
    torch.testing.assert_close(env.obs_buf["policy"][0], torch.full((660,), 100.0))
    for value in features.values():
        value[1] = -6
    env.obs_buf = manager.compute(update_history=True)
    first = env.obs_buf["policy"][1, :70].reshape(5, 14)
    torch.testing.assert_close(first[:, 0], torch.arange(-2, -7, -1, dtype=torch.float32))


def test_wrist_error_and_inactive_target_remain_consistent_after_mirroring():
    wrists = torch.tensor([[[0.2, 0.15, 0.1], [0.3, -0.16, 0.12]]])
    wrist_q = quat_from_euler_xyz(*torch.tensor([[0.2, 0.2], [-0.2, 0.3], [0.1, -0.4]]))
    targets = torch.tensor([[[0.28, 0.09, 0.18], [0, 0, 0]]])
    target_q = quat_from_euler_xyz(*torch.tensor([[0.08, 0], [0.24, 0], [-0.17, 0]]))
    pos, rot = compute_pose_error(wrists[0], wrist_q, targets[0], target_q, rot_error_type="axis_angle")
    errors = torch.cat((pos, rot), dim=-1)
    errors[1] = 0
    policy = torch.zeros(1, 109)
    policy[:, :14] = torch.cat((targets[0], target_q), dim=-1).flatten()
    policy[:, 14:16] = torch.tensor([1.0, 0.0])
    policy[:, 16:30] = torch.cat((wrists[0], wrist_q), dim=-1).flatten()
    policy[:, 30:42] = errors.flatten()

    reflected = mirror_mobile_policy(policy)
    assert reflected[0, 14:16].tolist() == [0.0, 1.0]
    at, aw = reflected[:, :14].reshape(2, 7), reflected[:, 16:30].reshape(2, 7)
    torch.testing.assert_close(at[0], torch.tensor([0, 0, 0, 0, 0, 0, 1.0]))
    torch.testing.assert_close(at[1, :3], torch.tensor([0.28, -0.09, 0.18]))
    pos, rot = compute_pose_error(aw[:, :3], aw[:, 3:], at[:, :3], at[:, 3:], rot_error_type="axis_angle")
    torch.testing.assert_close(reflected[0, 36:42], torch.cat((pos[1], rot[1])), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mirror_mobile_policy(reflected), policy, atol=0, rtol=0)


class _MirrorEnv(_UpperEnv):
    num_actions = 23

    def __init__(self, full=False):
        super().__init__()
        width, terms, history = (FULL_OBSERVATION_DIM, FULL_OBS_TERMS, OBS_HISTORY) if full else (109, OBS_TERMS, 1)
        self.state = torch.zeros(self.num_envs, width)
        self.cfg.observations = MobileObservationsCfg()
        self.observation_manager = SimpleNamespace(
            active_terms={"policy": [name for name, _ in terms]},
            group_obs_term_dim={"policy": [(width * history,) for _, width in terms]},
            group_obs_concatenate={"policy": True},
        )
        self.action_manager = SimpleNamespace(
            active_terms=[name for name, _ in ACTION_TERMS],
            action_term_dim=[width for _, width in ACTION_TERMS],
            get_term=lambda _: SimpleNamespace(_joint_names=UPPER_BODY_JOINT_NAMES),
        )
        bounds = torch.tensor([[-0.2, 0.05], [-0.25, 0.5], [-0.1, 0.1], [-0.2, 0.2], [-0.01, 0.02], [-0.02, 0.02]])
        self._curriculum = SimpleNamespace(initial_ranges=bounds.clone(), final_ranges=bounds)
        self.command_manager = SimpleNamespace(
            active_terms=["wrist_pose"], get_term=lambda _: SimpleNamespace(range_curriculum=self._curriculum)
        )

    def step(self, actions):
        self.state[:, :23] += actions * 0.01
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        return self.get_observations(), -self.state.square().sum(dim=1), dones, {}


@pytest.mark.parametrize("full", [False, True])
def test_opt_in_reaches_standard_ppo_update_with_original_and_mirrored_samples(full):
    env, agent = _MirrorEnv(full), _agent()
    assert create_upper_limb_runner(env, agent).alg.symmetry is None
    agent.mirror_augmentation = True
    assert create_upper_limb_runner(env, agent, evaluation=True).alg.symmetry is None
    runner = create_upper_limb_runner(env, agent)
    algorithm = runner.alg
    assert algorithm.symmetry.use_data_augmentation and not algorithm.symmetry.use_mirror_loss
    obs = env.get_observations()
    with torch.no_grad():
        for _ in range(agent.num_steps_per_env):
            actions = algorithm.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            algorithm.process_env_step(obs, rewards, dones, extras)
        algorithm.compute_returns(obs)
    weights = [p.detach().clone() for p in algorithm.actor.parameters()]
    losses = algorithm.update()
    assert all(torch.tensor(value).isfinite() for value in losses.values()) and "symmetry" in losses
    assert any(not torch.equal(old, new) for old, new in zip(weights, algorithm.actor.parameters(), strict=True))
    actions = torch.arange(1, 24, dtype=torch.float32).expand(env.num_envs, -1)
    batch, acts = augment_mobile_batch(env, obs, actions)
    assert batch.batch_size == torch.Size([2 * env.num_envs])
    torch.testing.assert_close(acts[: env.num_envs], actions)
    torch.testing.assert_close(acts[env.num_envs :, :6], actions[:, :6] * torch.tensor([1, -1, -1, 1, 1, -1]))
    torch.testing.assert_close(mirror_mobile_actions(acts[env.num_envs :]), actions, atol=0, rtol=0)
    policy = torch.arange(1, env.state.shape[-1] + 1, dtype=torch.float32).expand(env.num_envs, -1)
    torch.testing.assert_close(mirror_mobile_policy(mirror_mobile_policy(policy)), policy, atol=0, rtol=0)


def test_mirror_refuses_reordered_observations_and_asymmetric_lateral_ranges():
    env = _MirrorEnv()
    env._curriculum.final_ranges[2, 1] *= 2
    with pytest.raises(ValueError, match="symmetric vy"):
        validate_mobile_mirror_interface(env)
    env.observation_manager.active_terms["policy"][0] = "unknown"
    with pytest.raises(ValueError, match="standard 109-D"):
        validate_mobile_mirror_interface(env)


@pytest.mark.skipif(not os.environ.get("G1_COLLISION_ASSET_PATH"), reason="Opt-in: requires a local G1 USD asset")
def test_mirror_joint_signs_match_loaded_g1_usd_kinematics():
    import numpy as np
    from pxr import Usd, UsdPhysics
    from scipy.spatial.transform import Rotation

    stage = Usd.Stage.Open(os.environ["G1_COLLISION_ASSET_PATH"])
    stage.GetDefaultPrim().GetVariantSet("Physics").SetVariantSelection("SimplifiedPhysX")
    prims = {prim.GetName(): prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)}

    def joint_frame(position, orientation):
        frame = np.eye(4)
        frame[:3, :3] = Rotation.from_quat((*orientation.GetImaginary(), orientation.GetReal())).as_matrix()
        frame[:3, 3] = position
        return frame

    edges = []
    for name in MOBILE_OBSERVATION_JOINT_NAMES:
        joint = UsdPhysics.RevoluteJoint(prims[name])
        parent, child = (rel.GetTargets()[0].name for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()))
        frame_a = joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        frame_b = joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        axis = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
        edges.append((parent, child, frame_a, np.linalg.inv(frame_b), axis))

    def wrist_and_ankle_poses(angles):
        frames = {"pelvis": np.eye(4)}
        for angle, (parent, child, a, b_inv, axis) in zip(angles, edges, strict=True):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
            frames[child] = frames[parent] @ a @ rotation @ b_inv
        return np.stack(
            [frames[f"{side}_{link}"] for link in ("wrist_yaw_link", "ankle_roll_link") for side in ("left", "right")]
        )

    reflection = np.diag([1, -1, 1, 1])
    angles = torch.tensor(
        [0.34, -0.14, 0.2, 0.4, 0.23, -0.22, 0.47, 0.2, -0.25, 0.1, -0.38, -0.15, 0.29, -0.13, -0.26, 0.17, 0.2]
    )
    angles = torch.cat((angles, torch.linspace(-0.31, 0.35, 12)))
    pos_start = 5 * 42
    for joints in (torch.zeros_like(angles), angles, -0.63 * angles):
        policy = torch.zeros(FULL_OBSERVATION_DIM)
        history = policy[pos_start : pos_start + 5 * 29].reshape(5, 29)
        history.copy_(joints * torch.linspace(0.2, 1.0, 5)[:, None])
        mirrored = mirror_mobile_policy(policy)[pos_start : pos_start + 5 * 29].reshape(5, 29)
        for original, reflected in zip(history, mirrored, strict=True):
            expected = reflection @ wrist_and_ankle_poses(original.numpy())[[1, 0, 3, 2]] @ reflection
            np.testing.assert_allclose(wrist_and_ankle_poses(reflected.numpy()), expected, atol=2e-4)
