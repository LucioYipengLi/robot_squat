"""Opt-in train/save/play through the public entrypoint and real headless task scenes."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_TRAIN_PLAY_SMOKE") != "1", reason="Opt-in: launches Isaac Sim and uses a GPU")
@pytest.mark.parametrize(
    "task,family",
    [
        ("Template-G1-v0", "lower_limb_agent"),
        ("Template-G1-Upper-v0", "upper_limb_agent"),
        ("Template-G1-Upper-Mobile-v0", "upper_limb_agent"),
    ],
    ids=["lower_limb", "upper_g1", "upper_mobile"],
)
def test_headless_train_and_play_use_task_runner(tmp_path, task, family):
    project = Path(__file__).resolve().parents[1]

    def invoke(args, name):
        log = tmp_path / f"{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [sys.executable, *args],
                cwd=project,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=os.environ | {"PYTHONIOENCODING": "utf-8"},
                timeout=240,
                check=False,
            )
        assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-12000:]

    models = tmp_path / "models"
    common = [
        "--task",
        task,
        "--num_envs",
        "4",
        "--seed",
        "42",
        "--viz",
        "none",
        "physics=isaacsim_physx",
        "agent.actor.hidden_dims=[16]",
        "agent.critic.hidden_dims=[16]",
    ]
    if family == "lower_limb_agent":
        common.extend(
            [
                "env.episode_length_s=0.05",
                "env.commands.task_command.rel_mode_envs=(0,0,1,0)",
                "env.commands.task_command.ranges.lin_vel_x=(0.6,0.6)",
                "env.commands.task_command.ranges.lin_vel_y=(0,0)",
                "env.commands.task_command.ranges.ang_vel_z=(0,0)",
            ]
        )
    if family == "upper_limb_agent":
        # Exercise repeated physical/PD resets during the short rollout and playback.
        common.append("env.episode_length_s=0.05")
        common.append("env.commands.wrist_pose.resampling_time_range=[0.02,0.02]")
        common.append("env.rewards.target_speed.weight=-1e-4")
    if task == "Template-G1-Upper-Mobile-v0":
        # Guarantee coverage of episode-held targets even with only four envs.
        common.append("env.commands.wrist_pose.height_task.probability=1.0")
    invoke(
        [
            "-m",
            "g1.rl",
            "train",
            *common,
            "--max_iterations",
            "2",
            f"agent.experiment_name={models.as_posix()}",
            "agent.num_steps_per_env=4",
            "agent.algorithm.num_learning_epochs=1",
            "agent.algorithm.num_mini_batches=1",
        ],
        "train",
    )
    checkpoints = list(models.rglob("model_1.pt"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    infos = torch.load(checkpoint, weights_only=False, map_location="cpu")["infos"] or {}
    if family == "lower_limb_agent":
        state = infos["g1_curriculum"]
        assert state["family"] == "g1.lower_limb_agent" and state["phase"] == 1
        assert state["progress"]["control_steps"] == 8
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(checkpoint.parent)).Reload()
        assert events.Scalars("Episode_Reward/walk_joint_reference")
        assert events.Scalars("Episode_Reward/walk_support_timing")
        assert events.Scalars("Performance/task_command/walk/single_stance_fraction")
        assert events.Scalars("Performance/task_command/walk/hip_pitch_rms_rad")
    else:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        assert "g1_curriculum" not in infos
        events = EventAccumulator(str(checkpoint.parent)).Reload()
        tags = events.Tags()["scalars"]
        assert any(tag.startswith("Loss/") for tag in tags)
        assert not any(tag.startswith(("Performance/", "TaskDistribution/")) for tag in tags)
        for tag in (
            "Episode_Reward/joint_acc",
            "Episode_Reward/action_rate",
            "Episode_Reward/target_speed",
            "Episode_Reward/idle_arm_pose",
            "Episode_Reward/joint_vel",
            "Episode_Reward/end_effector_acc",
            "Episode_Reward/torque_limits",
        ):
            assert tag in tags
            assert all(math.isfinite(row.value) and row.value <= 0.0 for row in events.Scalars(tag))
        for tag in ("Episode_Reward/joint_vel", "Episode_Reward/end_effector_acc", "Episode_Reward/target_speed"):
            assert any(row.value < 0.0 for row in events.Scalars(tag))
        for tag in ("Episode_Reward/left_orientation", "Episode_Reward/right_orientation"):
            assert tag in tags
            assert all(math.isfinite(row.value) and row.value >= 0.0 for row in events.Scalars(tag))
        assert any(tag.startswith("Metrics/wrist_pose/") and "position_error_m" in tag for tag in tags)
        for metric in (
            "target_score",
            "actual_score",
            "position_ready_fraction",
            "map_coverage_fraction",
            "reward_available_fraction",
        ):
            tag = f"Metrics/wrist_pose/workspace_quality/{metric}"
            assert tag in tags
            assert all(math.isfinite(row.value) and 0 <= row.value <= 1.000001 for row in events.Scalars(tag))
        assert all(row.value == 0 for row in events.Scalars("Episode_Reward/workspace_quality"))
        workspace = infos["g1_workspace_quality"]
        assert workspace["position_tolerance_m"] == 0.01
        assert (checkpoint.parent / workspace["artifact"]).is_file()
        if task == "Template-G1-Upper-Mobile-v0":
            prefix = "Metrics/wrist_pose/task/height_shifted/"
            for metric in (
                "episodes",
                "failure_rate",
                "position_error_m",
                "orientation_error_rad",
                "position_ready_fraction",
                "height_shift_mean_m",
            ):
                assert prefix + metric in tags
                assert all(math.isfinite(row.value) for row in events.Scalars(prefix + metric))
            assert all(-0.150001 <= row.value <= -0.049999 for row in events.Scalars(prefix + "height_shift_mean_m"))
            vertical_tag = "Episode_Reward/base_vertical_acc"
            assert vertical_tag in tags
            assert all(math.isfinite(row.value) and row.value <= 0 for row in events.Scalars(vertical_tag))
            assert any(row.value < 0 for row in events.Scalars(vertical_tag))
            control = infos["g1_control_interface"]
            assert control["action_terms"] == ["lower_command", "height_offset", "joint_pos"]
            assert control["action_dimensions"] == [3, 1, 17] and control["policy_observation_dim"] == 107
            for metric in ("command_speed_m_s", "command_yaw_rad_s", "velocity_error_m_s", "yaw_error_rad_s"):
                assert f"FrozenLower/{metric}" in tags
            assert any(row.value < 0 for row in events.Scalars("Episode_Reward/lower_command_rate"))
            assert infos["g1_frozen_lower"]["iteration"] == 9999
            assert infos["g1_frozen_lower"]["interface"]["input_dim"] == 650
            assert (checkpoint.parent / "artifacts/frozen_lower_model.pt").is_file()
            assert (checkpoint.parent / "artifacts/params/env.yaml").is_file()
            assert all(
                f"FrozenLower/{name}" in tags
                for name in ("height_error_m", "height_m", "command_height_offset_m", "planar_speed_m_s", "tilt_rad")
            )
    # IsaacLab emits Python tuple/slice tags. Inspect the scalar node without
    # constructing unrelated configuration objects or enabling unsafe loaders.
    saved_env = yaml.compose((checkpoint.parent / "params/env.yaml").read_text(encoding="utf-8"))
    assert next(value.value for key, value in saved_env.value if key.value == "task_family") == family

    # Bound only the playback loop. Configuration, launch, model restore, policy
    # export, actual action inference/physics steps and cleanup all remain real.
    script = """
import sys
import torch
from isaaclab_rl.entrypoints.backends import play_rsl_rl
from g1.rl.__main__ import main
from g1.rl import runner as project_runner

raw_env = None
original_factory = project_runner.create_task_runner

def checked_factory(env, *args, **kwargs):
    global raw_env
    raw_env = env.unwrapped
    if raw_env.cfg.task_family == 'lower_limb_agent':
        assert env.num_actions == 12 and env.get_observations()['policy'].shape == (4, 650)
        assert raw_env.reward_manager.get_term_cfg('track_pelvis_height').weight == 2.0
        assert raw_env.reward_manager.get_term_cfg('vertical_bounce').weight == -0.5
    if raw_env.cfg.task_family == 'upper_limb_agent':
        from g1.tasks.upper_limb_agent.mdp.actions import UPPER_BODY_JOINT_NAMES
        robot = raw_env.scene['robot']
        mobile = hasattr(raw_env.action_manager.get_term('joint_pos'), 'frozen_policy')
        assert robot.is_fixed_base != mobile
        assert env.num_actions == (21 if mobile else 17)
        assert raw_env.action_manager.get_term('joint_pos')._joint_names == list(UPPER_BODY_JOINT_NAMES)
        acceleration_cfg = raw_env.reward_manager.get_term_cfg('joint_acc')
        action_ids = raw_env.action_manager.get_term('joint_pos')._joint_ids
        assert acceleration_cfg.func.__name__ == 'joint_acc_l2_substeps'
        assert acceleration_cfg.params == {'action_name': 'joint_pos'}
        assert not acceleration_cfg.func(raw_env, **acceleration_cfg.params).any()
        speed_cfg = raw_env.reward_manager.get_term_cfg('target_speed')
        assert speed_cfg.func.__name__ == 'joint_target_speed_excess_l2'
        assert speed_cfg.params == {'action_name': 'joint_pos', 'soft_limit': 2.5}
        assert not speed_cfg.func(raw_env, **speed_cfg.params).any()
        for name in ('joint_vel', 'torque_limits'):
            cfg = raw_env.reward_manager.get_term_cfg(name)
            assert cfg.params['asset_cfg'].joint_ids.tolist() == action_ids.tolist()
        assert raw_env.reward_manager.get_term_cfg('torque_limits').params['soft_ratio'] == 0.8
        observations = env.get_observations()['policy']
        assert observations.shape == (4, 107 if mobile else 93)
        assert set(raw_env.command_manager.active_terms) == {'wrist_pose'}
        command = raw_env.command_manager.get_term('wrist_pose')
        wrist_cfg = raw_env.reward_manager.get_term_cfg('end_effector_acc')
        assert wrist_cfg.func.__name__ == 'end_effector_relative_lin_acc_l2_substeps'
        assert wrist_cfg.params == {'action_name': 'joint_pos'}
        action_cfg = raw_env.action_manager.get_term('joint_pos').cfg
        assert action_cfg.acceleration_body_names == command.cfg.body_names
        assert action_cfg.acceleration_reference_body_name == command.cfg.reference_body_name
        torque_cfg = raw_env.reward_manager.get_term_cfg('torque_limits')
        for cfg in (wrist_cfg, torque_cfg):
            value = cfg.func(raw_env, **cfg.params)
            assert value.shape == (4,) and torch.isfinite(value).all() and (value >= 0).all()
        assert robot.body_names[command.reference_body_idx] == 'pelvis'
        assert tuple(robot.body_names[i] for i in command.body_indices) == command.cfg.body_names
        assert command.command.shape == (4, 14)
        assert command.active_mask.shape == (4, 2)
        assert command.active_mask.sum(dim=-1).min() >= 1
        if mobile:
            assert command.height_task_mask.all() and torch.isinf(command.time_left).all()
            assert ((command.height_shift >= -0.15) & (command.height_shift <= -0.05)).all()
        else:
            assert command.cfg.height_task is None and not command.height_task_mask.any()
        assert torch.isfinite(command.pose_error()[0]).all()
        errors = observations[:, 30:42].reshape(4, 2, 6)
        position_error, orientation_error = command.pose_error()
        torch.testing.assert_close(
            command.orientation_error_quat_squared(), torch.sin(orientation_error / 2).square(), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            errors[..., :3].norm(dim=-1), position_error * command.active_mask, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            errors[..., 3:].norm(dim=-1), orientation_error * command.active_mask, atol=1e-5, rtol=1e-5
        )
        from isaaclab.utils.math import subtract_frame_transforms
        pelvis = robot.data.body_link_pose_w.torch[:, command.reference_body_idx]
        default_pos, _ = subtract_frame_transforms(
            pelvis[:, None, :3].expand(-1, 2, -1),
            pelvis[:, None, 3:].expand(-1, 2, -1),
            command.default_wrist_pose_e[..., :3] + raw_env.scene.env_origins[:, None],
            command.default_wrist_pose_e[..., 3:],
        )
        expected = torch.tensor([[0.20, 0.15, 0.095], [0.20, -0.15, 0.095]], device=env.device)
        torch.testing.assert_close(default_pos, expected.expand(4, -1, -1), atol=0.005, rtol=0)
        original_mode = command.mode_id.clone()
        for mode in (0, 1, 2):
            command.mode_id.fill_(mode)
            command._debug_vis_callback(None)
        command.mode_id.copy_(original_mode)
        command._debug_vis_callback(None)
    return original_factory(env, *args, **kwargs)

project_runner.create_task_runner = checked_factory

def short_playback(step, **kwargs):
    fixed_pose = None
    if raw_env.cfg.task_family == 'upper_limb_agent' and raw_env.scene['robot'].is_fixed_base:
        robot = raw_env.scene['robot']
        fixed_pose = robot.data.body_link_pose_w.torch[:, 0].clone()
    with torch.inference_mode():
        for _ in range(3):
            if raw_env.cfg.task_family == 'upper_limb_agent':
                command = raw_env.command_manager.get_term('wrist_pose')
                goal = command.goal_pose_w.clone()
                length = raw_env.episode_length_buf.clone()
                shifted = command.height_task_mask.clone()
            step()
            if raw_env.cfg.task_family == 'upper_limb_agent' and command.cfg.height_task is not None:
                held = shifted & (raw_env.episode_length_buf > length)
                torch.testing.assert_close(command.goal_pose_w[held], goal[held], atol=0, rtol=0)
                assert (command.command_counter[held] == 1).all()
    if fixed_pose is not None:
        torch.testing.assert_close(robot.data.body_link_pose_w.torch[:, 0], fixed_pose)
    print('G1_DISPATCH_PLAYBACK_STEPS=3', flush=True)

play_rsl_rl.run_playback = short_playback
raise SystemExit(main(['play', *sys.argv[1:]]))
"""
    invoke(["-c", script, *common, "--checkpoint", str(checkpoint)], "play")
    assert "G1_DISPATCH_PLAYBACK_STEPS=3" in (tmp_path / "play.log").read_text(encoding="utf-8", errors="replace")
    assert (checkpoint.parent / "exported/policy.onnx").is_file()
