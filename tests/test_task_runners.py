"""Task dispatch, task isolation and real standard PPO updates without a simulator."""

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import create_rsl_rl_runner

from g1.rl import __main__ as cli
from g1.rl.evaluation.config import load_suite, validate_suite
from g1.rl.runner import create_task_runner
from g1.rl.task_registry import TASK_FAMILIES, TaskFamily, get_task_family, resolve_task_family
from g1.tasks.upper_limb_agent.runner import UpperLimbOnPolicyRunner


@pytest.fixture
def upper_task():
    task = "G1-Test-Upper-v0"
    gym.register(
        task,
        entry_point="g1.tasks.upper_limb_agent.env:TestEnv",
        kwargs={"env_cfg_entry_point": "g1.tasks.upper_limb_agent.config.g1.env_cfg:TestCfg"},
    )
    yield task
    gym.registry.pop(task, None)


def test_common_dispatch_import_does_not_load_tasks_or_rl_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import g1.rl.runner; "
            "assert not any(x in sys.modules for x in ('g1.tasks', 'torch', 'isaaclab', 'rsl_rl'))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_upper_runner_import_does_not_load_lower_training_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import g1.tasks.upper_limb_agent.runner; "
            "assert not any(x in sys.modules for x in "
            "('g1.tasks.lower_limb_agent.runner', 'g1.rl.diagnostics', "
            "'g1.tasks.lower_limb_agent.curriculum_state', 'g1.tasks.lower_limb_agent.config.g1.env_cfg', "
            "'g1.tasks.upper_limb_agent.config.g1.env_cfg', 'g1.tasks.upper_limb_agent.mdp'))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_upper_g1_resolves_its_own_configs():
    from isaaclab_tasks.utils.hydra import resolve_task_config

    task = "Template-G1-Upper-v0"
    assert resolve_task_family(task) == "upper_limb_agent"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    assert env_cfg.task_family == "upper_limb_agent"
    assert type(env_cfg).__module__ == "g1.tasks.upper_limb_agent.config.g1.env_cfg"
    assert type(agent_cfg).__module__ == "g1.tasks.upper_limb_agent.config.g1.agents.rsl_rl_ppo_cfg"
    assert env_cfg.commands.wrist_pose.asset_name == "robot" and env_cfg.curriculum is None
    assert env_cfg.commands.wrist_pose.reference_body_name == "pelvis"
    assert env_cfg.commands.wrist_pose.body_names == ("left_wrist_yaw_link", "right_wrist_yaw_link")
    for index, side in enumerate(("left", "right")):
        orientation = getattr(env_cfg.rewards, f"{side}_orientation")
        assert orientation.weight > 0
        assert orientation.func.__name__ == "end_effector_orientation_tracking"
        assert orientation.params["arm_index"] == index
        assert 0 < orientation.params["sigma_fine"] <= orientation.params["sigma_coarse"]
        assert 0 <= orientation.params["fine_weight"] <= 1
    assert len(env_cfg.actions.joint_pos.joint_names) == 17
    assert env_cfg.actions.joint_pos.class_type.__name__ == "UpperJointPositionAction"
    assert env_cfg.rewards.joint_acc.func.__name__ == "joint_acc_l2_substeps"
    assert env_cfg.rewards.joint_acc.params == {"action_name": "joint_pos"}
    assert env_cfg.rewards.target_speed.weight == 0.0
    assert env_cfg.rewards.workspace_quality.weight == 0.0
    assert env_cfg.rewards.workspace_quality.func.__name__ == "workspace_quality_bonus"
    assert env_cfg.commands.wrist_pose.workspace_quality.position_tolerance_m == 0.01
    assert env_cfg.commands.wrist_pose.height_task is None
    assert env_cfg.rewards.target_speed.func.__name__ == "joint_target_speed_excess_l2"
    assert env_cfg.rewards.target_speed.params == {"action_name": "joint_pos", "soft_limit": 2.5}
    for name in ("joint_vel", "torque_limits"):
        penalty = getattr(env_cfg.rewards, name)
        assert penalty.weight < 0
        assert penalty.params["asset_cfg"].joint_names == env_cfg.actions.joint_pos.joint_names
        assert penalty.params["asset_cfg"].preserve_order
    assert env_cfg.rewards.joint_vel.func.__name__ == "joint_vel_l2"
    assert env_cfg.rewards.torque_limits.func.__name__ == "joint_torque_limits"
    assert env_cfg.rewards.torque_limits.params["soft_ratio"] == 0.8
    assert env_cfg.rewards.end_effector_acc.weight < 0
    assert env_cfg.rewards.end_effector_acc.func.__name__ == "end_effector_relative_lin_acc_l2_substeps"
    assert env_cfg.rewards.end_effector_acc.params == {"action_name": "joint_pos"}
    assert env_cfg.actions.joint_pos.acceleration_body_names == env_cfg.commands.wrist_pose.body_names
    assert env_cfg.actions.joint_pos.acceleration_reference_body_name == env_cfg.commands.wrist_pose.reference_body_name
    assert env_cfg.actions.joint_pos.preserve_order
    assert env_cfg.scene.robot.spawn.fix_root_link
    assert not hasattr(env_cfg.events, "reset_base")
    assert not hasattr(env_cfg, "training_phase")
    assert agent_cfg.experiment_name == "g1_upper"
    assert agent_cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}


def test_mobile_upper_owns_velocity_commands_and_upper_actions_and_uses_latest_lower_legs():
    from isaaclab_tasks.utils.hydra import resolve_task_config

    from g1.tasks.lower_limb_agent.config.g1.utils.assets import G1_29DOF_CFG as lower

    task = "Template-G1-Upper-Mobile-v0"
    env, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"])
    assert resolve_task_family(task) == "upper_limb_agent"
    assert env.actions.joint_pos.class_type.__name__ == "FrozenLowerAction"
    assert list(vars(env.actions)) == ["lower_command", "height_offset", "joint_pos"]
    assert env.actions.lower_command.command_scale == (0.3, 0.1, 0.3)
    assert env.actions.lower_command.enabled
    assert env.actions.height_offset.height_offset_range == (-0.25, 0.02)
    assert env.actions.height_offset.enabled
    assert env.commands.wrist_pose.height_task.probability == 0.30
    assert env.commands.wrist_pose.height_task.shift_range == (-0.15, -0.05)
    assert env.commands.wrist_pose.height_task.hold_episode
    assert env.rewards.action_rate.func.__name__ == "action_term_rate_l2"
    assert env.rewards.action_rate.params == {"action_name": "joint_pos"}
    assert env.rewards.action_rate.weight == -0.03
    assert env.rewards.lower_command_rate.weight == -0.10
    assert env.rewards.base_vertical_acc.weight == -0.03
    assert env.rewards.base_vertical_acc.func.__name__ == "base_vertical_acc_excess_l2_substeps"
    assert env.rewards.base_vertical_acc.params == {"action_name": "joint_pos", "soft_limit": 4.0}
    assert env.rewards.lower_command_rate.func.__name__ == "lower_command_rate_l2"
    assert env.rewards.lower_command_rate.params == {
        "action_name": "lower_command",
        "height_action_name": "height_offset",
    }
    assert env.observations.policy.actions.func.__name__ == "term_action"
    assert env.observations.policy.lower_command.func.__name__ == "lower_velocity_command"
    assert env.observations.policy.height_offset.func.__name__ == "lower_height_offset"
    from g1.tasks.upper_limb_agent.mdp.actions import UpperJointPositionAction

    assert issubclass(env.actions.joint_pos.class_type, UpperJointPositionAction)
    assert env.rewards.joint_acc.func.__name__ == "joint_acc_l2_substeps"
    assert env.rewards.joint_acc.params == {"action_name": "joint_pos"}
    assert env.rewards.end_effector_acc.func.__name__ == "end_effector_relative_lin_acc_l2_substeps"
    assert env.rewards.end_effector_acc.params == {"action_name": "joint_pos"}
    assert env.actions.joint_pos.acceleration_reference_body_name == env.commands.wrist_pose.reference_body_name
    assert env.actions.joint_pos.acceleration_body_names == env.commands.wrist_pose.body_names
    assert env.rewards.target_speed.weight == 0.0
    assert env.rewards.target_speed.params == {"action_name": "joint_pos", "soft_limit": 2.5}
    assert len(env.actions.joint_pos.joint_names) == 17
    assert not env.scene.robot.spawn.fix_root_link
    for name in ("legs", "feet"):
        assert env.scene.robot.actuators[name] is not lower.actuators[name]
        assert to_dict(env.scene.robot.actuators[name]) == to_dict(lower.actuators[name])
    assert env.scene.robot.actuators["arms"].stiffness == 40
    assert env.events.reset_root is not None and env.events.reset_joints is not None
    assert env.terminations.fallen is not None and env.terminations.low_height is not None
    assert env.rewards.termination.weight < 0
    assert env.curriculum is None
    assert agent.experiment_name == "g1_upper_mobile"


def test_mobile_height_target_overrides_are_independent_of_fixed_upper_and_lower_commands():
    from isaaclab_tasks.utils.hydra import resolve_task_config

    mobile, _ = resolve_task_config(
        "Template-G1-Upper-Mobile-v0", "rsl_rl_cfg_entry_point",
        overrides=[
            "physics=isaacsim_physx",
            "env.commands.wrist_pose.height_task.probability=0.0",
            "env.commands.wrist_pose.height_task.shift_range=[-0.30,0.10]",
            "env.commands.wrist_pose.height_task.hold_episode=false",
        ],
    )
    height = mobile.commands.wrist_pose.height_task
    assert height.probability == 0 and tuple(height.shift_range) == (-0.30, 0.10) and not height.hold_episode
    assert mobile.actions.height_offset.height_offset_range == (-0.25, 0.02)
    fixed, _ = resolve_task_config(
        "Template-G1-Upper-v0", "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"]
    )
    assert fixed.commands.wrist_pose.height_task is None
    another, _ = resolve_task_config(
        "Template-G1-Upper-Mobile-v0", "rsl_rl_cfg_entry_point", overrides=["physics=isaacsim_physx"]
    )
    assert another.commands.wrist_pose.height_task.probability == 0.30


def test_family_is_inferred_from_registration_and_explicit_selection_is_a_check(upper_task):
    assert resolve_task_family(upper_task) == "upper_limb_agent"
    assert resolve_task_family(upper_task, expected="upper_limb_agent") == "upper_limb_agent"
    for task in ("Template-G1-v0", "Template-G1-Phase2-v0", "Template-G1-Phase3-v0"):
        assert resolve_task_family(task) == "lower_limb_agent"
    with pytest.raises(ValueError, match="belongs to lower_limb_agent"):
        resolve_task_family("Template-G1-v0", expected="upper_limb_agent")
    with pytest.raises(ValueError, match="not registered"):
        resolve_task_family("G1-Unregistered-Upper-v0")
    with pytest.raises(ValueError, match="Unknown task family"):
        get_task_family("missing_task")
    with pytest.raises(ValueError, match="environment task_family"):
        create_task_runner(SimpleNamespace(unwrapped=SimpleNamespace(cfg=SimpleNamespace())), object())


def test_new_family_must_own_its_runner_factory(monkeypatch):
    monkeypatch.setitem(TASK_FAMILIES, "another_task", TaskFamily("g1.tasks.lower_limb_agent.runner:create_runner"))
    with pytest.raises(ValueError, match="must implement its runner factory"):
        get_task_family("another_task")


@pytest.mark.parametrize("entry", [None, "other_package.env:Cfg", "g1.tasks.missing_task.config:Cfg"])
def test_unknown_registration_never_defaults_to_lower_limb(monkeypatch, upper_task, entry):
    monkeypatch.setitem(gym.spec(upper_task).kwargs, "env_cfg_entry_point", entry)
    with pytest.raises(ValueError, match="recognized project task family|Unknown task family"):
        resolve_task_family(upper_task)


def test_callable_registration_is_read_without_constructing_configuration(monkeypatch, upper_task):
    class Cfg:
        def __init__(self):
            pytest.fail("Task selection must not instantiate the environment configuration")

    Cfg.__module__ = "g1.tasks.upper_limb_agent.config.g1.env_cfg"
    monkeypatch.setitem(gym.spec(upper_task).kwargs, "env_cfg_entry_point", Cfg)
    assert resolve_task_family(upper_task) == "upper_limb_agent"


def test_runner_checks_resolved_family_against_live_configuration():
    with pytest.raises(ValueError, match="differs from environment task_family"):
        create_task_runner(_UpperEnv(), _agent(), task_family="lower_limb_agent")


@pytest.mark.parametrize("action", ["train", "play"])
def test_task_selection_is_visible_in_action_help(monkeypatch, capsys, action):
    backend = SimpleNamespace(create_rsl_rl_runner=create_rsl_rl_runner, run=lambda argv: 0)
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=lambda name: backend))
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert cli.main([action, "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "--task-family" in help_text and "upper_limb_agent" in help_text


@pytest.mark.parametrize("action", ["train", "play"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("equals_form", [False, True])
def test_upper_cli_forwards_native_options_and_restores_backend(
    monkeypatch, upper_task, action, fail, explicit, equals_form
):
    original_argv = sys.argv
    task_args = [f"--task={upper_task}"] if equals_form else ["--task", upper_task]
    native = [*task_args, "--seed", "42", "env.scene.num_envs=4"]
    if action == "play":
        native.append("--train_env_cfg")
    backend = SimpleNamespace(create_rsl_rl_runner=create_rsl_rl_runner)

    def run(argv):
        assert argv == native and sys.argv[1:] == native
        assert backend.create_rsl_rl_runner.func is create_task_runner
        assert backend.create_rsl_rl_runner.keywords == {
            "task_family": "upper_limb_agent",
            "snapshot_path": None,
            "evaluation": action == "play",
        }
        if fail:
            raise RuntimeError("backend failed")

    backend.run = run
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=lambda name: backend))
    monkeypatch.setenv("WORLD_SIZE", "1")
    family_args = ["--task-family", "upper_limb_agent"] if explicit else []
    args = [action, *family_args, *native]
    if fail:
        with pytest.raises(RuntimeError, match="backend failed"):
            cli.main(args)
    else:
        assert cli.main(args) == 0
    assert sys.argv is original_argv
    assert backend.create_rsl_rl_runner is create_rsl_rl_runner


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--task-family", "upper_limb_agent", "--task", "Template-G1-v0"],
        ["train", "--task-family", "upper_limb_agent", "--curriculum-snapshot", "state.json", "--checkpoint", "x.pt"],
        ["teleop", "--task-family", "upper_limb_agent"],
        ["eval", "--task-family", "upper_limb_agent", "--plan-only"],
    ],
)
def test_unsupported_combinations_fail_before_backend_import(monkeypatch, args):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=lambda name: pytest.fail(name)))
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_automatic_family_controls_task_specific_options(monkeypatch, upper_task, capsys):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(cli, "importlib", SimpleNamespace(import_module=lambda name: pytest.fail(name)))
    with pytest.raises(SystemExit):
        cli.main(["train", "--task", upper_task, "--curriculum-snapshot", "state.json", "--checkpoint", "x.pt"])
    assert "upper_limb_agent does not support --curriculum-snapshot" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["play", "--task", "Template-G1-v0", "--train_env_cfg"])
    assert "Curriculum playback requires play_mode()" in capsys.readouterr().err


def test_eval_help_keeps_simulator_free_dispatch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from g1.rl.__main__ import main; "
            "\ntry: main(['eval', '--help'])\nexcept SystemExit as e: assert e.code == 0\n"
            "assert not any(x in sys.modules for x in ('g1.tasks', 'isaaclab', 'torch'))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_eval_and_teleop_reject_upper_environment_without_family_flag(upper_task):
    suite = load_suite("g1_core")
    suite["task"] = upper_task
    with pytest.raises(ValueError, match="eval currently supports only"):
        validate_suite(suite)
    from g1.rl.teleop import _parse_args

    with pytest.raises(SystemExit) as error:
        _parse_args(["--task", upper_task])
    assert error.value.code == 2


def _agent():
    return SimpleNamespace(
        class_name="OnPolicyRunner",
        device="cpu",
        num_steps_per_env=4,
        save_interval=1,
        logger="tensorboard",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor={
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.1},
        },
        critic={"class_name": "MLPModel", "hidden_dims": [8], "activation": "elu"},
        algorithm={"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1},
    )


class _UpperEnv:
    """Small numerical task with no lower-limb commands, phases, events or monitors."""

    num_envs, num_actions, device, max_episode_length = 4, 2, "cpu", 4

    def __init__(self):
        self.cfg = SimpleNamespace(task_family="upper_limb_agent")
        self.unwrapped = self
        self.state = torch.zeros(self.num_envs, 3)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    def get_observations(self):
        return TensorDict({"policy": self.state.clone()}, batch_size=[self.num_envs])

    def step(self, actions):
        self.state[:, :2] += actions * 0.02
        self.episode_length_buf += 1
        rewards = -self.state.square().sum(dim=1)
        dones = self.episode_length_buf >= self.max_episode_length
        self.state[dones] = 0
        self.episode_length_buf[dones] = 0
        return self.get_observations(), rewards, dones, {}


def test_upper_real_ppo_matches_upstream_and_supports_save_load_play(tmp_path, monkeypatch):
    """Identical seeds produce identical PPO weights; the task runner adds no training hooks."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    monkeypatch.setenv("WORLD_SIZE", "1")
    torch.manual_seed(42)
    reference = OnPolicyRunner(_UpperEnv(), to_dict(_agent()), device="cpu")
    reference.logger.git_status_repos = []
    reference.learn(2)

    torch.manual_seed(42)
    runner = create_task_runner(_UpperEnv(), _agent(), str(tmp_path))
    runner.logger.git_status_repos = []
    assert type(runner) is UpperLimbOnPolicyRunner
    runner.learn(2)
    for name in ("actor_state_dict", "critic_state_dict"):
        expected, actual = reference.alg.save()[name], runner.alg.save()[name]
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    checkpoint = tmp_path / "model_1.pt"
    assert checkpoint.is_file()
    assert "g1_curriculum" not in (torch.load(checkpoint, weights_only=False)["infos"] or {})
    playback = create_task_runner(_UpperEnv(), _agent(), evaluation=True)
    playback.load(checkpoint, map_location="cpu")
    obs = _UpperEnv().get_observations()
    with torch.inference_mode():
        torch.testing.assert_close(
            playback.get_inference_policy()(obs), runner.get_inference_policy()(obs), rtol=0, atol=0
        )
    tags = EventAccumulator(str(tmp_path)).Reload().Tags()["scalars"]
    assert any(tag.startswith("Loss/") for tag in tags)
    assert not any(tag.startswith(("Performance/", "TaskDistribution/")) for tag in tags)
    assert not (tmp_path / "mode_comparison").exists()


def test_mobile_runner_keeps_frozen_parameters_outside_ppo_and_checks_model_identity(tmp_path, monkeypatch):
    from g1.tasks.upper_limb_agent.frozen_lower import DEFAULT_LOWER_CHECKPOINT

    path = Path(__file__).resolve().parents[1] / DEFAULT_LOWER_CHECKPOINT
    if not path.is_file():
        pytest.skip("The selected private Lower checkpoint is not installed")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    model = torch.nn.Linear(3, 2).requires_grad_(False)
    frozen = SimpleNamespace(
        path=path,
        interface={"input_dim": 650},
        metadata={"sha256": digest, "interface": {"input_dim": 650}},
        model=model,
    )
    env = _UpperEnv()
    env.action_manager = SimpleNamespace(
        active_terms=["joint_pos"], get_term=lambda name: SimpleNamespace(frozen_policy=frozen)
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    runner = create_task_runner(env, _agent(), str(tmp_path))
    runner.logger.git_status_repos = []
    runner.learn(2)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    checkpoint = tmp_path / "model_1.pt"
    infos = torch.load(checkpoint, weights_only=False)["infos"]
    assert infos["g1_frozen_lower"]["sha256"] == digest
    assert (tmp_path / infos["g1_frozen_lower"]["artifact"]).is_file()
    runner.load(checkpoint)
    frozen.metadata["sha256"] = "different-policy"
    with pytest.raises(ValueError, match="Frozen Lower"):
        runner.load(checkpoint)
    fixed = create_task_runner(_UpperEnv(), _agent(), evaluation=True)
    with pytest.raises(ValueError, match="mobile Upper"):
        fixed.load(checkpoint)


def test_mobile_control_checkpoint_records_layout_rejects_legacy_and_allows_command_overrides(tmp_path):
    class MobileEnv(_UpperEnv):
        num_actions = 21

        def __init__(self):
            super().__init__()
            self.state = torch.zeros(self.num_envs, 107)
            self.step_dt = 1 / 60
            self.observation_manager = SimpleNamespace(group_obs_dim={"policy": (107,)})
            self.command_cfg = SimpleNamespace(command_scale=(0.3, 0.1, 0.3), enabled=True)
            # Keep a historical range to verify loading into the expanded interface below.
            self.height_cfg = SimpleNamespace(height_offset_range=(-0.15, 0.02), enabled=True)
            terms = {
                "lower_command": SimpleNamespace(cfg=self.command_cfg),
                "height_offset": SimpleNamespace(cfg=self.height_cfg),
                "joint_pos": SimpleNamespace(_joint_names=[f"upper_{i}" for i in range(17)]),
            }
            self.action_manager = SimpleNamespace(
                active_terms=["lower_command", "height_offset", "joint_pos"],
                action_term_dim=[3, 1, 17],
                get_term=terms.__getitem__,
            )

    env = MobileEnv()
    runner = create_task_runner(env, _agent(), str(tmp_path))
    checkpoint = tmp_path / "model.pt"
    runner.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    control = saved["infos"]["g1_control_interface"]
    assert control["action_dimensions"] == [3, 1, 17] and control["policy_observation_dim"] == 107
    assert control["command_scale"] == [0.3, 0.1, 0.3] and control["commands_enabled"]
    assert control["schema_version"] == 2
    assert control["height_offset_range"] == [-0.15, 0.02] and control["height_enabled"]
    env.command_cfg.command_scale, env.command_cfg.enabled = (0.2, 0.1, 0.2), False
    env.height_cfg.height_offset_range, env.height_cfg.enabled = (-0.25, 0.02), False
    runner.load(checkpoint)
    env.action_manager.active_terms.reverse()
    with pytest.raises(ValueError, match="control interface mismatch"):
        runner.load(checkpoint)
    env.action_manager.active_terms.reverse()
    saved["infos"].pop("g1_control_interface")
    legacy = tmp_path / "legacy.pt"
    torch.save(saved, legacy)
    with pytest.raises(ValueError, match="20-action/106-observation"):
        runner.load(legacy)
    saved["infos"]["g1_control_interface"] = {
        **control,
        "schema_version": 1,
        "action_dimensions": [3, 17],
        "policy_observation_dim": 106,
    }
    torch.save(saved, legacy)
    with pytest.raises(ValueError, match="control interface mismatch"):
        runner.load(legacy)
    fixed = create_task_runner(_UpperEnv(), _agent(), evaluation=True)
    with pytest.raises(ValueError, match="mobile Upper"):
        fixed.load(checkpoint)


def test_runner_archives_workspace_map_without_changing_policy_or_blocking_legacy_load(tmp_path, monkeypatch):
    import json

    from g1.tasks.upper_limb_agent.mdp.workspace_quality import WorkspaceQualityCfg, WorkspaceQualityMap

    monkeypatch.setenv("WORLD_SIZE", "1")
    env = _UpperEnv()
    quality = WorkspaceQualityMap("data/workspace_q0_v1.npz", "cpu")
    command = SimpleNamespace(
        workspace_quality_map=quality, cfg=SimpleNamespace(workspace_quality=WorkspaceQualityCfg())
    )
    env.command_manager = SimpleNamespace(active_terms=["wrist_pose"], get_term=lambda name: command)
    runner = create_task_runner(env, _agent(), str(tmp_path))
    path = tmp_path / "checkpoint.pt"
    before = {key: value.clone() for key, value in runner.alg.save()["actor_state_dict"].items()}
    runner.save(path)
    saved = torch.load(path, weights_only=False, map_location="cpu")
    metadata = saved["infos"]["g1_workspace_quality"]
    assert metadata["map_sha256"] == quality.sha256
    assert hashlib.sha256((tmp_path / metadata["artifact"]).read_bytes()).hexdigest() == quality.sha256
    assert (
        json.loads((tmp_path / metadata["artifact"]).with_suffix(".json").read_text())["q_reference"]
        == quality.q_reference
    )
    for key, value in before.items():
        torch.testing.assert_close(saved["actor_state_dict"][key], value, rtol=0, atol=0)
    # Old checkpoints lack Q metadata; their network dimensions are still valid.
    saved["infos"].pop("g1_workspace_quality")
    legacy = tmp_path / "legacy.pt"
    torch.save(saved, legacy)
    runner.load(legacy, map_location="cpu")
    (tmp_path / metadata["artifact"]).write_bytes(b"wrong map")
    with pytest.raises(ValueError, match="Archived workspace Q"):
        runner.save(tmp_path / "another.pt")
