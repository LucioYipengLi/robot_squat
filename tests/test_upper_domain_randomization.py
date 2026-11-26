"""Phase composition, coherent goal observations and checkpoint migration on CPU."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.runners import OnPolicyRunner
from test_upper_limb_task import pose_command as _pose_command
from test_upper_range_curriculum import FINAL

from isaaclab.utils.math import compute_pose_error, quat_error_magnitude

from isaaclab_tasks.utils.hydra import resolve_task_config

from g1.rl.failures.cli import parse_args
from g1.tasks.upper_limb_agent.checkpoint_sampling import configure_dr_checkpoint_sampling
from g1.tasks.upper_limb_agent.config.g1.domain_randomization import (
    UpperDomainRandomizationCfg,
    configure_domain_randomization,
)
from g1.tasks.upper_limb_agent.mdp.adaptive_sampling import AdaptivePlanarSamplingCfg
from g1.tasks.upper_limb_agent.mdp.observations import (
    TargetPoseNoise,
    end_effector_pose_error,
    observed_wrist_targets,
)
from g1.tasks.upper_limb_agent.mdp.range_curriculum import RangeCurriculum, RangeCurriculumCfg
from g1.tasks.upper_limb_agent.runner import UpperLimbOnPolicyRunner

TASK = "Template-G1-Upper-Mobile-v0"
AGENT = "rsl_rl_cfg_entry_point"


@pytest.mark.parametrize("phase", [1, 2, 3, 4])
def test_phase_events_and_observation_noise(phase):
    import g1.tasks  # noqa: F401

    cfg, _ = resolve_task_config(TASK, AGENT, overrides=[f"env.domain_randomization.phase={phase}"])
    nominal, _ = resolve_task_config(TASK, AGENT, play_mode=True, overrides=[])
    assert nominal.domain_randomization.phase == 1
    configure_domain_randomization(cfg)
    policy = cfg.observations.policy
    assert policy.history_length == 5
    assert list(vars(policy)) == list(vars(nominal.observations.policy))
    noisy_terms = {name: term.noise.n_max for name, term in vars(policy).items() if getattr(term, "noise", None)}
    expected = dict(joint_pos=0.01, joint_vel=1.5, base_lin_vel=0.10, base_ang_vel=0.20, projected_gravity=0.05)
    assert noisy_terms == (expected if phase >= 2 else {})
    assert policy.wrist_pose_error.params.get("noisy_target", False) == (phase >= 2)
    assert (cfg.events.upper_gains is not None) == (phase >= 3)
    if phase >= 3:
        assert len(cfg.events.upper_gains.params["asset_cfg"].joint_names) == 17
        assert cfg.events.upper_gains.params["stiffness_distribution_params"] == (0.90, 1.10)
        assert cfg.events.upper_mass.params["recompute_inertia"]
        assert cfg.events.hand_payload.params["mass_distribution_params"] == (0.0, 0.30)
        order = list(vars(cfg.events))
        assert order.index("upper_mass") < order.index("hand_payload")
    assert (cfg.events.push_robot is not None) == (phase == 4)
    if phase == 4:
        assert cfg.events.push_robot.interval_range_s == (4.0, 6.0)
        assert cfg.events.push_robot.params["velocity_range"] == {"x": (-0.30, 0.30), "y": (-0.30, 0.30)}


def test_targets_and_errors_share_bounded_held_noise_without_changing_truth():
    term = _pose_command.__wrapped__()
    env = term._env
    env.cfg = SimpleNamespace(seed=42)
    env.common_step_counter = 0
    cfg = UpperDomainRandomizationCfg(phase=2)
    noise = env._target_pose_noise = TargetPoseNoise(env, cfg)
    term.mode_id[0] = 0  # inactive right hand must keep its neutral target and zero error
    clean_goal = term.goal_pose_w.clone()
    clean_targets = term.command.reshape(-1, 2, 7)
    target = observed_wrist_targets(env, "wrist_pose").reshape(-1, 2, 7)
    error = end_effector_pose_error(env, "wrist_pose", noisy_target=True).reshape(-1, 2, 6)
    obs_position, obs_rotation = term.current_pose_b()
    pos, rot = compute_pose_error(
        obs_position.reshape(-1, 3),
        obs_rotation.reshape(-1, 4),
        target[..., :3].reshape(-1, 3),
        target[..., 3:].reshape(-1, 4),
        rot_error_type="axis_angle",
    )
    expected = torch.cat((pos, rot), dim=-1).reshape_as(error)
    torch.testing.assert_close(error, torch.where(term.active_mask[..., None], expected, 0.0))
    assert (target[term.active_mask][:, :3] - clean_targets[term.active_mask][:, :3]).norm(dim=-1).max() <= 0.003001
    delta = quat_error_magnitude(target[term.active_mask][:, 3:], clean_targets[term.active_mask][:, 3:])
    assert delta.max() <= 0.010001 and delta.max() > 0.001
    torch.testing.assert_close(target[0, 1], clean_targets[0, 1])
    torch.testing.assert_close(term.goal_pose_w, clean_goal)

    env.common_step_counter = noise.period_steps - 1
    torch.testing.assert_close(observed_wrist_targets(env, "wrist_pose"), target.flatten(1))
    noise.reset(torch.tensor([0]))
    torch.testing.assert_close(observed_wrist_targets(env, "wrist_pose")[1:], target.flatten(1)[1:])
    env.common_step_counter = noise.period_steps
    assert not torch.allclose(observed_wrist_targets(env, "wrist_pose")[1:], target.flatten(1)[1:])


@pytest.mark.parametrize(("source_phase", "phase", "clear"), [(1, 2, True), (2, 2, False), (2, 3, True), (3, 4, True)])
def test_resume_keeps_level_probability_and_only_clears_cross_phase_statistics(
    tmp_path, monkeypatch, source_phase, phase, clear
):
    cfg = RangeCurriculumCfg(planar_sampling=AdaptivePlanarSamplingCfg())
    source = RangeCurriculum(cfg, FINAL, (-0.15, -0.05), (True,) * 3, 2, "cpu", planar_probability=0.2)
    source.level = 9
    source.planar_sampling.probability = 0.45
    source.planar_sampling.history = [(2, 500.0)]
    source.ema[:] = 0.9
    source.last_check = 3600
    infos = {"g1_upper_ranges": source.state_dict(4200)}
    if source_phase != 1:
        infos["g1_upper_domain_randomization"] = {
            "settings": UpperDomainRandomizationCfg(phase=source_phase).settings()
        }
    checkpoint = tmp_path / "model.pt"
    torch.save({"infos": infos}, checkpoint)
    course = RangeCurriculum(cfg, FINAL, (-0.15, -0.05), (True,) * 3, 2, "cpu", planar_probability=0.2)
    term = SimpleNamespace(range_curriculum=course)
    raw = SimpleNamespace(
        common_step_counter=4,
        reset=lambda: None,
        cfg=SimpleNamespace(domain_randomization=UpperDomainRandomizationCfg(phase=phase)),
        command_manager=SimpleNamespace(active_terms=["wrist_pose"], get_term=lambda _: term),
    )
    runner = UpperLimbOnPolicyRunner.__new__(UpperLimbOnPolicyRunner)
    runner.env = SimpleNamespace(unwrapped=raw)
    monkeypatch.setattr(OnPolicyRunner, "load", lambda *args: None)
    runner.load(checkpoint)
    assert course.level == 9 and course.planar_probability == 0.45
    assert course.clock_offset + raw.common_step_counter == 4200
    assert course.last_check == (4200 if clear else 3600)
    assert course.planar_sampling.history == ([] if clear else [(2, 500.0)])
    assert bool(course.ema.any()) != clear


def test_preconstruction_resume_restores_retract_and_legacy_archived_focus(tmp_path):
    import g1.tasks  # noqa: F401

    env, _ = resolve_task_config(TASK, AGENT, overrides=["env.commands.wrist_pose.range_curriculum.advance=false"])
    wrist = env.commands.wrist_pose
    target = {
        **{name: getattr(wrist, name).to_dict() for name in ("left_offsets", "right_offsets", "dual_targets")},
        "mode_probabilities": wrist.mode_probabilities,
        "resampling_time_range": wrist.resampling_time_range,
        "distance_reference_speed": wrist.distance_reference_speed,
        "class_probability": wrist.height_task.probability,
        "hold_episode": wrist.height_task.hold_episode,
        "regular_retract_sampling": {"probability": 0.35, "dual_probability": 0.25},
        "planar_task": wrist.planar_task.to_dict(),
    }
    course = {
        "contract": {"cfg": {"initial_level": 0}, "height_shift": wrist.height_task.shift_range, "target": target},
        "planar_sampling": {"cfg": wrist.range_curriculum.planar_sampling.to_dict()},
    }
    checkpoint = tmp_path / "model.pt"
    torch.save({"infos": {"g1_upper_ranges": course}}, checkpoint)
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text(
        "other: !!python/object/apply:builtins.slice [null, null, null]\n"
        "commands:\n  wrist_pose:\n    planar_turn_focus: {enabled: true, wide_probability: 0.6}\n"
    )
    configure_dr_checkpoint_sampling(env, checkpoint)
    assert wrist.regular_retract_sampling.probability == 0.35
    assert wrist.regular_retract_sampling.dual_probability == 0.25
    assert wrist.planar_turn_focus.enabled and wrist.planar_turn_focus.wide_probability == 0.6
    assert not wrist.range_curriculum.advance


def test_failure_evaluation_defaults_to_nominal_and_checks_push_phase():
    args = ["mine", "--checkpoint", "model.pt"]
    assert parse_args(args).dr_phase == 1
    with pytest.raises(SystemExit):
        parse_args([*args, "--dr-push-delta-v", "0.7"])
    stress = parse_args([*args, "--dr-phase", "4", "--dr-push-delta-v", "0.7"])
    assert stress.dr_phase == 4 and stress.dr_push_delta_v == 0.7


def test_probe_can_freeze_and_unfreeze_planar_without_resetting_course():
    cfg = RangeCurriculumCfg(
        advance=False,
        freeze_planar_sampling=True,
        check_period_steps=10,
        min_episodes_per_class=1,
        planar_sampling=AdaptivePlanarSamplingCfg(history_windows=2),
    )
    course = RangeCurriculum(cfg, FINAL, (-0.15, -0.05), (True,) * 3, 3, "cpu", planar_probability=0.2)
    course.planar_sampling.probability = 0.40
    ids = torch.arange(3)
    for step in (10, 20, 30, 40):
        if step == 30:
            cfg.freeze_planar_sampling = False
        course.complete(ids, ids, torch.ones(3), torch.tensor([False, False, True]), step, torch.full((3,), 10.0))
        assert course.planar_probability == (0.40 if step < 40 else 0.45)
