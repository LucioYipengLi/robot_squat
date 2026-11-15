"""Range progression, final-command costs and joint pose gating without a simulator."""

from types import SimpleNamespace

import pytest
import torch
from test_lower_command_actions import make_bridge
from test_upper_limb_task import pose_command as _pose_command

from isaaclab.managers import ActionManager, ActionTermCfg

from g1.tasks.upper_limb_agent.mdp.commands import BimanualPoseCommand
from g1.tasks.upper_limb_agent.mdp.frozen_lower_actions import (
    LowerHeightCommandAction,
    LowerHeightCommandActionCfg,
    LowerPelvisCommandAction,
    LowerPelvisCommandActionCfg,
    LowerVelocityCommandAction,
    LowerVelocityCommandActionCfg,
)
from g1.tasks.upper_limb_agent.mdp.range_curriculum import RangeCurriculum, RangeCurriculumCfg
from g1.tasks.upper_limb_agent.mdp.rewards import lower_command_magnitude_l2, lower_command_rate_l2

FINAL = ((-0.4, 0.05), (-0.5, 1.0), (-0.2, 0.2), (-0.8, 0.8), (-0.03, 0.03), (-0.03, 0.03))


@pytest.fixture
def command():
    return _pose_command.__wrapped__()


def state(**kwargs):
    return RangeCurriculum(RangeCurriculumCfg(**kwargs), FINAL, (-0.15, -0.05), (True, True), 4, "cpu")


def test_levels_are_episode_owned_and_targets_interpolate():
    c = state()
    torch.testing.assert_close(c.lower_ranges[0], torch.tensor(c.cfg.initial_lower_ranges))
    c.level = 10
    c.begin(torch.tensor([1]))
    torch.testing.assert_close(c.lower_ranges[1], torch.tensor(FINAL))
    torch.testing.assert_close(c.lower_ranges[0], torch.tensor(c.cfg.initial_lower_ranges))
    torch.testing.assert_close(c.target_scale(torch.tensor([0, 1])), torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(
        c.height_shift_ranges(torch.tensor([0, 1])), torch.tensor([[-0.05, -0.02], [-0.15, -0.05]])
    )


def test_gates_require_each_class_samples_quality_and_low_failure():
    c = state(min_episodes_per_class=1, check_period_steps=10)
    ids, categories = torch.tensor([0, 1]), torch.tensor([0, 1])
    c.complete(ids, categories, torch.tensor([1.0, 0.5]), torch.zeros(2, dtype=torch.bool), 10)
    assert c.level == 0 and not c.quality_ready
    c.complete(ids, categories, torch.ones(2), torch.tensor([False, True]), 20)
    assert c.level == 0 and not c.failure_ready
    for t in range(30, 100, 10):
        c.complete(ids, categories, torch.ones(2), torch.zeros(2, dtype=torch.bool), t)
        if c.level:
            break
    assert c.level == 1 and not c.initialized.any()
    # Already-running level-0 episodes cannot advance level 1.
    c.complete(ids, categories, torch.ones(2), torch.zeros(2, dtype=torch.bool), t + 10)
    assert c.level == 1 and not c.sample_ready
    c.begin(ids)
    c.complete(ids, categories, torch.ones(2), torch.zeros(2, dtype=torch.bool), t + 20)
    assert c.level == 2


def test_insufficient_samples_and_evaluation_do_not_advance():
    c = state(min_episodes_per_class=2, check_period_steps=10)
    c.complete(torch.tensor([0]), torch.tensor([0]), torch.ones(1), torch.zeros(1, dtype=torch.bool), 10)
    assert not c.initialized.any() and c.level == 0
    c.evaluation = True
    c.complete(torch.arange(4), torch.tensor([0, 0, 1, 1]), torch.ones(4), torch.zeros(4, dtype=torch.bool), 20)
    assert not c.counts.any() and c.level == 0


def test_state_restores_window_clock_and_rejects_changed_contract():
    c = state(check_period_steps=100)
    c.complete(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.75]), torch.tensor([True]), 40)
    saved = c.state_dict(40)
    restored = state(check_period_steps=100)
    restored.load_state_dict(saved, 3)
    assert restored.clock_offset == 37
    torch.testing.assert_close(restored.counts, torch.tensor([1, 0]))
    torch.testing.assert_close(restored.failures, torch.tensor([1, 0]))
    restored.complete(
        torch.tensor([], dtype=torch.long),
        torch.tensor([], dtype=torch.long),
        torch.tensor([]),
        torch.tensor([], dtype=torch.bool),
        63,
    )
    assert restored.last_check == 100
    with pytest.raises(ValueError, match="differs"):
        state(check_period_steps=200).load_state_dict(saved, 0)


def test_fixed_physical_references_preserve_change_costs_independently_of_magnitude():
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=1 / 60, scene={"robot": object()})
    old = LowerVelocityCommandAction(LowerVelocityCommandActionCfg(asset_name="robot"), env)
    new = LowerVelocityCommandAction(
        LowerVelocityCommandActionCfg(
            asset_name="robot", command_ranges=FINAL[1:4], reward_reference_scale=(0.3, 0.1, 0.3)
        ),
        env,
    )
    physical = torch.tensor([[0.15, -0.05, 0.15], [0.30, 0.0, -0.30], [-0.15, 0.10, 0.0]])
    old.process_actions(physical / torch.tensor([0.3, 0.1, 0.3]))
    new.process_actions(
        torch.where(physical < 0, physical / torch.tensor([0.5, 0.2, 0.8]), physical / torch.tensor([1.0, 0.2, 0.8]))
    )
    torch.testing.assert_close(new.processed_actions, old.processed_actions)
    torch.testing.assert_close(new.change_l2, old.change_l2)
    new.process_actions(torch.ones(3, 3))
    assert (new.reward_normalized > 1).all()
    env.action_manager = SimpleNamespace(get_term=lambda name: new)
    torch.testing.assert_close(lower_command_magnitude_l2(env), torch.full((3,), 3.0))
    old_h = LowerHeightCommandAction(LowerHeightCommandActionCfg(asset_name="robot"), env)
    new_h = LowerHeightCommandAction(
        LowerHeightCommandActionCfg(
            asset_name="robot", height_offset_range=(-0.4, 0.05), reward_reference_range=(-0.25, 0.02)
        ),
        env,
    )
    for h in (-0.10, 0.01, -0.25):
        old_h.process_actions(torch.full((3, 1), h / (0.25 if h < 0 else 0.02)))
        new_h.process_actions(torch.full((3, 1), h / (0.4 if h < 0 else 0.05)))
        torch.testing.assert_close(new_h.change_l2, old_h.change_l2)


def test_pelvis_vector_speed_and_partial_reset():
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=1 / 60, scene={"robot": object()})
    p = LowerPelvisCommandAction(LowerPelvisCommandActionCfg(asset_name="robot"), env)
    p.process_actions(torch.ones(3, 2))
    torch.testing.assert_close(p.processed_actions.norm(dim=-1), torch.full((3,), 0.02 / 60))
    before = p.processed_actions.clone()
    p.reset(torch.tensor([0]))
    assert not p.processed_actions[0].any()
    torch.testing.assert_close(p.processed_actions[1:], before[1:])


def test_final_command_matches_inference_observations_and_rewards(monkeypatch):
    env, bridge, calls = make_bridge(monkeypatch)
    env.step_dt = 1 / 60
    bridge.cfg.pelvis_action_name = "pelvis_offset"
    bridge.cfg.use_training_envelope = True
    bridge._update_motion_metrics = lambda: None
    infer = bridge.frozen_policy

    class Policy:
        command_training = {"envelope_depth_frac": 0.8, "envelope_vx_floor": 0.2, "envelope_wz_floor": 0.3}

        def __call__(self, observations):
            return infer(observations)

    bridge.frozen_policy = Policy()
    bridge._training_ranges = torch.tensor(((-0.405, 0.05), *FINAL[1:]))
    bridge._training_positive = (1.0, 0.8)
    bridge.lower_observations = SimpleNamespace(
        compute_group=lambda *a, **k: torch.cat(
            (bridge.lower_command, bridge.processed_actions, torch.zeros(3, 627)), -1
        )
    )
    env.action_manager = ActionManager(
        SimpleNamespace(
            lower_command=LowerVelocityCommandActionCfg(
                asset_name="robot",
                command_ranges=FINAL[1:4],
                reward_reference_scale=(0.3, 0.1, 0.3),
                range_command_name="wrist_pose",
            ),
            height_offset=LowerHeightCommandActionCfg(
                asset_name="robot",
                height_offset_range=FINAL[0],
                reward_reference_range=(-0.25, 0.02),
                range_command_name="wrist_pose",
            ),
            pelvis_offset=LowerPelvisCommandActionCfg(asset_name="robot", range_command_name="wrist_pose"),
            joint_pos=ActionTermCfg(asset_name="robot", class_type=lambda cfg, env: bridge),
        ),
        env,
    )
    a = torch.ones(3, 23)
    # Directly exercise the publishing stage before the frozen actor call.
    for name, values in (("lower_command", a[:, :3]), ("height_offset", a[:, 3:4]), ("pelvis_offset", a[:, 4:6])):
        env.action_manager.get_term(name).process_actions(values)
    bridge._publish_commands()
    assert not bridge.lower_command[:, 4:6].any()  # Moving: pelvis requests are disabled.
    for name, sl in (("lower_command", slice(1, 4)), ("height_offset", slice(0, 1)), ("pelvis_offset", slice(4, 6))):
        torch.testing.assert_close(env.action_manager.get_term(name).processed_actions, bridge.lower_command[:, sl])
    snapshot = [
        env.action_manager.get_term(n).change_l2.clone() for n in ("lower_command", "height_offset", "pelvis_offset")
    ]
    torch.testing.assert_close(lower_command_rate_l2(env, pelvis_action_name="pelvis_offset"), sum(snapshot))
    # The yaw envelope reduces the charged magnitude to the command Lower receives.
    before = bridge.lower_command.clone()
    expected_magnitude = (before[:, 1:4] / torch.tensor([1.0, 0.2, 0.8])).square().sum(-1)
    torch.testing.assert_close(lower_command_magnitude_l2(env), expected_magnitude)
    torch.testing.assert_close(bridge.lower_command, before)
    torch.testing.assert_close(lower_command_rate_l2(env, pelvis_action_name="pelvis_offset"), sum(snapshot))
    # Deep crouching forces all velocity components to zero and permits gradual pelvis control.
    env.action_manager.get_term("height_offset").process_actions(-torch.ones(3, 1))
    env.action_manager.get_term("pelvis_offset").process_actions(torch.ones(3, 2))
    bridge._publish_commands()
    assert not bridge.lower_command[:, 1:4].any()
    assert not lower_command_magnitude_l2(env).any()
    assert bridge.lower_command[:, 4:6].norm(dim=-1).max() <= 0.02 / 60 + 1e-8
    assert torch.isfinite(infer(bridge.lower_observations.compute_group())).all() and calls
    # Real manager order: new limits and all six commands reach inference in this step.
    curriculum = RangeCurriculum(RangeCurriculumCfg(), FINAL, (-0.15, -0.05), (True, True), 3, "cpu")
    env.command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(range_curriculum=curriculum))
    before = len(calls)
    env.action_manager.process_action(a)
    assert len(calls) == before + 1
    torch.testing.assert_close(calls[-1][:, :6], bridge.lower_command)
    assert (bridge.lower_command[:, 1] <= 0.100001).all()
    curriculum.level = 10
    curriculum.begin(torch.tensor([0]))
    env.action_manager.process_action(a)
    assert bridge.lower_command[0, 1] == 1 and bridge.lower_command[1, 1] <= 0.100001


def make_episode_actions():
    """Three environments retain different curriculum levels while decoding the same action."""
    curriculum = RangeCurriculum(RangeCurriculumCfg(), FINAL, (-0.15, -0.05), (True, True), 3, "cpu")
    for i, level in enumerate((0, 5, 10)):
        curriculum.level = level
        curriculum.begin(torch.tensor([i]))
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=1 / 60,
        scene={"robot": object()},
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(range_curriculum=curriculum)),
    )
    velocity = LowerVelocityCommandAction(
        LowerVelocityCommandActionCfg(
            asset_name="robot",
            command_ranges=FINAL[1:4],
            reward_reference_scale=(0.3, 0.1, 0.3),
            range_command_name="wrist_pose",
        ),
        env,
    )
    height = LowerHeightCommandAction(
        LowerHeightCommandActionCfg(
            asset_name="robot",
            height_offset_range=FINAL[0],
            reward_reference_range=(-0.25, 0.02),
            range_command_name="wrist_pose",
        ),
        env,
    )
    pelvis = LowerPelvisCommandAction(
        LowerPelvisCommandActionCfg(asset_name="robot", range_command_name="wrist_pose"), env
    )
    return curriculum, velocity, height, pelvis


@pytest.mark.parametrize("axis", [0, 1, 2, 3, 4, 5])
def test_entire_raw_interval_scales_into_each_episode_range(axis):
    c, v, h, p = make_episode_actions()
    term = (h, v, v, v, p, p)[axis]
    column = (0, 0, 1, 2, 0, 1)[axis]
    dimension = term.action_dim
    last = None
    for action in (-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0):
        term.process_actions(torch.full((3, dimension), action))
        value = max(-1, min(action, 1))
        scale = -c.lower_ranges[:, axis, 0] if value < 0 else c.lower_ranges[:, axis, 1]
        expected = scale * value
        torch.testing.assert_close(term.requested_actions[:, column], expected)
        if last is not None and -1 < action <= 1:
            assert (term.requested_actions[:, column] > last).all()
        last = term.requested_actions[:, column].clone()
    # Same raw action produces small continuous targets, including at level zero.
    term.process_actions(torch.full((3, dimension), 0.1))
    torch.testing.assert_close(term.requested_actions[:, column], c.lower_ranges[:, axis, 1] * 0.1)


def test_identical_physical_sequences_keep_change_costs_but_magnitude_tracks_episode_ranges():
    c, v, h, p = make_episode_actions()
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: v))
    for factor in (1.0, -0.5, 0.0):
        velocity = torch.tensor([0.03, 0.01, 0.04]).expand(3, -1) * factor
        bounds = c.lower_ranges[:, 1:4]
        scale = torch.where(velocity < 0, -bounds[..., 0], bounds[..., 1])
        v.process_actions(velocity / scale)
        torch.testing.assert_close(v.processed_actions, velocity)
        torch.testing.assert_close(v.change_l2, v.change_l2[0].expand(3))
        torch.testing.assert_close(v.reward_normalized, velocity / torch.tensor([0.3, 0.1, 0.3]))
        magnitude = lower_command_magnitude_l2(env)
        if factor:
            assert magnitude[0] > magnitude[1] > magnitude[2]
        else:
            assert not magnitude.any()
        height = torch.full((3, 1), -0.05 if factor > 0 else (0.005 if factor < 0 else 0.0))
        bounds = c.lower_ranges[:, :1]
        h.process_actions(height / torch.where(height < 0, -bounds[..., 0], bounds[..., 1]))
        torch.testing.assert_close(h.processed_actions, height)
        torch.testing.assert_close(h.change_l2, h.change_l2[0].expand(3))
        offset = torch.full((3, 2), 0.001 * factor)
        bounds = c.lower_ranges[:, 4:6]
        raw = offset / torch.where(offset < 0, -bounds[..., 0], bounds[..., 1])
        for _ in range(10):
            p.process_actions(raw)
            torch.testing.assert_close(p.change_l2, p.change_l2[0].expand(3), atol=1e-8, rtol=1e-5)
        torch.testing.assert_close(p.processed_actions, offset)


@pytest.mark.parametrize("fraction", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_magnitude_charges_identical_relative_use_across_episode_levels_and_signs(fraction):
    c, v, _, _ = make_episode_actions()
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: v))
    v.process_actions(torch.full((3, 3), fraction))
    commands, rate = v.processed_actions.clone(), v.change_l2.clone()
    expected = torch.full((3,), 3 * fraction**2)
    torch.testing.assert_close(lower_command_magnitude_l2(env), expected)
    # Advancing the global level alone leaves all ongoing episodes' costs intact.
    c.level = 7
    torch.testing.assert_close(lower_command_magnitude_l2(env), expected)
    torch.testing.assert_close(lower_command_magnitude_l2(env), expected)
    torch.testing.assert_close(v.processed_actions, commands, atol=0, rtol=0)
    torch.testing.assert_close(v.change_l2, rate, atol=0, rtol=0)


@pytest.mark.parametrize("direction", [-1, 1])
def test_magnitude_handles_zero_width_episode_directions(direction):
    c, v, _, _ = make_episode_actions()
    c.lower_ranges[0, 1:4] = 0
    c.lower_ranges[1, 1:4, 0 if direction < 0 else 1] = 0
    v.process_actions(torch.full((3, 3), float(direction)))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: v))
    torch.testing.assert_close(lower_command_magnitude_l2(env), torch.tensor([0.0, 0.0, 3.0]))


def test_episode_decoder_observes_partial_reset_level_changes_and_disabled_terms():
    c, v, h, p = make_episode_actions()
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: v))
    v.process_actions(torch.full((3, 3), 0.5))
    previous = v.processed_actions.clone()
    c.level = 1
    c.begin(torch.tensor([0]))
    v.reset(torch.tensor([0]))
    torch.testing.assert_close(v.processed_actions[1:], previous[1:])
    torch.testing.assert_close(lower_command_magnitude_l2(env), torch.tensor([0.0, 0.75, 0.75]))
    v.process_actions(torch.full((3, 3), 0.5))
    torch.testing.assert_close(v.processed_actions, c.lower_ranges[:, 1:4, 1] * 0.5)
    torch.testing.assert_close(lower_command_magnitude_l2(env), torch.full((3,), 0.75))
    for term in (v, h, p):
        term.cfg.enabled = False
        term.process_actions(torch.full((3, term.action_dim), 20.0))
        assert not term.processed_actions.any() and not term.change_l2.any()
    assert not lower_command_magnitude_l2(env).any()


def test_joint_pose_gate_is_simultaneous_and_idle_arm_is_ignored(command):
    term = command
    term.range_curriculum = state()
    term.mode_id[:] = torch.tensor([0, 1, 2])
    term.pose_error = lambda: (
        torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.02]]),
        torch.tensor([[0.0, 1.0], [0.0, 0.2], [0.0, 0.0]]),
    )
    term._update_metrics()
    torch.testing.assert_close(term._pose_ready_sum, torch.tensor([1.0, 0.0, 0.0]))
    term.pose_error = lambda: (
        torch.tensor([[0.0, 1.0], [0.0, 0.02], [0.0, 0.02]]),
        torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    term._update_metrics()
    torch.testing.assert_close(term._pose_ready_sum, torch.tensor([2.0, 0.0, 0.0]))


def test_pelvis_diagnostic_matches_original_lower_frame_including_opposite_foot_headings():
    from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz

    from g1.tasks.lower_limb_agent.mdp.commands.commands import SquatWalkCommand
    from g1.tasks.upper_limb_agent.mdp.kinematics import pelvis_xy_in_support_frame

    positions = torch.tensor([[[0, 0.1, 0], [0, -0.1, 0], [0.02, -0.01, 0.75]]]).expand(2, -1, -1).clone()
    yaw = torch.tensor([[0.2, 0.4, 0.3], [0.0, torch.pi, 0.6]])
    quats = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    data = SimpleNamespace(
        body_pos_w=SimpleNamespace(torch=positions),
        body_quat_w=SimpleNamespace(torch=quats),
        root_quat_w=SimpleNamespace(torch=quats[:, 2]),
    )
    original = SimpleNamespace(robot=SimpleNamespace(data=data), _support_foot_ids=[0, 1])
    origin, rotation = SquatWalkCommand.pelvis_support_frame_w(original)
    expected = quat_apply_inverse(rotation, positions[:, 2] - origin)[:, :2]
    torch.testing.assert_close(pelvis_xy_in_support_frame(data, 2, [0, 1]), expected)


def test_command_uses_per_episode_levels_for_goals_and_only_collects_natural_completions(command):
    env = command._env
    env.cfg = SimpleNamespace(
        actions=SimpleNamespace(
            lower_command=SimpleNamespace(command_ranges=FINAL[1:4]),
            height_offset=SimpleNamespace(height_offset_range=FINAL[0]),
            pelvis_offset=SimpleNamespace(offset_ranges=FINAL[4:]),
        )
    )
    env.common_step_counter = 0
    env.extras = {}
    env.reset_terminated = torch.zeros(3, dtype=torch.bool)
    env.reset_time_outs = torch.zeros(3, dtype=torch.bool)
    cfg = command.cfg.copy()
    cfg.range_curriculum = RangeCurriculumCfg(min_episodes_per_class=1)
    cfg.left_offsets.pos_x = cfg.right_offsets.pos_x = (0.10, 0.10)
    term = BimanualPoseCommand(cfg, env)
    term.reset()
    initial = term.goal_pose_w.clone()
    term.range_curriculum.level = 10
    term.reset(torch.tensor([0]))
    assert term.range_curriculum.episode_levels.tolist() == [10, 0, 0]
    torch.testing.assert_close(term.goal_pose_w[1:], initial[1:])
    # An explicit reset is excluded; a first-step natural failure is counted with quality zero.
    assert term.range_curriculum.counts.sum() == 0
    env.reset_terminated[0] = True
    term.reset(torch.tensor([0]))
    assert term.range_curriculum.counts[0] == 1 and term.range_curriculum.failures[0] == 1
    assert term.range_curriculum.score_sums[0] == 0


def test_runner_roundtrip_restores_level_ema_clock_and_freezes_playback(tmp_path):
    from test_task_runners import _agent, _UpperEnv

    from g1.rl.runner import create_task_runner

    class Env(_UpperEnv):
        def __init__(self):
            super().__init__()
            self.common_step_counter = 0
            self.command = SimpleNamespace(range_curriculum=state(), workspace_quality_map=None)
            self.command_manager = SimpleNamespace(active_terms=["wrist_pose"], get_term=lambda name: self.command)
            self.resets = 0

        def reset(self):
            assert self.command._restoring_ranges
            self.resets += 1

    env = Env()
    runner = create_task_runner(env, _agent())
    c = env.command.range_curriculum
    c.level = 4
    c.ema[:] = torch.tensor([0.6, 0.7])
    c.initialized[:] = True
    c.counts[:] = torch.tensor([20, 10])
    c.score_sums[:] = torch.tensor([14.0, 8.0])
    env.common_step_counter = 2350
    c.last_check = 1200
    filename = tmp_path / "model.pt"
    runner.save(filename)
    env2 = Env()
    playback = create_task_runner(env2, _agent(), evaluation=True)
    playback.load(filename)
    restored = env2.command.range_curriculum
    assert restored.level == 4 and restored.clock_offset == 2350 and restored.last_check == 1200
    torch.testing.assert_close(restored.ema, c.ema)
    torch.testing.assert_close(restored.counts, c.counts)
    assert restored.evaluation and env2.resets == 1
    restored.cfg.reset_on_load = True
    playback.load(filename)
    assert restored.level == 0 and not restored.counts.any() and not restored.initialized.any()
