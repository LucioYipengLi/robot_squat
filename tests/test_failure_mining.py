"""Failure clips must contain the terminal state, preserve reset boundaries and load without USD."""

import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from g1.rl.failures.cli import parse_args
from g1.rl.failures.pairs import PairedSwitchSummary
from g1.rl.failures.recording import FailureMiner, FailureRecorderTerm, force_category
from g1.rl.failures.storage import FailureArchive, TensorHistory, contact_statistics, write_json
from g1.rl.failures.switches import RegularSwitchTracker
from g1.rl.failures.viewer import read_index, sample_payload
from g1.tasks.upper_limb_agent.mdp.contacts import ARM_CONTACT_BODY_PATHS


def test_ring_wrap_and_partial_reset_keep_each_episode_separate():
    ring = TensorHistory(3, 2)
    for step in range(4):
        ring.push({"value": torch.tensor([[step], [100 + step]], dtype=torch.float32)})
    np.testing.assert_array_equal(ring.extract(0)["value"][:, 0], [1, 2, 3])
    ring.reset([0])
    ring.push({"value": torch.tensor([[4.0], [104.0]])})
    np.testing.assert_array_equal(ring.extract(0)["value"][:, 0], [4])
    np.testing.assert_array_equal(ring.extract(1)["value"][:, 0], [102, 103, 104])


def test_statistics_keep_substep_impulse_and_force_cancellation_visible():
    pair = np.zeros((2, 2, 2, 3))
    pair[:, :, 0, 0] = [[2, 0], [3, 4]]
    pair[:, :, 1, 0] = -pair[:, :, 0, 0]
    trace = {"pair_force": pair, "net_force": pair.sum(axis=-2)}
    stats = contact_statistics(trace, ["left_elbow", "right_elbow"], 0.01)
    assert stats["peak_net_normal_force_n"] == 0
    assert stats["peak_unattributed_normal_force_n"] == 0
    assert stats["partners"][0]["peak_normal_force_n"] == 4
    assert stats["partners"][0]["normal_impulse_ns"] == pytest.approx(0.09)
    assert stats["partners"][0]["time_above_1n_s"] == pytest.approx(0.03)
    assert stats["partners"][0]["contact_onsets"] == 2


def make_miner(tmp_path, *, episodes=10, max_failures=10, switches=False):
    n = 2
    proxy = lambda value: SimpleNamespace(torch=value)  # noqa: E731
    poses = torch.zeros(n, 3, 7)
    poses[..., 2], poses[..., 6] = 0.75, 1
    origin = torch.tensor([[10.0, 20, 0], [30.0, 40, 0]])
    poses[..., :3] += origin[:, None]
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=proxy(poses), joint_pos=proxy(torch.zeros(n, 3)), joint_vel=proxy(torch.zeros(n, 3))
        )
    )
    goal = poses[:, :2].clone()
    term = SimpleNamespace(
        goal_pose_w=goal,
        command_counter=torch.ones(n, dtype=torch.long),
        mode_id=torch.tensor([0, 2]),
        task_ids=torch.tensor([2, 0]),
        active_mask=torch.tensor([[True, False], [True, True]]),
        _sampling_quat_w=poses[:, 0, 3:].clone(),
        cfg=SimpleNamespace(resampling_time_range=(0.2, 0.4)),
        range_curriculum=SimpleNamespace(
            episode_levels=torch.tensor([0, 0]),
            cfg=SimpleNamespace(position_tolerance_m=0.015, orientation_tolerance_rad=0.10),
        ),
        evaluation_start_steps=torch.tensor([10, 0]),
        episode_deadline_steps=torch.tensor([20, 20]),
        planar_distance_m=torch.tensor([0.2, 0]),
        planar_heading_rad=torch.tensor([0.1, 0]),
        age=torch.ones(n, dtype=torch.long),
    )
    term.episode_age_steps = lambda: term.age
    term.pose_error = lambda: ((goal[..., :3] - poses[:, :2, :3]).norm(dim=-1), torch.zeros(n, 2))
    pair = torch.zeros(n, 3, 1, 2, 3)
    pair[:, :, 0, 0, 0] = torch.tensor([3.0, 2.0, 1.0])
    net = pair.sum(dim=-2)
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            normal_force_matrix_w_history=proxy(pair),
            net_normal_forces_w_history=proxy(net),
            contact_pos_w=proxy(torch.full((n, 1, 2, 3), float("nan"))),
        )
    )

    class Scene(dict):
        env_origins = origin

    sensor.find_bodies = lambda _: ([0], ["torso_link"])
    scene = Scene(robot=robot, torso_contacts=sensor, contact_forces=sensor)
    arm_pair = torch.zeros(n, 3, 1, len(ARM_CONTACT_BODY_PATHS), 3)
    scene["torso_contact_pairs"] = SimpleNamespace(
        data=SimpleNamespace(
            normal_force_matrix_w_history=proxy(arm_pair), net_normal_forces_w_history=proxy(arm_pair.sum(-2))
        )
    )
    ended, timeout = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    env = SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=0.01), decimation=2),
        step_dt=0.02,
        max_episode_length=100,
        num_envs=n,
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda _: term),
        action_manager=SimpleNamespace(
            action=torch.zeros(n, 23), get_term=lambda _: SimpleNamespace(lower_command=torch.zeros(n, 6))
        ),
        reset_terminated=ended,
        reset_time_outs=timeout,
        termination_manager=SimpleNamespace(
            active_terms=["base_contact", "time_out"],
            get_term=lambda name: ended if name == "base_contact" else timeout,
        ),
    )
    manifest = {"partners": ["left_elbow", "ground"], "physics_dt": 0.01, "control_dt": 0.02}
    tracker = RegularSwitchTracker(env, term, 20.0) if switches else None
    if tracker is not None:
        manifest["regular_switch_evaluation"] = tracker.settings()
    archive = FailureArchive(tmp_path / "archive", manifest)
    miner = FailureMiner(
        env,
        archive,
        manifest["partners"],
        history_s=0.1,
        episodes=episodes,
        max_failures=max_failures,
        switch_tracker=tracker,
    )
    env.failure_miner = miner
    recorder = FailureRecorderTerm.__new__(FailureRecorderTerm)
    recorder._env = env
    return miner, recorder, env, term, archive


def test_recorder_saves_terminal_frame_before_reset_and_never_leaks_old_episode(tmp_path):
    miner, recorder, env, term, archive = make_miner(tmp_path)
    recorder.record_post_step()
    assert archive.completed == 0
    term.age += 1
    env.scene["robot"].data.body_link_pose_w.torch[0, :, 2] = 0.11
    env.reset_terminated[0] = True
    recorder.record_post_step()
    with np.load(archive.directory / "clips/000000.npz", allow_pickle=False) as clip:
        assert clip["body_pose"].shape[0] == 2
        np.testing.assert_allclose(clip["body_pose"][-1, :, 2], 0.11)
        np.testing.assert_allclose(clip["body_pose"][-1, :, 0], 0)  # Env origin removed.
        np.testing.assert_array_equal(clip["pair_force"][-1, :, 0, 0], [2, 3])
    recorder.reset([0])
    term.age[0] = 1
    env.scene["robot"].data.body_link_pose_w.torch[0, :, 2] = 1.3
    recorder.record_post_step()
    with np.load(archive.directory / "clips/000001.npz", allow_pickle=False) as clip:
        assert len(clip["time"]) == 1
        np.testing.assert_allclose(clip["body_pose"][:, :, 2], 1.3)
    assert int(miner.history.lengths[1]) == 3
    assert archive.saved == archive.failed == archive.completed == 2
    archive.progress("episodes")
    archive.close()
    payload = sample_payload(archive.directory, "000000")
    assert payload["contact_pos"][0][0] == [None, None, None]
    assert len(read_index(archive.directory)["samples"]) == 2


def test_successes_have_target_summaries_but_no_failure_clip(tmp_path):
    _, recorder, env, _, archive = make_miner(tmp_path)
    env.reset_time_outs[1] = True
    recorder.record_post_step()
    archive.close()
    rows = [json.loads(line) for line in (archive.directory / "episodes.jsonl").read_text().splitlines()]
    assert rows[0]["category"] == "regular" and not rows[0]["failed"]
    assert rows[0]["reasons"] == ["time_out"] and "goal_pose" in rows[0]
    assert archive.completed == 1 and archive.saved == 0


def test_pair_recorder_ignores_unassigned_environment_and_labels_failure_clip(tmp_path):
    miner, recorder, env, term, archive = make_miner(tmp_path, switches=True)
    archive.pair_summary = PairedSwitchSummary([{"id": "one", "group": "in_up"}])
    miner.fixed_pairs = SimpleNamespace(
        trial_ids=torch.tensor([-1, 1]),
        metadata=lambda _: {"pair_id": "one", "pair_group": "in_up", "pair_side": "right"},
    )
    term.mode_id[1] = 1
    env.reset_terminated[:] = True
    recorder.record_post_step()
    archive.progress("interrupted")
    archive.close()

    assert archive.completed == archive.saved == 1
    sample = read_index(archive.directory)["samples"][0]
    assert (sample["env_id"], sample["pair_id"], sample["pair_side"]) == (1, "one", "right")
    report = json.loads((archive.directory / "summary.json").read_text())["switch_pair_evaluation"]["overall"]
    assert report["incomplete"] == report["right"]["failed_before_switch"] == 1


def test_regular_switch_denominator_includes_success_and_early_terminated_contacts(tmp_path):
    _, recorder, env, term, archive = make_miner(tmp_path, switches=True)
    poses = env.scene["robot"].data.body_link_pose_w.torch
    force = env.scene["torso_contact_pairs"].data.normal_force_matrix_w_history.torch
    # Initial regular goal is not a switch. This contact also precedes the new goal assignment.
    force[1, :, 0, 0, 0] = 7
    recorder.record_post_step()
    force.zero_()
    term.command_counter[1] += 1
    term.goal_pose_w[1, :, 0] -= 0.10
    for step in range(2, 13):
        term.age[1] = step
        if step == 11:
            poses[1, :2] = term.goal_pose_w[1]
            force[1, 0, 0, 0, 0] = 1.0  # Last physics substep at exactly 0.2 s.
        if step == 12:
            force[1, :, 0, 0, 0] = 8  # 0.21–0.22 s after refresh, outside the early window.
            env.reset_time_outs[1] = True
        recorder.record_post_step()

    recorder.reset([1])
    env.reset_time_outs[1] = False
    force.zero_()
    term.command_counter[1] = 1
    term.age[1] = 1
    recorder.record_post_step()
    term.command_counter[1] = 2
    term.age[1] = 2
    term.goal_pose_w[1, :, 0] -= 0.10
    force[1, 1, 0, :2, 0] = torch.tensor([12.0, -12.0])
    env.reset_terminated[1] = True
    recorder.record_post_step()
    archive.progress("episodes")
    archive.close()

    rows = [json.loads(line) for line in (archive.directory / "switches.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and [row["episode_id"] for row in rows] == [0, 1]
    safe, failed = rows
    assert [row["switch_index"] for row in rows] == [1, 1]
    assert all(row["direction"] == "retract" and row["position_jump_bin"] == "8_to_12cm" for row in rows)
    assert safe["switch_time_s"] == pytest.approx(0.02)
    assert safe["pre_switch_contact"] and safe["pre_switch_ready"]
    assert safe["peak_arm_force_0p2s_n"] == 1.0 and safe["contact_over_1n"] is False
    assert safe["new_goal_reach_time_s"] == pytest.approx(0.2)
    assert failed["contact_observed_s"] == pytest.approx(0.02)
    assert failed["peak_arm_force_0p2s_n"] == pytest.approx(24.0) and failed["contact_over_hard_threshold"]
    assert failed["terminated_within_1s"] and failed["new_goal_reached_within_1s"] is False
    summary = json.loads((archive.directory / "summary.json").read_text())
    summary = summary["regular_switches"]
    groups = summary["by_group"]
    assert groups[0]["direction"] == "retract" and groups[0]["arm_mode"] == "both"
    assert [group["pre_switch_contact"] for group in groups] == [False, True]
    assert summary["overall"]["contact_over_hard_threshold"] == {"count": 1, "assessed": 2, "rate": 0.5}
    assert read_index(archive.directory)["samples"][0]["episode_id"] == failed["episode_id"]


def test_short_timeout_is_censored_and_initial_goal_failures_are_separate(tmp_path):
    _, recorder, env, term, archive = make_miner(tmp_path, switches=True)
    recorder.record_post_step()
    term.command_counter[1] = 2
    term.age[1] += 1
    term.goal_pose_w[1, :, 1] += 0.10
    env.reset_time_outs[1] = True
    recorder.record_post_step()
    recorder.reset([1])
    term.command_counter[1] = 1
    term.age[1] = 1
    env.reset_time_outs[1] = False
    env.reset_terminated[1] = True
    recorder.record_post_step()
    archive.progress("episodes")
    archive.close()
    row = json.loads((archive.directory / "switches.jsonl").read_text())
    assert row["direction"] == "lateral" and row["contact_over_1n"] is None
    assert row["new_goal_reached_within_1s"] is None
    summary = json.loads((archive.directory / "summary.json").read_text())["regular_switches"]
    assert summary["overall"]["switches"] == 1
    assert summary["overall"]["contact_over_1n"]["assessed"] == 0
    assert summary["regular_episodes_without_switch"] == summary["regular_failures_before_first_switch"] == 1


@pytest.mark.parametrize("limit", ["episodes", "max_failures"])
def test_parallel_completions_respect_exact_archive_limits(tmp_path, limit):
    miner, recorder, env, _, archive = make_miner(tmp_path, **{limit: 1})
    env.reset_terminated[:] = True
    recorder.record_post_step()
    recorder.record_post_step()
    assert miner.done and archive.saved == archive.completed == 1
    archive.close()


def test_source_archive_cannot_be_overwritten_or_viewer_path_traversed(tmp_path):
    archive = FailureArchive(tmp_path / "existing", {})
    archive.close()
    with pytest.raises(FileExistsError):
        FailureArchive(archive.directory, {})
    for sample in ["../run.json", "000001/../../secret", "not-an-id"]:
        with pytest.raises(ValueError, match="sample id"):
            sample_payload(archive.directory, sample)


def test_interrupt_during_compression_does_not_count_an_uncommitted_failure(tmp_path, monkeypatch):
    _, recorder, env, _, archive = make_miner(tmp_path)
    env.reset_terminated[0] = True

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "savez_compressed", interrupt)
    with pytest.raises(KeyboardInterrupt):
        recorder.record_post_step()
    assert archive.saved == archive.failed == archive.completed == 0
    assert (archive.directory / "failures.jsonl").read_text() == ""
    archive.close()


@pytest.mark.parametrize("category", ["regular", "height_shifted", "planar", "mixed"])
def test_category_override_keeps_native_resampling_and_checkpoint_contract(category):
    from test_planar_timing import make_command

    term = make_command(probability=0.2)
    original = term._resample
    contract = term.range_curriculum.contract()
    force_category(term, category)
    term.reset()
    assert term._resample == original and term._teleop is None
    assert term.range_curriculum.contract() == contract
    if category != "mixed":
        assert (term.task_ids == {"regular": 0, "height_shifted": 1, "planar": 2}[category]).all()
        if category == "regular":
            assert torch.isfinite(term.time_left).all()
    else:
        assert len(term.task_ids) > 0


def test_help_and_offline_view_import_do_not_load_simulation_or_torch():
    script = "import sys; from g1.rl.__main__ import main; main(['failures','--help']);"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import g1.rl.failures.viewer; "
            "assert not any(n in sys.modules for n in ('isaaclab','pxr','torch','g1.tasks'))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "option,value",
    [("--num_envs", "0"), ("--history_s", "nan"), ("--episodes", "-1"), ("--level", "-1"), ("--level", "2.5")],
)
def test_invalid_mining_budgets_are_rejected(option, value):
    with pytest.raises(SystemExit):
        parse_args(["mine", "--checkpoint", "model.pt", option, value])


def test_mining_launch_boundary_has_no_usd_import(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({}, checkpoint)
    script = """
import sys
class UsdGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pxr' or fullname.startswith('pxr.'):
            raise RuntimeError('USD loaded before SimulationApp: '+fullname)
sys.meta_path.insert(0, UsdGuard())
import isaaclab.app
class LaunchReached(Exception): pass
def stop(cfg, args):
    assert args['headless'] and args['visualizer'] is None
    assert cfg.scene.num_envs == 2
    assert cfg.terminations.base_contact.params['threshold'] == 20.0
    raise LaunchReached()
isaaclab.app.launch_simulation = stop
from g1.rl.failures.cli import run
try:
    run(['mine','--checkpoint',sys.argv[1],'--num_envs','2'])
except LaunchReached:
    print('launch reached without USD')
else:
    raise AssertionError('launcher not called')
"""
    result = subprocess.run([sys.executable, "-c", script, str(checkpoint)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "launch reached without USD" in result.stdout


def test_json_writer_never_exports_nan_or_partial_documents(tmp_path):
    path = tmp_path / "data.json"
    write_json(path, {"contact": np.array([np.nan, np.inf, 2.0])})
    assert json.loads(path.read_text()) == {"contact": [None, None, 2.0]}
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("requested_level", [None, 0, 2, 10, 11])
def test_full_mining_flow_writes_loadable_archive_without_learning_or_terminal_state_loss(
    tmp_path, monkeypatch, requested_level
):
    import isaaclab.app
    import isaaclab.sim as sim_utils

    from isaaclab_rl import rsl_rl
    from isaaclab_rl.entrypoints import common

    from isaaclab_tasks import utils as task_utils

    from g1.rl import runner as runner_module
    from g1.rl.failures import geometry, mine, recording
    from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import UpperLimbMobileEnvCfg

    _, recorder, raw, term, unused = make_miner(tmp_path / "fixture")
    unused.close()
    raw.failure_miner = None
    cfg = UpperLimbMobileEnvCfg()
    cfg.sim.dt, cfg.decimation = 0.01, 2
    raw.cfg = cfg
    term.cfg = cfg.commands.wrist_pose
    raw.device = "cpu"
    raw.scene["robot"].body_names = ["pelvis", "torso_link", "left_elbow_link"]
    raw.scene["robot"].joint_names = ["joint0", "joint1", "joint2"]
    term.range_curriculum.level = 0
    term.elapsed_seconds = lambda ids: term.age[ids].float() * raw.step_dt
    raw.termination_manager.get_term_cfg = lambda _: SimpleNamespace(params={"threshold": 1.0})
    calls = []

    class Policy:
        def __call__(self, obs, **kwargs):
            calls.append(kwargs)
            return torch.zeros(2, 23)

        def reset(self, *args):
            pass

    policy = Policy()

    class Wrapped:
        def __init__(self, env, **kwargs):
            self.unwrapped = env.unwrapped

        def reset(self):
            term.age.zero_()
            term.range_curriculum.episode_levels.fill_(term.range_curriculum.level)
            return torch.zeros(2, 4), {}

        def step(self, actions):
            term.age += 1
            raw.scene["robot"].data.body_link_pose_w.torch[0, :, 2] = 0.11
            raw.reset_terminated[0], raw.reset_time_outs[1] = True, True
            recorder.record_post_step()
            dones = raw.reset_terminated | raw.reset_time_outs
            recorder.reset([0, 1])
            raw.scene["robot"].data.body_link_pose_w.torch[:, :, 2] = 0.75
            term.age.zero_()
            return torch.zeros(2, 4), torch.zeros(2), dones, {}

    raw_env = SimpleNamespace(unwrapped=raw)
    agent_cfg = SimpleNamespace(clip_actions=None, seed=0, device="cpu")

    def load(path):
        assert term.cfg.resampling_time_range == (2.0, 4.0)
        calls.append("loaded")
        term.range_curriculum.level = 3

    fake_runner = SimpleNamespace(load=load, get_inference_policy=lambda **kwargs: policy)

    def launch(cfg, args):
        assert requested_level != 11, "Out-of-range levels must fail before launching the simulator"
        return nullcontext()

    monkeypatch.setattr(isaaclab.app, "launch_simulation", launch)
    monkeypatch.setattr(task_utils, "resolve_task_config", lambda *a, **k: (cfg, agent_cfg))
    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", Wrapped)
    monkeypatch.setattr(rsl_rl, "handle_deprecated_rsl_rl_cfg", lambda cfg, version: cfg)
    monkeypatch.setattr(runner_module, "create_task_runner", lambda *a, **k: fake_runner)
    monkeypatch.setattr(common, "create_isaaclab_env", lambda *a, **k: raw_env)
    monkeypatch.setattr(common, "close_env", lambda env: calls.append("closed"))
    monkeypatch.setattr(sim_utils, "get_current_stage", lambda: SimpleNamespace(GetPrimAtPath=lambda _: None))
    monkeypatch.setattr(geometry, "contact_filters", lambda _: (["left_elbow", "ground"], ["/left", "/ground"]))
    monkeypatch.setattr(geometry, "export_geometry", lambda *a: {"meshes": [], "body_names": [], "edges": []})
    monkeypatch.setattr(recording, "force_category", lambda *a: None)
    monkeypatch.setattr(mine, "write_json", write_json)
    # The real YAML writer exercises config serialization; the runner has no learning method.
    checkpoint = tmp_path / "model.pt"
    saved_target = {"resampling_time_range": [2.0, 4.0]}
    torch.save({"infos": {"g1_upper_ranges": {"contract": {"target": saved_target}}}}, checkpoint)
    output = tmp_path / "mined"
    args = parse_args(
        [
            "mine",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--episodes",
            "2",
            "--num_envs",
            "2",
            "--policy_output",
            "sample",
            "--device",
            "cpu",
        ]
        + (["--level", str(requested_level)] if requested_level is not None else [])
        + (["--category", "regular", "--regular_refresh_s", "2", "3"] if requested_level == 1 else [])
    )
    if requested_level == 11:
        with pytest.raises(ValueError, match="--level must be between 0 and 10"):
            mine.run_mining(args)
        assert not output.exists() and not calls
        return
    assert mine.run_mining(args) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "episodes" and summary["completed_episodes"] == 2
    assert summary["saved_failures"] == 1 and summary["failed_episodes"] == 1
    assert calls == ["loaded", {"stochastic_output": True}, "closed"]
    assert (output / "effective_env.yaml").is_file()
    clip = sample_payload(output, "000000")
    assert clip["body_pose"][-1][0][2] == pytest.approx(0.11)
    index = read_index(output)
    assert index["run"]["terminations"]["base_contact"] == 1.0
    expected_level = requested_level if requested_level is not None else 3
    assert index["run"]["level"] == expected_level
    assert index["run"]["checkpoint_level"] == 3
    assert index["run"]["requested_level"] == requested_level
    assert index["run"]["checkpoint_regular_refresh_s"] == [2.0, 4.0]
    assert index["run"]["regular_refresh_s"] == ([2.0, 3.0] if requested_level == 1 else [2.0, 4.0])
    if requested_level == 1:
        assert summary["regular_switches"]["regular_episodes_without_switch"] == 1
        assert (output / "switches.jsonl").is_file()
    assert index["samples"][0]["level"] == expected_level
    assert term.range_curriculum.level == expected_level
