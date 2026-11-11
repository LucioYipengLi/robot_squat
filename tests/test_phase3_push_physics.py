"""Opt-in public train chain: old Phase 2 -> progressive Phase 3 -> Phase 3 resume."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("G1_PHASE3_PUSH_SMOKE") != "1", reason="Opt-in: requires Isaac Sim and a GPU")
def test_phase2_to_phase3_push_course_and_resume_in_physx(tmp_path):
    project = Path(__file__).resolve().parents[1]
    common = [
        "--num_envs",
        "4",
        "--seed",
        "42",
        "--viz",
        "none",
        "--max_iterations",
        "2",
        "physics=isaacsim_physx",
        "env.episode_length_s=0.1",
        "agent.actor.hidden_dims=[16]",
        "agent.critic.hidden_dims=[16]",
        "agent.num_steps_per_env=4",
        "agent.algorithm.num_learning_epochs=1",
        "agent.algorithm.num_mini_batches=1",
    ]
    # Wrap only inspection points; runner loading, event execution, PPO and saves stay real.
    checked_train = """
import sys
import torch
from g1.rl.__main__ import main
from g1.rl import runner as dispatch
from g1.tasks.lower_limb_agent.mdp.curriculums import curriculum_step

original_factory = dispatch.create_task_runner
push_calls = 0
restored = False

def factory(env, *args, **kwargs):
    raw = env.unwrapped
    event = raw.event_manager.get_term_cfg('push_robot')
    original_push = event.func
    def checked_push(environment, env_ids, **params):
        global push_calls
        for axis in ('x', 'y'):
            assert params['velocity_range'][axis] == (-0.1, 0.1)
        original_push(environment, env_ids, **params)
        push_calls += 1
    event.func = checked_push
    runner = original_factory(env, *args, **kwargs)
    original_load = runner.load
    def checked_load(path, *load_args, **load_kwargs):
        global restored
        saved = torch.load(path, map_location='cpu', weights_only=False)['infos']['g1_curriculum']
        result = original_load(path, *load_args, **load_kwargs)
        assert event.params['velocity_range'] == {'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}
        elapsed = saved['runtime']['checks'].get('push:push_robot', 0)
        state = raw._g1_curriculum_state['push:push_robot']
        assert state['last_update_step'] == curriculum_step(raw) - elapsed
        if saved['phase'] == 2:
            assert 'push_velocity_range' not in saved['distribution']
            assert not state['initialized']
        assert raw.cfg.curriculum.push_velocity_magnitude.params['survival_gate'] == 0.93
        restored = True
        return result
    runner.load = checked_load
    original_learn = runner.learn
    def checked_learn(*learn_args, **learn_kwargs):
        result = original_learn(*learn_args, **learn_kwargs)
        assert restored and push_calls > 0
        print(f'PHASE3_PUSH_CHECKS={push_calls}', flush=True)
        return result
    runner.learn = checked_learn
    return runner

dispatch.create_task_runner = factory
raise SystemExit(main(['train', *sys.argv[1:]]))
"""

    previous = None
    expected_steps = 0
    stages = (
        ("phase1", "Template-G1-v0"),
        ("phase2", "Template-G1-Phase2-v0"),
        ("phase3", "Template-G1-Phase3-v0"),
        ("phase3_resume", "Template-G1-Phase3-v0"),
    )
    # Reuse a prepared tiny Phase 2 checkpoint when rerunning only the transfer checks.
    if source := os.environ.get("G1_PHASE3_PUSH_SOURCE"):
        previous = Path(source).resolve()
        saved = torch.load(previous, weights_only=False, map_location="cpu")["infos"]["g1_curriculum"]
        assert saved["phase"] == 2 and "push_velocity_range" not in saved["distribution"]
        expected_steps = saved["progress"]["control_steps"]
        stages = stages[2:]
    for name, task in stages:
        models, log = tmp_path / name, tmp_path / f"{name}.log"
        args = ["--task", task, *common, f"agent.experiment_name={models.as_posix()}"]
        if previous is not None:
            args.extend(["--checkpoint", str(previous)])
        if name.startswith("phase3"):
            args.extend(
                [
                    "env.events.push_robot.interval_range_s=(0.02,0.02)",
                    "env.curriculum.push_velocity_magnitude.params.survival_gate=0.93",
                    "env.curriculum.push_velocity_magnitude.params.task_success_gate=0.75",
                ]
            )
            if name == "phase3_resume":
                # The checkpoint's actual range must override a different configured initial range.
                args.extend(
                    [
                        "env.events.push_robot.params.velocity_range.x=(-0.03,0.03)",
                        "env.events.push_robot.params.velocity_range.y=(-0.03,0.03)",
                    ]
                )
            command = [sys.executable, "-c", checked_train, *args]
        else:
            command = [sys.executable, "-m", "g1.rl", "train", *args]
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=project,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=os.environ | {"PYTHONIOENCODING": "utf-8"},
                timeout=240,
                check=False,
            )
        assert result.returncode == 0, log.read_text(encoding="utf-8", errors="replace")[-16000:]
        checkpoints = list(models.rglob("model_*.pt"))
        assert checkpoints
        previous = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
        state = torch.load(previous, weights_only=False, map_location="cpu")["infos"]["g1_curriculum"]
        expected_steps += 8
        assert state["progress"]["control_steps"] == expected_steps
        if name == "phase2":
            assert "push_velocity_range" not in state["distribution"]
        if name.startswith("phase3"):
            assert "PHASE3_PUSH_CHECKS=" in log.read_text(encoding="utf-8")
            assert state["distribution"]["push_velocity_range"] == {"x": [-0.1, 0.1], "y": [-0.1, 0.1]}
            assert "push:push_robot" in state["runtime"]["checks"]
            assert state["runtime"]["push_survival"] is not None
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            events = EventAccumulator(str(previous.parent)).Reload()
            for metric in ("max_delta_v_mps", "survival_ema", "task_success_rate"):
                assert events.Scalars(f"Curriculum/push_velocity_magnitude/{metric}")
