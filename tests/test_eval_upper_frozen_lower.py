"""Verify Upper Eval's explicit frozen Lower substitution without launching physics."""

import json
from types import SimpleNamespace

import pytest
import torch

from g1.rl.evaluation import cli, jobs
from g1.rl.evaluation.upper_fixed_targets import resolve_upper_height_range
from g1.tasks.upper_limb_agent import evaluation
from g1.tasks.upper_limb_agent.symmetry import FULL_OBSERVATION_DIM


def test_plan_pins_lower_and_resume_rejects_changed_sidecar(tmp_path, capsys):
    upper = tmp_path / "upper.pt"
    upper.write_bytes(b"upper checkpoint")
    lower = tmp_path / "lower" / "model.pt"
    params = lower.parent / "params"
    params.mkdir(parents=True)
    lower.write_bytes(b"lower checkpoint")
    lower_env = params / "env.yaml"
    lower_env.write_text("contract: 1", encoding="utf-8")
    output = tmp_path / "plan"
    args = [
        "--checkpoints",
        str(upper),
        "--frozen-lower-checkpoint",
        str(lower),
        "--upper-height-range",
        "-0.39",
        "0.05",
        "--plan-only",
        "--output",
        str(output),
    ]
    with pytest.raises(SystemExit):
        cli.run(args)
    assert "require an Upper Eval suite" in capsys.readouterr().err
    assert cli.run(["--suite", "g1_upper_fixed_targets", *args]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for job in manifest["jobs"]:
        inputs = job["checkpoint"]
        assert inputs["frozen_lower_override"]["sha256"] == jobs.digest_file(lower)
        assert inputs["upper_height_range"] == [-0.39, 0.05]
    assert cli.run(["--resume", str(output), "--plan-only"]) == 0
    with pytest.raises(SystemExit):
        cli.run(["--resume", str(output), "--upper-height-range", "-0.38", "0.05", "--plan-only"])
    assert "omit configuration overrides" in capsys.readouterr().err
    lower_env.write_text("contract: 2", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.run(["--resume", str(output), "--plan-only"])
    assert "Checkpoint input changed" in capsys.readouterr().err


def test_fixed_height_must_fit_both_models():
    upper, lower = (-0.40, 0.05), (-0.3900000000000002, 0.05)
    assert resolve_upper_height_range(upper, lower) == (lower[0], 0.05)
    assert resolve_upper_height_range(upper, lower, (-0.39, 0.05)) == (-0.39, 0.05)
    with pytest.raises(ValueError, match="must be within"):
        resolve_upper_height_range(upper, lower, (-0.40, 0.05))


@pytest.mark.parametrize("replace,fixed,expected", [(False, None, -0.40), (True, None, -0.39), (False, -0.39, -0.39)])
def test_upper_eval_prepares_expected_pair_and_height(tmp_path, monkeypatch, replace, fixed, expected):
    original = tmp_path / "artifacts" / "frozen_lower_model.pt"
    original.parent.mkdir()
    replacement = tmp_path / "new_lower.pt"
    for path, depth in ((original, -0.405), (replacement, -0.39)):
        state = {"family": "g1.lower_limb_agent", "distribution": {"ranges": {"height_offset": [depth, 0.05]}}}
        torch.save({"actor_state_dict": {}, "infos": {"g1_curriculum": state}}, path)
    upper = tmp_path / "upper.pt"
    lower_interface = {"input_dim": 650, "action_dim": 12}
    control = {
        "schema_version": 4,
        "policy_observation_dim": FULL_OBSERVATION_DIM,
        "height_offset_range": [-0.40, 0.05],
    }
    infos = {
        "g1_frozen_lower": {"sha256": jobs.digest_file(original), "interface": lower_interface},
        "g1_control_interface": control,
    }
    torch.save({"infos": infos}, upper)
    checkpoint = {"path": str(upper), "sha256": jobs.digest_file(upper)}
    selected = replacement if replace else original
    if replace:
        checkpoint["frozen_lower_override"] = {"path": str(replacement), "sha256": jobs.digest_file(replacement)}
    if fixed is not None:
        checkpoint["upper_height_range"] = [fixed, 0.05]
    cfg = SimpleNamespace(
        actions=SimpleNamespace(
            joint_pos=SimpleNamespace(checkpoint_path="unused"),
            height_offset=SimpleNamespace(height_offset_range=(-0.40, 0.05)),
        ),
        commands=SimpleNamespace(
            wrist_pose=SimpleNamespace(
                planar_turn_focus=SimpleNamespace(enabled=True),
                range_curriculum=SimpleNamespace(initial_lower_ranges=((-0.10, 0.01),)),
            )
        ),
        sim=SimpleNamespace(dt=1 / 120),
        scene=SimpleNamespace(),
        domain_randomization=SimpleNamespace(),
        decimation=2,
        episode_length_s=35,
    )
    benchmark = {"episode_length_s": 10, "level": 10, "control_dt": 1 / 60}
    saved, setup = evaluation.prepare_env_cfg(cfg, benchmark, checkpoint, 4, 42)
    assert cfg.actions.joint_pos.checkpoint_path == str(selected)
    assert cfg.actions.height_offset.height_offset_range == (expected, 0.05)

    loaded = SimpleNamespace(
        metadata={"sha256": jobs.digest_file(selected), "checkpoint": str(selected)}, interface=lower_interface
    )
    curriculum = cfg.commands.wrist_pose.range_curriculum
    curriculum.lower_ranges = torch.tensor([[[expected, 0.05]]])
    raw = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(frozen_policy=loaded)),
        command_manager=SimpleNamespace(get_term=lambda _: cfg.commands.wrist_pose),
    )
    monkeypatch.setattr(
        evaluation.UpperLimbOnPolicyRunner,
        "mobile_control_interface",
        lambda _: {**control, "height_offset_range": list(cfg.actions.height_offset.height_offset_range)},
    )
    monkeypatch.setattr(evaluation, "build_actor", lambda *_, **__: (object(), {"actor": {"class_name": "MLPModel"}}))
    env = SimpleNamespace(unwrapped=raw)
    _, _, metadata = evaluation.load_policy(env, None, checkpoint, saved, setup, device="cpu")
    assert metadata["frozen_lower_overridden"] is replace
    assert metadata["episode_height_range_m"] == pytest.approx([expected, 0.05])
    loaded.interface = {**lower_interface, "action_dim": 13}
    with pytest.raises(ValueError, match="Frozen Lower model or interface differs"):
        evaluation.load_policy(env, None, checkpoint, saved, setup, device="cpu")
