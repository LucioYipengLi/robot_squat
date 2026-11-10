"""Spatial-prior export, interpolation, mirroring and active-wrist position gating."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from g1.tasks.upper_limb_agent.mdp.workspace_quality import WorkspaceQualityMap

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_workspace_quality", ROOT / "scripts/prepare_workspace_quality.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def write_map(path, *, missing=False):
    coordinates = np.indices((2, 2, 2))
    quality = (0.1 + 0.1 * coordinates[0] + 0.2 * coordinates[1] + 0.3 * coordinates[2]).astype(np.float32)
    valid = np.ones_like(quality, dtype=bool)
    if missing:
        valid[1, 1, 1] = False
        quality[1, 1, 1] = 0
    metadata = {
        "schema_version": 1,
        "kind": "g1_workspace_quality",
        "coordinate_frame": "pelvis",
        "target_link": "left_wrist_yaw_link",
        "payload_mass_kg": 0,
        "aggregation": "max_all_saved_configurations",
        "right_mirror_axis": "y",
        "array_axis_order": "xyz",
        "q_reference": 1.0,
        "origin_xyz_m": [0, 0, 0],
        "step_xyz_m": [0.05] * 3,
        "shape_xyz": [2, 2, 2],
    }
    np.savez(path, quality=quality, valid=valid, metadata_json=np.array(json.dumps(metadata)))
    return WorkspaceQualityMap(path, "cpu")


def test_trilinear_query_known_nodes_boundary_and_input_immutability(tmp_path):
    field = write_map(tmp_path / "map.npz")
    points = torch.tensor([[0.0, 0.0, 0.0], [0.025, 0.025, 0.025], [0.05, 0.05, 0.05]])
    saved = points.clone()
    scores, valid = field.query(points)
    torch.testing.assert_close(scores, torch.tensor([0.1, 0.4, 0.7]))
    assert valid.all()
    torch.testing.assert_close(points, saved, rtol=0, atol=0)


def test_unknown_corner_is_not_blended_and_exact_known_node_remains_valid(tmp_path):
    field = write_map(tmp_path / "map.npz", missing=True)
    scores, valid = field.query(torch.tensor([[0.0, 0.0, 0.0], [0.025, 0.025, 0.025], [0.05, 0.05, 0.05]]))
    torch.testing.assert_close(scores, torch.tensor([0.1, 0.0, 0.0]))
    assert valid.tolist() == [True, False, False]


def test_outside_and_nonfinite_queries_are_unknown(tmp_path):
    field = write_map(tmp_path / "map.npz")
    points = torch.tensor([[-0.001, 0, 0], [0.051, 0, 0], [float("nan"), 0, 0], [float("inf"), 0, 0]])
    scores, valid = field.query(points)
    assert not valid.any() and not scores.any()
    assert torch.isfinite(scores).all()


def bimanual_targets():
    return torch.tensor([[[0.025, 0.025, 0.025], [0.025, -0.025, 0.025]]]).expand(3, -1, -1).clone()


def test_modes_mirror_normalization_all_active_gate_and_idle_exclusion(tmp_path):
    field = write_map(tmp_path / "map.npz")
    targets = bimanual_targets()
    active = torch.tensor([[True, False], [False, True], [True, True]])
    errors = torch.zeros(3, 2)
    errors[0, 1] = 100  # Idle hand must not affect the gate.
    targets[0, 1] = 100  # Nor its map coverage.
    state = field.evaluate(targets, targets, errors, active, 0.01)
    torch.testing.assert_close(state.reward, torch.full((3,), 0.4))
    assert state.position_ready.all() and state.map_covered.all()
    errors[2, 1] = 0.0101
    state = field.evaluate(targets, targets, errors, active, 0.01)
    torch.testing.assert_close(state.reward, torch.tensor([0.4, 0.4, 0.0]))
    errors[2, 1] = 0.0099
    targets[2, 1] = 100
    state = field.evaluate(targets, targets, errors, active, 0.01)
    assert state.reward[2] == 0 and not state.map_covered[2]


def test_score_uses_target_position_and_inactive_episode_has_zero_reward(tmp_path):
    field = write_map(tmp_path / "map.npz")
    targets = bimanual_targets()
    active = torch.ones(3, 2, dtype=torch.bool)
    actual = targets.clone()
    actual[:, :, 0] -= 0.005
    state = field.evaluate(targets, actual, torch.full((3, 2), 0.005), active, 0.01)
    torch.testing.assert_close(state.reward, torch.full((3,), 0.4))
    assert (state.actual_scores < state.target_scores).all()
    active.zero_()
    assert not field.evaluate(targets, actual, torch.zeros(3, 2), active, 0.01).reward.any()


@pytest.mark.parametrize("tolerance", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_tolerance_is_rejected(tmp_path, tolerance):
    field = write_map(tmp_path / "map.npz")
    with pytest.raises(ValueError, match="positive and finite"):
        field.evaluate(
            bimanual_targets(), bimanual_targets(), torch.zeros(3, 2), torch.ones(3, 2, dtype=torch.bool), tolerance
        )


def write_source(path):
    points = np.indices((2, 2, 2)).reshape(3, -1).T.astype(np.float32) * 0.05
    quality = np.column_stack([np.arange(16) / 20] * 3)
    meta = {
        "kind": "g1_batch_configuration_metrics",
        "schema_version": 2,
        "coordinate_frame": "pelvis",
        "target_link": "left_wrist_yaw_link",
        "payload_masses_kg": [0, 1, 3],
        "completed": 16,
        "composite_completed": 16,
        "source_ik_signature": {"grid": {"step_m": 0.05}},
        "composite_metrics": {"weights": [0.3, 0.3, 0.4]},
    }
    np.savez(
        path,
        points=points,
        point_indices=np.repeat(np.arange(8), 2),
        solution_counts=np.full(8, 2),
        composite_score=quality,
        metadata_json=np.array(json.dumps(meta)),
    )


def test_export_selects_max_of_real_samples_and_preserves_source(tmp_path):
    source, output = tmp_path / "source.npz", tmp_path / "export.npz"
    write_source(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = prepare.prepare_workspace_quality(source, output)
    field = WorkspaceQualityMap(output, "cpu")
    with np.load(source, allow_pickle=False) as raw:
        scores, valid = field.query(torch.tensor(raw["points"]))
    torch.testing.assert_close(scores, torch.arange(1, 16, 2) / 15)
    assert valid.all() and metadata["q_reference"] == 0.75
    assert metadata["source_metrics"]["sha256"] == digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(output.with_suffix(".json").read_text()) == field.metadata
    with pytest.raises(FileExistsError):
        prepare.prepare_workspace_quality(source, output)
    with pytest.raises(ValueError, match="distinct"):
        prepare.prepare_workspace_quality(source, source, overwrite=True)


def test_packaged_resource_matches_every_real_source_node():
    source = ROOT / "logs/analysis/left_5cm.batch.metrics_0.4ef.yaw-cropped.npz"
    if not source.exists():
        pytest.skip("Raw analysis data is optional; synthetic export test is self-contained")
    field = WorkspaceQualityMap("data/workspace_q0_v1.npz", "cpu")
    with np.load(source, allow_pickle=False) as raw:
        points, rows, q = raw["points"], raw["point_indices"], raw["composite_score"][:, 0]
    expected = np.full(len(points), -np.inf)
    np.maximum.at(expected, rows, q)
    score, valid = field.query(torch.tensor(points))
    assert valid.all()
    torch.testing.assert_close(score.double(), torch.tensor(expected / field.q_reference), atol=1e-6, rtol=1e-6)
    assert field.metadata["source_metrics"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
