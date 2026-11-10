"""Export a compact 0 kg best-Q map; source metrics remain read-only.

Example from the project root:
    uv run python scripts/prepare_workspace_quality.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "logs/analysis/left_5cm.batch.metrics_0.4ef.yaw-cropped.npz"
DEFAULT_OUTPUT = ROOT / "src/g1/tasks/upper_limb_agent/data/workspace_q0_v1.npz"


def prepare_workspace_quality(source: Path, output: Path, *, overwrite: bool = False) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    sidecar = output.with_suffix(".json")
    protected = (source, source.with_suffix(".json"))
    for destination in (output, sidecar):
        if any(
            destination == path or (destination.exists() and path.exists() and destination.samefile(path))
            for path in protected
        ):
            raise ValueError("Output must be distinct from the source metrics and its sidecar")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"{destination} already exists; use --overwrite to replace an exported map")
    if output.suffix != ".npz":
        raise ValueError("Output must have the .npz suffix")
    source_bytes = source.read_bytes()
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"]))
        points = archive["points"].astype(np.float64)
        point_indices = archive["point_indices"]
        counts = archive["solution_counts"]
        quality = archive["composite_score"]
    if meta["kind"] != "g1_batch_configuration_metrics" or meta["schema_version"] != 2:
        raise ValueError("Expected completed v2 batch metrics with stored composite scores")
    if meta["coordinate_frame"] != "pelvis" or meta["target_link"] != "left_wrist_yaw_link":
        raise ValueError("The initial map requires left_wrist_yaw_link in the pelvis frame")
    if float(meta["payload_masses_kg"][0]) != 0:
        raise ValueError("The first stored scenario must be the 0 kg load")
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or quality.ndim != 2 or quality.shape[1] != 3:
        raise ValueError("Invalid point or composite-score shapes")
    if (
        counts.shape != (len(points),)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or int(counts.sum()) != len(quality)
        or not np.array_equal(point_indices, np.repeat(np.arange(len(points)), counts))
        or meta["completed"] != len(quality)
        or meta["composite_completed"] != len(quality)
    ):
        raise ValueError("Metrics completion, counts and point/sample mapping are inconsistent")
    q0 = quality[:, 0]
    if not np.isfinite(points).all() or not np.isfinite(q0).all():
        raise ValueError("Points and 0 kg scores must be finite")
    step = float(meta["source_ik_signature"]["grid"]["step_m"])
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Grid step must be positive and finite")
    best = np.full(len(points), -np.inf)
    np.maximum.at(best, point_indices, q0)
    present = counts > 0
    if not present.any() or float(q0.max()) <= 0:
        raise ValueError("A positive Q reference and at least one saved configuration are required")
    origin = points.min(axis=0)
    coordinates = np.rint((points - origin) / step).astype(np.int64)
    if not np.allclose(origin + coordinates * step, points, atol=2e-6, rtol=0):
        raise ValueError("Points do not lie on the declared regular grid")
    shape = tuple((coordinates.max(axis=0) + 1).tolist())
    if min(shape) < 2 or np.prod(shape) > 10_000_000:
        raise ValueError("Expected a bounded 3D grid with at least two nodes per axis")
    linear = np.ravel_multi_index(coordinates.T, shape)
    if len(np.unique(linear)) != len(points):
        raise ValueError("Duplicate spatial grid nodes")
    grid = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    grid.reshape(-1)[linear[present]] = best[present]
    valid.reshape(-1)[linear[present]] = True
    document = {
        "schema_version": 1,
        "kind": "g1_workspace_quality",
        "coordinate_frame": "pelvis",
        "target_link": "left_wrist_yaw_link",
        "source_side": "left",
        "right_mirror_axis": "y",
        "payload_mass_kg": 0.0,
        "aggregation": "max_all_saved_configurations",
        "feasible_only": False,
        "q_reference": float(q0.max()),
        "normalization": "clip(Q / fixed_source_max_Q0, 0, 1)",
        "origin_xyz_m": origin.tolist(),
        "step_xyz_m": [step] * 3,
        "shape_xyz": list(shape),
        "array_axis_order": "xyz",
        "covered_nodes": int(present.sum()),
        "source_samples": len(quality),
        "source_metrics": {"filename": source.name, "sha256": hashlib.sha256(source_bytes).hexdigest()},
        "source_composite_metrics": meta["composite_metrics"],
        "meaning": "Best saved 0 kg configuration score per position; missing nodes are unknown",
    }
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    for destination, arrays in ((output, True), (sidecar, False)):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                if arrays:
                    np.savez_compressed(stream, quality=grid, valid=valid, metadata_json=np.array(encoded))
                else:
                    stream.write(encoded.encode("utf-8"))
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    document = prepare_workspace_quality(args.source, args.output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "covered_nodes": document["covered_nodes"],
                "q_reference": document["q_reference"],
            }
        )
    )


if __name__ == "__main__":
    main()
