"""Immutable spatial Q prior: map loading, interpolation, mirroring and position gating."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils import configclass

__all__ = ["WorkspaceQualityCfg", "WorkspaceQualityMap", "WorkspaceQualityState"]


@configclass
class WorkspaceQualityCfg:
    map_path: str = "data/workspace_q0_v1.npz"
    """Absolute path, or a path relative to the upper_limb_agent package directory."""
    position_tolerance_m: float = 0.01
    """All active wrists must satisfy this Euclidean position tolerance to earn Q."""


@dataclass
class WorkspaceQualityState:
    target_scores: torch.Tensor
    target_valid: torch.Tensor
    actual_scores: torch.Tensor
    actual_valid: torch.Tensor
    position_ready: torch.Tensor
    map_covered: torch.Tensor
    reward: torch.Tensor


class WorkspaceQualityMap:
    """A small, device-resident XYZ grid loaded once per command term.

    Missing/outside nodes remain unknown. Interpolation is valid only if every
    corner with nonzero weight is covered; exact known nodes remain queryable
    even when an adjacent zero-weight corner is unknown.
    """

    def __init__(self, path: str | Path, device: str | torch.device):
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            self.path = Path(__file__).resolve().parents[1] / self.path
        self.path = self.path.resolve()
        self.archive_bytes = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.archive_bytes).hexdigest()
        with np.load(io.BytesIO(self.archive_bytes), allow_pickle=False) as archive:
            quality, valid = archive["quality"], archive["valid"]
            self.metadata = json.loads(str(archive["metadata_json"]))
        meta = self.metadata
        if (
            meta.get("schema_version") != 1
            or meta.get("kind") != "g1_workspace_quality"
            or meta.get("coordinate_frame") != "pelvis"
            or meta.get("target_link") != "left_wrist_yaw_link"
            or meta.get("payload_mass_kg") != 0
            or meta.get("aggregation") != "max_all_saved_configurations"
            or meta.get("right_mirror_axis") != "y"
            or meta.get("array_axis_order") != "xyz"
        ):
            raise ValueError("Workspace Q metadata must describe the pelvis-frame left-wrist 0 kg best-Q map")
        self.q_reference = float(meta["q_reference"])
        origin, step = np.asarray(meta["origin_xyz_m"]), np.asarray(meta["step_xyz_m"])
        if (
            quality.ndim != 3
            or min(quality.shape) < 2
            or list(quality.shape) != meta["shape_xyz"]
            or valid.shape != quality.shape
            or valid.dtype != np.bool_
            or not valid.any()
            or not np.isfinite(quality).all()
            or not math.isfinite(self.q_reference)
            or self.q_reference <= 0
            or np.max(quality[valid]) > self.q_reference * (1 + 1e-6)
            or origin.shape != (3,)
            or step.shape != (3,)
            or not np.isfinite(origin).all()
            or not np.isfinite(step).all()
            or np.any(step <= 0)
        ):
            raise ValueError("Invalid workspace Q arrays, grid geometry or normalization reference")
        self._quality = torch.tensor(quality.reshape(-1), device=device, dtype=torch.float32)
        self._valid = torch.tensor(valid.reshape(-1), device=device, dtype=torch.bool)
        self._origin = torch.tensor(origin, device=device, dtype=torch.float32)
        self._step = torch.tensor(step, device=device, dtype=torch.float32)
        self._shape = torch.tensor(quality.shape, device=device, dtype=torch.long)
        self._corners = torch.tensor(
            [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=device, dtype=torch.long
        )
        self._strides = torch.tensor([quality.shape[1] * quality.shape[2], quality.shape[2], 1], device=device)

    def query(self, positions_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized scores and a separate validity mask for (..., 3) positions."""
        if positions_b.shape[-1] != 3:
            raise ValueError("Workspace Q positions must end in XYZ")
        positions = positions_b.reshape(-1, 3)
        finite = torch.isfinite(positions).all(-1)
        safe = torch.where(finite[:, None], positions, self._origin)
        coordinates = (safe - self._origin) / self._step
        # Float32 source coordinates differ slightly from exact 5 cm nodes.
        rounded = coordinates.round()
        coordinates = torch.where((coordinates - rounded).abs() < 1e-5, rounded, coordinates)
        inside = finite & ((coordinates >= 0) & (coordinates <= self._shape - 1)).all(-1)
        lower = torch.minimum(coordinates.floor().long().clamp_min(0), self._shape - 2)
        fraction = (coordinates - lower).clamp(0, 1)
        corners = lower[:, None] + self._corners
        linear = (corners * self._strides).sum(-1)
        weights = torch.where(self._corners.bool(), fraction[:, None], 1 - fraction[:, None]).prod(-1)
        valid = inside & (self._valid[linear] | (weights <= 1e-6)).all(-1)
        scores = ((self._quality[linear] * weights).sum(-1) / self.q_reference).clamp(0, 1)
        scores = torch.where(valid, scores, 0)
        return scores.reshape(positions_b.shape[:-1]), valid.reshape(positions_b.shape[:-1])

    def evaluate(
        self,
        targets_b: torch.Tensor,
        actual_b: torch.Tensor,
        position_errors_m: torch.Tensor,
        active: torch.Tensor,
        tolerance_m: float,
    ) -> WorkspaceQualityState:
        """Gate the mean active-target score on all active wrists' position errors.

        Inputs have wrist order left/right. Only positions participate; the right
        side queries (x,-y,z) in the left map. No controller state is modified.
        """
        if not math.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError("Workspace Q position tolerance must be positive and finite")
        targets, actual = targets_b.clone(), actual_b.clone()
        targets[:, 1, 1] *= -1
        actual[:, 1, 1] *= -1
        target_scores, target_valid = self.query(targets)
        actual_scores, actual_valid = self.query(actual)
        count = active.sum(-1)
        ready = ((position_errors_m <= tolerance_m) | ~active).all(-1) & (count > 0)
        covered = (target_valid | ~active).all(-1) & (count > 0)
        score = (target_scores * active).sum(-1) / count.clamp_min(1)
        return WorkspaceQualityState(
            target_scores,
            target_valid,
            actual_scores,
            actual_valid,
            ready,
            covered,
            torch.where(ready & covered, score, 0),
        )
