"""Use real USD composition to verify contact-channel paths and mesh-local coordinates."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from g1.rl.failures.geometry import contact_filters, export_geometry


def test_contact_filters_resolve_actual_body_and_ground_paths(tmp_path):
    robot_path, ground_path = tmp_path / "robot.usda", tmp_path / "ground.usda"
    stage = Usd.Stage.CreateNew(str(robot_path))
    root = UsdGeom.Xform.Define(stage, "/G1").GetPrim()
    stage.SetDefaultPrim(root)
    for name in ("pelvis", "torso_link", "left_elbow_link", "right_wrist_yaw_link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/G1/{name}").GetPrim())
    stage.GetRootLayer().Save()
    ground = Usd.Stage.CreateNew(str(ground_path))
    ground.SetDefaultPrim(UsdGeom.Xform.Define(ground, "/Ground").GetPrim())
    UsdGeom.Plane.Define(ground, "/Ground/Physics/ActualPlane")
    ground.GetRootLayer().Save()
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            robot=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(robot_path))),
            ground=SimpleNamespace(prim_path="/World/ground", spawn=SimpleNamespace(usd_path=str(ground_path))),
        ),
    )
    labels, paths = contact_filters(cfg)
    assert labels == ["left_elbow_link", "pelvis", "right_wrist_yaw_link", "ground/Physics/ActualPlane"]
    assert paths[-1] == "/World/ground/Physics/ActualPlane"
    assert all("torso_link" not in p for p in paths)


def test_export_geometry_is_relative_to_rigid_body_and_excludes_colliders():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    body = UsdGeom.Xform.Define(stage, "/Robot/torso_link")
    body.AddTranslateOp().Set((100.0, 20.0, 30.0))
    mesh = UsdGeom.Mesh.Define(stage, "/Robot/torso_link/visuals/shape")
    mesh.AddTranslateOp().Set((1.0, 0, 0))
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)])
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([4, 1, 2, 3])
    collider = UsdGeom.Mesh.Define(stage, "/Robot/torso_link/collisions/shape")
    collider.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    collider.GetFaceVertexCountsAttr().Set([3])
    collider.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    data = export_geometry(root, ["torso_link"])
    assert len(data["meshes"]) == 1
    mesh = data["meshes"][0]
    assert mesh["body"] == 0 and len(mesh["vertices"]) == 4
    np.testing.assert_allclose(np.asarray(mesh["vertices"]).min(0), [1, 0, 0])
    np.testing.assert_allclose(np.asarray(mesh["vertices"]).max(0), [2, 1, 0])
    assert len(mesh["triangles"]) == 2


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("G1_COLLISION_ASSET_PATH"), reason="Opt-in: local resolved G1 USD required")
def test_real_g1_visual_geometry_covers_all_links_and_preserves_kinematic_edges():
    stage = Usd.Stage.Open(os.environ["G1_COLLISION_ASSET_PATH"])
    root = stage.GetDefaultPrim()
    names = [p.GetName() for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    data = export_geometry(root, names)
    assert len(names) == 46 and len(data["edges"]) == 45
    represented = {names[mesh["body"]] for mesh in data["meshes"]}
    assert {"torso_link", "left_elbow_link", "right_elbow_link", "pelvis"} <= represented
    assert sum(len(mesh["vertices"]) for mesh in data["meshes"]) < 400000
    assert all(np.isfinite(mesh["vertices"]).all() for mesh in data["meshes"])
