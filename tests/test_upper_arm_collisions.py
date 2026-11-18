"""Arm geometry is restored before cloning, without changing Lower's collision model."""

import os
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils

from g1.tasks.upper_limb_agent.config.g1.utils.collisions import (
    ARM_COLLISION_LINK_NAMES,
    restore_arm_collisions,
    spawn_g1_with_arm_collisions,
)


@pytest.fixture
def simplified_asset(tmp_path):
    """A referenced collision library matches the composition used by the actual G1."""
    path = tmp_path / "simplified.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/G1").GetPrim())
    source = UsdGeom.Xform.Define(stage, "/colliders/arm").GetPrim()
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, source.GetPath().AppendChild("shape")).GetPrim())
    for name in (*ARM_COLLISION_LINK_NAMES, "left_hip_pitch_link", "torso_link", "left_ankle_roll_link"):
        body = UsdGeom.Xform.Define(stage, f"/G1/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        collision = UsdGeom.Xform.Define(stage, body.GetPath().AppendChild("collisions")).GetPrim()
        collision.GetReferences().AddInternalReference(source.GetPath())
        collision.SetInstanceable(True)
        collision.SetActive(name in ("torso_link", "left_ankle_roll_link"))
    stage.GetRootLayer().Save()
    return path


def enabled_colliders(root):
    return [
        p
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
    ]


def assert_restored(robot):
    stage = robot.GetStage()
    for name in ARM_COLLISION_LINK_NAMES:
        root = stage.GetPrimAtPath(robot.GetPath().AppendPath(f"{name}/collisions"))
        assert root.IsActive() and len(enabled_colliders(root)) == 1
        assert enabled_colliders(root)[0].IsInstanceProxy()
    assert not stage.GetPrimAtPath(robot.GetPath().AppendPath("left_hip_pitch_link/collisions")).IsActive()
    assert len(enabled_colliders(robot)) == 14


def test_restores_referenced_colliders_idempotently_without_editing_the_asset(simplified_asset):
    original = simplified_asset.read_bytes()
    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    robot.GetReferences().AddReference(str(simplified_asset))
    restore_arm_collisions(robot)
    assert_restored(robot)
    authored = stage.GetRootLayer().ExportToString()
    restore_arm_collisions(robot)
    assert stage.GetRootLayer().ExportToString() == authored
    assert simplified_asset.read_bytes() == original


def test_spawner_restores_each_clone_before_contact_reporting(simplified_asset):
    stage = Usd.Stage.CreateInMemory()
    for index in range(3):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{index}")
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(simplified_asset), func=spawn_g1_with_arm_collisions, activate_contact_sensors=True
    )
    with sim_utils.use_stage(stage):
        cfg.func("/World/envs/env_.*/Robot", cfg, translation=(1.0, 2.0, 3.0))
    for index in range(3):
        robot = stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot")
        assert_restored(robot)
        for name in ARM_COLLISION_LINK_NAMES:
            body = stage.GetPrimAtPath(robot.GetPath().AppendChild(name))
            assert "PhysxContactReportAPI" in body.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
            assert body.GetAttribute("physxContactReport:threshold").Get() == 0.0
        assert robot.GetAttribute("xformOp:translate").Get() == (1.0, 2.0, 3.0)


def test_missing_collision_root_reports_asset_error_before_restoring_other_nodes(simplified_asset):
    stage = Usd.Stage.Open(str(simplified_asset))
    last = ARM_COLLISION_LINK_NAMES[-1]
    stage.RemovePrim(f"/G1/{last}/collisions")
    with pytest.raises(RuntimeError, match=f"{last}/collisions"):
        restore_arm_collisions(stage.GetDefaultPrim())
    first = ARM_COLLISION_LINK_NAMES[0]
    assert not stage.GetPrimAtPath(f"/G1/{first}/collisions").IsActive()


def test_disabled_collision_shape_is_reported_instead_of_silently_spawning_a_visual_only_arm(simplified_asset):
    stage = Usd.Stage.Open(str(simplified_asset))
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/colliders/arm/shape")).GetCollisionEnabledAttr().Set(False)
    with pytest.raises(RuntimeError, match="no enabled collider"):
        restore_arm_collisions(stage.GetDefaultPrim())


def test_fixed_and_mobile_upper_use_repaired_spawner_and_lower_keeps_its_existing_asset():
    from g1.tasks.lower_limb_agent.config.g1.utils.assets import G1_29DOF_CFG as lower
    from g1.tasks.upper_limb_agent.config.g1.env_cfg import UpperLimbAgentEnvCfg
    from g1.tasks.upper_limb_agent.config.g1.mobile_env_cfg import UpperLimbMobileEnvCfg

    for config_type in (UpperLimbAgentEnvCfg, UpperLimbMobileEnvCfg):
        cfg = config_type()
        assert cfg.scene.robot.spawn.func is spawn_g1_with_arm_collisions
        assert cfg.scene.robot.spawn.variants == {"Physics": "SimplifiedPhysX"}
        cfg.teleop_mode()
        assert cfg.scene.robot.spawn.func is spawn_g1_with_arm_collisions
    assert lower.spawn.func in (sim_utils.spawn_from_usd, "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("G1_COLLISION_ASSET_PATH"), reason="Opt-in: requires a local G1 USD asset")
def test_actual_g1_restores_only_twelve_arm_colliders():
    from g1.tasks.upper_limb_agent.config.g1.utils.assets import mobile_robot_cfg
    from g1.tasks.upper_limb_agent.mdp.contacts import ARM_CONTACT_BODY_PATHS

    path = Path(os.environ["G1_COLLISION_ASSET_PATH"])
    assert path.is_file()
    source = Usd.Stage.Open(str(path))
    assert len(enabled_colliders(source.GetDefaultPrim())) == 31
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    # Reference the already resolved local asset directly; avoid another recursive asset download.
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot").GetPrim().GetReferences().AddReference(str(path))
    cfg = mobile_robot_cfg().spawn
    cfg.usd_path = str(path)
    cfg.fix_root_link = None  # Root topology requires a simulator; geometry and schemas do not.
    with sim_utils.use_stage(stage):
        cfg.func("/World/envs/env_0/Robot", cfg)
    robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    assert len(enabled_colliders(robot)) == 43
    for name in ARM_COLLISION_LINK_NAMES:
        assert len(enabled_colliders(stage.GetPrimAtPath(robot.GetPath().AppendChild(name)))) == 1
    assert not stage.GetPrimAtPath(robot.GetPath().AppendPath("left_hip_pitch_link/collisions")).IsActive()
    pelvis = stage.GetPrimAtPath(robot.GetPath().AppendChild("pelvis"))
    assert pelvis.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert pelvis.GetAttribute("newton:selfCollisionEnabled").Get() is True
    for relative_path in ARM_CONTACT_BODY_PATHS:
        body = stage.GetPrimAtPath(robot.GetPath().AppendPath(relative_path))
        assert body and body.HasAPI(UsdPhysics.RigidBodyAPI), relative_path
    assert len(enabled_colliders(source.GetDefaultPrim())) == 31
