"""Restore G1 arm colliders before contact reporting and environment cloning."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

    import isaaclab.sim as sim_utils


ARM_COLLISION_LINK_NAMES = tuple(
    f"{side}_{part}_link"
    for side in ("left", "right")
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch")
)


def restore_arm_collisions(robot_prim: Usd.Prim) -> None:
    """Reactivate existing collision subtrees without changing the simplified leg geometry."""
    from pxr import Usd, UsdPhysics

    stage = robot_prim.GetStage()
    paths = [robot_prim.GetPath().AppendPath(f"{name}/collisions") for name in ARM_COLLISION_LINK_NAMES]
    roots = [stage.GetPrimAtPath(path) for path in paths]
    missing = [str(path) for path, prim in zip(paths, roots, strict=True) if not prim]
    if missing:
        raise RuntimeError(f"G1 arm collision geometry is missing: {', '.join(missing)}")
    for root in roots:
        root.SetActive(True)
        if not any(
            prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        ):
            raise RuntimeError(f"G1 arm collision subtree has no enabled collider: {root.GetPath()}")


def spawn_g1_with_arm_collisions(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Load USD code at spawn time, then restore the prototype before reporting and cloning."""
    # Task configs are resolved before SimulationApp starts. The USD spawner imports pxr
    # through physics materials, so importing it at module scope loads the wrong USD runtime.
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file

    @sim_utils.clone
    def spawn(prim_path, cfg, **spawn_kwargs):
        prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, **spawn_kwargs)
        restore_arm_collisions(prim)
        return prim

    return spawn(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
