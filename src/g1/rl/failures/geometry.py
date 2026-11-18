"""USD inspection is called only after launch_simulation has selected the USD runtime."""

import numpy as np


def contact_filters(env_cfg):
    from pxr import Usd, UsdPhysics

    from isaaclab.utils.assets import retrieve_file_path

    path = retrieve_file_path(env_cfg.scene.robot.spawn.usd_path)
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("G1 USD has no default prim")
    bodies = sorted(
        str(p.GetPath().MakeRelativePath(root.GetPath()))
        for p in Usd.PrimRange(root)
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() != "torso_link"
    )
    ground_stage = Usd.Stage.Open(retrieve_file_path(env_cfg.scene.ground.spawn.usd_path))
    ground_root = ground_stage.GetDefaultPrim()
    if not ground_root:
        raise ValueError("Ground USD has no default prim")
    planes = [
        str(p.GetPath().MakeRelativePath(ground_root.GetPath()))
        for p in Usd.PrimRange(ground_root)
        if p.GetTypeName() == "Plane"
    ]
    if not bodies or not planes:
        raise ValueError("Could not resolve G1 rigid bodies or the actual ground-plane collision path")
    labels = bodies + [f"ground/{p}" for p in planes]
    paths = [f"{{ENV_REGEX_NS}}/Robot/{p}" for p in bodies]
    paths += [f"{env_cfg.scene.ground.prim_path}/{p}" for p in planes]
    return labels, paths


def export_geometry(robot_prim, body_names):
    """Export visual meshes in rigid-body coordinates once per run, without external textures."""
    from pxr import Usd, UsdGeom, UsdPhysics

    body_index = {name: index for index, name in enumerate(body_names)}
    cache = UsdGeom.XformCache()
    meshes, edges = [], set()
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if a and b and a[0].name in body_index and b[0].name in body_index:
                edges.add((body_index[a[0].name], body_index[b[0].name]))
        if not prim.IsA(UsdGeom.Mesh) or "/collisions/" in str(prim.GetPath()):
            continue
        body = prim.GetParent()
        while body and body != robot_prim and body.GetName() not in body_index:
            body = body.GetParent()
        if not body or body == robot_prim:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if points.ndim != 2 or not len(points) or counts is None or indices is None:
            continue
        # The G1 visual meshes duplicate vertices per face. Weld coincident points to
        # reduce the archive/viewer size without changing the surface geometry.
        points, inverse = np.unique(points, axis=0, return_inverse=True)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse())
        points = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        triangles, offset = [], 0
        for count in counts:
            face = inverse[np.asarray(indices[offset : offset + count], dtype=np.int64)]
            triangles.extend([int(face[0]), int(face[i]), int(face[i + 1])] for i in range(1, count - 1))
            offset += count
        if not triangles:
            continue
        name = body.GetName()
        color = [0.36, 0.60, 0.82] if name.startswith("left_") else [0.82, 0.56, 0.34]
        if not name.startswith(("left_", "right_")):
            color = [0.65, 0.69, 0.73]
        meshes.append(
            {
                "body": body_index[name],
                "vertices": points.round(6).tolist(),
                "triangles": triangles,
                "color": color,
            }
        )
    return {"body_names": body_names, "edges": sorted(edges), "meshes": meshes, "quaternion_order": "xyzw"}
