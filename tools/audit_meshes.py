#!/usr/bin/env python3
"""
Does the app's visual part math put every mesh where MuJoCo puts it?

The app draws part k of the animal at

    pos = body.xpos + body.xmat @ visual.pos
    rot = body.xmat @ quatToMat(visual.quat)          (quat is w, x, y, z)

with the mesh's own vertices, in the mesh's own frame — `FlyDynamics.visualWorld`.
MuJoCo draws the same geom at `geom_xpos`/`geom_xmat`. This compares the two at
the stance, on the asset the app actually carries (`build/fly_body.json`), so a
scatter on the screen (`uploads/IMG_2713.png`) has a number attached to it
instead of a description.

    python3 tools/audit_meshes.py [--asset build/fly_body.json] [--xml <fruitfly.xml>]
"""

from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

def quat_to_mat(q):
    """(w, x, y, z) -> 3x3, the same formula as FlyDynamics.quatToMat."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default=pathlib.Path("build/fly_body.json"), type=pathlib.Path)
    ap.add_argument("--xml", default=pathlib.Path(
        "data/flybody/flybody-main/flybody/fruitfly/assets/fruitfly.xml"),
        type=pathlib.Path)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    asset = json.loads(args.asset.read_text())

    # --- the stance ---------------------------------------------------------
    free = next((j for j in range(model.njnt)
                 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE), None)
    data.qpos[:] = 0
    if free is not None:
        data.qpos[model.jnt_qposadr[free]:model.jnt_qposadr[free] + 3] = \
            [0.0, 0.0, asset["stance_root_z"]]
        data.qpos[model.jnt_qposadr[free] + 3:model.jnt_qposadr[free] + 7] = [1, 0, 0, 0]
    n_set = 0
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in asset["stance_q"]:
            data.qpos[model.jnt_qposadr[j]] = asset["stance_q"][name]
            n_set += 1
    mujoco.mj_forward(model, data)
    print(f"stance: {n_set} hinges set, root z {asset['stance_root_z']:+.7f}")

    def body_id(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return i

    # --- compare ------------------------------------------------------------
    rows = []
    for k, v in enumerate(asset["visual"]):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, v["name"])
        if g < 0:
            rows.append((v["name"], np.inf, np.inf, -1, -1)); continue
        b = int(model.geom_bodyid[g])
        xpos, xmat = data.xpos[b], data.xmat[b].reshape(3, 3)
        true_pos = data.geom_xpos[g].copy()
        true_mat = data.geom_xmat[g].reshape(3, 3).copy()
        app_pos = xpos + xmat @ np.array(v["pos"])
        app_mat = xmat @ quat_to_mat(np.array(v["quat"]))
        d_pos = float(np.linalg.norm(app_pos - true_pos))
        # rotation difference: the angle of app_mat @ true_mat^T
        cos = (np.trace(app_mat @ true_mat.T) - 1) / 2
        d_ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        # and the body the asset names must be the body the model has
        named = body_id(v["body"]) if v.get("body") in [mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)] else -2
        rows.append((v["name"], d_pos, d_ang, b, named))

    worst_p = max(rows, key=lambda r: r[1])
    worst_a = max(rows, key=lambda r: r[2])
    print(f"parts compared: {len(rows)}")
    print(f"worst position error: {worst_p[1]:.3e} cm  ({worst_p[0]})")
    print(f"worst rotation error: {worst_a[2]:.3f} deg  ({worst_a[0]})")
    n_bad_p = sum(1 for r in rows if r[1] > 1e-6)
    n_bad_a = sum(1 for r in rows if r[2] > 0.05)
    print(f"parts with a position error > 1e-6 cm: {n_bad_p}")
    print(f"parts with a rotation error > 0.05 deg: {n_bad_a}")
    print(f"parts whose named body is not the geom's own body: "
          f"{sum(1 for r in rows if r[3] != r[4])}")
    print(f"\nthe {args.top} worst, by position error:")
    for name, dp, da, b, nb in sorted(rows, key=lambda r: -r[1])[:args.top]:
        print(f"  {name:<28} {dp:>10.3e} cm  {da:>8.3f} deg")
    return 0

if __name__ == "__main__":
    sys.exit(main())
