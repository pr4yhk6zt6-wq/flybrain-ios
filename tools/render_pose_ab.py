#!/usr/bin/env python3
"""
Draw the stance the way the phone draws it — twice: as the app drew it before
this fix, and as it draws it now.

The screen in `uploads/IMG_2713.png` showed the animal's parts scattered away
from a body that was standing perfectly (feet 6/6, COM z −0.0228 cm, six
contacts). Nothing in the physics, the asset or the solver was wrong: the
app's matrix → quaternion conversion was returning the *conjugate* of the
rotation, so SceneKit spun every part the wrong way about its own origin. This
draws the same pose with both readings, through the app's own camera, from the
same packed meshes and colours the app carries.

    # the pose dump: /tmp/golden build/fly_body.json build/fly_golden.json visual > /tmp/visual.txt
    python3 tools/render_pose_ab.py --pose /tmp/visual.txt \\
        --png docs/img/pose_conjugate_ab.png

`--pose` is the `visual` mode of `local/main.swift`: `V k name x y z` then the
nine numbers of the part's world rotation, in `FlyDynamics.Mat3` order — three
*columns* (c0, c1, c2), which is the reading that reproduces MuJoCo to 1e-15 cm
(`tools/audit_meshes.py`).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import render_still as rs            # the CPU rasteriser the reports use


def read_pose(path: pathlib.Path):
    """`V` rows as (name, position, matrix-in-FlyDynamics.order)."""
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if not p or p[0] != "V":
            continue
        out.append((p[2], np.array(p[3:6], float),
                    np.array(p[6:15], float).reshape(3, 3).T))
    return out


def mesh_soup(world_path: pathlib.Path, meshes_path: pathlib.Path):
    """The app's packed geometry: per part, vertices and triangles, + colour."""
    world = json.loads(world_path.read_text())
    raw = np.fromfile(meshes_path, dtype=np.float32)
    nV = world["body_geometry"]["n_vertex"]
    verts = raw[:nV * 3].reshape(nV, 3).astype(float)
    idx = raw[nV * 6:].view(np.int32)
    parts = []
    by_mesh = {m["name"]: m for m in world["body_geometry"]["meshes"]}
    for g in world["geoms"]:
        m = g.get("mesh")
        if m is None or m >= len(world["body_geometry"]["meshes"]):
            continue
        defn = world["body_geometry"]["meshes"][m]
        v0, vn = defn["vert"]
        f0, fn = defn["face"]
        faces = idx[f0 * 3:(f0 + fn) * 3].reshape(-1, 3) - v0
        rgb = g["rgba"][:3]
        parts.append((g["name"], verts[v0:v0 + vn], faces,
                      np.array([min(1.0, c * 1.6) for c in rgb])))
    return parts


def soup(parts, poses, conjugate: bool):
    """Triangles in world coordinates, drawn the way SceneKit draws them.

    The phone hands SceneKit a quaternion per part and the raw vertices; a
    quaternion rotates a vector as `v + 2w(q×v) + 2q×(q×v)`. Composing that
    with the mesh is what these two lines do explicitly, so the difference
    between the correct rotation and its conjugate is exactly what the screen
    showed.
    """
    tris, cols = [], []
    for (name, V, F, rgb), (_, pos, R) in zip(parts, poses):
        M = R.T if conjugate else R          # the bug was R.T
        tris.append((V @ M.T + pos)[F])
        cols.append(np.repeat(rgb[None, :], len(F), axis=0))
    return np.concatenate(tris), np.concatenate(cols)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default=pathlib.Path("/tmp/visual.txt"), type=pathlib.Path)
    ap.add_argument("--world", default=pathlib.Path("build/world.json"), type=pathlib.Path)
    ap.add_argument("--meshes", default=pathlib.Path("build/fly_meshes.bin"), type=pathlib.Path)
    ap.add_argument("--png", default=pathlib.Path("docs/img/pose_conjugate_ab.png"),
                    type=pathlib.Path)
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--fov", type=float, default=55.0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = read_pose(args.pose)
    parts = mesh_soup(args.world, args.meshes)
    if len(poses) != len(parts):
        print(f"pose dump has {len(poses)} parts, the world has {len(parts)}",
              file=sys.stderr)
        return 1
    print(f"{len(parts)} parts, {sum(len(p[2]) for p in parts):,} triangles")

    # The app's camera: azimuth 2.2 rad, elevation 0.42 rad, and close enough
    # that the animal fills the frame (WorldView's defaults).
    lookat = np.array([0.0, 0.0, -0.02])
    cam = rs.camera(lookat, 0.55, np.degrees(2.2), np.degrees(0.42), fov=args.fov)
    floor_z = -0.035

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), dpi=110)
    titles = ["what the phone drew — conjugate",
              "what it draws now — the rotation itself"]
    for ax, conjugate, title in zip(axes, (True, False), titles):
        tri, col = soup(parts, poses, conjugate)
        img = rs.render(tri, col, cam, w=args.size, h=args.size, floor_z=floor_z)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("the same stance, the same meshes, the same camera — one sign",
                 fontsize=12)
    fig.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
