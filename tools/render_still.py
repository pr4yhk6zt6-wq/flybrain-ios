#!/usr/bin/env python3
"""
Draw the animal without a graphics card.

MuJoCo's renderer wants an OpenGL context, and a CI machine does not have
one. This is a small z-buffer rasteriser over the body model's own meshes, so
the pictures in the reports can be made anywhere. It draws what the model
actually contains — the same OBJ meshes Janelia/DeepMind published — at the
pose the simulation produced.

    python3 tools/render_still.py --out docs/img/flybody_model.png

There is no lighting model worth the name here: one lamp, ambient, and the
surface's own colour. It is a picture of the geometry, not a render of a scene.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def triangle_soup(model, data, mj, grid=3e-4):
    """
    Every visible triangle in the animal, in world coordinates, with the
    colour the model gives it.
    """
    va, fa = model.mesh_vertadr, model.mesh_faceadr
    nvt, nft = model.mesh_vert.shape[0], model.mesh_face.shape[0]
    tris, cols = [], []
    for i in range(model.ngeom):
        if int(model.geom_type[i]) != 7:
            continue
        if model.geom_rgba[i][3] <= 0.01:
            continue
        mi = int(model.geom_dataid[i])
        v0, v1 = va[mi], (va[mi + 1] if mi + 1 < model.nmesh else nvt)
        f0, f1 = fa[mi], (fa[mi + 1] if mi + 1 < model.nmesh else nft)
        v = np.asarray(model.mesh_vert[v0:v1], dtype=np.float64)
        f = np.asarray(model.mesh_face[f0:f1], dtype=np.int64)

        # MuJoCo's geom_xpos / geom_xmat are the geom in the world frame and
        # already contain the geom's own pos and quat, so the only thing left
        # to apply is the mesh's scale. Adding the local rotation a second
        # time is the classic way to get an animal whose legs point inwards.
        Rw = np.asarray(data.geom_xmat[i], dtype=np.float64).reshape(3, 3)
        s = np.asarray(model.geom_size[i], dtype=np.float64)
        v = (Rw @ (v * s).T).T + np.asarray(data.geom_xpos[i], dtype=np.float64)

        mi = int(model.geom_matid[i]) if model.nmat else -1
        rgba = model.mat_rgba[mi] if 0 <= mi < model.nmat else model.geom_rgba[i]
        tris.append(v[f].reshape(-1, 9))
        cols.append(np.tile(np.asarray(rgba[:3], dtype=np.float64), (len(f), 1)))
    if not tris:
        raise SystemExit("the model has no visible mesh geometry")
    return np.concatenate(tris), np.concatenate(cols)


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------------------
# the camera
# ---------------------------------------------------------------------------

def camera(lookat, distance, azimuth, elevation, fov=38.0):
    """A pinhole camera on a sphere around `lookat`. Angles in degrees."""
    az, el = np.radians(azimuth), np.radians(elevation)
    eye = np.array([lookat[0] + distance * np.cos(el) * np.cos(az),
                    lookat[1] + distance * np.cos(el) * np.sin(az),
                    lookat[2] + distance * np.sin(el)])
    fwd = np.asarray(lookat, float) - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    return eye, right, true_up, fwd, np.tan(np.radians(fov) / 2.0)


def project(pts, eye, right, up, fwd, tan_half, w, h):
    rel = pts - eye
    x = rel @ right
    y = rel @ up
    z = rel @ fwd                       # depth, positive in front
    sx = (x / (z * tan_half) * 0.5 + 0.5) * w
    sy = (0.5 - y / (z * tan_half) * 0.5 * (h / w) * (w / h)) * h
    sy = (0.5 - (y / (z * tan_half)) * 0.5) * h
    return sx, sy, z


# ---------------------------------------------------------------------------
# rasterising
# ---------------------------------------------------------------------------

LIGHT = np.array([0.45, -0.75, 0.85])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def render(tri, col, cam, w=560, h=560, background=(0.97, 0.97, 0.95),
           floor_z=None, floor_colour=(0.88, 0.87, 0.84)):
    eye, right, up, fwd, tan_half = cam
    v = tri.reshape(-1, 3, 3)

    p = v.reshape(-1, 3)
    sx, sy, z = project(p, eye, right, up, fwd, tan_half, w, h)
    sx = sx.reshape(-1, 3)
    sy = sy.reshape(-1, 3)
    z = z.reshape(-1, 3)

    # face normals and flat shading, in world space
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 0, nrm / np.maximum(ln, 1e-12), 0.0)
    centre = v.mean(axis=1)
    to_eye = eye - centre
    to_eye /= np.maximum(np.linalg.norm(to_eye, axis=1, keepdims=True), 1e-12)
    facing = (nrm * to_eye).sum(axis=1)
    nrm = np.where((facing < 0)[:, None], -nrm, nrm)
    lam = np.abs(nrm @ LIGHT)
    shade = (0.30 + 0.70 * lam)[:, None]

    img = np.tile(np.asarray(background, float), (h, w, 1))
    zb = np.full((h, w), np.inf)

    # the floor, as a half-plane the camera can see: cheap and enough
    if floor_z is not None:
        # one ray per pixel: the ground is a plane, so this is closed-form
        yy, xx = np.mgrid[0:h, 0:w]
        dirs = ((right[None, None, :] * ((xx - w / 2) / (w / 2) * tan_half)[..., None]
                 + up[None, None, :] * (-(yy - h / 2) / (h / 2) * tan_half)[..., None]
                 + fwd[None, None, :])).reshape(-1, 3)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        hit = np.where(dirs[:, 2] < -1e-9,
                       (floor_z - eye[2]) / np.minimum(dirs[:, 2], -1e-9), np.inf)
        hit = hit.reshape(h, w)
        ok = np.isfinite(hit) & (hit > 0) & (hit < zb)
        # a grid, so the ground reads as ground
        safe = np.where(np.isfinite(hit), hit, 0.0)
        px = eye[0] + dirs[:, 0].reshape(h, w) * safe
        py = eye[1] + dirs[:, 1].reshape(h, w) * safe
        g = (np.floor(px / 0.05) + np.floor(py / 0.05)) % 2
        shaded = (np.asarray(floor_colour)[None, None, :]
                  * (0.62 + 0.38 * g)[..., None])
        img = np.where(ok[..., None], shaded, img)
        zb = np.where(ok, hit, zb)

    xs = np.stack([sx[:, 0], sx[:, 1], sx[:, 2]], axis=1)
    ys = np.stack([sy[:, 0], sy[:, 1], sy[:, 2]], axis=1)
    x0 = np.clip(np.floor(xs.min(axis=1)).astype(int), 0, w - 1)
    x1 = np.clip(np.ceil(xs.max(axis=1)).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(ys.min(axis=1)).astype(int), 0, h - 1)
    y1 = np.clip(np.ceil(ys.max(axis=1)).astype(int), 0, h - 1)
    keep = (x1 >= x0) & (y1 >= y0) & (z > 0.02).all(axis=1)

    rgb = np.clip(col * shade, 0.0, 1.0)
    for i in np.flatnonzero(keep):
        tx, ty = xs[i], ys[i]
        area = (tx[1] - tx[0]) * (ty[2] - ty[0]) - (tx[2] - tx[0]) * (ty[1] - ty[0])
        if abs(area) < 1e-12:
            continue
        gx0, gx1, gy0, gy1 = x0[i], x1[i] + 1, y0[i], y1[i] + 1
        px = np.arange(gx0, gx1) + 0.5
        py = np.arange(gy0, gy1) + 0.5
        PX, PY = np.meshgrid(px, py)
        w0 = ((tx[1] - PX) * (ty[2] - PY) - (tx[2] - PX) * (ty[1] - PY)) / area
        w1 = ((tx[2] - PX) * (ty[0] - PY) - (tx[0] - PX) * (ty[2] - PY)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = w0 * z[i, 0] + w1 * z[i, 1] + w2 * z[i, 2]
        m = inside & (zz < zb[gy0:gy1, gx0:gx1])
        if not m.any():
            continue
        zb[gy0:gy1, gx0:gx1] = np.where(m, zz, zb[gy0:gy1, gx0:gx1])
        img[gy0:gy1, gx0:gx1] = np.where(m[..., None], rgb[i], img[gy0:gy1, gx0:gx1])
    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default="data/flybody/flybody-main/flybody/"
                                      "fruitfly/assets/floor.xml", type=pathlib.Path)
    ap.add_argument("--out", default="docs/img/flybody_model.png", type=pathlib.Path)
    ap.add_argument("--settle-ms", type=int, default=200,
                    help="let it fall onto the floor first")
    ap.add_argument("--size", type=int, default=520)
    args = ap.parse_args()

    import mujoco
    m = mujoco.MjModel.from_xml_path(str(args.mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(args.settle_ms * 10):
        mujoco.mj_step(m, d)

    tri, col = triangle_soup(m, d, mujoco)
    print(f"  {len(tri):,} triangles from the model's own meshes")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    look = [0.0, 0.0, float(d.subtree_com[1][2]) + 0.03]
    views = [
        ("side", dict(distance=0.95, azimuth=90, elevation=-10)),
        ("top", dict(distance=0.95, azimuth=90, elevation=-88)),
        ("front", dict(distance=0.88, azimuth=180, elevation=-12)),
        ("head", dict(distance=0.24, azimuth=120, elevation=-14)),
    ]
    fig, ax = plt.subplots(1, len(views), figsize=(4.6 * len(views), 4.6))
    floor_z = float(m.geom_pos[0][2])
    for a, (name, c) in zip(np.atleast_1d(ax), views):
        lk = [0.045, 0.0, -0.05] if name == "head" else look
        cam = camera(lk, **c)
        im = render(tri, col, cam, w=args.size, h=args.size, floor_z=floor_z)
        a.imshow(im)
        a.set_title(name, fontsize=14)
        a.axis("off")
    fig.suptitle("The real flybody model (Janelia / Google DeepMind, "
                 "Vaxenburg et al.) - drawn from data/flybody/ in this repo",
                 fontsize=14)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
