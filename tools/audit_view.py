#!/usr/bin/env python3
"""
Look at the scene the app draws, from the app's own cameras, without a phone.

IMG_2715 is a screenshot of the Map screen on the device with the whole animal
apparently in pieces: a thorax filling the screen, legs and wings crossing the
frame at odd angles, big areas of flat grey. The panes in the same screenshot
show a fly standing on a grid, correctly assembled. The pose is therefore fine
and something about the *main camera* is not.

This tool settles which, and then keeps it settled. It reads the same three
files the app reads (`world/world.json`, `world/fly.bin`, `world/frames.bin`),
poses the animal exactly the way `FlyWorld.apply(frame:)` does, puts the camera
exactly where `WorldRig.set(azimuth:elevation:distance:)` puts it, and reports
the numbers that decide whether what is on the screen is an animal:

  * the animal's bounding radius about the rig origin (`WorldModel.render`
    centres the rig on the mean of the posed part origins, not on the body);
  * for each camera: how far the nearest surface is in front of the lens, what
    fraction of the animal is in front of the camera, what fraction is inside
    the frustum, and how many screens wide the animal projects to.

It also draws a picture: every mesh vertex as a dot, painted far-to-near, with
the floor grid for scale. That is not a renderer — there is no shading, no
hidden-surface removal within a triangle — it is a picture of *where the
geometry is in the frame*, which is the only question here.

    python3 tools/audit_view.py                     # report + pictures
    python3 tools/audit_view.py --gate              # and fail if a view is broken
    python3 tools/audit_view.py --frame 350

`--gate` is what CI runs, and it encodes the defect IMG_2715 reported: **the
closest the pinch can pull the camera must still frame the body.** At the old
`minDistance` of 0.22 cm the animal's body projected several screens wide and
only ~60% of its vertices were inside the frustum at all — a view that reads as
"the model is in pieces" when the model is fine.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

# The app's own numbers, by name, so a change in one place shows up here.
MAIN_FOV_Y = 38.0            # WorldRig.camera(main, 38, ...)
PANE_FOV_Y = 50.0
Z_NEAR = 0.002               # WorldRig.camera: c.zNear
Z_FAR = 6.0
PHONE = (828, 1792)          # the pixels of IMG_2715; the aspect is what matters
PANE = (134, 106)            # WorldScreen.paneSize, in points

# The body's own parts, by mesh name. "The body" here means the three big
# segments and their plates: the parts a person reads as the animal. Legs,
# wings, antennae and bristles are deliberately not in this list — a close-up
# that crops a wing tip is a close-up; a close-up that crops the thorax is the
# defect IMG_2715 reported.
BODY_PARTS = ("thorax", "head", "abdomen", "head_red", "head_black",
              "thorax_black", "abdomen_black")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load(root: pathlib.Path):
    manifest = json.loads((root / "world.json").read_text())
    blob = (root / "fly.bin").read_bytes()
    bg = manifest["body_geometry"]
    n_v, n_f = bg["n_vertex"], bg["n_face"]
    expect = (n_v * 3 + n_v * 3 + n_f * 3) * 4
    if len(blob) < expect:
        sys.exit(f"fly.bin is {len(blob)} bytes, the manifest needs {expect}")
    verts = np.frombuffer(blob, np.float32, count=n_v * 3,
                          offset=0).reshape(-1, 3).astype(np.float64)
    frames = np.frombuffer((root / "frames.bin").read_bytes(), np.float32)
    return manifest, verts, frames


def quat_matrix(q) -> np.ndarray:
    """The rotation of a quaternion stored (w, x, y, z), as FlyWorld reads it."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose(manifest, verts, frames, index: int):
    """World-space vertices and per-part origins at one recorded frame.

    The mesh table is `[offset, count]` — `thorax` is `[70258, 9158]`, which is
    only computable as an offset and a count — and `FlyWorld.buildGeometries`
    reads it that way. A tool that reads the second number as an end index takes
    a *backwards* slice for every mesh that does not start at zero, i.e. all but
    the head, and measures a differently broken animal.
    """
    parts = manifest["frames"]["parts"]
    stride = parts * 7
    n = manifest["frames"]["n"]
    if not 0 <= index < n:
        sys.exit(f"--frame must be 0..{n - 1}")
    table = frames[index * stride:(index + 1) * stride].reshape(parts, 7)

    n_v = manifest["body_geometry"]["n_vertex"]
    out = np.zeros((n_v, 3))
    owner = np.full(n_v, -1, int)
    origins = np.zeros((len(manifest["geoms"]), 3))
    name_of_part: dict[int, str] = {}
    for i, g in enumerate(manifest["geoms"]):
        origins[i] = table[g["part"]][:3]
        name_of_part[g["part"]] = g["name"]
    for g in manifest["geoms"]:
        if g.get("mesh") is None:
            continue
        vo, vn = manifest["body_geometry"]["meshes"][g["mesh"]]["vert"]
        p = table[g["part"]]
        out[vo:vo + vn] = verts[vo:vo + vn] @ quat_matrix(p[3:]).T + p[:3]
        owner[vo:vo + vn] = g["part"]
    weights = np.array([manifest["body_geometry"]["meshes"][g["mesh"]]["vert"][1]
                        if g.get("mesh") is not None else 0.0
                        for g in manifest["geoms"]], dtype=np.float64)
    return out, owner, origins, name_of_part, table, weights


def rig_centre(origins: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The app's rule for where the rig goes: the mean of the posed part
    origins, weighted by the number of vertices in each part's mesh.

    An unweighted mean sits off the animal, because the legs, wings and
    bristles are many parts with small meshes while the thorax is one part with
    a third of the model's vertices in it. `tools/measure_view.py` derives the
    camera limits about this same point.
    """
    w = weights.sum()
    if w <= 0:
        return origins.mean(axis=0)
    return (origins * (weights / w)[:, None]).sum(axis=0)


def camera_frame(eye: np.ndarray):
    """SceneKit's look-at, as `WorldRig.aim` computes it: the camera's −z axis
    points at the rig origin and the world's +z is up."""
    up = np.array([0.0, 0.0, 1.0])
    z = eye / np.linalg.norm(eye)                  # away from the target
    x = np.cross(up, z)
    xl = np.linalg.norm(x)
    x = np.array([1.0, 0.0, 0.0]) if xl < 1e-5 else x / xl
    y = np.cross(z, x)
    return x, y, z


def measure(verts_about: np.ndarray, eye: np.ndarray, fov_y: float,
            viewport: tuple[int, int], body_mask: np.ndarray) -> dict:
    x, y, z = camera_frame(eye)
    rel = verts_about - eye
    depth = -(rel @ z)                              # in front of the lens
    half_h = np.tan(np.radians(fov_y) / 2)
    half_w = half_h * viewport[0] / viewport[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndx = (rel @ x) / (depth * half_w)
        ndy = (rel @ y) / (depth * half_h)
    front = depth > Z_NEAR
    inframe = front & (np.abs(ndx) <= 1) & (np.abs(ndy) <= 1)

    def span(mask):
        if not mask.any():
            return float("inf")
        return float(max(np.nanmax(ndx[mask]) - np.nanmin(ndx[mask]),
                         np.nanmax(ndy[mask]) - np.nanmin(ndy[mask])) / 2)

    body = body_mask & front
    return {
        "eye": eye,
        "distance": float(np.linalg.norm(eye)),
        "nearest_surface": float(depth.min()) if front.any() else float("inf"),
        "behind_camera_pct": float(100 * (1 - front.mean())),
        "in_frame_pct": float(100 * inframe.mean()),
        "body_in_frame_pct": float(100 * (inframe & body_mask).sum()
                                   / max(1, body_mask.sum())),
        "body_span_screens": span(body),
        "all_span_screens": span(front),
        "ndx": ndx, "ndy": ndy, "front": front, "inframe": inframe,
        "body_mask": body_mask,
    }


def draw(path: pathlib.Path, m: dict, verts_about: np.ndarray, floor_z: float,
         viewport: tuple[int, int], title: str, grid_step: float = 0.027,
         grid_extent: float = 0.6) -> None:
    """Every vertex as a dot, painted far-to-near, with the floor grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = viewport
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_facecolor("#f6f4f0")
    ax.set_xticks([])
    ax.set_yticks([])

    # the floor grid, drawn as lines at floor_z (FlyWorld.dress)
    n = int(grid_extent / grid_step)
    for i in range(-n, n + 1):
        t = i * grid_step
        for p0, p1 in [((grid_extent, t, floor_z), (-grid_extent, t, floor_z)),
                       ((t, grid_extent, floor_z), (t, -grid_extent, floor_z))]:
            seg = np.array([p0, p1])
            sx, sy, sz = (m["ndx"] is not None and
                          project(seg, m) or (None, None, None))
            if sx is None:
                continue
            ax.plot(sx, sy, color="#d8d4cc", lw=0.5, zorder=1)

    order = np.argsort(m["depth_all"])[::-1]        # far first
    ok = m["front_all"][order]
    ax.scatter(m["ndx_all"][order][ok], m["ndy_all"][order][ok], s=1.2,
               c="#8a5a2b", linewidths=0, zorder=2)
    ax.set_title(title, fontsize=7, loc="left", pad=4)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec="#c33",
                               lw=0.6, zorder=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def project(points: np.ndarray, m: dict):
    """Project world points with the same camera the measurement used."""
    x, y, z = m["axes"]
    rel = points - m["eye"]
    depth = -(rel @ z)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndx = (rel @ x) / (depth * m["half_w"])
        ndy = (rel @ y) / (depth * m["half_h"])
    if not np.all(np.isfinite(ndx)):
        return None, None, None
    return ndx, ndy, depth


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=pathlib.Path, default=pathlib.Path("world"))
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("reports/view"))
    ap.add_argument("--gate", action="store_true",
                    help="fail if any camera the app can reach frames the "
                         "animal in a way that reads as broken")
    ap.add_argument("--distance", type=float, action="append", default=[],
                    help="extra distances to measure, in cm (repeatable)")
    args = ap.parse_args()

    manifest, verts, frames = load(args.world)
    posed, owner, origins, name_of_part, table, weights = pose(
        manifest, verts, frames, args.frame)
    centre = rig_centre(origins, weights)
    about = posed - centre
    radius = float(np.linalg.norm(about, axis=1).max())

    body_mask = np.zeros(len(posed), bool)
    for part, name in name_of_part.items():
        if name in BODY_PARTS:
            body_mask[owner == part] = True

    log(f"{args.world}/world.json: {len(manifest['geoms'])} geoms, "
        f"{manifest['body_geometry']['n_vertex']:,} vertices, "
        f"frame {args.frame} of {manifest['frames']['n']}")
    log(f"  rig origin (mean of the posed part origins): {centre.round(4)}")
    # The animal's own longest axis, as the manifest's mesh bounding box has
    # it: that is the number a person means by "the fly is this big".
    span = float(max(posed.max(axis=0) - posed.min(axis=0)))
    log(f"  animal radius about it: {radius:.4f} cm   "
        f"(its longest axis, from the posed meshes, is {span:.3f} cm)")
    log(f"  body parts: {int(body_mask.sum()):,} vertices of "
        f"{len(posed):,}")
    if not body_mask.any():
        log("  WARNING: none of BODY_PARTS matched a geom name in this "
            "manifest — the body-framing check below has nothing to measure")

    # The distances the app uses. Since tools/measure_view.py they live in the
    # manifest (`view`), measured from the animal rather than written in the
    # source; the fallbacks are the v0.7 constants, which is what an asset
    # without the block gets.
    view = manifest.get("view") or {}
    if not view:
        log("  note: this manifest has no `view` block (predates "
            "tools/measure_view.py); measuring the v0.7 constants instead")
    fov_main = float(view.get("fov_y", MAIN_FOV_Y))
    fov_pane = float(view.get("pane_fov_y", PANE_FOV_Y))
    home_d = float(view.get("home_distance_cm", 1.0))
    min_d = float(view.get("min_distance_cm", 0.22))
    max_d = float(view.get("max_distance_cm", 3.0))
    pane_d = float(view.get("pane_distance_cm", 0.306))

    # The cameras the app builds, by name: WorldRig.init and WorldModel.
    home_az, home_el = 2.2, 0.42
    def at(d):
        return np.array([d * np.cos(home_el) * np.cos(home_az),
                         d * np.cos(home_el) * np.sin(home_az),
                         d * np.sin(home_el)])

    pane_dir = np.array([0.03, 0.30, 0.05])
    pane_dir = pane_dir / np.linalg.norm(pane_dir)
    cameras = [
        (f"home  {home_d:.3f} cm", at(home_d), fov_main, PHONE),
        (f"min   {min_d:.3f} cm", at(min_d), fov_main, PHONE),
        (f"max   {max_d:.3f} cm", at(max_d), fov_main, PHONE),
        ("pane  left eye", pane_dir * pane_d, fov_pane, PANE),
        ("pane  right eye", pane_dir * np.array([1.0, -1.0, 1.0]) * pane_d,
         fov_pane, PANE),
    ]
    for d in args.distance:
        cameras.append((f"main  {d:.3f} cm", at(d), fov_main, PHONE))

    log("")
    log(f"    {'camera':<28} {'nearest':>9} {'behind':>8} {'in frame':>9} "
        f"{'body in frame':>14} {'body spans':>11} {'all spans':>10}")
    results = []
    for label, eye, fov, viewport in cameras:
        m = measure(about, eye, fov, viewport, body_mask)
        m["label"] = label
        m["fov"], m["viewport"] = fov, viewport
        m["axes"] = camera_frame(eye)
        m["half_h"] = np.tan(np.radians(fov) / 2)
        m["half_w"] = m["half_h"] * viewport[0] / viewport[1]
        # the per-vertex projection, for the picture
        x, y, z = m["axes"]
        rel = about - eye
        m["depth_all"] = -(rel @ z)
        with np.errstate(divide="ignore", invalid="ignore"):
            m["ndx_all"] = (rel @ x) / (m["depth_all"] * m["half_w"])
            m["ndy_all"] = (rel @ y) / (m["depth_all"] * m["half_h"])
        m["front_all"] = m["depth_all"] > Z_NEAR
        results.append(m)
        log(f"    {label:<28} {m['nearest_surface']:>8.4f}  "
            f"{m['behind_camera_pct']:>7.1f}% {m['in_frame_pct']:>8.1f}% "
            f"{m['body_in_frame_pct']:>13.1f}% "
            f"{m['body_span_screens']:>10.2f}x {m['all_span_screens']:>9.2f}x")
        draw(args.out / (label.split()[0] + "_" +
                         ("_".join(label.split()[1:3]).replace(" ", "").replace(".", "p")
                          or "view") + ".png"),
             m, about, manifest["floor_z"], viewport, label)

    log("")
    log("  'body in frame' is the thorax, head and abdomen and their plates: "
        "the parts a")
    log("  person reads as the animal. A camera that crops them is the defect; a "
        "camera")
    log("  that crops a wing tip is a close-up.")

    # ---- the gate -----------------------------------------------------------
    if not args.gate:
        log("")
        log(f"  wrote pictures to {args.out}/")
        return 0

    problems = []
    by_label = {m["label"].split()[0]: m for m in results}
    home = by_label.get("home")
    if home is not None:
        if home["body_in_frame_pct"] < 99.9:
            problems.append(
                f"home: the body is only {home['body_in_frame_pct']:.1f}% in "
                "frame — the default view must frame the whole animal")
        # The default view crops wing tips and the far antenna at any distance
        # that still reads as a fly rather than as a speck, and the screenshot
        # that started this (IMG_2715) shows that view working. What it may not
        # do is crop the animal: 90% is the allowance for bristles.
        if home["in_frame_pct"] < 90.0:
            problems.append(
                f"home: only {home['in_frame_pct']:.1f}% of the animal is in "
                "frame — the default view is the whole fly")
    near = by_label.get("min")
    if near is not None:
        # The defect IMG_2715 reported, as a rule: the closest the pinch can
        # pull the camera must still frame the body of the animal.
        if near["body_in_frame_pct"] < 99.9:
            problems.append(
                f"min ({min_d:.3f} cm): only {near['body_in_frame_pct']:.1f}% of "
                f"the body is in frame (it spans "
                f"{near['body_span_screens']:.2f} screens) — this is the camera "
                "IMG_2715 was looking through")
        if near["body_span_screens"] > 1.2:
            problems.append(
                f"min ({min_d:.3f} cm): the body spans "
                f"{near['body_span_screens']:.2f} screens at the closest zoom")
        if near["nearest_surface"] < 0.02:
            problems.append(
                f"min ({min_d:.3f} cm): the nearest surface is "
                f"{near['nearest_surface']*10:.2f} mm in front of the lens")
    far = by_label.get("max")
    if far is not None and far["body_span_screens"] < 0.08:
        problems.append(
            f"max ({max_d:.3f} cm): the body spans only "
            f"{far['body_span_screens']:.3f} screens — the animal is a dot at "
            "the furthest the pinch can pull back")
    for m in results:
        if m["label"].startswith("pane") and m["body_in_frame_pct"] < 99.0:
            problems.append(
                f"{m['label']}: only {m['body_in_frame_pct']:.1f}% of the body "
                "is in the pane")
    log("")
    for p in problems:
        log(f"  FAIL {p}")
    if problems:
        return 1
    log("  ok   every camera the app can reach frames the body of the animal")
    log(f"  wrote pictures to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
