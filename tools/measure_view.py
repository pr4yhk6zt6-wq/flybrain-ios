#!/usr/bin/env python3
"""
How close the camera may come: measured from the animal, written into the world.

`WorldModel` used to carry three constants — `minDistance = 0.22`,
`homeDistance = 1.0`, `maxDistance = 3.0` — and IMG_2715 is what the first of
them looks like on a phone. At 0.22 cm the lens is 2.2 mm from the rig origin,
the animal's *body* bounding radius is 1.47 mm, and a 38° lens sees a window
about 0.75 mm tall at that range: the thorax fills the screen, the legs and
wings cross it at angles, and the result reads as "the model is in pieces" — see
the pictures `tools/audit_view.py` writes, and the same screenshot's own panes,
which sit further out and show a fly standing correctly assembled.

The numbers belong to the asset, not to the source. This tool reads the world
`tools/step4_world.py` wrote, measures the animal as the app poses it, and
writes the camera limits into `world.json` under `view`:

    python3 tools/measure_view.py --write
    python3 tools/measure_view.py --print        # say what they are

The rules, and why:

  * `body_radius_cm` — the bounding radius of the thorax, head and abdomen (and
    their plates) about the rig origin, taken over the whole recording so a
    frame where the animal leans does not produce a tighter limit than it can
    honour. This is "the animal" for framing purposes: a close-up that crops a
    wing tip is a close-up, a close-up that crops the thorax is a bug.
  * `min_distance_cm = 1.05 * body_radius / tan(fov_y / 2)` — the closest
    distance at which the body still fits the lens, with 5 % of margin. At the
    minimum the body fills about half the screen's width and the whole animal
    (antennae to wing tips) is still inside the frame.
  * `home_distance_cm` — where the camera starts: far enough that the *whole*
    animal is inside the frame with margin, which is `all_radius / (0.85 *
    tan(fov_y/2))`, and never closer than 1.4 x the minimum.
  * `max_distance_cm = 2.5 * home` — far enough to see the walk, near enough
    that the animal is still an animal and not a dot.
  * `pane_distance_cm` — where the two eye panes' cameras sit: the same body
    rule at their wider lens (`pane_fov_y`), which is what makes those panes
    show the whole fly in the screenshot.

The rig origin is the app's own: `FlyWorld.centre()` is the mean of the posed
part origins weighted by how many vertices each part's mesh has, so the rig
follows where the geometry is rather than where the part origins are (the legs
and bristles have many parts with small meshes, and an unweighted mean sits off
the body). This tool implements that rule exactly, and `tools/audit_view.py`
checks the app against the file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

FOV_Y = 38.0                 # WorldRig.camera(main, 38, ...)
PANE_FOV_Y = 50.0            # WorldRig.camera(left/right, 50, ...)
MARGIN = 1.05                # the 5% of margin at the minimum
HOME_MARGIN = 0.85           # the fraction of the lens the animal fills at home
PANE_MARGIN = 1.15

# The screens these lenses are used on. This is the part the first version of
# this tool got wrong, and the audit caught it: a phone held upright is taller
# than it is wide, and SceneKit's `fieldOfView` is the *vertical* angle, so the
# horizontal half-angle is `tan(fov_y/2) * aspect` and it is the narrower one
# that decides whether the animal fits. At 0.38 cm the vertical window is 2.6 mm
# and the horizontal one is 1.2 mm — narrower than the body is wide — so a rule
# that only looked at the vertical said "fits" while the thorax hung off both
# sides. These are the app's numbers: `WorldScreen`'s panes, and the phone the
# screenshot came from.
PHONE = (828, 1792)
PANE = (134, 106)

BODY_PARTS = ("thorax", "head", "abdomen", "head_red", "head_black",
              "thorax_black", "abdomen_black")


def load(root: pathlib.Path):
    manifest = json.loads((root / "world.json").read_text())
    blob = (root / "fly.bin").read_bytes()
    n_v, n_f = manifest["body_geometry"]["n_vertex"], manifest["body_geometry"]["n_face"]
    expect = (n_v * 3 + n_v * 3 + n_f * 3) * 4
    if len(blob) < expect:
        sys.exit(f"fly.bin is {len(blob)} bytes, the manifest needs {expect}")
    verts = np.frombuffer(blob, np.float32, count=n_v * 3,
                          offset=0).reshape(-1, 3).astype(np.float64)
    frames = np.frombuffer((root / "frames.bin").read_bytes(), np.float32)
    return manifest, verts, frames


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_vertices(manifest, verts, frames, index):
    """World-space vertices at one frame, and which part each belongs to."""
    parts = manifest["frames"]["parts"]
    stride = parts * 7
    table = frames[index * stride:(index + 1) * stride].reshape(parts, 7)
    mesh_verts = manifest["body_geometry"]["meshes"]
    out = np.zeros_like(verts)
    owner = np.full(len(verts), -1, int)
    for g in manifest["geoms"]:
        if g.get("mesh") is None:
            continue
        vo, vn = mesh_verts[g["mesh"]]["vert"]
        p = table[g["part"]]
        out[vo:vo + vn] = verts[vo:vo + vn] @ quat_matrix(p[3:]).T + p[:3]
        owner[vo:vo + vn] = g["part"]
    return out, owner, table


def rig_centre(manifest, table) -> np.ndarray:
    """The app's rule, in one place: weight each posed part origin by the number
    of vertices in the mesh it carries, and average."""
    mesh_verts = manifest["body_geometry"]["meshes"]
    acc = np.zeros(3)
    total = 0.0
    for g in manifest["geoms"]:
        if g.get("mesh") is None:
            continue
        n = float(mesh_verts[g["mesh"]]["vert"][1])
        acc += table[g["part"]][:3] * n
        total += n
    if total <= 0:
        sys.exit("the manifest has no meshes with vertices")
    return acc / total


def measure(manifest, verts, frames, sample: int) -> dict:
    n = manifest["frames"]["n"]
    body_radius = all_radius = 0.0
    for i in range(0, n, max(1, sample)):
        posed, owner, table = pose_vertices(manifest, verts, frames, i)
        centre = rig_centre(manifest, table)
        rel = posed - centre
        names = {g["part"]: g["name"] for g in manifest["geoms"]}
        body = np.array([names.get(o, "") in BODY_PARTS for o in owner])
        all_radius = max(all_radius, float(np.linalg.norm(rel, axis=1).max()))
        if body.any():
            body_radius = max(body_radius,
                              float(np.linalg.norm(rel[body], axis=1).max()))
    if body_radius <= 0:
        sys.exit("no body parts matched — this manifest's geoms are not named "
                 f"as expected ({BODY_PARTS})")
    # The binding half-angle for each lens: the smaller of the vertical and the
    # horizontal, which is the horizontal on a phone held upright.
    def half_angle(fov_y: float, viewport) -> float:
        t = np.tan(np.radians(fov_y) / 2)
        return t * min(1.0, viewport[0] / viewport[1])

    t_main = half_angle(FOV_Y, PHONE)
    t_pane = half_angle(PANE_FOV_Y, PANE)
    t_home = np.tan(np.radians(FOV_Y) / 2)      # vertically: the wings and legs
    min_d = MARGIN * body_radius / t_main
    home_d = max(all_radius / (HOME_MARGIN * t_home), 1.15 * min_d)
    return {
        "fov_y": FOV_Y,
        "pane_fov_y": PANE_FOV_Y,
        "phone_px": list(PHONE),
        "pane_px": list(PANE),
        "body_radius_cm": round(body_radius, 6),
        "all_radius_cm": round(all_radius, 6),
        "min_distance_cm": round(min_d, 6),
        "home_distance_cm": round(home_d, 6),
        "max_distance_cm": round(2.5 * home_d, 6),
        "pane_distance_cm": round(PANE_MARGIN * body_radius / t_pane, 6),
        "rule": ("min = 1.05 * body_radius / (tan(fov_y/2) * min(1, w/h)); "
                 "home = max(all_radius / (0.85 * tan(fov_y/2)), 1.15 * min); "
                 "max = 2.5 * home; pane = 1.15 * body_radius / "
                 "(tan(pane_fov_y/2) * min(1, w_pane/h_pane)); rig origin = "
                 "vertex-weighted mean of the posed part origins"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=pathlib.Path, default=pathlib.Path("world"))
    ap.add_argument("--sample", type=int, default=25,
                    help="measure every Nth frame of the recording")
    ap.add_argument("--write", action="store_true",
                    help="write the block into world.json")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the block (this is also the default)")
    args = ap.parse_args()

    manifest, verts, frames = load(args.world)
    view = measure(manifest, verts, frames, args.sample)

    print(f"{args.world}/world.json — measured over every {args.sample}th of "
          f"{manifest['frames']['n']} frames")
    print(f"  body radius (thorax+head+abdomen about the rig origin): "
          f"{view['body_radius_cm']*10:.3f} mm")
    print(f"  whole animal radius:                                    "
          f"{view['all_radius_cm']*10:.3f} mm")
    binding = np.tan(np.radians(FOV_Y) / 2) * min(1.0, PHONE[0] / PHONE[1])
    print(f"  min   distance {view['min_distance_cm']:.4f} cm  "
          f"(v0.7 had 0.22 — the bug: the body was "
          f"{view['body_radius_cm'] / (0.22 * binding):.2f} screens wide there, "
          f"and the horizontal window at that range was only "
          f"{2 * 0.22 * binding * 10:.2f} mm)")
    print(f"  home  distance {view['home_distance_cm']:.4f} cm  "
          f"(v0.7 had 1.00)")
    print(f"  max   distance {view['max_distance_cm']:.4f} cm  "
          f"(v0.7 had 3.00)")
    print(f"  pane  distance {view['pane_distance_cm']:.4f} cm  "
          f"(the panes sat at 0.306 and framed the animal correctly)")

    if args.write:
        path = args.world / "world.json"
        manifest["view"] = view
        path.write_text(json.dumps(manifest, indent=1))
        print(f"  wrote the view block into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
