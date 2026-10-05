#!/usr/bin/env python3
"""
gait_geometry.py — derive the walking-gait constants in FlyBody.swift from the
CT-based flybody model, so none of them is a typed-in number.

For each leg segment this:

  1. searches joint-neutral postures inside the measured MJCF ranges and
     ranks them by the TRUE fore-aft travel of a fixed tarsus-tip material
     point when coxa+femur+tibia sweep together at equal amplitude through
     the same FK chain the renderer uses (FlyModel.solve / softrender).
     (A small-angle tangent gain measured at the neutral pose is NOT a
     valid predictor at the ~0.5 rad amplitudes a stride needs — the first
     version of this script used one and over-estimated travel ~2x.)
  2. solves the sweep amplitude A* whose travel equals the no-slip
     requirement  R = stride * duty  (stride = walkSpeedMax /
     stepFrequencyMax = 30 mm/s / 13 Hz = 2.31 mm, duty = 0.55 measured,
     Mendes et al. 2013), capped at the amplitude the joint ranges allow.
     The secant gain g = travel(A*) / (2 A*) is what FlyBody.swift keeps in
     gaitGainT* so the runtime formula  amp = min(cap, R / (2 g))
     reproduces A* exactly at the operating stride.
  3. builds the inverse-sweep knot table (gaitKnotT*): during stance the
     joint sweep follows these knots so the FOOT moves at constant speed
     even though the FK tip displacement is strongly nonlinear in joint
     angle.  Knot k solves  x(q_k) = x(+1) - (k/N) * travel.
  4. reports the residual slip at top speed where the morphology runs out
     of range (reported, not hidden — ASSUMPTIONS.md #6) and the mid-stance
     speed nonlinearity that remains.

It also reports the leg track: the rest-pose lateral distance between the
left and right tarsus lines.

Run:  python3 tools/gait_geometry.py [--model build/flymodel.bin]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "tools")
from softrender import (axis_angle, eye as I4, load_flymodel, quat_to_mat,
                        translate)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="build/flymodel.bin")
args = ap.parse_args()

parts, joints, V, I = load_flymodel(args.model)
name = {p["name"]: i for i, p in enumerate(parts)}
lo, hi = V[:, :3].min(0), V[:, :3].max(0)
SCALE = 0.25 / float((hi - lo).max())      # WorldRenderer normalisationScale
jr = {j["name"]: (j["lower"], j["upper"]) for j in joints}

# Operating point the gait constants are derived at (FlyMorphology).
WALK_SPEED_MAX = 30.0     # mm/s
STEP_FREQ_MAX = 13.0      # Hz
DUTY = 0.55               # Mendes et al. 2013
R_TRAVEL = WALK_SPEED_MAX / STEP_FREQ_MAX * DUTY    # mm, no-slip requirement
NKNOT = 8                 # stance inverse-mapping resolution


def assemble(angles):
    M = []
    for p in parts:
        par = I4() if p["parent"] < 0 else M[p["parent"]]
        local = translate(p["pos"]) @ quat_to_mat(p["quat"])
        for k in range(p["jcount"]):
            jd = joints[p["jstart"] + k]
            a = min(max(angles.get(jd["name"], 0.0), jd["lower"]), jd["upper"])
            if a != 0:
                local = local @ axis_angle(jd["axis"], a)
        M.append(par @ local)
    return M


def tip_vertex(partname):
    """Index (within the part) of the tarsus tip: the mesh vertex farthest
    from the thorax origin in the REST pose. Chosen ONCE — the travel must
    be the displacement of a fixed material point, not of a re-selected
    'farthest vertex', which jumps around and inflates the estimate."""
    p = parts[name[partname]]
    v = V[p["vstart"]:p["vstart"] + p["vcount"], :3]
    return int(np.argmax(np.linalg.norm(v, axis=1)))


def tip(M, partname, vert):
    p = parts[name[partname]]
    v = V[p["vstart"] + vert, :3]
    return (M[name[partname]][:3, :3] @ v + M[name[partname]][:3, 3]) * SCALE * 10


EPS = 0.02
SIDE = "left"        # the model is mirrored; one side is enough
TIPV = {seg: tip_vertex(f"tarsus_{seg}_{SIDE}") for seg in ("T1", "T2", "T3")}

print(f"no-slip travel requirement R = stride * duty = {R_TRAVEL:.3f} mm")
print()

for seg in ("T1", "T2", "T3"):
    pn = f"tarsus_{seg}_{SIDE}"
    vert = TIPV[seg]
    jn = [f"coxa_{seg}_{SIDE}", f"femur_{seg}_{SIDE}", f"tibia_{seg}_{SIDE}"]

    def tipx(base):
        return float(tip(assemble(base), pn, vert)[0])

    # ---- 1. posture search, ranked by TRUE travel -------------------------
    grids = [np.arange(jr[j][0] + 0.05, jr[j][1] - 0.04, st)
             for j, st in zip(jn, (0.1, 0.15, 0.15))]
    best = None
    for c0 in grids[0]:
        for f0 in grids[1]:
            for t0 in grids[2]:
                base = dict(zip(jn, (float(c0), float(f0), float(t0))))
                x0 = tipx(base)
                gains = []
                for j in jn:
                    a = dict(base)
                    a[j] = base[j] + EPS
                    gains.append((tipx(a) - x0) / EPS)
                if sum(abs(g) for g in gains) < 0.15:
                    continue
                signs = [1.0 if g >= 0 else -1.0 for g in gains]
                amax = 1e9
                for j, s in zip(jn, signs):
                    lwr, upr = jr[j]
                    n = base[j]
                    amax = min(amax, (upr - n) if s > 0 else (n - lwr),
                               (n - lwr) if s > 0 else (upr - n))
                tv = (tipx({j: base[j] + s * amax for j, s in zip(jn, signs)})
                      - tipx({j: base[j] - s * amax for j, s in zip(jn, signs)}))
                if best is None or tv > best[0]:
                    best = (tv, base, signs, amax)
    tvmax, base, signs, amax = best

    def travel(A):
        return (tipx({j: base[j] + s * A for j, s in zip(jn, signs)})
                - tipx({j: base[j] - s * A for j, s in zip(jn, signs)}))

    # ---- 2. amplitude solve: travel(A*) = R, capped at the range ----------
    if tvmax >= R_TRAVEL:
        loA, hiA = 0.01, amax
        for _ in range(40):
            mid = 0.5 * (loA + hiA)
            if travel(mid) < R_TRAVEL:
                loA = mid
            else:
                hiA = mid
        A = 0.5 * (loA + hiA)
        slip = 0.0
    else:
        A = amax
        slip = 1.0 - tvmax / R_TRAVEL
    tv = travel(A)
    gsec = tv / (2 * A)

    # ---- 3. inverse-sweep knots (constant stance-foot speed) --------------
    # x(q) falls through the sweep (tiny protraction-end bumps aside), so
    # each interior target has a unique root below the bump; bisect for it.
    xq = lambda q: tipx({j: base[j] + s * A * q for j, s in zip(jn, signs)})
    x0 = xq(1.0)
    knots = [1.0]
    for k in range(1, NKNOT + 1):
        target = x0 - (k / NKNOT) * tv          # x decreases through stance
        loQ, hiQ = -1.0, 1.0
        for _ in range(40):                      # bisect; x(q) rises with q
            mid = 0.5 * (loQ + hiQ)
            if xq(mid) > target:
                hiQ = mid
            else:
                loQ = mid
        q = 0.5 * (loQ + hiQ)
        assert abs(xq(q) - target) < 1e-3, f"{seg}: knot {k} did not converge"
        knots.append(q)
    assert all(a > b for a, b in zip(knots, knots[1:])), \
        f"{seg}: knots not monotone"

    # ---- 4. residual nonlinearity with the inverse mapping ----------------
    ss = np.linspace(1.0, -1.0, 41)
    qs = np.interp((1.0 - ss) / 2.0, np.linspace(0, 1, NKNOT + 1), knots)
    xs = np.array([xq(q) for q in qs])
    dx = np.diff(xs) / np.diff(ss)
    nl = float(np.mean(np.abs(dx / dx.mean() - 1)))

    print(f"{seg}: neutral=({base[jn[0]]:.2f}, {base[jn[1]]:.2f}, "
          f"{base[jn[2]]:.2f})  signs=({signs[0]:+.0f}, {signs[1]:+.0f}, "
          f"{signs[2]:+.0f})")
    print(f"    travel(Amax={amax:.2f})={tvmax:.3f} mm   A*={A:.3f} rad   "
          f"secant gain={gsec:.3f} mm/rad   slip@30mm/s={slip:.0%}   "
          f"nonlin={nl:.1%}")
    print(f"    knots=[{', '.join(f'{q:+.3f}' for q in knots)}]")
    print()

# ---- leg track: rest-pose tarsus spreads -----------------------------------
M = assemble({})
tracks = []
for seg in ("T1", "T2", "T3"):
    yl = tip(M, f"tarsus_{seg}_left", TIPV[seg])[1]
    # the right-side tip vertex is the mirrored mesh; use its own farthest
    vr = tip_vertex(f"tarsus_{seg}_right")
    yr = tip(M, f"tarsus_{seg}_right", vr)[1]
    tracks.append(abs(yl - yr))
print(f"rest-pose tarsus spreads (mm): "
      f"{', '.join(f'{t:.3f}' for t in tracks)}")
print(f"legTrack = mean = {np.mean(tracks):.3f} mm "
      f"-> FlyMorphology.legTrack = {np.mean(tracks) * 1e-3:.2e} m")
