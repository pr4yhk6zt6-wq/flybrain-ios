#!/usr/bin/env python3
"""
softrender.py — a CPU replica of FlyBrain's world pass.

It re-implements, in Python, exactly what these do on the phone:

    FlyModel.solve            (WorldRenderer.appendFly)
    WorldRenderer.buildInstances
    World.buildRoom / scenery
    ChaseCamera.matrix
    Shaders/World.metal       (worldVertex, worldFragment, sky)

so that "is the fly actually in the frame, and how big is it" becomes a
question we can answer by measuring pixels instead of by staring at a phone.

Usage:
    python3 tools/softrender.py out.png [--env kitchen|garden|lab] [--no-fly]
                                [--cam 0.9] [--scale-fix] [--rest]
"""

import argparse
import math
import struct

import numpy as np
from PIL import Image

# ------------------------------------------------------------------ matrices


def eye():
    return np.eye(4)


def translate(t):
    m = eye()
    m[0:3, 3] = t
    return m


def scale3(s):
    m = eye()
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def quat_to_mat(q):
    x, y, z, w = q
    m = eye()
    # Swift's float4x4(quaternion:) takes these as COLUMNS, which transposes
    # them into this row-major rotation matrix.
    m[0:3, 0:3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return m


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    m = eye()
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    m = eye()
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    m = eye()
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def axis_angle(axis, angle):
    a = axis / np.linalg.norm(axis)
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    m = eye()
    m[0:3, 0:3] = np.array([
        [t * a[0] * a[0] + c, t * a[0] * a[1] + s * a[2], t * a[0] * a[2] - s * a[1]],
        [t * a[0] * a[1] - s * a[2], t * a[1] * a[1] + c, t * a[1] * a[2] + s * a[0]],
        [t * a[0] * a[2] + s * a[1], t * a[1] * a[2] - s * a[0], t * a[2] * a[2] + c]])
    return m


def lookAt(eye_, target, up):
    f = target - eye_
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = eye()
    m[0, 0:3], m[1, 0:3], m[2, 0:3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye_, -u @ eye_, f @ eye_
    return m


def perspective(fov_y, aspect, near, far):
    y = 1 / math.tan(fov_y * 0.5)
    x = y / aspect
    z = far / (near - far)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1], m[2, 2], m[3, 2], m[2, 3] = x, y, z, -1, z * near
    return m


# ------------------------------------------------------- flymodel.bin loader

def load_flymodel(path):
    d = open(path, "rb").read()
    assert d[:8] == b"FLYMODEL"
    (ver, nparts, nverts, nidx, njoints, nsec) = struct.unpack_from("<IIIIII", d, 8)
    offs, lens = [], []
    for i in range(nsec):
        o, ln = struct.unpack_from("<QQ", d, 64 + i * 16)
        offs.append(o)
        lens.append(ln)

    parts, ps = [], 4 + 12 + 16 + 16 + 24 + 8
    for i in range(nparts):
        o = offs[0] + i * ps
        f = lambda k, o=o: struct.unpack_from("<f", d, o + k)[0]
        u = lambda k, o=o: struct.unpack_from("<I", d, o + k)[0]
        parts.append(dict(parent=struct.unpack_from("<i", d, o)[0],
                          pos=np.array([f(4), f(8), f(12)]),
                          quat=np.array([f(16), f(20), f(24), f(28)]),
                          colour=np.array([f(32), f(36), f(40), f(44)]),
                          vstart=u(48), vcount=u(52), istart=u(56), icount=u(60),
                          jstart=u(64), jcount=u(68), noff=u(72), nlen=u(76)))
    joints, js = [], 12 + 8 + 8
    for i in range(njoints):
        o = offs[1] + i * js
        f = lambda k, o=o: struct.unpack_from("<f", d, o + k)[0]
        u = lambda k, o=o: struct.unpack_from("<I", d, o + k)[0]
        joints.append(dict(axis=np.array([f(0), f(4), f(8)]),
                           lower=f(12), upper=f(16), noff=u(20), nlen=u(24)))
    names = d[offs[2]:offs[2] + lens[2]]
    for p in parts:
        p["name"] = names[p["noff"]:p["noff"] + p["nlen"]].decode("utf-8", "ignore")
    for j in joints:
        j["name"] = names[j["noff"]:j["noff"] + j["nlen"]].decode("utf-8", "ignore")
    V = np.frombuffer(d, dtype=np.float32, count=nverts * 6,
                      offset=offs[3]).reshape(-1, 6)
    I = np.frombuffer(d, dtype=np.uint32, count=nidx, offset=offs[4])
    return parts, joints, V, I


# ------------------------------------------------------- the primitive meshes

def cube_mesh():
    faces = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
             ((1, 0, 0), (0, 0, -1), (0, 1, 0)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]
    verts, idx = [], []
    for n, u, v in faces:
        n, u, v = np.array(n, float), np.array(u, float), np.array(v, float)
        base = len(verts) // 6
        for su, sv in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            p = (n + u * su + v * sv) * 0.5
            verts += [p[0], p[1], p[2], n[0], n[1], n[2]]
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(verts, np.float32).reshape(-1, 6), np.array(idx, np.uint32)


def quad_mesh():
    verts = []
    for x, z in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
        verts += [x, 0, z, 0, 1, 0]
    return (np.array(verts, np.float32).reshape(-1, 6),
            np.array([0, 1, 2, 0, 2, 3], np.uint32))


def icosphere_mesh(subdivisions):
    t = (1 + math.sqrt(5.0)) / 2
    pts = [np.array(p, float) for p in [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    pts = [p / np.linalg.norm(p) for p in pts]
    tris = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    cache = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        pts.append((pts[a] + pts[b]) * 0.5)
        pts[-1] = pts[-1] / np.linalg.norm(pts[-1])
        cache[key] = len(pts) - 1
        return len(pts) - 1

    for _ in range(subdivisions):
        nxt = []
        for a, b, c in tris:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nxt += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        tris = nxt
    verts = []
    for p in pts:
        h = p * 0.5
        verts += [h[0], h[1], h[2], p[0], p[1], p[2]]
    idx = []
    for a, b, c in tris:
        idx += [a, b, c]
    return np.array(verts, np.float32).reshape(-1, 6), np.array(idx, np.uint32)


# -------------------------------------------------------------- the world

ENVS = {
    "kitchen": dict(sky=(0.14, 0.12, 0.14), ground=(0.55, 0.42, 0.28),
                    checker=3.0, fog=0.03, ceiling=3.2),
    "garden": dict(sky=(0.38, 0.62, 0.86), ground=(0.22, 0.42, 0.16),
                   checker=10.0, fog=0.012, ceiling=8.0),
    "lab": dict(sky=(0.07, 0.08, 0.10), ground=(0.85, 0.86, 0.88),
                checker=1.5, fog=0.03, ceiling=3.2),
}

BOUNDS = 6.0


def build_world(env):
    objs = []
    e = ENVS[env]
    t, h, b = 0.25, e["ceiling"], BOUNDS
    span = b * 2 + t * 2
    for pos, size in [((0, h / 2, -b - t / 2), (span, h, t)),
                      ((0, h / 2, b + t / 2), (span, h, t)),
                      ((-b - t / 2, h / 2, 0), (t, h, span)),
                      ((b + t / 2, h / 2, 0), (t, h, span))]:
        objs.append(dict(kind="wall", pos=np.array(pos), size=np.array(size),
                         colour=(0.52, 0.47, 0.42), mesh="cube", amount=1))
    if env != "garden":
        objs.append(dict(kind="ceiling", pos=np.array((0, h + t / 2, 0)),
                         size=np.array((span, t, span)),
                         colour=(0.30, 0.29, 0.28), mesh="cube", amount=1))
    rng = np.random.default_rng(7)
    if env == "kitchen":
        for _ in range(5):
            s = float(rng.uniform(0.25, 0.8))
            objs.append(dict(kind="cube", pos=np.array(
                [float(rng.uniform(-4.8, 4.8)), s * 0.5,
                 float(rng.uniform(-4.8, 4.8))]),
                size=np.array([s, s, s]), colour=(0.45, 0.50, 0.62),
                mesh="cube", amount=1))
        for _ in range(2):
            s = float(rng.uniform(0.3, 0.5))
            hh = s * 2.0
            objs.append(dict(kind="pillar", pos=np.array(
                [float(rng.uniform(-4.8, 4.8)), hh * 0.5,
                 float(rng.uniform(-4.8, 4.8))]),
                size=np.array([s, hh, s]), colour=(0.60, 0.58, 0.52),
                mesh="cube", amount=1))
        objs.append(dict(kind="fruit", pos=np.array([1.4, 0.18, -1.1]),
                         size=np.array([0.18] * 3), colour=(0.85, 0.22, 0.30),
                         mesh="sphere", amount=1))
    return objs


# --------------------------------------------------------------- rasteriser

def render(W, H, draws, camPos, viewProj, env):
    colour = np.zeros((H, W, 3), np.float32)
    depth = np.full((H, W), np.inf, np.float32)
    flymask = np.zeros((H, W), bool)

    sky = np.array(env["sky"], np.float32)
    ambient = sky * 0.35 * 0.6 + np.array([0.25] * 3, np.float32) * 0.4
    sun = np.array([0.4, 0.85, 0.3], np.float32)
    sun = sun / np.linalg.norm(sun)

    # sky gradient: uv.y == 1 at the top of the screen
    t = (1.0 - (np.arange(H, dtype=np.float32) / (H - 1)))[:, None]
    grad = sky[None, None, :] * (0.55 * (1 - t) + 1.15 * t)[:, :, None]
    colour[:] = grad

    for (verts, idx, inst) in draws:
        M = inst["model"]
        R, tr = M[0:3, 0:3], M[0:3, 3]
        wp = verts[:, 0:3] @ R.T + tr
        nrm = verts[:, 3:6] @ R.T
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.where(ln > 1e-9, nrm / np.maximum(ln, 1e-9), 0.0)

        clip = np.concatenate([wp, np.ones((len(wp), 1))], axis=1) @ viewProj.T
        w = clip[:, 3]
        ok = w > 1e-6
        ndc = np.where(ok[:, None], clip[:, 0:3] / np.maximum(w, 1e-6)[:, None], 1e6)
        sx = (ndc[:, 0] * 0.5 + 0.5) * W
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H

        base_col = np.array(inst["colour"][0:3], np.float32)
        emissive = inst["colour"][3]
        checker = inst["checker"]
        is_fly = inst.get("fly", False)

        for t0, t1, t2 in idx.reshape(-1, 3):
            if not (ok[t0] and ok[t1] and ok[t2]):
                continue
            x0, y0, z0, iw0 = sx[t0], sy[t0], ndc[t0, 2], 1.0 / w[t0]
            x1, y1, z1, iw1 = sx[t1], sy[t1], ndc[t1, 2], 1.0 / w[t1]
            x2, y2, z2, iw2 = sx[t2], sy[t2], ndc[t2, 2], 1.0 / w[t2]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if area >= -1e-9:
                continue
            minx = max(int(math.floor(min(x0, x1, x2))), 0)
            maxx = min(int(math.ceil(max(x0, x1, x2))), W - 1)
            miny = max(int(math.floor(min(y0, y1, y2))), 0)
            maxy = min(int(math.ceil(max(y0, y1, y2))), H - 1)
            if minx > maxx or miny > maxy:
                continue
            px = np.arange(minx, maxx + 1, dtype=np.float32) + 0.5
            py = np.arange(miny, maxy + 1, dtype=np.float32) + 0.5
            X, Y = np.meshgrid(px, py)
            w0 = ((x1 - X) * (y2 - Y) - (x2 - X) * (y1 - Y)) / area
            w1 = ((x2 - X) * (y0 - Y) - (x0 - X) * (y2 - Y)) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            iw = w0 * iw0 + w1 * iw1 + w2 * iw2
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = np.where(np.abs(iw) > 1e-20, 1.0 / iw, 0.0)
            zz = (w0 * z0 * iw0 + w1 * z1 * iw1 + w2 * z2 * iw2) * inv
            hit = inside & (zz < depth[miny:maxy + 1, minx:maxx + 1])
            if not hit.any():
                continue
            b0 = (w0 * iw0 * inv)[hit]
            b1 = (w1 * iw1 * inv)[hit]
            b2 = (w2 * iw2 * inv)[hit]
            wpos = (wp[t0][None, :] * b0[:, None] + wp[t1][None, :] * b1[:, None]
                    + wp[t2][None, :] * b2[:, None])
            N = (nrm[t0][None, :] * b0[:, None] + nrm[t1][None, :] * b1[:, None]
                 + nrm[t2][None, :] * b2[:, None])
            Nl = np.linalg.norm(N, axis=1, keepdims=True)
            N = np.where(Nl > 1e-9, N / np.maximum(Nl, 1e-9), 0.0)

            base = np.repeat(base_col[None, :], len(wpos), axis=0)
            if checker > 0:
                c = np.floor(wpos[:, [0, 2]] * checker)
                f = np.mod(c[:, 0] + c[:, 1] + 2.0, 2.0)
                base = base * (0.74 + 0.26 * f)[:, None]
            ndl = np.maximum(N @ sun, 0.0)[:, None]
            lit = base * (ambient[None, :] + ndl * 0.85)
            V_ = camPos[None, :] - wpos
            Vl = np.linalg.norm(V_, axis=1, keepdims=True)
            V_ = np.where(Vl > 1e-9, V_ / np.maximum(Vl, 1e-9), 0.0)
            rim = np.power(1.0 - np.maximum((N * V_).sum(1), 0.0), 3.0)[:, None]
            lit = lit + base * rim * 0.25 + base * emissive
            dist = np.linalg.norm(wpos - camPos[None, :], axis=1)
            fog = np.clip(1.0 - np.exp(-dist * env["fog"]), 0, 1)[:, None]
            lit = lit * (1 - fog) + sky[None, :] * fog

            yy, xx = np.nonzero(hit)
            yys, xxs = yy + miny, xx + minx
            depth[yys, xxs] = zz[yy, xx]
            colour[yys, xxs] = lit
            if is_fly:
                flymask[yys, xxs] = True

    return colour, depth, flymask


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="build/flymodel.bin")
    ap.add_argument("--env", default="kitchen")
    ap.add_argument("--no-fly", action="store_true")
    ap.add_argument("--cam", type=float, default=0.9)
    ap.add_argument("--scale-fix", action="store_true")
    ap.add_argument("--rest", action="store_true", help="all joint angles 0")
    ap.add_argument("--W", type=int, default=390)
    ap.add_argument("--H", type=int, default=844)
    args = ap.parse_args()

    parts, joints, V, I = load_flymodel(args.model)
    lo, hi = V[:, :3].min(0), V[:, :3].max(0)
    longest = float((hi - lo).max())
    # assembled rest pose (what the fixed FlyModel does)
    restM = []
    for p_ in parts:
        par = eye() if p_["parent"] < 0 else restM[p_["parent"]]
        loc = translate(p_["pos"]) @ quat_to_mat(p_["quat"])
        restM.append(par @ loc)
    alo = np.array([np.inf] * 3)
    ahi = np.array([-np.inf] * 3)
    for i, p_ in enumerate(parts):
        if p_["icount"] == 0:
            continue
        v = V[p_["vstart"]:p_["vstart"] + p_["vcount"], :3]
        w = v @ restM[i][0:3, 0:3].T + restM[i][0:3, 3]
        alo = np.minimum(alo, w.min(0))
        ahi = np.maximum(ahi, w.max(0))
    bodyLength = float((ahi - alo)[0])
    if args.scale_fix:
        normOffset = -(alo + ahi) * 0.5
        scale = 0.25 / bodyLength
    else:
        normOffset = -(lo + hi) * 0.5
        scale = 0.25 / longest
    print(f"normalisationScale = {scale:.4f}   (current code: {0.25/longest:.4f})")
    print(f"assembled rest extent (cm) = {np.round(ahi-alo, 5)}  body={bodyLength:.5f}")
    print(f"  -> body {bodyLength*scale*10:.2f} mm, span {(ahi-alo)[1]*scale*10:.2f} mm, "
          f"height {(ahi-alo)[2]*scale*10:.2f} mm")

    pos = np.array([0.0, 0.35, 0.0])
    heading = pitch = roll = 0.0
    root = (translate(pos) @ rot_y(heading) @ rot_x(pitch) @ rot_z(roll)
            @ rot_x(-math.pi / 2) @ rot_z(math.pi / 2))
    norm = root @ scale3(np.array([scale] * 3)) @ translate(normOffset)

    angles = {j["name"]: 0.0 for j in joints}
    if not args.rest:
        # FlyBody.updateJointAngles() for a grounded fly (airborne == 0)
        angles.update({"wing_yaw_left": 1.4, "wing_yaw_right": 1.4})
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for seg in ("T1", "T2", "T3"):
                angles[f"coxa_abduct_{seg}_{side}"] = 0.1 * sign
                angles[f"femur_{seg}_{side}"] = -0.5
                angles[f"tibia_{seg}_{side}"] = 0.4
                angles[f"tarsus_{seg}_{side}"] = 0.1

    out = [None] * len(parts)
    for i, p in enumerate(parts):
        parentM = norm if p["parent"] < 0 else out[p["parent"]]
        local = translate(p["pos"]) @ quat_to_mat(p["quat"])
        for j in range(p["jcount"]):
            jd = joints[p["jstart"] + j]
            a = angles.get(jd["name"], 0.0)
            a = min(max(a, jd["lower"]), jd["upper"])
            if a != 0:
                local = local @ axis_angle(jd["axis"], a)
        out[i] = parentM @ local

    prims = {"cube": cube_mesh(), "sphere": icosphere_mesh(2), "quad": quad_mesh()}
    env = ENVS[args.env]
    draws = []

    draws.append((
        prims["quad"][0], prims["quad"][1],
        dict(model=translate(np.zeros(3))
             @ scale3(np.array([BOUNDS * 2.2, 1, BOUNDS * 2.2])),
             colour=env["ground"] + (0.0,), checker=env["checker"])))
    for o in build_world(args.env):
        size = o["size"] * (o["amount"] * 0.6 + 0.4)
        draws.append((prims[o["mesh"]][0], prims[o["mesh"]][1],
                      dict(model=translate(o["pos"]) @ scale3(size),
                           colour=o["colour"] + (0.12 if o["kind"] == "fruit"
                                                 else 0.0,), checker=0.0)))
    n_scene = len(draws)

    if not args.no_fly:
        for i, p in enumerate(parts):
            if p["icount"] == 0:
                continue
            is_wing = p["name"].startswith("wing")
            istart, icount = int(p["istart"]), int(p["icount"])
            base = int(p["vstart"])
            draws.append((V, I[istart:istart + icount],
                          dict(model=out[i],
                               colour=(p["colour"][0], p["colour"][1],
                                       p["colour"][2], 0.22 if is_wing else 0.0),
                               checker=0.0, fly=True)))

    distance = args.cam
    back = np.array([-math.sin(heading), 0.0, math.cos(heading)])
    camPos = pos + back * distance + np.array([0.0, 0.3, 0.0])
    view = lookAt(camPos, pos, np.array([0.0, 1.0, 0.0]))
    proj = perspective(math.radians(55), args.W / args.H, 0.01, 80)
    colour, depth, flymask = render(args.W, args.H, draws, camPos, proj @ view, env)

    Image.fromarray((np.clip(colour, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print(f"wrote {args.out}   scene draws={n_scene}  fly parts="
          f"{0 if args.no_fly else sum(1 for p in parts if p['icount'])}")

    n = int(flymask.sum())
    print(f"fly pixels: {n}  ({100.0 * n / (args.W * args.H):.2f}% of the frame)")
    if n:
        yy, xx = np.nonzero(flymask)
        print(f"fly bbox: x[{xx.min()}..{xx.max()}] y[{yy.min()}..{yy.max()}]  "
              f"-> {xx.max()-xx.min()+1} x {yy.max()-yy.min()+1} px")
        lit = colour[flymask]
        bg = colour[~flymask]
        print(f"fly   mean rgb = {np.round(lit.mean(0), 4)}")
        print(f"scene mean rgb = {np.round(bg.mean(0), 4)}")
        print(f"|contrast|     = {np.round(abs(lit.mean(0)-bg.mean(0)), 4)}")


if __name__ == "__main__":
    main()
