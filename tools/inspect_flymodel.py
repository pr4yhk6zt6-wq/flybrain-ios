#!/usr/bin/env python3
"""
inspect_flymodel.py — read flymodel.bin exactly the way FlyModel.swift does,
then reproduce WorldRenderer's root matrix, FlyModel.solve and ChaseCamera,
and report where the fly actually ends up on screen.

This is a diagnostic, not part of the build. It exists because "the fly is
invisible" is a question about numbers, and the numbers are the same in
Python as they are on the phone.
"""

import math
import struct
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "build/flymodel.bin"


def load(path):
    d = open(path, "rb").read()
    magic = d[:8]
    assert magic == b"FLYMODEL", magic
    (version, partCount, vertexCount, indexCount, jointCount, sectionCount) = \
        struct.unpack_from("<IIIIII", d, 8)
    offs, lens = [], []
    for i in range(sectionCount):
        o, ln = struct.unpack_from("<QQ", d, 64 + i * 16)
        offs.append(o)
        lens.append(ln)
    print(f"version={version} parts={partCount} verts={vertexCount} "
          f"indices={indexCount} joints={jointCount} sections={sectionCount}")
    for i, (o, ln) in enumerate(zip(offs, lens)):
        print(f"  section {i}: @{o:,} len {ln:,}")

    parts = []
    ps = 4 + 12 + 16 + 16 + 24 + 8
    for i in range(partCount):
        o = offs[0] + i * ps
        f = lambda k: struct.unpack_from("<f", d, o + k)[0]
        u = lambda k: struct.unpack_from("<I", d, o + k)[0]
        parts.append(dict(
            parent=struct.unpack_from("<i", d, o)[0],
            pos=np.array([f(4), f(8), f(12)]),
            quat=np.array([f(16), f(20), f(24), f(28)]),   # xyzw
            colour=np.array([f(32), f(36), f(40), f(44)]),
            vstart=u(48), vcount=u(52), istart=u(56), icount=u(60),
            jstart=u(64), jcount=u(68), noff=u(72), nlen=u(76)))

    joints = []
    js = 12 + 8 + 8
    for i in range(jointCount):
        o = offs[1] + i * js
        f = lambda k: struct.unpack_from("<f", d, o + k)[0]
        u = lambda k: struct.unpack_from("<I", d, o + k)[0]
        joints.append(dict(axis=np.array([f(0), f(4), f(8)]),
                           lower=f(12), upper=f(16), noff=u(20), nlen=u(24)))

    names = d[offs[2]:offs[2] + lens[2]]
    nm = lambda off, ln: names[off:off + ln].decode("utf-8", "ignore")
    for p in parts:
        p["name"] = nm(p["noff"], p["nlen"])
    for j in joints:
        j["name"] = nm(j["noff"], j["nlen"])

    V = np.frombuffer(d, dtype=np.float32,
                      count=vertexCount * 6, offset=offs[3]).reshape(-1, 6)
    I = np.frombuffer(d, dtype=np.uint32, count=indexCount, offset=offs[4])
    return parts, joints, V, I


# --------------------------------------------------------------- matrix maths

def eye():
    return np.eye(4)


def translate(t):
    m = eye()
    m[0:3, 3] = t
    return m


def scale(s):
    m = eye()
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def quat_to_mat(q):
    """xyzw -> the same 3x3 Swift builds in float4x4(quaternion:)."""
    x, y, z, w = q
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
        [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]])
    m = eye()
    m[0:3, 0:3] = r
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
    r = np.array([
        [t * a[0] * a[0] + c, t * a[0] * a[1] + s * a[2], t * a[0] * a[2] - s * a[1]],
        [t * a[0] * a[1] - s * a[2], t * a[1] * a[1] + c, t * a[1] * a[2] + s * a[0]],
        [t * a[0] * a[2] + s * a[1], t * a[1] * a[2] - s * a[0], t * a[2] * a[2] + c]])
    m = eye()
    m[0:3, 0:3] = r
    return m


def lookAt(eye_, target, up):
    f = (target - eye_) / np.linalg.norm(target - eye_)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye_)
    m[1, 3] = -np.dot(u, eye_)
    m[2, 3] = np.dot(f, eye_)
    return m


def perspective(fov_y, aspect, near, far):
    y = 1 / math.tan(fov_y * 0.5)
    x = y / aspect
    z = far / (near - far)
    # Swift columns: (x,0,0,0) (0,y,0,0) (0,0,z,-1) (0,0,z*near,0)
    m = np.zeros((4, 4))
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    m[3, 2] = -1
    m[2, 3] = z * near
    return m


# ------------------------------------------------------------------ the model

parts, joints, V, I = load(PATH)

print("\n--- per-part local extents (MJCF units, cm) ---")
for p in parts:
    if p["icount"] == 0:
        print(f"  {p['name']:<24} (no mesh)")
        continue
    v = V[p["vstart"]:p["vstart"] + p["vcount"], :3]
    lo, hi = v.min(0), v.max(0)
    print(f"  {p['name']:<24} v={p['vcount']:>6} tris={p['icount']//3:>6} "
          f"extent=[{hi[0]-lo[0]:.4f} {hi[1]-lo[1]:.4f} {hi[2]-lo[2]:.4f}]")

print("\n--- joint axes (a zero axis makes normalize() NaN in Swift) ---")
zero_axes = 0
for j in joints:
    n = float(np.linalg.norm(j["axis"]))
    if n < 1e-9:
        zero_axes += 1
        print(f"  !! {j['name']:<28} axis={np.round(j['axis'], 4)}  ZERO")
if zero_axes == 0:
    print("  all joint axes non-zero")
else:
    print(f"  {zero_axes} zero-axis joints -> NaN model matrices")

print("\n--- part colours ---")
for p in parts:
    if p["icount"] == 0:
        continue
    print(f"  {p['name']:<22} rgba={np.round(p['colour'], 3)}")

lo = V[:, :3].min(0)
hi = V[:, :3].max(0)
extent = hi - lo
longest = extent.max()
print(f"\nmodel AABB  lo={np.round(lo,5)} hi={np.round(hi,5)}")
print(f"model extent (MJCF units = cm) = {np.round(extent,5)}   longest={longest:.5f}")

# What FlyModel.swift computes
normScale = 0.25 / longest
normOffset = -(lo + hi) * 0.5
print(f"\nFlyModel.normalisationScale  = {normScale:.4f}")
print(f"FlyModel.normalisationOffset = {np.round(normOffset,5)}")
print(f"normalised extent (world units) = {np.round(extent*normScale,5)}")
print("  world X (lateral, was MJCF Y) = %.4f" % (extent[1] * normScale))
print("  world Y (up,      was MJCF Z) = %.4f" % (extent[2] * normScale))
print("  world Z (body axis, was MJCF X) = %.4f" % (extent[0] * normScale))

# Real D. melanogaster, for comparison
print("\nreal D. melanogaster (world units, 1 = 1 cm):")
print("  body length 2.5 mm = 0.25   wing length 2.39 mm = 0.239")
print("  wingspan ~5.2 mm   = 0.52   body width ~0.9 mm  = 0.09")

# ------------------------------------------------------------------ solve()

def solve(angles, root):
    out = [None] * len(parts)
    norm = root @ scale(np.array([normScale] * 3)) @ translate(normOffset)
    for i, p in enumerate(parts):
        parentM = norm if p["parent"] < 0 else out[p["parent"]]
        local = translate(p["pos"]) @ quat_to_mat(p["quat"])
        for j in range(p["jcount"]):
            ji = p["jstart"] + j
            a = angles[ji] if ji < len(angles) else 0.0
            a = min(max(a, joints[ji]["lower"]), joints[ji]["upper"])
            if a != 0:
                local = local @ axis_angle(joints[ji]["axis"], a)
        out[i] = parentM @ local
    return out


# FlyBody.reset(): position (0, 0.35, 0), heading 0
pos = np.array([0.0, 0.35, 0.0])
heading = pitch = roll = 0.0

root = (translate(pos) @ rot_y(heading) @ rot_x(pitch) @ rot_z(roll)
        @ rot_x(-math.pi / 2) @ rot_z(math.pi / 2))

angles = [0.0] * len(joints)
M = solve(angles, root)

# world-space AABB over every drawn part
wlo = np.array([np.inf] * 3)
whi = np.array([-np.inf] * 3)
for i, p in enumerate(parts):
    if p["icount"] == 0:
        continue
    v = V[p["vstart"]:p["vstart"] + p["vcount"], :3]
    v = (M[i][0:3, 0:3] @ v.T).T + M[i][0:3, 3]
    wlo = np.minimum(wlo, v.min(0))
    whi = np.maximum(whi, v.max(0))

print("\n--- fly in world space (rest pose, angles = 0) ---")
print(f"  AABB lo = {np.round(wlo,4)}   hi = {np.round(whi,4)}")
print(f"  size    = {np.round(whi-wlo,4)} world units (cm)")
print(f"  centre  = {np.round((wlo+whi)/2,4)}   (pose.position = {pos})")

# A few named landmarks
for want in ("thorax", "head", "abdomen", "wing_left", "wing_right",
             "tibia_T1_left", "antenna_left"):
    for i, p in enumerate(parts):
        if p["name"] == want and p["icount"]:
            v = V[p["vstart"]:p["vstart"] + p["vcount"], :3]
            v = (M[i][0:3, 0:3] @ v.T).T + M[i][0:3, 3]
            c = (v.min(0) + v.max(0)) * 0.5
            print(f"  {want:<16} centre = {np.round(c,4)}   "
                  f"size = {np.round(v.max(0)-v.min(0),4)}")
            break

# --------------------------------------------------------- the chase camera

distance, height = 0.9, 0.3
yaw = heading
back = np.array([-math.sin(yaw), 0.0, math.cos(yaw)])
camPos = pos + back * distance + np.array([0.0, height, 0.0])
lookAtPt = pos

aspect = 1170.0 / 2532.0          # iPhone 12/13/14 portrait
proj = perspective(math.radians(55), aspect, 0.01, 80)
view = lookAt(camPos, lookAtPt, np.array([0.0, 1.0, 0.0]))
vp = proj @ view

def project(p):
    q = vp @ np.array([p[0], p[1], p[2], 1.0])
    if q[3] <= 0:
        return None
    return q[0:3] / q[3]

corners = []
for x in (wlo[0], whi[0]):
    for y in (wlo[1], whi[1]):
        for z in (wlo[2], whi[2]):
            corners.append(np.array([x, y, z]))
ndc = [project(c) for c in corners]
ndc = [n for n in ndc if n is not None]
if ndc:
    n = np.array(ndc)
    sx = (n[:, 0].max() - n[:, 0].min()) * 0.5 * 1170
    sy = (n[:, 1].max() - n[:, 1].min()) * 0.5 * 2532
    zmin, zmax = n[:, 2].min(), n[:, 2].max()
    print("\n--- projected onto a 1170x2532 screen ---")
    print(f"  camera at {np.round(camPos,4)}, looking at {np.round(lookAtPt,4)}")
    print(f"  NDC z range = [{zmin:.5f}, {zmax:.5f}]   (near plane 0.01, far 80)")
    print(f"  fly covers  {sx:.1f} x {sy:.1f} px   "
          f"({100*sx/1170:.1f}% x {100*sy/2532:.1f}% of the screen)")
    print(f"  NDC x range = [{n[:,0].min():.3f}, {n[:,0].max():.3f}]")
    print(f"  NDC y range = [{n[:,1].min():.3f}, {n[:,1].max():.3f}]")

# Sanity: the ground plane and a wall, for scale
print("\n--- for comparison, the room ---")
for corner in [np.array([0.0, 0.0, 0.0]), np.array([6.0, 3.2, -6.0]),
               np.array([0.0, 3.2, -6.125])]:
    q = project(corner)
    print(f"  world {corner} -> NDC {None if q is None else np.round(q,3)}")
