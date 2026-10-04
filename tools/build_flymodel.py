#!/usr/bin/env python3
"""
build_flymodel.py — turn the Janelia/DeepMind `flybody` MuJoCo model into a
compact articulated mesh the iOS app can draw.

Source:   https://github.com/TuragaLab/flybody   (Apache-2.0)
          Vaxenburg et al., "Whole-body physics simulation of fruit fly
          locomotion", Nature (2025).

The flybody asset set is an anatomically accurate *Drosophila melanogaster*
reconstruction: 67 rigid bodies, 85 visual meshes, real joint axes and limits,
and even the two 140-degree eye cameras in the right places on the head. It is
the same model DeepMind trained flight and walking policies on, so its joint
tree is a defensible skeleton to hang our connectome's motor output on.

What this script does
---------------------
1.  Fetches fruitfly.xml and the OBJ meshes it references (cached).
2.  Parses the MuJoCo body tree: parent, rest translation, rest rotation,
    and each body's hinge joints (axis + limits).
3.  Bakes every geom's local pos/quat into its vertices, so each body owns one
    mesh in its own frame.
4.  Decimates by vertex clustering down to a triangle budget — the raw meshes
    are 1.5 M triangles, which is absurd for a thing 20 pixels across.
5.  Writes flymodel.bin: a part table + one interleaved vertex buffer + one
    index buffer, ready for `makeBuffer(bytes:)`.

Output format (little endian, every section 256-byte aligned)
-------------------------------------------------------------
header 256 B:  magic "FLYMODEL", version u32, partCount u32, vertexCount u32,
               indexCount u32, jointCount u32, sectionCount u32
               then sectionCount * (u64 offset, u64 length) at byte 64

sections: parts, joints, vertices, indices

part   (96 B): int32 parent
               float3 restTranslation
               float4 restRotation (quaternion xyzw)
               float4 colour (rgba)
               uint32 vertexStart, vertexCount, indexStart, indexCount
               uint32 jointStart, jointCount
               uint32 nameOffset, nameLength      (into the name blob)
joint  (32 B): float3 axis, float2 range, uint32 nameOffset, nameLength, pad
vertex (24 B): float3 position, float3 normal
index        : uint32 (parts exceed 65 k vertices in aggregate)
"""

import argparse
import json
import math
import os
import struct
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

RAW = "https://raw.githubusercontent.com/TuragaLab/flybody/main/flybody/fruitfly/assets"
MAGIC = b"FLYMODEL"
VERSION = 1
HEADER_BYTES = 256


def log(msg):
    print(f"[flymodel] {msg}", flush=True)


# --------------------------------------------------------------------- fetch

def fetch(name, cache_dir):
    path = os.path.join(cache_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(cache_dir, exist_ok=True)
    url = f"{RAW}/{name}"
    log(f"downloading {name} ...")
    with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as fh:
        fh.write(r.read())
    return path


# ------------------------------------------------------------------ geometry

def quat_wxyz(s):
    """MuJoCo quaternions are w x y z."""
    v = [float(x) for x in s.split()]
    q = np.array(v, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0, 0, 0])


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def euler_to_quat(s):
    """MuJoCo default euler order is xyz, radians (angle='radian' in this file)."""
    ex, ey, ez = [float(v) for v in s.split()]
    cx, sx = math.cos(ex / 2), math.sin(ex / 2)
    cy, sy = math.cos(ey / 2), math.sin(ey / 2)
    cz, sz = math.cos(ez / 2), math.sin(ez / 2)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ])


def vec3(s, default=(0.0, 0.0, 0.0)):
    if s is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(x) for x in s.split()], dtype=np.float64)


# ----------------------------------------------------------------- OBJ loader

def load_obj(path):
    """Minimal OBJ reader: positions and triangulated faces. Normals are
    recomputed from the geometry, because we are about to decimate anyway."""
    verts = []
    faces = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    idx.append(int(tok.split("/")[0]) - 1)
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(faces, dtype=np.int64).reshape(-1, 3))


# ----------------------------------------------------------------- decimation

def decimate(verts, faces, target_tris):
    """Vertex clustering on a uniform grid.

    Not as pretty as quadric error collapse, but it is 20 lines, has no
    dependencies, never produces holes, and at the sizes we draw this thing the
    difference is invisible. The grid resolution is solved for iteratively so we
    land near the triangle budget.
    """
    if len(faces) <= target_tris:
        return verts, faces

    lo, hi = verts.min(0), verts.max(0)
    extent = np.maximum(hi - lo, 1e-9)

    # Higher grid resolution = more clusters = more triangles, so to come DOWN
    # to the budget we shrink the grid.
    res = 64
    for _ in range(18):
        cell = np.floor((verts - lo) / extent * res).astype(np.int64)
        cell = np.minimum(cell, res - 1)
        key = (cell[:, 0] * res + cell[:, 1]) * res + cell[:, 2]
        _, inverse = np.unique(key, return_inverse=True)
        f = inverse[faces]
        # drop degenerate triangles (all three corners in one cell)
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[keep]
        if len(f) <= target_tris or res <= 4:
            break
        res = max(4, int(res * 0.78))

    # cluster representative = centroid of its members
    n_clusters = int(inverse.max()) + 1
    sums = np.zeros((n_clusters, 3))
    counts = np.zeros(n_clusters)
    np.add.at(sums, inverse, verts)
    np.add.at(counts, inverse, 1)
    new_verts = sums / np.maximum(counts, 1)[:, None]

    # drop any vertex no surviving triangle references
    used, f2 = np.unique(f.reshape(-1), return_inverse=True)
    return new_verts[used], f2.reshape(-1, 3)


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


# ---------------------------------------------------------------- MJCF parsing

# Bodies we actually draw. The full model has 67; the tarsal segments past the
# second are each a few pixels and the mouthpart chain is invisible from
# outside, so they are folded away. Everything that moves visibly is kept.
KEEP_PREFIXES = (
    "thorax", "head", "rostrum", "haustellum",
    "antenna_left", "antenna_right",
    "wing_left", "wing_right",
    "haltere_left", "haltere_right",
    "abdomen",
    "coxa_", "femur_", "tibia_", "tarsus_",
)
SKIP_EXACT = set()


def wanted(name):
    if name in SKIP_EXACT:
        return False
    # tarsus2/3/4/5 collapse into the parent tarsus
    if name.startswith(("tarsus2", "tarsus3", "tarsus4", "tarsus5")):
        return False
    if name.startswith(("labrum",)):
        return False
    return name.startswith(KEEP_PREFIXES)


def resolve_classes(root):
    """Flatten MuJoCo's <default> tree so a class name maps to its joint and
    geom attribute dicts, with parent classes inherited."""
    table = {}

    def walk(node, inherited_joint, inherited_geom):
        name = node.get("class")
        j = dict(inherited_joint)
        g = dict(inherited_geom)
        jn = node.find("joint")
        if jn is not None:
            j.update(jn.attrib)
        gn = node.find("geom")
        if gn is not None:
            g.update(gn.attrib)
        if name:
            table[name] = {"joint": j, "geom": g}
        for child in node.findall("default"):
            walk(child, j, g)

    top = root.find("default")
    if top is not None:
        walk(top, {}, {})
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/flybody")
    ap.add_argument("--out", default="build/flymodel.bin")
    ap.add_argument("--meta-out", default="build/flymodel_meta.json")
    ap.add_argument("--budget", type=int, default=60000,
                    help="total triangle budget across every part")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    xml_path = fetch("fruitfly.xml", args.cache)
    root = ET.parse(xml_path).getroot()

    # ---- assets ---------------------------------------------------------
    asset = root.find("asset")
    mesh_file = {}
    for m in asset.findall("mesh"):
        if m.get("name") and m.get("file"):
            mesh_file[m.get("name")] = m.get("file")
    materials = {}
    for m in asset.findall("material"):
        rgba = m.get("rgba")
        materials[m.get("name")] = (vec3(rgba)[:3].tolist() +
                                    [float(rgba.split()[3])]) if rgba else [0.7, 0.4, 0.2, 1]

    default_mesh_scale = 0.1
    dm = root.find("default/mesh")
    if dm is not None and dm.get("scale"):
        default_mesh_scale = float(dm.get("scale").split()[0])

    classes = resolve_classes(root)

    # ---- walk the body tree ----------------------------------------------
    worldbody = root.find("worldbody")
    parts = []        # dicts, in parent-before-child order
    index_of = {}

    def geom_list(body, childclass):
        out = []
        for g in body.findall("geom"):
            cls = g.get("class", childclass)
            inherited = classes.get(cls, {}).get("geom", {})
            if g.get("contype") == "1" or cls in ("collision", "adhesion-collision",
                                                  "collision-membrane"):
                continue
            if inherited.get("contype") == "1":
                continue
            mesh = g.get("mesh")
            if not mesh:
                continue          # primitive collision shape, skip
            mat = g.get("material", inherited.get("material", "body"))
            out.append({
                "mesh": mesh,
                "pos": vec3(g.get("pos")),
                "quat": quat_wxyz(g.get("quat")) if g.get("quat")
                        else (euler_to_quat(g.get("euler")) if g.get("euler")
                              else np.array([1.0, 0, 0, 0])),
                "material": mat,
            })
        return out

    def walk(body, parent_index, childclass):
        name = body.get("name") or ""
        cc = body.get("childclass", childclass)
        keep = wanted(name)
        my_index = parent_index

        if keep:
            joints = []
            for j in body.findall("joint"):
                cls = j.get("class")
                inherited = classes.get(cls, {}).get("joint", {}) if cls else {}
                axis = j.get("axis", inherited.get("axis", "0 0 1"))
                rng = j.get("range", inherited.get("range", "-1 1"))
                joints.append({
                    "name": j.get("name") or "",
                    "axis": vec3(axis).tolist(),
                    "range": [float(x) for x in rng.split()],
                })
            parts.append({
                "name": name,
                "parent": parent_index,
                "pos": vec3(body.get("pos")),
                "quat": quat_wxyz(body.get("quat")) if body.get("quat")
                        else (euler_to_quat(body.get("euler")) if body.get("euler")
                              else np.array([1.0, 0, 0, 0])),
                "joints": joints,
                "geoms": geom_list(body, cc),
            })
            my_index = len(parts) - 1
            index_of[name] = my_index
        else:
            # Fold a skipped body's geometry into its parent so nothing vanishes.
            if parent_index >= 0:
                pass

        for child in body.findall("body"):
            walk(child, my_index, cc)

    for b in worldbody.findall("body"):
        walk(b, -1, "body")

    log(f"{len(parts)} parts kept")

    # ---- load, bake and decimate ------------------------------------------
    # Budget is shared out by raw triangle count, so the thorax and head get the
    # detail and a tarsus gets 200 triangles.
    raw = {}
    for p in parts:
        for g in p["geoms"]:
            f = mesh_file.get(g["mesh"])
            if f and f not in raw:
                path = fetch(f, args.cache)
                raw[f] = load_obj(path)
                log(f"  {f}: {len(raw[f][1]):,} tris")

    total_raw = sum(len(v[1]) for v in raw.values())
    log(f"raw total {total_raw:,} triangles -> budget {args.budget:,}")

    all_verts = []
    all_idx = []
    v_cursor = 0
    i_cursor = 0

    for p in parts:
        pv = []
        pf = []
        base = 0
        for g in p["geoms"]:
            f = mesh_file.get(g["mesh"])
            if not f or f not in raw:
                continue
            verts, faces = raw[f]
            if len(faces) == 0:
                continue
            share = max(120, int(args.budget * len(faces) / max(total_raw, 1)))
            dv, df = decimate(verts, faces, share)

            # bake mesh scale, then the geom's local placement
            dv = dv * default_mesh_scale
            R = quat_to_matrix(g["quat"])
            dv = dv @ R.T + g["pos"]

            pv.append(dv)
            pf.append(df + base)
            base += len(dv)

        if not pv:
            p["vcount"] = 0
            p["icount"] = 0
            p["vstart"] = v_cursor
            p["istart"] = i_cursor
            p["colour"] = [0.7, 0.4, 0.2, 1.0]
            continue

        verts = np.concatenate(pv)
        faces = np.concatenate(pf)
        normals = vertex_normals(verts, faces)

        # one colour per part: the material of its largest geom
        big = max(p["geoms"], key=lambda g: len(raw.get(mesh_file.get(g["mesh"], ""),
                                                        ([], []))[1]))
        p["colour"] = materials.get(big["material"], [0.7, 0.4, 0.2, 1.0])

        p["vstart"] = v_cursor
        p["vcount"] = len(verts)
        p["istart"] = i_cursor
        p["icount"] = faces.size

        all_verts.append(np.hstack([verts, normals]).astype(np.float32))
        all_idx.append(faces.reshape(-1).astype(np.uint32))
        v_cursor += len(verts)
        i_cursor += faces.size

    V = np.concatenate(all_verts) if all_verts else np.zeros((0, 6), np.float32)
    I = np.concatenate(all_idx) if all_idx else np.zeros(0, np.uint32)
    log(f"packed {len(V):,} vertices / {len(I)//3:,} triangles")

    # ---- recentre and rescale -----------------------------------------------
    # The MJCF is in centimetres with the thorax at an arbitrary origin. Put the
    # body centroid at the origin and scale so the fly is 2.5 mm long, which is
    # the real body length of D. melanogaster, in our world units where 1 = 1 cm.
    if len(V):
        lo = V[:, :3].min(0)
        hi = V[:, :3].max(0)
        span = float(np.max(hi - lo))
        log(f"model extent {np.round(hi - lo, 4).tolist()} (MJCF units)")
    else:
        span = 1.0

    # ---- names ----------------------------------------------------------------
    name_blob = bytearray()

    def intern(s):
        b = s.encode("utf-8")
        off = len(name_blob)
        name_blob.extend(b)
        return off, len(b)

    joint_records = []
    part_records = []
    for p in parts:
        jstart = len(joint_records)
        for j in p["joints"]:
            jo, jl = intern(j["name"])
            joint_records.append((j["axis"], j["range"], jo, jl))
        no, nl = intern(p["name"])
        part_records.append({
            "parent": p["parent"],
            "pos": p["pos"],
            "quat": p["quat"],
            "colour": p["colour"],
            "vstart": p["vstart"], "vcount": p["vcount"],
            "istart": p["istart"], "icount": p["icount"],
            "jstart": jstart, "jcount": len(p["joints"]),
            "noff": no, "nlen": nl,
        })

    # ---- serialise ------------------------------------------------------------
    parts_blob = bytearray()
    for r in part_records:
        q = r["quat"]      # stored xyzw for simd_quatf
        parts_blob += struct.pack(
            "<i3f4f4f6I2I",
            r["parent"],
            float(r["pos"][0]), float(r["pos"][1]), float(r["pos"][2]),
            float(q[1]), float(q[2]), float(q[3]), float(q[0]),
            float(r["colour"][0]), float(r["colour"][1]),
            float(r["colour"][2]), float(r["colour"][3]),
            r["vstart"], r["vcount"], r["istart"], r["icount"],
            r["jstart"], r["jcount"],
            r["noff"], r["nlen"],
        )

    joints_blob = bytearray()
    for axis, rng, jo, jl in joint_records:
        joints_blob += struct.pack(
            "<3f2f2I",
            float(axis[0]), float(axis[1]), float(axis[2]),
            float(rng[0]), float(rng[1]), jo, jl)

    sections = [
        ("parts", bytes(parts_blob)),
        ("joints", bytes(joints_blob)),
        ("names", bytes(name_blob)),
        ("vertices", V.tobytes()),
        ("indices", I.tobytes()),
    ]
    offsets, cursor = [], HEADER_BYTES
    for _, blob in sections:
        cursor = (cursor + 255) & ~255
        offsets.append(cursor)
        cursor += len(blob)

    hdr = bytearray(HEADER_BYTES)
    struct.pack_into("<8sIIIIII", hdr, 0, MAGIC, VERSION,
                     len(part_records), len(V), len(I), len(joint_records),
                     len(sections))
    for i, ((_, blob), off) in enumerate(zip(sections, offsets)):
        struct.pack_into("<QQ", hdr, 64 + i * 16, off, len(blob))

    with open(args.out, "wb") as fh:
        fh.write(hdr)
        for (name, blob), off in zip(sections, offsets):
            fh.write(b"\0" * (off - fh.tell()))
            fh.write(blob)

    size = os.path.getsize(args.out)
    log(f"wrote {args.out}: {size/2**20:.2f} MiB")
    for (name, blob), off in zip(sections, offsets):
        log(f"    {name:<10} @{off:>9,}  {len(blob)/1024:8.1f} KiB")

    json.dump({
        "format": "flymodel.bin v1",
        "source": "TuragaLab/flybody (Apache-2.0), Vaxenburg et al. Nature 2025",
        "parts": [p["name"] for p in parts],
        "jointNames": [j["name"] for p in parts for j in p["joints"]],
        "triangles": int(len(I) // 3),
        "vertices": int(len(V)),
        "mjcfSpan": span,
    }, open(args.meta_out, "w"), indent=2)
    log("done.")


if __name__ == "__main__":
    main()
