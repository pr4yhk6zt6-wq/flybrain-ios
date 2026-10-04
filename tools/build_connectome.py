#!/usr/bin/env python3
"""
Step 2 — FlyWire FAFB v783  ->  packed binary for direct upload into Metal buffers.

Produces `flybrain.bin`: a single little-endian file whose sections are laid out so
that the iOS app can mmap it and hand each section straight to `MTLDevice.makeBuffer`
with zero parsing and zero copies.

Layout
------
  [0]      Header            256 B, magic "FLYBRAIN", version, counts, section table
  [1]      rootIDs           uint64  x N      FlyWire root ids (neuron inspector)
  [2]      positions         half3   x N      normalised to a unit cube, centred
  [3]      neuronMeta        8 B     x N      packed: system/superclass/nt/flags/type
  [4]      csrRowPtr         uint32  x (N+1)  CSR over OUTGOING edges (row = presyn)
  [5]      csrColIdx         uint32  x E      postsynaptic neuron index
  [6]      csrWeight         half    x E      signed: + excitatory, - inhibitory
  [7]      csrDelay          uint8   x E      conduction delay in 1 ms steps
  [8]      retinaIdx         uint32  x R      photoreceptor/LMC neurons, camera input
  [9]      retinaUV          half2   x R      retinotopic coordinate in [0,1]^2
  [10]     motorIdx          uint32  x M      descending neurons, motor readout

Neurons are sorted by (system, super_class, root_id) so every view mode is a
contiguous index range -- the renderer filters by range, never by predicate.

Usage:
  python3 tools/build_connectome.py --raw data/raw --out build/flybrain.bin
"""
from __future__ import annotations
import argparse, json, os, struct, sys, time
import numpy as np
import pandas as pd

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

MAGIC = b"FLYBRAIN"
VERSION = 1
HEADER_BYTES = 256

# ---------------------------------------------------------------- taxonomies
# Order matters: it defines the on-device enum and the view-mode ordering.
SYSTEMS = [
    "optic_lobe",            # 0
    "central_complex",       # 1
    "olfactory",             # 2
    "learning_memory",       # 3
    "gnathic",               # 4
    "antennal_mechanosensory",# 5
    "olfactory_lateral",     # 6
    "central_brain_other",   # 7
    "other",                 # 8
    "unassigned",            # 9
]
SYSTEM_ID = {s: i for i, s in enumerate(SYSTEMS)}

SUPER_CLASSES = [
    "optic", "central", "sensory", "visual_projection", "ascending",
    "descending", "sensory_ascending", "visual_centrifugal", "motor",
    "endocrine", "unknown",
]
SUPER_ID = {s: i for i, s in enumerate(SUPER_CLASSES)}

# neurotransmitter -> (id, sign). Glutamate is INHIBITORY in Drosophila (GluCl).
NT = {
    "ACH":  (0, +1.0),
    "GABA": (1, -1.0),
    "GLUT": (2, -1.0),
    "DA":   (3, +0.3),   # modulatory, weak
    "SER":  (4, +0.3),
    "OCT":  (5, +0.3),
}
NT_UNKNOWN = (6, +1.0)

_BASE_SYSTEM = {
    "LA":"optic_lobe","LO":"optic_lobe","LOP":"optic_lobe","ME":"optic_lobe",
    "MED":"optic_lobe","AOTU":"optic_lobe","AME":"optic_lobe","LOPLA":"optic_lobe",
    "LOPM":"optic_lobe",
    "AMMC":"antennal_mechanosensory",
    "EB":"central_complex","FB":"central_complex","PB":"central_complex",
    "NO":"central_complex","PON":"central_complex","SAD":"central_complex",
    "CAN":"central_complex","FLT":"central_complex","GA":"central_complex",
    "ICL":"central_complex","IDF":"central_complex","RUB":"central_complex",
    "AL":"olfactory","LH":"olfactory",
    "MB":"learning_memory","CA":"learning_memory","PED":"learning_memory",
    "SOG":"gnathic","GNG":"gnathic","PRW":"gnathic","SPS":"gnathic",
    "IPS":"gnathic","WED":"gnathic",
}
_CENTRAL_OTHER_PREFIX = ("SLP","AVLP","PVLP","PLP","SMP","SIP","SFS","GOR",
                         "CRE","VES","EPA","EPP","OCI")

def system_of(neuropil) -> str:
    if not isinstance(neuropil, str) or not neuropil:
        return "unassigned"
    base = neuropil.rsplit("_", 1)[0] if neuropil.endswith(("_L", "_R")) else neuropil
    if base in _BASE_SYSTEM:
        return _BASE_SYSTEM[base]
    if base.startswith(("ME", "LA", "LO", "AOTU", "AME")):
        return "optic_lobe"
    if base.startswith(_CENTRAL_OTHER_PREFIX):
        return "central_brain_other"
    if base.startswith(("DA", "VA", "DL", "VM", "DC", "VC", "VP", "VL")):
        return "olfactory_lateral"
    return "other"

# Lamina monopolars + inner photoreceptors: the camera drives these.
RETINA_TYPES = {"L1", "L2", "L3", "L4", "L5", "R7", "R8"}

# ------------------------------------------------------------------ argparse
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--out", default="build/flybrain.bin")
ap.add_argument("--meta", default="build/flybrain_meta.json")
ap.add_argument("--conduction-speed", type=float, default=0.5,
                help="m/s, for synthesising delays from soma distance")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
R = args.raw

# ============================================================ load the tables
log("loading neurons / classification / cell types ...")
neurons = pd.read_csv(f"{R}/neurons.csv.gz",
                      usecols=["root_id", "nt_type"],
                      dtype={"root_id": "int64"})
classif = pd.read_csv(f"{R}/classification.csv.gz",
                      usecols=["root_id", "flow", "super_class", "class", "side"],
                      dtype={"root_id": "int64"})
celltyp = pd.read_csv(f"{R}/consolidated_cell_types.csv.gz",
                      usecols=["root_id", "primary_type"],
                      dtype={"root_id": "int64"})

log("loading + deduplicating coordinates (centroid per neuron) ...")
coords = pd.read_csv(f"{R}/coordinates.csv.gz", usecols=["root_id", "position"],
                     dtype={"root_id": "int64"})
xyz = np.array([[int(v) for v in s.strip()[1:-1].split()]
                for s in coords["position"].to_numpy()], dtype=np.float64)
coords["x"], coords["y"], coords["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
coords = coords.groupby("root_id", as_index=True)[["x", "y", "z"]].mean()
log(f"  {len(coords):,} unique neurons with a centroid")

log("loading connections.csv.gz ...")
conn = pd.read_csv(f"{R}/connections.csv.gz",
                   usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
                   dtype={"pre_root_id": "int64", "post_root_id": "int64",
                          "neuropil": "category", "syn_count": "int32"})
log(f"  {len(conn):,} pair x neuropil rows")

# =================================================== dominant neuropil -> system
log("assigning each neuron its dominant neuropil ...")
by_post = (conn.groupby(["post_root_id", "neuropil"], observed=True)["syn_count"]
               .sum().reset_index())
dom = (by_post.loc[by_post.groupby("post_root_id")["syn_count"].idxmax()]
              .set_index("post_root_id")["neuropil"].astype(str))
# presynaptic-only neurons (pure afferents) fall back to their output neuropil
by_pre = (conn.groupby(["pre_root_id", "neuropil"], observed=True)["syn_count"]
              .sum().reset_index())
dom_pre = (by_pre.loc[by_pre.groupby("pre_root_id")["syn_count"].idxmax()]
                 .set_index("pre_root_id")["neuropil"].astype(str))
del by_post, by_pre

# ======================================================== assemble neuron table
log("assembling the neuron table ...")
N_tab = neurons.set_index("root_id")
N_tab = N_tab.join(classif.set_index("root_id"), how="left")
N_tab = N_tab.join(celltyp.set_index("root_id"), how="left")
N_tab = N_tab.join(coords, how="left")
assert not N_tab.index.duplicated().any(), "neuron table fanned out on join"

neuropil = dom.reindex(N_tab.index)
neuropil = neuropil.fillna(dom_pre.reindex(N_tab.index))
N_tab["neuropil"] = neuropil
N_tab["system"] = [system_of(v) for v in N_tab["neuropil"]]

missing_pos = N_tab["x"].isna().sum()
if missing_pos:
    log(f"  WARNING {missing_pos} neurons lack a position -> placed at centroid")
    for c in "xyz":
        N_tab[c] = N_tab[c].fillna(N_tab[c].mean())

N_tab["system_id"] = N_tab["system"].map(SYSTEM_ID).astype("uint8")
N_tab["super_id"] = (N_tab["super_class"].fillna("unknown")
                     .map(lambda s: SUPER_ID.get(s, SUPER_ID["unknown"]))).astype("uint8")
N_tab["nt_id"] = (N_tab["nt_type"].fillna("")
                  .map(lambda s: NT.get(s, NT_UNKNOWN)[0])).astype("uint8")
N_tab["nt_sign"] = (N_tab["nt_type"].fillna("")
                    .map(lambda s: NT.get(s, NT_UNKNOWN)[1])).astype("float32")

# ----------------------------------------------- deterministic ordering (view modes)
log("sorting neurons by (system, super_class, root_id) for contiguous view ranges ...")
N_tab = N_tab.reset_index().sort_values(
    ["system_id", "super_id", "root_id"], kind="mergesort").reset_index(drop=True)
N = len(N_tab)
index_of = pd.Series(np.arange(N, dtype=np.uint32), index=N_tab["root_id"].to_numpy())
log(f"  N = {N:,} neurons")

# contiguous [start,count) range per system, for the renderer's filter
system_ranges = {}
sid = N_tab["system_id"].to_numpy()
for name, i in SYSTEM_ID.items():
    hits = np.flatnonzero(sid == i)
    system_ranges[name] = {"start": int(hits[0]), "count": int(len(hits))} if len(hits) \
        else {"start": 0, "count": 0}
super_ranges = {}
sup = N_tab["super_id"].to_numpy()
for name, i in SUPER_ID.items():
    hits = np.flatnonzero(sup == i)
    super_ranges[name] = {"count": int(len(hits))}

# ============================================================== positions
log("normalising positions to a centred unit cube ...")
# FAFB14.1 raw voxels are anisotropic: 4 x 4 x 40 nm. Convert to nm first or the
# brain renders squashed along z.
VOXEL_NM = np.array([4.0, 4.0, 40.0])
P = N_tab[["x", "y", "z"]].to_numpy(np.float64) * VOXEL_NM
lo, hi = P.min(0), P.max(0)
span = (hi - lo).max()                       # isotropic scale: keep proportions
P = (P - (lo + hi) / 2.0) / span             # centred, longest axis spans [-0.5, 0.5]
log(f"  extent nm = {(hi-lo).astype(int).tolist()}  -> scale 1/{span:.0f}")
positions_f16 = P.astype(np.float16)

# distance matrix is impossible; keep nm positions for delay computation
P_nm = N_tab[["x", "y", "z"]].to_numpy(np.float64) * VOXEL_NM

# ============================================================== edges -> CSR
log("aggregating connections to unique directed pairs ...")
pair = (conn.groupby(["pre_root_id", "post_root_id"], observed=True)["syn_count"]
            .sum().rename("syn").reset_index())
del conn
log(f"  {len(pair):,} unique directed pairs")

# drop any edge touching a neuron that is not in the proofread neuron table
known = index_of.index
m = pair["pre_root_id"].isin(known) & pair["post_root_id"].isin(known)
dropped = int((~m).sum())
pair = pair[m]
log(f"  dropped {dropped:,} edges referencing non-neuron segments; {len(pair):,} remain")

pre_i = index_of.loc[pair["pre_root_id"].to_numpy()].to_numpy(np.uint32)
post_i = index_of.loc[pair["post_root_id"].to_numpy()].to_numpy(np.uint32)
syn = pair["syn"].to_numpy(np.float32)

# ---- signed weights --------------------------------------------------------
# FlyWire ships no synaptic weights. Sign comes from the PREsynaptic neuron's
# predicted neurotransmitter; magnitude is log1p(syn_count), which compresses the
# 1..93k synapse-count range into something a LIF membrane can integrate.
log("signing weights by presynaptic neurotransmitter ...")
nt_sign = N_tab["nt_sign"].to_numpy(np.float32)
w = np.log1p(syn) * nt_sign[pre_i]
# normalise so the MEDIAN total excitatory drive onto a neuron is 1.0; the app
# then has a single scalar gain to tune instead of per-edge magic numbers.
exc = np.where(w > 0, w, 0.0)
drive = np.bincount(post_i, weights=exc, minlength=N)
med = float(np.median(drive[drive > 0]))
w = (w / med).astype(np.float32)
log(f"  median excitatory drive {med:.2f} -> normalised; |w| range "
    f"[{np.abs(w).min():.4f}, {np.abs(w).max():.4f}]")

# ---- synthetic conduction delays ------------------------------------------
log(f"synthesising delays from soma distance @ {args.conduction_speed} m/s ...")
d_nm = np.linalg.norm(P_nm[post_i] - P_nm[pre_i], axis=1)
d_m = d_nm * 1e-9
delay_ms = d_m / args.conduction_speed * 1e3
delay = np.clip(np.round(delay_ms), 1, 255).astype(np.uint8)
log(f"  delay ms: min {delay.min()} median {int(np.median(delay))} "
    f"mean {delay.mean():.1f} max {delay.max()}")

# ---- CSR over outgoing edges ----------------------------------------------
log("building CSR (row = presynaptic neuron) ...")
order = np.argsort(pre_i, kind="stable")
pre_s, col_idx = pre_i[order], post_i[order]
weight, delay_s = w[order].astype(np.float16), delay[order]
row_ptr = np.zeros(N + 1, dtype=np.uint32)
np.cumsum(np.bincount(pre_s, minlength=N), out=row_ptr[1:])
E = len(col_idx)
assert row_ptr[-1] == E
log(f"  E = {E:,} edges; max out-degree = {int(np.diff(row_ptr).max()):,}")

# ================================================= retina + motor index lists
log("selecting camera-input (retina) and motor-readout neurons ...")
ptype = N_tab["primary_type"].fillna("").astype(str)
is_retina = ptype.isin(RETINA_TYPES).to_numpy()
retina_idx = np.flatnonzero(is_retina).astype(np.uint32)

# Retinotopic UV: the optic lobe is a curved sheet, so take the two principal
# axes of the retina point cloud and use those as the screen coordinates. Done
# per hemisphere, because the two eyes face opposite ways.
retina_uv = np.zeros((len(retina_idx), 2), dtype=np.float32)
side = N_tab["side"].fillna("").to_numpy()
for s in ("left", "right"):
    sel = np.flatnonzero(is_retina & (side == s))
    if len(sel) < 3:
        continue
    pts = P[sel]
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    uv = c @ vt[:2].T
    uv -= uv.min(0); uv /= np.maximum(uv.max(0), 1e-9)
    if s == "right":            # mirror so both eyes index the same image space
        uv[:, 0] = 1.0 - uv[:, 0]
    pos_in_list = np.searchsorted(retina_idx, sel)
    retina_uv[pos_in_list] = uv
    log(f"  retina {s}: {len(sel):,} cells")
retina_uv_f16 = retina_uv.astype(np.float16)

motor_mask = (N_tab["super_id"].to_numpy() == SUPER_ID["descending"]) | \
             (N_tab["super_id"].to_numpy() == SUPER_ID["motor"])
motor_idx = np.flatnonzero(motor_mask).astype(np.uint32)
log(f"  retina total {len(retina_idx):,} | motor readout {len(motor_idx):,}")

# ============================================================== neuron meta
# 8 bytes per neuron: system, super_class, nt, flags, cellTypeId(u16), pad(u16)
log("packing per-neuron metadata ...")
type_names = sorted(set(ptype[ptype != ""]))
type_id_of = {t: i + 1 for i, t in enumerate(type_names)}      # 0 = untyped
cell_type_id = ptype.map(lambda s: type_id_of.get(s, 0)).to_numpy(np.uint16)
flags = np.zeros(N, np.uint8)
flags |= (is_retina.astype(np.uint8) << 0)
flags |= (motor_mask.astype(np.uint8) << 1)
flags |= ((N_tab["flow"].fillna("") == "afferent").to_numpy().astype(np.uint8) << 2)
flags |= ((N_tab["flow"].fillna("") == "efferent").to_numpy().astype(np.uint8) << 3)
flags |= ((side == "left").astype(np.uint8) << 4)
flags |= ((side == "right").astype(np.uint8) << 5)

meta = np.zeros((N, 8), dtype=np.uint8)
meta[:, 0] = N_tab["system_id"].to_numpy(np.uint8)
meta[:, 1] = N_tab["super_id"].to_numpy(np.uint8)
meta[:, 2] = N_tab["nt_id"].to_numpy(np.uint8)
meta[:, 3] = flags
meta[:, 4:6] = cell_type_id.view(np.uint8).reshape(-1, 2)      # little-endian u16

# =================================================================== write it
log(f"writing {args.out} ...")
sections = [
    ("rootIDs",   N_tab["root_id"].to_numpy(np.uint64).tobytes()),
    ("positions", positions_f16.tobytes()),
    ("neuronMeta", meta.tobytes()),
    ("csrRowPtr", row_ptr.tobytes()),
    ("csrColIdx", col_idx.tobytes()),
    ("csrWeight", weight.tobytes()),
    ("csrDelay",  delay_s.tobytes()),
    ("retinaIdx", retina_idx.tobytes()),
    ("retinaUV",  retina_uv_f16.tobytes()),
    ("motorIdx",  motor_idx.tobytes()),
]

# 256-byte alignment keeps every section eligible for a no-copy MTLBuffer
offsets, cursor = [], HEADER_BYTES
for _, blob in sections:
    cursor = (cursor + 255) & ~255
    offsets.append(cursor)
    cursor += len(blob)
total = cursor

hdr = bytearray(HEADER_BYTES)
struct.pack_into("<8sIIIIII", hdr, 0, MAGIC, VERSION, N, E,
                 len(retina_idx), len(motor_idx), len(sections))
for i, ((_, blob), off) in enumerate(zip(sections, offsets)):
    struct.pack_into("<QQ", hdr, 64 + i * 16, off, len(blob))

with open(args.out, "wb") as fh:
    fh.write(hdr)
    for (name, blob), off in zip(sections, offsets):
        fh.write(b"\0" * (off - fh.tell()))
        fh.write(blob)
assert os.path.getsize(args.out) == total

log(f"  {total/2**20:.2f} MiB uncompressed")
for (name, blob), off in zip(sections, offsets):
    log(f"    {name:<12} @{off:>10,}  {len(blob)/2**20:7.2f} MiB")

# --------------------------------------------------------------- sidecar JSON
metadata = {
    "format": "flybrain.bin v1",
    "source": "FlyWire FAFB v783 (CC BY-NC 4.0, FlyWire Consortium)",
    "neurons": int(N), "edges": int(E),
    "synapses": int(syn.sum()),
    "systems": SYSTEMS, "systemRanges": system_ranges,
    "superClasses": SUPER_CLASSES, "superClassCounts": super_ranges,
    "neurotransmitters": {k: v[0] for k, v in NT.items()},
    "cellTypes": type_names,
    "retinaCount": int(len(retina_idx)), "motorCount": int(len(motor_idx)),
    "weightNormalisation": {"medianExcitatoryDrive": med,
                            "rule": "sign(NT) * log1p(syn_count) / median_drive"},
    "delayModel": {"conductionSpeed_m_per_s": args.conduction_speed,
                   "unit": "ms", "storage": "uint8 clamped 1..255"},
    "positionNormalisation": {"voxel_nm": VOXEL_NM.tolist(),
                              "extent_nm": (hi - lo).tolist(),
                              "scale_divisor_nm": float(span),
                              "note": "centred, isotropic, longest axis in [-0.5,0.5]"},
}
with open(args.meta, "w") as fh:
    json.dump(metadata, fh, indent=2)
log(f"wrote {args.meta}")
log("done.")
