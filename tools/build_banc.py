#!/usr/bin/env python3
"""
Build the embodied connectome from BANC v888 — brain AND ventral nerve cord,
from a single female fly, with real leg and wing motor neurons.

This supersedes build_connectome.py (FAFB v783, brain only). FAFB had 110 brain
motor neurons and no nerve cord, so "descending neuron fires -> leg moves" had to
be invented. BANC closes that loop with measured wiring:

    391 leg motor neurons   (front / middle / hind, both sides)
     60 wing motor neurons  (24 power, 24 steering, 12 tension)
     49 neck, 35 proboscis, 27 haltere, 2 jump/escape
   2903 proprioceptors, 6699 tactile, 1187 auditory, 3011 olfactory
   1316 descending, 1849 ascending, 12866 VNC intrinsic

Sources (both public, no authentication):
  https://storage.googleapis.com/flywire-data/codex/data/banc/888/
      neurons.csv.gz, connections_princeton.csv.gz
  https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/
      compiled_data/banc_888/banc_888_meta.feather   (positions, cell_function)

Output: flybanc.bin, format v2 — the v1 sections plus named neuron GROUPS, which
is what the virtual body reads to turn spikes into leg and wing commands.
"""
from __future__ import annotations
import argparse, json, os, struct, time
import numpy as np
import pandas as pd
import pyarrow.feather as feather

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

MAGIC = b"FLYBANC_"
VERSION = 2
HEADER_BYTES = 512

# ------------------------------------------------------------------ taxonomy
SYSTEMS = [
    "optic_lobe",            # 0
    "central_brain",         # 1
    "visual_projection",     # 2
    "descending",            # 3
    "ascending",             # 4
    "vnc_prothoracic",       # 5  T1  front legs
    "vnc_mesothoracic",      # 6  T2  middle legs + wings
    "vnc_metathoracic",      # 7  T3  hind legs + halteres
    "vnc_abdominal",         # 8  A1-A9
    "sensory",               # 9
    "motor",                 # 10
    "other",                 # 11
]
SYSTEM_ID = {s: i for i, s in enumerate(SYSTEMS)}

SUPER_CLASSES = [
    "optic_lobe_intrinsic", "central_brain_intrinsic", "sensory",
    "ventral_nerve_cord_intrinsic", "visual_projection", "ascending",
    "descending", "motor", "sensory_ascending", "visual_centrifugal",
    "visceral_circulatory", "sensory_descending", "unknown",
]
SUPER_ID = {s: i for i, s in enumerate(SUPER_CLASSES)}

# Glutamate is inhibitory in Drosophila (GluCl). Histamine likewise — it is the
# photoreceptor transmitter and hyperpolarises its targets.
NT = {
    "acetylcholine": (0, +1.0), "ach": (0, +1.0),
    "gaba":          (1, -1.0),
    "glutamate":     (2, -1.0), "glut": (2, -1.0),
    "dopamine":      (3, +0.3), "da": (3, +0.3),
    "serotonin":     (4, +0.3), "ser": (4, +0.3),
    "octopamine":    (5, +0.3), "oct": (5, +0.3),
    "histamine":     (6, -1.0),
    "tyramine":      (7, +0.3),
}
NT_UNKNOWN = (8, +1.0)

NOT_NEURONS = {"glia", "not_a_neuron", "trachea"}

# ---- the groups the virtual body actually reads -----------------------------
# Each entry: (group name, predicate over the neuron table).
# The motor pools, as groups. The app already reads named groups off the GPU
# (`SimulationEngine.groupRate`), so giving it one group per pool means step 6
# can ask the question it actually has -- "how fast is the pool that extends
# this leg's tibia firing?" -- without new Metal code, and without the app
# carrying its own copy of which neurons are in which pool.
from motor_pools import pool_groups, organ_groups   # noqa: E402


def build_group_specs():
    def fn(col, *values):
        vals = set(values)
        return lambda d: d[col].isin(vals)

    def fn_and(a, b):
        return lambda d: a(d) & b(d)

    specs = []

    # --- motor output, split by body part and side ---------------------------
    for part in ("front_leg", "middle_leg", "hind_leg"):
        for side in ("left", "right"):
            specs.append((f"motor_{part}_{side}",
                          lambda d, p=part, s=side:
                              (d["super_class"] == "motor") &
                              (d["body_part_effector"] == p) &
                              (d["side"] == s)))
    for fnname in ("wing_power", "wing_steering", "wing_tension"):
        for side in ("left", "right"):
            specs.append((f"motor_{fnname}_{side}",
                          lambda d, f=fnname, s=side:
                              (d["cell_function"] == f) & (d["side"] == s)))
    specs.append(("motor_neck",      fn("cell_function", "neck_motor")))
    specs.append(("motor_proboscis", fn("cell_function", "proboscis_motor")))
    specs.append(("motor_haltere",   fn("cell_function", "haltere_steering",
                                        "haltere_motor", "haltere_power")))
    specs.append(("motor_jump_escape", fn("cell_function", "jump_escape")))
    specs.append(("motor_abdomen",   lambda d: (d["super_class"] == "motor") &
                                               (d["body_part_effector"] == "abdomen")))

    # --- sensory input -------------------------------------------------------
    # BANC is missing most of the lamina and retina proper, so only ~1,360
    # cells carry a retina/lamina body-part tag. The lamina monopolars and
    # inner photoreceptors are present as CELL TYPES inside the optic lobe
    # though, and they are the right input layer anyway — same choice the FAFB
    # build made. ~8,500 cells instead of 1,360.
    RETINA_TYPES = {"L1", "L2", "L3", "L4", "L5", "R7", "R8"}
    specs.append(("sensory_vision",
                  lambda d: d["cell_type"].isin(RETINA_TYPES) |
                            d["body_part_sensory"].isin(
                                {"retina", "lamina", "interommatidial", "ocellus"})))
    specs.append(("sensory_olfactory",   fn("cell_function", "olfactory")))
    specs.append(("sensory_gustatory",   fn("cell_function", "gustatory")))
    specs.append(("sensory_auditory",    fn("cell_function", "auditory")))
    specs.append(("sensory_tactile",     fn("cell_function", "tactile")))
    specs.append(("sensory_proprioception", fn("cell_function", "proprioception")))
    specs.append(("sensory_nociception", fn("cell_function", "nociception")))
    specs.append(("sensory_hygro_thermo",
                  fn("cell_function", "hygrosensory", "thermosensory")))
    for part in ("front_leg", "middle_leg", "hind_leg"):
        specs.append((f"sensory_{part}",
                      lambda d, p=part: d["body_part_sensory"] == p))
    # Wing and haltere mechanosensors are lateralised: the campaniform fields
    # of each haltere encode rotation toward that side (Dickinson 1999 — the
    # equilibrium reflex is unilaterally lost for motions toward an ablated
    # haltere), so the body must be able to drive the two sides separately.
    # BANC annotates side on essentially all of them (haltere 216L/212R/11
    # unassigned; wing 272L/435R/45 unassigned).
    for side in ("left", "right"):
        specs.append((f"sensory_wing_{side}",
                      lambda d, s=side:
                          d["body_part_sensory"].isin({"wing_margin", "wing_base"}) &
                          (d["side"] == s)))
        specs.append((f"sensory_haltere_{side}",
                      lambda d, s=side:
                          (d["body_part_sensory"] == "haltere") &
                          (d["side"] == s)))
    specs.append(("sensory_antenna", fn("body_part_sensory", "antenna")))

    # --- the relay layers ----------------------------------------------------
    specs.append(("descending", fn("super_class", "descending")))
    specs.append(("ascending",  fn("super_class", "ascending",
                                   "sensory_ascending")))
    # --- the motor pools, per leg and side, as step 2 defined them ----------
    # These are the groups a closed loop reads: each one is the set of motor
    # neurons that innervates one muscle set on one leg, and the muscle set is
    # what makes one joint move one way.
    # The pools, and the sense organs they are closed onto. The organism's loop
    # is: organ -> cord -> pool -> muscle -> joint -> organ, and every arrow of
    # it is a name in this file, so both ends of the loop are read out of the
    # connectome the app is running rather than named on the phone.
    specs.extend(pool_groups())
    specs.extend(organ_groups())
    return specs


def neuromere_system(neuromere, super_class):
    if isinstance(neuromere, str):
        if neuromere == "T1": return "vnc_prothoracic"
        if neuromere == "T2": return "vnc_mesothoracic"
        if neuromere == "T3": return "vnc_metathoracic"
        if neuromere.startswith("A"): return "vnc_abdominal"
    sc = super_class if isinstance(super_class, str) else ""
    if sc == "optic_lobe_intrinsic":            return "optic_lobe"
    if sc == "visual_projection":               return "visual_projection"
    if sc == "visual_centrifugal":              return "optic_lobe"
    if sc == "central_brain_intrinsic":         return "central_brain"
    if sc == "descending":                      return "descending"
    if sc in ("ascending", "sensory_ascending", "sensory_descending"):
        return "ascending"
    if sc == "motor":                           return "motor"
    if sc == "sensory":                         return "sensory"
    if sc == "ventral_nerve_cord_intrinsic":    return "vnc_mesothoracic"
    return "other"


# ------------------------------------------------------------------- argparse
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/banc")
ap.add_argument("--out", default="build/flybanc.bin")
ap.add_argument("--meta-out", default="build/flybanc_meta.json")
# Fly axons are thin and unmyelinated; 0.1-0.5 m/s is the measured range. BANC
# spans about 1.4 mm, so 0.5 m/s collapses every delay to 1-2 ms. 0.25 m/s puts
# the spread at 1-7 ms, which is both defensible and visible.
ap.add_argument("--conduction-speed", type=float, default=0.25)
ap.add_argument("--min-syn", type=int, default=3,
                help="BANC's recommended pair threshold")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
R = args.raw

# ================================================================= load meta
log("reading banc_888_meta.feather ...")
# `cell_function_detailed` is here for the sense organs: the pools are selected
# by muscle (`peripheral_target_type`), and the organs by what kind of
# proprioceptor they are (`cell_function_detailed` — joint_angle, stretch,
# mechanical_strain), which is the predicate step 2 measured the reflex through.
cols = ["root_id", "root_position_nm", "side", "region", "neuromere", "nerve",
        "flow", "super_class", "cell_class", "cell_type",
        "body_part_sensory", "body_part_effector", "peripheral_target_type",
        "cell_function", "cell_function_detailed",
        "neurotransmitter_predicted", "proofread"]
meta = feather.read_table(f"{R}/banc_888_meta.feather", columns=cols).to_pandas()
log(f"  {len(meta):,} rows")

meta["super_class"] = meta["super_class"].fillna("unknown")
meta = meta[~meta["super_class"].isin(NOT_NEURONS)]
log(f"  {len(meta):,} after dropping glia / trachea / non-neurons")

# root_position_nm is "x, y, z" in nanometres already.
def parse_nm(s):
    if not isinstance(s, str) or not s.strip():
        return (np.nan, np.nan, np.nan)
    parts = s.replace(",", " ").split()
    if len(parts) < 3:
        return (np.nan, np.nan, np.nan)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return (np.nan, np.nan, np.nan)

log("parsing positions ...")
xyz = np.array([parse_nm(s) for s in meta["root_position_nm"].to_numpy()])
meta["x"], meta["y"], meta["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
has_pos = np.isfinite(xyz).all(axis=1)
log(f"  {has_pos.sum():,} of {len(meta):,} have a position "
    f"({has_pos.mean()*100:.1f}%)")

# Dropping unplaceable neurons outright costs over a million edges, because the
# 18,708 neurons without a representative point are concentrated in the optic
# lobe and are heavily connected. 98% of them do carry a region label, so place
# them at that region's centroid with a little jitter: wrong in detail, but it
# keeps their wiring in the simulation and puts them in roughly the right part
# of the brain on screen.
meta["has_position"] = has_pos
n_missing = int((~has_pos).sum())
if n_missing:
    log(f"  imputing {n_missing:,} positions from region centroids ...")
    placed = meta[has_pos]
    centroid = placed.groupby("region")[["x", "y", "z"]].mean()
    globalc = placed[["x", "y", "z"]].mean()
    rng = np.random.default_rng(7)
    miss = ~has_pos
    reg = meta.loc[miss, "region"]
    for axis in ("x", "y", "z"):
        vals = reg.map(centroid[axis]).to_numpy(np.float64)
        vals = np.where(np.isfinite(vals), vals, float(globalc[axis]))
        spread = float(placed[axis].std()) * 0.04
        meta.loc[miss, axis] = vals + rng.normal(0, spread, miss.sum())
    recovered = int(reg.notna().sum())
    log(f"    {recovered:,} placed by region, {n_missing - recovered:,} at the global centroid")

meta = meta.drop_duplicates(subset="root_id", keep="first")
meta["root_id"] = meta["root_id"].astype("int64")
log(f"  {len(meta):,} neurons carried forward")

# =========================================================== load connections
log("reading connections_princeton.csv.gz ...")
conn = pd.read_csv(f"{R}/connections_princeton.csv.gz",
                   dtype={"pre_root_id": "int64", "post_root_id": "int64",
                          "neuropil": "category", "syn_count": "int32",
                          "nt_type": "category"})
log(f"  {len(conn):,} pair x neuropil rows, "
    f"{int(conn['syn_count'].sum()):,} synapses")

log("aggregating to unique directed pairs ...")
pair = (conn.groupby(["pre_root_id", "post_root_id"], observed=True)["syn_count"]
            .sum().rename("syn").reset_index())
del conn
before = len(pair)
pair = pair[pair["syn"] >= args.min_syn]
log(f"  {before:,} pairs -> {len(pair):,} at >= {args.min_syn} synapses")

# keep only edges whose endpoints we can place in space
known = set(meta["root_id"].to_numpy())
m = pair["pre_root_id"].isin(known) & pair["post_root_id"].isin(known)
log(f"  dropping {int((~m).sum()):,} edges with an unplaceable endpoint")
pair = pair[m]

# ======================================================= order + index neurons
log("ordering neurons so every view mode is a contiguous range ...")
meta["system"] = [neuromere_system(n, s)
                  for n, s in zip(meta["neuromere"], meta["super_class"])]
meta["system_id"] = meta["system"].map(SYSTEM_ID).fillna(SYSTEM_ID["other"]).astype("uint8")
meta["super_id"] = meta["super_class"].map(
    lambda s: SUPER_ID.get(s, SUPER_ID["unknown"])).astype("uint8")

ntp = meta["neurotransmitter_predicted"].fillna("").astype(str).str.lower().str.strip()
meta["nt_id"] = ntp.map(lambda s: NT.get(s, NT_UNKNOWN)[0]).astype("uint8")
meta["nt_sign"] = ntp.map(lambda s: NT.get(s, NT_UNKNOWN)[1]).astype("float32")

meta = meta.sort_values(["system_id", "super_id", "root_id"],
                        kind="mergesort").reset_index(drop=True)
N = len(meta)
index_of = pd.Series(np.arange(N, dtype=np.uint32),
                     index=meta["root_id"].to_numpy())
log(f"  N = {N:,}")

system_ranges = {}
sid = meta["system_id"].to_numpy()
for name, i in SYSTEM_ID.items():
    hits = np.flatnonzero(sid == i)
    system_ranges[name] = ({"start": int(hits[0]), "count": int(len(hits))}
                           if len(hits) else {"start": 0, "count": 0})

# ================================================================= positions
log("normalising positions ...")
P = meta[["x", "y", "z"]].to_numpy(np.float64)
lo, hi = P.min(0), P.max(0)
span = (hi - lo).max()
Pn = (P - (lo + hi) / 2.0) / span
log(f"  extent nm = {(hi-lo).astype(int).tolist()}  scale 1/{span:.0f}")
positions_f16 = Pn.astype(np.float16)

# ==================================================================== edges
pre_i = index_of.loc[pair["pre_root_id"].to_numpy()].to_numpy(np.uint32)
post_i = index_of.loc[pair["post_root_id"].to_numpy()].to_numpy(np.uint32)
syn = pair["syn"].to_numpy(np.float32)

log("signing weights by presynaptic neurotransmitter ...")
nt_sign = meta["nt_sign"].to_numpy(np.float32)
w = np.log1p(syn) * nt_sign[pre_i]
exc = np.where(w > 0, w, 0.0)
drive = np.bincount(post_i, weights=exc, minlength=N)
med = float(np.median(drive[drive > 0]))
w = (w / med).astype(np.float32)
log(f"  {(w>0).mean()*100:.1f}% excitatory; median drive {med:.2f}")

log(f"synthesising delays @ {args.conduction_speed} m/s ...")
d_m = np.linalg.norm(P[post_i] - P[pre_i], axis=1) * 1e-9
delay = np.clip(np.round(d_m / args.conduction_speed * 1e3), 1, 255).astype(np.uint8)
log(f"  delay ms: median {int(np.median(delay))} max {int(delay.max())}")

log("building CSR ...")
order = np.argsort(pre_i, kind="stable")
pre_s, col_idx = pre_i[order], post_i[order]
weight, delay_s = w[order].astype(np.float16), delay[order]
row_ptr = np.zeros(N + 1, dtype=np.uint32)
np.cumsum(np.bincount(pre_s, minlength=N), out=row_ptr[1:])
E = len(col_idx)
log(f"  E = {E:,}; max out-degree {int(np.diff(row_ptr).max()):,}")

# ==================================================================== groups
log("resolving body groups ...")
for c in ("body_part_effector", "body_part_sensory", "cell_function",
          "side", "cell_type", "region"):
    meta[c] = meta[c].fillna("").astype(str)

group_indices, group_table = [], []
cursor = 0
for name, predicate in build_group_specs():
    idx = np.flatnonzero(predicate(meta).to_numpy()).astype(np.uint32)
    group_table.append({"name": name, "start": cursor, "count": int(len(idx))})
    group_indices.append(idx)
    cursor += len(idx)
    if len(idx):
        log(f"    {name:<28} {len(idx):>6,}")
group_indices = (np.concatenate(group_indices) if group_indices
                 else np.zeros(0, np.uint32))

# ---- the cells that drive more than one leg's pools -------------------------
# The sense organs' route to the motor pools is 85:1 private to the leg it
# serves (tools/coupling_probe.py measures it), so a six-legged gait cannot come
# from the organs: it has to come from cells that reach several legs' pools at
# once. There are 788 of them in BANC v888 and they carry 69,543 synapses into
# pool motor neurons — the substrate item 4 is built on. Finding them needs the
# graph and not just the neuron table, which is why this group is defined here
# rather than in motor_pools.py's predicates.
#
# The name says what it is: a *premotor* population, and it is deliberately not
# under the `pool:` prefix, because the app's pool tally (`SimulationEngine.
# poolSpikeSum`) counts exactly the `pool:` groups and this must not appear
# inside it.
LEG_OF_POOL = {}
for g in group_table:
    if g["name"].startswith("pool:"):
        LEG_OF_POOL[g["name"]] = g["name"].split(":")[1]
leg_ids = {leg: i for i, leg in enumerate(sorted(set(LEG_OF_POOL.values())))}
pool_leg = np.full(N, -1, np.int16)
for g in group_table:
    if g["name"] in LEG_OF_POOL:
        pool_leg[group_indices[g["start"]:g["start"] + g["count"]]] = \
            leg_ids[LEG_OF_POOL[g["name"]]]
post_leg = pool_leg[col_idx]
onto_pool = post_leg >= 0
if onto_pool.any():
    # One key per (cell, leg) pair, so a cell that hits the same leg twice is
    # one leg: the number this counts is the number of *legs* a cell reaches.
    key = pre_s[onto_pool].astype(np.int64) * 16 + post_leg[onto_pool]
    uniq = np.unique(key)
    legs_reached = np.bincount((uniq // 16).astype(np.int64), minlength=N)
    multi = np.flatnonzero(legs_reached >= 2).astype(np.uint32)
    group_table.append({"name": "premotor:multileg", "start": cursor,
                        "count": int(len(multi))})
    group_indices = np.concatenate([group_indices, multi])
    cursor += len(multi)
    log(f"    {'premotor:multileg':<28} {len(multi):>6,} "
        f"({int((legs_reached > 0).sum()):,} cells reach any pool)")

# ---- the same 7,033 cells, split by *which* legs they reach -----------------
# `premotor:multileg` says a cell reaches more than one leg. It does not say
# *which* ones, and that is the question item 4 now turns on
# (reports/item4_walking.md §8): the leg-selection under a patterned command is
# made in this population, so can the cells that reach one tripod be told apart
# from the cells that reach the other? The partition below makes that a count
# rather than an argument. Every cell that reaches any pool motor neuron lands in
# exactly one of nine groups:
#
#     premotor:leg:<leg>   it reaches that leg's pools and no other leg's
#     premotor:tripodA     ≥2 legs, all of them in {T1_left, T2_right, T3_left}
#     premotor:tripodB     ≥2 legs, all of them in {T1_right, T2_left, T3_right}
#     premotor:cross       at least one leg from each of the two tripods
#
# The tripod partition is the classical alternating one — the same one
# `tools/gait_probe.py` and `walk_loop.py` test with the tripod index — and it is
# a *hypothesis* about which legs move together, not a measurement. What is a
# measurement is `premotor:cross`: a cell that reaches a leg of each tripod can
# couple the two halves of a gait, and a cell that does not cannot. If a pattern
# on `premotor:tripodA` is to alternate anything, that count has to be small
# enough that the two groups are separable.
TRIPOD = {"A": ("T1_left", "T2_right", "T3_left"),
          "B": ("T1_right", "T2_left", "T3_right")}
# The six legs in `leg_ids` order, and one bit each.
LEG_ORDER = sorted(leg_ids, key=lambda k: leg_ids[k])
bit_of_id = {leg_ids[leg]: 1 << i for i, leg in enumerate(LEG_ORDER)}
mask_of_cell = np.zeros(N, np.int32)
cells_of_uniq = (uniq // 16).astype(np.int64)
legs_of_uniq = (uniq % 16).astype(np.int64)
for leg_id, bit in bit_of_id.items():
    sel = legs_of_uniq == leg_id
    mask_of_cell[cells_of_uniq[sel]] |= bit

mask_A = sum(bit_of_id[leg_ids[l]] for l in TRIPOD["A"])
mask_B = sum(bit_of_id[leg_ids[l]] for l in TRIPOD["B"])
single = {leg: np.flatnonzero(mask_of_cell == bit_of_id[leg_ids[leg]])
          for leg in LEG_ORDER}
A_only = np.flatnonzero((mask_of_cell & mask_B == 0)
                        & (mask_of_cell & mask_A != 0)
                        & (mask_of_cell & (mask_of_cell - 1) != 0))
B_only = np.flatnonzero((mask_of_cell & mask_A == 0)
                        & (mask_of_cell & mask_B != 0)
                        & (mask_of_cell & (mask_of_cell - 1) != 0))
cross = np.flatnonzero((mask_of_cell & mask_A != 0) & (mask_of_cell & mask_B != 0))

premotor_split = [("premotor:leg:" + leg, single[leg].astype(np.uint32))
                  for leg in LEG_ORDER]
premotor_split += [("premotor:tripodA", A_only.astype(np.uint32)),
                   ("premotor:tripodB", B_only.astype(np.uint32)),
                   ("premotor:cross", cross.astype(np.uint32))]
for name, idx in premotor_split:
    group_table.append({"name": name, "start": cursor, "count": int(len(idx))})
    group_indices = np.concatenate([group_indices, idx])
    cursor += len(idx)
    log(f"    {name:<28} {len(idx):>6,}")
reach_any = int((mask_of_cell > 0).sum())
split_total = sum(len(idx) for _, idx in premotor_split)
log(f"    {'premotor split total':<28} {split_total:>6,} of {reach_any:,} "
    f"cells that reach any pool")
if split_total != reach_any:
    raise SystemExit("the premotor split does not partition the cells that reach "
                     "a pool motor neuron — every such cell must land in exactly "
                     "one of the nine groups")

# ---- the same cells, split the other way: by the muscle they reach ---------
# The leg axis above is nearly common-mode (765 of the 788 multi-leg cells reach
# both tripods), and `tools/premotor_axis.py` measures what its populations
# would deliver to each tripod. The *muscle* axis is a different question with a
# different answer: the pool that reaches `long_tendon` delivers its net command
# +49% toward tripod A and the one that reaches `coxa_rotator_ant` delivers it
# -22% toward B, so a patterned command landing on those two populations does
# push the two halves apart — the lever the leg axis does not have. That is why
# these names exist: `walk_loop.py --drive-group premotor:muscle:long_tendon
# --drive-group-b premotor:muscle:coxa_rotator_ant --desc-pattern 20:1.0` is the
# experiment, and neither group can be driven until it is packed here.
#
# These groups *overlap* the nine above on purpose — a cell that reaches one leg
# and the tibia flexor pool belongs to both `premotor:leg:<leg>` and
# `premotor:muscle:tibia_flexor`. They answer different questions, and the
# partition check above is about the first of those only.
pool_muscle = np.full(N, -1, np.int16)
muscle_ids = sorted({g["name"].split(":")[2] for g in group_table
                     if g["name"].startswith("pool:")})
muscle_of = {m: i for i, m in enumerate(muscle_ids)}
for g in group_table:
    if g["name"].startswith("pool:"):
        pool_muscle[group_indices[g["start"]:g["start"] + g["count"]]] = \
            muscle_of[g["name"].split(":")[2]]
post_muscle = pool_muscle[col_idx]
muscle_groups = []
for m in muscle_ids:
    cells = np.unique(pre_s[onto_pool & (post_muscle == muscle_of[m])])
    muscle_groups.append((f"premotor:muscle:{m}", cells.astype(np.uint32)))
for name, idx in muscle_groups:
    group_table.append({"name": name, "start": cursor, "count": int(len(idx))})
    group_indices = np.concatenate([group_indices, idx])
    cursor += len(idx)
    log(f"    {name:<28} {len(idx):>6,}")
log(f"    {'premotor muscle axis':<28} "
    f"{len(muscle_groups)} groups, {len(muscle_ids)} muscles per leg")

# ---- retinotopic UV for the visual group, per hemisphere -------------------
vis = next(g for g in group_table if g["name"] == "sensory_vision")
vis_idx = group_indices[vis["start"]:vis["start"] + vis["count"]]
retina_uv = np.zeros((len(vis_idx), 2), np.float32)
side_arr = meta["side"].to_numpy()
for s in ("left", "right"):
    local = np.flatnonzero(side_arr[vis_idx] == s)
    if len(local) < 3:
        continue
    pts = Pn[vis_idx[local]]
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    uv = c @ vt[:2].T
    uv -= uv.min(0); uv /= np.maximum(uv.max(0), 1e-9)
    if s == "right":
        uv[:, 0] = 1.0 - uv[:, 0]
    retina_uv[local] = uv
    log(f"  retina {s}: {len(local):,}")
retina_uv_f16 = retina_uv.astype(np.float16)

# ============================================================== neuron meta
log("packing per-neuron metadata ...")
type_names = sorted(set(meta["cell_type"][meta["cell_type"] != ""]))
type_id = {t: i + 1 for i, t in enumerate(type_names)}
cell_type_id = meta["cell_type"].map(lambda s: type_id.get(s, 0)).to_numpy(np.uint16)

flags = np.zeros(N, np.uint8)
flags |= ((meta["super_class"] == "sensory").to_numpy().astype(np.uint8) << 0)
flags |= ((meta["super_class"] == "motor").to_numpy().astype(np.uint8) << 1)
flags |= ((meta["super_class"] == "descending").to_numpy().astype(np.uint8) << 2)
flags |= ((meta["super_class"] == "ascending").to_numpy().astype(np.uint8) << 3)
flags |= ((side_arr == "left").astype(np.uint8) << 4)
flags |= ((side_arr == "right").astype(np.uint8) << 5)
is_vnc = meta["neuromere"].notna().to_numpy().astype(np.uint8)
flags |= (is_vnc << 6)
flags |= ((~meta["has_position"].to_numpy()).astype(np.uint8) << 7)

nmeta = np.zeros((N, 8), np.uint8)
nmeta[:, 0] = meta["system_id"].to_numpy(np.uint8)
nmeta[:, 1] = meta["super_id"].to_numpy(np.uint8)
nmeta[:, 2] = meta["nt_id"].to_numpy(np.uint8)
nmeta[:, 3] = flags
nmeta[:, 4:6] = cell_type_id.view(np.uint8).reshape(-1, 2)

# =================================================================== write
log(f"writing {args.out} ...")
sections = [
    ("rootIDs",      meta["root_id"].to_numpy(np.uint64).tobytes()),
    ("positions",    positions_f16.tobytes()),
    ("neuronMeta",   nmeta.tobytes()),
    ("csrRowPtr",    row_ptr.tobytes()),
    ("csrColIdx",    col_idx.tobytes()),
    ("csrWeight",    weight.tobytes()),
    ("csrDelay",     delay_s.tobytes()),
    ("groupIndices", group_indices.tobytes()),
    ("retinaUV",     retina_uv_f16.tobytes()),
]
offsets, cursor = [], HEADER_BYTES
for _, blob in sections:
    cursor = (cursor + 255) & ~255
    offsets.append(cursor)
    cursor += len(blob)

hdr = bytearray(HEADER_BYTES)
struct.pack_into("<8sIIIIII", hdr, 0, MAGIC, VERSION, N, E,
                 len(group_indices), len(vis_idx), len(sections))
for i, ((_, blob), off) in enumerate(zip(sections, offsets)):
    struct.pack_into("<QQ", hdr, 64 + i * 16, off, len(blob))

with open(args.out, "wb") as fh:
    fh.write(hdr)
    for (name, blob), off in zip(sections, offsets):
        fh.write(b"\0" * (off - fh.tell()))
        fh.write(blob)

total = os.path.getsize(args.out)
log(f"  {total/2**20:.2f} MiB")
for (name, blob), off in zip(sections, offsets):
    log(f"    {name:<14} @{off:>10,}  {len(blob)/2**20:7.2f} MiB")

json.dump({
    "format": "flybanc.bin v2",
    "source": "BANC v888 (Bates et al. 2026) — brain AND ventral nerve cord, CC BY 4.0",
    "neurons": int(N), "edges": int(E), "synapses": int(syn.sum()),
    "minSynapses": args.min_syn,
    "systems": SYSTEMS, "systemRanges": system_ranges,
    "superClasses": SUPER_CLASSES,
    "neurotransmitters": {k: v[0] for k, v in NT.items()},
    "cellTypes": type_names,
    "groups": group_table,
    "visionGroup": vis,
    "weightRule": "sign(NT) * log1p(syn_count) / median_drive",
    "delayModel": {"speed_m_per_s": args.conduction_speed, "unit": "ms"},
    "positionNormalisation": {"extent_nm": (hi - lo).tolist(),
                              "scale_divisor_nm": float(span)},
}, open(args.meta_out, "w"), indent=2)
log(f"wrote {args.meta_out}")
log("done.")
