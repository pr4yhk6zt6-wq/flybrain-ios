#!/usr/bin/env python3
"""
STEP 2 — the leg sensorimotor circuit, taken out of the connectome itself.

This is the first place where the project asks a question of the wiring rather
than of a paper: **is the reflex arc a real fly uses to stand on its legs
actually present in BANC v888?**

The reflex in question is the one every insect leg has, the *resistance
reflex*: when the femur–tibia joint (the "knee") is forcibly bent, the
chordotonal organ inside the femur is stretched, and it excites the tibia
flexor motor neurons while inhibiting the extensor — the leg pushes back
against whatever is bending it. It is the foundation of standing, of load
support, and of the swing–stance alternation that walking is built on.

What this tool does NOT do is assume the answer. It extracts the subgraph
around one leg and measures:

  * which proprioceptors exist on that leg, by the function BANC gives them
    (`joint_angle` = chordotonal, `mechanical_strain` = campaniform, …)
  * whether they reach that leg's motor neurons at all, and in how many hops
  * whether any of those paths are monosynaptic, which is what a fast reflex
    needs
  * which premotor cell types sit on the paths, by BANC's own cell-type names
  * whether the arc stays inside the leg's own neuromere, as a local reflex
    must, or wanders through the brain

Output: `data/derived/leg_<T?>.npz` — a compact node/edge list for the
subgraph, small enough to keep, so the simulation and the body work from the
same extracted circuit.

Usage
  python3 tools/step2_legcircuit.py --leg front_leg --side left --hops 3
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

LEGS = {"front_leg": "T1", "middle_leg": "T2", "hind_leg": "T3"}

# What each of BANC's proprioceptor functions is, in the animal. These are the
# names BANC publishes in `cell_function_detailed`; the gloss is the standard
# anatomy of the organ that carries that name.
ORGAN = {
    "joint_angle":     "chordotonal organ (femoral / tibial) — reports joint angle",
    "vibro_position":  "chordotonal organ — reports position plus vibration",
    "position":        "hair plate / campaniform field — reports posture",
    "direction":       "hair plate — reports direction of joint movement",
    "mechanical_strain": "campaniform sensilla — reports load and cuticular strain",
    "vibro_tactile":   "campaniform sensilla — reports vibration and contact",
    "stretch":         "stretch receptor",
}

# Muscle name (BANC) -> the joint it moves (flybody vocabulary). Same table as
# tools/step1_anatomy.py, kept in one place per tool so each one is readable on
# its own; step 1 is the authority and this is a copy of it.
FLEXOR_MUSCLES = {"tibia_flexor_muscle", "accessory_tibia_flexor_muscle"}
EXTENSOR_MUSCLES = {"tibia_extensor_muscle"}


def load(banc: pathlib.Path, min_syn: int):
    import pandas as pd
    import pyarrow.feather as feather

    cols = ["banc_888_id", "root_888", "side", "neuromere", "region", "super_class",
            "cell_class", "cell_type", "cell_function", "cell_function_detailed",
            "body_part_sensory", "body_part_effector", "peripheral_target_type",
            "neurotransmitter_predicted", "root_position_nm"]
    meta = feather.read_table(banc / "banc_888_meta.feather", columns=cols).to_pandas()
    meta = meta.reset_index(drop=True)
    log(f"meta {len(meta):,} neurons")

    edges = pd.read_csv(banc / "connections_princeton.csv.gz",
                        dtype={"pre_root_id": "int64", "post_root_id": "int64",
                               "neuropil": "string", "syn_count": "int32",
                               "nt_type": "string"})
    log(f"edges {len(edges):,}")
    if min_syn > 1:
        edges = edges[edges.syn_count >= min_syn].reset_index(drop=True)
        log(f"  {len(edges):,} edges at >= {min_syn} synapses")

    id_to_idx = {int(v): i for i, v in enumerate(meta.root_888.fillna(0).astype("int64"))}
    pre = edges.pre_root_id.map(id_to_idx)
    post = edges.post_root_id.map(id_to_idx)
    keep = pre.notna() & post.notna()
    pre = pre[keep].to_numpy("int32")
    post = post[keep].to_numpy("int32")
    syn = edges.syn_count[keep].to_numpy("float32")
    log(f"  {len(pre):,} edges land on known neurons")
    return meta, pre, post, syn


def build_csr(n, pre, post, syn):
    order = np.argsort(pre, kind="stable")
    pre, post, syn = pre[order], post[order], syn[order]
    row = np.zeros(n + 1, dtype=np.int64)
    np.add.at(row, pre + 1, 1)
    np.cumsum(row, out=row)
    return row, post, syn


def bfs(row, col, sources, max_hops, allowed=None):
    """Breadth-first from `sources` over the connectome, up to max_hops."""
    n = len(row) - 1
    dist = np.full(n, -1, dtype=np.int8)
    parent = np.full(n, -1, dtype=np.int64)
    frontier = np.unique(sources)
    dist[frontier] = 0
    for hop in range(1, max_hops + 1):
        nxt = []
        for u in frontier:
            s, e = row[u], row[u + 1]
            targets = col[s:e]
            if allowed is not None:
                targets = targets[allowed[targets]]
            fresh = targets[dist[targets] < 0]
            if len(fresh):
                fresh = np.unique(fresh)
                dist[fresh] = hop
                parent[fresh] = u
                nxt.append(fresh)
        if not nxt:
            break
        frontier = np.concatenate(nxt)
    return dist, parent


def trace_path(parent, dist, target):
    path = [target]
    while parent[path[-1]] >= 0 and dist[path[-1]] > 0:
        path.append(int(parent[path[-1]]))
    return path[::-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banc", default="data/banc", type=pathlib.Path)
    ap.add_argument("--leg", default="front_leg", choices=list(LEGS))
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--min-syn", type=int, default=3)
    ap.add_argument("--out", default="data/derived", type=pathlib.Path)
    ap.add_argument("--json", default="reports/step2_legcircuit.json", type=pathlib.Path)
    a = ap.parse_args()

    meta, pre, post, syn = load(a.banc, a.min_syn)
    n = len(meta)
    row, col, _ = build_csr(n, pre, post, syn)

    leg, side = a.leg, a.side
    neuro = LEGS[leg]
    on_leg = meta.side.astype(str) == side

    # ---- sources: the leg's own proprioceptors ----------------------------
    proprio = (meta.cell_function.astype(str) == "proprioception")
    sensory_here = meta.body_part_sensory.astype(str).str.contains(leg, na=False) | \
                   meta.body_part_sensory.astype(str).str.contains(
                       f"coxa,{leg}", regex=False, na=False)
    src_mask = (proprio & on_leg & sensory_here).to_numpy()
    # The thoracic chordotonal organs are annotated by body part; hair plates and
    # campaniform fields sometimes only by neuromere, so take those too.
    src_mask |= (proprio & on_leg & (meta.neuromere.astype(str) == neuro)
                 & meta.body_part_sensory.notna()
                 & meta.region.astype(str).eq("ventral_nerve_cord")).to_numpy()
    sources = np.flatnonzero(src_mask)

    # ---- targets: the leg's own motor neurons, by muscle ------------------
    is_motor = (meta.super_class.astype(str) == "motor").to_numpy()
    tgt_mask = (is_motor & on_leg
                & (meta.body_part_effector.astype(str) == leg)).to_numpy()
    targets = np.flatnonzero(tgt_mask)
    flexor = targets[np.isin(meta.peripheral_target_type.astype(str).to_numpy()[targets],
                             list(FLEXOR_MUSCLES))]
    extensor = targets[np.isin(meta.peripheral_target_type.astype(str).to_numpy()[targets],
                               list(EXTENSOR_MUSCLES))]

    log(f"{leg} {side} ({neuro}): {len(sources)} proprioceptors, "
        f"{len(targets)} motor neurons ({len(flexor)} tibia flexor, "
        f"{len(extensor)} tibia extensor)")

    # ---- does the wiring connect them? ------------------------------------
    dist_all, parent_all = bfs(row, col, sources, a.hops)
    reachable = dist_all[targets] > 0
    log(f"  reachable motor neurons within {a.hops} hops: "
        f"{reachable.sum()}/{len(targets)} "
        f"(monosynaptic: {(dist_all[targets] == 1).sum()})")

    hop_hist = collections.Counter(int(d) for d in dist_all[targets] if d > 0)

    # ---- who is on the paths? ---------------------------------------------
    dist_flex = dist_all[flexor]
    dist_ext = dist_all[extensor]
    inter_counts = collections.Counter()
    for t in targets[reachable]:
        p = trace_path(parent_all, dist_all, int(t))
        for u in p[1:-1]:
            ct = meta.cell_type.iloc[u]
            sc = meta.super_class.iloc[u]
            inter_counts[f"{sc}:{ct}" if isinstance(ct, str) else f"{sc}:?"] += 1

    neuromeres = collections.Counter(str(meta.neuromere.iloc[u])
                                     for u in np.flatnonzero(dist_all > 0))

    # ---- the subgraph, kept so later tools do not re-derive it ------------
    nodes = np.flatnonzero(dist_all >= 0)
    node_of = {int(v): i for i, v in enumerate(nodes)}
    sel = np.isin(pre, nodes) & np.isin(post, nodes)
    sp, sq = pre[sel], post[sel]
    ss = syn[sel]
    a.out.mkdir(parents=True, exist_ok=True)
    outfile = a.out / f"leg_{neuro}_{side}.npz"
    np.savez_compressed(
        outfile,
        nodes=nodes.astype("int32"),
        edge_pre=np.array([node_of[int(v)] for v in sp], dtype="int32"),
        edge_post=np.array([node_of[int(v)] for v in sq], dtype="int32"),
        edge_syn=ss.astype("float32"),
        source_nodes=np.array([node_of[int(v)] for v in sources if int(v) in node_of],
                              dtype="int32"),
        flexor_nodes=np.array([node_of[int(v)] for v in flexor], dtype="int32"),
        extensor_nodes=np.array([node_of[int(v)] for v in extensor], dtype="int32"),
    )
    log(f"  subgraph {len(nodes):,} neurons, {sel.sum():,} edges -> {outfile}")

    # ---- report -----------------------------------------------------------
    R = {
        "leg": f"{neuro}_{side}",
        "proprioceptors": len(sources),
        "motor_neurons": int(len(targets)),
        "tibia_flexor_motor_neurons": int(len(flexor)),
        "tibia_extensor_motor_neurons": int(len(extensor)),
        "reachable_within_hops": int(reachable.sum()),
        "hops": a.hops,
        "monosynaptic": int((dist_all[targets] == 1).sum()),
        "hop_histogram": {str(k): int(v) for k, v in sorted(hop_hist.items())},
        "flexor_distances": {str(k): int(v) for k, v in
                             sorted(collections.Counter(
                                 int(d) for d in dist_flex if d > 0).items())},
        "extensor_distances": {str(k): int(v) for k, v in
                               sorted(collections.Counter(
                                   int(d) for d in dist_ext if d > 0).items())},
        "premotor_cell_types": inter_counts.most_common(25),
        "neuromeres_on_paths": neuromeres.most_common(),
        "subgraph_neurons": int(len(nodes)),
        "subgraph_edges": int(sel.sum()),
        "subgraph_file": str(outfile),
    }
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(R, indent=1))

    print()
    print(f"  leg {R['leg']}  proprioceptors {R['proprioceptors']}")
    print(f"  motor neurons {R['motor_neurons']} = "
          f"{R['tibia_flexor_motor_neurons']} flexor + "
          f"{R['tibia_extensor_motor_neurons']} extensor + others")
    print(f"  path lengths to motor neurons: {R['hop_histogram']}")
    print(f"  monosynaptic arcs: {R['monosynaptic']}")
    print(f"  flexor reached at {R['flexor_distances']}")
    print(f"  extensor reached at {R['extensor_distances']}")
    print(f"  top premotor cell types:")
    for k, v in inter_counts.most_common(8):
        print(f"     {v:6d}  {k}")
    print(f"  neuromeres on the paths: {neuromeres.most_common(6)}")
    print(f"  wrote {a.json} and {outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
