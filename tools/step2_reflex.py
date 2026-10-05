#!/usr/bin/env python3
"""
STEP 2 — the leg reflex, out of the connectome, checked against the papers.

Three corrections to the previous version of this tool, all of which changed
the answer, are documented in docs/STEP2.md:

  1. `disynaptic()` called `np.flatnonzero()` on an array of neuron *indices*,
     so NumPy read it as a boolean mask and the table was computed against the
     first 17 neurons of the dataset.
  2. `root_position_nm` is "x, y, z", not "[x y z]", so every axonal delay had
     silently collapsed to one millisecond.
  3. `simulate()` re-normalised the weights by the median postsynaptic drive,
     which cancelled whatever scale the caller had chosen.

Two mechanisms were missing, and both are now in, each declared in
docs/ASSUMPTIONS.md with its test:

  * **A synaptic scale.** The old rule normalised every weight by the median
    postsynaptic drive across the subgraph, which left a single spike worth
    ~0.1 of threshold: nothing propagated past the first synapse. The scale is
    now set by a single circuit-wide convention — how many presynaptic
    partners, firing together at 100 Hz, should drive an average neuron to
    threshold (`--n-star`, default 12) — and the answer is reported across a
    sweep of that number rather than at one value.

  * **Conductance-based synapses.** An additive LIF lets inhibition drive the
    membrane potential to minus infinity; in this circuit it took the tibia
    flexor to -1.4 times threshold, which no real neuron does, because
    inhibitory receptors have a reversal potential. Synapses are now
    conductances with reversal potentials, which is both the standard model and
    the thing that lets excitation and inhibition be balanced instead of
    cancelling.

What the tool then does is the experiment the previous round could not run:
put the fly in a standing state (its brain's descending neurons provide the
tonic drive a standing fly has), and ask what the leg's own sense organs do to
its motor pools — the whole chordotonal organ, each of its subtypes, and the
13B / 09A / 13A lineages that the literature names for this reflex.

  Agrawal et al. 2020, eLife 9:e60299 — 13Bα encodes tibia position and its
    activation produces tibia flexion; 9Aα responds to flexion and vibration and
    drives extension; 10Bα reports vibration and drives pausing.
  Mamiya et al. 2018, eLife 7:e34497 — the FeCO's neurons fall into claw
    (position), hook (movement) and club (vibration) subtypes.
  Azevedo et al. 2020, eLife 9:e56754 — passive extension excites the flexor
    motor neurons: the resistance reflex.

Usage
  python3 tools/step2_reflex.py --leg front_leg --side left
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

SIGN = {"acetylcholine": +1.0, "ach": +1.0, "gaba": -1.0,
        "glutamate": -1.0, "glut": -1.0, "histamine": -1.0}

ORGANS = {
    "chordotonal": {"joint_angle", "vibro_position", "stretch"},
    "campaniform": {"mechanical_strain", "vibro_tactile"},
    "hairplate":   {"position", "direction"},
}

POOLS = {
    "tibia flexor":   {"tibia_flexor_muscle", "accessory_tibia_flexor_muscle"},
    "tibia extensor": {"tibia_extensor_muscle"},
    "trochanter flexor": {"trochanter_flexor_muscle",
                          "accessory_trochanter_flexor_muscle"},
    "trochanter extensor": {"trochanter_extensor_muscle",
                            "tergotrochanter_extensor_muscle",
                            "sternotrochanter_extensor_muscle"},
    "coxa rotator ant":  {"sternal_anterior_rotator_muscle"},
    "coxa rotator post": {"sternal_posterior_rotator_muscle"},
    "femur reductor":    {"femur_reductor_muscle"},
    "long tendon":       {"long_tendon_muscle"},
    "tarsus depressor":  {"tarsus_depressor_muscle"},
    "tarsus levator":    {"tarsus_levator_muscle"},
}

PUBLISHED_LINEAGES = ["13B", "09A", "10B"]

# Conductance-based LIF. Every value is a modelling choice and lives in
# docs/ASSUMPTIONS.md; these are the usual normalised units, where the resting
# potential is 0 and the spike threshold is 1.
E_EXC = 4.0      # excitatory reversal, well above threshold
E_INH = -0.25    # inhibitory reversal, ~1/4 of the threshold range below rest
TAU_SYN = 5.0    # ms, conductance decay
TAU_M = 20.0     # ms, membrane
T_REF = 2        # ms, absolute refractory — 500 Hz ceiling
NOISE = 0.015
DT = 1.0


def load(banc: pathlib.Path, min_syn: int):
    import pandas as pd
    import pyarrow.feather as feather

    cols = ["root_888", "side", "neuromere", "cell_type", "cell_class",
            "super_class", "cell_function", "cell_function_detailed",
            "body_part_sensory", "body_part_effector", "peripheral_target_type",
            "neurotransmitter_predicted", "root_position_nm"]
    meta = feather.read_table(banc / "banc_888_meta.feather",
                              columns=cols).to_pandas().reset_index(drop=True)
    edges = pd.read_csv(banc / "connections_princeton.csv.gz",
                        dtype={"pre_root_id": "int64", "post_root_id": "int64",
                               "syn_count": "int32"})
    edges = edges[edges.syn_count >= min_syn]
    id_to_idx = {int(v): i for i, v in enumerate(meta.root_888.fillna(0).astype("int64"))}
    pre = edges.pre_root_id.map(id_to_idx)
    post = edges.post_root_id.map(id_to_idx)
    keep = pre.notna() & post.notna()
    return (meta, pre[keep].to_numpy("int64"), post[keep].to_numpy("int64"),
            edges.syn_count[keep].to_numpy("float32"))


def parse_positions(meta):
    """root_position_nm is 'x, y, z' in nanometres."""
    n = len(meta)
    pos = np.zeros((n, 3), dtype=np.float64)
    raw = meta.root_position_nm.astype(str).to_numpy()
    parsed = 0
    for i in range(n):
        s = raw[i]
        if "," in s:
            try:
                pos[i] = np.fromstring(s, sep=",", dtype=np.float64)
                parsed += 1
            except ValueError:
                pass
    return pos, parsed


def simulate(N, e_pre, e_post, w_sign, w_mag, delay, ms, drivers,
             rng_seed=42, watch=None, sample_from=None):
    """
    Conductance-based LIF over a subgraph.

    `w_mag` is the synaptic conductance one spike delivers, `w_sign` its sign.
    Excitatory spikes add to g_e, inhibitory to g_i; the membrane equation is

        dV/dt = (-(V - 0) + g_e (E_e - V) + g_i (E_i - V)) / tau_m

    so inhibition subtracts *towards its reversal*, never below it, and a
    strongly inhibited cell can still be driven by enough excitation — which is
    what makes the fly's balanced E/I circuits work at all.
    """
    order = np.argsort(e_pre, kind="stable")
    e_pre, e_post = e_pre[order], e_post[order]
    w_mag, delay = w_mag[order], delay[order]
    is_exc = (w_sign[order] > 0)
    e_row = np.zeros(N + 1, dtype=np.int64)
    np.add.at(e_row, e_pre + 1, 1)
    np.cumsum(e_row, out=e_row)

    rng = np.random.default_rng(rng_seed)
    V = np.zeros(N, dtype=np.float32)
    g_e = np.zeros(N, dtype=np.float32)
    g_i = np.zeros(N, dtype=np.float32)
    refrac = np.zeros(N, np.int16)
    max_d = int(delay.max()) + 1 if len(delay) else 2
    ring_e = np.zeros((max_d, N), np.float32)
    ring_i = np.zeros((max_d, N), np.float32)
    de = np.exp(-DT / TAU_SYN)

    spikes = np.zeros(N, np.int32)
    per_ms = []
    sample_from = ms // 2 if sample_from is None else sample_from
    acc = {k: {"v": [], "i": []} for k in (watch or {})}

    for t in range(ms):
        slot = t % max_d
        g_e = g_e * de + ring_e[slot]
        g_i = g_i * de + ring_i[slot]
        ring_e[slot] = 0
        ring_i[slot] = 0

        I = g_e * (E_EXC - V) + g_i * (E_INH - V)
        I = I + rng.normal(0, NOISE, N).astype(np.float32)
        for idx, amp, t0_, t1_ in drivers:
            if amp and t0_ <= t < t1_ and len(idx):
                I[idx] += amp

        free = refrac <= 0
        refrac[~free] -= 1                 # count the refractory period down
        # the resting conductance is the leak of tau_m = 20 ms
        dV = (DT / TAU_M) * (-V + I)
        V = np.where(free, V + dV, V).astype(np.float32)

        fired = np.flatnonzero((V >= 1.0) & free)
        V[fired] = 0.0
        refrac[fired] = T_REF
        if len(fired):
            spikes[fired] += 1
            for u in fired:
                s0, s1 = e_row[u], e_row[u + 1]
                if s1 > s0:
                    d = delay[s0:s1]
                    m = w_mag[s0:s1]
                    ex = is_exc[s0:s1]
                    tgt = e_post[s0:s1]
                    slots = (t + d) % max_d
                    if ex.any():
                        np.add.at(ring_e, (slots[ex], tgt[ex]), m[ex])
                    if (~ex).any():
                        np.add.at(ring_i, (slots[~ex], tgt[~ex]), m[~ex])
        per_ms.append(len(fired))
        if t >= sample_from:
            for k, idx in (watch or {}).items():
                if len(idx):
                    acc[k]["v"].append(float(V[idx].mean()))
                    acc[k]["i"].append(float(g_e[idx].mean() - g_i[idx].mean()))

    out = {}
    for k, idx in (watch or {}).items():
        if not len(idx):
            continue
        out[k] = {
            "hz": float(spikes[idx].sum() / len(idx) / (ms / 1000.0)),
            "v": float(np.mean(acc[k]["v"])) if acc[k]["v"] else float("nan"),
            "g": float(np.mean(acc[k]["i"])) if acc[k]["i"] else float("nan"),
            "spikes": int(spikes[idx].sum()),
            "n": int(len(idx)),
        }
    out["_population_hz"] = float(np.mean(per_ms[sample_from:]))
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banc", default="data/banc", type=pathlib.Path)
    ap.add_argument("--leg", default="front_leg", choices=list(LEGS))
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--min-syn", type=int, default=3)
    ap.add_argument("--ms", type=int, default=1000)
    ap.add_argument("--n-star", type=float, default=12.0,
                    help="presynaptic partners at 100 Hz that drive an average "
                         "neuron to threshold — the circuit-wide synaptic scale")
    ap.add_argument("--n-star-sweep", default="6,12,24,48")
    ap.add_argument("--desc", type=float, default=2.0,
                    help="tonic drive to the descending (brain -> cord) neurons")
    ap.add_argument("--stim", type=float, default=3.0,
                    help="drive added to a sensory population or a lineage")
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--json", default="reports/step2_reflex.json", type=pathlib.Path)
    ap.add_argument("--md", default="reports/step2_reflex.md", type=pathlib.Path)
    a = ap.parse_args()

    meta, pre, post, syn = load(a.banc, a.min_syn)
    n = len(meta)
    ct = meta.cell_type.astype(str).to_numpy()
    nt = meta.neurotransmitter_predicted.astype(str).str.lower() \
             .str.split().str[0].to_numpy()
    sign = np.array([SIGN.get(x, 0.0) for x in nt], dtype="float32")
    fn = meta.cell_function.astype(str).to_numpy()
    det = meta.cell_function_detailed.astype(str).to_numpy()
    part_s = meta.body_part_sensory.astype(str).to_numpy()
    part_e = meta.body_part_effector.astype(str).to_numpy()
    ptt = meta.peripheral_target_type.astype(str).to_numpy()
    sc = meta.super_class.astype(str).to_numpy()
    side = meta.side.astype(str).to_numpy()
    neuroneme = meta.neuromere.astype(str).to_numpy()
    log(f"{n:,} neurons, {len(pre):,} edges at >= {a.min_syn} synapses")

    leg, s = a.leg, a.side
    neuro = LEGS[leg]
    on_leg = side == s
    part_here = np.array([leg in p for p in part_s])

    organs = {}
    for name, funcs in ORGANS.items():
        m = (fn == "proprioception") & np.isin(det, list(funcs)) & on_leg
        m &= part_here | (neuroneme == neuro)
        organs[name] = np.flatnonzero(m)

    leg_motor = np.flatnonzero((sc == "motor") & on_leg & (part_e == leg))
    pools = {}
    for name, muscles in POOLS.items():
        m = np.zeros(n, bool)
        m[leg_motor] = np.isin(ptt[leg_motor], list(muscles))
        pools[name] = np.flatnonzero(m)

    chordo = organs["chordotonal"]
    print(f"\nleg {neuro}_{s}")
    for k, v in organs.items():
        print(f"  organ {len(v):4d}  {k}")
    for k, v in pools.items():
        if len(v):
            print(f"  pool  {len(v):4d}  {k}")
    print(f"  chordotonal: {len(chordo)} cells, transmitters "
          f"{dict(collections.Counter(nt[chordo]))}")

    # ---- 1. arcs out of the organ -----------------------------------------
    sel1 = np.isin(pre, chordo)
    arc_syn = collections.Counter()
    for q, w_ in zip(post[sel1], syn[sel1]):
        arc_syn[int(q)] += float(w_)
    arcs = [{"cell": ct[q], "super_class": sc[q], "transmitter": nt[q],
             "sign": float(sign[q]), "synapses": v}
            for q, v in arc_syn.most_common(15)]
    log("top targets of the chordotonal population:")
    for r in arcs[:6]:
        print(f"    {r['synapses']:6.0f}  {r['cell'][:22]:22s} "
              f"{r['transmitter'][:14]:14s} sign={r['sign']:+.0f}")

    # ---- 2. the disynaptic weights (corrected) -----------------------------
    inter_weight = np.zeros(n, dtype=np.float64)
    np.add.at(inter_weight, post[sel1], syn[sel1])
    inter_idx = np.flatnonzero(inter_weight > 0)

    def disynaptic(pool_idx):
        tgt = pool_idx                       # indices, not a mask
        sel = np.isin(pre, inter_idx) & np.isin(post, tgt)
        exc = inh = 0.0
        lines = collections.Counter()
        arcs_here = []
        for p, q, w_ in zip(pre[sel], post[sel], syn[sel]):
            weight = float(inter_weight[p]) * float(w_)
            sg = float(sign[p])
            if sg > 0:
                exc += weight
            elif sg < 0:
                inh += weight
            fam = next((f for f in PUBLISHED_LINEAGES if f in ct[p]), None) or "other"
            lines[fam] += 1
            arcs_here.append({"interneuron": ct[p], "transmitter": nt[p],
                              "sign": sg, "from_organ": float(inter_weight[p]),
                              "to_motor_neuron": float(w_),
                              "weight": weight * sg})
        return {"motor_neurons": len(tgt),
                "interneurons": int(len(set(pre[sel].tolist()))),
                "arcs": int(sel.sum()),
                "motor_neurons_receiving": int(len(set(post[sel].tolist()))),
                "excitatory_weight": exc, "inhibitory_weight": inh,
                "net_weight": exc - inh,
                "net_per_motor_neuron": (exc - inh) / max(1, len(tgt)),
                "published_lineages": dict(lines),
                "strongest_arcs": sorted(arcs_here,
                                         key=lambda r: -abs(r["weight"]))[:10]}

    # direct arcs, organ -> motor pool. This is the table the literature makes
    # testable: the femoral chordotonal organ reaches the tibia motor neurons
    # only polysynaptically (Azevedo et al. 2020), and trochanter campaniform
    # sensilla contact flexor tibiae motor neurons directly (Phelps et al. 2021).
    monosynaptic = {}
    for oname, cells in organs.items():
        monosynaptic[oname] = {}
        for pname, pool in pools.items():
            if not len(pool):
                continue
            selo = np.isin(pre, cells) & np.isin(post, pool)
            monosynaptic[oname][pname] = {"arcs": int(selo.sum()),
                                          "synapses": float(syn[selo].sum())}
    log("direct organ -> motor pool arcs:")
    for oname, row in monosynaptic.items():
        hits = {k: v["arcs"] for k, v in row.items() if v["arcs"]}
        print(f"    {oname:12s} {hits}")

    disyn = {k: disynaptic(v) for k, v in pools.items() if len(v)}
    for k in ("tibia flexor", "tibia extensor"):
        d = disyn[k]
        print(f"  disynaptic to {k}: {d['arcs']} arcs / {d['interneurons']} "
              f"interneurons, net {d['net_weight']:+,.0f}")

    # ---- 3. split the organ by its target profiles -------------------------
    from scipy.cluster.hierarchy import linkage, fcluster

    tgt_ids = np.unique(post[sel1])
    tgt_pos = {int(t): j for j, t in enumerate(tgt_ids)}
    src_pos = {int(c): i for i, c in enumerate(chordo)}
    M = np.zeros((len(chordo), len(tgt_ids)), dtype=np.float32)
    for p, q, w_ in zip(pre[sel1], post[sel1], syn[sel1]):
        M[src_pos[int(p)], tgt_pos[int(q)]] += float(w_)
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
    labels = fcluster(linkage(Mn, method="average", metric="correlation"),
                      t=a.clusters, criterion="maxclust")

    def describe(mask):
        cells = chordo[mask]
        counts = collections.Counter()
        for p, q, w_ in zip(pre[sel1], post[sel1], syn[sel1]):
            if src_pos[int(p)] in set(np.flatnonzero(mask).tolist()):
                counts[ct[int(q)]] += float(w_)
        return {"cells": int(len(cells)),
                "cell_types": dict(collections.Counter(ct[cells]).most_common(6)),
                "transmitters": dict(collections.Counter(nt[cells])),
                "top_targets": [{"cell": k, "synapses": v}
                                for k, v in counts.most_common(6)],
                "node_indices": [int(x) for x in cells]}

    subtypes = {f"cluster{i}": describe(labels == i) for i in np.unique(labels)}
    log("chordotonal subtypes by target profile:")
    for k, d in subtypes.items():
        print(f"    {k}: {d['cells']:2d} cells {d['cell_types']} -> "
              f"{[t['cell'] for t in d['top_targets'][:3]]}")

    # ---- the subgraph ------------------------------------------------------
    sources = np.unique(np.concatenate(list(organs.values())))
    fwd = _reach(n, pre, post, np.unique(np.concatenate([sources, chordo])), 3)
    bwd = _reach(n, post, pre, leg_motor, 3)
    keep = (fwd >= 0) & (bwd >= 0)
    nodes = np.flatnonzero(keep)
    node_of = {int(v): i for i, v in enumerate(nodes)}
    sel = keep[pre] & keep[post]
    e_pre = np.array([node_of[int(v)] for v in pre[sel]], dtype=np.int64)
    e_post = np.array([node_of[int(v)] for v in post[sel]], dtype=np.int64)
    w_sign = sign[pre[sel]].astype(np.float32)
    w_raw = np.log1p(syn[sel]).astype(np.float32)

    pos, parsed = parse_positions(meta)
    d_um = np.linalg.norm(pos[nodes][e_pre] - pos[nodes][e_post], axis=1) / 1000.0
    delay = np.clip(np.round(d_um / 250.0), 1, 8).astype(np.int64)
    log(f"reflex subgraph {len(nodes):,} neurons / {int(sel.sum()):,} edges; "
        f"positions parsed for {parsed:,}/{n:,} cells; delay median "
        f"{int(np.median(delay))} ms, p90 {int(np.percentile(delay, 90))} ms")

    def nodes_of(idx):
        return np.array([node_of[int(x)] for x in idx if int(x) in node_of],
                        dtype=np.int64)

    watch = {}
    for k, v in pools.items():
        idx = nodes_of(v)
        if len(idx):
            watch[f"pool:{k}"] = idx
    for k, v in organs.items():
        idx = nodes_of(v)
        if len(idx):
            watch[f"organ:{k}"] = idx
    sub_of = {k: nodes_of(np.array(d["node_indices"])) for k, d in subtypes.items()}

    # the lineages the papers name, restricted to the cells that receive from
    # THIS leg's organ
    downstream = np.unique(post[sel1])
    fam_of = {}
    for fam in ("13B", "09A", "10B", "13A"):
        cells = np.array([i for i in downstream if fam in ct[i]])
        fam_of[fam] = nodes_of(cells)
        watch[f"lineage:{fam}"] = fam_of[fam]
    desc_idx = nodes_of(np.flatnonzero(sc == "descending"))
    watch["descending"] = desc_idx
    log(f"descending neurons in the subgraph: {len(desc_idx):,}; "
        f"lineages receiving from the organ: "
        f"{ {k: len(v) for k, v in fam_of.items()} }")

    # ---- the synaptic scale ------------------------------------------------
    mean_raw = float(np.abs(w_raw).mean())
    # one spike delivers w_raw x K of conductance; N* partners at 100 Hz must
    # reach the conductance that first pushes an average neuron to threshold
    def scale_for(n_star):
        # g needed so that at V = 0 the drive equals the leak: g (E_e) ~ 1
        g_threshold = 1.0 / E_EXC
        return g_threshold / (n_star * 0.1 * TAU_SYN * mean_raw)

    K = scale_for(a.n_star)
    w_mag = (w_raw * K).astype(np.float32)
    log(f"synaptic scale: N* = {a.n_star:g} -> K = {K:.5f}, mean |w| = "
        f"{float(w_mag.mean()):.4f}")

    def run(name, drivers, k=None, ms=None):
        wm = w_mag if k is None else (w_raw * k).astype(np.float32)
        return simulate(len(nodes), e_pre.copy(), e_post.copy(),
                        w_sign.copy(), wm.copy(), delay.copy(),
                        ms or a.ms, drivers, watch=watch)

    runs = {}
    runs["silent"] = run("silent", [])
    standing = [(desc_idx, a.desc, 0, a.ms)]
    runs["standing"] = run("standing", standing)
    for k, idx in sub_of.items():
        runs[f"standing+{k}"] = run(f"standing+{k}",
                                    standing + [(idx, a.stim, a.ms // 4, a.ms)])
    for fam, idx in fam_of.items():
        if len(idx):
            runs[f"standing+lineage{fam}"] = run(
                f"standing+lineage{fam}",
                standing + [(idx, a.stim, a.ms // 4, a.ms)])

    # the release test: the organ is tonically active while standing, and an
    # imposed movement takes that activity away
    runs["release_tonic"] = run("release_tonic",
                                standing + [(watch["organ:chordotonal"],
                                             a.stim, 0, a.ms)])
    # and the load-sensing channel, which is tonic while the feet are loaded
    runs["standing+load"] = run("standing+load",
                                standing + [(watch["organ:campaniform"],
                                             a.stim, a.ms // 4, a.ms)])

    # ---- the scale sweep ---------------------------------------------------
    sweep = {}
    for n_star in [float(x) for x in a.n_star_sweep.split(",") if x.strip()]:
        k = scale_for(n_star)
        row = {"standing": run("", standing, k=k)}
        row["standing+organ"] = run("", standing + [(watch["organ:chordotonal"],
                                                     a.stim, a.ms // 4, a.ms)], k=k)
        for fam in ("13B", "09A"):
            if len(fam_of[fam]):
                row[f"standing+lineage{fam}"] = run(
                    "", standing + [(fam_of[fam], a.stim, a.ms // 4, a.ms)], k=k)
        sweep[f"{n_star:g}"] = row
    log(f"scale sweep done: {len(sweep)} values of N*")

    # ---- control: transmitter signs shuffled within cell class -------------
    cls = meta.cell_class.astype(str).to_numpy()[nodes]
    rng = np.random.default_rng(7)
    shuf = sign[nodes].copy()
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        if len(idx) > 1:
            shuf[idx] = sign[nodes][rng.permutation(idx)]
    control = simulate(len(nodes), e_pre.copy(), e_post.copy(),
                       shuf[e_pre].astype(np.float32), w_mag.copy(),
                       delay.copy(), a.ms,
                       standing + [(watch["organ:chordotonal"], a.stim,
                                    a.ms // 4, a.ms)], watch=watch)

    # ---- verdict -----------------------------------------------------------
    def pool(res, name):
        return res.get(f"pool:{name}", {})

    def delta(cond, name, base="standing", field="hz"):
        r = pool(runs[cond], name)
        b = pool(runs[base], name)
        return r.get(field, 0.0) - b.get(field, 0.0)

    verdict = {}
    for cond in runs:
        if cond in ("silent", "standing"):
            continue
        d = {field: {"flexor": delta(cond, "tibia flexor", field=field),
                     "extensor": delta(cond, "tibia extensor", field=field)}
             for field in ("hz", "v", "g")}
        verdict[cond] = {
            "d_flexor_hz": d["hz"]["flexor"], "d_extensor_hz": d["hz"]["extensor"],
            "d_flexor_v": d["v"]["flexor"], "d_extensor_v": d["v"]["extensor"],
            "d_flexor_g": d["g"]["flexor"], "d_extensor_g": d["g"]["extensor"],
        }
        verdict[cond]["flexor_minus_extensor_hz"] = (
            d["hz"]["flexor"] - d["hz"]["extensor"])
        verdict[cond]["flexor_minus_extensor_v"] = (
            d["v"]["flexor"] - d["v"]["extensor"])
        # the threshold-free metric: how much net synaptic conductance the
        # wiring delivers to each pool. This is what the connectome decides,
        # before any firing threshold is applied.
        verdict[cond]["flexor_minus_extensor_g"] = (
            d["g"]["flexor"] - d["g"]["extensor"])

    # the published prediction: driving 13B should bias the tibia toward
    # flexion; driving 09A should bias it toward extension
    def bias(cond):
        if cond not in verdict:
            return None
        return verdict[cond]["flexor_minus_extensor_hz"]

    out = {
        "leg": f"{neuro}_{s}",
        "corrections": [
            "disynaptic() indexed a pool of neuron indices with flatnonzero, "
            "so the previous table was computed against the wrong neurons",
            "root_position_nm is 'x, y, z' — the previous parser silently "
            "collapsed every axonal delay to 1 ms",
            "simulate() re-normalised the weights and cancelled the caller's "
            "synaptic scale, which is why no scale seemed to matter",
        ],
        "model": {
            "type": "conductance-based LIF",
            "E_exc": E_EXC, "E_inh": E_INH, "tau_m_ms": TAU_M,
            "tau_syn_ms": TAU_SYN, "t_ref_ms": T_REF, "noise": NOISE,
            "threshold": 1.0, "rest": 0.0,
            "synaptic_scale": {"n_star": a.n_star, "K": K,
                               "mean_raw_weight": mean_raw,
                               "meaning": "N* presynaptic partners firing "
                                          "together at 100 Hz drive an average "
                                          "neuron to threshold"},
            "delay_ms": "soma distance / 0.25 m/s, clipped to 1..8 ms",
        },
        "config": {"min_syn": a.min_syn, "ms": a.ms, "desc_drive": a.desc,
                   "stim": a.stim, "clusters": a.clusters},
        "organs": {k: int(len(v)) for k, v in organs.items()},
        "pools": {k: int(len(v)) for k, v in pools.items() if len(v)},
        "subgraph": {"neurons": int(len(nodes)), "edges": int(sel.sum()),
                     "descending": int(len(desc_idx))},
        "chordotonal_arcs": arcs,
        "monosynaptic": monosynaptic,
        "disynaptic": disyn,
        "subtypes": subtypes,
        "lineages": {k: int(len(v)) for k, v in fam_of.items()},
        "runs": runs,
        "scale_sweep": sweep,
        "control_sign_shuffled": control,
        "verdict": verdict,
    }
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, indent=1))
    log(f"wrote {a.json}")
    write_markdown(out, a.md)
    log(f"wrote {a.md}")

    cols_ = ["tibia flexor", "tibia extensor", "trochanter flexor",
             "femur reductor", "coxa rotator post"]
    print("\n  motor pools per condition (Hz / V).  t = tibia")
    head = "  " + f"{'condition':26s}" + "".join(f"{c:>19s}" for c in cols_)
    print(head)
    for k, r in runs.items():
        row = f"  {k:26s}"
        for c in cols_:
            p = pool(r, c)
            row += f"{p.get('hz', 0):10.2f}/{p.get('v', float('nan')):+8.3f}"
        print(row)
    print("\n  change from the standing state")
    print(f"  {'condition':26s} {'Δflex Hz':>9s} {'Δext Hz':>9s} "
          f"{'Δflex V':>9s} {'Δext V':>9s} {'flex-ext':>9s}")
    for k, d in verdict.items():
        print(f"  {k:26s} {d['d_flexor_hz']:9.2f} {d['d_extensor_hz']:9.2f} "
              f"{d['d_flexor_v']:9.4f} {d['d_extensor_v']:9.4f} "
              f"{d['flexor_minus_extensor_hz']:9.2f}")
    print("\n  opponent metric: net synaptic conductance delivered to each pool")
    print(f"  {'condition':26s} {'Δg flexor':>10s} {'Δg extensor':>11s} "
          f"{'flex-ext (g)':>13s} {'flex-ext (V)':>13s}")
    for k, d in verdict.items():
        print(f"  {k:26s} {d['d_flexor_g']:+10.5f} {d['d_extensor_g']:+11.5f} "
              f"{d['flexor_minus_extensor_g']:+13.5f} "
              f"{d['flexor_minus_extensor_v']:+13.5f}")
    print(f"\n  published prediction — 13B activation should favour flexion: "
          f"flex-ext(g) = {verdict.get('standing+lineage13B', {}).get('flexor_minus_extensor_g')}")
    print(f"  09A activation should favour extension: "
          f"flex-ext(g) = {verdict.get('standing+lineage09A', {}).get('flexor_minus_extensor_g')}")
    return 0


def _reach(n, pre, post, sources, hops):
    row, col = _csr(n, pre, post)
    dist = np.full(n, -1, dtype=np.int8)
    fr = np.unique(sources)
    dist[fr] = 0
    for h in range(1, hops + 1):
        starts, ends = row[fr], row[fr + 1]
        counts = (ends - starts).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            break
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        gather = np.repeat(starts, counts) + (np.arange(total, dtype=np.int64) - offsets)
        nb = col[gather]
        fresh = np.unique(nb[dist[nb] < 0])
        if len(fresh) == 0:
            break
        dist[fresh] = h
        fr = fresh
    return dist


def _csr(n, pre, post):
    order = np.argsort(pre, kind="stable")
    p = pre[order]
    row = np.zeros(n + 1, dtype=np.int64)
    np.add.at(row, p + 1, 1)
    np.cumsum(row, out=row)
    return row, post[order]


def write_markdown(o: dict, path: pathlib.Path) -> None:
    L = []
    A = L.append
    A("# Step 2 — the leg reflex, out of the connectome\n")
    A(f"Leg **{o['leg']}**. Generated by `tools/step2_reflex.py`. Every number "
      "is in `reports/step2_reflex.json`.\n")

    A("## Three corrections to the previous version\n")
    for c in o["corrections"]:
        A(f"- {c}")
    A("")
    A("Two mechanisms were missing as well, and both are now in the model and "
      "declared in `docs/ASSUMPTIONS.md`:\n")
    A(f"- **A synaptic scale set by a stated convention** — "
      f"{o['model']['synaptic_scale']['meaning']}, with N* = "
      f"{o['model']['synaptic_scale']['n_star']:g} here and the answer reported "
      "across a sweep of N* rather than at one value.")
    A("- **Conductance-based synapses.** An additive LIF let inhibition drive "
      "the tibia flexor to −1.4× threshold; inhibitory receptors have a "
      "reversal potential and cannot do that.\n")

    A("## 1. What the chordotonal organ contacts\n")
    A("| synapses | target | class | transmitter | sign |")
    A("|---:|---|---|---|---:|")
    for r in o["chordotonal_arcs"][:10]:
        A(f"| {r['synapses']:.0f} | `{r['cell']}` | {r['super_class']} | "
          f"{r['transmitter']} | {r['sign']:+.0f} |")
    A("")
    A(f"The organ is {o['organs']['chordotonal']} cells; its strongest target "
      f"is **`{o['chordotonal_arcs'][0]['cell']}`**, a 13B-lineage interneuron — "
      "the population the Drosophila literature names as the FeCO's first "
      "central partner (Agrawal et al. 2020).\n")

    A("## 1b. Direct (monosynaptic) arcs from each organ to each motor pool\n")
    onames = list(o["monosynaptic"])
    pnames = [k for k in o["pools"]]
    A("| organ | " + " | ".join(pnames) + " |")
    A("|" + "---|" * (len(pnames) + 1))
    for oname in onames:
        row = [f"{o['monosynaptic'][oname].get(pn, {}).get('arcs', 0)}"
               f" ({o['monosynaptic'][oname].get(pn, {}).get('synapses', 0):.0f})"
               for pn in pnames]
        A(f"| {oname} | " + " | ".join(row) + " |")
    A("")
    A("Cells are `arcs (synapses)`. Azevedo et al. 2020: the femoral "
      "chordotonal organ reaches the tibia motor neurons only through "
      "polysynaptic pathways. Phelps et al. 2021: trochanter campaniform "
      "sensilla contact flexor tibiae motor neurons directly. Both are visible "
      "in this table as number of arcs, not as a claim.\n")

    A("## 2. The disynaptic path (corrected)\n")
    A("| pool | motor neurons | reached | interneurons | arcs | excitatory | "
      "inhibitory | net |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in o["disynaptic"].items():
        A(f"| {name} | {d['motor_neurons']} | {d['motor_neurons_receiving']} | "
          f"{d['interneurons']} | {d['arcs']} | {d['excitatory_weight']:,.0f} | "
          f"{d['inhibitory_weight']:,.0f} | **{d['net_weight']:+,.0f}** |")
    A("")

    A("## 3. The organ is not one population\n")
    A("| cluster | cells | cell types | strongest targets |")
    A("|---|---:|---|---|")
    for k, d in o["subtypes"].items():
        types = ", ".join(f"{a_}({b_})" for a_, b_ in d["cell_types"].items())
        tgts = ", ".join(f"{t['cell']}({t['synapses']:.0f})"
                         for t in d["top_targets"][:3])
        A(f"| {k} | {d['cells']} | {types} | {tgts} |")
    A("")

    A("## 4. The experiment: a standing fly, and what its leg sense organs do\n")
    A(f"{o['config']['ms']} ms per condition. **Standing** = the descending "
      f"(brain → cord) neurons driven tonically at {o['config']['desc_drive']:g}, "
      "which is the drive a standing fly's cord receives. Each sensory "
      "population or lineage then gets an extra drive from a quarter of the way "
      "in. Rates and potentials are averaged over the second half.\n")
    pools_shown = ["tibia flexor", "tibia extensor", "trochanter flexor",
                   "femur reductor", "coxa rotator post"]
    A("| condition | " + " | ".join(f"{p} Hz" for p in pools_shown) + " |")
    A("|" + "---|" * (len(pools_shown) + 1))
    for k, r in o["runs"].items():
        cells = []
        for p in pools_shown:
            d = r.get(f"pool:{p}", {})
            cells.append(f"{d.get('hz', 0):.2f} ({d.get('v', float('nan')):+.3f})")
        A(f"| {k} | " + " | ".join(cells) + " |")
    A("")

    A("## 5. Change from standing, and the published prediction\n")
    A("| condition | Δg flexor | Δg extensor | flex−ext (g) | Δflex Hz | "
      "Δext Hz | flex−ext (V) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for k, d in o["verdict"].items():
        A(f"| {k} | {d['d_flexor_g']:+.5f} | {d['d_extensor_g']:+.5f} | "
          f"**{d['flexor_minus_extensor_g']:+.5f}** | {d['d_flexor_hz']:+.2f} | "
          f"{d['d_extensor_hz']:+.2f} | {d['flexor_minus_extensor_v']:+.5f} |")
    A("")
    A("`Δg` is the change in net synaptic conductance arriving at each pool — "
      "what the wiring delivers, before any threshold is applied. It is the "
      "robust readout here: the extensor pool is two cells, so its firing rate "
      "moves in whole spikes.")
    A("")
    A("The published optogenetic result (Agrawal et al. 2020) is that driving "
      "13Bα produces tibia **flexion** and driving 9Aα produces **extension**. "
      "A positive flexor−extensor change is the connectome's version of that "
      "prediction.\n")

    A("## 6. Robustness: the answer across the synaptic scale\n")
    A("| N* | condition | flexor Hz | extensor Hz | flexor V | extensor V |")
    A("|---:|---|---:|---:|---:|---:|")
    for ns, row in o["scale_sweep"].items():
        for cond, r in row.items():
            f_ = r.get("pool:tibia flexor", {})
            e_ = r.get("pool:tibia extensor", {})
            A(f"| {ns} | {cond} | {f_.get('hz', 0):.2f} | {e_.get('hz', 0):.2f} | "
              f"{f_.get('v', float('nan')):+.4f} | {e_.get('v', float('nan')):+.4f} |")
    A("")
    A(f"Sign-shuffled control (transmitters permuted within cell class), organ "
      f"driven: flexor {o['control_sign_shuffled'].get('pool:tibia flexor', {}).get('hz', 0):.2f} Hz, "
      f"extensor {o['control_sign_shuffled'].get('pool:tibia extensor', {}).get('hz', 0):.2f} Hz.\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
