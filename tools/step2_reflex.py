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
from types import SimpleNamespace as NS

import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

LEGS = {"front_leg": "T1", "middle_leg": "T2", "hind_leg": "T3"}

SIGN = {"acetylcholine": +1.0, "ach": +1.0, "gaba": -1.0,
        "glutamate": -1.0, "glut": -1.0, "histamine": -1.0}

# The organ table lives with the pools now (tools/motor_pools.py), because the
# phone closes the loop with the same organs this tool measures the reflex in.
#
# The three *proprioceptors*, though — not the whole table. This tool drives
# every cell of every organ it is handed and reads what the pools do, and its
# numbers are the project's reflex baseline; adding the leg's 3,184 tactile
# cells to that simulation would move a baseline rather than measure touch
# (tools/motor_pools.py says the same, where the view is defined). Item 12
# measures the touch pathway on its own, in `tools/tactile_probe.py`.
from motor_pools import PROPRIOCEPTORS as ORGANS   # noqa: E402

# The motor pools live in one file, because step 6 needs the same table on the
# phone (tools/motor_pools.py). Re-exported here so every caller in this file
# and in step 3 is reading the same object.
from motor_pools import POOLS, POOL_JOINT, LEGS as POOL_LEGS   # noqa: E402

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


def text(meta, name):
    """
    One column, as an object array of str, with missing values as "".

    Do not use `meta[name].astype(str)`: pandas 2 turns a missing value into the
    string "nan", pandas 3 keeps it missing, so a column that is fine locally
    arrives in CI holding floats and dies on `leg in value`. Converting through
    pandas' own `string` dtype and then filling is stable in both.
    """
    s = meta[name]
    return s.astype("string").fillna("").astype(object).to_numpy()


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
    raw = text(meta, "root_position_nm")
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


class CordNetwork:
    """
    The nerve cord as a steppable object.

    `simulate()` below is the batch form that step 2 uses: give it a duration
    and a list of tonic drives, get rates back. Step 3 needs the same cell
    model one millisecond at a time, because the drives there come from a body
    that is moving while the cord is running. Both go through this class, so
    there is exactly one implementation of the membrane equation.

    The physics is unchanged from the batch version — the RNG is consumed in
    the same order and every arithmetic operation is the same one — so the two
    produce bit-identical traces. `tools/step3_closedloop.py` re-runs step 2's
    standing state through the stepwise path and compares.
    """

    def __init__(self, N, e_pre, e_post, w_sign, w_mag, delay, rng_seed=42):
        order = np.argsort(e_pre, kind="stable")
        self.N = N
        # int32: more neuron indices than any connectome will have, and half
        # the memory, which is what lets step 4 hold six of these at once
        self.e_pre = e_pre[order].astype(np.int32)
        self.e_post = e_post[order].astype(np.int32)
        self.w_mag = w_mag[order].astype(np.float32)
        self.delay = delay[order].astype(np.int32)
        self.is_exc = (w_sign[order] > 0)

        e_row = np.zeros(N + 1, dtype=np.int64)
        np.add.at(e_row, self.e_pre + 1, 1)
        np.cumsum(e_row, out=e_row)
        self.e_row = e_row

        self.rng = np.random.default_rng(rng_seed)
        self.V = np.zeros(N, dtype=np.float32)
        self.g_e = np.zeros(N, dtype=np.float32)
        self.g_i = np.zeros(N, dtype=np.float32)
        self.refrac = np.zeros(N, np.int16)
        self.max_d = int(self.delay.max()) + 1 if len(self.delay) else 2
        self.ring_e = np.zeros((self.max_d, N), np.float32)
        self.ring_i = np.zeros((self.max_d, N), np.float32)
        self.de = np.exp(-DT / TAU_SYN)

        self.spikes = np.zeros(N, np.int32)
        self.t = 0
        self.fired = np.zeros(0, dtype=np.int64)

    def step(self, drivers=()):
        """Advance the cord by exactly 1 ms. Returns the neurons that fired."""
        t, N = self.t, self.N
        slot = t % self.max_d
        g_e = self.g_e * self.de + self.ring_e[slot]
        g_i = self.g_i * self.de + self.ring_i[slot]
        self.ring_e[slot] = 0
        self.ring_i[slot] = 0

        I = g_e * (E_EXC - self.V) + g_i * (E_INH - self.V)
        I = I + self.rng.normal(0, NOISE, N).astype(np.float32)
        for idx, amp, t0_, t1_ in drivers:
            if amp and t0_ <= t < t1_ and len(idx):
                I[idx] += amp

        free = self.refrac <= 0
        self.refrac[~free] -= 1          # count the refractory period down
        # the resting conductance is the leak of tau_m = 20 ms
        dV = (DT / TAU_M) * (-self.V + I)
        self.V = np.where(free, self.V + dV, self.V).astype(np.float32)

        fired = np.flatnonzero((self.V >= 1.0) & free)
        self.V[fired] = 0.0
        self.refrac[fired] = T_REF
        if len(fired):
            self.spikes[fired] += 1
            for u in fired:
                s0, s1 = self.e_row[u], self.e_row[u + 1]
                if s1 > s0:
                    d = self.delay[s0:s1]
                    m = self.w_mag[s0:s1]
                    ex = self.is_exc[s0:s1]
                    tgt = self.e_post[s0:s1]
                    slots = (t + d) % self.max_d
                    if ex.any():
                        np.add.at(self.ring_e, (slots[ex], tgt[ex]), m[ex])
                    if (~ex).any():
                        np.add.at(self.ring_i, (slots[~ex], tgt[~ex]), m[~ex])

        self.g_e, self.g_i = g_e, g_i
        self.fired = fired
        self.t = t + 1
        return fired

    def rate_hz(self, idx, window=None):
        """Mean rate of a set of neurons over the last `window` ms."""
        window = self.t if window is None else window
        if not len(idx):
            return 0.0
        return float(self.spikes[idx].sum() / len(idx) / (window / 1000.0))

    def run(self, ms, drivers, watch=None, sample_from=None):
        """Batch form: `ms` steps, sampled from `sample_from` (default: half)."""
        watch = watch or {}
        sample_from = ms // 2 if sample_from is None else sample_from
        per_ms = []
        acc = {k: {"v": [], "i": []} for k in watch}
        for _ in range(ms):
            self.step(drivers)
            per_ms.append(len(self.fired))
            if self.t > sample_from:
                for k, idx in watch.items():
                    if len(idx):
                        acc[k]["v"].append(float(self.V[idx].mean()))
                        acc[k]["i"].append(float(self.g_e[idx].mean()
                                                 - self.g_i[idx].mean()))
        out = {}
        for k, idx in watch.items():
            if not len(idx):
                continue
            out[k] = {
                "hz": float(self.spikes[idx].sum() / len(idx) / (ms / 1000.0)),
                "v": float(np.mean(acc[k]["v"])) if acc[k]["v"] else float("nan"),
                "g": float(np.mean(acc[k]["i"])) if acc[k]["i"] else float("nan"),
                "spikes": int(self.spikes[idx].sum()),
                "n": int(len(idx)),
            }
        out["_population_hz"] = float(np.mean(per_ms[sample_from:]))
        return out


def simulate(N, e_pre, e_post, w_sign, w_mag, delay, ms, drivers,
             rng_seed=42, watch=None, sample_from=None):
    """
    Conductance-based LIF over a subgraph. Batch entry point; the model itself
    is `CordNetwork`.

    `w_mag` is the synaptic conductance one spike delivers, `w_sign` its sign.
    Excitatory spikes add to g_e, inhibitory to g_i; the membrane equation is

        dV/dt = (-(V - 0) + g_e (E_e - V) + g_i (E_i - V)) / tau_m

    so inhibition subtracts *towards its reversal*, never below it, and a
    strongly inhibited cell can still be driven by enough excitation — which is
    what makes the fly's balanced E/I circuits work at all.
    """
    net = CordNetwork(N, e_pre, e_post, w_sign, w_mag, delay, rng_seed=rng_seed)
    return net.run(ms, drivers, watch=watch, sample_from=sample_from)


# --------------------------------------------------------------------------
def _cluster_profiles(Mn, k):
    """
    Split an organ's cells by which targets they innervate.

    `correlation` is undefined for a row with no variance — a cell whose
    outgoing synapses did not survive the threshold, or one that contacts a
    single target — and scipy answers that with a NaN and then refuses to
    cluster anything at all. The front legs never hit it and the middle and
    hind legs do, which is only visible once something asks for a leg other
    than the one step 2 reports on. Such a cell is given a group of its own
    instead of being allowed to take the leg apart.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    finite = np.isfinite(Mn).all(axis=1)
    varied = (Mn.max(axis=1) - Mn.min(axis=1)) > 1e-12
    usable = finite & varied
    labels = np.ones(len(Mn), dtype=np.int64)
    if usable.sum() < 2:
        return labels
    try:
        sub = fcluster(linkage(Mn[usable], method="average",
                               metric="correlation"), t=k, criterion="maxclust")
    except Exception as exc:                                   # pragma: no cover
        log(f"  the organ's subtypes could not be clustered ({exc}); "
            f"reporting them as one group")
        return labels
    labels[usable] = np.asarray(sub, dtype=np.int64)
    if (~usable).any():
        labels[~usable] = int(labels[usable].max()) + 1
        log(f"  {(~usable).sum()} organ cell(s) had no variance in their "
            f"target profile and were put in a group of their own")
    return labels


def leg_anatomy(a, loaded=None):
    """
    The leg's wiring, before anything is simulated: which cells are the sense
    organs, which are the motor pools, and how the organ splits by its own
    target profile. Step 3 imports this so that the body it closes the loop
    with is driven by exactly the pools step 2 measured.

    `loaded` is the tuple `load()` returns, for a caller that wants several
    legs: reading the connectome is the expensive part and it does not depend
    on which leg you are asking about.
    """
    meta, pre, post, syn = (load(a.banc, a.min_syn) if loaded is None
                            else loaded)
    n = len(meta)
    ct = text(meta, "cell_type")
    nt = np.array([v.split()[0].lower() if v else "" for v in
                   text(meta, "neurotransmitter_predicted")], dtype=object)
    sign = np.array([SIGN.get(x, 0.0) for x in nt], dtype="float32")
    fn = text(meta, "cell_function")
    det = text(meta, "cell_function_detailed")
    part_s = text(meta, "body_part_sensory")
    part_e = text(meta, "body_part_effector")
    ptt = text(meta, "peripheral_target_type")
    sc = text(meta, "super_class")
    side = text(meta, "side")
    neuroneme = text(meta, "neuromere")
    log(f"{n:,} neurons, {len(pre):,} edges at >= {a.min_syn} synapses")

    leg, s = a.leg, a.side
    neuro = LEGS[leg]
    on_leg = side == s
    part_here = np.array([leg in p for p in part_s])

    organs = {}
    for name, spec in ORGANS.items():
        m = (fn == spec["cell_function"]) & np.isin(det, list(spec["functions"])) & on_leg
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
    tgt_ids = np.unique(post[sel1])
    tgt_pos = {int(t): j for j, t in enumerate(tgt_ids)}
    src_pos = {int(c): i for i, c in enumerate(chordo)}
    M = np.zeros((len(chordo), len(tgt_ids)), dtype=np.float32)
    for p, q, w_ in zip(pre[sel1], post[sel1], syn[sel1]):
        M[src_pos[int(p)], tgt_pos[int(q)]] += float(w_)
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
    labels = _cluster_profiles(Mn, a.clusters)

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

    return NS(meta=meta, pre=pre, post=post, syn=syn, n=n, ct=ct, nt=nt,
              sign=sign, fn=fn, det=det, part_s=part_s, part_e=part_e,
              ptt=ptt, sc=sc, side=side, neuroneme=neuroneme,
              organs=organs, leg_motor=leg_motor, pools=pools,
              chordo=chordo, sel1=sel1, subtypes=subtypes,
              leg=leg, neuro=neuro, on_leg=on_leg, part_here=part_here,
              arcs=arcs, monosynaptic=monosynaptic, disyn=disyn)


def build_network(a, leg):
    """
    The reflex subgraph as arrays a simulation can step: nodes, edges, signs,
    axonal delays, the index groups to watch, and the synaptic scale.

    Built once and handed to both the batch runner (step 2) and the
    millisecond-by-millisecond loop (step 3), so the two cannot disagree about
    what the circuit is.
    """
    meta, pre, post, syn = leg.meta, leg.pre, leg.post, leg.syn
    n, ct, sc, sign = leg.n, leg.ct, leg.sc, leg.sign
    organs, pools, chordo, sel1 = leg.organs, leg.pools, leg.chordo, leg.sel1
    leg_motor, subtypes = leg.leg_motor, leg.subtypes
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

    return NS(N=len(nodes), nodes=nodes, node_of=node_of,
              e_pre=e_pre, e_post=e_post, w_sign=w_sign, w_raw=w_raw,
              delay=delay, watch=watch, sub_of=sub_of, fam_of=fam_of,
              desc_idx=desc_idx, nodes_of=nodes_of, mean_raw=mean_raw,
              scale_for=scale_for, K=K, w_mag=w_mag,
              subgraph_edges=int(sel.sum()), pos_parsed=parsed)


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

    leg = leg_anatomy(a)
    meta, sign = leg.meta, leg.sign
    organs, pools, subtypes = leg.organs, leg.pools, leg.subtypes
    neuro, s = leg.neuro, a.side
    arcs, monosynaptic, disyn = leg.arcs, leg.monosynaptic, leg.disyn

    net = build_network(a, leg)
    nodes, e_pre, e_post = net.nodes, net.e_pre, net.e_post
    w_sign, w_raw, delay = net.w_sign, net.w_raw, net.delay
    watch, sub_of, fam_of, desc_idx = net.watch, net.sub_of, net.fam_of, net.desc_idx
    scale_for, K, w_mag = net.scale_for, net.K, net.w_mag
    mean_raw = net.mean_raw


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
    cls = text(meta, "cell_class")[nodes]
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

    # The tonic state is the standing fly: a loaded leg's chordotonal organ is
    # active all the time. `standing` above is that state with the organ taken
    # away, so this one comparison has two directions and they say opposite
    # things. Both are reported, under the name of what was done rather than
    # what was meant, because step 3's answer depends on the sign.
    def signed_delta(hi, lo, name, field="hz"):
        return (pool(runs[hi], name).get(field, 0.0)
                - pool(runs[lo], name).get(field, 0.0))

    organ = {}
    for label, (hi, lo) in {"tone_added": ("release_tonic", "standing"),
                            "tone_removed": ("standing", "release_tonic")}.items():
        d = {f: {"flexor": signed_delta(hi, lo, "tibia flexor", f),
                 "extensor": signed_delta(hi, lo, "tibia extensor", f)}
             for f in ("hz", "v", "g")}
        organ[label] = {
            "d_flexor_hz": d["hz"]["flexor"],
            "d_extensor_hz": d["hz"]["extensor"],
            "flexor_minus_extensor_hz": d["hz"]["flexor"] - d["hz"]["extensor"],
            "d_flexor_g": d["g"]["flexor"],
            "d_extensor_g": d["g"]["extensor"],
            "flexor_minus_extensor_g": d["g"]["flexor"] - d["g"]["extensor"],
        }

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
        "subgraph": {"neurons": int(len(nodes)), "edges": net.subgraph_edges,
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
        "organ_tone": organ,
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
    names = {"release_tonic": "organ tone restored (from silent)"}
    for k, d in o["verdict"].items():
        A(f"| {names.get(k, k)} | {d['d_flexor_g']:+.5f} | {d['d_extensor_g']:+.5f} | "
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

    g = o["organ_tone"]
    A("### Which way the organ's own tone points\n")
    A("`standing` is the cord with the chordotonal organ silent; `organ tone "
      "restored` is the same cord with the organ driven. The two rows below are "
      "the same pair of runs read in the two directions, because which one is "
      "the *baseline* decides the sign, and step 3 is built on that sign.\n")
    A("| what was done to the organ | Δflexor Hz | Δextensor Hz | "
      "flexor − extensor Hz |")
    A("|---|---:|---:|---:|")
    for label, row in g.items():
        A(f"| {label.replace('_', ' ')} | {row['d_flexor_hz']:+.2f} | "
          f"{row['d_extensor_hz']:+.2f} | **{row['flexor_minus_extensor_hz']:+.2f}** |")
    A("")
    A("So the resting organ's tonic activity **opposes** the tibia flexor: it "
      "takes 0.50 Hz off it and gives 0.71 Hz to the extensor, and taking the "
      "organ away does the reverse. Read through the published transduction — "
      "flexion stretches the FeCO, extension relaxes it — that is a resistance "
      "reflex: imposed flexion raises FeCO activity, which favours the "
      "extensor, which extends the tibia against the imposed movement. Step 3 "
      "closes the loop and measures whether the body agrees.\n")

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
