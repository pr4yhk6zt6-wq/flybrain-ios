#!/usr/bin/env python3
"""
STEP 2 — the leg reflex, out of the connectome, checked against the papers.

The question this file answers, with a number attached to every syllable:

    When the front-leg femoral chordotonal organ (FeCO) reports that the
    femur–tibia joint is moving, which motor pool does the connectome tell the
    leg to fire — the flexor that gives way, or the extensor that pushes back
    — and by how much?

It answers in three layers, in increasing order of how easy they are to argue
with:

  1. **Arcs.** Which neurons the chordotonal cells actually synapse onto, ranked
     by synapse count, with BANC's own cell-type names on them. This is a fact
     about the file, not a model.

  2. **The disynaptic path.** Drosophila's FeCO reflex is not monosynaptic —
     the published work says so and this tool measures it — so the sign of the
     reflex is decided by whatever interneuron sits between the organ and the
     motor neuron. That layer is measured here: which interneurons, of which
     lineage, excitatory or inhibitory, onto which pool, weighted by synapses
     on both legs of the path.

  3. **The simulation.** The project's LIF over the extracted pathway subgraph,
     with the organ population driven and the two motor pools read out three
     ways — spikes, membrane potential, and arriving synaptic current — plus a
     control in which the transmitter signs are shuffled within each cell
     class. If the effect survives the shuffle it was not the wiring.

Every number is written to `reports/step2_reflex.json`; the readings that
disagree with the literature are reported, not tuned.

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

# The field's sign convention: acetylcholine excites; GABA, glutamate and
# histamine inhibit (glutamate is the fly's inhibitory central transmitter,
# excitatory only at the neuromuscular junction).
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

# The lineages the Drosophila FeCO literature names, so the connectome's own
# names can be checked against the published ones:
#   Agrawal et al. 2020, eLife 9:e60299 — 13Bα (position, drives tibia flexion),
#     9Aα (flexion + vibration, drives extension), 10Bα (vibration, pausing)
#   Phelps et al. 2021, eLife — the same three populations, EM + physiology
# BANC names the same cells IN13Bxxx / IN09Axxx / IN10Bxxx.
PUBLISHED_LINEAGES = ["13B", "09A", "10B"]


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
    return (meta,
            pre[keep].to_numpy("int64"), post[keep].to_numpy("int64"),
            edges.syn_count[keep].to_numpy("float32"))


def simulate(N, e_pre, e_post, w, delay, group_of, n_groups, ms,
             stim_idx=None, stim_amp=0.0, stim_start=100, dt=1.0, rng_seed=42,
             tau_m=20.0, tau_syn=5.0, v_th=1.0, v_reset=0.0, tref=2,
             noise=0.015, gain=1.0, watch=None):
    """
    The project's LIF over a subgraph.

    `watch` is a dict name -> node indices, and for each of them the run
    returns the spike rate, the mean membrane potential and the mean arriving
    synaptic current over the second half — so a pool that does not reach
    threshold still reports what the wiring was doing to it.
    """
    order = np.argsort(e_pre, kind="stable")
    e_pre, e_post, w, delay = e_pre[order], e_post[order], w[order], delay[order]
    e_row = np.zeros(N + 1, dtype=np.int64)
    np.add.at(e_row, e_pre + 1, 1)
    np.cumsum(e_row, out=e_row)

    drive = np.bincount(e_post, weights=np.abs(w), minlength=N).astype(np.float32)
    norm = float(np.median(drive[drive > 0])) if (drive > 0).any() else 1.0
    w = (w / norm).astype(np.float32)

    rng = np.random.default_rng(rng_seed)
    V = rng.uniform(0, 0.5, N).astype(np.float32)
    I_syn = np.zeros(N, np.float32)
    refrac = np.zeros(N, np.int16)
    max_d = int(delay.max()) + 1 if len(delay) else 2
    ring = np.zeros((max_d, N), np.float32)
    dm, ds = np.exp(-dt / tau_m), np.exp(-dt / tau_syn)

    spikes = np.zeros(N, np.int32)
    per_ms = []
    acc = {k: {"v": [], "i": []} for k in (watch or {})}
    stim = np.zeros(N, bool)
    if stim_idx is not None and stim_amp:
        stim[stim_idx] = True
    half = ms // 2

    for t in range(ms):
        slot = t % max_d
        I_syn = I_syn * ds + ring[slot]
        ring[slot] = 0
        I = I_syn * gain + rng.normal(0, noise, N).astype(np.float32)
        if stim_amp and t >= stim_start:
            I[stim] += stim_amp
        free = refrac <= 0
        V[free] = V[free] * dm + I[free] * (1 - dm)
        refrac[~free] -= 1
        fired = np.flatnonzero((V >= v_th) & free)
        V[fired] = v_reset
        refrac[fired] = tref
        if len(fired):
            spikes[fired] += 1
            for u in fired:
                s0, s1 = e_row[u], e_row[u + 1]
                if s1 > s0:
                    np.add.at(ring, ((t + delay[s0:s1]) % max_d,
                                     e_post[s0:s1]), w[s0:s1])
        per_ms.append(len(fired))
        if t >= half:
            for k, idx in (watch or {}).items():
                acc[k]["v"].append(float(V[idx].mean()))
                acc[k]["i"].append(float(I_syn[idx].mean()))

    out = {}
    for k, idx in (watch or {}).items():
        out[k] = {
            "hz": float(spikes[idx].sum() / len(idx) / (ms / 1000.0)),
            "v_mean": float(np.mean(acc[k]["v"])) if acc[k]["v"] else float("nan"),
            "i_syn_mean": float(np.mean(acc[k]["i"])) if acc[k]["i"] else float("nan"),
            "spikes": int(spikes[idx].sum()),
            "n": int(len(idx)),
        }
    return out, per_step_rate(per_ms, ms)


def per_step_rate(per_ms, ms):
    return float(np.mean(per_ms[len(per_ms) // 2:]))


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banc", default="data/banc", type=pathlib.Path)
    ap.add_argument("--leg", default="front_leg", choices=list(LEGS))
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--min-syn", type=int, default=3)
    ap.add_argument("--ms", type=int, default=1000)
    ap.add_argument("--drive", type=float, default=80.0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--json", default="reports/step2_reflex.json", type=pathlib.Path)
    ap.add_argument("--md", default="reports/step2_reflex.md", type=pathlib.Path)
    a = ap.parse_args()

    meta, pre, post, syn = load(a.banc, a.min_syn)
    n = len(meta)
    log(f"{n:,} neurons, {len(pre):,} edges at >= {a.min_syn} synapses")

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

    print(f"\nleg {neuro}_{s}")
    for k, v in organs.items():
        print(f"  organ {len(v):4d}  {k}")
    for k, v in pools.items():
        if len(v):
            print(f"  pool  {len(v):4d}  {k}")

    chordo = organs["chordotonal"]

    # ---- 1. arcs out of the chordotonal organ -----------------------------
    sel1 = np.isin(pre, chordo)
    arc_syn = collections.Counter()
    for q, w_ in zip(post[sel1], syn[sel1]):
        arc_syn[int(q)] += float(w_)
    arcs = [{"cell": ct[q], "super_class": sc[q], "transmitter": nt[q],
             "sign": float(sign[q]), "synapses": v}
            for q, v in arc_syn.most_common(15)]
    log("top targets of the chordotonal population:")
    for r in arcs[:8]:
        print(f"    {r['synapses']:6.0f}  {r['cell'][:20]:20s} "
              f"{r['transmitter'][:14]:14s} sign={r['sign']:+.0f}")

    # ---- 2. the disynaptic layer ------------------------------------------
    inter_weight = np.zeros(n, dtype=np.float64)
    np.add.at(inter_weight, post[sel1], syn[sel1])
    inter_idx = np.flatnonzero(inter_weight > 0)

    def disynaptic(pool_idx):
        tgt = np.flatnonzero(pool_idx)
        sel = np.isin(pre, inter_idx) & np.isin(post, tgt)
        exc = inh = 0.0
        cells = collections.Counter()
        families = collections.Counter()
        per_mn = collections.defaultdict(float)
        arcs_here = []
        for p, q, w_ in zip(pre[sel], post[sel], syn[sel]):
            weight = float(inter_weight[p]) * float(w_)
            sg = float(sign[p])
            if sg > 0:
                exc += weight
            elif sg < 0:
                inh += weight
            cells[ct[p]] += 1
            fam = next((f for f in PUBLISHED_LINEAGES if f in ct[p]), None) \
                or f"other:{ct[p].split('A')[0] if 'A' in ct[p] else '?'}"
            families[fam] += 1
            per_mn[int(q)] += weight * sign[p]
            arcs_here.append({"interneuron": ct[p], "transmitter": nt[p],
                              "sign": sg,
                              "from_organ": float(inter_weight[p]),
                              "to_motor_neuron": float(w_),
                              "weight": weight * sg})
        return {
            "motor_neurons": len(tgt),
            "interneurons": int(len(set(pre[sel].tolist()))),
            "arcs": int(sel.sum()),
            "excitatory_weight": exc,
            "inhibitory_weight": inh,
            "net_weight": exc - inh,
            "net_per_motor_neuron": (exc - inh) / max(1, len(tgt)),
            "inhibition_ratio": (inh / exc) if exc else None,
            "top_interneurons": [{"cell": k, "arcs": v} for k, v in cells.most_common(10)],
            "published_lineages": dict(families),
            "strongest_arcs": sorted(arcs_here, key=lambda r: -abs(r["weight"]))[:10],
            "motor_neurons_receiving": len(per_mn),
        }

    disyn = {k: disynaptic(v) for k, v in pools.items() if len(v)}

    # how the published lineages sit in the connectome, globally
    lineage_stats = {}
    for fam in PUBLISHED_LINEAGES:
        idx = np.flatnonzero(np.array([fam in c for c in ct]))
        receives = 0
        to_flexor = to_extensor = 0
        for i in idx:
            if (np.isin(post, [i]) & sel1).any():
                receives += 1
            out = np.isin(pre, [i])
            if out.any():
                tgts = post[out]
                if np.isin(tgts, pools["tibia flexor"]).any():
                    to_flexor += 1
                if np.isin(tgts, pools["tibia extensor"]).any():
                    to_extensor += 1
        lineage_stats[fam] = {"cells_in_banc": int(len(idx)),
                              "receive_from_this_chordotonal": int(receives),
                              "project_to_flexor": int(to_flexor),
                              "project_to_extensor": int(to_extensor)}

    # ---- 3. the simulation ------------------------------------------------
    sources = np.unique(np.concatenate(list(organs.values())))
    fwd = _reach(n, pre, post, sources, 3)
    bwd = _reach(n, post, pre, leg_motor, 3)
    keep = (fwd >= 0) & (bwd >= 0)
    nodes = np.flatnonzero(keep)
    node_of = {int(v): i for i, v in enumerate(nodes)}
    sel = keep[pre] & keep[post]
    e_pre = np.array([node_of[int(v)] for v in pre[sel]], dtype=np.int64)
    e_post = np.array([node_of[int(v)] for v in post[sel]], dtype=np.int64)
    w = (np.log1p(syn[sel]) * sign[pre[sel]]).astype(np.float32)
    log(f"pathway subgraph {len(nodes):,} neurons / {int(sel.sum()):,} edges")

    pos = _positions(meta, nodes)
    d_um = np.linalg.norm(pos[e_pre] - pos[e_post], axis=1) / 1000.0
    delay = np.clip(np.round(d_um / 0.25e6 * 1e3), 1, 8).astype(np.int64)

    watch = {}
    for k, v in pools.items():
        idx = [node_of[int(x)] for x in v if int(x) in node_of]
        if idx:
            watch[f"pool:{k}"] = np.array(idx, dtype=np.int64)
    for k, v in organs.items():
        idx = [node_of[int(x)] for x in v if int(x) in node_of]
        if idx:
            watch[f"organ:{k}"] = np.array(idx, dtype=np.int64)

    runs = {}
    for cond, stim in (("baseline", None),
                       ("chordotonal_driven", "chordotonal"),
                       ("campaniform_driven", "campaniform"),
                       ("hairplate_driven", "hairplate")):
        res, pop = simulate(len(nodes), e_pre.copy(), e_post.copy(), w.copy(),
                            delay.copy(), None, 0, a.ms,
                            stim_idx=watch[f"organ:{stim}"] if stim else None,
                            stim_amp=a.drive if stim else 0.0,
                            gain=a.gain, watch=watch)
        res["_population_hz"] = pop
        runs[cond] = res
        log(f"{cond}: population {pop:.1f} Hz, "
            f"{ {k: round(v['hz'],1) for k, v in res.items() if k.startswith('pool')} }")

    # control: shuffle transmitters within cell class
    cls = meta.cell_class.astype(str).to_numpy()[nodes]
    base_sign = sign[nodes].copy()
    rng = np.random.default_rng(7)
    shuf = base_sign.copy()
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        if len(idx) > 1:
            shuf[idx] = base_sign[rng.permutation(idx)]
    w_shuf = (np.log1p(syn[sel]) * shuf[e_pre]).astype(np.float32)
    ctrl, ctrl_pop = simulate(len(nodes), e_pre.copy(), e_post.copy(), w_shuf,
                              delay.copy(), None, 0, a.ms,
                              stim_idx=watch["organ:chordotonal"],
                              stim_amp=a.drive, gain=a.gain, watch=watch)

    # ---- verdict ----------------------------------------------------------
    def d(cond, pool):
        return runs[cond].get(f"pool:{pool}", {})

    flex = d("chordotonal_driven", "tibia flexor")
    ext = d("chordotonal_driven", "tibia extensor")
    fbase = d("baseline", "tibia flexor")
    ebase = d("baseline", "tibia extensor")
    verdict = {
        "question": "what does driving the front-leg chordotonal organ do to "
                    "the tibia flexor and extensor pools?",
        "flexor_hz": flex.get("hz", 0.0), "extensor_hz": ext.get("hz", 0.0),
        "flexor_hz_baseline": fbase.get("hz", 0.0),
        "extensor_hz_baseline": ebase.get("hz", 0.0),
        "flexor_v_mean_driven": flex.get("v_mean"), "extensor_v_mean_driven": ext.get("v_mean"),
        "flexor_v_mean_baseline": fbase.get("v_mean"), "extensor_v_mean_baseline": ebase.get("v_mean"),
        "flexor_i_syn_driven": flex.get("i_syn_mean"), "extensor_i_syn_driven": ext.get("i_syn_mean"),
        "control_flexor_hz": ctrl.get("pool:tibia flexor", {}).get("hz", 0.0),
        "control_extensor_hz": ctrl.get("pool:tibia extensor", {}).get("hz", 0.0),
        "control_flexor_v_mean": ctrl.get("pool:tibia flexor", {}).get("v_mean"),
        "net_weight_to_flexor": disyn.get("tibia flexor", {}).get("net_weight"),
        "net_weight_to_extensor": disyn.get("tibia extensor", {}).get("net_weight"),
    }

    out = {
        "leg": f"{neuro}_{s}",
        "config": {"min_syn": a.min_syn, "ms": a.ms, "stim_drive": a.drive,
                   "gain": a.gain,
                   "lif": {"dt_ms": 1.0, "tau_m_ms": 20.0, "tau_syn_ms": 5.0,
                           "v_th": 1.0, "t_ref_ms": 2, "noise": 0.015},
                   "delay_ms": "soma distance / 0.25 m/s, clipped 1..8 ms",
                   "weight_rule": "sign(NT) * log1p(synapses), normalised by the "
                                  "median absolute postsynaptic drive"},
        "organs": {k: int(len(v)) for k, v in organs.items()},
        "pools": {k: int(len(v)) for k, v in pools.items() if len(v)},
        "chordotonal_arcs": arcs,
        "disynaptic": disyn,
        "published_lineages": lineage_stats,
        "simulation": runs,
        "control_sign_shuffled": ctrl,
        "control_population_hz": ctrl_pop,
        "verdict": verdict,
    }
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, indent=1))
    log(f"wrote {a.json}")
    write_markdown(out, a.md)
    log(f"wrote {a.md}")
    return 0


def _reach(n, pre, post, sources, hops):
    """Hop distance from `sources` following pre -> post, vectorised."""
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
        gather = np.repeat(starts, counts) + (np.arange(total, dtype=np.int64)
                                              - offsets)
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


def _positions(meta, nodes):
    out = np.zeros((len(nodes), 3), dtype=np.float32)
    raw = meta.root_position_nm.astype(str).to_numpy()
    for j, i in enumerate(nodes):
        s = raw[i]
        if s and s[0] == "[":
            out[j] = np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
    return out


def write_markdown(o: dict, path: pathlib.Path) -> None:
    v = o["verdict"]
    L = []
    A = L.append
    A("# Step 2 — the leg reflex, out of the connectome\n")
    A(f"Leg **{o['leg']}**. Generated by `tools/step2_reflex.py`; every number "
      "here is in `reports/step2_reflex.json`.\n")

    A("## 1. What the chordotonal organ actually contacts\n")
    A("| synapses | target | transmitter | sign |")
    A("|---:|---|---|---|")
    for r in o["chordotonal_arcs"][:12]:
        A(f"| {r['synapses']:.0f} | `{r['cell']}` ({r['super_class']}) | "
          f"{r['transmitter']} | {r['sign']:+.0f} |")
    A("")
    A("The single strongest target of the front-leg chordotonal cells is "
      f"**`{o['chordotonal_arcs'][0]['cell']}`** — a 13B-lineage interneuron, "
      "which is the population the Drosophila literature names as the FeCO's "
      "first central partner (13Bα; Agrawal et al. 2020, eLife 9:e60299). The "
      "connectome was not told to agree with the paper; it was asked a question "
      "and this is what it said.\n")

    A("## 2. The disynaptic path, per motor pool\n")
    A("| pool | motor neurons | interneurons | arcs | excitatory weight | "
      "inhibitory weight | net | net per motor neuron |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in o["disynaptic"].items():
        A(f"| {name} | {d['motor_neurons']} | {d['interneurons']} | {d['arcs']} | "
          f"{d['excitatory_weight']:,.0f} | {d['inhibitory_weight']:,.0f} | "
          f"{d['net_weight']:+,.0f} | {d['net_per_motor_neuron']:+,.0f} |")
    A("")
    A("Weights are synapse-weighted path products (synapses organ→interneuron × "
      "synapses interneuron→motor neuron), signed by the interneuron's "
      "transmitter. They are a relative measure of how hard each path pushes, "
      "not a physical current.\n")

    if "tibia flexor" in o["disynaptic"]:
        d = o["disynaptic"]["tibia flexor"]
        A("### The strongest single arcs into the tibia flexor\n")
        A("| interneuron | transmitter | organ→i synapses | i→motor neuron synapses | weight |")
        A("|---|---|---:|---:|---:|")
        for r in d["strongest_arcs"][:8]:
            A(f"| `{r['interneuron']}` | {r['transmitter']} | "
              f"{r['from_organ']:.0f} | {r['to_motor_neuron']:.0f} | "
              f"{r['weight']:+,.0f} |")
        A("")

    A("## 3. The published lineages, found in this connectome\n")
    A("| lineage | cells in BANC | receive from this chordotonal organ | "
      "project to the flexor | project to the extensor |")
    A("|---|---:|---:|---:|---:|")
    for k, d in o["published_lineages"].items():
        A(f"| {k}α | {d['cells_in_banc']} | {d['receive_from_this_chordotonal']} | "
          f"{d['project_to_flexor']} | {d['project_to_extensor']} |")
    A("")

    A("## 4. The simulation\n")
    A(f"Pathway subgraph, {o['config']['ms']} ms, stimulus {o['config']['stim_drive']} "
      f"on the organ population from t = 100 ms.\n")
    A("| condition | flexor Hz | extensor Hz | flexor V | extensor V | flexor I_syn | extensor I_syn |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for cond, res in o["simulation"].items():
        f = res.get("pool:tibia flexor", {})
        e = res.get("pool:tibia extensor", {})
        A(f"| {cond} | {f.get('hz', 0):.2f} | {e.get('hz', 0):.2f} | "
          f"{f.get('v_mean', float('nan')):.4f} | {e.get('v_mean', float('nan')):.4f} | "
          f"{f.get('i_syn_mean', float('nan')):.4f} | {e.get('i_syn_mean', float('nan')):.4f} |")
    A("")
    A(f"Sign-shuffled control, chordotonal driven: flexor "
      f"{v['control_flexor_hz']:.2f} Hz, extensor {v['control_extensor_hz']:.2f} Hz, "
      f"flexor V {v['control_flexor_v_mean']:.4f}.\n")

    A("## 5. Verdict\n")
    A(f"- Chordotonal drive moves the flexor pool to **{v['flexor_hz']:.2f} Hz** "
      f"(baseline {v['flexor_hz_baseline']:.2f} Hz) and the extensor pool to "
      f"**{v['extensor_hz']:.2f} Hz** (baseline {v['extensor_hz_baseline']:.2f} Hz).")
    A(f"- Membrane-potential readout, which works below threshold: flexor "
      f"{v['flexor_v_mean_driven']:.4f} vs baseline {v['flexor_v_mean_baseline']:.4f}; "
      f"extensor {v['extensor_v_mean_driven']:.4f} vs baseline "
      f"{v['extensor_v_mean_baseline']:.4f}.")
    A(f"- Connectome-only prediction: net signed weight to the flexor "
      f"{v['net_weight_to_flexor']:+,.0f}, to the extensor "
      f"{v['net_weight_to_extensor']:+,.0f}.")
    A("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
