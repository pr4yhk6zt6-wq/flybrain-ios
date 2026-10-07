#!/usr/bin/env python3
"""
Item 12: does touch reach the leg motor pools, and what does a footfall do?

The gap docs/AUDIT.md row 12 names is not "the tactile cells are missing" — they
are in the shipped connectome, 3,184 of them (`sensory_tactile`, and now one
`organ:<seg>_<side>:tactile` group per leg). The gap is that *nothing drives
them*: the campaniform organ reports the load a leg carries, which is the same
number whether the foot has just landed or has been standing for a second.

This tool answers the two questions that decide whether wiring the organ is
worth it, off the device and on the files the app ships:

  1. the pathway. For every leg's tactile group: how many cells, how many
     outgoing synapses, and how many of them land on motor neurons. A sense
     organ that reaches no motor neuron is a readout, not a sensor.
  2. the loop. Drive one leg's tactile organ through a *footfall* — contact,
     lift, contact, the way `FlyCord`'s `case "tactile"` computes it from
     `fly_body.json`'s thresholds — and report what the six leg motor pools do:
     the same leg against the other five, and the down-phase against the
     air-phase. The drive law is transcribed from `FlyBrain/Sources/FlyCord.swift`
     (threshold 5 % of the standing load, hysteresis 0.5, phasic peak 2.0,
     adaptation 30 ms — assumptions #41 and #42) so that what this measures is
     what the phone will send.

The LIF and the loader are `tools/pool_probe.py`'s — one implementation, the
same one `tools/verify_banc.py` judges the Metal kernel against.

    python3 tools/tactile_probe.py               # every leg, and the pathway
    python3 tools/tactile_probe.py --gate        # what CI runs: fail if the
                                                 # organ is missing, empty, or
                                                 # reaches no motor neuron
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# One source of the loop: the reference LIF, its loader, and the organ law the
# cord runs. The law's four constants are the cord's own settings; they are
# repeated here because the Swift and the Python cannot import each other, and
# `--gate` checks the organ's *effect*, not the constants.
import motor_pools as mp                                          # noqa: E402

_spec = importlib.util.spec_from_file_location("pool_probe", HERE / "pool_probe.py")
pool_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pool_probe)

TONE = 2.5              # FlyCordSettings.tone (assumption #5)
TACTILE = "tactile"     # the organ's slug, from the organ table
# FlyCordSettings, item 12 (assumptions #41, #42).
CONTACT_FRACTION = 0.05
HYSTERESIS = 0.5
PEAK = 2.0
ADAPT_MS = 30.0
DT_MS = 1.0
DESCENDING = "descending"   # FlyCordSettings.descendingGroup


def load(bin_path, meta_path):
    net = pool_probe.load(bin_path, meta_path)
    return net


def target_classes(net, indices):
    """The super-classes of a set of neuron indices, from the pack's own meta."""
    names = net["meta"]["superClasses"]        # a list, indexed by super-class id
    ids = net["neuronMeta"][indices, 1]
    out = {}
    for i in ids:
        key = names[int(i)]
        out[key] = out.get(key, 0) + 1
    return out


def outgoing(net, indices):
    rowptr, colidx, weight = net["rowPtr"], net["colIdx"], net["weight"]
    parts = [np.arange(rowptr[i], rowptr[i + 1]) for i in indices
             if rowptr[i + 1] > rowptr[i]]
    if not parts:
        return np.array([], np.int64), np.array([], np.float32)
    k = np.concatenate(parts)
    return colidx[k], weight[k]


def pathway(net, group):
    """The organ's cells, its outgoing synapses, and how many are to motor neurons."""
    cells = net["members"](group)
    t, w = outgoing(net, cells)
    if len(t) == 0:
        return dict(cells=len(cells), edges=0, sites=0.0, motor_sites=0.0,
                    classes={})
    classes = target_classes(net, t)
    super_of = net["neuronMeta"][:, 1]
    motor_id = list(net["meta"]["superClasses"]).index("motor")
    m = super_of[t] == motor_id
    return dict(cells=len(cells), edges=len(t), sites=float(w.sum()),
                motor_sites=float(w[m].sum()), classes=classes)


def footfall_drive(ms=800, down_ms=300, air_ms=100):
    """
    The organ's drive through one lift-and-land, computed the way the cord
    computes it.

    The contact signal is idealised — a foot is either carrying its own weight
    or it is in the air — because the real one is the solver's `footForce` on
    the phone and cannot be run here. What this *does* reproduce exactly is the
    receptor: the threshold, the hysteresis, the phasic peak and the adaptation
    are the cord's, so the drive sequence below is the one the phone sends.
    """
    h, down, was_down, out = 0.0, True, None, []
    a = min(1.0, DT_MS / ADAPT_MS)
    for t in range(ms):
        phase_down = (t // (down_ms + air_ms)) % 2 == 0
        load = 1.0 if phase_down else 0.0          # in units of the standing load
        on, off = CONTACT_FRACTION, CONTACT_FRACTION * HYSTERESIS
        down = (load > off) if was_down else (load > on)
        if down:
            h = PEAK if not was_down else h + (1 - h) * a
        else:
            h += (0 - h) * a
        was_down = down
        out.append(TONE * h)
    return np.array(out), np.array([(t // (down_ms + air_ms)) % 2 == 0
                                    for t in range(ms)])


def leg_pools(net):
    out = {}
    for g in net["meta"]["groups"]:
        name = g["name"]
        if name.startswith("pool:"):
            out.setdefault(name.split(":")[1], []).append(name)
    return out


def run_pattern(net, group, drive, ms, descending=True):
    """
    Drive one organ and return every leg pool's spike count per millisecond.

    The *brain* is on, at the cord's own tone: `FlyCord.update` drives the
    descending population on every millisecond of every phase, and the organs
    are what modulate the pools on top of that. A probe that left the tone out
    would be measuring a different animal — one whose only input is a single
    sense organ.
    """
    pools = [g["name"] for g in net["meta"]["groups"] if g["name"].startswith("pool:")]
    lif = pool_probe.LIF(net, 12.0, seed=42)
    members = {p: net["members"](p) for p in pools}
    hit = {p: np.zeros(ms) for p in pools}
    for t in range(ms):
        if descending:
            lif.drive(DESCENDING, TONE)
        lif.drive(group, float(drive[t]))
        sp = lif.step()
        idx = np.flatnonzero(sp) if sp.dtype == bool else sp
        for p in pools:
            hit[p][t] = np.isin(idx, members[p]).sum()
    return pools, hit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=ROOT / "build" / "flybanc.bin", type=pathlib.Path)
    ap.add_argument("--meta", default=ROOT / "build" / "flybanc_meta.json", type=pathlib.Path)
    ap.add_argument("--out", default=ROOT / "reports" / "tactile_probe.json", type=pathlib.Path)
    ap.add_argument("--ms", type=int, default=800)
    ap.add_argument("--gate", action="store_true",
                    help="fail if the organ is missing, empty, or reaches no motor neuron")
    args = ap.parse_args()

    net = load(args.bin, args.meta)
    report = {"source": str(args.bin), "tone": TONE,
              "threshold": {"contact_fraction": CONTACT_FRACTION,
                            "hysteresis": HYSTERESIS, "peak": PEAK,
                            "adapt_ms": ADAPT_MS},
              "organs": {}, "footfall": {}}

    print(f"connectome: {net['N']:,} neurons, {net['e']:,} edges at >= 3 synapses")
    print(f"{'organ':28s} {'cells':>6s} {'edges':>7s} {'sites':>9s} {'to motor':>9s}")
    problems = []
    for leg, neuro in mp.LEGS.items():
        for side in mp.SIDES:
            group = mp.organ_slug(neuro, side, TACTILE)
            if group not in net["groups"]:
                problems.append(f"the connectome has no group '{group}'")
                print(f"{group:28s} {'—':>6s}  (missing)")
                continue
            p = pathway(net, group)
            report["organs"][group] = p
            print(f"{group:28s} {p['cells']:6d} {p['edges']:7d} {p['sites']:9.1f} "
                  f"{p['motor_sites']:9.1f}")
            if p["cells"] == 0:
                problems.append(f"'{group}' is empty")
            if p["motor_sites"] <= 0:
                problems.append(f"'{group}' reaches no motor neuron")

    # The leg the footfall is run on: the middle left, the leg step 2's baseline
    # uses, so the two reports describe the same animal.
    group = mp.organ_slug("T2", "left", TACTILE)
    drive, down = footfall_drive(ms=args.ms)
    pools, hit = run_pattern(net, group, drive, args.ms)
    own = [p for p in pools if p.startswith("pool:T2_left:")]
    others = [p for p in pools if not p.startswith("pool:T2_left:")]
    print(f"\nthe footfall: 300 ms down, 100 ms in the air, driving {group}")
    print(f"  drive: {drive.min():.2f} .. {drive.max():.2f} (tone {TONE})")
    print(f"  the leg's own pools ({len(own)}):")
    for p in sorted(own):
        # A rate is a count divided by the time it was counted over, once: the
        # first version of this line divided by the phase length twice and read
        # `mean()` as if it were a sum, which understated every rate below by the
        # length of the phase in milliseconds.
        rate_down = float(hit[p][down].sum()) * 1000.0 / max(1, int(down.sum()))
        rate_air = float(hit[p][~down].sum()) * 1000.0 / max(1, int((~down).sum()))
        report["footfall"][p] = {"down_hz": rate_down, "air_hz": rate_air}
        print(f"    {p.split(':')[2]:22s} down {rate_down:7.2f} Hz   air {rate_air:7.2f} Hz"
              f"   {'+' if rate_air > rate_down else ''}{rate_air - rate_down:+7.2f}")
    want = [p for p in others if p.startswith("pool:T2_right:")]
    print(f"  the other side's pools ({len(want)}):")
    for p in sorted(want):
        rate = float(hit[p].sum()) * 1000.0 / args.ms
        report["footfall"][p] = {"mean_hz": rate}
        print(f"    {p.split(':')[2]:22s} mean {rate:7.2f} Hz")
    report["footfall"]["_note"] = ("rates are spikes per pool per phase, scaled to Hz "
                                   "— a pool is many cells; compare pools, not "
                                   "cells")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    # Like for like, on the same leg: each of the three organs driven at the
    # cord's tone for 400 ms, with the brain's tone on the descending population
    # as it always is. This is the number that says what the touch channel adds:
    # a leg's own pools, against the other side's.
    print("\nthe three organs of the same leg, each driven at the tone (400 ms):")
    compare = {}
    for kind in sorted(mp.DRIVEN_ORGANS):
        g = mp.organ_slug("T2", "left", kind)
        if g not in net["groups"]:
            continue
        pools, hit = run_pattern(net, g, np.full(400, TONE), 400)
        own_hz = sum(float(h[100:].sum()) for p_, h in hit.items()
                     if p_.startswith("pool:T2_left:")) * 1000.0 / 300.0
        off_hz = sum(float(h[100:].sum()) for p_, h in hit.items()
                     if p_.startswith("pool:T2_right:")) * 1000.0 / 300.0
        compare[kind] = {"own_hz": own_hz, "other_side_hz": off_hz}
        print(f"  {kind:28s} the leg's own pools {own_hz:8.2f} Hz   "
              f"the other side {off_hz:8.2f} Hz")
    report["compare"] = compare

    print(f"\nwrote {args.out}")

    if args.gate:
        if problems:
            for p in problems:
                print(f"  FAIL {p}")
            return 1
        print("  gate ok: six tactile organs, all populated, all reaching motor neurons")
    return 0 if not problems else (0 if not args.gate else 1)


if __name__ == "__main__":
    sys.exit(main())
