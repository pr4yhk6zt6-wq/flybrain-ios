#!/usr/bin/env python3
"""
Does the cord couple the legs to each other?

Item 4 of the brief is the walking circuits: six legs that coordinate, with a
gait that *emerges* rather than being scheduled. There are two ways that can
happen in this project, and they are not equivalent:

  1. the connectome couples them — an organ drive on one leg changes the motor
     pools of another, through the interneurons the BANC reconstruction
     actually contains; or
  2. they do not couple, and a gait has to be built from a declared
     approximation (a local-oscillator model, labelled PLACEHOLDER, with the
     coupling strength as its parameter and a test that pins it).

Which of those is true is a measurement, not a design decision, and this is the
measurement. It runs the same LIF as tools/pool_probe.py at the app's operating
point, once as a baseline (the tone of assumption #5 on the descending group and
on every organ channel) and once per perturbation, where a single organ group is
pushed the way a knee push pushes it and nothing else is touched:

    the reflex input, at full stretch: x = 1, so #11's law
    tone·(1 − polarity·kappa·x) = 2.5·(1 − 0.5) = 1.25, i.e. the organ is
    *silenced* (a stretched chordotonal organ is silenced, and a compressed one
    fires: the same delta, opposite sign)

The response is read out as the per-leg mean rate over that leg's own pools, in
Hz and as a fraction of the baseline — because the question is not whether a
number changed (it will) but whether the change on the leg that was pushed is
any bigger than the change on the five legs that were not.

    python3 tools/coupling_probe.py                 # the default 200 ms
    python3 tools/coupling_probe.py --ms 400

Exit code is 0 either way: this measures a property of the published dataset, it
does not assert a threshold. The threshold belongs in the report that decides
how item 4 is built.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def load_pool_probe():
    """pool_probe.py, imported as a module — the LIF is its, not a copy."""
    spec = importlib.util.spec_from_file_location("pool_probe", HERE / "pool_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def common_premotor(c, pools_of, legs, n):
    """Which cells drive more than one leg's pools at once?

    The organ -> pool route above is nearly private to a leg (85:1), so if the
    six legs are coordinated by this connectome at all, the substrate is not the
    sense organs: it is interneurons that synapse onto the motor pools of more
    than one leg. Those cells *are* the tripod pattern, if the reconstruction has
    one — a cell that reaches the pools of both a front and a hind leg on
    opposite sides is exactly what a gait needs, and its strength is countable.

    Reads the reverse graph of the packed CSR (scipy's column slice, not a
    second copy of the edges).
    """
    import scipy.sparse as sp

    row, col = c["rowPtr"].astype(np.int32), c["colIdx"].astype(np.int32)
    A = sp.csr_matrix((np.ones(len(col), np.uint8), col, row), shape=(n, n))
    legs_of = np.zeros(n, np.uint8)          # how many legs' pools a cell reaches
    pair = {}
    per_leg = {}
    for leg in legs:
        cells = np.concatenate([c["members"](p) for p in pools_of[leg]])
        presyn = np.unique(A[:, cells].nonzero()[0])
        per_leg[leg] = presyn
        legs_of[presyn] += 1
    for i, a in enumerate(legs):
        for b in legs[i + 1:]:
            pair[(a, b)] = int(np.intersect1d(per_leg[a], per_leg[b]).size)

    print("\ncells driving the motor pools of more than one leg\n")
    total = int((legs_of > 0).sum())
    for k in range(1, len(legs) + 1):
        n_k = int((legs_of == k).sum())
        print(f"  onto {k} leg{'s' if k > 1 else ' '}: {n_k:7,d} cells"
              f"   ({n_k / max(total, 1) * 100:5.1f}% of the cells that reach any pool)")
    multi = np.isin(legs_of, np.arange(2, len(legs) + 1))
    print(f"\n  multi-leg premotor cells: {int(multi.sum()):,} of {total:,} "
          f"({multi.mean() * 100:.1f}%)")

    print("\n  shared premotor cells per leg pair (and the pairs a tripod gait needs)")
    tripod = {("T1_left", "T2_right"), ("T2_right", "T3_left"), ("T1_left", "T3_left"),
              ("T1_right", "T2_left"), ("T2_left", "T3_right"), ("T1_right", "T3_right")}
    print("  " + "".join(f"{l:>11s}" for l in legs))
    for a in legs:
        rowtxt = ""
        for b in legs:
            key = (a, b) if (a, b) in pair else (b, a)
            rowtxt += f"{pair.get(key, 0) if a != b else 0:11,d}"
        print(f"  {a:>10s} {rowtxt}")
    tripod_cells = 0
    for (a, b) in tripod:
        key = (a, b) if (a, b) in pair else (b, a)
        tripod_cells += pair.get(key, 0)
    print(f"\n  the six tripod pairs share {tripod_cells:,} premotor cells "
          f"(pairs counted once each)")
    # how strong are they? synapses from multi-leg cells into pools
    strength = int(multi[col].sum())
    print(f"  synapses from multi-leg cells into pool motor neurons: {strength:,}")
    return 0 if int(multi.sum()) > 0 else 1


def out_edges(col, row, cells):
    """Every outgoing edge of `cells`, concatenated — one pass, no Python loop."""
    s = row[cells].astype(np.int64)
    e = row[cells + 1].astype(np.int64)
    counts = e - s
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    idx = np.repeat(s, counts) + (np.arange(total) - offsets)
    return col[idx].astype(np.int64)


def structural(c, organs_of, pools_of, legs, n):
    """Is there a wire from one leg's sense organs to another leg's pools?

    The dynamic probe below asks what the network *does*, which in a chaotic LIF
    is a noisy question — a push on one leg changes the whole trajectory. The
    connectome's wiring is not noisy: this walks the graph the binary was packed
    from and reports, for each pair of legs, how many synapses leave one leg's
    organ cells for the other leg's pool motor neurons (one hop) and how many of
    those motor neurons are reachable within two hops. That is the answer to
    "can the cord couple these legs at all", and it is the number item 4 has to
    be built on.
    """
    row, col = c["rowPtr"], c["colIdx"]
    pool_sets = {leg: np.zeros(n, bool) for leg in legs}
    organ_cells = {}
    for leg in legs:
        for name in organs_of[leg]:
            organ_cells.setdefault(leg, []).extend(c["members"](name).tolist())
        pool_sets[leg][np.concatenate([c["members"](p) for p in pools_of[leg]])] = True
    pool_cells = {leg: np.flatnonzero(pool_sets[leg]) for leg in legs}

    print("\nstructural coupling — synapses out of one leg's organs into another "
          "leg's pools\n")
    print("  organs of    " + "".join(f"{l:>11s}" for l in legs) + "   (direct synapses)")
    direct_own, direct_other = [], []
    reach_own, reach_other = [], []
    for src in legs:
        cells = np.array(sorted(set(organ_cells[src])), dtype=np.int64)
        one = out_edges(col, row, cells)
        two = out_edges(col, row, np.unique(one))
        rowtxt = ""
        for dst in legs:
            n_direct = int(pool_sets[dst][one].sum())
            n_reach = int(pool_sets[dst][two].sum())
            frac = n_reach / max(1, len(pool_cells[dst]))
            rowtxt += f"{frac:11.2f}"
            if dst == src:
                direct_own.append(n_direct)
                reach_own.append(frac)
            else:
                direct_other.append(n_direct)
                reach_other.append(frac)
        print(f"  {src:>10s}   {rowtxt}   {int(pool_sets[src][one].sum())}")
    print("\n  (the numbers in the matrix are the fraction of that leg's pool "
          "motor neurons reachable from the source leg's organs within two hops)")
    do, dc = np.array(direct_own, float), np.array(direct_other, float)
    ro, rc = np.array(reach_own), np.array(reach_other)
    print(f"\n  one hop,   own leg: {do.mean():8.1f} synapses   "
          f"other legs: {dc.mean():8.1f}   ratio {do.mean() / max(dc.mean(), 1e-9):.2f}")
    print(f"  two hops,  own leg: {ro.mean():8.3f} of pools   "
          f"other legs: {rc.mean():8.3f}   ratio {ro.mean() / max(rc.mean(), 1e-9):.2f}")
    if rc.mean() > 0.05 and ro.mean() < 0.9:
        print("\n  ⇒ other legs' pools are reachable from this leg's organs: the")
        print("    connectome does couple the legs, so item 4's gait can be built")
        print("    from the cord's own wiring rather than from an invented oscillator.")
    else:
        print("\n  ⇒ the coupling is weak or absent on this route; item 4 needs a")
        print("    different route (interneurons of another type) or a declared")
        print("    PLACEHOLDER oscillator with the coupling as its parameter.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=pathlib.Path("build/flybanc.bin"), type=pathlib.Path)
    ap.add_argument("--meta", default=pathlib.Path("build/flybanc_meta.json"),
                    type=pathlib.Path)
    ap.add_argument("--ms", default=200, type=int)
    ap.add_argument("--gain", default=12.0, type=float,
                    help="the app's operating point (SimulationEngine's default)")
    ap.add_argument("--tone", default=2.5, type=float,
                    help="assumption #5 — the brain's tone on the descending group")
    ap.add_argument("--structural-only", action="store_true",
                    help="skip the dynamic half (it is the slow half: one run per leg)")
    ap.add_argument("--require-coupling", action="store_true",
                    help="exit 1 if the binary carries no multi-leg premotor cell — "
                         "that substrate is what item 4's gait is built on")
    args = ap.parse_args()

    pp = load_pool_probe()
    c = pp.load(args.bin, args.meta)

    legs = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
    pools_of = {}
    for leg in legs:
        pools_of[leg] = sorted(n for n in c["groups"] if n.startswith(f"pool:{leg}:"))
    organs_of = {}
    for leg in legs:
        organs_of[leg] = sorted(n for n in c["groups"] if n.startswith(f"organ:{leg}:"))
    present = [leg for leg in legs if pools_of[leg] and organs_of[leg]]
    if not present:
        print("no leg has both pools and organs in this binary", file=sys.stderr)
        return 1
    print(f"{args.bin}: {c['N']:,} neurons, legs {len(present)}, "
          f"pools per leg {[len(pools_of[l]) for l in present]}, "
          f"organ channels per leg {[len(organs_of[l]) for l in present]}")

    def rates(drives):
        _, window, _ = pp.run(c, args.ms, args.gain, drives)
        out = {}
        for leg in present:
            per_pool = [pp.hz(c, window, p, args.ms) for p in pools_of[leg]]
            out[leg] = float(np.mean(per_pool))
        return out

    base_drives = {"descending": args.tone, "sensory_vision": 1.0}
    for leg in present:
        for organ in organs_of[leg]:
            base_drives[organ] = args.tone

    structural(c, organs_of, pools_of, present, c["N"])
    coupled = common_premotor(c, pools_of, present, c["N"])
    if args.require_coupling:
        # Two things have to be true of the *packed binary*, not of the dataset:
        # the cells are there (the section above walks the packed CSR), and they
        # are named, so the app can read their rate off the GPU like any other
        # group. The first is the dataset's property; the second is ours, and it
        # is the one a future edit to build_banc.py could quietly drop.
        named = c["groups"].get("premotor:multileg")
        if named is None:
            print("\nFAIL the binary carries no 'premotor:multileg' group — the app "
                  "cannot ask the GPU for the rate of the cells that couple the "
                  "legs", file=sys.stderr)
            return 1
        print(f"\n  the binary names them too: premotor:multileg = "
              f"{named['count']:,} cells")
    if args.structural_only:
        return 0 if (coupled == 0 or not args.require_coupling) else 1
    if args.require_coupling and coupled != 0:
        print("\nFAIL no cell in this binary drives the pools of more than one leg",
              file=sys.stderr)
        return 1

    t0 = time.time()
    baseline = rates(base_drives)
    print(f"\nbaseline ({args.ms} ms, gain {args.gain:g}, tone {args.tone:g}): "
          + "  ".join(f"{leg} {baseline[leg]:.1f} Hz" for leg in present)
          + f"   [{time.time() - t0:.0f} s]")

    # The pushed leg's own organ channels, silenced by a full stretch (#11).
    print(f"\nper-leg pool rate, as a fraction of the baseline, when one leg's "
          f"chordotonal organ is stretched to x = 1\n")
    head = "  pushed leg  " + "".join(f"{leg:>11s}" for leg in present) + "   (Hz, pushed)"
    print(head)
    own, other = [], []
    for leg in present:
        drives = dict(base_drives)
        for organ in organs_of[leg]:
            if organ.endswith(":chordotonal"):
                drives[organ] = args.tone * 0.5      # 1 − kappa·x, kappa 0.5, x = 1
        got = rates(drives)
        row = ""
        for other_leg in present:
            frac = got[other_leg] / baseline[other_leg] if baseline[other_leg] > 1e-9 else 0.0
            row += f"{frac:11.3f}"
            if other_leg == leg:
                own.append(frac)
            else:
                other.append(frac)
        print(f"  {leg:>10s}  {row}   {got[leg]:8.1f}")

    own_a, oth_a = np.array(own), np.array(other)
    print(f"\n  the pushed leg's own pools:  {own_a.mean():.3f} ± {own_a.std():.3f} "
          f"of baseline   (n = {len(own_a)})")
    print(f"  the other five legs' pools:  {oth_a.mean():.3f} ± {oth_a.std():.3f} "
          f"of baseline   (n = {len(oth_a)})")
    print(f"  effect = own − other = {own_a.mean() - oth_a.mean():+.3f} of baseline")
    print("\n  Read it like this: if the six legs were independent, a push on one")
    print("  would move its own pools and leave the others at 1.000 — a large")
    print("  effect. An effect near zero means the push moves every leg equally,")
    print("  which is what a single global drive looks like, not what a cord does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
