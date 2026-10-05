#!/usr/bin/env python3
"""
Does the cord put the six legs into a tripod — or into one lump?

Item 4 needs a gait that emerges from the connectome, and the previous
measurement (tools/coupling_probe.py) settled the substrate: the sense organs
reach their own leg's pools 85:1 over any other leg's, while 788 cells reach
more than one leg's pools at once, carrying 69,543 synapses. That is the wiring.
This is the dynamics: when the cord runs at the app's operating point, do the
six legs' motor pools *differ in time* from each other?

The test is a correlation, and the prediction of a tripod gait is specific. In
the alternating tripod the legs divide into two sets that swing together and
stance together:

    set A = {T1_left, T2_right, T3_left}
    set B = {T1_right, T2_left, T3_right}

so the pairs *within* a set should co-vary more than the pairs *across* sets. If
instead the cord is one lump — every leg's pools moved by the same common drive
— then within-set and across-set correlations are equal, and the difference is
zero. That difference is the number this tool reports:

    tripod index = mean corr(within set) − mean corr(across sets)

What it does *not* claim: a positive index is not yet a gait. It is the cord
showing that the two tripods are two, which is the thing a gait has to be built
out of — or the thing that says it cannot be and an oscillator has to be
declared instead (PLACEHOLDER, with its coupling as the parameter).

    python3 tools/gait_probe.py                  # 400 ms, the app's point
    python3 tools/gait_probe.py --ms 1000 --bins 20

Per-millisecond spike counts for every pool group come back from the LIF run
(pool_probe.run's `track`), binned to `--bins` ms, so nothing here re-implements
the network.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# The two tripods, by the legs' own names (BANC's T1/T2/T3 = front/middle/hind).
SET_A = ["T1_left", "T2_right", "T3_left"]
SET_B = ["T1_right", "T2_left", "T3_right"]


def load_pool_probe():
    spec = importlib.util.spec_from_file_location("pool_probe", HERE / "pool_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tripod_index(traces, legs):
    """Per-leg rate traces -> the within-set minus across-set correlation.

    The traces are z-scored per leg first, so a leg that simply fires faster
    than another does not count as co-varying with anything: the question is
    *shape*, not level.
    """
    z = {}
    for leg in legs:
        x = traces[leg]
        sd = x.std()
        z[leg] = (x - x.mean()) / sd if sd > 1e-9 else x * 0.0

    def corr(a, b):
        return float(np.corrcoef(z[a], z[b])[0, 1]) if z[a].std() > 1e-9 and z[b].std() > 1e-9 else 0.0

    within, across = [], []
    for i, a in enumerate(legs):
        for b in legs[i + 1:]:
            same = (a in SET_A and b in SET_A) or (a in SET_B and b in SET_B)
            (within if same else across).append(corr(a, b))
    w, c = np.array(within), np.array(across)
    return float(w.mean()), float(c.mean()), float(w.mean() - c.mean()), w, c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=pathlib.Path("build/flybanc.bin"), type=pathlib.Path)
    ap.add_argument("--meta", default=pathlib.Path("build/flybanc_meta.json"),
                    type=pathlib.Path)
    ap.add_argument("--ms", default=400, type=int)
    ap.add_argument("--bins", default=20, type=int, help="ms per bin of the traces")
    ap.add_argument("--gain", default=12.0, type=float)
    ap.add_argument("--tone", default=2.5, type=float)
    ap.add_argument("--drive-premotor", default=0.0, type=float,
                    help="also drive premotor:multileg at this current (0 = not driven)")
    ap.add_argument("--no-vision", action="store_true",
                    help="leave the photoreceptors undriven too: the descending "
                         "tone is then the only structured input the cord has")
    ap.add_argument("--silent-organs", action="store_true",
                    help="do not drive the organ groups: what the cord does on its "
                         "own (descending tone + retina only)")
    args = ap.parse_args()

    pp = load_pool_probe()
    c = pp.load(args.bin, args.meta)
    legs = SET_A + SET_B
    pools_of = {leg: sorted(n for n in c["groups"] if n.startswith(f"pool:{leg}:"))
                for leg in legs}
    order = [p for leg in legs for p in pools_of[leg]]      # track order = flat pools
    if any(not pools_of[leg] for leg in legs):
        print("this binary does not carry pools for all six legs", file=sys.stderr)
        return 1

    organs = [] if args.silent_organs else sorted(
        n for n in c["groups"] if n.startswith("organ:"))
    drives = {"descending": args.tone, **{o: args.tone for o in organs}}
    if not args.no_vision:
        drives["sensory_vision"] = 1.0
    if args.drive_premotor:
        drives["premotor:multileg"] = args.drive_premotor

    t0 = time.time()
    _, _, tracked = pp.run(c, args.ms, args.gain, drives, track=order)
    sizes = np.array([max(len(c["members"](p)), 1) for p in order], float)
    rate = tracked / sizes[None, :] * 1000.0                 # Hz per millisecond

    nbins = max(2, args.ms // args.bins)
    usable = (len(rate) // nbins) * nbins
    binned = rate[:usable].reshape(nbins, -1, order.__len__()).sum(1) / (args.ms / nbins)
    traces = {}
    at = 0
    for leg in legs:
        k = len(pools_of[leg])
        traces[leg] = binned[:, at:at + k].mean(1)
        at += k

    w, a, idx, ww, aa = tripod_index(traces, legs)
    print(f"\n{args.bin.name}: {args.ms} ms at gain {args.gain:g}, tone {args.tone:g}"
          + (f", premotor driven at {args.drive_premotor:g}" if args.drive_premotor else "")
          + f"  ({time.time() - t0:.0f} s, {nbins} bins of {args.ms / nbins:.0f} ms)")
    print(f"\n  per-leg pool rate, Hz  " +
          "  ".join(f"{leg} {traces[leg].mean():.1f}" for leg in legs))
    print(f"  within a tripod: mean corr {w:+.3f}   across: {a:+.3f}   "
          f"tripod index {idx:+.3f}")
    print(f"  within-set pairs {np.round(ww, 2).tolist()}")
    print(f"  across-set pairs {np.round(aa, 2).tolist()}")
    if idx > 0.15:
        print("\n  ⇒ the two tripods are two: within-set pairs co-vary more than")
        print("    across-set pairs. A gait can be built from this cord.")
    elif idx < -0.15:
        print("\n  ⇒ the correlation runs the other way: the legs alternate")
        print("    between the sets, i.e. the two tripods are anti-correlated.")
    else:
        print("\n  ⇒ no tripod in the rates: the six legs move as one lump at this")
        print("    operating point. Item 4 then needs the coupling to be *made*")
        print("    (a declared oscillator, PLACEHOLDER, parameter measured by this")
        print("    tool) rather than read off the cord.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
