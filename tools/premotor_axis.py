#!/usr/bin/env python3
"""
The other axis: split the premotor cells by the *muscles* they reach, not the
legs — and measure which muscle groups deliver an asymmetric command to the two
tripods.

`reports/item4_walking.md` §8 measured that the legs which lift under a
patterned command change with the population the command lands on, and §9 then
measured *why* the obvious lever is missing: of the 788 multi-leg premotor
cells, 765 reach both tripods, and a cell that reaches both cannot drive them
apart. That closes one axis (legs) and leaves the other one open. The cord's
premotor population is not organised only by leg: each leg's motor neurons are
grouped by *muscle* — `pool:T2_right:tibia_flexor` and `pool:T2_right:tibia_extensor`
are different targets, and a cell that reaches one is not the same cell as one
that reaches the other. If the legs are not separable but the muscles are, then
the substrate for a gait is a muscle axis, not a leg axis, and the alternation
would be produced by driving an extensor-like population against a flexor-like
one — the way the animal's own stance and swing phases are produced.

This tool measures that without assuming any of it. For each muscle group it
takes the population of premotor cells that reach that muscle *anywhere* (which
is what a group named `premotor:muscle:<m>` would contain) and adds up the
synaptic weight that population delivers onto the pool motor neurons of each
tripod. The asymmetry

    (weight onto tripod A - weight onto tripod B) / (their sum)

is the number that decides whether driving it can move the two halves apart at
all: at 0 the command is common-mode and the two tripods cannot be separated by
it no matter how it is patterned; at +1 or -1 it is entirely one half. A gait
needs *two* populations with opposite signs, so the tool also prints the most
opposed pair it finds — the lever a pattern could alternate, if one exists.

The muscle names come from the pools themselves (there are ten per leg:
coxa_rotator_ant/post, trochanter_flexor/extensor, femur_reductor, tibia_flexor/
extensor, tarsus_levator/depressor, long_tendon). Which of them lifts a leg and
which pushes it is a question for the body model — `tools/stand_body.py` and the
MuJoCo model are where that is measured — and this tool deliberately does not
answer it. It only measures which of them would deliver a command to one tripod
and not the other, which is a property of the connectome alone.

    python3 tools/premotor_axis.py
    python3 tools/premotor_axis.py --banc build/flybanc.bin --json tmp/premotor_axis.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

SECTIONS = ["rootIDs", "positions", "neuronMeta", "csrRowPtr", "csrColIdx",
            "csrWeight", "csrDelay", "groupIndices", "retinaUV"]
TRIPOD = {"A": ("T1_left", "T2_right", "T3_left"),
          "B": ("T1_right", "T2_left", "T3_right")}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load(path: pathlib.Path, meta_path: pathlib.Path):
    blob = path.read_bytes()
    magic, version, n, e, gic, retina, section_count = \
        struct.unpack_from("<8sIIIIII", blob, 0)
    if magic != b"FLYBANC_" or version != 2:
        sys.exit(f"{path} is not a flybanc.bin v2")
    view = memoryview(blob)
    sec = {}
    for i, name in enumerate(SECTIONS[:section_count]):
        off, length = struct.unpack_from("<QQ", blob, 64 + i * 16)
        sec[name] = view[off:off + length]
    meta = json.loads(meta_path.read_text())
    return n, e, sec, meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--banc", type=pathlib.Path,
                    default=pathlib.Path("build/flybanc.bin"))
    ap.add_argument("--meta", type=pathlib.Path,
                    default=pathlib.Path("build/flybanc_meta.json"))
    ap.add_argument("--json", type=pathlib.Path, default=None,
                    help="write the tables here as json")
    ap.add_argument("--legs", nargs="*", default=[],
                    help="muscle groups to break down leg by leg")
    ap.add_argument("--top", type=int, default=6,
                    help="how many rows of each table to print")
    args = ap.parse_args()

    n, e, sec, meta = load(args.banc, args.meta)
    row = np.frombuffer(sec["csrRowPtr"], dtype=np.uint32)
    col = np.frombuffer(sec["csrColIdx"], dtype=np.uint32).astype(np.int64)
    # The packed weight is half precision on disk (tools/build_banc.py writes
    # `weight.astype(np.float16)`), so the sums below are in that rule's units:
    # sign(NT) * log1p(syn_count) / median_drive. They are compared with each
    # other and never with a synapse count.
    wt = np.frombuffer(sec["csrWeight"], dtype=np.float16)
    gi = np.frombuffer(sec["groupIndices"], dtype=np.uint32)
    groups = {g["name"]: (int(g["start"]), int(g["count"]))
              for g in meta["groups"]}

    def members(name: str) -> np.ndarray:
        start, count = groups[name]
        return gi[start:start + count].astype(np.int64)

    log(f"{args.banc}: {n:,} neurons, {e:,} edges, {len(groups)} groups")

    # ---- the pool motor neurons, by leg and by muscle ----------------------
    pool_names = sorted(name for name in groups if name.startswith("pool:"))
    legs = sorted({name.split(":")[1] for name in pool_names})
    muscles = sorted({name.split(":")[2] for name in pool_names})
    leg_of = {leg: i for i, leg in enumerate(legs)}
    mus_of = {m: i for i, m in enumerate(muscles)}

    pool_leg = np.full(n, -1, np.int16)
    pool_mus = np.full(n, -1, np.int16)
    for name in pool_names:
        _, leg, mus = name.split(":")
        pool_leg[members(name)] = leg_of[leg]
        pool_mus[members(name)] = mus_of[mus]

    tripod = np.zeros(len(legs), np.int8)          # +1 A, -1 B, 0 neither
    for leg in TRIPOD["A"]:
        tripod[leg_of[leg]] = +1
    for leg in TRIPOD["B"]:
        tripod[leg_of[leg]] = -1

    # Two control partitions, because a muscle group that biases one tripod
    # could be biasing something correlated with it instead: the sides, or the
    # segments. A number that only shows up under the tripod split is a number
    # about the tripod split; a number that shows up under all three is about
    # the animal's left-right or front-back gradient and says nothing about a
    # gait. Both controls are computed and printed, and the reading of the main
    # table is only as good as the difference between them.
    def partition(pairs) -> np.ndarray:
        sign = np.zeros(len(legs), np.int8)
        for leg, s in pairs.items():
            sign[leg_of[leg]] = s
        return sign

    SIDE = partition({**{f"T{i}_left": +1 for i in (1, 2, 3)},
                      **{f"T{i}_right": -1 for i in (1, 2, 3)}})
    SEGMENT = partition({**{f"T1_{s}": +1 for s in ("left", "right")},
                         **{f"T3_{s}": -1 for s in ("left", "right")}})

    log(f"  {len(pool_names)} pools: {len(legs)} legs x {len(muscles)} muscles")

    # ---- every edge that lands on a pool motor neuron ----------------------
    # `pre_of_edge` is the same construction tools/verify_premotor.py uses: two
    # walks of the same CSR, and this one is the third independent one.
    pre_of_edge = np.repeat(np.arange(n, dtype=np.int64),
                            np.diff(row).astype(np.int64))
    onto = pool_leg[col] >= 0
    pre_e = pre_of_edge[onto]
    post_e = col[onto]
    w_e = wt[onto].astype(np.float64)
    leg_e = pool_leg[post_e]
    mus_e = pool_mus[post_e]
    tri_e = tripod[leg_e]

    log(f"  {len(pre_e):,} edges land on a pool motor neuron "
        f"({w_e.sum():,.0f} synapses)")

    # ---- which cells reach any pool, and on how many legs ------------------
    legbit = np.zeros(n, np.int32)
    np.bitwise_or.at(legbit, pre_e,
                     (np.int32(1) << leg_e.astype(np.int32)))
    n_legs = np.array([bin(int(b)).count("1") for b in legbit], np.int8)
    reach_any = np.flatnonzero(legbit > 0)
    multi = reach_any[n_legs[reach_any] > 1]
    log(f"  cells that reach any pool: {len(reach_any):,}  "
        f"(multi-leg: {len(multi):,})")

    # ---- the asymmetry of a driven population ------------------------------
    def asymmetry(driven: np.ndarray, sign_e: np.ndarray = None) -> dict:
        """The *net* command a population delivers to each half of a partition.

        The packed weight is signed by the presynaptic transmitter
        (`tools/build_banc.py`: `sign(NT) * log1p(syn_count) / median_drive`),
        so these are net sums — excitation minus inhibition — and not shares of
        a positive total. The first version of this tool divided them as if
        they were magnitudes and printed -408% for a ratio that cannot leave
        [-1, 1]; the denominator here is `|net_A| + |net_B|`, which keeps the
        index bounded and reads 0 when the two halves get the same net drive.
        """
        sign_e = tri_e if sign_e is None else sign_e
        mask = np.zeros(n, bool)
        mask[driven] = True
        sel = mask[pre_e]
        a = w_e[sel & (sign_e > 0)].sum()
        b = w_e[sel & (sign_e < 0)].sum()
        denom = abs(a) + abs(b)
        return {"cells": int(len(driven)), "onto_A": float(a),
                "onto_B": float(b),
                "asym": float((a - b) / denom) if denom else 0.0}

    def pct(x: float) -> str:
        return f"{100 * x:+.1f}%"

    def net(a: float, b: float) -> str:
        """The two halves' net commands, in the weight rule's own units."""
        return f"{a:+.0f} vs {b:+.0f}"

    # ---- 1. the muscle axis over every cell that reaches a pool ------------
    rows = []
    for m in muscles:
        driven = np.unique(pre_e[mus_e == mus_of[m]])
        r = asymmetry(driven)
        r["muscle"] = m
        rows.append(r)
    rows.sort(key=lambda r: -abs(r["asym"]))

    log("")
    log("  by muscle group — the population that reaches this muscle, anywhere")
    log(f"    {'muscle':<20} {'cells':>6} {'net A vs net B':>22} {'asym':>8}")
    for r in rows:
        log(f"    {r['muscle']:<20} {r['cells']:>6,} "
            f"{net(r['onto_A'], r['onto_B']):>22} {pct(r['asym']):>8}")

    # ---- 1b. the same populations under the two control partitions ---------
    # `side` is the tripod split with the left and right halves swapped onto
    # one another (all three left legs against all three right legs), and
    # `segment` is front against back. If a muscle group's tripod number is
    # really a side number or a segment number, it shows up here too, and the
    # tables above are a fact about the fly's anatomy rather than a lever for a
    # gait. Printed in the same order as the main table, so the columns can be
    # read across.
    log("")
    log("  the same populations, split by two control partitions of the legs")
    log("  (a number that is really about left-right or front-back shows up "
        "here too)")
    log(f"    {'muscle':<20} {'tripod':>9} {'side':>9} {'segment':>9}")
    for r in rows:
        m = mus_of[r["muscle"]]
        drv = np.unique(pre_e[mus_e == m])
        side = asymmetry(drv, SIDE[leg_e])["asym"]
        seg = asymmetry(drv, SEGMENT[leg_e])["asym"]
        log(f"    {r['muscle']:<20} {pct(r['asym']):>9} {pct(side):>9} "
            f"{pct(seg):>9}")
        r["asym_side"] = side
        r["asym_segment"] = seg

    # ---- 1c. the candidates, leg by leg ------------------------------------
    # The tripod key mixes sides and segments (A is T1_left, T2_right, T3_left),
    # so a tripod number can be produced by a pure segment effect with the legs
    # of one segment disagreeing. This prints, for the groups that matter, the
    # net command to each leg's pools, where that shows up.
    if args.legs:
        for name in args.legs:
            if name not in mus_of:
                log(f"    ({name} is not a muscle in this binary)")
                continue
            drv = np.unique(pre_e[mus_e == mus_of[name]])
            mask = np.zeros(n, bool)
            mask[drv] = True
            sel = mask[pre_e]
            log("")
            log(f"  {name}: {len(drv):,} cells that reach it, net command "
                f"per leg")
            for leg in legs:
                m = sel & (leg_e == leg_of[leg])
                log(f"    {leg:<10} {w_e[m].sum():+8.1f}  over {int(m.sum()):>5} "
                    f"edges from {len(np.unique(pre_e[m])):>5} cells"
                    + ("   (tripod A)" if tripod[leg_of[leg]] > 0 else
                       "   (tripod B)"))

    # ---- 2. the same, restricted to the 788 multi-leg cells ----------------
    multi_mask = np.zeros(n, bool)
    multi_mask[multi] = True
    rows_multi = []
    for m in muscles:
        driven = np.unique(pre_e[(mus_e == mus_of[m]) & multi_mask[pre_e]])
        r = asymmetry(driven)
        r["muscle"] = m
        rows_multi.append(r)
    rows_multi.sort(key=lambda r: -abs(r["asym"]))

    log("")
    log("  the same, restricted to the 788 multi-leg cells (the population "
        "§8/§9 measured)")
    log(f"    {'muscle':<20} {'cells':>6} {'net A vs net B':>22} {'asym':>8}")
    for r in rows_multi:
        log(f"    {r['muscle']:<20} {r['cells']:>6,} "
            f"{net(r['onto_A'], r['onto_B']):>22} {pct(r['asym']):>8}")

    # ---- 3. calibration: the groups that have already been driven ----------
    log("")
    log("  calibration — groups with a measured gait number "
        "(reports/item4_walking.md §6-§8)")
    calib = {}
    for name in ("descending", "premotor:multileg", "premotor:cross",
                 "premotor:tripodA", "premotor:tripodB"):
        if name not in groups:
            continue
        r = asymmetry(members(name))
        calib[name] = r
        log(f"    {name:<20} {r['cells']:>6,} "
            f"{net(r['onto_A'], r['onto_B']):>22} {pct(r['asym']):>8}")

    # ---- 4. is there a pair to alternate? ----------------------------------
    # Two populations can alternate the tripods only if driving one pushes the
    # weight toward A and driving the other pushes it toward B: opposite signs
    # and a magnitude to go with them. This is the specific thing item 4 is
    # looking for, so it is printed as a decision, not as a table.
    best = None
    for i, ra in enumerate(rows):
        for rb in rows[i + 1:]:
            if ra["asym"] * rb["asym"] >= 0:
                continue
            # how far apart the two commands would put the halves
            sep = abs(ra["asym"] - rb["asym"])
            score = sep * min(abs(ra["asym"]), abs(rb["asym"]))
            if best is None or score > best[0]:
                best = (score, ra, rb, sep)
    log("")
    if best is None:
        log("  No muscle group delivers an asymmetric command to the two "
            "tripods at all:")
        log("  every population above is common-mode, so no pair of them can "
            "alternate")
        log("  the tripods. On this axis the cord has no lever either.")
    else:
        score, ra, rb, sep = best
        log(f"  The most opposed pair on this axis: {ra['muscle']} "
            f"({pct(ra['asym'])}) against {rb['muscle']} ({pct(rb['asym'])})")
        log(f"  -> driving them alternately moves the command balance between "
            f"the tripods by {100 * sep:.1f} points.")
        log("     That is a lever, and it is in the connectome — the next "
            "measurement is to")
        log("     pack it as `premotor:muscle:<name>` and drive it with "
            "walk_loop.py.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"muscle": rows, "muscle_multi": rows_multi, "calibration": calib,
             "legs": legs, "muscles": muscles,
             "best_pair": None if best is None else
             {"separation": best[3], "a": best[1], "b": best[2]}},
            indent=1))
        log("")
        log(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
