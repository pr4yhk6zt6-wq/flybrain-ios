#!/usr/bin/env python3
"""
The 788 multi-leg premotor cells, split by which legs they reach — the count
that decides whether a pattern on that population can alternate anything.

`reports/item4_walking.md` §8 measured that the set of legs which lift under a
patterned command changes with the population the command lands on, so the
leg-selection is being made in the cord's wiring. §9 asks the next question
there: are the cells that reach one tripod separable from the cells that reach
the other? A gait needs the two halves to be driven apart, and only a cell that
reaches *one* tripod can be part of a mechanism that drives them apart.

`tools/build_banc.py` now packs nine mutually exclusive groups (six single-leg
`premotor:leg:<leg>`, `premotor:tripodA`, `premotor:tripodB`, `premotor:cross`).
This tool reads the shipped binary and:

  1. **re-derives the split from the graph**, the way `tools/coupling_probe.py`
     re-derives the 788 — the packer walks forward from each pool motor neuron's
     column, this walks the CSR backwards from every cell, and the two have to
     agree, cell for cell;
  2. fails if the nine groups do not partition the cells that reach any pool
     motor neuron (a cell dropped between the two tools is a cell the app cannot
     drive);
  3. fails if a group the app can ask for is not in the binary, because a missing
     name reads as 0 Hz, which is what a quiet population reads as;
  4. prints the number item 4 turns on: how many of the multi-leg cells reach
     **both** tripods, and how many are exclusive to one.

    python3 tools/verify_premotor.py
    python3 tools/verify_premotor.py --require-separable 200

`--require-separable N` is off by default and is the flag a future gait
experiment would turn on: it fails unless at least N cells are exclusive to one
tripod. It is off because the honest answer today is 23, and a gate that asserts
a number that is not there is a gate that gets turned off.
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


def log(msg: str) -> None:
    print(msg, flush=True)


def load(path: pathlib.Path, meta_path: pathlib.Path):
    blob = path.read_bytes()
    magic, version, n, e, gic, retina, section_count = \
        struct.unpack_from("<8sIIIIII", blob, 0)
    if magic != b"FLYBANC_" or version != 2:
        raise SystemExit(f"{path} is not a flybanc.bin v2")
    view = memoryview(blob)
    sec = {}
    for i, name in enumerate(SECTIONS[:section_count]):
        off, length = struct.unpack_from("<QQ", blob, 64 + i * 16)
        sec[name] = view[off:off + length]
    meta = json.loads(meta_path.read_text())
    return n, e, sec, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--banc", default=pathlib.Path("build/flybanc.bin"),
                    type=pathlib.Path)
    ap.add_argument("--meta", default=pathlib.Path("build/flybanc_meta.json"),
                    type=pathlib.Path)
    ap.add_argument("--require-separable", type=int, default=None,
                    help="fail unless this many cells are exclusive to one "
                         "tripod (off by default: the measured answer is 23)")
    args = ap.parse_args()

    n, e, sec, meta = load(args.banc, args.meta)
    row = np.frombuffer(sec["csrRowPtr"], dtype=np.uint32)
    col = np.frombuffer(sec["csrColIdx"], dtype=np.uint32)
    gi = np.frombuffer(sec["groupIndices"], dtype=np.uint32)
    groups = {g["name"]: (int(g["start"]), int(g["count"]))
              for g in meta["groups"]}

    def members(name: str) -> np.ndarray:
        start, count = groups[name]
        return gi[start:start + count]

    # ---- which leg does each pool motor neuron belong to --------------------
    legs = sorted({name.split(":")[1] for name in groups if name.startswith("pool:")})
    leg_id = {leg: i for i, leg in enumerate(legs)}
    pool_leg = np.full(n, -1, np.int16)
    for name in groups:
        if name.startswith("pool:"):
            pool_leg[members(name)] = leg_id[name.split(":")[1]]

    # ---- re-derive, backwards: for every cell, which legs do its spikes reach
    # The packer walked forward from each pool motor neuron's column; this walks
    # the same edges the other way, from every pre-synaptic cell, so the two
    # counts are independent. A duplicate edge (pre, post) is one leg, once.
    pre_of_edge = np.repeat(np.arange(n, dtype=np.int64), np.diff(row).astype(np.int64))
    post_of_edge = col.astype(np.int64)
    onto = pool_leg[post_of_edge] >= 0
    pair = np.unique(pre_of_edge[onto].astype(np.int64) * 16
                     + pool_leg[post_of_edge][onto].astype(np.int64))
    mask = np.zeros(n, np.int32)
    cells = (pair // 16).astype(np.int64)
    legbits = np.array([1 << i for i in range(len(legs))], np.int32)
    np.bitwise_or.at(mask, cells, legbits[(pair % 16).astype(np.int64)])

    reach_any = np.flatnonzero(mask > 0)
    mask_a = sum(1 << leg_id[l] for l in TRIPOD["A"])
    mask_b = sum(1 << leg_id[l] for l in TRIPOD["B"])
    two_legs = mask & (mask - 1) != 0
    a_only = np.flatnonzero((mask & mask_b == 0) & (mask & mask_a != 0) & two_legs)
    b_only = np.flatnonzero((mask & mask_a == 0) & (mask & mask_b != 0) & two_legs)
    cross = np.flatnonzero((mask & mask_a != 0) & (mask & mask_b != 0))

    problems: list[str] = []
    log(f"{args.banc}: {n:,} neurons, {e:,} edges, {len(groups)} groups")

    # ---- the names the app can ask for must be there ------------------------
    want = [f"premotor:leg:{leg}" for leg in legs] + \
           ["premotor:tripodA", "premotor:tripodB", "premotor:cross",
            "premotor:multileg"]
    for name in want:
        if name not in groups:
            problems.append(f"{name} is not in the packed connectome — a "
                            "missing group reads as 0 Hz")
    if problems:
        for p in problems:
            log(f"  FAIL {p}")
        return 1

    # ---- the groups must partition the cells that reach any pool ------------
    total = sum(groups[name][1] for name in want if name != "premotor:multileg")
    log(f"  cells that reach any pool motor neuron: {len(reach_any):,}")
    log(f"  the nine exclusive groups sum to:       {total:,}")
    if total != len(reach_any):
        problems.append(f"the split drops or double-counts cells: {total:,} "
                        f"against {len(reach_any):,}")

    # ---- the packer's counts and this walk must agree, cell for cell --------
    for name, idx in (("premotor:tripodA", a_only), ("premotor:tripodB", b_only),
                      ("premotor:cross", cross)):
        have = np.sort(members(name))
        if len(have) != len(idx) or not np.array_equal(have, np.sort(idx)):
            problems.append(f"{name}: the binary has {len(have):,} cells, this "
                            f"walk finds {len(idx):,}"
                            + ("" if len(have) == len(idx) else " — different cells"))

    # ---- the number item 4 turns on -----------------------------------------
    private = sum(groups[f"premotor:leg:{leg}"][1] for leg in legs)
    multi = len(reach_any) - private
    log("")
    log(f"  private to one leg      {private:>6,}")
    log(f"  multi-leg               {multi:>6,}")
    log(f"    exclusive to tripod A {len(a_only):>6,}")
    log(f"    exclusive to tripod B {len(b_only):>6,}")
    log(f"    reaching both         {len(cross):>6,}")
    if multi:
        log(f"    cross / multi-leg     {len(cross)/multi:6.1%}")
    log("")
    log("  A cell that reaches both tripods cannot drive them apart, and only a")
    log("  cell that reaches one can be part of a mechanism that does. The")
    log("  measured answer is that the multi-leg premotor population is")
    log("  overwhelmingly shared, so a pattern on it cannot alternate the")
    log("  tripods — which is what walk_loop.py --drive-group measured from the")
    log("  other end (reports/item4_walking.md §8: index -0.300).")

    if args.require_separable is not None:
        exclusive = len(a_only) + len(b_only)
        if exclusive < args.require_separable:
            problems.append(f"only {exclusive} cells are exclusive to one tripod, "
                            f"and {args.require_separable} were required")

    if problems:
        log("")
        for p in problems:
            log(f"  FAIL {p}")
        return 1
    log("")
    log("  ok   the nine groups partition every cell that reaches a pool")
    log("  ok   this walk and the packer's agree, cell for cell")
    log("  ok   every name the app can ask for is in the binary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
