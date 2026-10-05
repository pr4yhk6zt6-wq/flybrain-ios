#!/usr/bin/env python3
"""
The descending neurons have to actually reach the leg's motor pools.

The cord's tone is the brain's, and `FlyCord` now puts it where the reference
puts it: on the descending population (`tools/step3_closedloop.py` injects
`args.desc` into `net.desc_idx` on every millisecond of every mode). That is a
statement about the *shipped* connectome, not about the source database, so it
is checked here, in `build/flybanc.bin` — the exact artifact that is copied into
the app bundle:

    python3 tools/verify_descending.py

It fails if the group is absent, if the app's own name for it has drifted from
the name in the binary, or if the 1,316 descending cells do not synapse onto the
motor neurons the pools are made of. Two-hop reachability is reported as well:
one synapse is a projection, two is a pathway, and the brief's item 9 is about
the pathway.

Reads `build/flybanc.bin` (v2) the way `FlyBrain/Sources/Connectome.swift` does:
a 64-byte header, then a table of (offset, length) per section, 256-byte aligned.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys

import numpy as np

SECTIONS = ["rootIDs", "positions", "neuronMeta", "csrRowPtr", "csrColIdx",
            "csrWeight", "csrDelay", "groupIndices", "retinaUV"]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_banc(path: pathlib.Path, meta_path: pathlib.Path):
    blob = path.read_bytes()
    magic, version, n, e, group_index_count, retina, section_count = \
        struct.unpack_from("<8sIIIIII", blob, 0)
    if magic != b"FLYBANC_" or version != 2:
        raise SystemExit(f"{path} is not a flybanc.bin v2 (magic {magic!r}, v{version})")
    view = memoryview(blob)
    sections = {}
    for i, name in enumerate(SECTIONS[:section_count]):
        off, length = struct.unpack_from("<QQ", blob, 64 + i * 16)
        sections[name] = view[off:off + length]
    meta = json.loads(meta_path.read_text())
    return n, e, sections, meta, group_index_count, retina


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--banc", default=pathlib.Path("build/flybanc.bin"), type=pathlib.Path)
    ap.add_argument("--meta", default=pathlib.Path("build/flybanc_meta.json"), type=pathlib.Path)
    ap.add_argument("--cord", default=pathlib.Path("FlyBrain/Sources/FlyCord.swift"),
                    type=pathlib.Path)
    ap.add_argument("--min-direct", type=int, default=1000,
                    help="descending -> pool synapses required (measured 3,283)")
    ap.add_argument("--min-covered", type=float, default=0.9,
                    help="fraction of pool motor neurons reachable within two hops")
    args = ap.parse_args()

    n, e, sec, meta, group_index_count, retina = load_banc(args.banc, args.meta)
    row = np.frombuffer(sec["csrRowPtr"], dtype=np.uint32)
    col = np.frombuffer(sec["csrColIdx"], dtype=np.uint32)
    gi = np.frombuffer(sec["groupIndices"], dtype=np.uint32)
    groups = {g["name"]: (int(g["start"]), int(g["count"])) for g in meta["groups"]}
    log(f"{args.banc}: {n:,} neurons, {e:,} edges, {len(groups)} groups, "
        f"{group_index_count:,} group indices, {retina:,} retina UVs")

    def members(name: str) -> np.ndarray:
        start, count = groups[name]
        return gi[start:start + count].astype(np.int64)

    # --- the app's own name for the group must be the binary's ---------------
    cord = args.cord.read_text()
    m = re.search(r"descendingGroup:\s*String\s*=\s*\"([^\"]+)\"", cord)
    if not m:
        log(f"FAIL {args.cord} does not declare a descendingGroup default")
        return 1
    app_name = m.group(1)
    if app_name not in groups:
        log(f"FAIL FlyCord's descendingGroup is {app_name!r}, which is not a group "
            f"in {args.meta.name}")
        return 1
    desc = members(app_name)
    log(f"descending group {app_name!r}: {len(desc):,} cells "
        f"(FlyCord.swift declares the same name)")

    # --- the motor neurons the pools are made of ----------------------------
    pools = sorted(g for g in groups if g.startswith("pool:"))
    pool_cells = np.concatenate([members(p) for p in pools]) if pools else np.zeros(0, np.int64)
    pool_set = np.zeros(n, dtype=bool)
    pool_set[pool_cells] = True
    if pool_cells.size == 0:
        log("FAIL the binary carries no pool: groups at all")
        return 1

    # --- one hop -------------------------------------------------------------
    start, end = row[desc], row[desc + 1]
    out = np.concatenate([col[s:t] for s, t in zip(start, end)]) if len(desc) else \
        np.zeros(0, dtype=np.uint32)
    direct = int(pool_set[out].sum())
    log(f"  {len(out):,} synapses out of the descending cells "
        f"({len(out) / max(1, len(desc)):.1f} per cell)")
    log(f"  into the {pool_cells.size} pool motor neurons, directly: {direct:,}")

    # --- two hops ------------------------------------------------------------
    targets = np.unique(out.astype(np.int64))
    relay = targets[~pool_set[targets]]
    reached = set(np.unique(out[pool_set[out]]).tolist())
    edges2 = 0
    st, en = row[relay], row[relay + 1]
    for s, t in zip(st, en):
        if t > s:
            row_targets = col[s:t]
            hit = pool_set[row_targets]
            if hit.any():
                edges2 += int(hit.sum())
                reached.update(row_targets[hit].tolist())
    covered = len(reached) / pool_cells.size
    log(f"  through {relay.size:,} interneurons, two hops: {edges2:,} synapses, "
        f"{len(reached)}/{pool_cells.size} pool motor neurons reachable "
        f"({covered * 100:.1f}%)")

    # --- every leg's pools have to be reachable -----------------------------
    legs = {}
    for p in pools:
        legs.setdefault(p.split(":")[1], []).extend(members(p).tolist())
    worst = None
    for leg, cells in sorted(legs.items()):
        cells = np.array(sorted(set(cells)))
        hit = np.isin(cells, np.array(sorted(reached)))
        leg_edges = int(np.isin(cells, out).sum())
        if worst is None or hit.mean() < worst[1]:
            worst = (leg, float(hit.mean()), leg_edges)
        log(f"  {leg:<10} {hit.sum():>3}/{cells.size:<3} motor neurons reachable, "
            f"{leg_edges:>4} direct descending synapses")

    ok = True
    if direct < args.min_direct:
        log(f"FAIL only {direct:,} descending -> pool synapses "
            f"(need {args.min_direct:,}): the tone would be injected into cells "
            "that do not talk to the leg's motor neurons")
        ok = False
    if covered < args.min_covered:
        log(f"FAIL only {covered * 100:.1f}% of the pool motor neurons are within "
            f"two hops of a descending cell (need {args.min_covered * 100:.0f}%)")
        ok = False
    if worst and worst[1] < args.min_covered:
        log(f"FAIL {worst[0]}'s pools are {worst[1] * 100:.1f}% reachable "
            f"(need {args.min_covered * 100:.0f}%)")
        ok = False
    log("OK — the descending cells reach every leg's motor pools" if ok
        else "the loop's brain input is not connected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
