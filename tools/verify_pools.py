"""
The pools must be in the connectome the app loads.

Step 6 drives the muscles from motor pools and reads them through the app's own
named groups, so the pools have to be in the sidecar the app reads — with the
right size, and none of them empty. This checks the built file, not the
intention: a pool that is missing, or present with zero neurons, is read by the
app as *silence*, which looks exactly like a plausible animal doing nothing.

The counts are cross-checked against the pool predicates applied to the BANC
meta table the same way `build_banc.py` applies them. Counts are order
independent, so this is the check that can be done from the sidecar alone; the
indices themselves are written from the same predicate call that writes the
count, so they cannot drift from it without the count drifting too.
"""
import json
import pathlib
import sys

import numpy as np
import pyarrow.feather as feather

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from motor_pools import pool_groups   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = ROOT / "build" / "flybanc_meta.json"
RAW = ROOT / "data" / "banc"

# build_banc.py's own filter, in build_banc.py's own words.
NOT_NEURONS = {"glia", "not_a_neuron", "trachea"}
COLS = ["root_id", "side", "neuromere", "super_class",
        "body_part_effector", "peripheral_target_type"]


def log(*a):
    print(*a, flush=True)


def main() -> int:
    if not META.exists():
        log(f"no {META} — build the connectome binary first")
        return 1
    sidecar = json.loads(META.read_text())
    groups = {g["name"]: g for g in sidecar.get("groups", [])}
    log(f"{META.name}: {len(groups)} groups, {sidecar['neurons']:,} neurons")

    wanted = pool_groups()
    missing = [n for n, _ in wanted if n not in groups]
    if missing:
        log(f"FAIL  {len(missing)} of {len(wanted)} pool groups are missing "
            f"from the connectome, e.g. {missing[:4]}")
        return 1
    empty = [n for n, _ in wanted if groups[n]["count"] == 0]
    if empty:
        log(f"FAIL  {len(empty)} pool groups are empty, e.g. {empty[:4]}")
        return 1

    m = feather.read_table(f"{RAW}/banc_888_meta.feather",
                           columns=COLS).to_pandas()
    m["super_class"] = m["super_class"].fillna("unknown")
    m = m[~m["super_class"].isin(NOT_NEURONS)]
    m = m.drop_duplicates(subset="root_id", keep="first")
    for c in ("side", "body_part_effector", "peripheral_target_type"):
        m[c] = m[c].fillna("").astype(str)

    bad = 0
    total = 0
    for name, pred in wanted:
        want = int(np.count_nonzero(pred(m).to_numpy()))
        got = int(groups[name]["count"])
        total += got
        if want != got:
            log(f"FAIL  {name}: the sidecar says {got}, the predicate selects "
                f"{want}")
            bad += 1
    if bad:
        return 1

    log(f"ok    all {len(wanted)} pools present, {total} pool neurons, none "
        f"empty, every count agrees with the predicate")
    by_leg = {}
    for name, _ in wanted:
        leg = name.split(":")[1]
        by_leg.setdefault(leg, 0)
        by_leg[leg] += groups[name]["count"]
    log("      per leg: " + "  ".join(f"{k} {v}"
                                      for k, v in sorted(by_leg.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
