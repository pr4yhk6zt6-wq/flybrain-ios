#!/usr/bin/env python3
"""
Do the motor pools fire under the configuration the app actually runs?

The device build behind `uploads/IMG_2713.png` printed
`0/42 pools firing · 0.0 Hz · organs campaniform+chordotonal · desc 2.5` while
its cord was driving the descending population every millisecond. That is a
statement about the loop's *input*, and it can be checked off-device: this runs
the same LIF the app's kernel reproduces (`tools/verify_banc.py`'s NumPy
reference) with the app's own numbers, and reports what the 42 pool groups do.

    python3 tools/pool_probe.py                     # every condition below
    python3 tools/pool_probe.py --ms 300 --gain 6

Conditions, because the difference between them is the whole point:

  app          gain 6.0, descending 2.5, organs 2.5, as the cord drives them
               (FlyCord.update → setGroupDrive), vision silent
  app x1.5     the same, with the drive that `driveGroups` builds — the kernel
               computes `I = iSyn*gain + externalInput*P.externalDrive + noise`
               and externalDrive is 1.5, so a group drive of 2.5 arrives as 3.75
  reference    gain 12.0, descending 2.5 — `tools/step3_closedloop.py --desc 2.5
               --n-star 12`, the configuration the loop's law was measured in
  vision       gain 12.0, descending 2.5, and the photoreceptors driven at 1.5,
               i.e. the whole sensorimotor path the brief asks about

The numbers to look for: `pool Hz mean`, `pools firing` (the HUD's n/42, a pool
counts as firing at ≥ 1 Hz after the cord's 60 ms muscle filter), and
`descending Hz` — a tone of 2.5 that does not light up the cells it is injected
into is a scaling mistake, not a silent animal.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time

import numpy as np

SECTIONS = ["rootIDs", "positions", "neuronMeta", "csrRowPtr", "csrColIdx",
            "csrWeight", "csrDelay", "groupIndices", "retinaUV"]


def load(bin_path: pathlib.Path, meta_path: pathlib.Path):
    buf = np.memmap(bin_path, dtype=np.uint8, mode="r")
    magic, version, n, e, n_group_idx, n_vision, n_sections = \
        struct.unpack_from("<8sIIIIII", buf, 0)
    assert magic == b"FLYBANC_" and version == 2, (magic, version)
    sec = {}
    for i, name in enumerate(SECTIONS):
        off, length = struct.unpack_from("<QQ", buf, 64 + i * 16)
        sec[name] = (off, length)

    def view(name, dtype, cols=1):
        off, length = sec[name]
        a = np.frombuffer(buf, dtype=dtype, count=length // np.dtype(dtype).itemsize,
                          offset=off)
        return a.reshape(-1, cols) if cols > 1 else a

    meta = json.loads(meta_path.read_text())
    groups = {g["name"]: g for g in meta["groups"]}
    group_idx = view("groupIndices", np.uint32)

    def members(name):
        g = groups[name]
        return group_idx[g["start"]:g["start"] + g["count"]].astype(np.int64)

    return dict(n=n, e=e, N=n,
                rowPtr=view("csrRowPtr", np.uint32),
                colIdx=view("csrColIdx", np.uint32),
                weight=view("csrWeight", np.float16).astype(np.float32),
                delay=view("csrDelay", np.uint8),
                meta=meta, groups=groups, members=members)


def run(c, ms, gain, drives, seed=42, report_every=0):
    """The LIF of tools/verify_banc.py, with an arbitrary drive per group."""
    N, rowPtr, colIdx, W, delay = c["N"], c["rowPtr"], c["colIdx"], c["weight"], c["delay"]
    DT, TAU_M, TAU_SYN = 1.0, 20.0, 5.0
    V_TH, T_REF, NOISE = 1.0, 2, 0.015
    rng = np.random.default_rng(seed)
    V = rng.uniform(0, 0.5, N).astype(np.float32)
    I_syn = np.zeros(N, np.float32)
    refrac = np.zeros(N, np.int16)
    max_d = int(delay.max()) + 1
    ring = np.zeros((max_d, N), np.float32)

    I_ext = np.zeros(N, np.float32)
    for group, value in drives.items():
        I_ext[c["members"](group)] = value

    dm, ds = np.exp(-DT / TAU_M), np.exp(-DT / TAU_SYN)
    spikes = np.zeros(N, np.int64)
    lag = 50                                   # ignore the start-up transient
    window = np.zeros(N, np.int64)
    for t in range(ms):
        slot = t % max_d
        I_syn = I_syn * ds + ring[slot]
        ring[slot] = 0
        I = I_syn * gain + I_ext + rng.normal(0, NOISE, N).astype(np.float32)
        free = refrac <= 0
        V[free] = V[free] * dm + I[free] * (1 - dm)
        refrac[~free] -= 1
        fired = np.flatnonzero((V >= V_TH) & free)
        V[fired] = 0.0
        refrac[fired] = T_REF
        spikes[fired] += 1
        if t >= ms - lag:
            window[fired] += 1
        if len(fired):
            s0 = rowPtr[fired].astype(np.int64)
            s1 = rowPtr[fired + 1].astype(np.int64)
            if (s1 - s0).sum():
                e = np.concatenate([np.arange(a, b) for a, b in zip(s0, s1)])
                np.add.at(ring, ((t + delay[e].astype(np.int64)) % max_d,
                                 colIdx[e]), W[e])
        if report_every and t % report_every == 0:
            print(f"    t={t:4d} fired={len(fired):6,}", flush=True)
    return spikes, window


def hz(c, counts, name, ms):
    idx = c["members"](name)
    return float(counts[idx].sum()) / max(len(idx), 1) / (ms / 1000.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=pathlib.Path("build/flybanc.bin"), type=pathlib.Path)
    ap.add_argument("--meta", default=pathlib.Path("build/flybanc_meta.json"), type=pathlib.Path)
    ap.add_argument("--ms", type=int, default=300)
    ap.add_argument("--only", default=None, help="run one condition by name")
    args = ap.parse_args()

    c = load(args.bin, args.meta)
    meta = c["meta"]
    # The 42 pools `FlyCord` actually carries: the body asset places seven per
    # leg, and a pool it cannot place is not in the loop at all.
    cord_pools = []
    asset_path = pathlib.Path("build/fly_body.json")
    if asset_path.exists():
        asset = json.loads(asset_path.read_text())
        for leg in asset.get("legs", {}).values():
            for pool in leg.get("pools", {}).values():
                g = pool.get("group")
                if g:
                    cord_pools.append(g)
        cord_pools = sorted(set(cord_pools))
    organs = [g["name"] for g in meta["groups"] if g["name"].startswith("organ:")]
    pools = [g["name"] for g in meta["groups"] if g["name"].startswith("pool:")]
    print(f"{args.bin}: {c['N']:,} neurons, {len(pools)} pools, {len(organs)} organ groups")

    conditions = {
        "app": dict(gain=6.0, drives={"descending": 2.5,
                                      **{o: 2.5 for o in organs}}),
        "app x1.5": dict(gain=6.0, drives={"descending": 2.5 * 1.5,
                                           **{o: 2.5 * 1.5 for o in organs}}),
        "reference": dict(gain=12.0, drives={"descending": 2.5,
                                             **{o: 2.5 for o in organs}}),
        "vision": dict(gain=12.0, drives={"descending": 2.5,
                                          **{o: 2.5 for o in organs},
                                          "sensory_vision": 1.5}),
        # the device's own configuration: the app's default gain, the drive as
        # the kernel builds it (organ/descending × externalDrive), and the flat
        # retinal drive BrainEngine sets at load when there is no camera
        "device": dict(gain=6.0, drives={"descending": 2.5 * 1.5,
                                         **{o: 2.5 * 1.5 for o in organs},
                                         "sensory_vision": 1.0}),
    }

    rows = []
    for name, cfg in conditions.items():
        if args.only and name != args.only:
            continue
        t0 = time.time()
        spikes, window = run(c, args.ms, cfg["gain"], cfg["drives"],
                             report_every=0)
        pool_hz = np.array([hz(c, window, p, args.ms) for p in pools])
        firing = int((pool_hz >= 1.0).sum())
        row = dict(
            condition=name, gain=cfg["gain"],
            descending=hz(c, window, "descending", args.ms),
            organ_mean=float(np.mean([hz(c, window, o, args.ms) for o in organs])),
            pool_mean=float(pool_hz.mean()),
            pool_max=float(pool_hz.max()),
            pools_firing=firing, pools=len(pools),
            population=float(window.sum() / c["N"] / (args.ms / 1000.0)),
            never_fired_pct=float((spikes == 0).mean() * 100),
            seconds=time.time() - t0)
        rows.append(row)
        print(f"\n{name}: gain {cfg['gain']}, "
              f"{args.ms} ms of simulated cord  ({row['seconds']:.0f} s to run)")
        print(f"  descending        {row['descending']:8.2f} Hz")
        print(f"  organ groups      {row['organ_mean']:8.2f} Hz (mean)")
        print(f"  pool groups       {row['pool_mean']:8.2f} Hz (mean), "
              f"{row['pool_max']:.2f} Hz (best)")
        print(f"  pools firing      {row['pools_firing']}/{row['pools']} "
              f"— the HUD's line")
        print(f"  population        {row['population']:8.2f} Hz, "
              f"never fired {row['never_fired_pct']:.1f}%")
        best = np.argsort(-pool_hz)[:5]
        print("  loudest pools: " + ", ".join(f"{pools[i].split(':')[1]}/"
                                              f"{pools[i].split(':')[2]} "
                                              f"{pool_hz[i]:.1f} Hz" for i in best))
        # the cord only carries the pools the body asset could place (seven per
        # leg); those are the ones `FlyCord.activePools` counts
        if cord_pools:
            ch = np.array([pool_hz[pools.index(p)] if p in pools else 0.0
                           for p in cord_pools])
            print(f"  the cord's own pools ({len(cord_pools)}): "
                  f"{int((ch >= 1.0).sum())} firing at >= 1 Hz, "
                  f"mean {ch.mean():.2f} Hz, best {ch.max():.2f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
