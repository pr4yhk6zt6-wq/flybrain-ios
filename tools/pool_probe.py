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

  device       the build behind uploads/IMG_2713.png: gain 6.0, and every drive
               carrying the old kernel convention (external input multiplied by
               `externalDrive` 1.5, so a tone of 2.5 arrived as 3.75)
  app          the app as it stands: gain 12.0 (SimulationEngine's default), the
               tone of assumption #5 (2.5) on the descending group and on both
               organ channels, drives added raw (ASSUMPTIONS #25)
  app x1.5     the same at the old convention's amplitude, for the difference
  reference    gain 12.0, drives 2.5 — `tools/step3_closedloop.py --desc 2.5
               --n-star 12`, the configuration the loop's law was measured in
  vision       gain 12.0, drives 2.5, and the photoreceptors driven at 1.5, i.e.
               the whole sensorimotor path the brief asks about

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

# The rate window: the last 50 ms of a run, so the start-up transient (a
# uniform membrane potential, no synaptic current yet) is not averaged into the
# number. Rates are quoted per this many milliseconds — `hz()` divides by it —
# and it used to be passed `args.ms` by mistake, which made every "Hz" in this
# tool's output four times too small at the default 200 ms window: a rate is a
# count divided by the time it was counted over, and the counts only ever cover
# this window.
WINDOW_MS = 50


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

    return dict(n=n, e=e, N=n, n_group_idx=n_group_idx,
                rowPtr=view("csrRowPtr", np.uint32),
                colIdx=view("csrColIdx", np.uint32),
                weight=view("csrWeight", np.float16).astype(np.float32),
                delay=view("csrDelay", np.uint8),
                # 8 bytes per neuron, in the pack's own order: system id,
                # super-class id, transmitter id, flags, then the cell type as a
                # little-endian uint16. A tool that asks "what are this group's
                # targets" reads the answer here rather than in the feather, so
                # it describes the connectome the app actually loaded.
                neuronMeta=view("neuronMeta", np.uint8).reshape(-1, 8),
                meta=meta, groups=groups, members=members)


class LIF:
    """The reference network, as a thing that can be stepped.

    `tools/verify_banc.py` runs a window and reports; a closed loop has to run
    one millisecond, read what the pools did, and decide what to drive next — so
    the same equations live here behind `step()`, and `run()` is a loop over it.
    There is one implementation of the LIF in this repository and this is it:
    the kernel in `FlyBrain/Shaders/LIF.metal` is its port, and the kernels are
    judged against the numbers it produces.

    The state is exactly the reference's — membrane, synaptic current,
    refractory counters, the delay ring, the RNG — and the update is its update,
    line for line, including the noise draw's order and shape. `drive(name,
    value)` writes a group's external current, which is what
    `SimulationEngine.setGroupDrive` does on the phone; `None` clears it, which
    is the state of every group the body is not driving.
    """

    DT, TAU_M, TAU_SYN = 1.0, 20.0, 5.0
    V_TH, T_REF, NOISE = 1.0, 2, 0.015

    def __init__(self, c, gain, seed=42):
        self.c = c
        self.gain = float(gain)
        N = c["N"]
        self.rowPtr, self.colIdx = c["rowPtr"], c["colIdx"]
        self.W, self.delay = c["weight"], c["delay"]
        self.rng = np.random.default_rng(seed)
        self.V = self.rng.uniform(0, 0.5, N).astype(np.float32)
        self.I_syn = np.zeros(N, np.float32)
        self.refrac = np.zeros(N, np.int16)
        # Delays run 1..19 ms, so a ring of max+1 rows carries every write
        # without ever landing on the slot being read at the same instant.
        self.max_d = int(self.delay.max()) + 1
        self.ring = np.zeros((self.max_d, N), np.float32)
        self.I_ext = np.zeros(N, np.float32)
        # np.float64, not float: the reference multiplies a float32 array by
        # this scalar, and NumPy promotes to float64 for a NumPy scalar where a
        # Python float would stay float32. Same algebra, different rounding, and
        # the difference shows up as ~15 spikes in 30,000 over 200 ms — enough
        # to make the port's numbers disagree with the reference it is measured
        # against. (Verified: with these two as np.float64 the probe reproduces
        # its pre-refactor output byte for byte.)
        self.dm = np.exp(-self.DT / self.TAU_M)
        self.ds = np.exp(-self.DT / self.TAU_SYN)
        self.t = 0
        self.last = np.zeros(0, np.int64)        # last step's fired indices
        self._members = {}

    def members(self, name):
        idx = self._members.get(name)
        if idx is None:
            idx = self._members[name] = self.c["members"](name)
        return idx

    def drive(self, name, value):
        """Set (or clear, with None) one group's external current."""
        self.I_ext[self.members(name)] = 0.0 if value is None else value

    def step(self):
        """One millisecond. `self.last` is the fired indices, in order."""
        t = self.t
        slot = t % self.max_d
        self.I_syn = self.I_syn * self.ds + self.ring[slot]
        self.ring[slot] = 0
        I = self.I_syn * self.gain + self.I_ext + \
            self.rng.normal(0, self.NOISE, self.c["N"]).astype(np.float32)
        free = self.refrac <= 0
        self.V[free] = self.V[free] * self.dm + I[free] * (1 - self.dm)
        self.refrac[~free] -= 1
        fired = np.flatnonzero((self.V >= self.V_TH) & free)
        self.V[fired] = 0.0
        self.refrac[fired] = self.T_REF
        if len(fired):
            s0 = self.rowPtr[fired].astype(np.int64)
            s1 = self.rowPtr[fired + 1].astype(np.int64)
            if (s1 - s0).sum():
                e = np.concatenate([np.arange(a, b) for a, b in zip(s0, s1)])
                np.add.at(self.ring, ((t + self.delay[e].astype(np.int64)) % self.max_d,
                                      self.colIdx[e]), self.W[e])
        self.t += 1
        self.last = fired
        return fired

    def counts(self, names):
        """Spikes this millisecond for each named group (an array)."""
        fired = self.last
        out = np.zeros(len(names), np.int64)
        if not len(fired):
            return out
        for k, name in enumerate(names):
            out[k] = int(np.isin(fired, self.members(name)).sum())
        return out


def run(c, ms, gain, drives, seed=42, report_every=0, track=()):
    """`ms` milliseconds of the reference network with a fixed drive per group.

    Returns (spike counts per neuron over the whole run, the same over the last
    50 ms — the window the rates are quoted from, with the start-up transient
    left out — and, when `track` is given, a (ms, len(track)) array of the
    tracked groups' spikes per millisecond.
    """
    net = LIF(c, gain, seed)
    for group, value in drives.items():
        net.drive(group, value)
    spikes = np.zeros(c["N"], np.int64)
    window = np.zeros(c["N"], np.int64)
    tracked = np.zeros((ms, len(track)), np.int64) if track else None
    lag = min(WINDOW_MS, ms)                   # ignore the start-up transient
    for t in range(ms):
        fired = net.step()
        if len(fired):
            spikes[fired] += 1
            if t >= ms - lag:
                window[fired] += 1
        if tracked is not None:
            tracked[t] = net.counts(track)
        if report_every and t % report_every == 0:
            print(f"    t={t:4d} fired={len(fired):6,}", flush=True)
    return spikes, window, tracked


def hz(c, counts, name, window_ms=WINDOW_MS):
    """A group's firing rate, in Hz, from a count over `window_ms` milliseconds."""
    idx = c["members"](name)
    return float(counts[idx].sum()) / max(len(idx), 1) / (window_ms / 1000.0)


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

    # `externalInput` is added raw since ASSUMPTIONS #25 — one convention for the
    # groups and the photoreceptors, matching tools/verify_banc.py — so a drive
    # in this table is the current it is written as. The "device" row is the
    # build behind uploads/IMG_2713.png, which ran the *old* convention at the
    # old gain; every other row is the app as it stands.
    conditions = {
        "device": dict(gain=6.0, drives={"descending": 2.5 * 1.5,
                                         **{o: 2.5 * 1.5 for o in organs},
                                         "sensory_vision": 1.0}),
        "app": dict(gain=12.0, drives={"descending": 2.5,
                                       **{o: 2.5 for o in organs},
                                       "sensory_vision": 1.0}),
        "app x1.5": dict(gain=12.0, drives={"descending": 2.5 * 1.5,
                                            **{o: 2.5 * 1.5 for o in organs},
                                            "sensory_vision": 1.0}),
        "reference": dict(gain=12.0, drives={"descending": 2.5,
                                             **{o: 2.5 for o in organs}}),
        "vision": dict(gain=12.0, drives={"descending": 2.5,
                                          **{o: 2.5 for o in organs},
                                          "sensory_vision": 1.5}),
    }

    rows = []
    for name, cfg in conditions.items():
        if args.only and name != args.only:
            continue
        t0 = time.time()
        spikes, window, tracked = run(c, args.ms, cfg["gain"], cfg["drives"],
                                      report_every=0, track=cord_pools)
        pool_hz = np.array([hz(c, window, p) for p in pools])
        firing = int((pool_hz >= 1.0).sum())
        row = dict(
            condition=name, gain=cfg["gain"],
            descending=hz(c, window, "descending"),
            organ_mean=float(np.mean([hz(c, window, o) for o in organs])),
            pool_mean=float(pool_hz.mean()),
            pool_max=float(pool_hz.max()),
            pools_firing=firing, pools=len(pools),
            population=float(window.sum() / c["N"] / (WINDOW_MS / 1000.0)),
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
        # The HUD's own readout line, in the HUD's own units and over the same
        # kind of window, so a screenshot of the phone and this run on the same
        # binary can be read side by side — the same three quantities:
        # `net` is every spike in the connectome per second, `groups` is the
        # named-group machinery the kernels are encoded against, and
        # `group spikes` is the callers' own tally (SimulationEngine.groupSpikeSum
        # on the device, summed here over all 113 groups for this window).
        win_ms = WINDOW_MS
        def tally(names):
            # counts over the window = rate x window x population
            return sum(hz(c, window, name) * len(c["members"](name))
                       * (win_ms / 1000.0) for name in names)
        print(f"  the HUD's line    over {win_ms} ms · "
              f"net {row['population'] * c['N']:.0f} spk/s · "
              f"groups {len(c['groups'])}/{c['n_group_idx']} · "
              f"spikes {tally(c['groups']):.0f} (pools {tally(cord_pools):.0f})")
        if tracked is not None and len(cord_pools):
            # FlyCord, line for line: the pool's rate in Hz, then the muscle
            # filter, activation += (rate − activation)·(dt/tau), dt = 1 ms and
            # tau = 60 ms (assumption #7). `activePools` counts activation >= 1,
            # and the HUD prints that count, the mean activation, the *measured*
            # descending rate and the organ mean beside the tone it was given.
            sizes = np.array([max(len(c["members"](g)), 1) for g in cord_pools])
            rate = tracked / sizes[None, :] * 1000.0
            act = np.zeros(len(cord_pools))
            for t in range(rate.shape[0]):
                act += (rate[t] - act) * (1.0 / 60.0)
            tone = cfg["drives"].get("descending", 0.0)
            print(f"  the HUD should read {int((act >= 1.0).sum())}/{len(cord_pools)} "
                  f"pools · {act.mean():.1f} Hz · desc {row['descending']:.1f} Hz "
                  f"(tone {tone:g}) · organs {row['organ_mean']:.1f} Hz")

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
