#!/usr/bin/env python3
"""
Odor as a spatial and temporal field, and the drive it puts on the antenna.

Brief item 11. The connectome has carried 3,000 olfactory receptor neurons as a
named group (`sensory_olfactory`, from BANC's `cell_function == "olfactory"`)
since the first packing, and nothing has ever driven them: the environment had a
floor, a grid and three lights, and no smell. This is the field it was missing,
and the measurement that the drive reaches the brain — not the claim.

## The field

An odor source in a wind. Real plumes are not smooth clouds: a filamentous,
intermittent structure is the whole point of insect odor navigation (Murlis,
Elkinton & Cardé 1992, *Annu. Rev. Entomol.*; Celani et al. 2014, *PNAS* — the
receptor's problem is detecting filaments, not measuring concentration). The
model here has the two properties those papers say matter and nothing else:

    mean concentration      c̄(p) = C0 * (s0 / (s0 + s)) * exp(-r² / 2σ(s)²)
                                    * exp(-s / Λ)
    plume widening          σ(s) = σ0 + α·s
    intermittency           c(p, t) = c̄ * max(0, 1 + m·sin(2π f·t − k·s + φ))

where `s` is distance downwind of the source and `r` the crosswind distance.
Upwind of the source the concentration is zero. The constants are labelled in
`docs/ASSUMPTIONS.md`; the honest summary is that this is a **kinematic** plume
with literature-shaped statistics — it is not a solution of the advection-
diffusion equation, and it does not claim to be. It is deterministic given
(seed, source, wind, time), which is what a run has to be to be comparable with
another run.

## The measurement

`--probe` drives the ORN group with the concentration sampled where an antenna
would be — at the animal's own head, which is where the body model puts it — and
measures, over the same LIF the app runs:

  1. **Does the drive reach the brain at all?** ORN rate and population rate
     against a zero-odor control, same seed, same everything else, so the
     difference is attributable to the smell;
  2. **Where it goes**: the ORNs project to the antennal lobe, and what leaves
     the lobe is what the fly can act on, so the tool reports the ascending,
     descending and leg-motor rates as well — an odor that moves nothing outside
     the antenna is a light bulb, not a sense;
  3. **Dose–response**: ORN rate against concentration, which must be monotone
     or the field is decorative. A saturating curve is expected and fine.

    python3 tools/odor_field.py --probe            # the measurement
    python3 tools/odor_field.py --gate             # the CI gate
    python3 tools/odor_field.py --figure docs/img/odor_field.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# ---------------------------------------------------------------- the field
# Every constant here is a labelled place-holder except where a source is named.
# See docs/ASSUMPTIONS.md #31-#36: the geometry is an ENGINEERING PLACEHOLDER,
# the intermittency is an APPROXIMATION of turbulent statistics, and the
# conversion from concentration to a receptor current is an APPROXIMATION of
# the app's own tone convention (assumption #10).
DEFAULTS = dict(
    source_cm=(0.0, 0.0, 0.0),     # where the odor is released, world frame
    wind_cms=(0.30, 0.0, 0.0),     # the air, in cm/s (a fly walks ~1 cm/s)
    c0=1.0,                        # concentration at the source, in "units":
                                   # the unit is the concentration that drives
                                   # the ORNs at `baseline_drive`
    decay_cm=1.20,                 # Λ: mean concentration falls e-fold per this
    sigma0_cm=0.045,               # σ0: plume width at the source
    widening=0.22,                 # α: σ grows this much per cm downwind
    s0_cm=0.05,                    # smoothing so the source is not a singularity
    duty=0.20,                     # D: the fraction of time a fixed point is
                                   # inside a filament (literature: 10-30%)
    gust_hz=3.5,                   # f: how often a filament passes a fixed point
    wavenumber=6.0,                # k: the phase lag per cm downwind, which is
                                   # what makes a filament arrive at the far end
                                   # of the plume later than at the near end
    # The LIF fires when its membrane passes 1.0, and a group's membrane tracks
    # its external current, so a drive below 1.0 is a drive that cannot make a
    # receptor fire at all. The app's tone is 2.5 (`FlyCord`), so this is the
    # same order: an APPROXIMATION of the receptor's gain (assumption #34), not
    # a measured dose-response, but one that is at least in the units the rest
    # of the loop is in.
    baseline_drive=2.6,
    antenna_cm=(0.052, -0.0, -0.009),   # where the antennae are, from the body
                                   # model's own head geom (FlyWorld's head is
                                   # at +x and the antennae at its front)
)


class Plume:
    """The odor field: concentration at a point at a time.

    Pure and importable on purpose. The app needs the same field, and the way to
    keep two implementations honest is to have one definition of what the field
    *is* — this class, with the constants it was measured with — so that the
    port can be diffed against it number for number rather than eyeballed.
    """

    def __init__(self, **kw):
        self.p = dict(DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if v is not None})
        self.source = np.array(self.p["source_cm"], float)
        w = np.array(self.p["wind_cms"], float)
        self.speed = float(np.linalg.norm(w))
        if self.speed <= 0:
            raise ValueError("the wind must not be still: without advection there "
                             "is no plume, only a smell")
        self.wind = w / self.speed

    # -- the field itself -----------------------------------------------------
    def mean(self, points) -> np.ndarray:
        """Time-mean concentration at world points, shape (..., 3) -> (...)."""
        d = np.asarray(points, float) - self.source
        s = d @ self.wind                      # downwind distance, may be < 0
        r = np.linalg.norm(d - s[..., None] * self.wind, axis=-1)
        c = (self.p["c0"] * (self.p["s0_cm"] / (self.p["s0_cm"] + np.maximum(s, 0.0)))
             * np.exp(-(r ** 2) / (2 * (self.p["sigma0_cm"] + self.p["widening"]
                                        * np.maximum(s, 0.0)) ** 2))
             * np.exp(-np.maximum(s, 0.0) / self.p["decay_cm"]))
        return np.where(s >= 0, c, 0.0)

    def at(self, points, t_ms: float) -> np.ndarray:
        """The concentration a receptor actually sees: filaments of odor passing
        through clean air, not a smooth cloud.

        This is the property the odor-navigation literature turns on (Murlis et
        al. 1992; Celani et al. 2014): an antenna in a plume spends most of its
        time in *clean air* and meets the odor as discrete packets, which is why
        a receptor needs a fast transient response rather than an accurate mean.
        A sinusoid around the mean cannot produce that — it is above any
        threshold more than half the time — so the modulation here is a duty
        cycle: inside a filament (phase in [0, D)) the concentration is a raised
        cosine rising to `2/D` times the mean, and outside it, zero. The time
        mean is preserved by construction (`mean(w) = D/2`), the peak-to-mean is
        `2/D` (10 at the default 20% duty), and the phase slides downwind at
        `wavenumber` radians per cm, so the far end of the plume sees the same
        filament later.
        """
        d = np.asarray(points, float) - self.source
        s = np.maximum(d @ self.wind, 0.0)
        D, f, k = self.p["duty"], self.p["gust_hz"], self.p["wavenumber"]
        phase = np.mod(f * (t_ms / 1000.0) - k * s / (2 * np.pi), 1.0)
        inside = phase < D
        w = np.where(inside, 0.5 * (1 - np.cos(2 * np.pi * phase / D)), 0.0)
        return self.mean(points) * (2.0 / D) * w

    # -- what the body does with it ------------------------------------------
    def drive(self, points, t_ms: float) -> np.ndarray:
        """The external current the ORNs get, in the app's tone units.

        A receptor's rate grows with the odor and saturates; the app's own
        convention for an external drive is a current on the group, so the
        conversion is `baseline_drive * c / (1 + c)` — the simplest saturating
        map with the right two limits, and an APPROXIMATION (assumption #34),
        not a measured dose–response.
        """
        c = self.at(points, t_ms)
        return self.p["baseline_drive"] * c / (1.0 + c)

    def facts(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.p.items()}


# ---------------------------------------------------------------- the binary
SECTIONS = ["rootIDs", "positions", "neuronMeta", "csrRowPtr", "csrColIdx",
            "csrWeight", "csrDelay", "groupIndices", "retinaUV"]


def load_cord(bin_path: pathlib.Path, meta_path: pathlib.Path):
    """The packed connectome, as `pool_probe.load` returns it."""
    spec = importlib.util.spec_from_file_location("pool_probe", HERE / "pool_probe.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    return pp, pp.load(bin_path, meta_path)


def group_size(c: dict, name: str) -> int:
    """How many cells a group holds, out of the packed probe (`pool_probe.load`
    returns the group table as `groups`, keyed by name)."""
    g = c["groups"].get(name)
    return int(g["count"]) if g else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", type=pathlib.Path,
                    default=ROOT / "build" / "flybanc.bin")
    ap.add_argument("--meta", type=pathlib.Path,
                    default=ROOT / "build" / "flybanc_meta.json")
    ap.add_argument("--probe", action="store_true",
                    help="drive the ORNs from the field and measure the brain")
    ap.add_argument("--gate", action="store_true",
                    help="fail unless the field and its path into the brain hold")
    ap.add_argument("--figure", type=pathlib.Path, default=None)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--write", action="store_true",
                    help="write the field's parameters into world/world.json, "
                         "where the app reads them (like the `view` block)")
    ap.add_argument("--golden", type=pathlib.Path, default=None,
                    help="write sample values for the Swift port to be pinned to")
    ap.add_argument("--ms", type=int, default=120,
                    help="connectome milliseconds per condition in --probe")
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    plume = Plume()
    p = plume.p
    out: dict = {"field": plume.facts()}

    # ---- the field's own properties, measured ------------------------------
    print("the field")
    dists = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    pts = np.stack([dists, np.zeros_like(dists), np.full_like(dists, -0.009)], axis=-1)
    c_mean = plume.mean(pts)
    print(f"  wind {p['wind_cms']} cm/s from a source at {p['source_cm']}, "
          f"decay {p['decay_cm']:g} cm, widening {p['widening']:g}/cm")
    print(f"    {'downwind':>9} {'time-mean c':>12} {'filament peak':>14}")
    for d, cm in zip(dists, c_mean):
        peak = float(max(plume.at(np.array([[d, 0, -0.009]]), t)[0]
                         for t in np.linspace(0, 2000 / p["gust_hz"], 400)))
        print(f"    {d:>7.2f} cm {cm:>12.4f} {peak:>14.4f}")
    up = plume.mean(np.array([[-0.5, 0.0, -0.009]]))[0]
    print(f"  upwind of the source: c = {up:.4f} (a plume does not reach "
          f"upwind)")
    out["downwind"] = [{"s_cm": float(d), "mean": float(c), "peak": float(
        max(plume.at(np.array([[d, 0, -0.009]]), t)[0]
            for t in np.linspace(0, 2000 / p["gust_hz"], 400)))}
        for d, c in zip(dists, c_mean)]

    # What an antenna sees standing in the plume: the trace that matters is the
    # one at a *fixed* point, because that is the intermittency — how long the
    # receptor is in clean air between filaments — and it is the number a fly's
    # navigation is built on.
    ts = np.arange(0.0, 2000.0, 1.0)
    here = np.array([0.30, 0.0, -0.009])
    trace = np.array([plume.at(here[None, :], tt)[0] for tt in ts])
    clean = float((trace < 0.1 * trace.mean()).mean())
    inside = float((trace > 0.0).mean())
    peak_ratio = float(trace.max() / max(trace.mean(), 1e-9))
    print(f"  an antenna standing 3 mm downwind: mean {trace.mean():.3f}, "
          f"peak {trace.max():.3f}, peak/mean {peak_ratio:.1f}")
    print(f"    inside a filament {100 * inside:.0f}% of the time, in clean air "
          f"(<10% of the mean) {100 * clean:.0f}% of the time")
    out["filaments"] = {"mean": float(trace.mean()), "peak": float(trace.max()),
                        "peak_over_mean": peak_ratio,
                        "duty_inside": inside, "duty_clean": clean}

    problems = []
    if not np.all(np.diff(c_mean) < 0):
        problems.append("the time-mean concentration does not fall with distance")
    if up != 0.0:
        problems.append(f"the plume reaches upwind of its source (c = {up})")
    if peak_ratio < 3.0:
        problems.append(f"the trace is not intermittent: peak/mean {peak_ratio:.2f}")
    if not (0.05 < 1 - clean < 0.95 or clean < 0.05):
        pass
    if inside < 0.05 or inside > 0.9:
        problems.append(f"the antenna is inside a filament {100 * inside:.0f}% of "
                        "the time — that is a cloud, not a plume "
                        "(literature: 10-30%)")
    if np.any(~np.isfinite(trace)):
        problems.append("the field produced a non-finite concentration")

    # ---- the drive reaches the brain ---------------------------------------
    probe = None
    if args.probe or args.gate:
        pp, c = load_cord(args.bin, args.meta)
        n_orn = group_size(c, "sensory_olfactory")
        print("")
        print("the drive")
        print(f"  sensory_olfactory in {args.bin.name}: {n_orn:,} cells")
        if n_orn <= 0:
            problems.append("sensory_olfactory is not in the binary — the field "
                            "would drive nothing")
            probe = {"orn_cells": 0}
        else:
            # the antenna sits where the body model's head is; the animal walks
            # upwind through the plume over the run, so the drive sweeps the
            # whole response rather than sitting at one value
            groups = ["sensory_olfactory", "ascending", "descending",
                      "pool:T1_left:trochanter_flexor", "leg_motor"]
            names = {gv: gv for gv in groups
                     if group_size(c, gv) > 0 or gv.startswith("pool:")}
            probe = {"orn_cells": n_orn, "ms": args.ms, "conditions": {}}
            for label, c_mult in (("no odor", 0.0), ("far", 0.05),
                                  ("edge", 0.25), ("in it", 1.0),
                                  ("strong", 5.0)):
                net = pp.LIF(c, args.gain, seed=args.seed)
                fired_orn = 0
                drive_peak = 0.0
                rate_sums = {gv: 0.0 for gv in groups if gv != "sensory_olfactory"}
                # The antenna stands 3 mm downwind of the source, which is where
                # a fly that has found the plume would be, and the filaments
                # arrive at it as time passes: the drive varies because the
                # *field* does, not because the animal is moving. (The first
                # version started the animal 1 cm upwind and walked it at 1 cm/s
                # — over 120 ms that is 1.2 mm, so the antenna never reached the
                # source and every ORN rate was zero. The drive's peak is now
                # reported with every condition so that mistake cannot repeat
                # quietly.)
                here = np.array([[0.30, 0.0, -0.009]])
                for t in range(args.ms):
                    drive = float(plume.drive(here, float(t))[0]) * c_mult
                    drive_peak = max(drive_peak, drive)
                    net.drive("sensory_olfactory", drive)
                    # the app turns the retina on with a flat 1.0 (`BrainEngine
                    # .load`), so this runs the brain at the app's operating
                    # point rather than in the dark
                    net.drive("sensory_vision", 1.0)
                    net.step()
                    fired_orn += int(np.isin(net.last, net.members("sensory_olfactory")).sum())
                    for gv in rate_sums:
                        members = net.members(gv) if gv != "leg_motor" else None
                        if members is not None:
                            rate_sums[gv] += float(np.isin(net.last, members).sum())
                orn_hz = fired_orn / n_orn * 1000.0 / args.ms
                record = {"orn_hz": orn_hz, "drive_peak": drive_peak}
                for gv, total in rate_sums.items():
                    size = group_size(c, gv)
                    record[gv.replace(":", "_") + "_hz"] = (
                        total / size * 1000.0 / args.ms if size else 0.0)
                probe["conditions"][label] = record
                print(f"  {label:<10} drive {drive_peak:>4.2f}  ORN {orn_hz:>6.2f} Hz   "
                      + "  ".join(f"{gv.split(':')[-1]} "
                                  f"{record[gv.replace(':', '_') + '_hz']:>6.2f}"
                                  for gv in rate_sums))
            base = probe["conditions"]["no odor"]["orn_hz"]
            top = probe["conditions"]["strong"]["orn_hz"]
            print(f"  ORN rate rises from {base:.2f} Hz with no odor to "
                  f"{top:.2f} Hz at the strongest drive")
            rates = [probe["conditions"][k]["orn_hz"] for k in
                     ("far", "edge", "in it", "strong")]
            # `b >= a - eps`: the rate must not *fall* as the concentration
            # rises. Written the other way round first — `b <= a` — which
            # demanded that it fall, and failed a run whose dose-response was
            # correct. It failed loudly, which is the only reason it was found
            # in one run rather than after a wrong conclusion.
            if not all(b >= a - 1e-9 for a, b in zip(rates, rates[1:])):
                problems.append(f"the ORN dose-response falls with concentration: "
                                f"{rates}")
            if rates[-1] <= base:
                problems.append("the odor does not raise the ORN rate at all")
            moved = [gv for gv in rate_sums
                     if abs(probe["conditions"]["strong"][gv.replace(":", "_") + "_hz"]
                            - probe["conditions"]["no odor"][gv.replace(":", "_") + "_hz"]) > 0.5]
            print(f"  groups the odor moved by more than 0.5 Hz: "
                  f"{', '.join(moved) if moved else 'none'}")
            probe["moved_groups"] = moved
            # The claim is not "receptors fire" — it is "the smell is a sense".
            # An odor that raises the ORNs and moves nothing beyond them is a
            # light bulb on the antenna, and the gate has to say so: the first
            # rehearsal of this tool (a drive below threshold, `baseline_drive`
            # 0.6 instead of 2.6) left the ORN rate at a few spikes from the
            # network and moved nothing, and still passed.
            if not moved:
                problems.append("the odor raises the receptor rate but moves "
                                "nothing outside the antenna: no group in "
                                "{ascending, descending, leg pools} changed by "
                                "more than 0.5 Hz")
            if top < 5.0:
                problems.append(f"the strongest condition reaches only "
                                f"{top:.2f} Hz on the receptors, which is not a "
                                "response")
            out["probe"] = probe

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=130)
        axes[0].semilogy(dists, c_mean, "o-")
        axes[0].set_xlabel("downwind distance (cm)")
        axes[0].set_ylabel("time-mean concentration")
        axes[0].set_title("the plume falls with distance", fontsize=9)
        axes[1].plot(ts, trace, lw=0.7, color="#8a5a2b")
        axes[1].axhline(trace.mean(), ls="--", lw=0.7, color="k")
        axes[1].set_xlabel("time (ms)")
        axes[1].set_ylabel("concentration at the antenna")
        axes[1].set_title(f"filaments passing a standing antenna "
                          f"(inside {100 * inside:.0f}% of the time)",
                          fontsize=9)
        cross = np.linspace(-0.4, 0.4, 161)
        prof = plume.mean(np.stack([np.full_like(cross, 0.3), cross,
                                    np.full_like(cross, -0.009)], axis=-1))
        axes[2].plot(cross * 10, prof)
        axes[2].set_xlabel("crosswind distance at 0.3 cm (mm)")
        axes[2].set_ylabel("concentration")
        axes[2].set_title("the plume has width", fontsize=9)
        for ax in axes:
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure)
        plt.close(fig)
        print(f"\n  wrote {args.figure}")

    if args.write:
        path = ROOT / "world" / "world.json"
        manifest = json.loads(path.read_text())
        manifest["odor"] = plume.facts()
        path.write_text(json.dumps(manifest, indent=1))
        print(f"  wrote the odor block into {path}")

    if args.golden:
        # The Swift port has to agree with this one number for number, the same
        # way the body solver is pinned to the golden trace, and a test that
        # carries its own expected values is a test that cannot drift with the
        # tool it is supposed to be checking.
        samples = []
        # The times are *chosen* rather than stepped: a filament is 20% of the
        # cycle, so a regular time grid samples clean air and pins the port to
        # zeros — a golden file that cannot tell a correct port from a broken
        # one. For each point this scans a few cycles and takes the peak, a
        # zero, and something in between, so the samples exercise the whole
        # formula: the downwind falloff, the crosswind profile, the upwind
        # cutoff and the filament phase.
        scan = np.arange(0.0, 400.0, 2.0)
        for pt in ([0.30, 0.0, -0.009], [0.9, 0.0, -0.009],
                   [0.30, 0.10, -0.009], [-0.4, 0.0, -0.009],
                   [3.0, 0.0, -0.009]):
            one = np.array([pt], float)
            trace = np.array([float(plume.drive(one, tt)[0]) for tt in scan])
            top = trace.max()
            t_peak = float(scan[int(trace.argmax())])
            t_zero = float(scan[int(np.argmin(np.abs(trace)))])
            half = np.argmin(np.abs(trace - top / 2))
            for t_ms in sorted({t_zero, float(scan[half]), t_peak}):
                samples.append({"point": pt, "t_ms": t_ms,
                                "drive": float(plume.drive(one, t_ms)[0])})
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        args.golden.write_text(json.dumps(samples, indent=1))
        print(f"  wrote {args.golden} ({len(samples)} samples)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"  wrote {args.json}")

    if args.gate:
        print("")
        for pr in problems:
            print(f"  FAIL {pr}")
        if problems:
            return 1
        print("  ok   the field is a plume, and its drive reaches the brain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
