#!/usr/bin/env python3
"""
Odor as a spatial and temporal field, and the drive it puts on the antenna.

Brief item 11. The connectome has carried 3,000 olfactory receptor neurons as a
named group (`sensory_olfactory`, from BANC's `cell_function == "olfactory"`)
since the first packing, and nothing had ever driven them: the environment had a
floor, a grid and three lights, and no smell. This is the field it was missing,
and the measurement of what the smell does — not the claim.

## The field

An odor source in a wind. Real plumes are not smooth clouds: a filamentous,
intermittent structure is the whole point of insect odor navigation (Murlis,
Elkinton & Cardé 1992, *Annu. Rev. Entomol.*; Celani et al. 2014, *PNAS* — the
receptor's problem is detecting filaments, not measuring concentration). The
model here has the two properties those papers say matter and nothing else:

    mean concentration   c̄(p) = C0 * (s0/(s0+s)) * exp(-r²/2σ(s)²) * exp(-s/Λ)
    plume widening       σ(s) = σ0 + α·s
    intermittency        phase = mod(f·t - k·s/2π, 1)
                         c(p,t) = c̄ * (2/D) * ½(1 - cos(2π·phase/D))  if phase < D
                         c(p,t) = 0                                    otherwise

where `s` is distance downwind of the source and `r` the crosswind distance.
Upwind of the source the concentration is zero. Inside a filament the
concentration is a raised cosine rising to `2/D` times the mean; the time mean
is preserved by construction (`mean(w) = D/2`), the peak-to-mean is `2/D` (10 at
the default 20% duty), and the phase slides downwind at `k` radians per cm, so
the far end of the plume sees the same filament later. The constants are
labelled in `docs/ASSUMPTIONS.md`; the honest summary is that this is a
**kinematic** plume with literature-shaped statistics — it is not a solution of
the advection-diffusion equation, and it does not claim to be. It is
deterministic given (source, wind, time), which is what a run has to be to be
comparable with another run.

## Where the animal stands

The source is 5.5 mm in front of the animal and the air blows *onto* it
(`wind_cms` points at the animal), so the plume goes over the animal's head and
walking upwind is walking towards the food. The antennae are where the body
model's own head geom is — the same point `WorldModel.sampleOdor()` reads on the
phone — which puts them 0.5 cm downwind of the source: in the plume.

## The measurement

`--probe` drives the ORN group with the field sampled at the antennae and
measures, over the same LIF the app runs, with the same window and seed:

  1. **Does the animal smell at all?** The ORN rate against a no-odor control
     (the drive exactly zero). The window is a full 500 ms after 100 ms of
     warm-up, because the first version of this measurement used 120 ms — less
     than one gust cycle and inside the LIF's start-up transient — and reported
     a group that actually idles at ~19 Hz as "0.00 Hz", and every number in its
     table as a transient;
  2. **Where the smell goes**: the cells the receptor group actually synapses
     onto, read out of the packed connectome rather than assumed, and then the
     ascending and descending populations and a leg motor pool. A smell that
     moves nothing outside the antenna is a light bulb, not a sense;
  3. **How far it reaches**: the current at which the receptors start firing
     (bisected on the LIF, not guessed), and from that the distance downwind at
     which a filament peak still crosses it — the animal's smelling range;
  4. **Dose–response**: ORN rate against concentration, which must be monotone
     or the field is decorative.

    python3 tools/odor_field.py --probe            # the measurement
    python3 tools/odor_field.py --gate             # the CI gate
    python3 tools/odor_field.py --write            # the world the app reads
    python3 tools/odor_field.py --golden build/odor_golden.json
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
# See docs/ASSUMPTIONS.md #26-#29: the geometry is an ENGINEERING PLACEHOLDER,
# the intermittency is an APPROXIMATION of turbulent statistics, and the
# conversion from concentration to a receptor current is an APPROXIMATION of
# the app's own drive convention.
DEFAULTS = dict(
    source_cm=(0.30, 0.0, 0.0),    # where the odor is released, world frame: a
                                   # patch 2.5 mm in front of the animal, which
                                   # stands at the origin facing +x — the animal
                                   # has found the food and is standing on it
    wind_cms=(-0.30, 0.0, 0.0),    # the air, in cm/s (a fly walks ~1 cm/s). It
                                   # blows *onto* the animal, so the plume goes
                                   # over its head and walking upwind (+x) is
                                   # walking to the source
    c0=1.0,                        # concentration at the source, in "units"
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
    # The receptor current a saturating concentration gives, in the app's tone
    # units. This is the field's operating point and it is *set by measurement*,
    # not by taste. Two measurements decide it:
    #
    #  * a ladder of currents on the ORN group (--probe prints it) says the
    #    group idles at ~20 Hz in clean air and leaves that at a current of
    #    about 3.9 (the 20% criterion the gate uses);
    #  * so `baseline_drive` is chosen so the animal's *own* antennae sit well
    #    above that: at 45 the antennae's filament peak is a current of ~25,
    #    which is 7x the threshold, and the distance at which a filament peak
    #    still crosses it comes out at ~1.5 cm — the animal smells the patch it
    #    is standing on, and loses it 1.5 cm further downwind. A smaller
    #    current (the first version used 2.6) put the animal's own antennae
    #    *below* the receptors' threshold: the app shipped a fly that could not
    #    smell its own food.
    #
    # The value is an ENGINEERING PLACEHOLDER (assumption #34). The threshold
    # and the range it produces are measured, by this tool, on every run.
    baseline_drive=45.0,
    antenna_cm=(0.052, -0.0, -0.009),   # where the antennae are, world frame,
                                   # from the body model's own head geom (the
                                   # animal faces +x and the antennae are at the
                                   # front of its head); `WorldModel.sampleOdor`
                                   # reads the same geom on the phone
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

    # -- where things are -----------------------------------------------------
    def point(self, s_cm: float, across_cm: float = 0.0,
              z_cm: float | None = None) -> np.ndarray:
        """A world point `s_cm` downwind of the source, `across_cm` crosswind.

        The z is the plane every sample in this tool has always used for the
        antenna — the height of the body model's head geom — so that a point at
        the antenna's own downwind distance is the antenna.
        """
        along = self.source + s_cm * self.wind
        side = np.array([-self.wind[1], self.wind[0], 0.0]) * across_cm
        z = DEFAULTS["antenna_cm"][2] if z_cm is None else z_cm
        return np.array([along[0] + side[0], along[1] + side[1], z])

    def downwind(self, point) -> float:
        """Distance downwind of the source (negative upwind of it)."""
        return float((np.asarray(point, float) - self.source) @ self.wind)

    def filament_peak(self, s_cm: float, samples: int = 400) -> float:
        """The highest concentration a filament puts at a point, which is
        `2/D` times the mean there."""
        one = self.point(s_cm)[None, :]
        cycle = 1000.0 / self.p["gust_hz"]
        return float(max(self.at(one, t)[0] for t in np.linspace(0.0, cycle, samples)))

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
        cosine rising to `2/D` times the mean, and outside it, zero.
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


def first_targets(c: dict, name: str) -> np.ndarray:
    """The cells a group's cells synapse onto, read out of the packed
    connectome's own CSR — the only honest answer to "where does this go?".

    The smell's target population is not assumed to be a named group: the
    receptors' synapses land on 881 cells, 615 of which are not themselves
    labelled olfactory. What the named populations do is reported separately.
    """
    row, col = c["rowPtr"], c["colIdx"]
    mine = set(c["members"](name).tolist())
    reach = np.unique(np.concatenate([col[row[i]:row[i + 1]]
                                      for i in c["members"](name)]))
    return reach[~np.isin(reach, np.array(sorted(mine), dtype=reach.dtype))]


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
    ap.add_argument("--ms", type=int, default=860,
                    help="connectome milliseconds per condition, counted after "
                         "the warm-up: three gust cycles at the default 3.5 Hz, "
                         "because a filament is a discrete event and a window "
                         "that is not a whole number of cycles reports how many "
                         "peaks happened to land in it")
    ap.add_argument("--ladder-ms", type=int, default=500,
                    help="milliseconds per rung of the threshold ladder: the "
                         "receptor rate is a strong, seed-stable signal there, "
                         "so this one is short")
    ap.add_argument("--warmup", type=int, default=100,
                    help="milliseconds run before counting, so the LIF's own "
                         "start-up transient is not averaged into a rate")
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    plume = Plume()
    p = plume.p
    out: dict = {"field": plume.facts()}

    # ---- the field's own properties, measured ------------------------------
    print("the field")
    dists = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    pts = np.stack([plume.point(float(d)) for d in dists])
    c_mean = plume.mean(pts)
    antenna = np.array(p["antenna_cm"], float)
    s_ant = plume.downwind(antenna)
    print(f"  wind {p['wind_cms']} cm/s from a source at {p['source_cm']}; the "
          f"antennae are {s_ant:.2f} cm downwind of it")
    print(f"  decay {p['decay_cm']:g} cm, widening {p['widening']:g}/cm, "
          f"duty {p['duty']:g} at {p['gust_hz']:g} Hz")
    print(f"    {'downwind':>9} {'time-mean c':>12} {'filament peak':>14}")
    for d, cm in zip(dists, c_mean):
        print(f"    {d:>7.2f} cm {cm:>12.4f} {plume.filament_peak(float(d)):>14.4f}")
    up = float(plume.mean(plume.point(-0.5)[None, :])[0])
    print(f"  upwind of the source: c = {up:.4f} (a plume does not reach "
          f"upwind)")
    out["downwind"] = [{"s_cm": float(d), "mean": float(c),
                        "peak": plume.filament_peak(float(d))}
                       for d, c in zip(dists, c_mean)]

    # What an antenna sees standing in the plume: the trace that matters is the
    # one at a *fixed* point, because that is the intermittency — how long the
    # receptor is in clean air between filaments — and it is the number a fly's
    # navigation is built on.
    ts = np.arange(0.0, 2000.0, 1.0)
    trace = np.array([plume.at(antenna[None, :], tt)[0] for tt in ts])
    clean = float((trace < 0.1 * trace.mean()).mean())
    inside = float((trace > 0.0).mean())
    peak_ratio = float(trace.max() / max(trace.mean(), 1e-9))
    print(f"  the antennae standing {s_ant * 10:.1f} mm downwind: mean "
          f"{trace.mean():.3f}, peak {trace.max():.3f}, peak/mean "
          f"{peak_ratio:.1f}")
    print(f"    inside a filament {100 * inside:.0f}% of the time, in clean air "
          f"(<10% of the mean) {100 * clean:.0f}% of the time")
    out["filaments"] = {"mean": float(trace.mean()), "peak": float(trace.max()),
                        "peak_over_mean": peak_ratio,
                        "duty_inside": inside, "duty_clean": clean,
                        "antenna_downwind_cm": s_ant}

    problems = []
    if not np.all(np.diff(c_mean) < 0):
        problems.append("the time-mean concentration does not fall with distance")
    if up != 0.0:
        problems.append(f"the plume reaches upwind of its source (c = {up})")
    if peak_ratio < 3.0:
        problems.append(f"the trace is not intermittent: peak/mean {peak_ratio:.2f}")
    if inside < 0.05 or inside > 0.9:
        problems.append(f"the antenna is inside a filament {100 * inside:.0f}% of "
                        "the time — that is a cloud, not a plume "
                        "(literature: 10-30%)")
    if np.any(~np.isfinite(trace)):
        problems.append("the field produced a non-finite concentration")

    # ---- the drive reaches the brain ---------------------------------------
    # Two things changed here after the first version of this measurement, and
    # both had produced a confident wrong answer rather than merely untidy code:
    #
    #  * the window was 120 ms — less than one gust cycle, and inside the LIF's
    #    start-up transient. It reported "no odor: 0.00 Hz" for a group that
    #    actually idles at ~19 Hz, and every rate in its table was a transient;
    #  * the "antenna" was a point 3 mm downwind of the source that the app never
    #    samples. The app samples the body model's head geom. That point is now
    #    the animal's own condition, and the other distances are what it would
    #    smell if it walked.
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
            orn = c["members"]("sensory_olfactory")
            reach = first_targets(c, "sensory_olfactory")
            row, col = c["rowPtr"], c["colIdx"]
            n_syn = int(sum(row[i + 1] - row[i] for i in orn))
            watch = [("ascending", c["members"]("ascending")),
                     ("descending", c["members"]("descending"))]
            for name in ("pool:T1_left:trochanter_flexor", "leg_motor"):
                if group_size(c, name) > 0:
                    watch.append((name.split(":")[-1], c["members"](name)))
            print(f"  its synapses land on {reach.size:,} cells that are not "
                  f"olfactory receptors ({n_syn:,} synapses); beyond them: "
                  + ", ".join(f"{n} {ids.size:,}" for n, ids in watch))
            print(f"  window {args.ms} ms after {args.warmup} ms of warm-up, "
                  f"gain {args.gain:g}, seed {args.seed}, {c['n']:,} cells")

            def measure(s_cm, mult=1.0, drive_off=False, ms=None,
                        warmup=None, seed=None):
                """Run the LIF with the field's drive (optionally scaled) at one
                downwind distance, and count spikes over the window."""
                net = pp.LIF(c, args.gain, seed=args.seed if seed is None else seed)
                ms = args.ms if ms is None else ms
                warm = args.warmup if warmup is None else warmup
                here = plume.point(s_cm if s_cm is not None else s_ant)[None, :]
                counts = {"orn": 0, "reach": 0}
                counts.update({name: 0 for name, _ in watch})
                peak = 0.0
                for t in range(warm + ms):
                    d = 0.0 if drive_off else float(plume.drive(here, float(t))[0]) * mult
                    if t >= warm:
                        peak = max(peak, d)
                    # the app turns the retina on with a flat 1.0
                    # (`BrainEngine.load`), so this runs the brain at the app's
                    # operating point rather than in the dark
                    net.drive("sensory_olfactory", d)
                    net.drive("sensory_vision", 1.0)
                    net.step()
                    if t >= warm:
                        counts["orn"] += int(np.isin(net.last, orn).sum())
                        counts["reach"] += int(np.isin(net.last, reach).sum())
                        for name, ids in watch:
                            counts[name] += int(np.isin(net.last, ids).sum())
                hz = {"orn": counts["orn"] / orn.size * 1000.0 / ms,
                      "reach": counts["reach"] / reach.size * 1000.0 / ms}
                for name, ids in watch:
                    hz[name] = counts[name] / ids.size * 1000.0 / ms
                return peak, hz

            # The animal's own condition is the antennae's own point at the
            # field's own scale; everything else is the animal walking.
            conditions = [("no odor at all (drive 0)", None, 1.0),
                          ("4 mm upwind of the source", -0.40, 1.0),
                          ("2.5 cm downwind", 2.50, 1.0),
                          ("1.2 cm downwind", 1.20, 1.0),
                          ("3 mm downwind", 0.30, 1.0),
                          (f"the antennae ({s_ant * 10:.1f} mm)", s_ant, 1.0),
                          ("1 cm downwind", 1.00, 1.0)]
            label0 = "no odor at all (drive 0)"
            results = {}
            print("")
            names = ["orn", "reach"] + [n for n, _ in watch]
            print(f"  {'condition':<28}" + "".join(f"{n:>13}" for n in names))
            for label, pt, mult in conditions:
                peak, hz = measure(pt, mult, drive_off=(pt is None))
                results[label] = {"drive_peak": peak, **hz}
                print(f"  {label:<28}" + "".join(
                    f"{hz[n]:>13.2f}" for n in names))
            print(f"    the columns are Hz: {', '.join(names)} "
                  f"(the last ones are populations, not the receptors)")
            base = results[label0]
            own = results[f"the antennae ({s_ant * 10:.1f} mm)"]

            # ---- the threshold, measured on the LIF -----------------------
            # Not a parameter anyone chose: the current on the group at which it
            # starts to fire. A ladder rather than a bisection, because a
            # bisection assumes the response is monotone and this one is noisy —
            # the first version of this code bisected and landed on 3.26 when the
            # ladder shows the crossing at ~1.6. The criterion is the one the
            # gate uses: a 20% rise over the group's own clean-air rate.
            levels = [0.05, 0.08, 0.11, 0.15, 0.22, 0.40, 1.0]
            ladder = []
            print("")
            print(f"  the receptors' threshold: currents at the antennae, walked "
                  f"up a ladder (clean air is {base['orn']:.2f} Hz, so the "
                  f"criterion is {1.2 * base['orn']:.2f} Hz)")
            for m in levels:
                peak, hz = measure(s_ant, m, ms=args.ladder_ms)
                ladder.append({"mult": m, "drive_peak": peak, "orn_hz": hz["orn"],
                               "reach_hz": hz["reach"]})
                fires = "  <- fires" if hz["orn"] >= 1.2 * base["orn"] else ""
                print(f"    peak drive {peak:>6.2f}   ORN {hz['orn']:>7.2f} Hz"
                      f"{fires}")
            rising = [r for r in ladder if r["orn_hz"] >= 1.2 * base["orn"]]
            step = (levels[1] - levels[0]) * own["drive_peak"]
            if rising:
                threshold = rising[0]["drive_peak"]
                print(f"  threshold: about {threshold:.2f} (+/- {step / 2:.2f}) — "
                      f"the antennae's own filament peak is "
                      f"{own['drive_peak'] / threshold:.1f}x above it")
            else:
                threshold = float("inf")
                print("  threshold: the receptors never leave their clean-air "
                      "rate on this ladder")
            ordinate = [r["orn_hz"] for r in ladder]
            if not all(b >= a - 1e-9 for a, b in zip(ordinate, ordinate[1:])):
                problems.append("the ORN dose-response falls with concentration: "
                                f"{[round(v, 2) for v in ordinate]}")
            # and from that, how far from the source a filament can still be
            # smelt: the largest distance where a peak crosses the threshold
            def peak_drive(s):
                pk = plume.filament_peak(s)
                return plume.p["baseline_drive"] * pk / (1 + pk)
            if np.isfinite(threshold):
                a, b = 0.05, 6.0
                for _ in range(40):
                    mid = 0.5 * (a + b)
                    if peak_drive(mid) >= threshold:
                        a = mid
                    else:
                        b = mid
                range_cm = a
                print(f"  range: a filament peak crosses that threshold out to "
                      f"{range_cm:.2f} cm downwind, and not from "
                      f"{range_cm + 0.05:.2f} cm")
            else:
                range_cm = 0.0
            out["probe"] = {"orn_cells": int(orn.size), "reach_cells": int(reach.size),
                            "orn_synapses": n_syn, "ms": args.ms,
                            "warmup": args.warmup, "gain": args.gain,
                            "seed": args.seed, "conditions": results,
                            "ladder": ladder, "threshold_drive": threshold,
                            "range_cm": range_cm}

            # ---- what the gate is allowed to claim ------------------------
            if own["orn"] < 1.2 * base["orn"]:
                problems.append(
                    f"the animal standing in its own plume smells nothing: the "
                    f"ORN rate is {own['orn']:.2f} Hz against {base['orn']:.2f} Hz "
                    f"in clean air — the field the app ships is below the "
                    f"receptors' threshold")
            if own["reach"] < 1.05 * base["reach"]:
                problems.append(
                    f"the smell does not reach the cells the receptors synapse "
                    f"onto: {own['reach']:.2f} Hz against {base['reach']:.2f} Hz "
                    f"with no odor (they are read out of the connectome, not "
                    f"assumed)")
            if range_cm < 1.0:
                problems.append(
                    f"the smelling range is {range_cm * 10:.1f} mm — under a body "
                    f"length: the plume is a bubble, not a field")
            # The smell's effect *beyond* its first targets is measured and
            # reported, because it is the honest state of the pathway, but it is
            # not gated: a bar here would either be met by noise or force the
            # number up. `reports/item11_odor.md` says what the VNC does.
            MIN_POP = 50        # below this a "rate" is a handful of cells and
                                # its change is not evidence of anything
            print("  beyond the receptors: " + ", ".join(
                f"{n} {base[n]:.2f} -> {own[n]:.2f} Hz" for n, _ in watch))
            moved = [n for n, ids in watch
                     if ids.size >= MIN_POP and abs(own[n] - base[n]) > 0.5]
            small = [f"{n} ({ids.size} cells)" for n, ids in watch
                     if ids.size < MIN_POP]
            print(f"    populations that moved by more than 0.5 Hz: "
                  f"{', '.join(moved) if moved else 'none'}"
                  + (f"; too few cells to read: {', '.join(small)}" if small else ""))
            out["probe"]["moved_groups"] = moved

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
        axes[1].set_ylabel("concentration at the antennae")
        axes[1].set_title(f"filaments passing a standing antenna "
                          f"(inside {100 * inside:.0f}% of the time)",
                          fontsize=9)
        cross = np.linspace(-0.4, 0.4, 161)
        prof = plume.mean(np.stack([plume.point(0.3, float(x)) for x in cross]))
        axes[2].plot(cross * 10, prof)
        axes[2].set_xlabel("crosswind distance at 3 mm (mm)")
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
        # cutoff and the filament phase. The points are at distances downwind of
        # the source, so the table survives the source being moved.
        scan = np.arange(0.0, 400.0, 2.0)
        grabs = [(s_ant, 0.0), (1.0, 0.0), (1.5, 0.0), (-0.4, 0.0), (3.0, 0.0),
                 (1.0, 0.1)]
        for s_cm, across in grabs:
            one = plume.point(s_cm, across)[None, :]
            tr = np.array([float(plume.drive(one, tt)[0]) for tt in scan])
            top = tr.max()
            t_peak = float(scan[int(tr.argmax())])
            t_zero = float(scan[int(np.argmin(np.abs(tr)))])
            half = np.argmin(np.abs(tr - top / 2))
            for t_ms in sorted({t_zero, float(scan[half]), t_peak}):
                samples.append({"point": [float(v) for v in one[0]],
                                "t_ms": t_ms,
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
        print("  ok   the field is a plume, the animal in it smells, and the "
              "smell reaches the cells the receptors synapse onto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
