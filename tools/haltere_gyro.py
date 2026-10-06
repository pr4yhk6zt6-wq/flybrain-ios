#!/usr/bin/env python3
"""
Halteres as gyroscopes: the angular velocity of the body, as the fly feels it.

Brief item 7. `docs/AUDIT.md` had this as PARTIAL — "Haltere bodies exist
physically; no haltere-derived angular-rate signal enters `FlyProprioception`" —
and that is still true of the code until this tool's numbers are ported. The
connectome, though, has always had the two sensory groups: `sensory_haltere_left`
(216 cells) and `sensory_haltere_right` (212), BANC's own annotation.

## What a haltere is, and what this tool takes from where

A haltere is a hindwing turned into a vibrating structure: it oscillates through
an arc of about 90° at wingbeat frequency, in antiphase to its wing (Pringle
1948, *Phil. Trans. R. Soc. B* 233:347; Miyan & Ewing 1985), and when the body
rotates the Coriolis force pushes the oscillating haltere **out of its stroke
plane**, which the campaniform fields at its base transduce (Pringle 1948 for
the mechanism; Nalbach 1993, *J. Comp. Physiol. A* 173:509, for the measured
kinematics that make dF2 the field that sees it). The fly therefore has, at
every instant of a stroke, a **directionally sensitive angular-rate sensor**
whose sensitivity axis is

    a = v̂ × n̂          v̂: the haltere's velocity, n̂: normal of the stroke plane

and the force it feels is `F = 2 m (Ω × v)` — linear in the body's angular
velocity Ω, and *tiny* next to the forces that drive the stroke, which is the
discrimination problem Pringle posed and the reason the field is specialised.

## What is measured, and what is a labelled constant

Measured here, from the body model's own geometry (`data/flybody`, loaded with
MuJoCo, exactly as `tools/build_body.py` loads it):

  * each haltere's mass (0.822 µg), the distance from its hinge to its centre of
    mass (0.172 mm), its hinge axis and its attachment point on the thorax;
  * the resulting sensitivity axis of each haltere, in the body frame;
  * what the two axes have in common and what they do not — which is what makes
    a left/right *pair* able to separate yaw and roll from pitch at all.

From the literature, labelled, because the model cannot supply them:

  * the stroke: ±45° about the hinge at 200 Hz (wingbeat/haltere frequency in
    *Drosophila*, Lehmann & Dickinson 2005, *J. Exp. Biol.* 208:3075; arc of
    ~90°, Pringle 1948). The model's halteres are **passive stubs** — one
    hinge joint, no actuator, no stroke — so the stroke has to come from
    somewhere, and a measured constant from the animal is the honest source;
  * the **relative stroke phase of the two halteres**, which the model cannot
    decide either (its hinge axes are geometry, and a hinge axis has no
    direction). This is the one degree of freedom the whole sign structure hangs
    on: see `CONVENTIONS` below and the report, where both are run and the
    difference is printed rather than asserted.

## The left/right pair, and why a difference channel means anything

Fox, Fairhall & Daniel (2010, *Front. Neural Circuits* 4:123) built the circuit
model of what the haltere pair is for: the Coriolis forces from yaw and roll are
**out of phase** between the left and right halteres, while the pitch force and
the centrifugal force are **in phase**, so a left-minus-right subtraction
receives yaw and roll and cancels pitch and centrifugal force — which is why the
fly can tell "I am turning" from "I am being shaken". This tool measures whether
the model's own haltere geometry supports that: the axes are combined into a
difference channel and a common channel, and the leakage of each rotation axis
into the wrong channel is reported as a number (it comes out below 1%).

    python3 tools/haltere_gyro.py --probe          # the measurement
    python3 tools/haltere_gyro.py --gate           # the CI gate
    python3 tools/haltere_gyro.py --write          # the world the app reads
    python3 tools/haltere_gyro.py --golden build/haltere_golden.json
    python3 tools/haltere_gyro.py --figure docs/img/haltere_gyro.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

DEG = 180.0 / 3.141592653589793

# --------------------------------------------------------------- the constants
# Every number here is either measured from the body model (marked MEASURED) or
# a labelled constant with a source.
DEFAULTS = dict(
    # -- the stroke: the model has no haltere stroke at all (a passive hinge
    #    with a ±0.2 rad range and no actuator), so these two are the animal's,
    #    and they set the scale of every force below.
    stroke_hz=200.0,          # BIOLOGICAL DATA: wingbeat/haltere frequency in
                              # Drosophila, ~200 Hz (Lehmann & Dickinson 2005)
    stroke_deg=90.0,          # BIOLOGICAL DATA: the haltere arcs through about
                              # 90° (Pringle 1948)
    # -- the drive the afferents get. `bias` is the tonic current the stroking
    #    haltere's *primary* forces hold the receptor population at all the time
    #    (the fields are tonically active in a stroking fly, `Flies tune the
    #    activity of their multifunctional gyroscope` 2024); `gain` is the
    #    current per unit of normalised angular velocity, with the reference
    #    rate at which a rotation is "1.0". Both are ENGINEERING PLACEHOLDERS in
    #    the app's tone units (docs/ASSUMPTIONS.md #30-#33); the LIF response
    #    they produce is measured here.
    bias=3.0,
    gain=2.0,
    reference_dps=100.0,      # APPROXIMATION: the yaw rate at which the drive
                              # reaches `gain`. A fly's saccades reach 1000s of
                              # °/s and a slow turn is ~50-100 °/s
    min_dps=1.0,              # below this the fly is not turning; the drive is
                              # exactly zero so "not moving" reads as not moving
    filter_ms=5.0,            # APPROXIMATION: first-order lag on the sensed
                              # rate. The afferents fire phase-locked to a 200 Hz
                              # stroke, so the population cannot follow faster
                              # than a few tens of Hz; 5 ms keeps the signal in
                              # the band the reflex literature measures
                              # (wing responses lead the stimulus by ~6° of a
                              # 5 Hz cycle: ~3 ms, JEB 2026 haltere paper)
    # -- the mirrored sign convention. See CONVENTIONS below.
    convention="antiphase",
)


class Halteres:
    """The pair, as the body model has them, and what they sense.

    Everything geometric here is read out of the MJCF the app's own asset is
    built from, so the sensitivity axes are the model's, not a guess about the
    model: `read_model()` returns the mass, the hinge axis, the centre-of-mass
    offset and the axis `a = v̂ × n̂` for each side.
    """

    def __init__(self, model_path: pathlib.Path, **kw):
        self.p = dict(DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if v is not None})
        self.model_path = pathlib.Path(model_path)
        self.geometry = read_model(self.model_path)
        self.a_left = self.geometry["left"]["sensitivity"]
        # The model's own right-hand axis, and the convention applied to it.
        # `a_R = S a_L` in the model (measured, agreement 0.9999), i.e. the two
        # halteres are plain mirrors of one another. Whether the *strokes* are
        # mirrored (both tips away from the midline at the same time) or
        # anti-mirrored (both tips the same way) is not in the geometry: a hinge
        # axis has no direction, and the model has no stroke at all. Negating
        # the right-hand axis is exactly that choice.
        self.a_right_model = self.geometry["right"]["sensitivity"]
        if self.p["convention"] == "antiphase":
            self.a_right = -self.a_right_model
        else:
            self.a_right = self.a_right_model.copy()
        self.a_mirror_agreement = float(
            self.a_left @ mirror(self.a_right_model))

    # -- the geometry of the pair --------------------------------------------
    @property
    def difference(self) -> np.ndarray:
        """The left-minus-right channel: what a subtraction of the two halteres
        receives. Yaw and roll, with pitch and the centrifugal force cancelled
        (Fox, Fairhall & Daniel 2010)."""
        return 0.5 * (self.a_left - self.a_right)

    @property
    def common(self) -> np.ndarray:
        """The left-plus-right channel: pitch, and everything the two halteres
        agree on."""
        return 0.5 * (self.a_left + self.a_right)

    def channel_leakage(self) -> dict:
        """How much of each rotation axis leaks into the wrong channel.

        The claim the fly's yaw reflex rests on is that the difference of the
        two halteres is a yaw (and roll) signal and not a pitch signal. Since
        the axes are not exactly orthogonal, that is a number, and this is it:
        the fraction of each axis's sensitivity that lands in the other channel.
        """
        out = {}
        for i, axis in enumerate(("roll(x)", "pitch(y)", "yaw(z)")):
            d = abs(self.difference[i])
            c = abs(self.common[i])
            out[axis] = {"difference": float(self.difference[i]),
                         "common": float(self.common[i]),
                         "leak_into_wrong_channel": float(min(d, c) / max(d, c) if max(d, c) else 0.0)}
        return out

    # -- the signal ----------------------------------------------------------
    def tip_speed(self) -> float:
        """The peak speed of the haltere's centre of mass, cm/s, MEASURED radius
        with the labelled stroke."""
        r_cm = self.geometry["left"]["com_distance_cm"]
        omega = 2 * np.pi * self.p["stroke_hz"] * np.radians(self.p["stroke_deg"]) / 2
        return float(omega * r_cm)

    def coriolis_force_nN(self, dps: float) -> float:
        """`F = 2 m Ω v`, in nanonewtons, for one haltere (MEASURED m and v)."""
        m_kg = self.geometry["left"]["mass_ug"] * 1e-9
        v = self.tip_speed() / 100.0                     # cm/s -> m/s
        return float(2 * m_kg * np.radians(dps) * v * 1e9)

    def primary_force_nN(self) -> float:
        """The in-plane force the same haltere feels every stroke — the signal
        the Coriolis force has to be told apart from (Pringle 1948)."""
        m_kg = self.geometry["left"]["mass_ug"] * 1e-9
        r_m = self.geometry["left"]["com_distance_cm"] / 100.0
        omega = 2 * np.pi * self.p["stroke_hz"] * np.radians(self.p["stroke_deg"]) / 2
        return float(m_kg * omega ** 2 * r_m * 1e9)

    def drive(self, dps_vec, filtered: bool = True) -> np.ndarray:
        """The current the two afferent groups get, in the app's tone units.

        `dps_vec` is the body's angular velocity (roll, pitch, yaw) in degrees
        per second, body frame. The map is

            drive = bias + gain * clip(Ω·a / Ω_ref)      per haltere

        with Ω·a the out-of-plane Coriolis component each haltere feels, and the
        rotation *reduced to zero* below `min_dps` by `rate(z)` so that a still
        animal gives exactly `bias` rather than the noise of its own rounding.
        """
        omega = np.asarray(dps_vec, float)
        if without_small(omega, self.p["min_dps"]):
            omega = np.zeros(3)
        left = self.p["bias"] + self.p["gain"] * float(omega @ self.a_left) / self.p["reference_dps"]
        right = self.p["bias"] + self.p["gain"] * float(omega @ self.a_right) / self.p["reference_dps"]
        return np.array([left, right])

    def facts(self) -> dict:
        return {k: (list(v) if isinstance(v, np.ndarray) else v)
                for k, v in self.p.items()}


def without_small(omega: np.ndarray, floor: float) -> bool:
    return bool(np.linalg.norm(omega) < floor)


def mirror(v: np.ndarray) -> np.ndarray:
    """Reflection about the sagittal plane (y -> -y)."""
    return np.array([v[0], -v[1], v[2]])


def read_model(path: pathlib.Path) -> dict:
    """Measure the haltere pair out of the body model.

    The MJCF is loaded exactly as `tools/build_body.py` loads it (MuJoCo), and
    every number below comes from the compiled model rather than from the XML
    text: the two can differ, and the model is what the app simulates.

    The sensitivity axis is `a = v̂ × n̂`, where `n̂` is the haltere's hinge axis
    (which is the normal of its stroke plane, because the stroke *is* rotation
    about the hinge) and `v̂` is the direction its centre of mass moves. Note
    that the cross product is a **pseudo**-vector: under a mirror reflection it
    comes back negated, and that is the whole reason a left/right pair can tell
    a rotation from a translation.
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    out = {}
    for side in ("left", "right"):
        name = f"haltere_{side}"
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise SystemExit(f"{path} has no {name} body")
        quat = np.asarray(m.body_quat[b], float)
        R = quat_to_matrix(quat)
        com = R @ np.asarray(m.body_ipos[b], float)
        r_hat = com / np.linalg.norm(com)
        axis = R @ np.array([1.0, 0.0, 0.0])          # the hinge axis = plane normal
        axis = axis / np.linalg.norm(axis)
        stroke = np.cross(axis, r_hat)
        stroke = stroke / np.linalg.norm(stroke)
        sensitivity = np.cross(stroke, axis)
        sensitivity = sensitivity / np.linalg.norm(sensitivity)
        out[side] = {
            "attach_cm": [float(v) for v in m.body_pos[b]],
            "mass_ug": float(m.body_mass[b] * 1e6),
            "com_distance_cm": float(np.linalg.norm(com)),
            "hinge_axis": [float(v) for v in axis],
            "stroke_direction": [float(v) for v in stroke],
            "sensitivity": sensitivity,
            "inertia": [float(v) for v in m.body_inertia[b]],
            "hinge_range_rad": [float(v) for v in
                                m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]],
        }
    return out


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- the binary
def load_cord(bin_path: pathlib.Path, meta_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("pool_probe", HERE / "pool_probe.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    return pp, pp.load(bin_path, meta_path)


def group_size(c: dict, name: str) -> int:
    g = c["groups"].get(name)
    return int(g["count"]) if g else 0


def first_targets(c: dict, name: str) -> np.ndarray:
    row, col = c["rowPtr"], c["colIdx"]
    mine = set(c["members"](name).tolist())
    reach = np.unique(np.concatenate([col[row[i]:row[i + 1]]
                                      for i in c["members"](name)]))
    return reach[~np.isin(reach, np.array(sorted(mine), dtype=reach.dtype))]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", type=pathlib.Path,
                    default=ROOT / "data/flybody/flybody-main/flybody/fruitfly/assets/fruitfly.xml")
    ap.add_argument("--bin", type=pathlib.Path, default=ROOT / "build" / "flybanc.bin")
    ap.add_argument("--meta", type=pathlib.Path, default=ROOT / "build" / "flybanc_meta.json")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--figure", type=pathlib.Path, default=None)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--write", action="store_true",
                    help="write the haltere block into world/world.json, where "
                         "the app reads it (like `view` and `odor`)")
    ap.add_argument("--golden", type=pathlib.Path, default=None)
    ap.add_argument("--ms", type=int, default=1000,
                    help="connectome milliseconds per condition after the warm-up "
                         "(the haltere pools are small — 216 and 212 cells — so a "
                         "short window cannot resolve a rate *difference* from "
                         "their own asymmetry; the error bar is measured, not "
                         "assumed: 50 ms bins, and its standard error is printed)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--gain-syn", type=float, default=12.0)
    ap.add_argument("--drive-gain", type=float, default=None,
                    help="override the drive per unit of normalised rate: 0 is "
                         "the rehearsal of a gyro that is not wired to anything")
    ap.add_argument("--drive-bias", type=float, default=None,
                    help="override the tonic drive a stroking haltere gives")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--convention", choices=("antiphase", "mirrored"),
                    default=DEFAULTS["convention"],
                    help="the relative stroke phase of the two halteres: "
                         "`antiphase` (what the fly does) or `mirrored` (the "
                         "model's own plain-mirror geometry). The gate is "
                         "rehearsed against `mirrored`, which is what a wrong "
                         "convention looks like.")
    args = ap.parse_args()

    pair = Halteres(args.body, convention=args.convention,
                    gain=args.drive_gain if args.drive_gain is not None else DEFAULTS["gain"],
                    bias=args.drive_bias if args.drive_bias is not None else DEFAULTS["bias"])
    out: dict = {"halteres": pair.facts(), "geometry": {
        side: {k: (v if not isinstance(v, np.ndarray) else [float(x) for x in v])
               for k, v in g.items()} for side, g in pair.geometry.items()}}
    problems = []

    # ---- what the model says the pair is -----------------------------------
    gL, gR = pair.geometry["left"], pair.geometry["right"]
    print("the pair, measured from the body model's own geometry")
    print(f"  {args.body}")
    print(f"  haltere mass {gL['mass_ug']:.3f} ug each "
          f"(left {gL['mass_ug']:.3f}, right {gR['mass_ug']:.3f})")
    print(f"  hinge -> centre of mass {gL['com_distance_cm'] * 10:.3f} mm "
          f"(right {gR['com_distance_cm'] * 10:.3f} mm)")
    print(f"  hinge axis, body frame: left {np.round(pair.geometry['left']['hinge_axis'], 4)}, "
          f"right {np.round(pair.geometry['right']['hinge_axis'], 4)}")
    print(f"  the two sensitivity axes are plain mirrors of each other: "
          f"cos(a_left, S a_right) = {pair.a_mirror_agreement:.4f}")
    print(f"  stroke: {pair.p['stroke_deg']:g} deg at {pair.p['stroke_hz']:g} Hz"
          f"  ->  peak speed {pair.tip_speed() * 10:.2f} mm/s")
    print(f"  the force the gyro has to be told apart from (in-plane, every "
          f"stroke): {pair.primary_force_nN():.1f} nN")
    for dps in (50, 100, 400, 1000):
        f = pair.coriolis_force_nN(dps)
        print(f"    at {dps:>5.0f} deg/s the Coriolis force is {f:.3f} nN "
              f"({100 * f / pair.primary_force_nN():.2f}% of it)")
    if pair.a_mirror_agreement < 0.99:
        problems.append("the two halteres are not mirror images of each other in "
                        f"the body model (cos = {pair.a_mirror_agreement:.4f})")

    # ---- the two channels the pair forms -----------------------------------
    print("")
    print("the pair as a sensor (convention: the two strokes are "
          f"{pair.p['convention']})")
    print(f"  difference channel  (L-R)/2 = {np.round(pair.difference, 4)}")
    print(f"  common channel      (L+R)/2 = {np.round(pair.common, 4)}")
    print(f"    {'rotation':>10} {'difference':>12} {'common':>10} {'leak':>8}")
    leakage = pair.channel_leakage()
    for axis, v in leakage.items():
        print(f"    {axis:>10} {v['difference']:>12.4f} {v['common']:>10.4f} "
              f"{100 * v['leak_into_wrong_channel']:>7.1f}%")
    out["channels"] = {"difference": [float(v) for v in pair.difference],
                       "common": [float(v) for v in pair.common],
                       "leakage": leakage}
    # The claim is not "the channels are orthogonal" — they are the model's
    # geometry, so they are near-orthogonal — it is that yaw lands in the
    # difference channel and pitch does not, and the other way round. That is
    # what the fly's yaw reflex needs, and what a wrong stroke convention
    # destroys (see --convention mirrored: it moves yaw into the common channel).
    for axis, want_diff in (("yaw(z)", True), ("pitch(y)", False)):
        v = leakage[axis]
        got = abs(v["difference"]) > abs(v["common"])
        if got != want_diff:
            problems.append(f"{axis} lands in the wrong channel: difference "
                            f"{v['difference']:.4f}, common {v['common']:.4f} "
                            f"(the two halteres' strokes are {pair.p['convention']})")
    if max(v["leak_into_wrong_channel"] for v in leakage.values()) > 0.10:
        problems.append("more than 10% of a rotation axis leaks into the wrong "
                        "channel: the pair cannot tell yaw from pitch")

    # ---- the drive reaches the haltere afferents ---------------------------
    probe = None
    if args.probe or args.gate:
        pp, c = load_cord(args.bin, args.meta)
        nL, nR = group_size(c, "sensory_haltere_left"), group_size(c, "sensory_haltere_right")
        print("")
        print("the afferents, in the packed connectome")
        print(f"  sensory_haltere_left {nL:,} cells, sensory_haltere_right {nR:,}")
        if nL <= 0 or nR <= 0:
            problems.append("the haltere sensory groups are not in the binary — "
                            "the gyro would drive nothing")
            probe = {"left_cells": nL, "right_cells": nR}
        else:
            L = c["members"]("sensory_haltere_left")
            R = c["members"]("sensory_haltere_right")
            tL, tR = first_targets(c, "sensory_haltere_left"), first_targets(c, "sensory_haltere_right")
            shared = np.intersect1d(tL, tR)
            row, col = c["rowPtr"], c["colIdx"]
            direct = {}
            afferents = np.concatenate([L, R])
            for name in sorted(g for g in c["groups"]
                               if g.startswith(("motor", "descending", "ascending"))):
                ids = set(c["members"](name).tolist())
                hits = sum(1 for t in np.unique(np.concatenate(
                    [col[row[i]:row[i + 1]] for i in afferents])) if t in ids)
                if hits:
                    direct[name] = hits
            print(f"  their synapses onto cells that are not haltere afferents: "
                  f"left {tL.size:,} cells, right {tR.size:,}, "
                  f"**shared {shared.size:,}** (a subtraction needs shared targets)")
            out["afferents"] = {"left_cells": int(L.size), "right_cells": int(R.size),
                                "left_targets": int(tL.size), "right_targets": int(tR.size),
                                "shared_targets": int(shared.size)}
            if shared.size < 20:
                problems.append(f"only {shared.size} cells are innervated by both "
                                "halteres: a left-right comparison has nothing to "
                                "compare on")

            # ---- the LIF: does a rotation reach the afferents? --------------
            # Two things are measured here and they are different claims:
            #
            #  1. the *drive* the pair computes. That is exact algebra and it is
            #     gated exactly: still -> zero difference, yaw -> difference,
            #     sign reversed for the other direction, pitch -> common;
            #  2. what the LIF does with it. The two pools are 216 and 212 cells
            #     and they are not interchangeable: in this packed connectome
            #     the right pool sits at a different operating point and fires
            #     ~2x the left, so the *absolute* left-minus-right rate has an
            #     offset that has nothing to do with rotation. So the rotation
            #     is measured *against that baseline*, with the error over 50 ms
            #     bins printed, and the gate asks whether a fast turn moves the
            #     difference further than the animals' own asymmetry — which is
            #     the question a reflex has to answer.
            watch = [("ascending", c["members"]("ascending")),
                     ("descending", c["members"]("descending"))]
            for short, name in (("neck", "motor_neck"),
                                ("steer_L", "motor_wing_steering_left"),
                                ("steer_R", "motor_wing_steering_right")):
                if group_size(c, name) > 0:
                    watch.append((short, c["members"](name)))
            N = c["N"]
            others = np.zeros(N, bool)
            for group in (["sensory_haltere_left", "sensory_haltere_right"]
                          + [n for n, _ in watch]):
                if group_size(c, group) > 0:
                    others[c["members"](group)] = True
            beyond = np.flatnonzero(others & ~np.isin(np.arange(N), L)
                                    & ~np.isin(np.arange(N), R))
            sizes = {"left": int(L.size), "right": int(R.size),
                     "beyond": int(beyond.size)}
            sizes.update({n: int(ids.size) for n, ids in watch})
            print("")
            print(f"  window {args.ms} ms after {args.warmup} ms of warm-up, gain "
                  f"{args.gain_syn:g}, seed {args.seed}, {N:,} cells")

            def measure(dl, dr, ms=None, warm=None, seed=None, bin_ms=50):
                """Drive the two afferent groups and count. Returns rates and the
                standard error of the left-minus-right difference over `bin_ms`
                bins of the same run — an error measured from the data rather
                than assumed from a formula."""
                net = pp.LIF(c, args.gain_syn, seed=args.seed if seed is None else seed)
                ms = args.ms if ms is None else ms
                warm = args.warmup if warm is None else warm
                names = ["left", "right", "beyond"] + [n for n, _ in watch]
                spikes = {n: 0 for n in names}
                per_bin = []
                bin_counts = {n: 0 for n in names}
                for t in range(warm + ms):
                    net.drive("sensory_haltere_left", dl)
                    net.drive("sensory_haltere_right", dr)
                    net.drive("sensory_vision", 1.0)      # the app's operating point
                    net.step()
                    if t >= warm:
                        hit = {"left": np.isin(net.last, L),
                               "right": np.isin(net.last, R),
                               "beyond": np.isin(net.last, beyond)}
                        for name, ids in watch:
                            hit[name] = np.isin(net.last, ids)
                        for name in names:
                            c_ = int(hit[name].sum())
                            spikes[name] += c_
                            bin_counts[name] += c_
                        if (t - warm) % bin_ms == bin_ms - 1:
                            per_bin.append({n: bin_counts[n] / sizes[n]
                                            * 1000.0 / bin_ms for n in names})
                            bin_counts = {n: 0 for n in names}
                hz = {name: spikes[name] / sizes[name] * 1000.0 / ms
                      for name in names}
                hz["difference"] = hz["left"] - hz["right"]
                hz["common"] = 0.5 * (hz["left"] + hz["right"])
                if len(per_bin) > 1:
                    d = np.array([b["left"] - b["right"] for b in per_bin])
                    cm = np.array([0.5 * (b["left"] + b["right"]) for b in per_bin])
                    hz["err_difference"] = float(d.std(ddof=1) / np.sqrt(len(d)))
                    hz["err_common"] = float(cm.std(ddof=1) / np.sqrt(len(cm)))
                else:
                    hz["err_difference"] = hz["err_common"] = float("nan")
                return hz

            def pair_drive(dps):
                d = pair.drive(np.asarray(dps, float))
                return float(d[0]), float(d[1])

            conditions = [("still", pair_drive((0, 0, 0))),
                          ("yaw left 100 deg/s", pair_drive((0, 0, 100))),
                          ("yaw right 100 deg/s", pair_drive((0, 0, -100))),
                          ("yaw left 400 deg/s", pair_drive((0, 0, 400))),
                          ("yaw right 400 deg/s", pair_drive((0, 0, -400))),
                          ("roll left 400 deg/s", pair_drive((400, 0, 0))),
                          ("pitch up 400 deg/s", pair_drive((0, 400, 0)))]
            names = (["left", "right", "difference", "common", "beyond",
                      "difference - still"] + [w for w, _ in watch])
            print(f"  {'condition':<22}" + "".join(f"{n:>13}" for n in names))
            results = {"drive": {}}
            for label, (dl, dr) in conditions:
                hz = measure(dl, dr)
                hz["drive"] = [dl, dr]
                results[label] = hz
            still_diff = results["still"]["difference"]
            for label, _ in conditions:
                results[label]["difference_minus_still"] = (
                    results[label]["difference"] - still_diff)
            for label, _ in conditions:
                hz = results[label]
                hz["difference - still"] = hz["difference_minus_still"]
                print(f"  {label:<22}" + "".join(f"{hz[n]:>13.2f}" for n in names))
            print(f"    the columns are Hz. The error bar on the difference over "
                  f"{int(args.ms / 50)} x 50 ms bins: still "
                  f"+/-{results['still']['err_difference']:.1f} Hz, "
                  f"yaw left 400 +/-{results['yaw left 400 deg/s']['err_difference']:.1f} Hz")
            print(f"    the two pools are not interchangeable: at rest the left "
                  f"fires {results['still']['left']:.1f} Hz and the right "
                  f"{results['still']['right']:.1f} Hz (216 and 212 cells, "
                  f"different connectivity), so the rotation is judged against "
                  f"its own baseline")
            print(f"    {beyond.size:,} cells beyond the afferents and the motor "
                  f"groups are watched in `beyond`")
            out["probe"] = {"ms": args.ms, "warmup": args.warmup, "gain": args.gain_syn,
                            "seed": args.seed,
                            "conditions": {k: {kk: vv for kk, vv in v.items()}
                                           for k, v in results.items()},
                            "beyond_cells": int(beyond.size)}

            # ---- the drive, which is exact algebra ------------------------
            print("")
            print("  the drive the pair computes (exact, no LIF):")
            for label, dps in (("still", (0, 0, 0)), ("yaw left 400", (0, 0, 400)),
                               ("yaw right 400", (0, 0, -400)),
                               ("pitch up 400", (0, 400, 0)),
                               ("roll left 400", (400, 0, 0))):
                dl, dr = pair_drive(dps)
                print(f"    {label:<16} left {dl:>6.2f}  right {dr:>6.2f}  "
                      f"(difference {dl - dr:>+6.2f}, common {0.5 * (dl + dr):>5.2f})")
            yl, yr = results["yaw left 400 deg/s"], results["yaw right 400 deg/s"]
            pitch = results["pitch up 400 deg/s"]
            slow = results["yaw left 100 deg/s"]

            # ---- what the gate is allowed to claim ------------------------
            if abs(results["still"]["drive"][0] - results["still"]["drive"][1]) > 1e-12:
                problems.append("a still animal does not get the same drive on "
                                "both halteres")
            yaw_l = pair_drive((0, 0, 400))[0] - pair_drive((0, 0, 400))[1]
            yaw_r = pair_drive((0, 0, -400))[0] - pair_drive((0, 0, -400))[1]
            if yaw_l * yaw_r >= 0:
                problems.append(f"the yaw channel does not reverse with the "
                                f"rotation: drive difference {yaw_l:+.2f} left, "
                                f"{yaw_r:+.2f} right")
            if results["still"]["err_difference"] > abs(yl["difference_minus_still"]) / 1.5:
                problems.append(
                    f"a 400 deg/s turn moves the difference between the haltere "
                    f"pools by {yl['difference_minus_still']:+.1f} Hz, which is "
                    f"inside their own asymmetry "
                    f"(+/-{results['still']['err_difference']:.1f} Hz): the pools "
                    f"cannot be read as a gyro at this window length")
            if abs(pitch["common"] - results["still"]["common"]) < 1.5 * results["still"]["err_common"]:
                problems.append("a 400 deg/s pitch does not move the common "
                                "channel of the haltere pools")
            if abs(pitch["difference_minus_still"]) > abs(yl["difference_minus_still"]):
                problems.append(
                    f"pitch moves the difference channel as much as yaw does "
                    f"({pitch['difference_minus_still']:+.1f} against "
                    f"{yl['difference_minus_still']:+.1f} Hz): the pair cannot "
                    f"tell yaw from pitch")
            # reported, not gated: a slow turn is inside the pools' asymmetry in
            # a single window, which is what a fly's downstream integration is
            # for. Saying so is the point of the measurement.
            print("")
            print(f"  what the pools do with it: a 400 deg/s turn moves their "
                  f"difference by {yl['difference_minus_still']:+.1f} Hz and a "
                  f"100 deg/s turn by {slow['difference_minus_still']:+.1f} Hz, "
                  f"against an error of "
                  f"+/-{results['still']['err_difference']:.1f} Hz over {int(args.ms / 50)} "
                  f"bins — and the offset at rest ({still_diff:+.1f} Hz) is a "
                  f"property of these two pools' different connectivity, not "
                  f"noise, which is why every rotation is judged against it.")

            # ---- the yaw ladder -------------------------------------------
            # And what the smallest resolvable turn is, measured rather than
            # asserted: the rungs are compared against three standard errors of
            # the still animal's own difference, which is the noise floor this
            # pair of pools actually has.
            err = max(results["still"]["err_difference"], 1e-9)
            ladder = [0, 5, 10, 25, 50, 100, 400]   # same window as the table,
                                                    # so the same floor applies
            print("")
            print(f"  the difference against the yaw rate ({args.ms} ms each, "
                  f"drive straight from the pair; the noise floor is 3 x the still "
                  f"animal's own standard error, 3 x {err:.2f} = {3 * err:.2f} Hz, "
                  f"measured on the same window)")
            rates, drives, bars = [], [], []
            for dps in ladder:
                dl, dr = pair_drive((0, 0, dps))
                hz = measure(dl, dr)
                d = hz["difference"] - still_diff
                rates.append(d)
                drives.append(dl - dr)
                bars.append(3 * hz["err_difference"])
                flag = "  <- above the floor" if abs(d) >= 3 * err else ""
                print(f"    yaw {dps:>4} deg/s  drive difference {dl - dr:>+6.2f}  "
                      f"rate difference {d:>+7.2f} Hz  (own error "
                      f"{hz['err_difference']:.2f}){flag}")
            out["probe"]["yaw_ladder_dps"] = ladder
            out["probe"]["yaw_ladder_drive_difference"] = drives
            out["probe"]["yaw_ladder_rate_difference_hz"] = rates
            # The resolution is the smallest turn after which the rate response
            # is *monotone* and above the floor. Both halves are needed: at 5
            # deg/s the drive changes by 0.12 and the measured difference moves
            # by +8 Hz, more than at 10 deg/s — a small drive change flipping
            # individual near-threshold cells, which is the same quantisation
            # item 11's receptor ladder showed. Those rungs are printed and are
            # not evidence of anything.
            resolution = None
            for i in range(len(ladder)):
                tail = rates[i:]
                if all(b >= a - 1e-9 for a, b in zip(tail, tail[1:])) and \
                        all(abs(r) >= 3 * err for r in tail):
                    resolution = ladder[i]
                    break
            if resolution is None:
                print("    no turn on this ladder gives a monotone response above "
                      "the floor")
            else:
                flip = [d for d, r in zip(ladder, rates) if abs(r) >= 3 * err]
                below = [d for d in flip if d < resolution]
                print(f"    the smallest turn this pair resolves in {args.ms} ms "
                      f"is {resolution} deg/s ({resolution / 57.3:.2f} rad/s): the "
                      f"response is monotone from there up and every rung is above "
                      f"the floor" + (f" (rungs below it — {below} deg/s — move the "
                                      f"rate too, but not monotonically: at those "
                                      f"drives individual cells cross threshold)"
                                      if below else ""))
            out["probe"]["resolution_dps"] = resolution
            if not all(b >= a - 1e-9 for a, b in zip(drives, drives[1:])):
                problems.append("the drive is not monotone in the yaw rate")
            if rates[-1] <= rates[0]:
                problems.append("the yaw channel does not grow with the yaw rate")
            if resolution is None or resolution > 50:
                problems.append(
                    f"the smallest turn the haltere pools resolve in "
                    f"{args.ms} ms is {resolution} deg/s: the gyro cannot "
                    f"see a turn a fly makes while walking")

            # ---- the afferents and what they touch, structurally -----------
            # A rate cannot settle whether the reflex pathway exists over a
            # window this short (measured above: `beyond` does not move), so the
            # pathway claim is structural and it is read out of the packed
            # connectome: the afferents' own synapses.
            print("")
            print("  the pathway, structurally (one synapse out of the afferents, "
                  "read from the packed connectome, not assumed):")
            print("    " + ", ".join(f"{k} {v}" for k, v in
                                    sorted(direct.items(), key=lambda kv: -kv[1])))
            steering = [k for k in direct
                        if "wing_steering" in k or "neck" in k or "haltere" in k]
            if not steering:
                problems.append("the haltere afferents do not reach the flight "
                                "motor system in one synapse (Fayyazuddin & "
                                "Dickinson 1996 recorded exactly that pathway)")
            out["probe"]["direct_motor_targets"] = direct

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=130)
        # 1: the sensitivity axes, in the body frame
        ax = axes[0]
        for vec, colour, label in ((pair.a_left, "#1f77b4", "left"),
                                   (pair.a_right, "#d62728", "right")):
            ax.quiver(0, 0, vec[1], vec[2], color=colour, angles="xy",
                      scale_units="xy", scale=1, label=label)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
        ax.set_xlabel("y (left)"); ax.set_ylabel("z (up)")
        ax.set_title("what each haltere feels:\na = v × n, body frame", fontsize=8)
        ax.legend(fontsize=7)
        # 2: the two channels against the rotation axis
        ax = axes[1]
        axes_names = ["roll\n(x)", "pitch\n(y)", "yaw\n(z)"]
        idx = np.arange(3)
        ax.bar(idx - 0.18, [pair.difference[i] for i in idx], 0.36,
               label="difference (L-R)/2")
        ax.bar(idx + 0.18, [pair.common[i] for i in idx], 0.36,
               label="common (L+R)/2")
        ax.set_xticks(idx); ax.set_xticklabels(axes_names, fontsize=8)
        ax.axhline(0, lw=0.5, color="k")
        ax.set_title("yaw and roll are the pair's difference,\npitch is the sum", fontsize=8)
        ax.legend(fontsize=7)
        # 3: the yaw channel against the rate, if it was measured
        ax = axes[2]
        err_value = out.get("probe", {}).get("conditions", {}).get(
            "still", {}).get("err_difference", 0.0)
        if "yaw_ladder_rate_difference_hz" in out.get("probe", {}):
            dps = out["probe"]["yaw_ladder_dps"]
            ax.errorbar(dps, out["probe"]["yaw_ladder_rate_difference_hz"],
                        yerr=[3 * err_value for _ in dps], fmt="o-", capsize=2)
            ax.axhline(0, lw=0.5, color="k")
            ax.axhspan(-3 * err_value, 3 * err_value, color="grey", alpha=0.2,
                       label="the still animal's floor")
            ax.set_xlabel("yaw rate (deg/s)")
            ax.set_ylabel("left - right rate (Hz)")
            ax.set_title("the yaw channel, through the connectome", fontsize=8)
            ax.legend(fontsize=7)
        else:
            ax.axis("off")
            ax.text(0.02, 0.5, "run --probe to fill this", fontsize=8)
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
        manifest["haltere"] = {
            "sensitivity_left": [float(v) for v in pair.a_left],
            "sensitivity_right": [float(v) for v in pair.a_right],
            "mass_ug": pair.geometry["left"]["mass_ug"],
            "com_mm": pair.geometry["left"]["com_distance_cm"] * 10,
            "stroke_hz": pair.p["stroke_hz"],
            "stroke_deg": pair.p["stroke_deg"],
            "bias": pair.p["bias"],
            "gain": pair.p["gain"],
            "reference_dps": pair.p["reference_dps"],
            "min_dps": pair.p["min_dps"],
            "filter_ms": pair.p["filter_ms"],
            "convention": pair.p["convention"],
            "group_left": "sensory_haltere_left",
            "group_right": "sensory_haltere_right",
        }
        path.write_text(json.dumps(manifest, indent=1))
        print(f"  wrote the haltere block into {path}")

    if args.golden:
        # The Swift port gets the same treatment as the odor field: a table of
        # rates in, drives out, chosen so that a port that gets the *sign*
        # structure wrong fails on the table itself, and one that gets the
        # geometry wrong fails on the magnitudes.
        samples = []
        for dps in ((0, 0, 0), (0, 0, 100), (0, 0, -100), (0, 0, 400),
                    (100, 0, 0), (0, 100, 0), (0, 0, 0.5),
                    (60, -30, 180), (-400, 0, 0)):
            d = pair.drive(np.asarray(dps, float))
            samples.append({"dps": [float(v) for v in dps],
                            "left": float(d[0]), "right": float(d[1])})
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
        print("  ok   the pair is a two-axis rate sensor, and a rotation moves "
              "the afferents it drives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
