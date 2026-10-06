#!/usr/bin/env python3
"""
Six legs, one cord, one animal — the closed loop, headless.

`tools/gait_probe.py` measured the cord with a *constant* drive and found no
tripod. That is the cord as a rate source, not the cord as an animal: in the app
the pools are wired to muscles, the muscles to a body, and the body's joints and
foot loads back into the sense organs the cord is listening to. This tool is that
loop, and nothing else:

    connectome (pool_probe's LIF, 1 ms per step)
        -> each pool's rate -> its muscle's activation (tau 60 ms, assumption #7)
            -> the balance of each joint's antagonist pair
                -> excitation = posture + gain*(balance − reference)   (FlyCord)
                    -> fly_aba's muscles -> torque -> the articulated body
                        -> joint angles and foot forces
                            -> chordotonal and campaniform organ drives (#11, #12)
                                -> back into the connectome

Every law in it is the one the app runs (`FlyCord.swift`, `FlyLiveBody.swift`),
and every number comes from the same assets (`build/fly_body.json`,
`build/flybanc.bin`), so what it finds can be ported rather than reinterpreted.

What it measures: whether six legs wired this way *step*. A leg is stepping if
its foot load rises above and falls back below the load it carries standing, and
a gait is alternating if the two tripods' loads move against each other. Both are
reported per run, next to the control runs that say what the body does with the
cord disconnected and with the loop open.

    python3 tools/walk_loop.py --ms 400
    python3 tools/walk_loop.py --ms 800 --gain 6 --json /tmp/walk.json

Cost: the LIF is ~25 ms of wall clock per millisecond of cord, the body ~1 ms per
100 us substep, so 400 ms of loop is ~20 s.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fly_aba as fa                                            # noqa: E402

TONE = 2.5          # assumption #5, the brain's tone
KAPPA = 1.0         # assumption #11 — `FlyCordSettings.kappa`: how much of the
                    # joint's anatomical range the chordotonal organ spans its
                    # drive over. 1 = the whole range. It is a named constant
                    # here because it was a literal once: this tool ran the organ
                    # at 0.5 (`value = tone * (1 - 0.5 * x)`) while
                    # `FlyCord.swift` and step 3 both ran it at 1.0, so §4 and §5
                    # of reports/item4_walking.md described an animal whose organ
                    # gain was half the phone's. `tools/check_cord_laws.py` now
                    # fails the build if the three files stop agreeing, and if
                    # either organ law stops using its named κ.
DRIVE_GROUP = "descending"   # the population the brain's command lands on.
                    # `descending` (1,316 cells) is what the app drives. The
                    # other candidate is `premotor:multileg` (788 cells, the
                    # only cells in BANC that reach more than one leg's pools):
                    # driving *it* asks whether the tripod's leg set is chosen
                    # by the cord's own coupling or by the body's geometry,
                    # which is the question §7 of reports/item4_walking.md
                    # leaves open.
POLARITY = 1.0      # assumption #11 — +1: flexion stretches the organ, which
                    # silences it (FeCO, published). -1 reverses the reflex.
COMMAND_GAIN = 0.5  # assumption #24 — `FlyCordSettings.gain`, excitation per unit
                    # of balance. Not to be confused with the LIF's current gain
                    # (`SimulationEngine.gain`, 12): that one scales synaptic
                    # current into membrane, this one scales a balance into a
                    # muscle command, and they are three and a half orders of
                    # magnitude apart in meaning. Passing 12 here saturates the
                    # command at ±1 on every joint and thrashes the body.
TAU_MUSCLE_MS = 60.0  # assumption #7
CALIBRATE_MS = 300.0  # assumption #10
SET_B = [("T1", "right"), ("T2", "left"), ("T3", "right")]


def load_pool_probe():
    spec = importlib.util.spec_from_file_location("pool_probe", HERE / "pool_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Cord:
    """`FlyCord.swift`, same laws and same order of operations.

    The one difference is that this one is told the state instead of reading it
    off a phone: `update(angles, rates, loads)` takes the joint angles, the foot
    loads and the pool rates the connectome just produced.
    """

    def __init__(self, body, asset, gain, tone=TONE,
                 channels=("chordotonal", "campaniform"),
                 closed_calibration=False, kappa=KAPPA, polarity=POLARITY,
                 drive_group=DRIVE_GROUP, drive_group_b=None):
        """`gain` is the *command* gain (assumption #24), not the LIF's.

        `closed_calibration` decides what the reference balance is measured
        *in*. The app measures it with the organs clamped at their standing
        value (assumption #10) and then runs with the organs live, so the
        network is calibrated in one operating point and run in another — and
        the difference shows up as a constant offset on every joint. With this
        on, the reference is the balance the closed loop itself holds while
        standing, which is the regime it will run in.
        """
        self.body = body
        self.tone = tone
        self.gain = gain
        self.kappa = kappa
        self.polarity = polarity
        self.drive_group = drive_group
        # A second population, driven half a cycle out of phase with the first.
        # This is the muscle-axis experiment (reports/item4_walking.md §10): the
        # leg axis has no lever (765 of the 788 multi-leg cells reach both
        # tripods), so the command is split between two *muscle* populations
        # instead and the question is whether the cord turns that antiphase
        # command into alternating legs. With no second group this does nothing
        # and the loop is the one §6-§8 ran.
        self.drive_group_b = drive_group_b
        self.closed_calibration = closed_calibration
        self.pattern = None          # (Hz, amplitude) on the descending group
        self.channels = set(channels)

        hinge_of = {j["name"]: i for i, j in enumerate(body.hinges)}
        self.pools = []
        self.organs = []
        for leg_key, leg in asset["legs"].items():
            for name, p in leg["pools"].items():
                full = leg["joints"][p["joint"]]["name"]
                if full not in hinge_of:
                    continue
                self.pools.append(dict(leg=leg_key, name=name, group=p["group"],
                                       hinge=hinge_of[full],
                                       sign=1 if p["qpos_sign"] >= 0 else -1))
            for kind, o in leg["organs"].items():
                joint = o.get("joint") or leg["joints"]["tibia"]["name"]
                j = leg["joints"]["tibia"]
                span = abs(j["range"][1] - j["range"][0])
                self.organs.append(dict(leg=leg_key, kind=kind, group=o["group"],
                                        hinge=hinge_of[joint], rest=j["rest"],
                                        half_span=max(1e-6, span / 2)))
        self.pools.sort(key=lambda p: p["group"])
        self.organs.sort(key=lambda o: o["group"])
        self.driven = sorted({p["hinge"] for p in self.pools})

        n = len(self.pools)
        self.rate = np.zeros(n)
        self.activation = np.zeros(n)
        self.balance = np.zeros(len(self.driven))
        self.reference = np.zeros(len(self.driven))
        self._calib_sum = np.zeros(len(self.driven))
        self.calibrating_ms = 0.0
        self.phase = "calibrating"
        self.reference_load = {}
        # Per-phase readouts: the calibration half and the closed half are
        # different regimes (the first has the organs clamped at tone), and a
        # number averaged over both describes neither.
        self.history = {"calib_desc": [], "calib_pool": [], "run_desc": [],
                        "run_pool": [], "run_offset": []}
        # The load each leg carries *standing* — the campaniform organ's own
        # normalisation. Measured across the calibration window and taken as a
        # median, because one sample of a penalty contact is a number with a
        # millisecond of the solver's noise in it.
        self._calib_loads = {leg: [] for leg in self.body.legs}
        self.descending_rate = 0.0
        self.organ_rate = 0.0

    # -- the cord, one millisecond ------------------------------------------
    def update(self, net, dt_ms, angles, loads, feedback=True, t_ms=0.0):
        """Drive the net, step it, read the rates back, return the offset.

        `feedback=False` holds every organ at its standing value, so the cord's
        rates stop depending on the body: what is left is the cord driving the
        muscles of a body that cannot answer it. That is the control for "is the
        motion the loop's or the network's".
        """
        # The brain's command. A constant tone is a *hold*; real descending
        # commands are patterned, and the pattern is the experiment: does the
        # cord's own wiring turn a modulated command into alternating tripods,
        # or do all six legs simply follow the drive in phase? `pattern` is
        # (frequency Hz, amplitude in the same units as the tone).
        tone = self.tone
        tone_b = None
        if self.pattern:
            f, a = self.pattern
            tone = max(0.0, self.tone + a * np.sin(2 * np.pi * f * t_ms / 1000.0))
            if self.drive_group_b:
                # Half a cycle later — same mean, same amplitude, opposite phase.
                tone_b = max(0.0, self.tone - a * np.sin(2 * np.pi * f * t_ms / 1000.0))
        net.drive(self.drive_group, tone)
        if self.drive_group_b:
            net.drive(self.drive_group_b, self.tone if tone_b is None else tone_b)
        # The app has no camera in world mode until the user grants one, and
        # `BrainEngine.load` gives the retina a flat 1.0 so the brain is alive
        # the moment the view appears. Same drive here, so this loop's operating
        # point is the app's and not a darker one.
        net.drive("sensory_vision", 1.0)
        n_org = 0
        org_sum = 0.0
        clamped = (not feedback) or (self.phase == "calibrating"
                                     and not self.closed_calibration)
        for o in self.organs:
            if clamped or o["kind"] not in self.channels:
                value = self.tone
            elif o["kind"] == "chordotonal":
                x = np.clip((angles[o["hinge"]] - o["rest"]) / o["half_span"], -1, 1)
                value = self.tone * (1 - self.polarity * self.kappa * x)  # #11
            else:
                ref = self.reference_load.get(o["leg"], 0.0)
                value = self.tone * (loads.get(o["leg"], 0.0) / ref) if ref > 1e-9 else self.tone
            net.drive(o["group"], max(0.0, value))
            if o["kind"] in self.channels:
                org_sum += max(0.0, value)
                n_org += 1
        self.organ_rate = org_sum / n_org if n_org else 0.0

        net.step()
        counts = net.counts([p["group"] for p in self.pools])
        sizes = np.array([max(len(net.members(p["group"])), 1) for p in self.pools])
        self.rate = counts / sizes * 1000.0 / dt_ms      # Hz, per pool
        self.descending_rate = float(
            net.counts(["descending"])[0] / max(len(net.members("descending")), 1)
            * 1000.0 / dt_ms)
        if self.phase == "calibrating":
            for leg, force in loads.items():
                self._calib_loads.setdefault(leg, []).append(float(force))
        key = "calib" if self.phase == "calibrating" else "run"
        self.history[key + "_desc"].append(self.descending_rate)
        self.history[key + "_pool"].append(float(self.rate.mean()))

        alpha = min(1.0, dt_ms / TAU_MUSCLE_MS)
        self.activation += (self.rate - self.activation) * alpha

        for k, hinge in enumerate(self.driven):
            up = down = 0.0
            for i, p in enumerate(self.pools):
                if p["hinge"] != hinge:
                    continue
                if p["sign"] > 0:
                    up += self.activation[i]
                else:
                    down += self.activation[i]
            total = up + down
            self.balance[k] = up / total if total > 1e-9 else 0.0

        if self.phase == "calibrating":
            self._calib_sum += self.balance
            self.calibrating_ms += dt_ms
            if self.calibrating_ms >= CALIBRATE_MS:
                self.reference = self._calib_sum / max(1, self.calibrating_ms / dt_ms)
                tail = {leg: v[-50:] for leg, v in self._calib_loads.items() if len(v) >= 10}
                self.reference_load = {leg: float(np.median(v)) for leg, v in tail.items()}
                self.phase = "running"

        offset = np.zeros(self.body.nj)
        if self.phase == "running":
            for k, hinge in enumerate(self.driven):
                offset[hinge] = np.clip(self.gain * (self.balance[k] - self.reference[k]),
                                        -1, 1)
            self.history["run_offset"].append(float(np.abs(offset).max()))
        # While calibrating there is nothing to command *towards* yet: `reference`
        # is still zero and `gain*(balance - 0)` would be a push of up to gain,
        # applied for 300 ms to an animal that is supposed to be holding its
        # stance so the stance can be measured. Step 3's "clamped" mode commands
        # every joint to the rest pose for exactly this phase; so does this, by
        # returning no offset at all and letting muscle tone hold the animal.
        return offset


def run_loop(args) -> dict:
    pp = load_pool_probe()
    body = fa.FlyBody(args.asset)
    asset = body.raw
    c = pp.load(args.bin, args.meta)
    sim = fa.FlySim(body, limits=True)
    sim.reset()
    posture = fa.hold_excitation(body)
    dt = body.dt_default

    # Settle the stance the way `FlyLiveBody.startUp` does: twenty milliseconds
    # at a tenth of the step, so the stiff stance mode is not excited by the
    # first step. Muscle tone only — no cord, no offset.
    for _ in range(int(round(0.020 / (dt / 10)))):
        sim.step(dt / 10, fa.muscle_torque(body, sim.q, sim.qd, posture))

    net = pp.LIF(c, args.gain, seed=args.seed)
    cord = Cord(body, asset, args.command_gain, tone=args.tone,
                closed_calibration=args.closed_calibration,
                kappa=args.kappa, polarity=args.polarity,
                drive_group=args.drive_group,
                drive_group_b=args.drive_group_b)
    if args.desc_pattern:
        f, a = args.desc_pattern
        cord.pattern = (f, a)

    legs = sorted(asset["legs"].keys())
    knee_hinge = {leg: next(o["hinge"] for o in cord.organs
                            if o["leg"] == leg and o["kind"] == "chordotonal")
                  for leg in legs}
    sub = max(1, int(round(1.0 / (dt * 1000.0))))       # physics steps per cord ms

    trace = {leg: dict(load=[], knee=[]) for leg in legs}
    com_xy = []
    offsets_seen = []
    t0 = time.time()
    for t in range(args.ms):
        angles = sim.q
        loads = dict(sim.foot_force)
        if args.no_cord:
            # Muscle tone only: the connectome is not run at all, so the body's
            # own stability is the whole story. Nothing below this line can
            # affect the cord, and the cord cannot affect the body.
            exc = np.clip(posture, -1.0, 1.0)
            for _ in range(sub):
                sim.step(dt, fa.muscle_torque(body, sim.q, sim.qd, exc))
            for leg in legs:
                trace[leg]["load"].append(sim.foot_force[leg])
                trace[leg]["knee"].append(sim.q[knee_hinge[leg]])
            com_xy.append(sim.com().copy())
            continue
        # The control: the cord runs, but its organs are never told what the body
        # did. What is left is the cord's own dynamics plus physics.
        offset = cord.update(net, 1.0, angles, loads,
                             feedback=not args.no_feedback, t_ms=float(t))
        offsets_seen.append(float(np.abs(offset).max()))
        exc = np.clip(posture + offset, -1.0, 1.0)
        for _ in range(sub):
            sim.step(dt, fa.muscle_torque(body, sim.q, sim.qd, exc))
        for leg in legs:
            trace[leg]["load"].append(sim.foot_force[leg])
            trace[leg]["knee"].append(sim.q[knee_hinge[leg]])
        com_xy.append(sim.com().copy())

    wall = time.time() - t0
    com = np.array(com_xy)

    # ---- what happened ----------------------------------------------------
    out = {"ms": args.ms, "gain": args.gain, "command_gain": args.command_gain,
           "tone": args.tone, "kappa": args.kappa, "polarity": args.polarity,
           "drive_group": args.drive_group,
           "drive_group_b": args.drive_group_b,
           "cord": not args.no_cord, "feedback": not args.no_feedback,
           "phase_at_end": cord.phase, "wall_s": wall, "legs": {},
           "history": {k: [round(v, 3) for v in vals] for k, vals in cord.history.items()}}
    ref_load = {leg: float(np.median(np.array(trace[leg]["load"])[:20]))
                for leg in legs}
    for leg in legs:
        load = np.array(trace[leg]["load"])
        knee = np.array(trace[leg]["knee"])
        ref = max(ref_load[leg], 1e-9)
        rel = load / ref
        # A step is the load leaving the leg: below a quarter of standing. A
        # foot that never leaves the floor is not stepping, it is planted.
        lifted = rel < 0.25
        crossings = int(np.sum(lifted[1:] & ~lifted[:-1]))
        out["legs"][leg] = dict(
            load_mean=float(load.mean()), load_ref=ref,
            lifted_frac=float(lifted.mean()), steps=crossings,
            knee_min=float(knee.min()), knee_max=float(knee.max()),
            knee_ptp=float(np.ptp(knee)))
    com0 = com[0]
    out["com_drift"] = float(np.linalg.norm(com[-1][:2] - com0[:2]))
    out["com_z"] = [float(com[0][2]), float(com[-1][2])]
    out["legs_leaving_floor"] = int(sum(1 for leg in legs if out["legs"][leg]["steps"] > 0))

    # Tripod index over the load traces: the same measure gait_probe.py applies
    # to the pools' rates, now applied to what the feet actually did.
    def tripod(z):
        def corr(a, b):
            za, zb = z[a] - z[a].mean(), z[b] - z[b].mean()
            if za.std() < 1e-12 or zb.std() < 1e-12:
                return 0.0
            return float((za * zb).mean() / (za.std() * zb.std()))
        within, across = [], []
        for i, a in enumerate(legs):
            for b in legs[i + 1:]:
                sa = (a.split("_")[0], a.split("_")[1])
                sb = (b.split("_")[0], b.split("_")[1])
                (within if (sa in SET_B) == (sb in SET_B) else across).append(corr(a, b))
        return float(np.mean(within)), float(np.mean(across))

    loads = {leg: np.array(trace[leg]["load"]) for leg in legs}
    w, a = tripod(loads)
    out["load_within"], out["load_across"], out["load_tripod_index"] = w, a, w - a
    knees = {leg: np.array(trace[leg]["knee"]) for leg in legs}
    w2, a2 = tripod(knees)
    out["knee_within"], out["knee_across"], out["knee_tripod_index"] = w2, a2, w2 - a2
    out["offset_max"] = float(np.max(offsets_seen))
    out["descending_rate"] = cord.descending_rate
    out["mean_pool_rate"] = float(cord.rate.mean())
    out["mean_activation"] = float(cord.activation.mean())
    return out, trace


def show(out: dict, trace: dict) -> None:
    what = ("cord disconnected (muscle tone only)" if not out["cord"] else
            ("open loop (organs not told what the body did)" if not out["feedback"]
             else "closed loop"))
    print(f"\n{out['ms']} ms of {what} — LIF gain {out['gain']:g}, command gain "
          f"{out['command_gain']:g}, tone {out['tone']:g}, kappa {out['kappa']:g}, "
          f"polarity {out['polarity']:+.0f}"
          f"  [{out['wall_s']:.0f} s wall]")
    if out["cord"]:
        h = out["history"]
        def mean(x):
            return float(np.mean(x)) if x else float("nan")
        print(f"  cord, calibrating: desc {mean(h['calib_desc']):5.1f} Hz · pool "
              f"{mean(h['calib_pool']):5.1f} Hz   |   closed loop: desc "
              f"{mean(h['run_desc']):5.1f} Hz · pool {mean(h['run_pool']):5.1f} Hz · "
              f"max |offset| {out['offset_max']:.2f}")
    print(f"\n  {'leg':<10}{'load µN':>9}{'of stance':>11}{'airborne':>10}"
          f"{'steps':>7}{'knee ptp rad':>14}")
    for leg, d in sorted(out["legs"].items()):
        print(f"  {leg:<10}{d['load_mean']:9.2f}{d['load_mean'] / max(d['load_ref'], 1e-9):11.2f}"
              f"{d['lifted_frac'] * 100:9.1f}%{d['steps']:7d}{d['knee_ptp']:14.3f}")
    print(f"\n  legs that left the floor at least once: {out['legs_leaving_floor']}/6")
    print(f"  COM: z {out['com_z'][0]:+.5f} -> {out['com_z'][1]:+.5f} cm, "
          f"horizontal drift {out['com_drift']:.5f} cm")
    print(f"  foot-load tripod index {out['load_tripod_index']:+.3f} "
          f"(within {out['load_within']:+.3f}, across {out['load_across']:+.3f})")
    print(f"  knee tripod index      {out['knee_tripod_index']:+.3f} "
          f"(within {out['knee_within']:+.3f}, across {out['knee_across']:+.3f})")
    if out["legs_leaving_floor"] == 0:
        print("\n  ⇒ no leg left the floor: this loop does not walk. It stands —")
        print("    and the report says so rather than calling the stance a gait.")
    else:
        print(f"\n  ⇒ {out['legs_leaving_floor']} legs leave the floor. Whether that is")
        print("    a gait or a stumble is what the tripod index and the per-leg")
        print("    traces above are for; a gait needs the two tripods to alternate.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="build/fly_body.json")
    ap.add_argument("--bin", default=pathlib.Path("build/flybanc.bin"), type=pathlib.Path)
    ap.add_argument("--meta", default=pathlib.Path("build/flybanc_meta.json"),
                    type=pathlib.Path)
    ap.add_argument("--ms", type=int, default=400)
    ap.add_argument("--gain", type=float, default=12.0,
                    help="the LIF's current gain (SimulationEngine.gain)")
    ap.add_argument("--command-gain", type=float, default=COMMAND_GAIN,
                    help="FlyCord's balance -> excitation gain (assumption #24)")
    ap.add_argument("--tone", type=float, default=TONE)
    ap.add_argument("--kappa", type=float, default=KAPPA,
                    help="the chordotonal organ's range fractionation "
                         "(assumption #11, FlyCordSettings.kappa). Default is "
                         "the value the phone runs; step 3 sweeps 1/4/16.")
    ap.add_argument("--polarity", type=float, default=POLARITY,
                    help="+1: flexion stretches the organ and silences it "
                         "(published). -1 is the control experiment.")
    ap.add_argument("--drive-group", default=DRIVE_GROUP,
                    help="which population the brain's command lands on: "
                         "`descending` (the app) or `premotor:multileg` (the "
                         "788 cells that reach more than one leg's pools)")
    ap.add_argument("--drive-group-b", default=None,
                    help="a second population, driven antiphase to --drive-group "
                         "when --desc-pattern is given (the muscle-axis "
                         "experiment, reports/item4_walking.md §10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cord", action="store_true", help="muscle tone only")
    ap.add_argument("--no-feedback", action="store_true",
                    help="the cord runs, the organs are held at tone")
    ap.add_argument("--desc-pattern", type=lambda s: tuple(float(x) for x in s.split(":")),
                    default=None, metavar="HZ:AMP",
                    help="modulate the descending command: tone + AMP*sin(2 pi HZ t)")
    ap.add_argument("--closed-calibration", action="store_true",
                    help="measure the reference balance with the organs live "
                         "(the regime the loop runs in) instead of clamped")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out, trace = run_loop(args)
    show(out, trace)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
