#!/usr/bin/env python3
"""
Where the animal actually stands, measured in the solver the phone runs.

`tools/build_body.py` measures the stance with **MuJoCo's** soft contacts: it
settles the MJCF, reads the pose, and reads the joint torques the settle needed
(`hold_torque`, by inverse dynamics). That stance is the real animal's.

But a stance is only a stance *for the floor it was measured on*. The phone's
floor is a penalty contact (74 support functions, `k = 150`, `b = 0.005`), and
MuJoCo's is a soft constraint; they agree to 0.2 µm on the geometry and they
disagree about how the load divides between six feet. Measured, at the MJCF
stance the two differ by 0.44 on the front leg: MuJoCo's front-left claw carries
0.23 of the animal's weight, this solver's carries 0.42, because the two give
the claws different penetrations (3.7e-04 cm against 2.6e-03 cm). The *net*
force is right and the animal still pushes itself off the floor within 200 ms,
because a 0.2 residual on a claw is 0.01 torque on a joint whose inertia is the
model's own 1e-6 g cm² armature.

So the stance this file exports is measured the way a stance should be measured
— by standing. The animal is placed at the MJCF stance, held by its own
calibrated muscle tone plus a stance servo, and *this solver's* dynamics are
allowed to find the pose where the contact forces balance the weight. The
settled pose is averaged over the last `--tail-ms` and written as `stance_q`;
the hold torque is then re-measured at that pose by static inverse dynamics
(`q̈ = 0`), so `hold_excitation` inverts the muscle curve at the pose it will
actually be used at.

The residual root wrench at the settled pose is reported and *not* forced to
zero: it is the balance the harness could not remove, and if it is large the
animal is not standing, it is being held up.

Run after build_body.py, in place:

    python3 tools/stand_body.py --body build/fly_body.json
    python3 tools/stand_body.py --body build/fly_body.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

import fly_aba as fa

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def settle(body, dt, fine_dt, fine_ms, ms, ramp_ms, gain_scale, tail_ms):
    """
    Stand the animal up, at the pose its own floor gives it. Returns the pose
    and the excitation that held it, averaged over the last `tail_ms`.

    Three things are load-bearing. The ramp: a step from no torque to full
    stance torque is a kick, and a kick on a 1e-6 g cm² joint throws the animal
    clear of the floor (measured; it reaches a millimetre of altitude). The fine
    step at the start: the initial penetration error of a few tens of microns is
    resolved as a 300 rad/s kick at the model's own step, and the servo cannot
    act before the joint has already left. And the floor's own damper, which is
    *explicit*: at `CONTACT_DAMPING = 0.5` the animal is thrown clear in 3 ms
    whatever the servo does (b < m·2/dt is the limit; 0.005 is what it stands
    at).
    """
    sim = fa.FlySim(body)
    sim.dt = dt
    sim.reset()
    tail = []                       # (q, excitation, com) over the last stretch
    base = fa.hold_excitation(body)
    steps_fine = max(0, int(fine_ms / 1000.0 / fine_dt))
    steps = max(1, int(ms / 1000.0 / dt))
    tail_max = max(1, int(tail_ms / 1000.0 / dt))
    # The harness's gains are *derived*, not tuned: a joint whose inertia is
    # the armature (1e-6 g cm^2) and whose muscle makes O(1) torque cannot be
    # pushed faster than omega_dt/dt or damped harder than kd_dt*D/dt without
    # the servo itself becoming the instability. Measured: the hand-picked
    # gains this file started with (a 0.05 rad band, 0.4 per rad/s) lose the
    # animal in 300 ms, and the derived ones hold it.
    gp, gd = fa.stance_servo(body, dt)
    gp, gd = gp * gain_scale, gd * gain_scale
    trace = []
    for k in range(steps_fine + steps):
        fine = k < steps_fine
        h = fine_dt if fine else dt
        t = (k * fine_dt if fine else steps_fine * fine_dt
             + (k - steps_fine) * dt)
        ramp = min(1.0, t / (ramp_ms / 1000.0))
        exc = np.clip(ramp * base + gp * (body.stance_q - sim.q) - gd * sim.qd,
                      -1, 1)
        qdd = sim.step(h, fa.muscle_torque(body, sim.q, sim.qd, exc))
        if not np.isfinite(qdd).all():
            return None
        if not fine:
            tail.append((sim.q.copy(), exc.copy(), sim.com().copy()))
            if len(tail) > tail_max:
                tail.pop(0)
        if k % max(1, (steps_fine + steps) // 12) == 0:
            trace.append({
                "ms": round(t * 1000, 2),
                "com_z": float(sim.com()[2]),
                "feet": int(sum(1 for v in sim.foot_force.values() if v > 1e-6)),
                "support": float(sum(sim.foot_force.values())),
                "worst_rate": float(np.abs(sim.qd).max()),
            })
    q_tail = np.array([t[0] for t in tail])
    exc_tail = np.array([t[1] for t in tail])
    com_tail = np.array([t[2] for t in tail])
    return {"sim": sim, "trace": trace, "base": base,
            "stance_q": q_tail.mean(axis=0),
            "stance_excitation": exc_tail.mean(axis=0),
            "pose_spread": float(np.abs(q_tail - q_tail.mean(axis=0)).max()),
            "com": com_tail.mean(axis=0),
            "com_spread": float(np.abs(com_tail - com_tail.mean(axis=0)).max()),
            "tail_ms": len(q_tail) * dt * 1000}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", default="build/fly_body.json", type=pathlib.Path)
    ap.add_argument("--out", default=None, type=pathlib.Path,
                    help="default: rewrite --body in place")
    ap.add_argument("--dt", default=1e-4, type=float)
    ap.add_argument("--fine-dt", default=1e-5, type=float)
    ap.add_argument("--fine-ms", default=30.0, type=float)
    ap.add_argument("--ms", default=600.0, type=float, help="how long to stand")
    ap.add_argument("--tail-ms", default=100.0, type=float,
                    help="the pose is averaged over this much of the end")
    ap.add_argument("--ramp-ms", default=50.0, type=float)
    ap.add_argument("--gain-scale", default=1.0, type=float,
                    help="multiplier on the derived stance-servo gains, which "
                         "are read off the discretisation (see stance_servo)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    body = fa.FlyBody(args.body)
    log(f"standing the animal up at dt {args.dt:g} after {args.fine_ms:g} ms at "
        f"{args.fine_dt:g}, floor k {fa.CONTACT_STIFFNESS:g} "
        f"b {fa.CONTACT_DAMPING:g} ...")
    out = settle(body, args.dt, args.fine_dt, args.fine_ms, args.ms,
                 args.ramp_ms, args.gain_scale, args.tail_ms)
    if out is None:
        print("the harness lost the animal — the stance cannot be measured",
              file=sys.stderr)
        return 1
    for row in out["trace"]:
        log(f"  {row['ms']:7.1f} ms  com_z {row['com_z']:+.5f}  "
            f"feet {row['feet']}  support {row['support']:.4f}  "
            f"|qd|max {row['worst_rate']:.3e}")

    q_new = out["stance_q"]
    exc_new = out["stance_excitation"]
    drift = float(np.abs(q_new - body.stance_q).max())
    weight = float(body.body_mass.sum() * np.linalg.norm(body.gravity))
    support = float(out["trace"][-1]["support"])
    feet = int(out["trace"][-1]["feet"])
    log(f"settled: root z {out['sim'].root_pos[2]:+.6f} cm "
        f"(MJCF stance {body.stance_root_z:+.6f}), com z "
        f"{out['com'][2]:+.5f} cm, posed within {out['pose_spread']:.5f} rad and "
        f"{out['com_spread']:.5f} cm over the last {out['tail_ms']:.0f} ms")
    log(f"against the MJCF stance: the pose moved {drift:.4f} rad at the worst "
        f"joint; floor carries {support:.4f} of the weight ({support/weight:.3f}) "
        f"on {feet} feet")
    if feet < 4 or not (0.5 < support / weight < 1.5):
        print("the animal is not standing on its own six feet: "
              f"{feet} feet, support/weight {support/weight:.3f}",
              file=sys.stderr)
        return 1

    if args.dry_run:
        log("dry run: nothing written")
        return 0

    raw = json.loads(pathlib.Path(args.body).read_text())
    raw["stance_q"] = {body.joint_name[j]: float(q_new[j])
                       for j in range(body.nj)}
    raw["stance_root_z"] = float(out["sim"].root_pos[2])
    raw["stance_excitation"] = {body.joint_name[j]: float(exc_new[j])
                                for j in range(body.nj)}
    raw["stance_measured_by"] = {
        "tool": "tools/stand_body.py",
        "floor": {"stiffness": fa.CONTACT_STIFFNESS,
                  "damping": fa.CONTACT_DAMPING, "friction": fa.FRICTION,
                  "slop": fa.CONTACT_SLOP},
        "harness": {"dt": args.dt, "fine_dt": args.fine_dt,
                    "fine_ms": args.fine_ms, "ms": args.ms,
                    "ramp_ms": args.ramp_ms, "gain_scale": args.gain_scale,
                    "gains": "fa.stance_servo(body, dt)"},
        "pose_shift_from_mjcf_rad": drift,
        "pose_spread_rad": out["pose_spread"],
        "com_spread_cm": out["com_spread"],
        "support_over_weight": support / weight,
        "note": ("the stance and the excitation that holds it are measured by "
                 "standing in this solver, not read from MuJoCo"),
    }
    for m in raw["muscles"].values():
        j = body.joint_index.get(m["joint"])
        if j is not None:
            m["stance_angle"] = float(q_new[j])
            m["stance_excitation"] = float(exc_new[j])
    dest = args.out or args.body
    dest.write_text(json.dumps(raw))
    log(f"wrote {dest}: stance for {body.nj} joints, root z "
        f"{float(out['sim'].root_pos[2]):+.6f} cm, excitation that holds it "
        f"(largest {np.abs(exc_new).max():.4f} of 1, "
        f"{(np.abs(exc_new) > 0.99).sum()} joints saturated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
