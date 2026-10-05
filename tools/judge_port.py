#!/usr/bin/env python3
"""
Settle a disagreement between the two solvers with MuJoCo as the judge.

`tools/fly_aba.py` is the reference and `FlyBrain/Sources/FlyDynamics.swift`
must reproduce its golden trace, but when the two disagree the reference is not
automatically right — it is a transliteration's source, not a ground truth. The
ground truth is MuJoCo, which both are trying to reproduce. This script hands
one state to all three and reports who is closer.

The comparison is on the **joint rates after one step**, not on `qacc`:

  * MuJoCo's own `d.qacc` is not the acceleration it integrates. Joint damping
    is handled implicitly inside `mj_step`'s integrator, so `d.qacc` is off by
    tens of percent while the step it produces is right to 1e-6 relative.
    Comparing `qacc` makes a correct solver look broken (measured: median
    relative 0.28 on a state where the rates after the step agree to 2.5e-6).
  * The rates after the step are what both solvers actually integrate, with
    their damping folded in exactly the same way, so they are comparable.

`--limits` matters: the port's hard stops are a clamp and a rate-kill, MuJoCo's
are soft constraints, and `FlySim` can be built either way. States whose joints
sit outside their range are clamped by one side and not the other, so the
default here clips every state inside every joint's range and nothing is
clamped on either side.

  python3 tools/judge_port.py --swift /tmp/golden --seeds 5 11 23 41

Compile the driver first:

  swiftc -O FlyBrain/Sources/FlyDynamics.swift local/main.swift -o /tmp/golden
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fly_aba as fa                                          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MJCF = (ROOT / "data/flybody/flybody-main/flybody/fruitfly/"
                "assets/fruitfly.xml")


def one_state(body, mjcf, seed: int, limits: bool) -> dict:
    """One step from one random state, in all three implementations."""
    import mujoco as mj
    dt = body.dt_default
    rng = np.random.default_rng(seed)
    lo = np.where(body.limited, body.lo + 1e-3, -np.inf)
    hi = np.where(body.limited, body.hi - 1e-3, np.inf)
    q0 = np.clip(rng.uniform(-0.3, 0.3, body.nj), lo, hi)
    qd0 = rng.uniform(-2, 2, body.nj)
    v0 = rng.uniform(-1, 1, 3)
    om0 = rng.uniform(-3, 3, 3)

    sim = fa.FlySim(body, floor_z=-1e6, limits=limits)
    sim.dt = dt
    sim.reset(root_z=0.0, q=q0, qd=qd0, v=v0, omega=om0)
    return {"dt": dt, "state": {"q": q0, "qd": qd0, "v": v0, "omega": om0}, "sim": sim}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", default=str(ROOT / "build/fly_body.json"),
                    type=pathlib.Path)
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF), type=pathlib.Path)
    ap.add_argument("--swift", default="/tmp/golden",
                    help="the compiled local/ driver")
    ap.add_argument("--golden", default=str(ROOT / "build/fly_golden.json"),
                    type=pathlib.Path, help="only used to decode the asset")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[5, 11, 23, 41, 57, 63, 77, 91, 103, 121])
    ap.add_argument("--limits", default="true", choices=["true", "false"],
                    help="joint limits: keep the same on both sides")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="fail if the port and the reference differ by more")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    import mujoco as mj
    limits = args.limits == "true"
    body = fa.FlyBody(args.body)
    print(f"body {args.body.name}: {body.nj} hinges, {body.nbody - 1} parts, "
          f"limits {limits}, the port compiled at {args.swift}")

    rows = []
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        state_path = pathlib.Path(tmp) / "state.json"
        for seed in args.seeds:
            case = one_state(body, args.mjcf, seed, limits)
            sim, dt, st = case["sim"], case["dt"], case["state"]
            m, d = fa._mujoco_pair(body, sim, args.mjcf)
            for j, jj in enumerate(body.hinges):
                d.qfrc_applied[jj["dofadr"]] = 0.0
            mj.mj_step(m, d)
            qv_mj = np.array([d.qvel[jj["dofadr"]] for jj in body.hinges])
            sim.step(dt, np.zeros(body.nj))
            qv_py = sim.qd.copy()
            json.dump({"q": st["q"].tolist(), "qd": st["qd"].tolist(),
                       "v": st["v"].tolist(), "omega": st["omega"].tolist(),
                       "rootPos": [0.0, 0.0, 0.0], "rootQuat": [1.0, 0.0, 0.0, 0.0],
                       "tau": [0.0] * body.nj, "dt": dt},
                      state_path.open("w"))
            out = subprocess.run([args.swift, str(args.body), str(args.golden),
                                  "judge", str(state_path)],
                                 capture_output=True, text=True)
            if "V " not in out.stdout:
                print("the driver printed no rates:", out.stdout[-400:],
                      out.stderr[-400:], file=sys.stderr)
                return 2
            qv_sw = np.zeros(body.nj)
            for line in out.stdout.split("\n"):
                if line.startswith("V "):
                    _, j, val = line.split()
                    qv_sw[int(j)] = float(val)
            dp = float(np.abs(qv_py - qv_mj).max())
            ds = float(np.abs(qv_sw - qv_mj).max())
            dsp = float(np.abs(qv_sw - qv_py).max())
            ok = ok and dsp < args.tol
            rows.append({"seed": seed, "python_vs_mujoco": dp,
                         "swift_vs_mujoco": ds, "swift_vs_python": dsp})
            flag = "ok  " if dsp < args.tol else "FAIL"
            print(f"  {flag} seed {seed:4d}  python-mujoco {dp:.3e}   "
                  f"swift-mujoco {ds:.3e}   swift-python {dsp:.3e} rad/s")

    worst_sw = max(r["swift_vs_mujoco"] for r in rows)
    worst_py = max(r["python_vs_mujoco"] for r in rows)
    print(f"worst over {len(rows)} states: python-mujoco {worst_py:.3e}   "
          f"swift-mujoco {worst_sw:.3e}")
    print("VERDICT: " + ("the port is the same solver as the reference"
                         if ok else "the port still disagrees"))
    if args.json:
        args.json.write_text(json.dumps({"ok": ok, "seeds": rows}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
