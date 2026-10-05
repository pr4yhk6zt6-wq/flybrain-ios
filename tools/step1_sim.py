#!/usr/bin/env python3
"""
STEP 1 — the physics spine: the fly standing on the ground, headless.

This file owns exactly one claim, and it proves it by measurement:

    The anatomical model from `flybody`, once the adhesion actuators that the
    Janelia/DeepMind model ships are used as they are documented, supports the
    animal's own weight on its own six feet, and the pose is stable — nothing
    sags, nothing sinks, nothing pops.

There is no controller here and no behaviour. It is the ground truth that
everything later has to beat: any walking controller must stand on this spine.

Muscle model, as shipped (not invented here):
  the 70 joint actuators are affine position actuators,
        force = gain * (ctrl - qpos)
  which is MuJoCo's standard linear muscle approximation — a spring whose rest
  length is commanded by the motor neuron. ctrl = 0 therefore commands "every
  joint at its anatomical rest angle", which is the standing pose of the scan.

Usage
  python3 tools/step1_sim.py --seconds 2.0 --render out/stand.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def foot_bodies(model, mujoco):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    return [i for i, n in enumerate(names)
            if n and n.startswith(("claw_", "tarsus4_", "tarsus3_"))]


def run(mjcf, seconds, dt_report=0.01):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # The fly starts standing on the floor in the sampled rest pose. Record
    # where that is, because it is the target the physics has to hold.
    rest_com = data.subtree_com[1].copy()
    feet = foot_bodies(model, mujoco)
    rest_foot_z = min(data.xpos[b][2] for b in feet)

    ctrl = np.zeros(model.nu)          # every muscle at its rest length
    n_steps = int(seconds / model.opt.timestep)
    every = max(1, int(dt_report / model.opt.timestep))

    trace = []
    contact_counts = []
    for step in range(n_steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        ncon_feet = sum(1 for i in range(data.ncon)
                        if model.geom_bodyid[data.contact[i].geom1] in feet
                        or model.geom_bodyid[data.contact[i].geom2] in feet)
        if step % every == 0:
            trace.append({
                "t": float(data.time),
                "com_z_mm": float(data.subtree_com[1][2]) * 10,
                "com_drift_mm": float(np.linalg.norm(
                    (data.subtree_com[1] - rest_com)[:2])) * 10,
                "min_foot_z_mm": float(min(data.xpos[b][2] for b in feet)) * 10,
                "foot_contacts": int(ncon_feet),
                "ncon": int(data.ncon),
                "max_qvel": float(np.abs(data.qvel).max()),
            })
            contact_counts.append(ncon_feet)

    return model, data, trace, rest_foot_z * 10, feet


def summarise(trace, model):
    com = np.array([p["com_z_mm"] for p in trace])
    drift = np.array([p["com_drift_mm"] for p in trace])
    feet = np.array([p["foot_contacts"] for p in trace])
    qvel = np.array([p["max_qvel"] for p in trace])
    half = len(trace) // 2
    return {
        "com_z_mm_first": float(com[0]),
        "com_z_mm_last": float(com[-1]),
        "com_z_mm_change": float(com[-1] - com[0]),
        "com_z_mm_sd_second_half": float(com[half:].std()),
        "horizontal_drift_mm": float(drift[-1]),
        "feet_touching_floor_mean_second_half": float(feet[half:].mean()),
        "feet_touching_frames_0_second_half": int((feet[:half] == 0).sum()),
        "max_joint_speed_second_half": float(qvel[half:].max()),
        "settled": bool(abs(com[-1] - com[0]) < 0.10
                        and drift[-1] < 0.05
                        and qvel[half:].max() < 1.0),
    }


def render(model, data, out: pathlib.Path, width=900, height=640):
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = mujoco.Renderer(model, height, width)
    # A free camera placed by hand: the model's own two cameras are the fly's
    # own eyes and point wherever the fly points, which is not useful here.
    views = [("side", 90, -6, 0.42), ("front", 0, -14, 0.42),
             ("three-quarter", 138, -26, 0.50), ("from above", 135, -62, 0.50)]

    fig, axes = plt.subplots(1, len(views), figsize=(3.5 * len(views), 2.7))
    if len(views) == 1:
        axes = [axes]
    for ax, (label, azim, elev, dist) in zip(axes, views):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance = dist                 # cm: the whole animal is 0.3 cm
        cam.elevation = elev
        cam.azimuth = azim
        cam.lookat[:] = [0, 0, -0.06]
        r.update_scene(data, camera=cam)
        ax.imshow(r.render())
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle("flybody · 67 parts · 102 DOF · 0.985 mg — standing on its own six feet",
                 fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flybody", default="data/flybody", type=pathlib.Path)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--render", type=pathlib.Path, default=None)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    mjcf = a.flybody / "flybody-main/flybody/fruitfly/assets/floor.xml"
    if not mjcf.exists():
        print(f"missing {mjcf}", file=sys.stderr)
        return 2

    print(f"standing the fly for {a.seconds} s of MuJoCo time "
          f"(100 µs steps, {int(a.seconds/1e-4)} of them) …")
    model, data, trace, rest_foot_z, feet = run(mjcf, a.seconds)
    s = summarise(trace, model)

    print(f"  COM height      {s['com_z_mm_first']:+.4f} -> {s['com_z_mm_last']:+.4f} mm "
          f"(change {s['com_z_mm_change']:+.5f} mm, second-half sd {s['com_z_mm_sd_second_half']:.5f})")
    print(f"  horizontal drift {s['horizontal_drift_mm']:.5f} mm")
    print(f"  feet on the floor {s['feet_touching_floor_mean_second_half']:.2f} of 6 "
          f"(frames with none: {s['feet_touching_frames_0_second_half']})")
    print(f"  max joint speed, second half {s['max_joint_speed_second_half']:.4f} rad/s")
    print(f"  VERDICT: {'STANDING' if s['settled'] else 'NOT STABLE'}")

    if a.render:
        render(model, data, a.render)
        print(f"  wrote {a.render}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"summary": s, "trace": trace}, indent=1))
        print(f"  wrote {a.json}")

    return 0 if s["settled"] else 1


if __name__ == "__main__":
    sys.exit(main())
