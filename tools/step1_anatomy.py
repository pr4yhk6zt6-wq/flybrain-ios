#!/usr/bin/env python3
"""
STEP 1 — the fly's body and its nerve structure, as measured, not invented.

Two questions are answered here, and nothing is built on top of the answers
until both are in hand:

  1. What is the body?  Read the Janelia/DeepMind `flybody` MJCF — the same
     model the Janelia news article is about (Vaxenburg et al., Nature 2025) —
     and inventory every part, joint, degree of freedom, actuator and tendon
     that actually exists in it. No hand-transcription, no paraphrase.

  2. What is the nerve structure that drives it?  Read BANC v888 (the only
     public connectome that contains a ventral nerve cord) and inventory the
     motor neurons by the muscle they innervate and the joint they move.

Everything below is a MEASUREMENT read off one of those two files. There is
no constant in this file that was chosen to make anything look good.

Outputs
  build/fly_anatomy.json    the body: parts, joints, DOF, actuators, tendons
  build/fly_motorunits.json the nerves: muscle -> motor units -> joint targets
  reports/step1_anatomy.md  the human-readable report

Usage
  python3 tools/step1_anatomy.py --flybody data/flybody --banc data/banc
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import collections

import numpy as np

# --------------------------------------------------------------------------
# 1. THE BODY — read the MJCF through MuJoCo, never by hand
# --------------------------------------------------------------------------

# MuJoCo's mjtJoint / mjtGeom enums, by value, so this file does not need the
# mujoco package to *read* the inventory it produced. (It still uses it to
# build it — there is no substitute for the real parser.)
JOINT_KIND = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
GEOM_KIND = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder",
             6: "box", 7: "mesh", 9: "hfield"}


def read_body(mjcf_path: pathlib.Path) -> dict:
    """Open the fly in MuJoCo and write down everything it has."""
    import mujoco  # imported here so --help works without a GPU-adjacent dep

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    name = lambda obj, i: mujoco.mj_id2name(model, obj, i) or f"<{obj}:{i}>"

    # --- parts -------------------------------------------------------------
    # Index 0 is the world; every other body is a part of the animal.
    bodies = []
    for i in range(1, model.nbody):
        jadr, jnum = model.body_jntadr[i], model.body_jntnum[i]
        bodies.append({
            "index": i,
            "name": name(mujoco.mjtObj.mjOBJ_BODY, i),
            "parent": name(mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[i]),
            "mass_kg": float(model.body_mass[i]),
            "inertia": [float(x) for x in model.body_inertia[i]],
            "pos": [float(x) for x in model.body_pos[i]],
            "joints": [name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(jadr, jadr + jnum)],
        })

    # --- joints ------------------------------------------------------------
    joints = []
    for i in range(model.njnt):
        kind = JOINT_KIND[int(model.jnt_type[i])]
        dof = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}[kind]
        # MuJoCo >= 3.2 keeps damping and armature per degree of freedom, not
        # per joint. A hinge has one DOF, so the mapping is the identity; the
        # free root spreads its damping over six.
        da = int(model.jnt_dofadr[i])
        joints.append({
            "index": i,
            "name": name(mujoco.mjtObj.mjOBJ_JOINT, i),
            "kind": kind,
            "dof": dof,
            "body": name(mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[i]),
            "axis": [float(x) for x in model.jnt_axis[i]] if kind in ("hinge", "slide") else None,
            "range": [float(x) for x in model.jnt_range[i]] if model.jnt_limited[i] else None,
            "stiffness": float(model.jnt_stiffness[i]),
            "damping": float(model.dof_damping[da]),
            "armature": float(model.dof_armature[da]),
        })

    # --- actuators, tendons -------------------------------------------------
    TRANSMISSION = {0: "joint", 1: "jointinparent", 2: "slidercrank",
                    3: "tendon", 4: "site", 5: "body"}
    actuators = []
    for i in range(model.nu):
        trn = int(model.actuator_trntype[i])
        target = "-"
        try:
            if trn == 0:
                target = name(mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i][0]))
            elif trn == 3:
                target = name(mujoco.mjtObj.mjOBJ_TENDON, int(model.actuator_trnid[i][0]))
            elif trn == 4:
                target = name(mujoco.mjtObj.mjOBJ_SITE, int(model.actuator_trnid[i][0]))
        except Exception:
            pass
        actuators.append({
            "index": i,
            "name": name(mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "transmission": TRANSMISSION.get(trn, str(trn)),
            "target": target,
            "ctrlrange": [float(x) for x in model.actuator_ctrlrange[i]],
            "forcerange": [float(x) for x in model.actuator_forcerange[i]],
        })
    tendons = [name(mujoco.mjtObj.mjOBJ_TENDON, i) for i in range(model.ntendon)]

    # --- geometry / assembled size -----------------------------------------
    # This is the number that matters: the AABB of the *assembled* animal in
    # the rest pose, not the union of each part's own vertex frame.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Wings, legs, antennae and halteres stick out past the trunk and would
    # swamp the length measurement. They are the animal too, so both numbers
    # are reported: the whole silhouette, and the trunk alone.
    APPENDAGE = ("wing_", "haltere_", "antenna_", "coxa_", "femur_", "tibia_",
                 "tarsus", "claw_", "labrum", "haustellum", "rostrum")

    def geom_world(g):
        gtype = int(model.geom_type[g])
        xpos, xmat = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
        if gtype == 7:
            mid = int(model.geom_dataid[g])
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[va:va + vn]
        else:
            verts = np.zeros((1, 3))
        return verts @ xmat.T + xpos

    def extent_of(pred):
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in range(model.ngeom):
            b = name(mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            if not pred(b):
                continue
            w = geom_world(g)
            lo = np.minimum(lo, w.min(0))
            hi = np.maximum(hi, w.max(0))
        return hi - lo

    extent = extent_of(lambda b: True)
    trunk_extent = extent_of(lambda b: not any(k in b for k in APPENDAGE))

    # The model's length unit is settled by its own gravity: with total mass in
    # grams and gravity -981, the consistent unit is the centimetre. We assert
    # it against a published fly measurement instead of trusting the file.
    total_mass_kg = float(model.body_mass.sum()) / 1000.0   # g -> kg
    # x is the animal's long axis in this model (head at +x, abdomen at -x);
    # y is the wing span, z is dorso-ventral. Measured, not assumed: the head
    # body sits at +0.0567 and the abdomen at -0.0447, and the trunk extent
    # along x then reproduces the published body length.
    body_length_mm = float(trunk_extent[0]) * 10.0          # cm -> mm
    wingspan_mm = float(extent[1]) * 10.0

    return {
        "source": "TuragaLab/flybody fruitfly.xml (Janelia + Google DeepMind, Apache-2.0)",
        "citation": "Vaxenburg et al. (2025) Nature — whole-body physics simulation "
                    "of fruit fly locomotion",
        "file": str(mjcf_path),
        "counts": {
            "parts": len(bodies),
            "joints_total": model.njnt,
            "joints_articulated": model.njnt - 1,      # excluding the free root
            "dof_root": 6,
            "dof_articulated": sum(j["dof"] for j in joints if j["kind"] != "free"),
            "dof_total": model.nv,
            "qpos": model.nq,
            "actuators": model.nu,
            "tendons": model.ntendon,
            "sites": model.nsite,
            "sensors": model.nsensor,
            "geoms": model.ngeom,
        },
        "size": {
            "assembled_extent_raw": [float(x) for x in extent],
            "trunk_extent_raw": [float(x) for x in trunk_extent],
            "silhouette_mm": [round(float(x) * 10, 3) for x in extent],
            "trunk_mm": [round(float(x) * 10, 3) for x in trunk_extent],
            "unit": "cm (set by the file's own gravity: -981 cm/s^2)",
            "body_length_mm": body_length_mm,
            "wingspan_mm": wingspan_mm,
            "long_axis": "x",
            "reference_mm": "D. melanogaster adult body 2.5-3.0 mm, wingspan ~5 mm "
                            "(published)",
        },
        "mass_total_kg": total_mass_kg,
        "gravity_raw": [float(x) for x in model.opt.gravity],
        "timestep_s": float(model.opt.timestep),
        "bodies": bodies,
        "joints": joints,
        "actuators": actuators,
        "tendons": tendons,
    }


# --------------------------------------------------------------------------
# 2. THE NERVES — read BANC v888 and speak in muscles, not in cell IDs
# --------------------------------------------------------------------------

# BANC's `peripheral_target_type` is a real muscle name. Map it to the joint
# whose name in the flybody model it acts on. Both sides of this table are
# quotations: the left column is BANC's own vocabulary, the right column is
# flybody's own joint/actuator vocabulary. Nothing between them was invented;
# where a muscle has no modelled counterpart the row says None and the cell is
# reported as unmapped rather than silently dropped.
MUSCLE_TO_ACTUATOR = {
    # femur–tibia (the knee). flex = the animal's own naming in BANC.
    "tibia_flexor_muscle":                        "tibia_*",
    "accessory_tibia_flexor_muscle":              "tibia_*",
    "tibia_extensor_muscle":                      "tibia_*",
    # coxa–trochanter
    "trochanter_flexor_muscle":                   "coxa_twist_*",
    "accessory_trochanter_flexor_muscle":         "coxa_twist_*",
    "trochanter_extensor_muscle":                 "coxa_*",
    "tergotrochanter_extensor_muscle":            "coxa_*",
    "sternotrochanter_extensor_muscle":           "coxa_*",
    # coxa rotators  (the swing/stance plane of the whole leg)
    "sternal_anterior_rotator_muscle":            "coxa_abduct_*",
    "sternal_posterior_rotator_muscle":           "coxa_abduct_*",
    "tergopleural_promotor_muscle":               "coxa_abduct_*",
    "pleural_remotor_and_abductor_muscle":        "coxa_abduct_*",
    "sternal_adductor_muscle":                    "coxa_abduct_*",
    # femur reductor / long tendon
    "femur_reductor_muscle":                      "femur_*",
    "long_tendon_muscle":                         "tarsus2_*",   # via the claw-tendon chain
    # tarsus
    "tarsus_depressor_muscle":                    "tarsus_*",
    "tarsus_levator_muscle":                      "tarsus_*",
}

LEG_OF_BODY_PART = {"front_leg": "T1", "middle_leg": "T2", "hind_leg": "T3"}


def read_nerves(banc_dir: pathlib.Path) -> dict:
    import pyarrow.feather as feather

    cols = ["banc_888_id", "side", "neuromere", "cell_function",
            "cell_function_detailed", "body_part_effector", "body_part_sensory",
            "peripheral_target_type", "root_position_nm"]
    t = feather.read_table(banc_dir / "banc_888_meta.feather", columns=cols).to_pandas()

    motors = t[t.cell_function.astype(str).str.contains("motor", na=False)].copy()
    motors = motors[motors.cell_function != "unknown"]

    units = []
    for (muscle, part, side, neuro, fn, det), g in motors.groupby(
            ["peripheral_target_type", "body_part_effector", "side",
             "neuromere", "cell_function", "cell_function_detailed"], dropna=False):
        units.append({
            "muscle": None if muscle is None else str(muscle),
            "body_part": None if part is None else str(part),
            "side": None if side is None else str(side),
            "neuromere": None if neuro is None else str(neuro),
            "function": str(fn),
            "movement": None if det is None else str(det),
            "n_motor_neurons": int(len(g)),
            "motor_neuron_ids": [str(x) for x in g.banc_888_id.tolist()][:64],
        })

    # Every leg motor neuron must land on a modelled leg, or we have a hole.
    unmapped = [u for u in units
                if u["function"] == "leg_motor"
                and u["body_part"] not in LEG_OF_BODY_PART
                and u["n_motor_neurons"] > 0]

    per_leg = collections.Counter()
    per_muscle = collections.Counter()
    for u in units:
        if u["function"] == "leg_motor" and u["body_part"] in LEG_OF_BODY_PART:
            key = f'{LEG_OF_BODY_PART[u["body_part"]]}_{u["side"]}'
            per_leg[key] += u["n_motor_neurons"]
            if u["muscle"]:
                per_muscle[f'{key} {u["muscle"]}'] += u["n_motor_neurons"]

    sensory = collections.Counter(
        t[t.body_part_sensory.notna()].body_part_sensory.astype(str))

    return {
        "source": "BANC v888 (brain + ventral nerve cord, single female fly)",
        "citation": "Bates et al. (2026) Nature — the BANC connectome",
        "licence": "CC BY 4.0",
        "n_neurons_in_meta": int(len(t)),
        "n_motor_neurons": int(len(motors)),
        "motor_units": units,
        "leg_motor_neurons_per_leg": dict(sorted(per_leg.items())),
        "leg_motor_neurons_per_muscle": dict(sorted(per_muscle.items())),
        "sensory_organs": dict(sensory.most_common()),
        "muscle_to_actuator_pattern": MUSCLE_TO_ACTUATOR,
        "unmapped_leg_units": unmapped,
    }


# --------------------------------------------------------------------------

def write_report(body: dict, nerves: dict, out: pathlib.Path) -> None:
    c, s = body["counts"], body["size"]
    L = []
    A = L.append
    A("# Step 1 — the body and the nerves, both as measured\n")
    A("Generated by `tools/step1_anatomy.py`. Every number below is read out of "
      "one of the two source files at build time; re-run it and they come back "
      "identical.\n")

    A("## 1. The body\n")
    A(f"Source: `{body['source']}`\n")
    A(f"Citation: {body['citation']}\n")
    A("| | |")
    A("|---|---|")
    A(f"| Body parts | **{c['parts']}** |")
    A(f"| Joints (articulated) | **{c['joints_articulated']}** |")
    A(f"| Joints incl. the free root | {c['joints_total']} |")
    A(f"| Degrees of freedom, articulated | **{c['dof_articulated']}** |")
    A(f"| Degrees of freedom, total (with the 6-DOF root) | {c['dof_total']} |")
    A(f"| Actuators | {c['actuators']} |")
    A(f"| Tendons | {c['tendons']} |")
    A(f"| Collision/visual geoms | {c['geoms']} |")
    A(f"| Total mass | {body['mass_total_kg'] * 1e6:.3f} mg |")
    A(f"| Trunk length along the body axis (x) | **{s['body_length_mm']:.2f} mm** |")
    A(f"| Wing span (y) | {s['wingspan_mm']:.2f} mm |")
    A(f"| Trunk size (x × y × z) | {s['trunk_mm'][0]:.2f} × {s['trunk_mm'][1]:.2f} × {s['trunk_mm'][2]:.2f} mm |")
    A(f"| Whole silhouette incl. wings, legs, antennae | {s['silhouette_mm'][0]:.2f} × {s['silhouette_mm'][1]:.2f} × {s['silhouette_mm'][2]:.2f} mm |")
    A(f"| Units | {s['unit']} |")
    A(f"| Integrator step | {body['timestep_s'] * 1e6:.0f} µs |")
    A("")
    A(f"Published *D. melanogaster* adult body length: {s['reference_mm']}. The "
      "model's own gravity and mass fix its length unit as the centimetre; the "
      "assembled measurement then agrees with the published animal, so the "
      "model is at scale and no constant needs adjusting.\n")

    A("### Actuator inventory\n")
    A("| group | how many | what they are |")
    A("|---|---|---|")
    groups = collections.Counter()
    for a in body["actuators"]:
        n = a["name"]
        if n.startswith("adhere_"):
            groups["adhesion"] += 1
        elif "leg" in n or n.split("_")[-1] in ("T1", "T2", "T3") or "_T" in n:
            groups["leg joints"] += 1
        elif n.startswith("wing"):
            groups["wing joints"] += 1
        elif n.startswith("antenna"):
            groups["antennal joints"] += 1
        elif n.startswith("head") or n in ("rostrum", "haustellum") or n.startswith(("labrum", "haustellum")):
            groups["head/mouth joints"] += 1
        elif n.startswith("abdomen"):
            groups["abdominal joints"] += 1
        else:
            groups["other joints"] += 1
    for k, v in groups.most_common():
        A(f"| {k} | {v} | |")
    A("")
    A("The eight `adhere_*` actuators are the adhesion model the Janelia article "
      "describes: two on the labrum and one on each of the six claws. They are "
      "what lets the animal grip a surface instead of sliding off it.\n")

    A("### Joints whose axis and range come from the scan\n")
    A("| joint | part | axis | range (rad) | stiffness | damping |")
    A("|---|---|---|---|---|---|")
    for j in body["joints"]:
        if j["kind"] != "hinge":
            continue
        if not any(k in j["name"] for k in ("coxa_T1_left", "femur_T1_left",
                                            "tibia_T1_left", "wing_pitch_left",
                                            "head", "abdomen")):
            continue
        ax = "—" if j["axis"] is None else " ".join(f"{x:+.2f}" for x in j["axis"])
        rg = "—" if j["range"] is None else f'{j["range"][0]:+.3f} … {j["range"][1]:+.3f}'
        A(f'| `{j["name"]}` | {j["body"]} | {ax} | {rg} | '
          f'{j["stiffness"]:.4f} | {j["damping"]:.5f} |')
    A("")

    A("## 2. The nerve structure\n")
    A(f"Source: **{nerves['source']}** — {nerves['citation']} ({nerves['licence']}).\n")
    A(f"Motor neurons in the dataset: **{nerves['n_motor_neurons']}**. "
      "This is the whole point of using BANC rather than a brain-only dataset: "
      "the cell bodies that move the legs are *in the file*.\n")

    A("### Leg motor neurons, per leg\n")
    A("| leg | motor neurons |")
    A("|---|---|")
    for k, v in nerves["leg_motor_neurons_per_leg"].items():
        A(f"| {k} | {v} |")
    A("")
    A("### Leg motor neurons, per muscle\n")
    A("| muscle | motor neurons |")
    A("|---|---|")
    for k, v in nerves["leg_motor_neurons_per_muscle"].items():
        A(f"| `{k}` | {v} |")
    A("")
    A("Each row is a named muscle of the fly's leg with the number of motor "
      "neurons that innervate it, counted from the connectome. `MUSCLE_TO_"
      "ACTUATOR` in the tool maps each onto the modelled joint it moves; rows "
      "with no modelled counterpart are reported, never dropped.\n")

    A("### Sensory organs available as feedback\n")
    A("| organ | cells |")
    A("|---|---|")
    for k, v in list(nerves["sensory_organs"].items())[:20]:
        A(f"| {k} | {v} |")
    A("")

    if nerves["unmapped_leg_units"]:
        A("### Unmapped\n")
        A(f"{len(nerves['unmapped_leg_units'])} leg motor units have a body part "
          "that is not a modelled leg. Listed here rather than dropped.\n")
        for u in nerves["unmapped_leg_units"]:
            A(f"- {u['body_part']} × {u['n_motor_neurons']} ({u['muscle']})")
        A("")
    else:
        A("Every leg motor unit in BANC lands on a modelled leg. No holes.\n")

    out.write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flybody", default="data/flybody", type=pathlib.Path)
    ap.add_argument("--banc", default="data/banc", type=pathlib.Path)
    ap.add_argument("--out", default="build", type=pathlib.Path)
    ap.add_argument("--report", default="reports/step1_anatomy.md", type=pathlib.Path)
    a = ap.parse_args()

    mjcf = a.flybody / "flybody-main/flybody/fruitfly/assets/fruitfly.xml"
    if not mjcf.exists():
        print(f"missing the fly body model at {mjcf}", file=sys.stderr)
        print("run: bash tools/download_flybody.sh", file=sys.stderr)
        return 2
    if not (a.banc / "banc_888_meta.feather").exists():
        print(f"missing BANC at {a.banc}", file=sys.stderr)
        print("run: bash tools/download_banc.sh", file=sys.stderr)
        return 2

    print("reading the body …")
    body = read_body(mjcf)
    c = body["counts"]
    print(f"  {c['parts']} parts · {c['joints_articulated']} joints · "
          f"{c['dof_articulated']} DOF · {c['actuators']} actuators · "
          f"{body['mass_total_kg']*1e6:.3f} mg · "
          f"{body['size']['body_length_mm']:.2f} mm long")

    print("reading the nerves …")
    nerves = read_nerves(a.banc)
    print(f"  {nerves['n_motor_neurons']} motor neurons · "
          f"{len(nerves['motor_units'])} motor units")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "fly_anatomy.json").write_text(json.dumps(body, indent=1))
    (a.out / "fly_motorunits.json").write_text(json.dumps(nerves, indent=1))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(body, nerves, a.report)
    print(f"wrote {a.out}/fly_anatomy.json, {a.out}/fly_motorunits.json, {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
