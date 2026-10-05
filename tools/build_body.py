#!/usr/bin/env python3
"""
BUILD THE BODY — the fly as something that can be *simulated*, not just drawn.

`tools/step4_world.py` records what the animal did. That recording is a
measurement, and it stays, but it is a film. This tool exists because the app
should not be showing a film: the animal in your hand should be the animal
running, one millisecond of nerve cord and one hundred microseconds of physics
at a time, on the phone.

So this writes the one file the phone needs to *be* the flybody model rather
than play it back:

  build/fly_body.json    the rigid-body tree, the joints, the collision geoms,
                         the muscles, and the leg/organ/pool tables
  build/fly_meshes.bin   the welded visual meshes (same format as world/fly.bin)

Every number in it is read out of two sources at build time and nothing is
typed in by hand:

  * **TupagaLab/Google DeepMind `flybody`** (Apache-2.0) — the tree, the joint
    axes and ranges, the masses and inertias, the geoms, the actuator table.
  * **BANC v888** (CC BY 4.0), through step 2's own vocabulary — which muscle
    pool moves which joint, and in which direction. The direction is *measured*
    on the assembled animal by `step3_closedloop.Body.measure_directions`,
    imported rather than re-implemented, so there is one source of truth.

The one thing that is not read out of a file is the **muscle torque scale**:
the model gives a position servo, not a muscle, so it does not state how much
torque a joint can make. That is measured here as the signed torque the
animal's own actuators need to hold its stance — inverse dynamics at the
settled pose, so it includes the floor and the weight above each joint — and
it is reported per joint in the JSON under `muscles[].hold_torque`; the scale
`max_torque` is then a `--headroom` multiple of its magnitude. That is an ENGINEERING PLACEHOLDER with a test —
`tools/fly_aba.py --verify` fails if the resulting muscles cannot hold the
animal up — and it is the only invented quantity in this file.

Usage
  python3 tools/build_body.py                     # -> build/fly_body.json
  python3 tools/build_body.py --headroom 4        # the declared multiplier
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import step4_world as s4                                    # noqa: E402
import step3_closedloop as s3                               # noqa: E402
import step2_reflex as s2                                   # noqa: E402

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


LEG_PARTS = ("front_leg", "middle_leg", "hind_leg")          # BANC's names
LEG_SEGMENT = {"front_leg": "T1", "middle_leg": "T2", "hind_leg": "T3"}

# The joints step 3 and step 4 hand to the cord. Exactly three per leg, which
# is what the connectome's pools can actually be placed on
# (docs/ASSUMPTIONS.md #9): the coxa, the femur, and the knee.
DRIVEN_JOINTS = ("coxa_abduct", "femur", "tibia")

# Which joint each pool moves, and whether it moves it up or down. This is
# step 3's table, imported, not copied: build_mapper() is the authority on
# pool -> joint and it is the same code path the off-device experiments use.
POOL_ACTION = s3.POOL_ACTION
POOLS = s2.POOLS

# The two sense organs the body can tell the cord about, in step 2's
# vocabulary. Chordotonal = joint position, campaniform = load.
ORGAN_FUNCTIONS = s2.ORGANS


# The names the app asks the connectome for. They come from the table that
# *writes* the connectome (tools/motor_pools.py), not from a second spelling
# here: a pool called `motor_front_leg_left_tibia_flexor` in this asset and
# `pool:T1_left:tibia_flexor` in flybanc.bin is a loop that reads silence, and
# silence looks exactly like an animal doing nothing.
import motor_pools as _mp                                          # noqa: E402


def motor_pool_group(part: str, side: str, pool: str) -> str:
    return _mp.pool_slug(_mp.LEGS[part], side, pool)


def organ_group(part: str, side: str, organ: str) -> str:
    return _mp.organ_slug(_mp.LEGS[part], side, organ)


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------

def read_tree(model, name) -> dict:
    """
    The rigid-body tree, exactly as MuJoCo compiled it — expanded so that every
    link carries exactly one joint.

    MuJoCo lets a body carry several joints (the head has three: abduct, twist,
    pitch), which compose as a chain of rotations inside that one body. An
    articulated-body solver needs one joint per link, so the 25 multi-joint
    bodies are expanded here into chains: one link per joint, the body's own
    mass, inertia and geometry on the last link of its chain. Nothing is
    approximated — the expansion is the same mechanism, written out.
    """
    m = model

    links = []
    link_of_body = {}
    for i in range(m.nbody):
        jadr, jnum = int(m.body_jntadr[i]), int(m.body_jntnum[i])
        jids = list(range(jadr, jadr + jnum))
        parent_link = 0 if i == 0 else link_of_body[int(m.body_parentid[i])]
        base = name("body", i)

        if i == 0:
            links.append({
                "index": 0, "name": base, "body": base, "frame": True,
                "parent_index": 0, "parent": base, "mass": 0.0,
                "com": [0.0, 0.0, 0.0], "inertia": [0.0, 0.0, 0.0],
                "iquat": [1.0, 0.0, 0.0, 0.0], "pos": [0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0], "joint": None,
                "geoms": [name("geom", g) for g in range(m.ngeom)
                          if int(m.geom_bodyid[g]) == 0],
            })
            link_of_body[0] = 0
            continue

        if not jids:
            # a body with no joint of its own: welded to its parent
            links.append({
                "index": len(links), "name": base, "body": base, "frame": True,
                "parent_index": parent_link, "parent": links[parent_link]["name"],
                "mass": float(m.body_mass[i]),
                "com": [float(x) for x in m.body_ipos[i]],
                "inertia": [float(x) for x in m.body_inertia[i]],
                "iquat": [float(x) for x in m.body_iquat[i]],
                "pos": [float(x) for x in m.body_pos[i]],
                "quat": [float(x) for x in m.body_quat[i]],
                "joint": None,
                "geoms": [name("geom", g) for g in range(m.ngeom)
                          if int(m.geom_bodyid[g]) == i],
            })
            link_of_body[i] = len(links) - 1
            continue

        for k, jid in enumerate(jids):
            last = (k == len(jids) - 1)
            jn = name("joint", jid)
            links.append({
                "index": len(links),
                "name": base if last else f"{base}__{jn}",
                "body": base,
                "frame": last,
                "parent_index": parent_link,
                "parent": links[parent_link]["name"],
                "mass": float(m.body_mass[i]) if last else 0.0,
                "com": [float(x) for x in m.body_ipos[i]] if last else [0.0, 0.0, 0.0],
                "inertia": [float(x) for x in m.body_inertia[i]] if last else [0.0, 0.0, 0.0],
                "iquat": [float(x) for x in m.body_iquat[i]] if last else [1.0, 0.0, 0.0, 0.0],
                "pos": [float(x) for x in m.body_pos[i]] if k == 0 else [0.0, 0.0, 0.0],
                "quat": [float(x) for x in m.body_quat[i]] if k == 0 else [1.0, 0.0, 0.0, 0.0],
                "joint": jn,
                "geoms": ([name("geom", g) for g in range(m.ngeom)
                           if int(m.geom_bodyid[g]) == i] if last else []),
            })
            parent_link = len(links) - 1
        link_of_body[i] = parent_link

    joints = []
    for j in range(m.njnt):
        kind = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[int(m.jnt_type[j])]
        da = int(m.jnt_dofadr[j])
        jn = name("joint", j)
        link = next(x for x in links if x["joint"] == jn)
        joints.append({
            "index": j,
            "name": jn,
            "kind": kind,
            "body": link["name"],
            "source_body": name("body", int(m.jnt_bodyid[j])),
            "axis": [float(x) for x in m.jnt_axis[j]],
            "anchor": [float(x) for x in m.jnt_pos[j]],
            "range": [float(x) for x in m.jnt_range[j]],
            "limited": bool(m.jnt_limited[j]),
            "stiffness": float(m.jnt_stiffness[j]),
            # The angle the joint's own spring pulls towards. Nonzero all over
            # the fly (rostrum 0.8, wing_yaw 1.5, wing_pitch -1.0 rad): a
            # solver that springs towards zero instead of towards the animal's
            # rest pose is off by `stiffness * spring_ref` on every one of
            # them, which is small torque and enormous acceleration on the
            # wing and mouth parts.
            "spring_ref": float(m.qpos_spring[int(m.jnt_qposadr[j])]),
            "damping": float(m.dof_damping[da]),
            "armature": float(m.dof_armature[da]),
            "qposadr": int(m.jnt_qposadr[j]),
            "dofadr": da,
        })

    # --- collision geoms ---------------------------------------------------
    # Everything that is not a mesh, on a body of the animal. The mesh geoms
    # are the ones you look at; these are the ones the floor feels.
    KIND = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
            5: "cylinder", 6: "box", 7: "mesh"}
    collision = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        t = int(m.geom_type[g])
        if b == 0 or t == 7:
            continue
        collision.append({
            "name": name("geom", g),
            "body": name("body", b),
            "type": KIND[t],
            "size": [float(x) for x in m.geom_size[g]],
            "pos": [float(x) for x in m.geom_pos[g]],
            "quat": [float(x) for x in m.geom_quat[g]],
        })

    return {"bodies": links, "joints": joints, "collision": collision,
            "link_of_body": {name("body", i): link_of_body[i]
                             for i in range(m.nbody)}}


def read_actuators(model, name) -> list:
    """
    The model's own actuators, with their force law.

    flybody's actuators are affine: `force = gainprm[0]*ctrl + biasprm[1]*q`,
    which is a muscle with an active gain and a *passive stiffness* — the
    model's own statement about how stiff a fly's muscles are, and the largest
    published number in the whole file. Reading only the ctrlrange, as this
    used to, throws away both.
    """
    out = []
    for i in range(model.nu):
        t = int(model.actuator_trntype[i])
        row = {
            "index": i,
            "name": name("actuator", i),
            "transmission": {0: "joint", 1: "jointinparent", 3: "tendon",
                             4: "site", 5: "body"}.get(t, str(t)),
            "ctrlrange": [float(x) for x in model.actuator_ctrlrange[i]],
            "gainprm": [float(x) for x in model.actuator_gainprm[i]],
            "biasprm": [float(x) for x in model.actuator_biasprm[i]],
            "gear": [float(x) for x in model.actuator_gear[i]],
        }
        if t in (0, 1):
            row["joint"] = name("joint", int(model.actuator_trnid[i][0]))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# The muscle torque scale — the one measured, one declared quantity
# ---------------------------------------------------------------------------

def hold_torques(args) -> dict:
    """
    What the animal's own actuators have to do to stand, per joint, signed.

    The model's actuators are position servos, so the file never says how much
    torque a joint can make. What it *can* be asked is what torque holds the
    animal where it stands: settle the animal on the floor from the model's own
    keyframe, then ask MuJoCo's *inverse* dynamics for the generalized force
    that produces zero acceleration at that pose. That force is signed and it
    carries the whole load — the weight of the animal above the joint and the
    floor pushing back through the feet — which is exactly what a muscle has to
    supply.

    The obvious reading of the same idea is wrong and was worth the rebuild:
    `qfrc_actuator` at the stance is the servo's own passive spring, `-0.1*q`,
    which at a stance near q = 0 is a fraction of a hundredth of the animal's
    weight and always points the same way. Twice over it says nothing about
    load: it is a spring, and it has no sign. Muscles that are always extensors
    splat the animal (measured: |qd| reaching 1.4e4 rad/s in 20 ms, then NaN).

    Returns {joint_name: signed torque}, the settled root height, and the
    contact count, so the caller can check the animal it measured is the animal
    that stood.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(args.floor_xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = 0.0                     # every servo at its anatomical rest
    steps = int(round(args.settle_s / m.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(m, d)

    # The stance is the state to hold, so ask for the force that holds it.
    # `mj_inverse` needs the constraint forces of a forward pass at the same
    # state, which `mj_forward` provides, and it deliberately ignores actuator
    # force: what comes back is what the actuators would have to supply.
    mujoco.mj_forward(m, d)
    d.qacc[:] = 0.0
    d.qfrc_applied[:] = 0.0
    mujoco.mj_inverse(m, d)

    name = lambda obj, i: mujoco.mj_id2name(m, obj, i) or f"<{obj}:{i}>"  # noqa: E731
    out = {}
    for i in range(m.nu):
        if int(m.actuator_trntype[i]) != 0:
            continue
        j = int(m.actuator_trnid[i][0])
        out[name(mujoco.mjtObj.mjOBJ_JOINT, j)] = float(
            d.qfrc_inverse[int(m.jnt_dofadr[j])])
    # the pose those torques hold, hinge by hinge: an animal standing on the
    # floor has its legs compressed by its own weight, and a solver that starts
    # from the keyframe instead starts from a pose the torques do not hold
    stance_q = {}
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        stance_q[name(mujoco.mjtObj.mjOBJ_JOINT, j)] = float(
            d.qpos[int(m.jnt_qposadr[j])])
    return out, float(d.qpos[2]), int(d.ncon), stance_q


def muscles(args, tree, leg_table, hold, headroom, actuators) -> dict:
    """
    One muscle per joint, with its maximum torque and its passive stiffness.

    A joint in the fly is pulled by antagonist muscles, so what the phone needs
    per joint is (a) how much torque the pair can make together, (b) how stiff
    the pair is when it is doing nothing, and (c) which of the two directions a
    given motor pool pulls. (c) is measured on the animal by step 3 and lives in
    `leg_table`. (a) and (b) are *both in the model*: each actuator is written
    as `gainprm[0]*ctrl + biasprm[1]*q`, so the full-activation torque is
    `gainprm[0]*max|ctrl|` and `-biasprm[1]` is the passive stiffness of the
    muscle at rest. The leg joints come out at 0.4-0.8, which is what keeps a
    1e-6 g cm^2 joint from being thrown around by its own feet.

    Joints the file gives no actuator (36 of 102) fall back to `headroom` times
    the torque the animal's own settle needed, with no passive stiffness: a
    PLACEHOLDER, marked `"source": "placeholder"` in the JSON so nothing
    downstream can mistake it for data.
    """
    hinge = [j for j in tree["joints"] if j["kind"] == "hinge"]
    # magnitudes scale the muscle; the signed torque is the stance feedforward
    loaded = {j["name"]: abs(hold.get(j["name"], 0.0)) for j in hinge}
    reference = max(loaded.values()) if loaded else 1.0
    floor = args.torque_floor * reference      # a joint that carries nothing
                                               # still needs a muscle
    by_joint = {a["joint"]: a for a in actuators if "joint" in a}
    out = {}
    for j in hinge:
        name = j["name"]
        hold_t = loaded[name]
        act = by_joint.get(name)
        if act is not None and abs(act["gainprm"][0]) > 0:
            ctrl = max(abs(x) for x in act["ctrlrange"]) or 1.0
            max_t = abs(act["gainprm"][0]) * ctrl
            stiffness = max(0.0, -act["biasprm"][1])
            source = "flybody actuator (gainprm, biasprm)"
        else:
            max_t = headroom * max(hold_t, floor)
            stiffness = 0.0
            source = "placeholder: headroom x the stance torque"
        out[name] = {
            "joint": name,
            # signed: the torque that holds the stance (inverse dynamics)
            "hold_torque": float(hold.get(name, 0.0)),
            "max_torque": max_t,
            "max_torque_source": source,
            # passive muscle stiffness, `-biasprm[1]`: pulls the joint towards
            # the file's own control zero, which is the stance
            "passive_stiffness": stiffness,
            # The length-tension curve is centred on the joint's own spring
            # reference — the angle MuJoCo's compiler was given as this joint's
            # rest pose in flybody — and is as wide as the joint's anatomical
            # range. flybody publishes no muscle optimal angles, so this is an
            # APPROXIMATION and is declared as one (docs/ASSUMPTIONS.md #20);
            # the rest angle itself is read from the model, not invented.
            "optimal_angle": j.get("spring_ref", 0.0),
            "spring_ref": j.get("spring_ref", 0.0),
            "range": j["range"] if j["limited"] else [-3.14159, 3.14159],
            "damping": j["damping"],
            "armature": j["armature"],
            "stiffness": j["stiffness"],
        }
    return out


# ---------------------------------------------------------------------------
# The legs: joints, feet, organs, pools
# ---------------------------------------------------------------------------

def leg_tables(args, tree, name_of) -> dict:
    """
    Six legs, each with the three joints the cord can address, the muscle pools
    that pull them, the sense organs, and the bits of the leg that touch the
    floor.

    `step3_closedloop.Body` is constructed once per leg to measure directions —
    the same class the off-device experiments use, asked the same question, so
    the sign of every pool's action is the sign that tool reported.
    """
    out = {}
    for part in LEG_PARTS:
        seg = LEG_SEGMENT[part]
        for side in ("left", "right"):
            b = s3.Body(args.mjcf, part, side)
            joints = {}
            for key in DRIVEN_JOINTS:
                j = b.joint[key]
                joints[key] = {
                    "name": j["name"],
                    "range": [float(x) for x in j["range"]],
                    "rest": float(b.rest[key]),
                    "direction": {k: int(v) for k, v in b.direction[key].items()},
                    "actuator": j["actuator"],
                }
            # Which pool pulls which joint, and which way: step 3's own mapper,
            # fed the same pool list, through the same function.
            placed, driven, unplaced = s3.build_mapper(
                _PoolsShim(POOLS), _BodyShim(b))
            pools = {}
            for pool, info in placed.items():
                pools[pool] = {
                    "joint": info["joint"],
                    "action": info["action"],
                    # +1: a bigger pool rate opens the joint's qpos. Measured,
                    # not assumed (step3_closedloop.Body.measure_directions).
                    "qpos_sign": int(info["qpos_sign"]),
                    "group": motor_pool_group(part, side, pool),
                }
            feet = sorted(nm for nm in
                          (name_of("body", i)
                           for i in range(len(tree["bodies"])))
                          if nm.startswith(("claw_", "tarsus3_", "tarsus4_"))
                          and nm.endswith(f"_{seg}_{side}"))
            adhesion = f"adhere_claw_{seg}_{side}"
            out[f"{seg}_{side}"] = {
                "part": part,
                "segment": seg,
                "side": side,
                "joints": joints,
                "pools": pools,
                "pools_not_placed": unplaced,
                "feet_bodies": feet,
                "adhesion_actuator": adhesion,
                "organs": {
                    # The femoral chordotonal organ, which the literature puts
                    # on the femur-tibia joint (step 2 §2): joint position.
                    "chordotonal": {
                        "joint": joints["tibia"]["name"],
                        "group": organ_group(part, side, "chordotonal"),
                        "functions": sorted(ORGAN_FUNCTIONS["chordotonal"]),
                    },
                    # Campaniform sensilla measure strain, and in this body the
                    # strain a sensillum sees is the load its leg is carrying.
                    "campaniform": {
                        "leg_bodies": feet,
                        "group": organ_group(part, side, "campaniform"),
                        "functions": sorted(ORGAN_FUNCTIONS["campaniform"]),
                    },
                },
            }
            del b
    return out


class _PoolsShim:
    """One attribute, so step 3's mapper can be called on this leg."""

    def __init__(self, pools):
        self.watch = {f"pool:{p}": np.zeros(1) for p in pools}


class _BodyShim:
    def __init__(self, body):
        self.joint = body.joint
        self.direction = body.direction


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flybody", default="data/flybody", type=pathlib.Path)
    ap.add_argument("--out", default="build/fly_body.json", type=pathlib.Path)
    ap.add_argument("--meshes", default="build/fly_meshes.bin", type=pathlib.Path)
    ap.add_argument("--mesh-grid", type=float, default=1e-4,
                    help="weld tolerance for the visual meshes (model units)")
    ap.add_argument("--settle-s", type=float, default=1.5,
                    help="how long the animal is given to settle before its "
                         "stance torques are read")
    ap.add_argument("--headroom", type=float, default=4.0,
                    help="multiple of the measured stance torque a joint's "
                         "muscles are allowed (ASSUMPTIONS #20)")
    ap.add_argument("--torque-floor", type=float, default=0.1,
                    help="fraction of the largest measured torque that even an "
                         "unloaded joint gets, so no joint is unmuscled")
    args_ = ap.parse_args()

    import mujoco

    assets = args_.flybody / "flybody-main/flybody/fruitfly/assets"
    args_.mjcf = assets / "fruitfly.xml"
    args_.floor_xml = assets / "floor.xml"
    for path in (args_.mjcf, args_.floor_xml):
        if not path.exists():
            print(f"missing {path}\nrun: bash tools/download_flybody.sh",
                  file=sys.stderr)
            return 2

    log("opening the body model ...")
    model = mujoco.MjModel.from_xml_path(str(args_.mjcf))

    OBJ = {"body": mujoco.mjtObj.mjOBJ_BODY, "joint": mujoco.mjtObj.mjOBJ_JOINT,
           "geom": mujoco.mjtObj.mjOBJ_GEOM,
           "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
           "tendon": mujoco.mjtObj.mjOBJ_TENDON, "mesh": mujoco.mjtObj.mjOBJ_MESH}

    def name_of(kind: str, i: int) -> str:
        return mujoco.mj_id2name(model, OBJ[kind], i) or f"<{kind}:{i}>"

    tree = read_tree(model, name_of)
    tree["actuators"] = read_actuators(model, name_of)
    log(f"  {len(tree['bodies']) - 1} parts, "
        f"{sum(1 for j in tree['joints'] if j['kind'] == 'hinge')} hinge joints, "
        f"{len(tree['collision'])} collision geoms")

    log(f"settling the standing animal for {args_.settle_s}s to measure what "
        f"holding the stance costs ...")
    hold, root_z, contacts, stance_q = hold_torques(args_)
    log(f"  it stood: root z {root_z:+.5f} cm, {contacts} contacts touching, "
        f"largest joint torque {max(abs(v) for v in hold.values()):.6g}")

    log("measuring each leg's joints, pools and directions "
        "(step 3's own Body class) ...")
    legs = leg_tables(args_, tree, name_of)
    for key, leg in legs.items():
        log(f"  {key:9s} {len(leg['joints'])} joints, {len(leg['pools'])} pools, "
            f"{len(leg['feet_bodies'])} foot bodies")

    actuators = read_actuators(model, name_of)
    muscles_ = muscles(args_, tree, legs, hold, args_.headroom, actuators)

    # --- the visual meshes --------------------------------------------------
    # The same packing step 4 uses, called rather than copied, so the phone's
    # live animal and the recording are the same geometry.
    log("welding and packing the visual meshes ...")
    args_.meshes.parent.mkdir(parents=True, exist_ok=True)
    mesh_manifest = s4.write_body(model, args_.meshes, grid=args_.mesh_grid)

    keep = s4.visible_geoms(model)
    visual = s4.geom_manifest(model, keep)

    # --- where the floor is ------------------------------------------------
    # The floor the off-device model stands on, read out of the file that
    # provides it, plus where the animal's lowest point actually is at the
    # stance — the two differ by the contact softness, and both are reported.
    floor_z = floor_of(args_.floor_xml)
    stance_support_z = lowest_support(model, tree)

    body = {
        "format": "flybody-live v1",
        "source": {
            "body": "TuragaLab/flybody fruitfly.xml (Janelia + Google DeepMind, "
                    "Apache-2.0)",
            "citation": "Vaxenburg et al. (2025) Nature — whole-body physics "
                        "simulation of fruit fly locomotion",
            "nerves": "BANC v888 (Bates et al. 2026, CC BY 4.0), through the "
                      "pools and organs of tools/step2_reflex.py",
            "generated_by": "tools/build_body.py",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        # Model units, straight out of the MJCF: gravity is -981, so length is
        # the centimetre, mass the gram (0.985 mg of fly) and time the second.
        "units": {"length": "cm", "mass": "g", "time": "s", "angle": "rad",
                  "force": "g*cm/s^2", "torque": "g*cm^2/s^2",
                  "note": "1 unit of torque = 1e-9 N*m"},
        "gravity": [float(x) for x in model.opt.gravity],
        "timestep_s": float(model.opt.timestep),
        "floor_z": floor_z,
        "stance_support_z": stance_support_z,
        "stance_root_z": root_z,
        # the hinge angles the animal itself settled at, under its own
        # actuators, with the floor holding it up
        "stance_q": stance_q,
        "counts": {
            "bodies": len(tree["bodies"]) - 1,
            "hinges": sum(1 for j in tree["joints"] if j["kind"] == "hinge"),
            "dof": int(model.nv),
            "collision_geoms": len(tree["collision"]),
            "visual_geoms": len(visual),
            "actuators": len(tree["actuators"]),
            "muscles": len(muscles_),
        },
        "muscle_model": {
            "kind": "activation -> force, with length and velocity dependence",
            "max_torque_rule": f"{args_.headroom} x the measured stance torque, "
                               f"floored at {args_.torque_floor} x the largest",
            "status": "calibration is ENGINEERING PLACEHOLDER (ASSUMPTIONS #20); "
                      "the tree, joints, masses and geoms are measured",
        },
        "bodies": tree["bodies"],
        "joints": tree["joints"],
        "collision": tree["collision"],
        "actuators": tree["actuators"],
        "muscles": muscles_,
        "legs": legs,
        "visual": visual,
        "meshes": mesh_manifest,
        "geometry": {"n_vertex": mesh_manifest["n_vertex"],
                     "n_face": mesh_manifest["n_face"],
                     "file": args_.meshes.name,
                     "bytes": mesh_manifest["bytes"]},
    }

    args_.out.parent.mkdir(parents=True, exist_ok=True)
    args_.out.write_text(json.dumps(body, indent=1))
    log(f"wrote {args_.out} ({args_.out.stat().st_size/1e6:.1f} MB) and "
        f"{args_.meshes} ({args_.meshes.stat().st_size/1e6:.1f} MB)")
    log(f"  floor at z = {floor_z:+.6f} cm (the plane floor.xml provides); "
        f"at the stance the lowest collision point sits at "
        f"{stance_support_z:+.6f} cm, i.e. {(-stance_support_z)*10:.2f} mm "
        f"below the floor — the contact softness, not a floating animal")
    return 0


def lowest_support(model, tree) -> float:
    """
    The height of the lowest point of the animal, at the stance.

    A collision geom is a convex shape, so its lowest point is its centre minus
    its support function along +z. Taking the minimum over the geoms gives the
    floor the animal is standing on, measured rather than guessed.
    """
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)

    def support(geom, n):
        t = int(model.geom_type[geom])
        s = model.geom_size[geom]
        r = d.geom_xmat[geom].reshape(3, 3)
        if t == 2:                                     # sphere
            return float(s[0])
        if t == 3:                                     # capsule
            axis = r[:, 2]
            return float(s[0] + s[1] * abs(np.dot(axis, n)))
        if t == 4:                                     # ellipsoid
            return float(np.linalg.norm(r @ s[:3]))
        if t == 5:                                     # cylinder
            axis = r[:, 2]
            a = abs(np.dot(axis, n))
            return float(s[0] * np.sqrt(max(0.0, 1 - a * a)) + s[1] * a)
        if t == 6:                                     # box
            nn = r.T @ n
            return float(np.abs(nn) @ s[:3])
        return 0.0

    z_min = np.inf
    n = np.array([0.0, 0.0, 1.0])
    for g in tree["collision"]:
        gi = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g["name"])
        if gi < 0:
            continue
        z_min = min(z_min, float(d.geom_xpos[gi][2]) - support(gi, n))
    return float(z_min) if np.isfinite(z_min) else 0.0


def floor_of(floor_xml: pathlib.Path) -> float:
    """
    Where the floor is, out of the file that provides it.

    `fruitfly.xml` has no floor — it is a fly in empty space — and
    `floor.xml` includes it and adds a plane. Reading the plane's own height
    keeps the phone's floor in the same place as the off-device one instead of
    assuming zero.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(floor_xml))
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) == 0 and int(m.geom_type[g]) == 0:
            return float(m.geom_pos[g][2])
    return 0.0


if __name__ == "__main__":
    sys.exit(main())
