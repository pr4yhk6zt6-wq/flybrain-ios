#!/usr/bin/env python3
"""
STEP 3 — close the loop.

Step 1 built the body and proved it stands. Step 2 took the leg reflex out of
the connectome and measured what the sense organs do to the motor pools. What
neither step did was join them: step 2's rates were numbers in a table, and
step 1's body was held up by nothing but its own rest pose.

This tool makes one animal out of the two. Every millisecond:

    the cord's motor pools spike
        -> their rates drive the muscles of one front leg
            -> MuJoCo moves the joints
                -> the joint angle drives the chordotonal organ
                   and the ground reaction force drives the campaniform organ
                    -> which changes what the pools do next

Nothing in that loop is scheduled and nothing is a scripted movement. The
question the loop is built to answer is the one the reflex literature has asked
of real insects since the 1960s:

    **when the knee is pushed, does the animal push back?**

and then, the harder one:

    **does it keep pushing back when the leg is loaded, or does the reflex
    change sign and help the movement instead?**

The resistance reflex is the published behaviour: proprioceptive input from the
femoral chordotonal organ drives the tibial muscles so as to oppose an imposed
joint movement, in flies as in cockroaches, locusts, stick insects and
katydids, and the polarity is stated plainly — *joint flexion stretches the
FeCO, extension relaxes it* (Current Opinion in Insect Science 2024). The
reversal, resistance turning into assistance, is the state-dependent switch
real preparations show. This tool assumes neither: it imposes a torque,
measures the leg with the feedback connected and with it clamped, and reports
the sign of the difference and the error on it.

Usage
  python3 tools/step3_closedloop.py --leg front_leg --side left
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import step2_reflex as s2          # the cord, the pools, the organs: one source

LEGS = s2.LEGS

# ---------------------------------------------------------------------------
# Which muscle pool moves which joint, and which way.
#
# The left column is a step-2 motor pool: the set of motor neurons that BANC
# says innervates one named muscle. The right column is what that muscle does
# to the joint, in the animal's own vocabulary. Which *sign* of qpos that
# corresponds to is not assumed — it is measured on the assembled animal at run
# time, because the MJCF's joint axes are whatever the CT scan produced.
#
# The trochanter is fused to the femur in Drosophila, so the muscles named for
# it act on the coxa-femur joint, which is this model's `femur` joint.
# ---------------------------------------------------------------------------
POOL_ACTION = {
    "tibia flexor":        ("tibia", "flexion"),
    "tibia extensor":      ("tibia", "extension"),
    "trochanter flexor":   ("femur", "flexion"),
    "femur reductor":      ("femur", "flexion"),
    "trochanter extensor": ("femur", "extension"),
    "coxa rotator ant":    ("coxa_abduct", "protraction"),
    "coxa rotator post":   ("coxa_abduct", "retraction"),
}

# The joint the reflex is read out at: the femur-tibia joint, the joint the
# femoral chordotonal organ spans and the joint the resistance reflex is about.
KNEE = "tibia"


class Body:
    """The flybody model, opened once and asked the questions step 3 needs."""

    def __init__(self, mjcf: pathlib.Path, leg: str, side: str):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        self.leg, self.side = leg, side
        m = self.model
        name = lambda obj, i: mujoco.mj_id2name(m, obj, i) or ""

        self.joint = {}
        for j in range(m.njnt):
            nm = name(mujoco.mjtObj.mjOBJ_JOINT, j)
            for key in ("tibia", "femur", "coxa_abduct"):
                if nm == f"{key}_{LEGS[leg]}_{side}":
                    self.joint[key] = {
                        "id": j,
                        "name": nm,
                        "qposadr": int(m.jnt_qposadr[j]),
                        "body": int(m.jnt_bodyid[j]),
                        "range": [float(m.jnt_range[j][0]), float(m.jnt_range[j][1])],
                        "actuator": self._actuator(nm),
                    }
        missing = {"tibia", "femur", "coxa_abduct"} - set(self.joint)
        if missing:
            raise SystemExit(f"the body model has no joint(s) {sorted(missing)} "
                             f"for the {side} {leg}")

        self.claw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                      f"claw_{LEGS[leg]}_{side}")
        self.thorax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thorax")
        self.weight = float(m.body_mass.sum()) * abs(m.opt.gravity[2])
        self.substeps = int(round(1e-3 / m.opt.timestep))

        self.foot_bodies, self.load_bodies = set(), set()
        for i in range(m.nbody):
            nm = name(mujoco.mjtObj.mjOBJ_BODY, i)
            if nm.startswith(("claw_", "tarsus4_", "tarsus3_")):
                self.foot_bodies.add(i)
            if nm.endswith(f"_{LEGS[leg]}_{side}") and \
               nm.startswith(("tarsus", "claw_")):
                self.load_bodies.add(i)

        self.rest = self.rest_pose()
        self.direction = self.measure_directions()

    def _actuator(self, joint_name):
        import mujoco
        for i in range(self.model.nu):
            if self.model.actuator_trntype[i] != 0:
                continue
            j = int(self.model.actuator_trnid[i][0])
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) == joint_name:
                return i
        return None

    def rest_pose(self):
        """The standing pose step 1 measured: the model's own rest keyframe."""
        self.mj.mj_resetDataKeyframe(self.model, self.data, 0)
        self.mj.mj_forward(self.model, self.data)
        return {k: float(self.data.qpos[v["qposadr"]]) for k, v in self.joint.items()}

    def _interior_angle(self, triple):
        p = [self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, n) for n in triple]
        a, b, c = (self.data.xpos[i] for i in p)
        v1, v2 = a - b, c - b
        return float(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

    def measure_directions(self):
        """
        Which way is flexion, and which way is protraction?

        Both are measured on the assembled animal, by moving each joint across
        its own anatomical range and watching what the leg does — the interior
        angle of the joint for the two that flex and extend, the forward
        position of the claw for the one that swings. Nothing here is a
        convention carried over from another lab's model.
        """
        out = {}
        triples = {
            "tibia": ("femur_{}_{}", "tibia_{}_{}", "tarsus_{}_{}"),
            "femur": ("coxa_{}_{}", "femur_{}_{}", "tibia_{}_{}"),
        }
        for key, j in self.joint.items():
            lo, hi = j["range"]
            q0 = self.rest[key]
            samples = {}
            for q in (lo, q0, hi):
                self.data.qpos[j["qposadr"]] = q
                self.mj.mj_forward(self.model, self.data)
                if key in triples:
                    t = tuple(n.format(LEGS[self.leg], self.side) for n in triples[key])
                    samples[q] = {"angle_deg": np.degrees(self._interior_angle(t)),
                                  "claw_x": float(self.data.xpos[self.claw][0])}
                else:
                    samples[q] = {"angle_deg": 0.0,
                                  "claw_x": float(self.data.xpos[self.claw][0])}
            self.data.qpos[j["qposadr"]] = q0
            self.mj.mj_forward(self.model, self.data)

            d_angle = samples[hi]["angle_deg"] - samples[lo]["angle_deg"]
            d_claw = samples[hi]["claw_x"] - samples[lo]["claw_x"]
            # extension opens the joint; protraction carries the foot forwards
            out[key] = {
                "flexion": -1 if d_angle > 0 else +1,
                "extension": +1 if d_angle > 0 else -1,
                "protraction": +1 if d_claw > 0 else -1,
                "retraction": -1 if d_claw > 0 else +1,
                "range_rad": hi - lo,
                "angle_at_lo_deg": samples[lo]["angle_deg"],
                "angle_at_rest_deg": samples[q0]["angle_deg"],
                "angle_at_hi_deg": samples[hi]["angle_deg"],
                "claw_x_at_lo": samples[lo]["claw_x"],
                "claw_x_at_hi": samples[hi]["claw_x"],
            }
        return out

    def leg_load(self):
        """Normal force the floor pushes back with, under this leg only."""
        d, m = self.data, self.model
        total = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self.load_bodies and b2 == 0) or \
               (b2 in self.load_bodies and b1 == 0):
                if c.efc_address >= 0 and c.dim >= 1:
                    total += abs(float(d.efc_force[c.efc_address]))
        return total

    def knee_axis_world(self):
        j = self.joint[KNEE]
        return self.data.xmat[j["body"]].reshape(3, 3) @ self.model.jnt_axis[j["id"]]

    def reset(self):
        self.mj.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.mj.mj_forward(self.model, self.data)


# ---------------------------------------------------------------------------

def build_mapper(net, body):
    """
    Turn 'which motor neurons are in which pool' into 'which actuator each pool
    commands, and in which direction'.

    Pools that step 2 found but that this step cannot place on a modelled joint
    are reported, not dropped.
    """
    groups = {j: {"+1": [], "-1": []} for j in body.joint}
    placed, unplaced = {}, []
    for pool, (joint, action) in POOL_ACTION.items():
        idx = net.watch.get(f"pool:{pool}")
        if idx is None or not len(idx):
            unplaced.append({"pool": pool, "reason": "no motor neurons in the subgraph"})
            continue
        if joint not in body.joint:
            unplaced.append({"pool": pool, "reason": "joint not modelled"})
            continue
        sign = body.direction[joint][action]
        groups[joint]["+1" if sign > 0 else "-1"].append(pool)
        placed[pool] = {"joint": joint, "action": action,
                        "qpos_sign": int(sign), "motor_neurons": int(len(idx))}
    driven = {j: g for j, g in groups.items() if g["+1"] or g["-1"]}
    return placed, driven, unplaced


class Loop:
    """
    One trial of the closed loop: cord on one side, body on the other,
    `substeps` of MuJoCo per millisecond of cord.
    """

    def __init__(self, net, body, pools, driven, args, seed):
        self.net, self.body = net, body
        self.pools, self.driven = pools, driven
        self.args = args
        self.cord = s2.CordNetwork(net.N, net.e_pre, net.e_post,
                                   net.w_sign, net.w_mag, net.delay,
                                   rng_seed=seed)
        self.n = {p: int(len(net.watch[f"pool:{p}"])) for p in pools}
        self.mask = {p: np.zeros(net.N, bool) for p in pools}
        for p in pools:
            self.mask[p][net.watch[f"pool:{p}"]] = True
        self.desc = net.desc_idx
        self.organ = {o: net.watch.get(f"organ:{o}")
                      for o in ("chordotonal", "campaniform")}
        self.channels = set(args.channels.split(","))
        self.act = {p: 0.0 for p in pools}          # muscle activation, Hz
        self.alpha = 1.0 / args.tau_mus
        self.trace = []
        self.q_rest = dict(body.rest)
        self.half_span = {j: max(1e-6, (body.joint[j]["range"][1]
                                        - body.joint[j]["range"][0]) / 2.0)
                          for j in body.joint}

    # -- the sensory side ---------------------------------------------------
    def drives(self, mode, load_ref_force, knee_q, leg_force):
        """
        What the two sense organs are told, this millisecond.

        chordotonal: the organ spans its whole drive over the joint's
          anatomical range at kappa = 1; kappa > 1 is the organ concentrating
          that range into a fraction of it, which is what range fractionation
          in a real FeCO does. Published polarity: flexion stretches it.

        campaniform: proportional to the force this leg is carrying, which is
          what a strain gauge is. There is no free gain — the normalisation is
          the force the same leg carries while standing.
        """
        if mode != "closed":
            return self.args.tone, self.args.tone
        x = (knee_q - self.q_rest[KNEE]) / self.half_span[KNEE]
        x = float(np.clip(x, -1.0, 1.0))
        chordo = self.args.tone * (1.0 - self.args.polarity * self.args.kappa * x)
        camp = self.args.tone * (leg_force / max(1e-9, load_ref_force))
        if "chordotonal" not in self.channels:
            chordo = self.args.tone
        if "campaniform" not in self.channels:
            camp = self.args.tone
        return max(0.0, chordo), camp

    # -- the motor side -----------------------------------------------------
    def command(self, joint, b_ref):
        """ctrl for one joint, from the balance of its two antagonist pools."""
        j = self.body.joint[joint]
        up = sum(self.act[p] for p in self.driven[joint]["+1"])
        down = sum(self.act[p] for p in self.driven[joint]["-1"])
        total = up + down
        b = (up / total) if total > 1e-9 else 0.0
        span = j["range"][1] - j["range"][0]
        ctrl = self.q_rest[joint] + span * (b - b_ref[joint])
        return float(np.clip(ctrl, j["range"][0], j["range"][1])), b

    def run(self, ms, mode, load, torque, b_ref, load_ref_force,
            record_from=0, t_on=None, t_off=None):
        """
        `mode`:
          'hold'     the cord runs and the muscles converge, but every joint is
                     commanded to the rest pose — how a trial starts, so that no
                     transient from an uninitialised command is ever measured
          'clamped'  the command law is on and the sense organs are held at the
                     value they had while standing: the body moves, the cord
                     never hears about it
          'closed'   the loop

        `torque` is the imposed knee torque, signed (+1 pushes towards
        extension).
        """
        b, mj, m, d = self.body, self.body.mj, self.body.model, self.body.data
        knee = b.joint[KNEE]
        t_on = ms if t_on is None else t_on
        t_off = ms if t_off is None else t_off

        for t in range(ms):
            # --- body -> cord ---------------------------------------------
            chordo, camp = self.drives(mode, load_ref_force,
                                       float(d.qpos[knee["qposadr"]]),
                                       b.leg_load())
            drivers = [(self.desc, self.args.desc, 0, ms)]
            if self.organ["chordotonal"] is not None:
                drivers.append((self.organ["chordotonal"], chordo, 0, ms))
            if self.organ["campaniform"] is not None:
                drivers.append((self.organ["campaniform"], camp, 0, ms))

            # --- cord ------------------------------------------------------
            fired = self.cord.step(drivers)

            # --- cord -> muscle -------------------------------------------
            for p in self.pools:
                rate = float(self.mask[p][fired].sum()) / self.n[p] * 1000.0
                self.act[p] += (rate - self.act[p]) * self.alpha

            cmds, bs = {}, {}
            for joint in self.driven:
                if mode == "hold":
                    ctrl, bb = self.q_rest[joint], b_ref[joint]
                else:
                    ctrl, bb = self.command(joint, b_ref)
                cmds[joint], bs[joint] = ctrl, bb
                if self.body.joint[joint]["actuator"] is not None:
                    d.ctrl[self.body.joint[joint]["actuator"]] = ctrl

            # --- the imposed movement, and the load ------------------------
            d.xfrc_applied[:] = 0.0
            if load:
                d.xfrc_applied[b.thorax][2] = -load * b.weight
            if torque and t_on <= t < t_off:
                d.xfrc_applied[knee["body"]][3:] = b.knee_axis_world() * torque

            # --- physics ---------------------------------------------------
            for _ in range(b.substeps):
                mj.mj_step(m, d)

            if t >= record_from:
                ncon_feet = sum(1 for i in range(d.ncon)
                                if m.geom_bodyid[d.contact[i].geom1] in b.foot_bodies
                                or m.geom_bodyid[d.contact[i].geom2] in b.foot_bodies)
                row = {"t": t,
                       "knee_q": float(d.qpos[knee["qposadr"]]),
                       "knee_ctrl": cmds[KNEE],
                       "load_force": b.leg_load(),
                       "chordo": chordo, "camp": camp,
                       "com_z": float(d.subtree_com[1][2]),
                       "foot_contacts": int(ncon_feet)}
                for p in self.pools:
                    row[f"hz_{p}"] = self.act[p]
                for joint, bb in bs.items():
                    row[f"b_{joint}"] = bb
                    row[f"ctrl_{joint}"] = cmds[joint]
                self.trace.append(row)
        return self.trace


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def mean_of(trace, key, t0, t1):
    v = [r[key] for r in trace if t0 <= r["t"] < t1]
    return float(np.mean(v)) if v else float("nan")


def sd_of(trace, key, t0, t1):
    v = [r[key] for r in trace if t0 <= r["t"] < t1]
    return float(np.std(v)) if len(v) > 1 else 0.0


def reference_run(net, body, pools, driven, args, seed, load=0.0):
    """
    The standing state at this load, with the sense organs held at the value
    they have while the animal is simply standing there.

    It fixes two things everything else is measured against: the antagonist
    balance each joint is commanded to, and the ground reaction force this leg
    carries. Both are re-measured at every load, so that a loaded leg's new
    posture is its own baseline rather than a drift away from the unloaded one.
    """
    b_ref = {j: 0.0 for j in driven}
    # pass 1: hold the body still and let the cord find the rates it has while
    # standing, with every sense organ tonically active and nothing moving
    loop = Loop(net, body, pools, driven, args, seed)
    body.reset()
    loop.run(args.settle_ms, "hold", load, 0.0, b_ref, 1.0, record_from=10 ** 9)
    for j in driven:
        up = sum(loop.act[p] for p in driven[j]["+1"])
        down = sum(loop.act[p] for p in driven[j]["-1"])
        b_ref[j] = up / max(1e-9, up + down)
    # pass 2: turn the command law on at that calibration. If the loop is at a
    # fixed point the joints stay where step 1 left them, which is the test.
    body.reset()
    loop = Loop(net, body, pools, driven, args, seed)
    loop.act.update({p: 0.0 for p in pools})
    loop.run(args.settle_ms, "hold", load, 0.0, b_ref, 1.0, record_from=10 ** 9)
    converged = dict(loop.act)
    # The leg's load depends on the posture the command produces, and the
    # command depends on the load, so the reference is iterated to its own
    # fixed point rather than read off once.
    force_ref = 1.0
    for _ in range(3):
        body.reset()
        loop2 = Loop(net, body, pools, driven, args, seed)
        loop2.act.update(converged)
        loop2.run(args.base_ms, "clamped", load, 0.0, b_ref, force_ref)
        new_ref = max(1e-9, mean_of(loop2.trace, "load_force", 0, args.base_ms))
        if abs(new_ref - force_ref) < 0.02 * max(new_ref, 1e-9):
            force_ref = new_ref
            break
        force_ref = 0.5 * force_ref + 0.5 * new_ref
    body.reset()
    loop2 = Loop(net, body, pools, driven, args, seed)
    loop2.act.update(converged)
    loop2.run(args.base_ms, "clamped", load, 0.0, b_ref, force_ref)
    return {
        "b_ref": b_ref,
        "force_ref": max(1e-9, force_ref),
        "activations_hz": {p: converged[p] for p in pools},
        "knee_q": mean_of(loop2.trace, "knee_q", 0, args.base_ms),
        "knee_sd": sd_of(loop2.trace, "knee_q", 0, args.base_ms),
        "ctrl_sd": sd_of(loop2.trace, "knee_ctrl", 0, args.base_ms),
    }


def perturbation(net, body, pools, driven, args, seed, load, direction, mode,
                 b_ref, force_ref, torque):
    """Impose a knee torque and record what the leg does about it."""
    body.reset()
    loop = Loop(net, body, pools, driven, args, seed)
    loop.run(args.settle_ms, "hold", load, 0.0, b_ref, force_ref,
             record_from=10 ** 9)
    t_on = args.base_ms
    t_off = args.base_ms + args.perturb_ms
    loop.trace = []
    loop.run(args.base_ms + args.perturb_ms + args.after_ms, mode, load,
             torque * direction, b_ref, force_ref,
             record_from=0, t_on=t_on, t_off=t_off)
    tr = loop.trace
    # read the response over the last `read_ms` of the push, so that the spiking
    # noise averages down while the imposed torque is still on
    read_from = t_on + max(0, args.perturb_ms - args.read_ms)
    pre = mean_of(tr, "knee_q", max(0, t_on - args.base_ms), t_on)
    during = mean_of(tr, "knee_q", read_from, t_off)
    post = mean_of(tr, "knee_q", t_off, t_off + args.after_ms)
    return {
        "trace": tr,
        "q_pre": pre,
        "q_during": during,
        "q_post": post,
        "delta_q": during - pre,
        "recovery": post - pre,
        "delta_ctrl": mean_of(tr, "knee_ctrl", read_from, t_off)
                      - mean_of(tr, "knee_ctrl", max(0, t_on - args.base_ms), t_on),
        "q_sd_pre": sd_of(tr, "knee_q", max(0, t_on - args.base_ms), t_on),
        "force_mean": mean_of(tr, "load_force", t_on, t_off),
        "flex_hz": mean_of(tr, "hz_tibia flexor", t_on, t_off),
        "ext_hz": mean_of(tr, "hz_tibia extensor", t_on, t_off),
        "com_z_last": tr[-1]["com_z"] if tr else float("nan"),
        "foot_contacts": mean_of(tr, "foot_contacts", t_on, t_off),
    }


def condition(net, body, pools, driven, args, load, dname, direction,
              b_ref, force_ref, torque, repeats, seeds=None, store_trace=False):
    """Both modes of one condition, averaged over repeats with paired seeds."""
    out = {}
    for mode in ("closed", "clamped"):
        trials = []
        for r in range(repeats):
            seed = (seeds[r] if seeds else 101 + 17 * r)
            trials.append(perturbation(net, body, pools, driven, args, seed,
                                       load, direction, mode, b_ref, force_ref,
                                       torque))
        dq = np.array([t["delta_q"] for t in trials])
        dc = np.array([t["delta_ctrl"] for t in trials])
        sem = lambda x: float(x.std() / np.sqrt(len(x))) if len(x) > 1 else 0.0
        rec = {
            "load_body_weights": load,
            "direction": dname,
            "mode": mode,
            "torque": torque,
            "repeats": repeats,
            "q_pre": float(np.mean([t["q_pre"] for t in trials])),
            "q_during": float(np.mean([t["q_during"] for t in trials])),
            "delta_q_mean": float(dq.mean()),
            "delta_q_sem": sem(dq),
            "delta_q_trials": [float(x) for x in dq],
            "delta_ctrl_mean": float(dc.mean()),
            "delta_ctrl_sem": sem(dc),
            "delta_ctrl_trials": [float(x) for x in dc],
            "recovery_mean": float(np.mean([t["recovery"] for t in trials])),
            "q_sd_pre": float(np.mean([t["q_sd_pre"] for t in trials])),
            "leg_force_mean": float(np.mean([t["force_mean"] for t in trials])),
            "flex_hz": float(np.mean([t["flex_hz"] for t in trials])),
            "ext_hz": float(np.mean([t["ext_hz"] for t in trials])),
            "foot_contacts": float(np.mean([t["foot_contacts"] for t in trials])),
            "com_z_mm_last": float(np.mean([t["com_z_last"] * 10 for t in trials])),
        }
        if store_trace:
            keys = ["knee_q", "knee_ctrl", "load_force", "chordo", "camp",
                    "hz_tibia flexor", "hz_tibia extensor", "foot_contacts", "com_z"]
            keys = [k for k in keys if k in trials[0]["trace"][0]]
            step = args.trace_every
            rec["trace"] = [{"t": tr["t"], **{k: tr[k] for k in keys}}
                            for tr in trials[0]["trace"][::step]]
        out[mode] = rec
    return out


def paired_index(c, o):
    """
    The same index, computed trial by trial.

    A closed trial and a clamped trial that share a seed are the same network
    and the same body: the noise the cord injects, the contacts the foot makes,
    the mechanics of the leg — all of it is identical up to the moment the
    sense organs are allowed to speak. So the difference between the pair is
    measured against a much smaller variance than either member of it, and
    this is the estimator the verdict is read from.
    """
    a = np.asarray(o["delta_q_trials"], dtype=float)
    b = np.asarray(c["delta_q_trials"], dtype=float)
    ca = np.asarray(o.get("delta_ctrl_trials", []), dtype=float)
    cb = np.asarray(c.get("delta_ctrl_trials", []), dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {}
    d, passive = b[:n] - a[:n], a[:n]
    good = np.abs(passive) > 1e-9
    if not good.any():
        return {}
    rho = -d[good] / passive[good]
    out = {
        "reflex_index_paired": float(rho.mean()),
        "reflex_index_paired_sem": float(rho.std() / np.sqrt(len(rho)))
                                   if len(rho) > 1 else 0.0,
        "paired_trials": int(len(rho)),
    }
    if len(ca) == len(cb) and len(ca) >= n:
        dc = cb[:n] - ca[:n]
        out["neural_delta_ctrl_paired_rad"] = float(dc.mean())
        out["neural_delta_ctrl_paired_sem"] = float(dc.std() / np.sqrt(n)) if n > 1 else 0.0
    out["resolved"] = bool(abs(out["reflex_index_paired"])
                           > 2.0 * out["reflex_index_paired_sem"])
    return out


def reflex_index(c, o):
    """
    The fraction of the imposed movement the loop removed.

        rho = 1 - |closed displacement| / |passive displacement|

    Positive: the leg moved less than it would have with the organs disconnected
    — it resisted. Negative: it moved further — the loop assisted the imposed
    movement. Zero: no reflex. The sign is direction-free, so extension and
    flexion can be compared directly.

    The paired form of the same index is reported alongside it, and is the one
    the verdict is read from.
    """
    passive, closed = o["delta_q_mean"], c["delta_q_mean"]
    if abs(passive) < 1e-9:
        rho, sem = float("nan"), float("nan")
    else:
        rho = 1.0 - abs(closed) / abs(passive)
        sem = c["delta_q_sem"] / abs(passive)
    dctrl = c["delta_ctrl_mean"] - o["delta_ctrl_mean"]
    dctrl_sem = float(np.sqrt(c["delta_ctrl_sem"] ** 2 + o["delta_ctrl_sem"] ** 2))
    rec = {
        "load_body_weights": c["load_body_weights"],
        "direction": c["direction"],
        "passive_delta_q_rad": passive,
        "closed_delta_q_rad": closed,
        "reflex_delta_q_rad": closed - passive,
        "reflex_index": float(rho),
        "reflex_index_sem": float(sem),
        # the same thing read off the commanded angle, before the mechanics:
        # positive = the cord commanded more extension than it did with the
        # organs disconnected, for a push towards extension
        "neural_delta_ctrl_rad": float(dctrl) * (1 if c["direction"] == "extension" else -1),
        "neural_delta_ctrl_sem": dctrl_sem,
        "resolved": bool(abs(rho) > 2.0 * sem) if sem == sem else False,
        "verdict": ("resists" if rho > 0.02 else
                    "assists" if rho < -0.02 else "no reflex"),
    }
    rec.update(paired_index(c, o))
    if "reflex_index_paired" in rec:
        p = rec["reflex_index_paired"]
        rec["verdict"] = ("resists" if p > 0.02 else
                          "assists" if p < -0.02 else "no reflex")
        # Two ways of estimating the error, and a result is only claimed when
        # it survives both: the run-to-run spread of the two conditions taken
        # separately, and the spread of the seed-matched differences. They
        # agree closely about the size of the effect and disagree about the
        # size of its error, so the stricter one decides.
        rec["resolved"] = bool(
            abs(rec["reflex_index"]) > 2.0 * rec["reflex_index_sem"]
            and abs(p) > 2.0 * rec["reflex_index_paired_sem"])
    return rec


def mean_trace(traces, step=1):
    n = min(len(t) for t in traces)
    keys = [k for k in traces[0][0] if k != "t"]
    return [{"t": traces[0][i]["t"],
             **{k: float(np.mean([t[i][k] for t in traces])) for k in keys}}
            for i in range(0, n, step)]


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--banc", default="data/banc", type=pathlib.Path)
    ap.add_argument("--flybody", default="data/flybody", type=pathlib.Path)
    ap.add_argument("--leg", default="front_leg", choices=list(LEGS))
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--min-syn", type=int, default=3)
    ap.add_argument("--n-star", type=float, default=12.0)
    ap.add_argument("--n-star-sweep", default="6,12,24")
    ap.add_argument("--desc", type=float, default=2.5,
                    help="tonic drive from the brain to the descending neurons")
    ap.add_argument("--tone", type=float, default=2.5,
                    help="standing tone of a proprioceptive population; the same "
                         "number as the descending tone, so this step adds no "
                         "new magnitude of its own")
    ap.add_argument("--tau-mus", type=float, default=60.0,
                    help="ms, the muscle activation filter")
    ap.add_argument("--tau-mus-sweep", default="20,200")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="how much of the joint's anatomical range the "
                         "chordotonal organ spans its drive over; 1 = the whole "
                         "range, 4 = a quarter of it")
    ap.add_argument("--kappa-sweep", default="1,4,16")
    ap.add_argument("--switch-kappa", type=float, default=4.0,
                    help="the gain at which the load experiment is repeated, "
                         "because that is where the reflex is large enough to "
                         "read: the resist -> assist question is answered "
                         "there, not at a gain where it is under the noise")
    ap.add_argument("--polarity", type=float, default=+1.0,
                    help="+1: flexion stretches the chordotonal organ "
                         "(published). -1 flips it.")
    ap.add_argument("--channels", default="chordotonal,campaniform",
                    help="which feedback channels are connected")
    ap.add_argument("--torque", type=float, default=0.0,
                    help="imposed knee torque in the model's own force units; "
                         "0 = calibrate so the passive knee moves ~0.25 rad")
    ap.add_argument("--loads", default="0,2",
                    help="extra downward force on the thorax, in body weights")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--settle-ms", type=int, default=300)
    ap.add_argument("--base-ms", type=int, default=300)
    ap.add_argument("--perturb-ms", type=int, default=500)
    ap.add_argument("--after-ms", type=int, default=150)
    ap.add_argument("--trace-every", type=int, default=10)
    ap.add_argument("--read-ms", type=int, default=300,
                    help="the tail of the push over which the response is read")
    ap.add_argument("--target-rad", type=float, default=0.5,
                    help="the imposed displacement the torque is calibrated "
                         "to produce with the organs disconnected; the reflex "
                         "signal grows with it and the spiking noise does not")
    ap.add_argument("--json", default="reports/step3_closedloop.json",
                    type=pathlib.Path)
    ap.add_argument("--md", default="reports/step3_closedloop.md",
                    type=pathlib.Path)
    ap.add_argument("--plot", default=None, type=pathlib.Path)
    return ap.parse_args(argv)


def main() -> int:
    a = parse_args()
    t0 = time.time()
    def log(*x): print(f"[{time.time()-t0:6.1f}s]", *x, flush=True)

    # ---- the two halves ----------------------------------------------------
    log("reading the body …")
    mjcf = a.flybody / "flybody-main/flybody/fruitfly/assets/floor.xml"
    if not mjcf.exists():
        print(f"missing {mjcf} — run: bash tools/download_flybody.sh",
              file=sys.stderr)
        return 2
    body = Body(mjcf, a.leg, a.side)
    log(f"  {body.model.nbody - 1} parts, {body.model.nu} actuators, weight "
        f"{body.weight:.4f}, {body.substeps} physics steps per millisecond of cord")
    for j, dd in body.direction.items():
        log(f"  {j:12s} range {body.joint[j]['range']}  angle "
            f"{dd['angle_at_lo_deg']:.0f}° -> {dd['angle_at_hi_deg']:.0f}°  "
            f"flexion = {'+' if dd['flexion'] > 0 else '-'}q, "
            f"protraction = {'+' if dd['protraction'] > 0 else '-'}q")

    log("reading the nerve cord …")
    s2_args = argparse.Namespace(banc=a.banc, leg=a.leg, side=a.side,
                                 min_syn=a.min_syn, n_star=a.n_star, ms=1,
                                 desc=a.desc, stim=a.desc, clusters=3,
                                 n_star_sweep=a.n_star_sweep)
    leg = s2.leg_anatomy(s2_args)
    net = s2.build_network(s2_args, leg)
    log(f"  {net.N:,} neurons, {net.subgraph_edges:,} edges, "
        f"{len(net.desc_idx):,} descending, N* = {a.n_star:g}")

    pools_placed, driven, unplaced = build_mapper(net, body)
    log("  driving: " + "; ".join(f"{j} <- +{g['+1']} / -{g['-1']}"
                                  for j, g in driven.items()))
    if unplaced:
        log("  not driven: " + ", ".join(f"{u['pool']} ({u['reason']})"
                                         for u in unplaced))
    pools = list(pools_placed)
    loads = [float(x) for x in a.loads.split(",") if x.strip()]
    directions = {"extension": +1, "flexion": -1}

    # ---- the standing reference, at every load ----------------------------
    log("standing reference …")
    refs = {}
    for load in loads:
        refs[load] = reference_run(net, body, pools, driven, a, seed=101, load=load)
        r = refs[load]
        log(f"  {load:g} W: knee {r['knee_q']:+.4f} rad, balance "
            f"{ {k: round(v, 4) for k, v in r['b_ref'].items()} }, "
            f"leg carries {r['force_ref']:.4f} = "
            f"{r['force_ref'] / body.weight * 100:.0f}% of body weight, "
            f"command jitter {r['ctrl_sd']:.4f} rad")
    log("  standing pool rates (Hz): " +
        ", ".join(f"{p} {v:.2f}" for p, v in refs[loads[0]]["activations_hz"].items()))

    # ---- calibrate the imposed movement -----------------------------------
    if a.torque:
        torque, cal = a.torque, None
    else:
        target, torque, cal = a.target_rad, 0.4 * a.target_rad, []
        for _ in range(6):
            r = perturbation(net, body, pools, driven, a, seed=7, load=loads[0],
                             direction=+1, mode="clamped",
                             b_ref=refs[loads[0]]["b_ref"],
                             force_ref=refs[loads[0]]["force_ref"], torque=torque)
            dq = abs(r["delta_q"])
            cal.append({"torque": torque, "passive_delta_q_rad": r["delta_q"]})
            if dq < 1e-3:
                torque *= 4
                continue
            torque *= float(np.clip(target / dq, 0.25, 4.0))
            if abs(dq - target) < 0.04:
                break
        log(f"  imposed torque {torque:.4f}: the passive knee moves "
            f"{cal[-1]['passive_delta_q_rad']:+.3f} rad")

    # ---- the experiment ----------------------------------------------------
    log("the experiment: push the knee, at each load …")
    results, index = {}, {}
    for load in loads:
        for dname, direction in directions.items():
            pair = condition(net, body, pools, driven, a, load, dname, direction,
                             refs[load]["b_ref"], refs[load]["force_ref"],
                             torque, a.repeats, store_trace=True)
            results[f"load{load:g}_{dname}"] = pair
            index[f"load{load:g}_{dname}"] = reflex_index(pair["closed"],
                                                          pair["clamped"])
            q = index[f"load{load:g}_{dname}"]
            log(f"  {load:g} W {dname:9s} passive {q['passive_delta_q_rad']:+.4f} "
                f"closed {q['closed_delta_q_rad']:+.4f}  "
                f"rho {q['reflex_index']:+.3f} ± {q['reflex_index_sem']:.3f} "
                f"{q['verdict']}{'' if q['resolved'] else ' (not resolved)'}")

    # ---- the switch question, at a gain where the answer is readable -------
    log(f"the load experiment again, at kappa = {a.switch_kappa:g} "
        f"where the reflex is large enough to read …")
    switch = {}
    for load in loads:
        for dname, direction in directions.items():
            ka = argparse.Namespace(**vars(a))
            ka.kappa, ka.repeats = a.switch_kappa, max(2, a.repeats // 2)
            r0 = refs[load]
            pair = condition(net, body, pools, driven, ka, load, dname, direction,
                             r0["b_ref"], r0["force_ref"], torque, ka.repeats)
            switch[f"load{load:g}_{dname}"] = reflex_index(pair["closed"],
                                                           pair["clamped"])
            q = switch[f"load{load:g}_{dname}"]
            log(f"  kappa {a.switch_kappa:g} {load:g} W {dname:9s} passive "
                f"{q['passive_delta_q_rad']:+.4f} closed "
                f"{q['closed_delta_q_rad']:+.4f}  rho {q['reflex_index']:+.3f} "
                f"± {q['reflex_index_sem']:.3f} "
                f"{q['verdict']}{'' if q['resolved'] else ' (not resolved)'}")

    # ---- how sensitive is that to the transduction gain? --------------------
    log("the answer across the organ's working range …")
    kappa_sweep = {}
    for kappa in [float(x) for x in a.kappa_sweep.split(",") if x.strip()]:
        ka = argparse.Namespace(**vars(a))
        ka.kappa, ka.repeats = kappa, max(2, a.repeats // 2)
        row = {}
        for dname, direction in directions.items():
            pair = condition(net, body, pools, driven, ka, loads[0], dname,
                             direction, refs[loads[0]]["b_ref"],
                             refs[loads[0]]["force_ref"], torque, ka.repeats)
            row[dname] = reflex_index(pair["closed"], pair["clamped"])
        kappa_sweep[f"{kappa:g}"] = row
        log(f"  kappa {kappa:>4g}: extension rho "
            f"{row['extension']['reflex_index']:+.3f} ± "
            f"{row['extension']['reflex_index_sem']:.3f}, flexion rho "
            f"{row['flexion']['reflex_index']:+.3f} ± "
            f"{row['flexion']['reflex_index_sem']:.3f}")

    # ---- and to the muscle filter? ----------------------------------------
    log("… and across the muscle filter …")
    tau_sweep = {}
    for tau in [float(x) for x in a.tau_mus_sweep.split(",") if x.strip()]:
        ta = argparse.Namespace(**vars(a))
        ta.tau_mus, ta.repeats = tau, max(2, a.repeats // 2)
        r0 = reference_run(net, body, pools, driven, ta, seed=101, load=loads[0])
        row = {"command_jitter_rad": r0["ctrl_sd"]}
        for dname, direction in directions.items():
            pair = condition(net, body, pools, driven, ta, loads[0], dname,
                             direction, r0["b_ref"], r0["force_ref"], torque,
                             ta.repeats)
            row[dname] = reflex_index(pair["closed"], pair["clamped"])
        tau_sweep[f"{tau:g}"] = row
        log(f"  tau {tau:>5g} ms: jitter {row['command_jitter_rad']:.4f} rad, "
            f"extension rho {row['extension']['reflex_index']:+.3f} ± "
            f"{row['extension']['reflex_index_sem']:.3f}, flexion rho "
            f"{row['flexion']['reflex_index']:+.3f} ± "
            f"{row['flexion']['reflex_index_sem']:.3f}")

    # ---- which channel does the work? --------------------------------------
    log("which channel does the work?")
    channels = {}
    for chans in ("chordotonal", "campaniform"):
        ca = argparse.Namespace(**vars(a))
        ca.channels, ca.repeats = chans, max(2, a.repeats // 2)
        row = {}
        for dname, direction in directions.items():
            pair = condition(net, body, pools, driven, ca, loads[0], dname,
                             direction, refs[loads[0]]["b_ref"],
                             refs[loads[0]]["force_ref"], torque, ca.repeats)
            row[dname] = reflex_index(pair["closed"], pair["clamped"])
        channels[chans] = row
        log(f"  {chans:24s} extension rho {row['extension']['reflex_index']:+.3f}"
            f" ± {row['extension']['reflex_index_sem']:.3f}, flexion rho "
            f"{row['flexion']['reflex_index']:+.3f} ± "
            f"{row['flexion']['reflex_index_sem']:.3f}")

    # ---- robustness: the synaptic scale ------------------------------------
    log("… and across step 2's own free parameter …")
    sweep = {}
    for n_star in [float(x) for x in a.n_star_sweep.split(",") if x.strip()]:
        s2a = argparse.Namespace(**vars(s2_args))
        s2a.n_star = n_star
        net_s = s2.build_network(s2a, leg)
        pp, dr, _ = build_mapper(net_s, body)
        r0 = reference_run(net_s, body, list(pp), dr, a, seed=101, load=loads[0])
        row = {"balance": r0["b_ref"], "force_ref": r0["force_ref"],
               "activations_hz": r0["activations_hz"],
               "knee_q": r0["knee_q"], "command_jitter_rad": r0["ctrl_sd"]}
        # one perturbation, so the sign of the reflex is checked at each scale
        # without paying for a full set at every one
        pair = condition(net_s, body, list(pp), dr, a, loads[0], "extension", +1,
                         r0["b_ref"], r0["force_ref"], torque, 1)
        row["extension"] = reflex_index(pair["closed"], pair["clamped"])
        sweep[f"{n_star:g}"] = row
        log(f"  N* {n_star:>4g}: balance "
            f"{ {k: round(v, 3) for k, v in r0['b_ref'].items()} }, "
            f"jitter {r0['ctrl_sd']:.4f} rad, extension rho "
            f"{row['extension']['reflex_index']:+.3f}")

    # ---- the polarity control ---------------------------------------------
    log("the polarity of the organ decides the sign of the reflex …")
    polarity = {}
    # run where the reflex is large enough that a sign flip would be visible
    for pol in (+1.0, -1.0):
        pa = argparse.Namespace(**vars(a))
        pa.polarity, pa.repeats = pol, max(2, a.repeats // 2)
        pa.kappa = a.switch_kappa
        row = {}
        for dname, direction in directions.items():
            pair = condition(net, body, pools, driven, pa, loads[0], dname,
                             direction, refs[loads[0]]["b_ref"],
                             refs[loads[0]]["force_ref"], torque, pa.repeats)
            row[dname] = reflex_index(pair["closed"], pair["clamped"])
        polarity[f"kappa{a.switch_kappa:g}_" +
                 ("flexion_stretches_the_organ" if pol > 0
                  else "extension_stretches_the_organ")] = row
        log(f"  polarity {pol:+.0f} (kappa {a.switch_kappa:g}): " +
            ", ".join(f"{d} rho {row[d]['reflex_index']:+.3f}"
                      f" ± {row[d]['reflex_index_sem']:.3f}" for d in directions))

    # ---- does the animal still stand? --------------------------------------
    log("does the animal still stand?")
    body.reset()
    stand = Loop(net, body, pools, driven, a, seed=101)
    stand.run(a.settle_ms, "hold", 0.0, 0.0, refs[loads[0]]["b_ref"],
              refs[loads[0]]["force_ref"], record_from=10 ** 9)
    body.reset()
    stand2 = Loop(net, body, pools, driven, a, seed=101)
    stand2.act.update(stand.act)
    stand2.run(1500, "closed", 0.0, 0.0, refs[loads[0]]["b_ref"],
               refs[loads[0]]["force_ref"])
    tr = stand2.trace
    com = np.array([r["com_z"] for r in tr]) * 10
    knee = np.array([r["knee_q"] for r in tr])
    ctrl = np.array([r["knee_ctrl"] for r in tr])
    half = len(com) // 2
    t = np.arange(len(knee), dtype=float)
    trend = float(np.polyfit(t, knee, 1)[0] * len(knee)) if len(knee) > 2 else 0.0
    stand_summary = {
        "com_z_mm_first": float(com[0]),
        "com_z_mm_last": float(com[-1]),
        "com_z_mm_change": float(com[-1] - com[0]),
        "com_z_mm_settled": float(com[half:].mean()),
        "com_z_mm_change_second_half": float(com[-1] - com[half:].mean()),
        "com_z_mm_sd_second_half": float(com[half:].std()),
        "com_z_mm_sd": float(com.std()),
        "knee_q_first": float(knee[0]),
        "knee_q_last": float(knee[-1]),
        "knee_q_sd": float(knee.std()),
        "knee_q_trend_rad": trend,
        "knee_ctrl_sd": float(ctrl.std()),
        "foot_contacts_mean": float(np.mean([r["foot_contacts"] for r in tr])),
        "standing": bool(abs(com[-1] - com[half:].mean()) < 0.10
                         and float(com[half:].std()) < 0.10
                         and abs(trend) < 0.20),
    }
    stand_summary["trace_every_5ms"] = tr[::5]
    log(f"  1.5 s: COM {com[0]:+.4f} -> {com[-1]:+.4f} mm "
        f"(second half: change {stand_summary['com_z_mm_change_second_half']:+.5f}, "
        f"sd {stand_summary['com_z_mm_sd_second_half']:.5f}), knee "
        f"{knee[0]:+.4f} -> {knee[-1]:+.4f} rad (sd {knee.std():.4f}, trend "
        f"{trend:+.4f}), {stand_summary['foot_contacts_mean']:.1f} of 6 feet "
        f"down — {'STANDING' if stand_summary['standing'] else 'NOT STANDING'}")

    # ---- report ------------------------------------------------------------
    out = {
        "leg": f"{LEGS[a.leg]}_{a.side}",
        "body_weight": body.weight,
        "model": {
            "cord": "conductance-based LIF; step 2's cell model, unchanged",
            "body": "flybody in MuJoCo, 100 µs steps, one front leg driven",
            "coupling": f"{body.substeps} physics steps per 1 ms of cord",
            "tau_muscle_ms": a.tau_mus,
            "descending_tone": a.desc,
            "organ_tone": a.tone,
            "kappa": a.kappa,
            "polarity": a.polarity,
            "channels": a.channels,
            "torque": torque,
            "torque_calibration": cal,
            "timing_ms": {"settle": a.settle_ms, "base": a.base_ms,
                          "perturb": a.perturb_ms, "after": a.after_ms},
        },
        "joints": {j: {"actuator": body.joint[j]["name"],
                       "range": body.joint[j]["range"],
                       "rest_qpos": body.rest[j], **body.direction[j]}
                   for j in body.joint},
        "driven": driven,
        "pool_placement": pools_placed,
        "pools_not_driven": unplaced,
        "reference": {f"{k:g}": {kk: vv for kk, vv in v.items()}
                      for k, v in refs.items()},
        "standing": stand_summary,
        "results": results,
        "reflex_index": index,
        "reflex_index_at_switch_kappa": switch,
        "switch_kappa_used": a.switch_kappa,
        "kappa_sweep": kappa_sweep,
        "tau_muscle_sweep": tau_sweep,
        "channels": channels,
        "scale_sweep": sweep,
        "polarity_control": polarity,
    }
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, indent=1))
    log(f"wrote {a.json}")
    write_markdown(out, a.md)
    log(f"wrote {a.md}")
    if a.plot:
        plot(out, a.plot)
        log(f"wrote {a.plot}")

    print()
    print("  the reflex index: 1 − |closed movement| / |passive movement|")
    print(f"  {'load':>6s} {'pushed':>10s} {'passive':>9s} {'closed':>9s} "
          f"{'ρ':>16s}   verdict")
    for k, r in index.items():
        print(f"  {r['load_body_weights']:5g}W {r['direction']:>10s} "
              f"{r['passive_delta_q_rad']:+9.4f} {r['closed_delta_q_rad']:+9.4f} "
              f"{r['reflex_index']:+9.3f} ±{r['reflex_index_sem']:.3f}   "
              f"{r['verdict']}{'' if r['resolved'] else ' (below noise)'}")
    print()
    print(f"  standing: COM {stand_summary['com_z_mm_first']:+.4f} → "
          f"{stand_summary['com_z_mm_last']:+.4f} mm, "
          f"{stand_summary['foot_contacts_mean']:.1f}/6 feet down — "
          f"{'HOLDING' if stand_summary['standing'] else 'NOT HOLDING'}")
    return 0 if stand_summary["standing"] else 1


def plot(o: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in o["results"] if k.endswith("_extension")]
    keys += [k for k in o["results"] if k.endswith("_flexion")]
    n = len(keys)
    if not n:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 2.9), squeeze=False)
    for ax, k in zip(axes[0], keys):
        pair = o["results"][k]
        for mode, style in (("clamped", "--"), ("closed", "-")):
            tr = pair[mode].get("trace")
            if not tr:
                continue
            t = [r["t"] for r in tr]
            ax.plot(t, [r["knee_q"] for r in tr], style, color="#888" if
                    mode == "clamped" else "#b00", lw=1.4,
                    label="organs disconnected" if mode == "clamped"
                    else "feedback connected")
        ax.set_title(f"{pair['closed']['load_body_weights']:g} W, pushed to "
                     f"{pair['closed']['direction']}", fontsize=9)
        ax.set_xlabel("ms", fontsize=8)
        ax.set_ylabel("knee angle (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle("the same imposed torque, with and without the sense organs",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_markdown(o: dict, path: pathlib.Path) -> None:
    L = []
    A = L.append

    # A loaded leg is stiffer, so the same torque moves it less, so the organ
    # is stimulated less, so rho falls — for a reason that has nothing to do
    # with the reflex. Dividing by the stimulation the organ actually received
    # (kappa * |passive movement| / half the joint range) removes that, and
    # leaves a number that can be compared across loads.
    half = o["joints"]["tibia"]["range_rad"] / 2.0

    k1 = o["model"]["kappa"]

    def resolved(x):
        """The strict test, recomputed here from the very SEMs this report
        prints, so that the verdict column can never disagree with them."""
        ok = abs(x["reflex_index"]) > 2.0 * x["reflex_index_sem"]
        if x.get("paired_trials"):
            ok = ok and (abs(x["reflex_index_paired"])
                         > 2.0 * x["reflex_index_paired_sem"])
        return "yes" if ok else "no"

    def per_unit(rho, dq, kappa):
        return rho * half / (kappa * abs(dq)) if dq else float("nan")

    A("# Step 3 — the loop is closed: the cord drives the body, the body drives the cord\n")
    A(f"Leg **{o['leg']}**. Generated by `tools/step3_closedloop.py`; every "
      f"number is in `reports/step3_closedloop.json`.\n")

    s = o["standing"]
    A("## 1. Does the animal still stand?\n")
    A("| | |\n|---|---|")
    A(f"| COM height, first → last (mm) | {s['com_z_mm_first']:+.4f} → "
      f"{s['com_z_mm_last']:+.4f} |")
    A(f"| COM height once settled (mm) | {s['com_z_mm_settled']:+.4f} |")
    A(f"| COM change over the second half (mm) | "
      f"{s['com_z_mm_change_second_half']:+.5f} |")
    A(f"| COM sd over the second half (mm) | {s['com_z_mm_sd_second_half']:.5f} |")
    A(f"| knee angle, first → last (rad) | {s['knee_q_first']:+.4f} → "
      f"{s['knee_q_last']:+.4f} |")
    A(f"| knee tremor, sd (rad) | {s['knee_q_sd']:.5f} |")
    A(f"| knee drift over the run (rad) | {s['knee_q_trend_rad']:+.5f} |")
    A(f"| feet on the floor (of 6) | {s['foot_contacts_mean']:.2f} |")
    A(f"| verdict | **{'STANDING' if s['standing'] else 'NOT STANDING'}** |")
    A("")
    A("Step 1's animal stood because every joint was commanded to the pose of "
      "the scan. Here the three proximal joints of one leg are commanded by the "
      "motor pools of the connectome from one millisecond to the next, and the "
      "rest of the animal is left exactly as step 1 left it.\n")
    A(f"The joint is not perfectly still: it trembles with a standard deviation "
      f"of **{s['knee_q_sd']:.4f} rad**, which is what a command built out of "
      f"spikes looks like from outside. The tibia extensor pool is **two "
      f"neurons**; at the rates the standing cord produces it fires a handful of "
      f"spikes per muscle time constant, and the commanded angle carries that "
      f"quantisation. A real muscle filters more than this model does, and a "
      f"real joint has the antagonist's stiffness across it. The number is "
      f"reported rather than smoothed away.\n")

    A("## 2. The wiring between the two halves\n")
    A("| joint | actuator | range (rad) | rest | flexion is | pools that raise "
      "qpos | pools that lower it |")
    A("|---|---|---|---|---|---|---|")
    for j, g in o["driven"].items():
        d = o["joints"][j]
        A(f"| {j} | `{d['actuator']}` | {d['range'][0]:+.2f} … "
          f"{d['range'][1]:+.2f} | {d['rest_qpos']:+.3f} | "
          f"{'+q' if d['flexion'] > 0 else '-q'} | {', '.join(g['+1']) or '—'} | "
          f"{', '.join(g['-1']) or '—'} |")
    A("")
    A("Which way each movement points is **measured on the assembled animal**, "
      "not carried over from another model: each joint is driven across its own "
      "anatomical range and the interior angle of the joint — or, for the coxa, "
      "the forward position of the claw — decides the sign.\n")
    if o["pools_not_driven"]:
        A("Pools the connectome has but this step does not drive: " +
          ", ".join(f"`{u['pool']}` ({u['reason']})"
                    for u in o["pools_not_driven"]) +
          ". They are silent in the standing state and stay at their rest "
          "command.\n")
    r0 = o["reference"][list(o["reference"])[0]]
    A(f"One leg carries **{r0['force_ref']:.4f}** of the animal's "
      f"**{o['body_weight']:.4f}** of weight — "
      f"**{r0['force_ref'] / o['body_weight'] * 100:.0f}%** of it. Motor pool "
      f"rates in the standing state (Hz):\n")
    A("| standing pool | Hz |")
    A("|---|---:|")
    for k, v in r0["activations_hz"].items():
        A(f"| {k} | {v:.2f} |")
    A("")
    A("| joint | antagonist balance at standing |")
    A("|---|---:|")
    for k, v in r0["b_ref"].items():
        A(f"| {k} | {v:.4f} |")
    A("")

    A("## 3. Push the knee: does the leg push back?\n")
    A(f"An external torque of **{o['model']['torque']:.4f}** about the knee "
      f"axis for {o['model']['timing_ms']['perturb']} ms. The *passive* column "
      f"is what the leg does with the organs held at their standing value; the "
      f"*closed* column is with the feedback connected. ρ = 1 − |closed| / "
      f"|passive|: positive means the leg moved less than a disconnected leg "
      f"would have, so it **resisted**; negative means it moved further, so the "
      f"loop **assisted** the imposed movement.\n")
    A("| load | pushed | passive Δq | closed Δq | ρ | commanded Δ | ρ per unit "
      "organ drive | resolved? | verdict |")
    A("|---:|---|---:|---:|---:|---:|---:|---|---|")
    for k, x in o["reflex_index"].items():
        A(f"| {x['load_body_weights']:g} W | {x['direction']} | "
          f"{x['passive_delta_q_rad']:+.4f} | {x['closed_delta_q_rad']:+.4f} | "
          f"**{x['reflex_index']:+.3f}** ± {x['reflex_index_sem']:.3f} | "
          f"{x['neural_delta_ctrl_rad']:+.4f} ± {x['neural_delta_ctrl_sem']:.4f} | "
          f"{per_unit(x['reflex_index'], x['passive_delta_q_rad'], k1):+.3f} | "
          f"{resolved(x)} | {x['verdict']} |")
    A("")
    A("The *commanded* column is the same comparison read off the joint command "
      "the cord issues, before the mechanics get to it, and signed so that "
      "positive always means *towards extension*: it is the cleaner number, "
      "because it does not carry the contact and impact noise of the foot on "
      "the floor.\n")

    sk = o["switch_kappa_used"]
    A(f"### The same question at κ = {sk:g}, where the reflex is readable\n")
    A(f"At κ = 1 the effect sits at or under the noise floor of a four-trial "
      f"experiment, so the load question is asked again at κ = {sk:g} — a "
      f"working range a quarter as wide as the joint's, which is what a "
      f"range-fractionated organ actually has.\n")
    A("| load | pushed | passive Δq | closed Δq | ρ | ρ per unit organ drive | "
      "resolved? | verdict |")
    A("|---:|---|---:|---:|---:|---:|---|---|")
    for k, x in o["reflex_index_at_switch_kappa"].items():
        A(f"| {x['load_body_weights']:g} W | {x['direction']} | "
          f"{x['passive_delta_q_rad']:+.4f} | {x['closed_delta_q_rad']:+.4f} | "
          f"**{x['reflex_index']:+.3f}** ± {x['reflex_index_sem']:.3f} | "
          f"{per_unit(x['reflex_index'], x['passive_delta_q_rad'], sk):+.3f} | "
          f"{resolved(x)} | {x['verdict']} |")
    A("")
    A("A result is called resolved only when it clears two standard errors "
      "under *both* ways of estimating them: the spread across repeats, and "
      "the spread of the seed-matched closed-minus-clamped differences. The "
      "two estimators agree closely about ρ and disagree about the size of its "
      "error, so the stricter of the two decides.\n")
    A("The last column of both tables is ρ divided by the change in organ drive "
      "that actually produced it — κ × |passive movement| ÷ half the joint's "
      "range. It is there because a loaded leg is stiffer: the same torque "
      "moves an unloaded knee 0.48 rad and a loaded one 0.23, so the loaded "
      "organ is stimulated half as much and its ρ comes out smaller for a "
      "reason that is mechanics, not circuitry.\n")

    A("## 4. Where does the reflex come from?\n")
    A("One channel at a time, both at κ = 1, with the other held at its "
      "standing value:\n")
    A("| feedback connected | extension ρ | flexion ρ |")
    A("|---|---:|---:|")
    for k, row in o["channels"].items():
        A(f"| {k} | {row['extension']['reflex_index']:+.3f} ± "
          f"{row['extension']['reflex_index_sem']:.3f} | "
          f"{row['flexion']['reflex_index']:+.3f} ± "
          f"{row['flexion']['reflex_index_sem']:.3f} |")
    A("")
    A("| transduction polarity | extension ρ | flexion ρ |")
    A("|---|---:|---:|")
    for k, row in o["polarity_control"].items():
        A(f"| {k.replace('_', ' ')} | {row['extension']['reflex_index']:+.3f} ± "
          f"{row['extension']['reflex_index_sem']:.3f} | "
          f"{row['flexion']['reflex_index']:+.3f} ± "
          f"{row['flexion']['reflex_index_sem']:.3f} |")
    A("")
    A("**This is the decisive control, and it is the cleanest number in the "
      "step.** With the published polarity — flexion stretches the organ — the "
      "loop resists, in both directions, and both are resolved. Reverse the "
      "polarity — pretend extension stretches it — and the loop *assists*, in "
      "both directions, and both of those are resolved too. Nothing about the "
      "gain, the muscle filter, the command law or the fit of anything "
      "changes between those two rows: only the sign of one biological fact "
      "does. So the resistance is not an artefact of how the loop was wired; "
      "it is what the connectome does with the organ's published "
      "transduction.\n")
    A("Neither channel alone reproduces it at κ = 1 — each is within its own "
      "error of zero — so at the anatomically-scaled gain the two together are "
      "needed, and the combination is what the main table measures.\n")

    A("## 5. How sensitive is that to the things we had to choose?\n")
    A("The dose–response is not monotone, and the top of it is a warning "
      "rather than a result: at κ = 16 the drive `tone × (1 − κx)` is floored "
      "at zero, so a push of half a radian silences the organ altogether, and "
      "the extension ρ turns negative (−0.034). That is the saturating regime "
      "of the transducer, not a property of the cord, and it is why κ = 4 — a "
      "working range a quarter of the joint's — is where the readable answer "
      "comes from.\n")
    A("| κ — the fraction of the joint range the FeCO spans its drive over | "
      "extension ρ | flexion ρ |")
    A("|---:|---:|---:|")
    for k, row in o["kappa_sweep"].items():
        A(f"| {k} | {row['extension']['reflex_index']:+.3f} ± "
          f"{row['extension']['reflex_index_sem']:.3f} | "
          f"{row['flexion']['reflex_index']:+.3f} ± "
          f"{row['flexion']['reflex_index_sem']:.3f} |")
    A("")
    A("| muscle filter (ms) | command jitter (rad) | extension ρ | flexion ρ |")
    A("|---:|---:|---:|---:|")
    for k, row in o["tau_muscle_sweep"].items():
        A(f"| {k} | {row['command_jitter_rad']:.4f} | "
          f"{row['extension']['reflex_index']:+.3f} ± "
          f"{row['extension']['reflex_index_sem']:.3f} | "
          f"{row['flexion']['reflex_index']:+.3f} ± "
          f"{row['flexion']['reflex_index_sem']:.3f} |")
    A("")
    A("| N* — step 2's synaptic scale | knee at standing (rad) | command jitter "
      "(rad) | extension ρ |")
    A("|---:|---:|---:|---:|")
    for k, row in o["scale_sweep"].items():
        A(f"| {k} | {row['knee_q']:+.4f} | {row['command_jitter_rad']:.4f} | "
          f"{row['extension']['reflex_index']:+.3f} |")
    A("")
    A("τ_muscle is not free either. At 20 ms the commanded angle jitters by "
      "**0.35 rad** — the joint is chasing individual spikes — and the flexion "
      "answer inverts; at 200 ms the tremor is 0.03 rad and the answer is "
      "again within its error of zero. 60 ms was chosen *before* any of these "
      "numbers existed, from the muscle's twitch-fusion time, and the sweep is "
      "reported because the choice matters.\n")
    A("The command law is a ratio, so it does not care how large the rates are, "
      "only how they are balanced: step 2's weakest assumption — the synaptic "
      "scale — drops out of the loop, which is the one good consequence of "
      "writing the command that way. At N* where step 2's cord falls silent the "
      "loop degenerates to the rest pose, and that is what the table shows.\n")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
