#!/usr/bin/env python3
"""
Step 4 — the world.

Steps 1 to 3 progressively handed the animal over to its own nervous system:
step 1 measured the body, step 2 took the leg reflex out of the connectome,
step 3 closed the loop on ONE leg and found that the animal still stands and
that the leg resists being pushed.

This step drives all six legs, each from its own copy of the cord, and then
does the thing the other three could not: it lets you watch. It writes a
recording of the animal — every visible part of the body, frame by frame — and
a viewer that plays it back, because a standing fly is a claim until it is
something you can look at.

Nothing here decides what the fly should do. There is no gait, no schedule, no
target velocity and no clock anywhere in the controller. Each leg's cord gets a
tonic drive from the brain, its own two sense organs, and the wiring of the
connectome; where the animal goes is where that takes it.

Outputs
  world/fly.bin        the body's visible geometry, welded and packed once
  world/frames.bin     one row of positions and orientations per frame
  world/world.json     what everything is, and what the animal did
  reports/step4_world.md / .json

Run the viewer with `python3 tools/serve_world.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import step2_reflex as s2                                    # noqa: E402
import step3_closedloop as s3                                # noqa: E402

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------
# The six legs, in the animal's own nomenclature. BANC calls them front,
# middle and hind; the body model calls them T1, T2 and T3.
# ---------------------------------------------------------------------------
LEG_NAMES = list(s2.LEGS)                 # front_leg, middle_leg, hind_leg
SIDES = ("left", "right")
JOINTS = ("coxa_abduct", "femur", "tibia")
KNEE = "tibia"

# The muscle pool -> joint table is step 3's, and is not duplicated here.
POOL_ACTION = s3.POOL_ACTION


class Animal:
    """
    One body, six cords.

    The cords do not talk to each other directly — each is the <=3-hop
    subgraph around one leg's sense organs and motor neurons, which is what
    step 2 built. Anything the legs do in concert has to come out of the body:
    through the floor, through the thorax they are all hanging from, and
    through the tonic descending drive they share. That is a limitation and it
    is stated as one.
    """

    def __init__(self, args):
        import mujoco

        self.mj = mujoco
        mjcf = str(args.mjcf)
        self.model = mujoco.MjModel.from_xml_path(mjcf)
        self.data = mujoco.MjData(self.model)
        self.m = self.model
        self.d = self.data
        self.substeps = int(round(1e-3 / self.model.opt.timestep))
        self.weight = float(self.model.body_mass.sum()) * abs(
            self.model.opt.gravity[2])
        self.thorax = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        "thorax")

        # -- one step-3 Body per leg, used only to measure that leg ---------
        # (a Body owns its own MjData, so the measurements cannot disturb the
        #  one the animal is about to be simulated with)
        self.joint, self.rest, self.direction = {}, {}, {}
        self.load_bodies, self.claw = {}, {}
        for leg in LEG_NAMES:
            for side in SIDES:
                b = s3.Body(args.mjcf, leg, side)
                key = (leg, side)
                self.joint[key] = b.joint
                self.rest[key] = b.rest
                self.direction[key] = b.direction
                self.load_bodies[key] = b.load_bodies
                self.claw[key] = b.claw
                del b
        self.foot_bodies = set()
        for i in range(self.model.nbody):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if nm.startswith(("claw_", "tarsus4_", "tarsus3_")):
                self.foot_bodies.add(i)
        self.reset()

        self.half_span = {(leg, side, j):
                          max(1e-6, (self.joint[(leg, side)][j]["range"][1]
                                     - self.joint[(leg, side)][j]["range"][0]) / 2.0)
                          for leg in LEG_NAMES for side in SIDES for j in JOINTS}
        self.jspan = {(leg, side, j):
                      (self.joint[(leg, side)][j]["range"][1]
                       - self.joint[(leg, side)][j]["range"][0])
                      for leg in LEG_NAMES for side in SIDES for j in JOINTS}

    # -- the world ---------------------------------------------------------
    def reset(self):
        self.mj.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def leg_load(self, leg, side):
        """Normal force the floor pushes back with, under one leg only."""
        d, m, bodies = self.data, self.model, self.load_bodies[(leg, side)]
        total = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in bodies and b2 == 0) or (b2 in bodies and b1 == 0):
                if c.efc_address >= 0 and c.dim >= 1:
                    total += abs(float(d.efc_force[c.efc_address]))
        return total

    def foot_contacts(self):
        d, m = self.data, self.model
        return sum(1 for i in range(d.ncon)
                   if m.geom_bodyid[d.contact[i].geom1] in self.foot_bodies
                   or m.geom_bodyid[d.contact[i].geom2] in self.foot_bodies)

    def com(self):
        return np.array([float(self.data.subtree_com[1][0]),
                         float(self.data.subtree_com[1][1]),
                         float(self.data.subtree_com[1][2])])


class SixLegLoop:
    """
    The loop of step 3, six times over: body -> organ -> cord -> muscle ->
    joint, one millisecond at a time.
    """

    def __init__(self, animal, nets, args, seed):
        self.A = animal
        self.args = args
        self.nets = nets
        self.alpha = 1.0 / args.tau_mus
        self.cord, self.mask, self.n_pool, self.act = {}, {}, {}, {}
        self.desc, self.organ, self.pools, self.driven = {}, {}, {}, {}
        for key, net in nets.items():
            self.cord[key] = s2.CordNetwork(net.N, net.e_pre, net.e_post,
                                            net.w_sign, net.w_mag, net.delay,
                                            rng_seed=seed)
            self.desc[key] = net.desc_idx
            self.organ[key] = {o: net.watch.get(f"organ:{o}")
                               for o in ("chordotonal", "campaniform")}
            placed, driven, unplaced = s3.build_mapper(net, _Shim(animal, key))
            self.pools[key] = [p for p in POOL_ACTION
                               if net.watch.get(f"pool:{p}") is not None
                               and len(net.watch[f"pool:{p}"])]
            self.driven[key] = driven
            self.mask[key] = {p: np.zeros(net.N, bool) for p in self.pools[key]}
            for p in self.pools[key]:
                self.mask[key][p][net.watch[f"pool:{p}"]] = True
            self.n_pool[key] = {p: int(len(net.watch[f"pool:{p}"]))
                                for p in self.pools[key]}
            self.act[key] = {p: 0.0 for p in self.pools[key]}
        self.place_report = {f"{k[0]}_{k[1]}": {"driven": self.driven[k]}
                             for k in nets}

    def drives(self, key, force_ref):
        """What this leg's two sense organs are told, this millisecond."""
        leg, side = key
        a = self.args
        knee = self.A.joint[key][KNEE]
        q = float(self.A.d.qpos[knee["qposadr"]])
        x = (q - self.A.rest[key][KNEE]) / self.A.half_span[key + (KNEE,)]
        x = float(np.clip(x, -1.0, 1.0))
        chordo = a.tone * (1.0 - a.polarity * a.kappa * x)
        camp = a.tone * (self.A.leg_load(leg, side) / max(1e-9, force_ref))
        return max(0.0, chordo), camp

    def command(self, key, joint, b_ref):
        """ctrl for one joint, from the balance of its two antagonist pools."""
        up = sum(self.act[key][p] for p in self.driven[key][joint]["+1"])
        down = sum(self.act[key][p] for p in self.driven[key][joint]["-1"])
        total = up + down
        b = (up / total) if total > 1e-9 else 0.0
        j = self.A.joint[key][joint]
        lo, hi = j["range"]
        ctrl = self.A.rest[key][joint] + (hi - lo) * (b - b_ref[key + (joint,)])
        return float(np.clip(ctrl, lo, hi)), b

    def step(self, ms_total, b_ref, force_ref, nudge=None, on_frame=None,
             hold=False):
        """
        Advance `ms_total` milliseconds.

        `nudge`    callable(t) -> a 6-vector of external force on the thorax.
                   The world pushing the animal. Nothing in the controller
                   ever calls this; it is the equivalent of a puff of air.
        `hold`     command every joint to the pose of the scan and let the
                   muscles converge there. Every run starts this way, because
                   at t = 0 no pool has fired yet and the balance — and so the
                   command — is meaningless until they have.
        `on_frame` callable(t) called once per millisecond, so the caller can
                   decide what to record without this class knowing about
                   files, formats or frame rates.
        """
        A, d, m, mj = self.A, self.A.d, self.A.m, self.A.mj
        rows = []
        for t in range(ms_total):
            for key in self.cord:
                chordo, camp = self.drives(key, force_ref[key])
                drivers = [(self.desc[key], self.args.desc, 0, ms_total)]
                if self.organ[key]["chordotonal"] is not None:
                    drivers.append((self.organ[key]["chordotonal"], chordo, 0, ms_total))
                if self.organ[key]["campaniform"] is not None:
                    drivers.append((self.organ[key]["campaniform"], camp, 0, ms_total))
                fired = self.cord[key].step(drivers)
                fired = np.asarray(fired, dtype=np.int64)
                for p in self.pools[key]:
                    rate = float(self.mask[key][p][fired].sum()) \
                        / self.n_pool[key][p] * 1000.0
                    self.act[key][p] += (rate - self.act[key][p]) * self.alpha
                for joint in self.driven[key]:
                    if hold:
                        ctrl = self.A.rest[key][joint]
                    else:
                        ctrl, _b = self.command(key, joint, b_ref)
                    act = self.A.joint[key][joint]["actuator"]
                    if act is not None:
                        d.ctrl[act] = ctrl

            d.xfrc_applied[:] = 0.0
            if nudge is not None:
                f = nudge(t)
                if f is not None:
                    d.xfrc_applied[A.thorax][:3] += f[:3]
                    d.xfrc_applied[A.thorax][3:] += f[3:]

            for _ in range(A.substeps):
                mj.mj_step(m, d)

            if on_frame is not None:
                on_frame(t)
        return rows


class _Shim:
    """The three attributes of step 3's Body that `build_mapper` reads."""

    def __init__(self, animal, key):
        self.joint = animal.joint[key]
        self.direction = animal.direction[key]


# ---------------------------------------------------------------------------
# Finding the stance: the fixed point of the whole animal
# ---------------------------------------------------------------------------

def reference_run(animal, nets, args, seed, passes=4, record=None):
    """
    The pose at which the cord's commands and the body's answer agree.

    Step 3 did this for one leg. Six legs have to be solved together, because
    they are all hanging off the same thorax: a leg that takes more weight
    changes the load every other leg reports. So the whole animal is relaxed
    at once — measure every joint's antagonist balance, move every command
    towards what that balance implies, run again — until it stops moving.
    """
    b_ref = {k + (j,): 0.5 for k in nets for j in JOINTS}
    force = {k: 1.0 for k in nets}
    trail = []
    for p in range(passes):
        loop = SixLegLoop(animal, nets, args, seed)
        # the muscles converge onto the standing pose, joints held there
        loop.step(args.settle_ms, b_ref, force, hold=True)
        for _ in range(3):
            new_force = {}
            for k in nets:
                new_force[k] = max(1e-6, animal.leg_load(*k))
            relax = 0.5 if p < passes - 1 else 1.0
            for k in nets:
                force[k] += (new_force[k] - force[k]) * relax
            new_b = {}
            for k in nets:
                for j in JOINTS:
                    up = sum(loop.act[k][pp] for pp in loop.driven[k][j]["+1"])
                    dn = sum(loop.act[k][pp] for pp in loop.driven[k][j]["-1"])
                    new_b[k + (j,)] = (up / (up + dn)) if (up + dn) > 1e-9 else 0.5
            for key in new_b:
                b_ref[key] += (new_b[key] - b_ref[key]) * relax
            loop.step(args.settle_ms, b_ref, force)
        moved = max(abs(new_b[k] - b_ref[k]) for k in new_b) if new_b else 0.0
        trail.append({"pass": p, "max_balance_move": moved,
                      "com_z": float(animal.com()[2]),
                      "contacts": animal.foot_contacts()})
        log(f"  stance pass {p+1}/{passes}: largest balance move {moved:.4f}, "
            f"COM z {animal.com()[2]:+.5f}, {animal.foot_contacts()} feet down")
        if moved < 0.002 and p >= 1:
            break
    return b_ref, force, trail


# ---------------------------------------------------------------------------
# Recording the animal
# ---------------------------------------------------------------------------

def visible_geoms(model):
    """The parts you can see: the meshes, and the floor. Not the collision
    capsules — those are physics, not appearance."""
    out = []
    for i in range(model.ngeom):
        if model.geom_rgba[i][3] <= 0.01:
            continue
        if int(model.geom_type[i]) == 7 or int(model.geom_bodyid[i]) == 0:
            out.append(i)
    return out


def weld(verts, faces, grid):
    """
    The body model's meshes store every triangle's three corners separately,
    so 818,000 vertices describe about 132,000 distinct points. Snapping them
    together is not simplification — it is the same surface — and it is what
    makes the animal small enough to load in a browser.
    """
    q = np.floor(np.asarray(verts) / grid).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    f = inv[np.asarray(faces)]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    if not len(f):
        return np.zeros((0, 3), "float32"), np.zeros((0, 3), "int32")
    used, remap = np.unique(f, return_inverse=True)
    v = np.asarray(verts)[used].astype(np.float32)
    # smooth normals, averaged over the faces that now share each point
    n = np.zeros_like(v, dtype=np.float64)
    tri = v[remap].reshape(-1, 3, 3).astype(np.float64)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, remap.reshape(-1, 3)[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 0, n / np.maximum(ln, 1e-12), np.array([0.0, 0.0, 1.0]))
    return v, remap.reshape(-1, 3).astype(np.int32), n.astype(np.float32)


def write_body(model, path: pathlib.Path, grid=1e-4):
    """The animal's appearance, packed once: vertices, normals, faces."""
    import mujoco
    va, fa = model.mesh_vertadr, model.mesh_faceadr
    nvert_total = model.mesh_vert.shape[0]
    nface_total = model.mesh_face.shape[0]
    meshes, verts, norms, faces = [], [], [], []
    off_v = off_f = 0
    for i in range(model.nmesh):
        v0 = va[i]
        v1 = va[i + 1] if i + 1 < model.nmesh else nvert_total
        f0 = fa[i]
        f1 = fa[i + 1] if i + 1 < model.nmesh else nface_total
        v, f, n = weld(model.mesh_vert[v0:v1], model.mesh_face[f0:f1], grid)
        meshes.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i),
            "vert": [off_v, len(v)], "face": [off_f, len(f)],
        })
        verts.append(v)
        norms.append(n)
        faces.append(f + off_v)
        off_v += len(v)
        off_f += len(f)
    V = np.concatenate(verts).astype("<f4")
    N = np.concatenate(norms).astype("<f4")
    F = np.concatenate(faces).astype("<i4")
    with open(path, "wb") as fh:
        fh.write(V.tobytes())
        fh.write(N.tobytes())
        fh.write(F.tobytes())
    log(f"  body geometry: {len(V):,} points, {len(F):,} triangles, "
        f"{path.stat().st_size/1e6:.1f} MB "
        f"(model had {nvert_total:,} raw vertices)")
    return {"meshes": meshes, "n_vertex": int(len(V)), "n_face": int(len(F)),
            "bytes": int(path.stat().st_size)}


def geom_colour(model, i):
    """
    What a part looks like.

    The colour is on the geom's *material*, not on the geom: flybody names
    them — body, red, ocelli, black, bristle-brown, brown, membrane — and
    every mesh geom's own rgba is left at the default grey. Reading the
    geom's rgba gives an animal the colour of wet concrete.
    """
    mi = int(model.geom_matid[i]) if model.nmat else -1
    src = model.mat_rgba[mi] if 0 <= mi < model.nmat else model.geom_rgba[i]
    return [float(x) for x in src]


def geom_manifest(model, keep):
    import mujoco
    out = []
    for i in keep:
        t = int(model.geom_type[i])
        row = {
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}",
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[i])) or "world",
            "type": t,
            "rgba": geom_colour(model, i),
            "pos": [float(x) for x in model.geom_pos[i]],
            "quat": [float(x) for x in model.geom_quat[i]],
            "size": [float(x) for x in model.geom_size[i]],
        }
        if t == 7:
            row["mesh"] = int(model.geom_dataid[i])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# What the animal did
# ---------------------------------------------------------------------------

def behaviour(rows, hz, animal):
    com = np.array([r["com"] for r in rows])
    t = np.array([r["t"] for r in rows], dtype=float) / 1000.0
    xy = com[:, :2]
    # Two different questions, because a standing animal trembles: the
    # straight-line distance between where it started and where it ended, and
    # the length of the path it walked. The second is measured on 100 ms
    # windows, because summing single-frame steps counts the tremor as travel.
    net_disp = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0
    w = max(1, int(0.1 * hz))
    coarse = xy[::w]
    step = np.linalg.norm(np.diff(coarse, axis=0), axis=1)
    dist = float(step.sum())
    # one speed per frame, not one per window: how far it moved over the last
    # `w` frames. The HUD indexes this with the frame number.
    sp = np.zeros(len(xy))
    for i in range(w, len(xy)):
        sp[i] = np.linalg.norm(xy[i] - xy[i - w]) / (w / hz)
    speed = sp
    contacts = np.array([r["contacts"] for r in rows], dtype=float)
    legs = sorted(rows[0]["loads"].keys())
    duty = {}
    for L in legs:
        f = np.array([r["loads"][L] for r in rows])
        on = f > 0.02 * max(1e-9, f.max())
        duty[L] = float(on.mean())
    out = {
        "seconds": float(t[-1] - t[0]) if len(t) else 0.0,
        "net_displacement": net_disp,
        "path_length": dist,
        "body_lengths_per_second": dist / 0.27 / max(1e-9, (t[-1] - t[0])),
        "net_body_lengths_per_second": net_disp / 0.27 / max(1e-9, (t[-1] - t[0])),
        "mean_speed": float(speed.mean()) if len(speed) else 0.0,
        "max_speed": float(speed.max()) if len(speed) else 0.0,
        "com_height_first": float(com[0][2]), "com_height_last": float(com[-1][2]),
        "com_height_min": float(com[:, 2].min()), "com_height_max": float(com[:, 2].max()),
        "foot_contacts_mean": float(contacts.mean()),
        "foot_contacts_min": float(contacts.min()),
        "leg_contact_duty": duty,
        "fell_over": bool(com[:, 2].min() < com[0][2] - 0.05),
        "units": "model units; 1 unit = 1 cm, the animal is ~0.27 units long",
    }
    return out, {"t": t.tolist(), "xy": xy.tolist(), "z": com[:, 2].tolist(),
                 "contacts": contacts.tolist(),
                 "speed": (speed.tolist() if len(speed) else [])}


# ---------------------------------------------------------------------------

def rss_mb():
    """Resident memory, so a six-cord run can say what it is using."""
    try:
        return int(pathlib.Path("/proc/self/statm").read_text().split()[1]) * 4096 / 1e6
    except Exception:
        return -1.0


def build_cords(args, legs):
    """
    One cord per leg.

    The connectome is read once — it is the same connectome whichever leg you
    ask about — and each leg's anatomy is dropped as soon as its network
    exists, because six of them at once is what a single leg never had to be.
    """
    import gc
    raw = s2.load(args.banc, args.min_syn)
    nets = {}
    for leg in legs:
        for side in SIDES:
            la = argparse.Namespace(**vars(args))
            la.leg, la.side = leg, side
            anat = s2.leg_anatomy(la, loaded=raw)
            nets[(leg, side)] = s2.build_network(la, anat)
            del anat
            gc.collect()
            log(f"  cord for the {side} {leg.replace('_', ' ')}: "
                f"{nets[(leg, side)].N:,} neurons, "
                f"{nets[(leg, side)].subgraph_edges:,} edges "
                f"[{rss_mb():.0f} MB resident]")
    del raw
    gc.collect()
    return nets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--banc", default="data/banc", type=pathlib.Path)
    ap.add_argument("--mjcf", default="data/flybody/flybody-main/flybody/"
                                      "fruitfly/assets/floor.xml", type=pathlib.Path)
    ap.add_argument("--out", default="world", type=pathlib.Path)
    ap.add_argument("--json", default="reports/step4_world.json", type=pathlib.Path)
    ap.add_argument("--md", default="reports/step4_world.md", type=pathlib.Path)
    ap.add_argument("--legs", default=",".join(LEG_NAMES))
    ap.add_argument("--ms", type=int, default=8000,
                    help="milliseconds of animal to record")
    ap.add_argument("--settle-ms", type=int, default=400)
    ap.add_argument("--pre-ms", type=int, default=500,
                    help="closed-loop milliseconds run before the camera "
                         "starts, so the recording is not the transient of "
                         "switching the loop on")
    ap.add_argument("--hz", type=int, default=100,
                    help="frames per second of animal, recorded")
    ap.add_argument("--min-syn", type=int, default=3)
    ap.add_argument("--clusters", type=int, default=3,
                    help="step 2's chordotonal subtypes; step 4 builds "
                         "them but does not report on them")
    ap.add_argument("--n-star", type=float, default=12.0)
    ap.add_argument("--desc", type=float, default=2.5)
    ap.add_argument("--tone", type=float, default=2.5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--polarity", type=float, default=+1.0)
    ap.add_argument("--tau-mus", type=float, default=60.0)
    ap.add_argument("--nudge", default="3000,6000",
                    help="milliseconds at which the world pushes the animal")
    ap.add_argument("--nudge-force", type=float, default=0.35,
                    help="the push, as a multiple of the animal's weight")
    ap.add_argument("--nudge-ms", type=int, default=100,
                    help="how long the push lasts")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.channels = "chordotonal,campaniform"

    legs = [x for x in args.legs.split(",") if x]
    args.out.mkdir(parents=True, exist_ok=True)

    log("the body …")
    animal = Animal(args)
    animal.reset()
    log(f"  {animal.model.nbody-1} parts, {animal.model.nu} actuators, "
        f"{animal.substeps} physics steps per millisecond of cord, "
        f"weight {animal.weight:.4f}")

    log(f"the cords, one per leg ({len(legs)*2} of them) …")
    nets = build_cords(args, legs)

    log("finding the stance: the pose the cord commands and the body agrees on …")
    b_ref, force, trail = reference_run(animal, nets, args, args.seed)
    log(f"  stance found. COM z {animal.com()[2]:+.5f}, "
        f"{animal.foot_contacts()} feet down")
    for k in sorted(nets):
        log(f"  {k[0][:4]} {k[1]:5s} carries {force[k]:.4f} "
            f"= {100*force[k]/animal.weight:4.1f}% of body weight, "
            f"balance " + ", ".join(f"{j} {b_ref[k+(j,)]:.3f}" for j in JOINTS))

    # ---- the animal, in the world ----------------------------------------
    nudges = [int(x) for x in args.nudge.split(",") if x.strip()]
    log(f"recording {args.ms} ms of animal at {args.hz} Hz "
        f"(after {args.pre_ms} ms of closed loop so the recording is not the "
        f"transient of switching the loop on); "
        f"the world pushes it at {[n + args.pre_ms for n in nudges]} ms …")
    loop = SixLegLoop(animal, nets, args, args.seed)
    if args.pre_ms:
        loop.step(args.pre_ms, b_ref, force)
        log(f"  settled: COM z {animal.com()[2]:+.5f}, "
            f"x {animal.com()[0]:+.4f}, {animal.foot_contacts()} feet down")
    keep = visible_geoms(animal.model)
    stride = max(1, int(round(1000.0 / args.hz)))
    n_frames = args.ms // stride

    frames = np.zeros((n_frames, len(keep), 7), dtype="<f4")
    rows = []
    push = args.nudge_force * animal.weight

    def nudge(t):
        for ms in nudges:
            if ms <= t < ms + args.nudge_ms:
                return np.array([0.0, -push, 0.0, 0.0, 0.0, 0.0])
        return None

    f = 0

    def on_frame(t):
        nonlocal f
        if t % stride or f >= n_frames:
            return
        gx = animal.d.geom_xpos[keep]
        gm = animal.d.geom_xmat[keep].reshape(-1, 3, 3)
        frames[f, :, 0:3] = gx
        frames[f, :, 3:7] = mat_to_quat(gm)
        rows.append(dict(
            t=t, com=animal.com().tolist(), contacts=animal.foot_contacts(),
            loads={f"{k[0][:4]}{k[1][0]}": animal.leg_load(*k) for k in nets},
            knee={f"{k[0][:4]}{k[1][0]}":
                  float(animal.d.qpos[animal.joint[k][KNEE]["qposadr"]])
                  for k in nets}))
        f += 1
        if t % 1000 == 0:
            log(f"  {t/1000:.1f}s / {args.ms/1000:.1f}s  COM z "
                f"{animal.com()[2]:+.5f}  {animal.foot_contacts()} feet down  "
                f"x {animal.com()[0]:+.4f}")

    loop.step(args.ms, b_ref, force, nudge=nudge, on_frame=on_frame)
    frames = frames[:f]

    # the per-frame measurements are small; write them before the large files,
    # so nothing that goes wrong later can cost the numbers
    (args.out / "behaviour.json").write_text(json.dumps(rows))
    log("packing the animal and the recording …")
    body = write_body(animal.model, args.out / "fly.bin")
    (args.out / "frames.bin").write_bytes(frames.astype("<f4").tobytes())
    log(f"  frames: {frames.shape[0]} x {frames.shape[1]} parts, "
        f"{(args.out/'frames.bin').stat().st_size/1e6:.1f} MB")

    beh, series = behaviour(rows, args.hz, animal)
    log(f"  it ended {beh['net_displacement']:.4f} units from where it started "
        f"({beh['net_displacement']/0.27:.2f} body lengths) and walked a path "
        f"of {beh['path_length']:.4f} units ({beh['body_lengths_per_second']:.2f} "
        f"body lengths per second), {beh['foot_contacts_mean']:.1f} of 6 feet "
        f"down on average")

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "legs": [f"{l}_{s}" for l in legs for s in SIDES],
        "model": {
            "cord": "six connectome-derived cords, one per leg, LIF, 1 ms",
            "body": "flybody in MuJoCo, 100 µs steps, 18 joints under cord control",
            "tau_muscle_ms": args.tau_mus, "desc_tone": args.desc,
            "organ_tone": args.tone, "kappa": args.kappa,
            "polarity": args.polarity, "n_star": args.n_star,
            "recorded_ms": args.ms, "pre_roll_ms": args.pre_ms,
            "frame_hz": args.hz,
            "nudges_ms": [n + args.pre_ms for n in nudges],
            "nudge_force": args.nudge_force, "nudge_ms": args.nudge_ms,
            "seed": args.seed,
        },
        "weight": animal.weight,
        "floor_z": float(animal.model.geom_pos[0][2]),
        "body_geometry": body,
        "geoms": geom_manifest(animal.model, keep),
        "frames": {"n": int(frames.shape[0]), "parts": int(frames.shape[1]),
                   "stride_ms": stride},
        "stance": {"balance": {f"{k[0]}_{k[1]}|{j}": b_ref[k + (j,)]
                               for k in nets for j in JOINTS},
                   "leg_force": {f"{k[0]}_{k[1]}": force[k] for k in nets},
                   "passes": trail},
        "behaviour": beh,
        "series": series,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(manifest, indent=1))
    (args.out / "world.json").write_text(json.dumps(manifest, indent=1))
    log(f"wrote {args.json} and {args.out/'world.json'}")
    write_markdown(manifest, args.md)
    log(f"wrote {args.md}")
    return 0


def mat_to_quat(mats):
    """
    MuJoCo's 3x3 rotation matrices -> quaternions, (w, x, y, z).

    scipy's Rotation is the one dependency that will do this in bulk; doing it
    by hand is where sign and branch errors live, and a quaternion that is
    wrong by a sign makes the animal's legs rotate the wrong way on screen.
    """
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(mats, dtype=np.float64)
                             .reshape(-1, 3, 3)).as_quat()   # x, y, z, w
    out = np.empty_like(q)
    out[:, 0], out[:, 1:] = q[:, 3], q[:, :3]                # -> w, x, y, z
    return out.astype("<f4")


def write_markdown(o: dict, path: pathlib.Path) -> None:
    L = []
    A = L.append
    b = o["behaviour"]
    A("# Step 4 — the animal, in a world you can look at\n")
    A("Generated by `tools/step4_world.py`. Every number is in "
      "`reports/step4_world.json`.\\n")
    A("## What it is\n")
    A("All six legs, each driven by its own copy of the ventral nerve cord — "
      "the ≤3-hop subgraph step 2 took out of the BANC connectome, around one "
      "leg's sense organs and motor neurons. Every millisecond each leg's "
      "organs are told where the joint is and how much load the leg is "
      "carrying; the cord runs; the pools' spike rates are filtered at a "
      "muscle time constant; the antagonists' balance commands the joint. "
      "There is no clock, no gait, no target, and nothing that knows where "
      "the animal is meant to go.\\n")
    A(f"**{len(o['legs'])} legs, {len(o['geoms'])} visible parts, "
      f"{o['frames']['n']} frames at {o['model']['frame_hz']} Hz.**\\n")
    A("## What it did\n")
    A("| | |")
    A("|---|---|")
    A(f"| seconds of animal | {b['seconds']:.2f} |")
    A(f"| net displacement, start to end | {b['net_displacement']:.4f} units "
      f"({b['net_displacement']/0.27:.2f} body lengths) |")
    A(f"| length of the path it walked | {b['path_length']:.4f} units |")
    A(f"| speed along that path | {b['body_lengths_per_second']:.2f} body "
      f"lengths per second |")
    A(f"| mean / max speed | {b['mean_speed']:.4f} / {b['max_speed']:.4f} |")
    A(f"| feet on the floor | {b['foot_contacts_mean']:.2f} of 6 "
      f"(min {b['foot_contacts_min']:.0f}) |")
    A(f"| COM height, first → last | {b['com_height_first']:+.4f} → "
      f"{b['com_height_last']:+.4f} |")
    A(f"| it fell over | {'yes' if b['fell_over'] else 'no'} |")
    A("")
    A("Per-leg contact duty cycle — the fraction of the recording each leg is "
      "carrying load:\\n")
    A("| leg | duty |")
    A("|---|---:|")
    for k, v in sorted(b["leg_contact_duty"].items()):
        A(f"| {k} | {v:.2f} |")
    A("")
    A("## The stance it found\n")
    A("The pose the cord commands and the body agrees on, solved for the "
      "whole animal at once, because six legs hang off one thorax and a leg "
      "that takes more weight changes what every other leg reports:\\n")
    A("| leg | coxa_abduct | femur | tibia | share of body weight |")
    A("|---|---:|---:|---:|---:|")
    st = o["stance"]
    for leg in o["legs"]:
        A(f"| {leg} | " + " | ".join(
            f"{st['balance'][f'{leg}|{j}']:.3f}" for j in JOINTS) +
          f" | {100*st['leg_force'][leg]/o['weight']:.1f}% |")
    A("")
    A("## How to watch it\n")
    A("```\npython3 tools/serve_world.py\n```\n")
    A("## What is assumed\n")
    A("The six cords are built independently and do not share interneurons: "
      "each is the subgraph around one leg. Anything the legs do together has "
      "to come out of the body — through the floor, through the thorax they "
      "all hang from, and through the tonic descending drive they share. That "
      "is a limitation, and it is the first thing step 5 owes.\\n")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
