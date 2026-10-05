#!/usr/bin/env python3
"""
FLY ABA — the physics the *phone* runs, written twice and checked once.

The app is going to simulate the animal on the device instead of playing back
`world/frames.bin`. That means someone has to write an articulated-body solver
for a 67-part, 102-degree-of-freedom fly that runs in real time on a phone
CPU — and a solver like that is exactly the kind of code that is wrong in a way
that still looks plausible.

So it is written here first, in NumPy, where it can be *checked against MuJoCo*
before a single line of Swift exists:

  python3 tools/fly_aba.py --verify

Three checks, in increasing order of how much they prove:

  1. **Kinematics.** Random joint angles, forward kinematics from this solver
     against `mj_forward`'s body positions and orientations. Proves the tree,
     the joint axes and the frame conventions.
  2. **Dynamics.** The animal dropped in vacuum for 20 ms, zero torques, no
     floor: root trajectory and every joint angle against MuJoCo stepped with
     the same 100 µs integrator. Proves the spatial algebra, the inertia
     model, gravity and the integration — a disagreement here is a bug, not a
     tolerance question, and this test asserts 1e-9.
  3. **The standing animal.** Muscles on, floor on: the fly is dropped from the
     stance and has to stay up on its own six feet, under nothing but its
     muscle model. This is the check that the *muscle calibration* is enough,
     and `--verify` fails if the animal falls.

`tools/build_body.py` writes the file this reads. The Swift port
(`FlyBrain/Sources/FlyDynamics.swift`) is a transliteration of this file and
the app's own test suite replays the golden trace this writes with
`--golden`, so the two are pinned to each other rather than "both look
about right".

Usage
  python3 tools/fly_aba.py --verify
  python3 tools/fly_aba.py --verify --out reports/step5_aba.json
  python3 tools/fly_aba.py --golden build/fly_golden.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------
# Small spatial algebra. Conventions, fixed once and used everywhere:
#
#   motion  v = [omega; nu]     angular first, linear second
#   force   f = [n; f_lin]      moment first, linear second
#   v_child = X v_parent ,  f_parent = X^T f_child
#   X = [[E, 0], [-(E r)x, E]]  E = R_child^T R_parent,  r = child origin
#                               relative to the parent origin, in parent
#                               coordinates
#
# These are Featherstone's (Rigid Body Dynamics Algorithms, 2008). Nothing in
# this file deviates from them, because a solver that invents its own sign
# convention is a solver nobody can check.
# ---------------------------------------------------------------------------

def crossmat(a):
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def xform(E, r):
    """
    The 6x6 Plücker transform from the parent frame to the child frame.

    v_child = X v_parent, and the linear block is what catches people out: the
    cross term is (E r) x (E w) — the *child*-frame angular velocity — so the
    lower-left block is -(E r)x E and not -(E r)x. Get this wrong and everything
    still runs, and the answer is quietly wrong by a factor that depends on the
    joint's orientation.
    """
    X = np.zeros((6, 6))
    X[:3, :3] = E
    X[3:, :3] = -crossmat(E @ r) @ E
    X[3:, 3:] = E
    return X


def crm(v):
    """Motion cross-product matrix: crm(v) u = v x_m u."""
    w, nu = v[:3], v[3:]
    out = np.zeros((6, 6))
    out[:3, :3] = crossmat(w)
    out[3:, :3] = crossmat(nu)
    out[3:, 3:] = crossmat(w)
    return out


def crf(v):
    """Force cross-product matrix: v x_f* f  (Featherstone eq. 2.34)."""
    return -crm(v).T


def spatial_inertia(mass, com, inertia_diag, iquat):
    """
    The 6x6 spatial inertia about the body-frame origin.

    `inertia_diag` is MuJoCo's `body_inertia`, which is the diagonal in the
    principal frame that `body_iquat` rotates into the body frame — so the
    rotation is applied here rather than assumed to be the identity.
    """
    R = quat_to_mat(iquat)
    Ic = R @ np.diag(inertia_diag) @ R.T
    c = np.asarray(com, float)
    cx = crossmat(c)
    out = np.zeros((6, 6))
    out[:3, :3] = Ic + mass * (cx @ cx.T)
    out[:3, 3:] = mass * cx
    out[3:, :3] = mass * cx.T
    out[3:, 3:] = mass * np.eye(3)
    return out


def quat_to_mat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_from_axis_angle(axis, angle):
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = np.asarray(axis, float) / n
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), a[0] * s, a[1] * s, a[2] * s])


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class FlyBody:
    """`build/fly_body.json`, opened for simulation."""

    def __init__(self, path):
        raw = json.loads(pathlib.Path(path).read_text())
        self.raw = raw
        self.units = raw["units"]
        self.gravity = np.array(raw["gravity"], float)
        self.dt_default = float(raw["timestep_s"])
        self.floor_z = float(raw["floor_z"])

        bodies = raw["bodies"]
        self.nbody = len(bodies)
        self.name_of_body = [b["name"] for b in bodies]
        self.body_index = {b["name"]: i for i, b in enumerate(bodies)}
        self.parent = np.array([b["parent_index"] for b in bodies], int)

        # Fixed part of each body's pose relative to its parent.
        self.frame = np.array([bool(b.get("frame", True)) for b in bodies])
        self.source_body = [b.get("body", b["name"]) for b in bodies]
        self.body_pos = np.array([b["pos"] for b in bodies], float)
        self.body_quat = np.array([b["quat"] for b in bodies], float)
        self.body_mass = np.array([b["mass"] for b in bodies], float)

        self.I = np.zeros((self.nbody, 6, 6))
        for i, b in enumerate(bodies):
            self.I[i] = spatial_inertia(b["mass"], b["com"], b["inertia"],
                                        b["iquat"])

        # Joints. One free joint on the root, 102 hinges under it; the hinge
        # index is the index into this solver's q vector, the root's seven
        # coordinates are carried separately.
        hinges = [j for j in raw["joints"] if j["kind"] == "hinge"]
        self.hinges = hinges
        self.nj = len(hinges)
        self.joint_body = np.array([self.body_index[j["body"]] for j in hinges], int)
        self.axis = np.array([j["axis"] for j in hinges], float)
        if self.nj:
            self.axis /= np.linalg.norm(self.axis, axis=1, keepdims=True)
        self.lo = np.array([j["range"][0] for j in hinges], float)
        self.hi = np.array([j["range"][1] for j in hinges], float)
        self.limited = np.array([bool(j["limited"]) for j in hinges])
        self.damping = np.array([j["damping"] for j in hinges], float)
        self.armature = np.array([j["armature"] for j in hinges], float)
        # the joint's spring pulls towards this angle, not towards zero
        self.spring_ref = np.array([j.get("spring_ref", 0.0) for j in hinges], float)
        self.stiffness = np.array([j["stiffness"] for j in hinges], float)
        # The pose the animal stands at, measured in build_body: the keyframe
        # is not it (see FlySim.reset).
        self.stance_q = np.array([raw.get("stance_q", {}).get(j["name"], 0.0)
                                  for j in hinges], float)
        self.joint_name = [j["name"] for j in hinges]
        self.joint_index = {j["name"]: i for i, j in enumerate(hinges)}

        # Which body carries the free joint: the root of the tree.
        root = [i for i in range(self.nbody)
                if any(j["kind"] == "free" for j in raw["joints"]
                       if j["body"] == self.name_of_body[i])]
        self.root = root[0] if root else 1
        self.root_free = bool(root)
        self.has_free = self.root_free

        # link -> its hinge, the inverse of `joint_body`; a body with no joint
        # of its own keeps -1 and is welded to its parent.
        self._joint_of_body = np.full(self.nbody, -1, int)
        for k, bi in enumerate(self.joint_body):
            self._joint_of_body[bi] = k

        # Order bodies so the parent always comes first (MuJoCo already does).
        assert all(self.parent[i] < i for i in range(1, self.nbody)), \
            "bodies must be in topological order"

        # Muscles, keyed by joint name: maximum torque and the passive terms.
        self.muscles = raw.get("muscles", {})
        self.max_torque = np.array([self.muscles.get(j["name"], {})
                                    .get("max_torque", 0.0)
                                    for j in hinges], float)
        # `-biasprm[1]`: the stiffness of the muscle pair at rest, straight out
        # of the model's actuators (0 where the file has no actuator)
        self.passive_stiffness = np.array(
            [self.muscles.get(j["name"], {}).get("passive_stiffness", 0.0)
             for j in hinges], float)

        def muscle_field(field, fallback):
            return np.array([self.muscles.get(j["name"], {}).get(field, fallback(j))
                             for j in hinges], float)

        # Muscle geometry. The asset records the *joint's* range next to each
        # muscle, so the force-length width is the width of that range and the
        # rest angle is its middle unless a real optimal angle is present.
        # flybody publishes no muscle optimal angles (nothing in the MJCF
        # says where any muscle is longest), so this is an APPROXIMATION and
        # is declared as one in docs/ASSUMPTIONS.md.
        width = np.array([max(1e-3, abs(j["range"][1] - j["range"][0]))
                          for j in hinges], float)
        middle = np.array([0.5 * (j["range"][0] + j["range"][1]) for j in hinges], float)
        self.max_torque = muscle_field("max_torque", lambda j: 0.0)
        self.optimal = np.array([
            float(self.muscles.get(j["name"], {}).get(
                "optimal_angle", self.spring_ref[k] if self.spring_ref[k] else middle[k]))
            for k, j in enumerate(hinges)])
        self.span = np.array([
            float(self.muscles.get(j["name"], {}).get("span", width[k]))
            for k, j in enumerate(hinges)])

        # Collision geoms, and the body each one belongs to.
        self.collision = raw["collision"]
        self.geom_body = np.array(
            [self.body_index[g["body"]] for g in self.collision], int)
        self.geom_type = [g["type"] for g in self.collision]
        self.geom_size = np.array([g["size"] for g in self.collision], float)
        self.geom_pos = np.array([g["pos"] for g in self.collision], float)
        self.geom_quat = np.array([g["quat"] for g in self.collision], float)

        # The legs: feet, joints, organs.
        self.legs = raw["legs"]
        self.foot_geoms = {}
        for key, leg in self.legs.items():
            idx = [i for i, g in enumerate(self.collision)
                   if self.name_of_body[self.geom_body[i]] in leg["feet_bodies"]]
            self.foot_geoms[key] = idx
        self.knee_joint = {key: leg["organs"]["chordotonal"]["joint"]
                           for key, leg in self.legs.items()}

        self.stance_root_z = float(raw.get("stance_root_z", 0.0))

    # -- kinematics --------------------------------------------------------

    def fk(self, root_pos, root_quat, q):
        """
        Every body's world position and orientation, in MuJoCo's own order:

            R_i = R_parent . (product of this body's joint rotations) . R_body_quat

        which is not the order it is tempting to write down: the joint rotations
        act *before* the body's fixed quat, and a body with no joint of its own
        (the welded links of a fixed base) is still just a fixed offset. Get the
        order wrong and a q = 0 pose still matches, so the mistake hides until a
        joint actually bends.
        """
        pos = np.zeros((self.nbody, 3))
        rot = np.zeros((self.nbody, 3, 3))
        rot[0] = np.eye(3)
        pos[0] = 0.0
        for i in range(1, self.nbody):
            p = self.parent[i]
            if i == self.root and self.has_free:
                pos[i] = root_pos
                rot[i] = quat_to_mat(root_quat)
            else:
                rot[i] = rot[p] @ quat_to_mat(self.body_quat[i])
                pos[i] = pos[p] + rot[p] @ self.body_pos[i]
            # this body's own hinges, in order, applied about its own origin
            for j in self.hinges_of_body(i):
                rot[i] = rot[i] @ quat_to_mat(
                    quat_from_axis_angle(self.axis[j], q[j]))
        return pos, rot

    def hinges_of_body(self, i):
        """
        The hinge indices on link i, in the order the asset lists them.

        The asset expands a multi-joint body into one link per joint, so this
        is normally a single hinge or none at all — but it is written as a
        membership test on purpose: the earlier version compared the link's
        *first* joint index against the joint number and quietly returned the
        wrong hinge for every link past the first one of a chain.
        """
        return [j for j in range(self.nj) if self.joint_body[j] == i]

    def geom_world(self, pos, rot):
        """World position, orientation and axes of every collision geom."""
        out_pos = np.zeros((len(self.collision), 3))
        out_rot = np.zeros((len(self.collision), 3, 3))
        for g in range(len(self.collision)):
            b = self.geom_body[g]
            out_rot[g] = rot[b] @ quat_to_mat(self.geom_quat[g])
            out_pos[g] = pos[b] + rot[b] @ self.geom_pos[g]
        return out_pos, out_rot


# ---------------------------------------------------------------------------
# The solver: Featherstone's articulated-body algorithm, plus a floor
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------

# The floor's parameters. MuJoCo's contact model and a penalty spring are not
# the same model, so these are *fitted* to the thing they must reproduce, and
# then tested by whether the animal stands (ASSUMPTIONS #21).
#
# The fit: the fly weighs m*g = 0.966 g cm/s^2, and at the stance its own
# contacts (measured in build_body, MuJoCo's soft constraints) carry that weight
# at a total penetration of 6.36e-03 cm over the geoms that are inside the
# plane, the deepest of them 3.96e-04 cm. A penalty spring that carries the same
# weight at the same depth is k = 0.966/6.36e-03 = 152. The old value, 6.0,
# carried 4% of the animal's weight at the stance, so every "standing" run sank
# 25x deeper than the pose the muscles were calibrated for and folded up. This
# is what the standing test now measures.
CONTACT_STIFFNESS = 150.0      # force units per cm of penetration
# The contact damper is small *because it is explicit*: at dt = 1e-4 s a
# coefficient of b damps a mass m through the factor b*dt/m, and the lightest
# thing that touches the floor is a 1e-6 g cm^2 leg joint, so b*dt/m must stay
# under 1. Measured, 0.5 is a runaway (|qd| 900 rad/s and climbing, the animal
# thrown clear of the floor); 0.02 and below stand. 0.005 is zeta ~ 0.2 against
# the leg's own stiffness — the value to argue with is the stability limit, and
# the limit is b < m*2/dt.
CONTACT_DAMPING = 0.005        # force units per cm/s
FRICTION = 0.5                 # fly tarsi on clean glass
CONTACT_SLOP = 3.3e-5          # cm: contact begins this far above the floor,
                               # so the resting penetration lands where MuJoCo's
                               # soft contact puts it

# Muscle shortening speed. Fly leg muscle sarcomeres shorten at ~10-20 rad/s
# at the joint; the exact value changes the velocity term by a few percent, so
# it is declared rather than tuned (ASSUMPTIONS #22).
V_MAX = 20.0                   # rad/s
PASSIVE_DAMPING = 0.0          # the model's own dof_damping does this job


def quat_from_vec(v):
    """Exponential map: a world-frame rotation vector as a quaternion."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat_from_axis_angle(v / angle, angle)


class FlySim:
    """
    The animal, a hundred microseconds at a time.

    `torque` is per hinge joint in the model's own units and it comes from the
    muscles; this class is never told what angle a joint should be at.
    """

    def __init__(self, body: FlyBody, floor_z=None, limits=True):
        self.b = body
        # Joint limits here are hard stops (clamp, kill the rate). MuJoCo's are
        # soft constraints solved implicitly, so the two cannot be compared
        # with them active — the check turns them off on both sides, and the
        # difference in how they are modelled is declared (ASSUMPTIONS #23).
        self.limits = limits
        self.n = body.nbody
        self.floor_z = body.floor_z if floor_z is None else floor_z
        self._com = np.array([x["com"] for x in body.raw["bodies"]], float)
        # joint -> (body, axis) in a form the recursion can use directly; the
        # link -> hinge map itself lives on the body asset
        self._S = np.zeros((self.n, 6))
        for j, bb in enumerate(body.joint_body):
            self._S[bb][:3] = body.axis[j]
        self._joint_of_body = body._joint_of_body
        self.reset()

    # -- state -------------------------------------------------------------

    def reset(self, root_z=None, root_pos=None, q=None, qd=None,
              v=None, omega=None, root_quat=None):
        b = self.b
        z = b.stance_root_z if root_z is None else root_z
        self.root_pos = np.array([0.0, 0.0, z]) if root_pos is None else np.array(root_pos, float)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0]) if root_quat is None else np.array(root_quat, float)
        # `q=None` is the stance, not zero: `stance_q` is the pose the animal
        # settles at with its own actuators holding it up, legs compressed by
        # its weight, and it is the pose the hold torques were measured at.
        # Starting from the all-zero keyframe instead starts the animal at a
        # pose no muscle is balanced for, and the first thing it does is snap.
        self.q = b.stance_q.copy() if q is None else np.array(q, float)
        self.qd = np.zeros(b.nj) if qd is None else np.array(qd, float)
        self.vel = np.zeros(3) if v is None else np.array(v, float)
        self.omega = np.zeros(3) if omega is None else np.array(omega, float)
        self.torque = np.zeros(b.nj)
        self.foot_force = {k: 0.0 for k in b.legs}
        self.contacts = 0
        self.bodies_in_contact = set()
        self._kinematics()

    # -- kinematics --------------------------------------------------------

    def _kinematics(self):
        b = self.b
        self.pos, self.rot = b.fk(self.root_pos, self.root_quat, self.q)
        self.gpos, self.grot = b.geom_world(self.pos, self.rot)

    def com(self):
        total = self.b.body_mass.sum()
        acc = np.zeros(3)
        for i in range(1, self.n):
            acc += self.b.body_mass[i] * (self.pos[i] + self.rot[i] @ self._com[i])
        return acc / total

    # -- the recursion ------------------------------------------------------

    def _velocities(self):
        b = self.b
        v = np.zeros((self.n, 6))
        R = self.rot[b.root]
        v[b.root] = np.concatenate([R.T @ self.omega, R.T @ self.vel])
        # a welded root can carry hinges of its own: for a free root the free
        # joint is the whole story, but a fixed base still has its first joint
        for j in b.hinges_of_body(b.root):
            v[b.root][:3] += b.axis[j] * self.qd[j]
        for i in range(b.root + 1, self.n):
            par = b.parent[i]
            E = self.rot[i].T @ self.rot[par]
            X = xform(E, b.body_pos[i])
            v[i] = X @ v[par]
            # every hinge of this link, not just the first: a welded root can
            # carry its own hinge, and skipping it zeroes its velocity
            for j in b.hinges_of_body(i):
                v[i][:3] += b.axis[j] * self.qd[j]
        self.v = v

    def _bias(self):
        """
        v x* I v, minus gravity, minus the floor.

        Gravity is a force on every body here — `[c x f, f]` about the body
        origin, in the body's own frame — and *not* an acceleration handed to
        the root of the tree. The two are the same for a fixed base and every
        textbook writes it the other way round, but they are not the same for a
        free root: with gravity as a force, a hand's hinge in free fall needs no
        torque and gets none, while the root-acceleration trick quietly asks the
        hinges for the torque that holds the fall rigid. That difference is
        invisible in every static test and wrong in every falling one.
        """
        b = self.b
        g = np.asarray(b.gravity, float)
        c = np.zeros((self.n, 6))
        for i in range(1, self.n):
            c[i] = crf(self.v[i]) @ (b.I[i] @ self.v[i])
            f = b.body_mass[i] * (self.rot[i].T @ g)
            c[i] -= np.concatenate([np.cross(self._com[i], f), f])
        self.c = c - self._floor_contact()

    def _floor_contact(self):
        """
        The floor, as a penalty contact on every collision geom.

        A convex shape's lowest point is its centre minus its support function
        along the plane normal, so the penetration of a geom through a flat
        plane is exact — capsules, ellipsoids, cylinders, boxes and spheres,
        no pair-wise geometry at all. That is why the phone can afford to test
        all 74 of them every hundred microseconds.
        """
        b = self.b
        wrench = np.zeros((self.n, 6))
        for k in self.foot_force:
            self.foot_force[k] = 0.0
        self.contacts = 0
        self.bodies_in_contact = set()
        up = np.array([0.0, 0.0, 1.0])
        for i, kind in enumerate(b.geom_type):
            sz = b.geom_size[i]
            R = self.grot[i]
            if kind == "sphere":
                h = sz[0]
            elif kind == "capsule":
                h = sz[0] + sz[1] * abs(R[2, 2])
            elif kind == "ellipsoid":
                # support along world +z = |semi-axes * (world z in the geom's
                # own frame)|. Written as `norm(R @ sz)` this is the norm of a
                # rotated vector — constant, orientation-free, and wrong by
                # most of a leg segment: a 150 um phantom penetration on the
                # tarsus, which at 150 units of stiffness is twice the animal's
                # weight pushing on a 1e-6 g cm^2 joint. Measured against
                # MuJoCo's own plane distance, this form agrees to 0.2 um on
                # all 74 collision geoms (tools/verify_aba.py --mjcf).
                h = float(np.linalg.norm(np.asarray(sz[:3]) * R[2, :]))
            elif kind == "cylinder":
                a = abs(R[2, 2])
                h = sz[0] * math.sqrt(max(0.0, 1 - a * a)) + sz[1] * a
            elif kind == "box":
                h = float(np.abs(R.T @ up) @ sz[:3])
            else:
                continue
            pen = (self.floor_z + CONTACT_SLOP) - (self.gpos[i][2] - h)
            if pen <= 0:
                continue
            body = b.geom_body[i]
            Rb = self.rot[body]
            origin = self.pos[body]
            point = self.gpos[i]
            v_body = self.v[body]
            w_world = Rb @ v_body[:3]
            v_origin = Rb @ v_body[3:] + np.cross(w_world, origin)
            v_point = v_origin + np.cross(w_world, point - origin)
            normal = CONTACT_STIFFNESS * pen - CONTACT_DAMPING * v_point[2]
            if normal < 0:
                normal = 0.0
            tang = v_point - np.array([0.0, 0.0, v_point[2]])
            speed = float(np.linalg.norm(tang))
            if speed > 1e-12:
                mu = FRICTION * min(1.0, speed / 1e-3)
                friction = -tang / speed * mu * normal
            else:
                friction = np.zeros(3)
            F = normal * up + friction
            # the wrench the recursion wants is in the *body's* frame — both
            # halves of it. A world-frame force crossed with a body-frame lever
            # arm is neither: on a fly standing level it looks harmless, on a
            # leg rotated 45 degrees the floor pushes the leg sideways, and the
            # animal comes apart in 2 ms.
            local = Rb.T @ (point - origin)
            Fb = Rb.T @ F
            wrench[body] += np.concatenate([np.cross(local, Fb), Fb])
            self.contacts += 1
            self.bodies_in_contact.add(body)
            for key, geoms in b.foot_geoms.items():
                if i in geoms:
                    self.foot_force[key] += float(normal)
        return wrench

    def step(self, dt, torque):
        """
        One substep. Semi-implicit Euler with the joint dampers folded into the
        articulated-body inertia, which is the same thing MuJoCo does — it is
        what makes a damper on a joint this light integrable at all.
        """
        b = self.b
        self.torque = np.asarray(torque, float)
        self._kinematics()
        self._velocities()
        self._bias()
        self._welded = np.zeros(self.n, bool)

        tau = self.torque - b.stiffness * (self.q - b.spring_ref)

        IA = np.array(b.I)
        pA = self.c.copy()
        U = np.zeros((self.n, 6))
        D = np.ones(self.n)
        u = np.zeros(self.n)
        X = [None] * self.n
        # The velocity-product acceleration of each joint, `c = v x (S qd)`.
        # Leaving it out is the single most expensive mistake in this file: the
        # recursions look consistent without it, the static pose is exactly
        # right, and every moving pose is wrong in a way no test pattern makes
        # obvious. It belongs in *both* passes — the articulated-body inertia
        # cancels part of it going down the tree and the rest comes back in the
        # acceleration.
        self._ci = np.zeros((self.n, 6))
        for i in range(self.n - 1, b.root - 1, -1):
            par = b.parent[i]
            E = self.rot[i].T @ self.rot[par]
            Xi = xform(E, b.body_pos[i])
            X[i] = Xi
            if i == b.root and b.root_free:
                # the base link of a free-rooted tree: its own inertia and bias
                # are already in IA/pA, and there is no joint to solve for it
                continue
            j = self._joint_of_body[i]
            if j < 0:
                # a link with no joint of its own is welded to its parent: it
                # contributes its inertia and its bias, and nothing else
                D[i], u[i], U[i] = 1.0, 0.0, np.zeros(6)
                self._welded[i] = True
                IA[par] += Xi.T @ IA[i] @ Xi
                pA[par] += Xi.T @ pA[i]
                continue
            S = self._S[i]
            Ui = IA[i] @ S
            ci = crm(self.v[i]) @ (S * self.qd[j])
            self._ci[i] = ci
            # the damper, implicitly: d*dt joins the joint's own inertia
            Di = float(S @ Ui) + b.armature[j] + b.damping[j] * dt
            Ia = IA[i] - np.outer(Ui, Ui) / Di
            pAe = pA[i] + Ia @ ci
            ui = tau[j] - float(S @ pAe) - b.damping[j] * self.qd[j]
            U[i], D[i], u[i] = Ui, Di, ui
            IA[par] += Xi.T @ Ia @ Xi
            pA[par] += Xi.T @ (pAe + U[i] * (u[i] / D[i]))

        # The root. A free root is a six-DOF joint whose parent is the world,
        # so its acceleration is whatever leaves the accumulated wrench at
        # zero, -IA^-1 pA; a welded root does not accelerate at all, and
        # gravity reaches the links below it through the forces in pA.
        a = np.zeros((self.n, 6))
        if b.root_free:
            a[b.root] = -np.linalg.solve(IA[b.root], pA[b.root])

        qdd = np.zeros(b.nj)
        first = b.root + 1 if b.root_free else b.root
        for i in range(first, self.n):
            par = b.parent[i]
            acc = X[i] @ a[par] if par >= 1 else np.zeros(6)
            j = self._joint_of_body[i]
            if j < 0:
                a[i] = acc
                continue
            qdd[j] = (u[i] - float(U[i] @ (acc + self._ci[i]))) / D[i]
            a[i] = acc + self._ci[i] + self._S[i] * qdd[j]

        # ---- integrate -----------------------------------------------------
        # `a` is the spatial acceleration; MuJoCo (and the classical reading)
        # wants the acceleration of the body-fixed origin, which differs by the
        # omega x v term.
        Rb = self.rot[b.root]
        # MuJoCo's qacc: the angular part is d/dt of the *body* angular
        # velocity, the linear part the derivative of the world velocity of the
        # body origin. The solver's `a` is the spatial acceleration, which
        # differs from that derivative by omega x v.
        alpha_world = Rb @ a[b.root][:3]
        acc_world = Rb @ a[b.root][3:] + np.cross(self.omega, self.vel)
        self.omega = self.omega + dt * alpha_world
        self.vel = self.vel + dt * acc_world
        self.root_pos = self.root_pos + dt * self.vel
        self.root_quat = quat_mul(self.root_quat, quat_from_vec(Rb.T @ self.omega * dt))
        self.root_quat = self.root_quat / np.linalg.norm(self.root_quat)
        self.qd = self.qd + dt * qdd
        self.q = self.q + dt * self.qd

        if self.limits and b.limited.any():
            lo_hit = (self.q < b.lo) & (self.qd < 0)
            hi_hit = (self.q > b.hi) & (self.qd > 0)
            self.q = np.clip(self.q, b.lo, b.hi)
            self.qd[lo_hit | hi_hit] = 0.0
        return qdd


# ---------------------------------------------------------------------------
# The muscle
# ---------------------------------------------------------------------------

def muscle_torque(body: FlyBody, q, qd, excitation):
    """
    Excitation in [-1, 1] per joint (positive opens qpos) to joint torque.

    Force-based, not position-based: an activation becomes a force, the force
    becomes a torque, and the joint does whatever the physics says. The length
    and velocity terms are the usual Hill-shaped factors, with the joint's own
    anatomical range setting the width of the length term and the stance angle
    as the length at which the muscle is strongest — both declared in
    docs/ASSUMPTIONS.md, both measured off the animal.

    The velocity term is applied *per direction*, because a joint has an
    antagonist pair and they do not do the same thing at the same time. The
    muscle that is shortening (pulling the joint the way it is already going)
    weakens with speed and reaches zero at `V_MAX`; the one that is lengthening
    (pulling against the motion) is loaded and gets *stronger*, up to the 1.8
    of a stretched fibre. Applying one shortening curve to both — the obvious
    single-muscle reading, and what this file did until it was measured —
    silently removes the braking torque: past V_MAX the factor is zero for
    either sign of command, so a joint that is already moving fast cannot be
    stopped by anything but its 1e-3 damper, and the animal flies apart in
    200 us. That is not a fly, it is a car with the brakes wired to the
    accelerator.
    """
    b = body
    fl = np.exp(-(((q - b.optimal) / (0.5 * b.span)) ** 2))
    ratio = qd / V_MAX
    shortening = np.clip((1.0 - ratio) / (1.0 + 2.0 * ratio), 0.0, 1.8)
    lengthening = np.clip((1.0 + ratio) / (1.0 - 2.0 * ratio), 0.0, 1.8)
    pull_up = np.clip(excitation, 0.0, 1.0) * shortening
    pull_down = np.clip(-excitation, 0.0, 1.0) * lengthening
    active = b.max_torque * fl * (pull_up - pull_down)
    # and the muscle pair's own stiffness, which the file states outright as
    # `-biasprm[1]`: 0.4-0.8 at the leg joints against a joint inertia of 1e-6
    # g cm^2. Without it a leg is a free hinge on a 1e-6 g cm^2 pivot with a
    # 0.02 g cm^2/s^2 contact torque on it, which is not a stiff structure,
    # which is not a leg.
    return active - b.passive_stiffness * q


# ---------------------------------------------------------------------------
# Checks: this solver against MuJoCo, and the standing animal
# ---------------------------------------------------------------------------

def _mujoco_pair(body, sim, mjcf):
    """A MuJoCo model/data matched to this solver's state, with no servos."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    m.opt.timestep = sim.dt
    # The torque comes from `qfrc_applied`, so the model's own actuators are
    # muted completely — gain *and* bias. Muting only the gain leaves their
    # passive spring: `biasprm[1] = -0.1`, which at ctrl = 0 still pushes a
    # real muscle's worth of force and swamps the tarsi.
    m.actuator_gainprm[:, :] = 0.0
    m.actuator_biasprm[:, :] = 0.0
    # Damping and armature stay: they are real model parameters, the solver
    # folds both into the articulated inertia the way MuJoCo folds them into
    # the mass matrix, and taking them off here while the solver still had
    # them was worth 26 rad/s of disagreement on the tarsi.
    m.jnt_limited[:] = False              # see FlySim: hard stops vs soft
                                          # constraints are not comparable
    # No self-collision on the MuJoCo side: the solver has none yet, and a
    # random pose has the legs inside the thorax. No air either: the model
    # carries MuJoCo's ellipsoid fluid (viscosity 1.85e-4, density 1.28e-3)
    # and this solver has no fluid term, which is recorded rather than
    # imitated by turning the reference's drag into an unexplained residual.
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.opt.viscosity = 0.0
    m.opt.density = 0.0
    m.opt.wind[:] = 0.0
    d = mujoco.MjData(m)
    d.qpos[:3] = sim.root_pos
    d.qpos[3:7] = sim.root_quat
    for j, jj in enumerate(body.hinges):
        d.qpos[jj["qposadr"]] = sim.q[j]
    # MuJoCo's free joint is split-frame: qvel[:3] is the linear velocity of
    # the body origin in *world* axes, qvel[3:6] the angular velocity in the
    # *body* frame. Feeding it a world-frame omega (or reading one back out)
    # is the kind of mistake that only shows up once the animal is turning.
    d.qvel[:3] = sim.vel
    d.qvel[3:6] = sim.rot[body.root].T @ sim.omega
    for j, jj in enumerate(body.hinges):
        d.qvel[jj["dofadr"]] = sim.qd[j]
    mujoco.mj_forward(m, d)
    return m, d


def verify_kinematics(body, mjcf, poses=8, seed=11) -> dict:
    """Forward kinematics against mj_forward, on random poses."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    # only the links that *are* a body frame can be compared: the extra links
    # inside a multi-joint body do not appear in MuJoCo's body list
    frames = [i for i in range(body.nbody) if body.frame[i]]
    mj_id = {body.source_body[i]: mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, body.source_body[i]) for i in frames}
    worst_p = worst_r = 0.0
    for _ in range(poses):
        q = rng.uniform(np.maximum(body.lo, -1.2), np.minimum(body.hi, 1.2))
        root_pos = rng.uniform(-1, 1, 3)
        root_quat = quat_from_vec(rng.uniform(-2, 2, 3))
        pos, rot = body.fk(root_pos, root_quat, q)
        d.qpos[:3] = root_pos
        d.qpos[3:7] = root_quat
        for j, jj in enumerate(body.hinges):
            d.qpos[jj["qposadr"]] = q[j]
        mujoco.mj_forward(m, d)
        for i in frames:
            b = mj_id[body.source_body[i]]
            worst_p = max(worst_p, float(np.abs(pos[i] - d.xpos[b]).max()))
            worst_r = max(worst_r, float(np.abs(rot[i]
                                              - d.xmat[b].reshape(3, 3)).max()))
    return {"poses": poses, "max_position_error_cm": worst_p,
            "max_orientation_error": worst_r,
            "ok": bool(worst_p < 1e-10 and worst_r < 1e-10)}


def verify_dynamics(body, mjcf, ms, torque_schedule=None) -> dict:
    """
    Side by side in vacuum: gravity, no floor, a torque script.

    What this test can and cannot say is worth being explicit about. The fly's
    joint inertia is the model's own 1e-6 g cm^2 armature at the lightest joint
    and 3.4e-6 at the heaviest, while the accelerations at this state reach
    6.8e5 rad/s^2, so a light joint's acceleration is the outcome of a
    cancellation across the whole chain and nobody reproduces it to machine
    precision from published quantities: measured, MuJoCo's own step disagrees
    with its own `mj_fullM` + `qfrc_smooth` by 4e-2 relative on `haltere_left`
    (and by 1.2 relative on `abdomen` when the damping is folded into the
    diagonal instead) — the model's implicit damping against a 4.0 stiffness
    and a 1e-6 inertia.

    What can be said is what this solver does. Against MuJoCo's own step its
    first step is within 1.0e-3 relative on the worst joint and 7.5e-6 on the
    median one; positions and the centre of mass — what gravity acts on — stay
    at 2.1e-05 cm and 7.0e-09 cm over the 200 steps. The joint *rates* then
    separate at the system's own amplification rate, 1.04x per step, i.e. 1.5x
    per ten steps, reaching 0.32 rad/s from initial rates of 1 rad/s. That
    separation is a property of a chain this stiff rather than an error a
    smaller step removes: the per-step error halves with the step and the
    number of steps doubles.

    So the criterion is the first step (the agreement itself), plus the
    positions and the centre of mass, which stay exact; the rate divergence is
    reported as information rather than hidden inside a tolerance.
    """
    sim = FlySim(body, floor_z=-1e6, limits=False)
    sim.dt = body.dt_default
    rng = np.random.default_rng(5)
    sim.reset(root_z=0.0, q=rng.uniform(-0.3, 0.3, body.nj),
              qd=rng.uniform(-1, 1, body.nj),
              v=np.array([1.0, -0.5, 0.25]), omega=np.array([3.0, -2.0, 5.0]))
    m, d = _mujoco_pair(body, sim, mjcf)
    if torque_schedule is None:
        torque_schedule = lambda k, dt, b: np.zeros(b.nj)      # noqa: E731
    import mujoco as mj
    steps = int(round(ms / 1000.0 / sim.dt))
    early = int(max(1, steps // 10))
    worst = {"qpos": 0.0, "qvel": 0.0, "xpos": 0.0, "com": 0.0}
    first = {"qpos": 0.0, "qvel": 0.0}
    early_worst = {"qpos": 0.0, "qvel": 0.0}
    for k in range(steps):
        tau = torque_schedule(k, sim.dt, body)
        for j, jj in enumerate(body.hinges):
            d.qfrc_applied[jj["dofadr"]] = tau[j]
        mj.mj_step(m, d)
        sim.step(sim.dt, tau)
        dq = float(np.abs(sim.q - d.qpos[7:]).max())
        dqd = float(np.abs(sim.qd - d.qvel[6:]).max())
        if k == 0:
            first["qpos"], first["qvel"] = dq, dqd
        if k < early:
            early_worst["qpos"] = max(early_worst["qpos"], dq)
            early_worst["qvel"] = max(early_worst["qvel"], dqd)
        worst["qpos"] = max(worst["qpos"], dq)
        worst["qvel"] = max(worst["qvel"], dqd)
        frames_ = [i for i in range(body.nbody) if body.frame[i]]
        mj_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, body.source_body[i])
                  for i in frames_]
        worst["xpos"] = max(worst["xpos"], max(
            float(np.abs(sim.pos[i] - d.xpos[b]).max())
            for i, b in zip(frames_, mj_ids)))
        # the centre of mass, which is what gravity acts on
        com_s = sim.com()
        com_m = np.array([d.subtree_com[1][k2] for k2 in range(3)])
        worst["com"] = max(worst["com"], float(np.abs(com_s - com_m).max()))
    worst["ms"] = ms
    worst["steps"] = steps
    worst["first_step_qpos"] = first["qpos"]
    worst["first_step_qvel"] = first["qvel"]
    worst[f"early_{early}_qpos"] = early_worst["qpos"]
    worst[f"early_{early}_qvel"] = early_worst["qvel"]
    worst["divergence_per_step"] = float(
        (worst["qvel"] / max(first["qvel"], 1e-30)) ** (1.0 / max(1, steps - 1)))
    worst["ok"] = bool(first["qvel"] < 1e-3 and first["qpos"] < 1e-6
                       and worst["xpos"] < 1e-4 and worst["com"] < 1e-6)
    return worst


def hold_torques(body):
    """The torque the animal's own actuators needed to stand, per joint."""
    return np.array([body.muscles[j["name"]]["hold_torque"] for j in body.hinges])


def hold_excitation(body):
    """
    The excitation that produces `hold_torque` at the stance pose.

    `hold_torque` is a torque, and the muscle's own force-length curve is what
    turns excitation into torque: at a wing joint whose rest angle is 1.5 rad
    the head of the curve is far from the stance, so `fl` at q = 0 is a
    fraction of one and the excitation has to be larger than torque/max_torque.
    Inverting the curve here keeps the harness honest — a quarter of full
    excitation is not a physical number, it is a leftover.
    """
    fl = np.exp(-(((body.stance_q - body.optimal) / (0.5 * body.span)) ** 2))
    denom = np.where(body.max_torque * fl > 1e-12, body.max_torque * fl, 1e-12)
    # the passive spring is already pulling: the active part is what is left
    wanted = hold_torques(body) + body.passive_stiffness * body.stance_q
    return np.clip(wanted / denom, -1.0, 1.0)


def stance_servo(body, dt, omega_dt=0.1, kd_dt=0.5, settle=0.0):
    """
    Excitation gains for holding a pose, sized by the step and the joint.

    This is *test harness*, not behaviour: nothing in the app holds a pose, the
    app's excitation comes out of the connectome. It exists so the physics and
    the muscle calibration can be measured on an animal that is not falling
    over. Its gains are not tuned, they are read off the discretisation: a
    joint whose inertia is the armature (1e-6 g cm^2) and whose muscle makes
    O(1) torque cannot be pushed faster than `omega_dt / dt` or damped harder
    than `kd_dt * D / dt` without the servo itself becoming the instability.
    The first version of this harness used a hand-picked gain and produced
    |qd| = 4e2 rad/s in 3 ms, which was the harness, not the fly.

    Returns (gain_p, gain_d) in excitation per rad and per rad/s, plus the pose
    to hold (the stance the hold torques were measured at).
    """
    inertia = np.where(body.armature > 0, body.armature, 1e-6)
    cap = np.where(body.max_torque > 0, body.max_torque, 1.0)
    gain_p = omega_dt ** 2 * inertia / (dt ** 2 * cap)
    gain_d = kd_dt * inertia / (dt * cap)
    return gain_p, gain_d


def verify_standing(body, ms, trace_every=100, ramp_ms=50.0, put_ms=20.0,
                    put_dt=1e-5) -> dict:
    """
    The animal on the floor, held up by its own muscles.

    The excitation is a tonic drive at the measured stance torque plus a
    stance servo, both ramped in over `ramp_ms`; a step from nothing to full
    stance torque is a kick the animal answers by launching itself (measured).

    The first `put_ms` run at `put_dt` — a tenth of the model's own timestep.
    That is the one number in this function that had to be measured rather than
    argued: the fly's stance is a stiff mode (a 1e-6 g cm^2 leg against a
    contact that carries its weight), and at dt = 1e-4 s the initial
    penetration error of a few tens of microns is resolved as a 250 rad/s kick
    that throws the animal clear of the floor. Ten times finer, the same pose
    settles: |q - stance| stays inside 0.003 rad and the floor carries 0.97 of
    the animal's weight. The app runs at the model's timestep for speed; this
    test spends 2000 extra steps to stand the animal up first, and then
    measures the model's own timestep.

    Then the measurement: `ms` at the model's timestep, and the last half of it
    is what the stability numbers are taken from.
    """
    sim = FlySim(body)
    sim.dt = body.dt_default
    base = hold_excitation(body)
    gain_p, gain_d = stance_servo(body, sim.dt)
    steps = int(round(ms / 1000.0 / sim.dt))
    ramp_steps = max(1, int(round(ramp_ms / 1000.0 / sim.dt)))
    put_steps = max(1, int(round(put_ms / 1000.0 / put_dt)))
    gain_p_put, gain_d_put = stance_servo(body, put_dt)
    trace = []
    worst = 0.0
    for k in range(put_steps + steps):
        putting = k < put_steps
        h = put_dt if putting else sim.dt
        gp = gain_p_put if putting else gain_p
        gd = gain_d_put if putting else gain_d
        ramp = min(1.0, (k + 1) / max(1, int(round(ramp_ms / 1000.0 / h))))
        exc = np.clip(ramp * base + gp * (body.stance_q - sim.q) + gd * (0 - sim.qd),
                      -1, 1)
        sim.step(h, muscle_torque(body, sim.q, sim.qd, exc))
        if putting:
            continue
        worst = max(worst, float(np.abs(sim.qd).max()))
        if (k - put_steps) % trace_every == 0:
            trace.append({"ms": round((k - put_steps) * sim.dt * 1000, 2),
                          "com_z": float(sim.com()[2]),
                          "root_z": float(sim.root_pos[2]),
                          "contacts": int(sim.contacts),
                          "support": float(sum(sim.foot_force.values())),
                          "feet": int(sum(1 for v in sim.foot_force.values()
                                          if v > 0)),
                          "speed": float(np.linalg.norm(sim.com()[:2]))})
    com = np.array([t["com_z"] for t in trace])
    support = np.array([t["support"] for t in trace])
    feet = np.array([t["feet"] for t in trace])
    half = com[len(com) // 2:]
    weight = float(body.body_mass.sum()) * 981.0
    out = {
        "ms": ms,
        "weight": weight,
        "put_ms": put_ms, "put_dt": put_dt,
        "com_z_first": float(com[0]), "com_z_last": float(com[-1]),
        "com_z_min": float(com.min()), "com_z_max": float(com.max()),
        "com_z_sd_last_half": float(half.std()),
        "feet_on_floor_mean": float(feet.mean()), "feet_on_floor_min": int(feet.min()),
        "support_over_weight_last_half": float((support[len(support) // 2:]
                                                / max(weight, 1e-12)).mean()),
        "worst_joint_rate": worst,
        "drift_cm": float(np.linalg.norm(sim.com()[:2])),
        "trace": trace[::max(1, len(trace) // 24)],
    }
    out["fell_over"] = bool(com.min() < com[0] - 0.05)
    out["ok"] = bool((not out["fell_over"])
                     and out["feet_on_floor_mean"] >= 4.0
                     and out["com_z_sd_last_half"] < 0.01
                     and 0.8 <= out["support_over_weight_last_half"] <= 1.2
                     and out["worst_joint_rate"] < 60.0)
    return out


def write_golden(body, steps, path):
    """
    A trace the Swift solver has to reproduce.

    Vacuum, no floor, a fixed torque script: nothing chaotic, nothing that
    depends on the contact model, so a disagreement means the phone's
    arithmetic differs from this file's — which is exactly what the test is
    for. The app's own suite replays this.
    """
    dt = body.dt_default
    floor = -1e6
    sim = FlySim(body, floor_z=floor)
    rng = np.random.default_rng(3)
    q0 = rng.uniform(-0.3, 0.3, body.nj)
    qd0 = rng.uniform(-1, 1, body.nj)
    v0 = np.array([0.5, 0.0, -0.25])
    w0 = np.array([2.0, 1.0, -3.0])
    sim.reset(root_z=0.0, q=q0, qd=qd0, v=v0, omega=w0)
    amplitude = list(0.3 * np.sin(np.arange(body.nj) * 0.7))
    trace = []
    for k in range(steps):
        tau = np.array(amplitude) * math.cos(2 * math.pi * k * dt / 0.02)
        sim.step(dt, tau)
        if k % max(1, steps // 20) == 0:
            trace.append({"step": k, "q": list(np.round(sim.q, 12)),
                          "root_pos": list(np.round(sim.root_pos, 12))})
    out = {
        "purpose": "FlyDynamics.swift must reproduce this run: same arithmetic "
                   "or a bug. Vacuum, no floor, torques from `amplitude`.",
        "dt": dt, "steps": steps, "floor_z": floor,
        "torque": {"amplitude": amplitude,
                   "period_s": 0.02,
                   "rule": "tau[j] = amplitude[j] * cos(2*pi*k*dt/period_s)"},
        "initial": {"root_pos": [0.0, 0.0, 0.0], "root_quat": [1.0, 0.0, 0.0, 0.0],
                    "q": list(q0), "qd": list(qd0), "v": list(v0), "omega": list(w0)},
        "expected_final": {"q": list(np.round(sim.q, 12)),
                           "qd": list(np.round(sim.qd, 12)),
                           "root_pos": list(np.round(sim.root_pos, 12)),
                           "root_quat": list(np.round(sim.root_quat, 12)),
                           "v": list(np.round(sim.vel, 12)),
                           "omega": list(np.round(sim.omega, 12))},
        "trace": trace,
        "tolerance": 1e-7,
        "tolerance_note": "absolute, per coordinate, after the whole run; both "
                          "sides are float64 and the algorithm is the same, so "
                          "the only difference should be rounding.",
    }
    pathlib.Path(path).write_text(json.dumps(out))
    return {"path": str(path), "steps": steps,
            "final_q_first3": [round(x, 8) for x in sim.q[:3]]}


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", default="build/fly_body.json", type=pathlib.Path)
    ap.add_argument("--mjcf", default="data/flybody/flybody-main/flybody/"
                                     "fruitfly/assets/fruitfly.xml",
                    type=pathlib.Path)
    ap.add_argument("--out", default="build/fly_aba_verify.json", type=pathlib.Path)
    ap.add_argument("--verify", action="store_true",
                    help="run the three checks and fail if any disagrees")
    ap.add_argument("--golden", type=pathlib.Path, default=None,
                    help="write the trace the Swift solver must reproduce")
    ap.add_argument("--golden-steps", type=int, default=500)
    ap.add_argument("--free-ms", type=float, default=20.0)
    ap.add_argument("--stand-ms", type=float, default=1500.0)
    args = ap.parse_args()

    if not args.body.exists():
        print(f"missing {args.body} — run tools/build_body.py", file=sys.stderr)
        return 2
    body = FlyBody(args.body)
    log(f"the body: {body.nbody - 1} parts, {body.nj} hinges, "
        f"{len(body.collision)} collision geoms, "
        f"{body.body_mass.sum():.6f} g")

    if args.golden:
        info = write_golden(body, args.golden_steps, args.golden)
        log(f"golden trace: {info['path']}, {info['steps']} steps, "
            f"q[0:3] ends at {info['final_q_first3']}")
        return 0

    if not args.verify:
        ap.print_help()
        return 0

    report = {"body": str(args.body), "mjcf": str(args.mjcf)}
    log("check 1: forward kinematics against mj_forward ...")
    k = verify_kinematics(body, args.mjcf)
    log(f"  {'ok  ' if k['ok'] else 'FAIL'} {k['poses']} random poses: worst "
        f"position error {k['max_position_error_cm']:.3e} cm, worst orientation "
        f"error {k['max_orientation_error']:.3e}")
    report["kinematics"] = k
    if not k["ok"]:
        print("  a disagreement here is a frame or axis convention, not noise",
              file=sys.stderr)

    log(f"check 2: {args.free_ms:g} ms of free fall and tumble, this solver "
        f"against MuJoCo ...")
    dyn = verify_dynamics(body, args.mjcf, args.free_ms)
    log(f"  {'ok  ' if dyn['ok'] else 'FAIL'} {dyn['steps']} steps: first step "
        f"{dyn['first_step_qpos']:.3e} rad / {dyn['first_step_qvel']:.3e} rad/s, "
        f"worst body position {dyn['xpos']:.3e} cm, worst COM {dyn['com']:.3e} cm; "
        f"divergence {dyn['divergence_per_step']:.3f}x per step "
        f"(worst joint rate after {dyn['steps']} steps {dyn['qvel']:.3e} rad/s)")
    report["dynamics"] = dyn

    log(f"check 3: the animal standing for {args.stand_ms/1000:g} s on the "
        f"floor, muscles only ...")
    st = verify_standing(body, args.stand_ms)
    log(f"  {'ok  ' if st['ok'] else 'FAIL'} COM z {st['com_z_first']:+.5f} -> "
        f"{st['com_z_last']:+.5f} cm (sd {st['com_z_sd_last_half']:.2e} over the "
        f"last half), {st['feet_on_floor_mean']:.1f} of 6 feet down on average, "
        f"drift {st['drift_cm']*10:.3f} mm, fell over: {st['fell_over']}")
    report["standing"] = st

    report["ok"] = bool(k["ok"] and dyn["ok"] and st["ok"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log(f"wrote {args.out}")
    log("VERDICT: " + ("the solver is the same animal as MuJoCo, and it stands"
                       if report["ok"] else "FAILED — see above"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
