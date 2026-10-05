#!/usr/bin/env python3
"""
Verify the articulated-body solver against MuJoCo, term by term.

The solver in `fly_aba.py` is what will run on the phone, so it has to be
checked against the reference the body asset came from, not against itself.
Each stage of the recursion is compared on its own, on models small enough
that a failure says which stage broke:

    1  kinematics      body positions and orientations, random joint angles
    2  velocities      spatial velocities of every link
    3  inertia         H(q), assembled two ways (composite + CRBA)
    4  bias forces     C(q, qd) via recursive Newton-Euler
    5  accelerations   qdd from the solver vs MuJoCo, one step
    6  trajectories    N steps of both, from random states

The mini models are self-contained MJCF, so this runs in a second and needs no
data download. `--mjcf` adds the real flybody model to the same battery.

    python3 tools/verify_aba.py
    python3 tools/verify_aba.py --mjcf data/flybody/.../fruitfly.xml
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import build_body as bb  # noqa: E402
import fly_aba as fa  # noqa: E402

try:
    import mujoco as mj
except ImportError:  # pragma: no cover - the tool is useless without it
    print("mujoco is required: pip install mujoco")
    raise SystemExit(2)

OBJ = {
    "body": mj.mjtObj.mjOBJ_BODY,
    "joint": mj.mjtObj.mjOBJ_JOINT,
    "geom": mj.mjtObj.mjOBJ_GEOM,
}

# Long enough to expose a frame error (a wrong axis shows up once a joint
# bends), short enough that this stays a fast check.
STEPS = 50

PENDULUM = """
<mujoco><compiler angle="radian"/><option timestep="1e-5"/>
 <worldbody>
  <body name="arm" pos="0 0 0">
    <joint name="j" type="hinge" axis="0 1 0" range="-3 3"/>
    <geom type="sphere" size="0.02" mass="0.001" pos="0.1 0 0"/>
  </body>
 </worldbody></mujoco>
"""

CHAIN = """
<mujoco><compiler angle="radian"/><option timestep="1e-5"/>
 <worldbody>
  <body name="a" pos="0.1 0.05 -0.02" quat="0.9 0.1 0.3 0.3">
    <geom type="sphere" size="0.05" mass="0.002"/>
    <body name="b" pos="0.12 0 0.03" quat="0.95 0.2 0.1 -0.2">
      <joint name="j" type="hinge" axis="0.3 0.8 0.5" range="-3 3" damping="0.0"/>
      <geom type="sphere" size="0.05" mass="0.001"/>
      <body name="c" pos="0.1 0 0" quat="0.98 0.05 -0.1 0.15">
        <joint name="k" type="hinge" axis="0.4 0.2 -0.6" range="-3 3"/>
        <geom type="sphere" size="0.05" mass="0.001" pos="0.03 0.01 -0.02"/>
      </body>
    </body>
  </body>
 </worldbody></mujoco>
"""

BRANCH = """
<mujoco><compiler angle="radian"/><option timestep="1e-5"/>
 <worldbody>
  <body name="trunk" pos="0 0 0">
    <freejoint/>
    <geom type="sphere" size="0.05" mass="0.004"/>
    <body name="arm_l" pos="0.1 0 0" quat="0.97 0 0.2 0.1">
      <joint name="l1" type="hinge" axis="0 1 0" range="-3 3"/>
      <geom type="sphere" size="0.03" mass="0.001" pos="0.05 0 0"/>
      <body name="claw_l" pos="0.08 0 0" quat="0.99 0.1 0 0">
        <joint name="l2" type="hinge" axis="1 0 0" range="-3 3"/>
        <geom type="sphere" size="0.02" mass="0.0005"/>
      </body>
    </body>
    <body name="arm_r" pos="-0.1 0 0">
      <joint name="r1" type="hinge" axis="0 1 0" range="-3 3"/>
      <geom type="sphere" size="0.03" mass="0.001"/>
    </body>
  </body>
 </worldbody></mujoco>
"""

MULTI = """
<mujoco><compiler angle="radian"/><option timestep="1e-5"/>
 <worldbody>
  <body name="base" pos="0 0 0">
    <freejoint/>
    <geom type="sphere" size="0.05" mass="0.003"/>
    <body name="head" pos="0.08 0 0.02" quat="0.96 0.1 -0.2 0.15">
      <joint name="pitch" type="hinge" axis="0 1 0" range="-3 3"/>
      <joint name="twist" type="hinge" axis="1 0 0" range="-3 3"/>
      <joint name="roll" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom type="sphere" size="0.03" mass="0.001"/>
    </body>
  </body>
 </worldbody></mujoco>
"""


# --------------------------------------------------------------------------
# model plumbing


def asset_from_xml(xml: str, path: Path) -> tuple:
    """Compile an MJCF snippet and read the same tree the asset compiler does."""
    path.write_text(xml)
    model = mj.MjModel.from_xml_path(str(path))
    name = lambda kind, i: (mj.mj_id2name(model, OBJ[kind], i) or f"<{kind}:{i}>")

    tree = bb.read_tree(model, name)
    raw = {
        "units": {"length": "cm"},
        "gravity": [float(x) for x in model.opt.gravity],
        "timestep_s": float(model.opt.timestep),
        "floor_z": -1e6,                       # no floor: vacuum comparison
        "stance_root_z": 0.0,
        "bodies": tree["bodies"],
        "joints": tree["joints"],
        "collision": tree["collision"],
        "actuators": [],
        "muscles": {},
        "legs": {},
        "visual": [],
        "meshes": {},
    }
    out = path.with_suffix(".json")
    out.write_text(json.dumps(raw))
    return model, fa.FlyBody(out), out


def make_sim(model, body, **kw) -> "fa.FlySim":
    sim = fa.FlySim(body, floor_z=-1e6, limits=False, **kw)
    sim.dt = float(model.opt.timestep)
    return sim


def relerr(mine, ref, floor=0.0) -> float:
    """
    Relative error that survives a zero reference.

    `|mine - ref| / |ref|` is useless where the truth is zero and both sides
    are rounding noise — it reports order-one failures for a perfect solver.
    Scaling by whichever side is larger fixes the sign of that mistake but not
    its magnitude, since two noise vectors of the same size still differ by
    their own size. So the scale has a floor, given in the units of the
    quantity being compared: below it the check is 'both answers are zero',
    above it a wrong answer cannot hide.
    """
    mine = np.asarray(mine, float)
    ref = np.asarray(ref, float)
    scale = max(float(np.abs(mine).max()), float(np.abs(ref).max()), float(floor))
    return float(np.abs(mine - ref).max()) / max(scale, 1e-30)


def char_scales(body) -> dict:
    """
    Characteristic magnitude of each quantity, from the model itself.

    These turn `relerr` into an atol+rtol comparison: at the scale of the
    model, `relerr / tol` is a relative error, and where the true answer is
    zero it is an absolute one expressed in units of the model. Both are
    needed — a fly falling in vacuum has zero hinge acceleration, so the check
    has to call that zero without also calling 0.4 rad/s^2 zero, which is what
    a missing velocity-product term looks like.
    """
    total = float(np.sum(body.body_mass)) or 1e-6
    glen = float(np.abs(np.asarray(body.gravity, float)).max()) or 1.0
    span = max([float(np.linalg.norm(b["pos"])) for b in body.raw["bodies"]] + [1e-3])
    dt = float(getattr(body, "timestep", 0.0) or body.dt_default)
    return {
        "H": total * span * span,
        "C": total * glen * span,
        "acc": glen,
        "q": 1.0,
        "vel": glen * dt,
        "pos": span,
    }


def strip_passive(model, body, passive=False) -> None:
    """
    `passive=True` keeps damping, armature and joint springs — the fly's own
    passive properties — and removes only the parts the solver deliberately
    does not model yet (muscle actuators, self-collision, air). The other
    direction, `passive=False`, is for the models whose dynamics are being
    checked term by term with nothing in the way.
    """
    """
    Remove every passive term from both sides — nothing left to blame.

    Contacts go too. `check_*` samples joint angles from the model's own
    ranges, and a fly at a random pose has its legs inside its own thorax, so
    MuJoCo answers with a contact force of 1e10 and the solver, which has no
    self-collision at all yet, answers with gravity. Comparing those two
    numbers says nothing about either. The floor is already absent from the
    solver side (floor_z = -1e6) and the contacts are off here, so the
    dynamics rows are pure rigid-body dynamics; contacts are checked
    separately, against the floor, in `verify_standing`.
    """
    if not passive:
        model.dof_damping[:] = 0.0
        model.dof_armature[:] = 0.0
        model.jnt_stiffness[:] = 0.0
        body.damping = np.zeros(body.nj)
        body.armature = np.zeros(body.nj)
        body.stiffness = np.zeros(body.nj)
    model.jnt_limited[:] = False
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    # The fly's muscles are springs before they are motors: the actuator bias
    # `-0.1 * q` is the passive stiffness of the real muscle, and at ctrl = 0
    # it still pushes — 0.02 of a unit, which at the tarsus' own inertia is an
    # acceleration of 4e7. The solver is being asked about rigid-body dynamics
    # here, with no muscle force at all, so both sides run with the actuators
    # off; the passive muscle is a real part of the model and is checked
    # against MuJoCo separately, as a force.
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    # And the air. `fruitfly.xml` sets viscosity 1.85e-4 / density 1.28e-3, so
    # MuJoCo's ellipsoid fluid model is adding a drag force to every link that
    # moves — 1.9 units of generalized force at 3 cm/s, which is larger than
    # gravity on the tarsi. The solver has no fluid model yet (recorded in
    # docs/ASSUMPTIONS.md, needed for the wing, negligible for walking), so
    # both sides run in vacuum here.
    model.opt.viscosity = 0.0
    model.opt.density = 0.0
    model.opt.wind[:] = 0.0


def mj_gravity(model) -> np.ndarray:
    return np.concatenate([np.zeros(3), -np.asarray(model.opt.gravity, float)])


def mj_inertia(model, data) -> np.ndarray:
    H = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, data, H)
    return H


def full_mass(model, data) -> np.ndarray:
    H = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, data, H)
    return H


# --------------------------------------------------------------------------
# reference implementations, written straight from the equations


def reference_h_c(sim: "fa.FlySim", qdd=None):
    """Joint-space inertia (composite rigid body) and bias (Newton-Euler)."""
    b = sim.b
    n = sim.n
    if qdd is None:
        qdd = np.zeros(b.nj)

    Ic = np.array(b.I)
    for i in range(n - 1, b.root - 1, -1):
        if i == b.root and b.root_free:
            continue
        par = b.parent[i]
        E = sim.rot[i].T @ sim.rot[par]
        Xi = fa.xform(E, b.body_pos[i])
        Ic[par] = Ic[par] + Xi.T @ Ic[i] @ Xi

    H = np.zeros((b.nj, b.nj))
    for i in range(1, n):
        j = sim._joint_of_body[i]
        if j < 0:
            continue
        H[j, j] = b.armature[j] + sim._S[i] @ Ic[i] @ sim._S[i]
        k, F = i, Ic[i] @ sim._S[i]
        while True:
            par = b.parent[k]
            if par < 1:
                break
            E = sim.rot[k].T @ sim.rot[par]
            Xi = fa.xform(E, b.body_pos[k])
            F = Xi.T @ F
            k = par
            jj = sim._joint_of_body[k]
            if jj >= 0:
                # armature is a rotor inertia *at* joint jj: MuJoCo adds it
                # to the diagonal of M only (mj_fullM), never to a cross term
                H[jj, j] = H[j, jj] = sim._S[k] @ F

    a = np.zeros((n, 6))
    f = np.zeros((n, 6))
    g = np.asarray(b.gravity, float)
    for i in range(1, n):
        par = b.parent[i]
        E = sim.rot[i].T @ sim.rot[par]
        Xi = fa.xform(E, b.body_pos[i])
        v = sim.v[i]
        j = sim._joint_of_body[i]
        a[i] = Xi @ a[par] if par >= 1 else np.zeros(6)
        if j >= 0:
            # the joint's velocity-product acceleration, v x (S qd): the term
            # that makes every moving pose wrong when it is left out
            a[i] = a[i] + fa.crm(v) @ (sim._S[i] * sim.qd[j]) + sim._S[i] * qdd[j]
        f[i] = b.I[i] @ a[i] + fa.crf(v) @ (b.I[i] @ v)
        # gravity is a force on every body, so that a free root and a welded
        # root are the same algorithm
        fg = b.body_mass[i] * (sim.rot[i].T @ g)
        f[i] -= np.concatenate([np.cross(sim._com[i], fg), fg])
    C = np.zeros(b.nj)
    for i in range(n - 1, b.root - 1, -1):
        if i == b.root and b.root_free:
            continue
        j = sim._joint_of_body[i]
        if j >= 0:
            C[j] = sim._S[i] @ f[i]
        par = b.parent[i]
        if par >= 1:
            E = sim.rot[i].T @ sim.rot[par]
            Xi = fa.xform(E, b.body_pos[i])
            f[par] = f[par] + Xi.T @ f[i]
    return H, C


# --------------------------------------------------------------------------
# the checks


def check_kinematics(model, body, rng) -> float:
    worst = 0.0
    for _ in range(5):
        q = rng.uniform(body.lo + 1e-3, body.hi - 1e-3)
        d = mj.MjData(model)
        for j, jj in enumerate(body.hinges):
            d.qpos[jj["qposadr"]] = q[j]
        mj.mj_forward(model, d)
        sim = make_sim(model, body)
        sim.reset(q=q, qd=np.zeros(body.nj))
        sim._kinematics()
        for i, link in enumerate(body.raw["bodies"]):
            if not link["frame"] or i == 0:
                continue
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, link["body"])
            worst = max(worst,
                        float(np.abs(sim.pos[i] - d.xpos[bid]).max()),
                        float(np.abs(sim.rot[i] - d.xmat[bid].reshape(3, 3)).max()))
    return worst


def check_velocities(model, body, rng) -> float:
    """
    Spatial velocities of every link against MuJoCo's own Jacobians.

    `mj_objectVelocity` is not usable here: it reports the velocity of the
    body's *centre of mass* in a frame that depends on flags, while the solver
    carries the velocity of the body *origin*. The Jacobian at the origin is
    the unambiguous statement: v_world = J qvel, and J is exactly the matrix
    the velocity recursion is supposed to reproduce.
    """
    worst = 0.0
    for _ in range(3):
        q = rng.uniform(body.lo + 1e-3, body.hi - 1e-3)
        qd = rng.uniform(-3, 3, body.nj)
        d = mj.MjData(model)
        for j, jj in enumerate(body.hinges):
            d.qpos[jj["qposadr"]] = q[j]
            d.qvel[jj["dofadr"]] = qd[j]
        mj.mj_forward(model, d)
        sim = make_sim(model, body)
        sim.reset(q=q, qd=qd)
        sim._kinematics()
        sim._velocities()
        for i, link in enumerate(body.raw["bodies"]):
            if not link["frame"] or i == 0:
                continue
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, link["body"])
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mj.mj_jac(model, d, jacp, jacr, d.xpos[bid], bid)
            mine = np.concatenate([sim.rot[i] @ sim.v[i][:3],
                                   sim.rot[i] @ sim.v[i][3:]])
            ref = np.concatenate([jacr @ d.qvel, jacp @ d.qvel])
            worst = max(worst, float(np.abs(mine - ref).max()))
    return worst


def sim_local(sim, i) -> np.ndarray:
    """Spatial velocity of link i in its own frame, MuJoCo's ordering."""
    v = sim.v[i]
    return np.concatenate([v[:3], v[3:]])


def check_inertia_bias(model, body, rng, with_velocity=True):
    """`mj_fullM` is the physical inertia plus the *diagonal* `dof_armature`;
    integrator damping is not part of it. Measured against the real fly this
    matches `FlySim`'s composite inertia to roundoff, on both modes."""
    sc = char_scales(body)
    worst_h = worst_c = 0.0
    for _ in range(3):
        q = rng.uniform(body.lo + 1e-3, body.hi - 1e-3)
        qd = rng.uniform(-3, 3, body.nj) if with_velocity else np.zeros(body.nj)
        d = mj.MjData(model)
        for j, jj in enumerate(body.hinges):
            d.qpos[jj["qposadr"]] = q[j]
            d.qvel[jj["dofadr"]] = qd[j]
        mj.mj_forward(model, d)
        sim = make_sim(model, body)
        sim.reset(q=q, qd=qd)
        sim._kinematics()
        sim._velocities()
        H, C = reference_h_c(sim)
        Hm = full_mass(model, d)
        Cref = np.asarray(d.qfrc_bias)
        if body.root_free:
            off = model.nv - body.nj
            Hm = Hm[off:, off:]
            Cref = Cref[off:]
        worst_h = max(worst_h, relerr(H, Hm, sc["H"]))
        worst_c = max(worst_c, relerr(C, Cref, sc["C"]))
    return worst_h, worst_c


def check_acceleration(model, body, rng, with_velocity=True):
    """
    One step of each, compared as a *velocity*, not as an acceleration.

    MuJoCo applies joint damping implicitly — the velocity update is
    `qd / (1 + dt*d/m)`, which is the whole reason a 1e-8 g tarsus with a
    1e-2 damper is stable at 1e-4 s at all — but `d.qacc` still reports the
    *explicit* acceleration `-d*qd/m`, five orders of magnitude away from what
    the integrator actually used. The solver keeps the implicit form in its
    acceleration, as it must, so the two agree on the velocity after a step and
    cannot agree on the acceleration that produced it. Comparing velocities
    tests the same physics without depending on which of the two the API
    happens to print.
    """
    sc = char_scales(body)
    worst = 0.0
    for _ in range(3):
        q = rng.uniform(body.lo + 1e-3, body.hi - 1e-3)
        qd = rng.uniform(-3, 3, body.nj) if with_velocity else np.zeros(body.nj)
        d = mj.MjData(model)
        for j, jj in enumerate(body.hinges):
            d.qpos[jj["qposadr"]] = q[j]
            d.qvel[jj["dofadr"]] = qd[j]
        mj.mj_forward(model, d)
        sim = make_sim(model, body)
        sim.reset(q=q, qd=qd)
        sim.step(sim.dt, np.zeros(body.nj))
        mj.mj_step(model, d)
        ref = np.asarray(d.qvel)[[j["dofadr"] for j in body.hinges]]
        worst = max(worst, relerr(sim.qd, ref, sc["vel"]))
    return worst


def check_trajectory(model, body, rng, steps=STEPS, with_velocity=True):
    """
    Two integrators on the same body for `steps` steps.

    The single-step row matters more than the long one. A fly whose springs are
    at their rest length is not an equilibrium: over 50 steps (5 ms) the joints
    are already whipping at tens of radians per second, so the two trajectories
    separate by the system's own Lyapunov amplification — measured at ~1.6x per
    step, i.e. 1e10 over the run. The residual after 50 steps is dominated by
    that growth, not by how well the two agree; the 1-step row on the same seed
    measures the agreement itself, and it is at machine precision when the pose
    is near the rest configuration and ~1e-9 relative when the pose is random
    enough to push some dofs to |qdd| ~ 5e5.
    """
    sc = char_scales(body)
    worst_q = worst_qd = 0.0
    worst_root = 0.0
    worst_1q = worst_1qd = 0.0
    for _ in range(2):
        q = rng.uniform(body.lo + 1e-3, body.hi - 1e-3)
        qd = rng.uniform(-3, 3, body.nj) if with_velocity else np.zeros(body.nj)
        d = mj.MjData(model)
        for j, jj in enumerate(body.hinges):
            d.qpos[jj["qposadr"]] = q[j]
            d.qvel[jj["dofadr"]] = qd[j]
        mj.mj_forward(model, d)
        sim = make_sim(model, body)
        sim.reset(q=q, qd=qd)
        for k in range(steps):
            mj.mj_step(model, d)
            sim.step(sim.dt, np.zeros(body.nj))
            if k == 0:
                worst_1q = max(worst_1q, relerr(sim.q, d.qpos[
                    [j["qposadr"] for j in body.hinges]], sc["q"]))
                worst_1qd = max(worst_1qd, relerr(sim.qd, d.qvel[
                    [j["dofadr"] for j in body.hinges]], sc["vel"]))
        worst_q = max(worst_q, relerr(sim.q, d.qpos[
            [j["qposadr"] for j in body.hinges]], sc["q"]))
        if body.root_free:
            scale_p = max(float(np.abs(d.qpos[:3]).max()), 1e-6)
            worst_root = max(worst_root,
                             float(np.abs(sim.root_pos - d.qpos[:3]).max()) / scale_p,
                             float(np.abs(np.abs(sim.root_quat) - np.abs(d.qpos[3:7])).max()),
                             float(np.abs(sim.vel - d.qvel[:3]).max()) / max(
                                 float(np.abs(d.qvel[:3]).max()), 1e-6))
        worst_qd = max(worst_qd, relerr(sim.qd, d.qvel[
            [j["dofadr"] for j in body.hinges]], sc["vel"]))
    return worst_q, worst_qd, worst_root, worst_1q, worst_1qd


def battery(name, xml, rng, tmp: Path, steps=STEPS):
    """
    Every check gets its own generator, seeded from one number per model, so a
    failure can be reproduced on its own instead of only in sequence.
    """
    model, body, _ = asset_from_xml(xml, tmp / f"{name}.xml")
    strip_passive(model, body)
    seed = int(rng.integers(1 << 30))
    fresh = lambda: np.random.default_rng(seed * 7919 + 13)
    print(f"   (seed {seed})")
    rows = []
    rows.append(("kinematics pos+rot", check_kinematics(model, body, fresh()), 1e-12))
    if model.nv:
        rows.append(("velocities", check_velocities(model, body, fresh()), 1e-11))
    for mode, wv in (("static", False), ("moving", True)):
        h, c = check_inertia_bias(model, body, fresh(), wv)
        rows.append((f"inertia H ({mode})", h, 1e-10))
        rows.append((f"bias C ({mode})", c, 1e-10 if not wv else 1e-8))
        rows.append((f"accel qdd ({mode})", check_acceleration(model, body, fresh(), wv),
                     1e-9 if not wv else 1e-6))
    q, qd, root, q1, qd1 = check_trajectory(model, body, fresh(), steps, True)
    rows.append((f"trajectory q (1 step)", q1, 1e-9))
    rows.append((f"trajectory qd (1 step)", qd1, 1e-6))
    rows.append((f"trajectory q ({steps} steps)", q, 1e-9))
    rows.append((f"trajectory qd ({steps} steps)", qd, 1e-6))
    if model.nq > 7 or body.root_free:
        rows.append((f"trajectory root ({steps} steps)", root, 1e-6))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", help="the real body, for the same battery")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", help="write the numbers here")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tmp = Path(tempfile.mkdtemp(prefix="verify_aba_"))
    cases = [("pendulum", PENDULUM), ("chain", CHAIN), ("branch", BRANCH),
             ("multi-joint", MULTI)]

    report = {}
    bad = 0
    for name, xml in cases:
        print(f"\n{name}")
        rows = battery(name, xml, rng, tmp, args.steps)
        report[name] = {}
        for label, value, tol in rows:
            ok = value <= tol
            bad += 0 if ok else 1
            print(f"   {label:26s} {value:9.3e}   tol {tol:7.1e}   "
                  f"{'ok' if ok else 'FAIL'}")
            report[name][label] = {"value": value, "tol": tol, "ok": bool(ok)}

    if args.mjcf:
        path = Path(args.mjcf)
        model = mj.MjModel.from_xml_path(str(path))
        raw = json.loads(json.dumps(_mjcf_asset(model, path)))
        asset = tmp / "fly.json"
        asset.write_text(json.dumps(raw))
        body = fa.FlyBody(asset)
        seed = int(rng.integers(1 << 30))
        fresh = lambda: np.random.default_rng(seed * 7919 + 13)  # noqa: E731
        strip_passive(model, body, passive=True)
        print(f"\nflybody (seed {seed})")
        report["flybody"] = {}
        rows = [("kinematics pos+rot", check_kinematics(model, body, fresh()), 1e-12),
                ("velocities", check_velocities(model, body, fresh()), 1e-11)]
        # passive: the fly's own damping, armature and joint springs are in
        # play on both sides, which is the model the phone will run
        for mode, wv in (("static", False), ("moving", True)):
            h, c = check_inertia_bias(model, body, fresh(), wv)
            rows.append((f"inertia H ({mode}, passive)", h, 1e-10))
            rows.append((f"bias C ({mode}, passive)", c, 1e-10 if not wv else 1e-8))
            rows.append((f"step qd ({mode}, passive)",
                         check_acceleration(model, body, fresh(), wv),
                         1e-9 if not wv else 1e-5))
        q, qd, root, q1, qd1 = check_trajectory(model, body, fresh(), args.steps, True)
        # one step, same seed: the agreement itself
        rows.append((f"trajectory q (1 step, passive)", q1, 1e-6))
        rows.append((f"trajectory qd (1 step, passive)", qd1, 1e-5))
        # and {steps} steps: this row is Lyapunov-limited, not error-limited —
        # a fly with its springs at rest is not an equilibrium, so the two
        # trajectories separate by the amplification of the motion itself. The
        # amplification factor is printed below the table.
        rows.append((f"trajectory q ({args.steps} steps, passive)", q, 1e-3))
        rows.append((f"trajectory qd ({args.steps} steps, passive)", qd, 1e-2))
        rows.append((f"trajectory root ({args.steps} steps, passive)", root, 1e-4))
        grown = (q, qd, root, q1, qd1)
        strip_passive(model, body)
        rows.append(("bias C (moving, bare)", check_inertia_bias(model, body, fresh(), True)[1], 1e-8))
        for mode, wv in (("static", False), ("moving", True)):
            h, c = check_inertia_bias(model, body, rng, wv)
            rows.append((f"inertia H ({mode})", h, 1e-10))
            rows.append((f"bias C ({mode})", c, 1e-10 if not wv else 1e-8))
            rows.append((f"accel qdd ({mode})", check_acceleration(model, body, rng, wv),
                         1e-9 if not wv else 1e-6))
        q, qd, root, q1, qd1 = check_trajectory(model, body, rng, args.steps, True)
        rows.append((f"trajectory q ({args.steps} steps)", q, 1e-9))
        rows.append((f"trajectory qd ({args.steps} steps)", qd, 1e-6))
        rows.append((f"trajectory root ({args.steps} steps)", root, 1e-6))
        for label, value, tol in rows:
            ok = value <= tol
            bad += 0 if ok else 1
            print(f"   {label:26s} {value:9.3e}   tol {tol:7.1e}   "
                  f"{'ok' if ok else 'FAIL'}")
            report["flybody"][label] = {"value": value, "tol": tol, "ok": bool(ok)}

        g = grown
        print(f"   -- passive trajectory amplification over {args.steps} steps: "
              f"q {g[0] / max(g[3], 1e-30):.2e}x, qd {g[1] / max(g[4], 1e-30):.2e}x "
              f"(divergence of the motion, not solver error)")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))

    print("\n" + ("all checks pass" if not bad else f"{bad} check(s) failed"))
    return 1 if bad else 0


def _mjcf_asset(model, path):
    """The full asset for a model already compiled, with the real gravity."""
    name = lambda kind, i: (mj.mj_id2name(model, OBJ[kind], i) or f"<{kind}:{i}>")
    tree = bb.read_tree(model, name)
    raw = {
        "units": {"length": "cm"},
        "gravity": [float(x) for x in model.opt.gravity],
        "timestep_s": float(model.opt.timestep),
        "floor_z": -1e6,
        "stance_root_z": 0.0,
        "bodies": tree["bodies"],
        "joints": tree["joints"],
        "collision": tree["collision"],
        "actuators": [],
        "muscles": {},
        "legs": {},
        "visual": [],
        "meshes": {},
    }
    return raw


if __name__ == "__main__":
    raise SystemExit(main())
