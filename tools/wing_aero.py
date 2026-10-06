#!/usr/bin/env python3
"""
The wings: what they are, what they feel, and what force they make.

Brief item 6. `docs/AUDIT.md` had it PARTIAL — "both wings have joints, muscles
and mesh; `FlyDynamics` carries wing-joint torque like any joint; the connectome
carries the wing motor pools (12+12 power, 12+12 steering, 6+6 tension) as named
groups the cord can already read. **Gap:** no aerodynamic force model and no
wing-beat state." This is the force model and the beat state, and the
measurement that the forces are the right size to hold the animal up.

## What is measured here, from the body model

`data/flybody`'s MJCF, loaded with MuJoCo exactly as `tools/build_body.py`
loads it, gives each wing's real geometry: the mesh triangles give the planform
**area** (the mesh is a closed 3 µm-thick shell, so its triangle areas double
count: the planform is computed by projection, not by summing faces), the
**span** and the chord distribution give the mean chord and the aspect ratio,
and the second moment of the area about the hinge — `∫r² dA`, the number a
flapping wing's force is actually proportional to — is integrated over the same
triangles. The hinge axes are taken into the thorax frame, so which joint
strokes the wing and which one pitches it is read out rather than assumed.

## The force law, and where its coefficients come from

Quasi-steady blade element: the wing is divided into spanwise strips, each strip
feels

    dF = ½ ρ |v|² (C_L(α) ĉ + C_D(α) n̂) c(r) dr

with `v` the strip's velocity **relative to the air**, `α = atan2(−v_n, v_c)`
the angle of attack, and the coefficients the measured *Drosophila* fits from
Dickinson, Lehmann & Sane (1999, *Science* 284:1954):

    C_L(α) = 0.225 + 1.58 sin(2.13α − 7.2°)
    C_D(α) = 1.92 − 1.55 cos(2.13α − 9.8°)

Those are BIOLOGICAL DATA at a Reynolds number of ~100, which is where the tool
measures this wing to be. The rotational-circulation and wake-capture terms of
that paper are **not** modelled: this is the quasi-steady part, and the report
says what that leaves out (it is the difference between lift/weight ≈ 1 and the
1.5–2 a real fly gets from the rotational mechanism).

## The stroke, and what is a labelled constant

The model's wings are **springs with no actuators** (three hinges each, 0.01
stiffness, and the actuator table lists the three wing joints but `build_body.py`
drives only the legs), so a stroke has to come from somewhere. It comes from the
thorax: a fly's flight muscles are stretch-activated and the thorax is a
resonant spring-mass system, so the *frequency* is set by thoracic mechanics
(~200 Hz in *Drosophila*: Lehmann & Dickinson 2005, *J. Exp. Biol.* 208:3075)
and the *amplitude* by how hard the power muscles are driven. The tool models
exactly that and nothing else:

    stroke      φ(t) = Φ · sin(2π f t)          Φ from the measured joint range
    pitch       α(t) = α_mid − α_rot · cos(4π f t)   rotation at each reversal
    amplitude   Φ = Φ_min + a · (Φ_max − Φ_min),  a from the wing motor pool's rate

Every one of those is a labelled constant or a measurement (docs/ASSUMPTIONS.md
#34–#37). The `a → Φ` map is an APPROXIMATION of the thorax's behaviour, and it
is the one place where a *rate* becomes a *kinematic*: everything downstream of
it — the strip forces, the cycle-mean lift, the asymmetry torque — is computed,
not asserted.

    python3 tools/wing_aero.py --probe        # the measurement
    python3 tools/wing_aero.py --gate         # the CI gate
    python3 tools/wing_aero.py --write        # the world the app reads
    python3 tools/wing_aero.py --golden build/wing_golden.json
    python3 tools/wing_aero.py --figure docs/img/wing_aero.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEG = 180.0 / np.pi

DEFAULTS = dict(
    # -- the air and the animal
    rho_g_cm3=1.225e-3,        # BIOLOGICAL DATA: air at 20 °C, 1.225 kg/m³
    body_mass_ug=985.0,        # MEASURED: the body model's own total mass
    g_cm_s2=981.0,             # standard gravity, in cm/s²
    # -- the stroke: the model has no actuator on the wing hinges, so the beat
    #    is the animal's (see the module docstring).
    stroke_hz=200.0,           # BIOLOGICAL DATA: Lehmann & Dickinson 2005
    amplitude_max_deg=82.0,    # MEASURED: the wing_yaw joint's own range, ±82°
    amplitude_full_deg=77.5,   # BIOLOGICAL DATA: the fly's stroke amplitude at
                               # full drive (~155° peak to peak, Lehmann &
                               # Dickinson 2005); the joint allows 164°
    alpha_mid_deg=45.0,        # APPROXIMATION: mid-stroke angle of attack. The
                               # value the tool then *measures* is the one that
                               # makes lift equal weight (--probe prints it)
    alpha_rot_deg=30.0,        # APPROXIMATION: the sweep of α about the middle,
                               # i.e. the wing rotating as the stroke reverses
                               # (Dickinson et al. 1999 measured 90-180° of
                               # rotation per reversal; this is a smooth stand-in)
    # -- the thorax, i.e. rate -> amplitude
    rate_full_hz=60.0,         # APPROXIMATION: the wing power motor pool's rate
                               # at full drive. Set so that the app's own tone
                               # gives a beat that is *below* full amplitude:
                               # the reference, not a claim about the animal
    n_station=12,              # strips along the span
    # -- the two wings' groups
    power_left="motor_wing_power_left",
    power_right="motor_wing_power_right",
    steering_left="motor_wing_steering_left",
    steering_right="motor_wing_steering_right",
)


class Wing:
    """One wing, measured out of the body model."""

    def __init__(self, geom: dict):
        self.name = geom["name"]
        self.area_cm2 = geom["area_cm2"]            # planform
        self.span_cm = geom["span_cm"]              # reach from the hinge
        self.length_cm = geom["length_cm"]          # base to tip
        self.r2dA_cm4 = geom["second_moment_cm4"]   # ∫r² dA about the hinge
        self.chord_cm = geom["chord_cm"]            # mean chord = A / span
        self.mass_ug = geom["mass_ug"]
        self.com_cm = geom["com_cm"]                # hinge -> CoM
        self.stations = geom["stations"]            # (r, chord) along the span
        self.station_area_cm2 = geom["station_area_cm2"]
        self.hinge_cm = geom["hinge_cm"]            # attachment, thorax frame
        self.stroke_axis = geom["stroke_axis"]      # unit, thorax frame
        self.span_axis = geom["span_axis_thorax"]   # unit, thorax frame
        self.plane_normal = geom["plane_normal"]    # unit, thorax frame
        self.mesh_plane_normal = geom["mesh_plane_normal"]   # the rest pose
        # The strips are binned by their distance from the *hinge* (which is
        # where the force model's r²dA is taken, and where the joint is), but a
        # strip's speed is set by its distance from the *stroke axis*, which is
        # that distance times sin θ. If the wing lay in the stroke plane this
        # would be 1 and it would not matter; it does not lie exactly in it, so
        # it is measured rather than assumed.
        a = np.asarray(self.span_axis, float) / max(np.linalg.norm(self.span_axis), 1e-12)
        n = np.asarray(self.stroke_axis, float) / max(np.linalg.norm(self.stroke_axis), 1e-12)
        self.sin_theta = float(np.linalg.norm(a - (a @ n) * n))

    @property
    def aspect_ratio(self) -> float:
        """The wing's length² / area, which is how the aerodynamics literature
        quotes it (2.7-3.0 for a *Drosophila* wing)."""
        return self.length_cm ** 2 / self.area_cm2

    @property
    def r2_hat(self) -> float:
        """√(∫r²dA / A): the radius whose velocity the cycle-mean force is
        equivalent to. 0.5–0.7 R for a fly wing, and measured here."""
        return float(np.sqrt(self.r2dA_cm4 / self.area_cm2))


# Rehearsal hooks, so that `--gate` can be shown to have teeth rather than
# asserted to: each one makes the model deliberately wrong in a specific way,
# and the gate has to catch it. Nothing else in the tool reads them.
REHEARSAL = {"mirror_convention": False, "lift_from": "stroke_plane"}


def wing_asymmetry(wings: dict) -> dict:
    """How far the body model's two wings are from being identical.

    They are not: their planform areas differ by 6.6e-05 and their ∫r²dA by
    7.5e-05, which puts a floor of that order under every left-right number this
    tool reports. It is small — 0.006 % of the flight force, 0.1 % of the
    steering signal — but it is a property of the model and not of the animal,
    so it is measured and written down, and the force model integrates each
    wing's own strips rather than one table for both."""
    l, r = wings["left"], wings["right"]
    return {"area_rel": abs(l.area_cm2 - r.area_cm2) / l.area_cm2,
            "second_moment_rel": abs(l.r2dA_cm4 - r.r2dA_cm4) / l.r2dA_cm4,
            "sin_theta_rel": abs(l.sin_theta - r.sin_theta) / l.sin_theta}


def read_wing(m, name: str, side: str) -> dict:
    """Measure one wing out of the compiled model."""
    import mujoco
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise SystemExit(f"the body model has no {name}")
    R = quat_to_matrix(np.asarray(m.body_quat[b], float))
    # the wing's own frame -> the thorax frame (the wing's parent)
    tri_areas, tri_centroids, tri_normals = [], [], []
    verts_all = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != b or int(m.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        adr, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fadr, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        # The mesh vertices are in the *geom's* frame, not the body's: the
        # geom carries its own rotation as well as its offset, and dropping the
        # rotation puts the wing in the wrong place inside its own body frame.
        # (Measured the honest way — forward kinematics into the world — the
        # wing reaches 2.65 mm from the hinge; without the geom's rotation the
        # same mesh only reached 1.89 mm, and the tool was integrating a wing
        # that does not exist.)
        Rg = quat_to_matrix(np.asarray(m.geom_quat[g], float))
        v = np.array(m.mesh_vert[adr:adr + nv]) @ Rg.T + np.asarray(m.geom_pos[g], float)
        f = np.array(m.mesh_face[fadr:fadr + nf])
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        a = 0.5 * np.linalg.norm(n, axis=1)
        keep = a > 0
        tri_areas.append(a[keep])
        tri_centroids.append(tri[keep].mean(axis=1))
        tri_normals.append(n[keep] / np.maximum(np.linalg.norm(n[keep], axis=1)[:, None], 1e-30))
        verts_all.append(v)
    verts = np.concatenate(verts_all)
    areas = np.concatenate(tri_areas)
    centroids = np.concatenate(tri_centroids)
    normals = np.concatenate(tri_normals)
    # The wing lies in a plane; the direction it is thinnest in is the normal
    # of that plane, and the planform is the wing's shadow on it. The shell is
    # closed, so its triangle areas double count: every weight below is halved
    # once, here, and used consistently — an early version halved it in the
    # planform and not in the second moment, which inflated the forces by
    # exactly 2.
    extent = verts.max(0) - verts.min(0)
    thin = int(np.argmin(extent))
    weight = 0.5 * np.abs(normals[:, thin]) * areas        # sums to the planform
    if thin != int(np.argmin(extent)):
        pass
    planform = float(weight.sum())
    # Every triangle's distance from the hinge, which is the axis the wing
    # turns about and therefore the lever arm of every force on it.
    hinge = np.zeros(3)
    r = np.linalg.norm(centroids - hinge, axis=1)
    second = float((r ** 2 * weight).sum())                # ∫r² dA about the hinge
    r_min, r_max = float(r.min()), float(r.max())
    # The wing's own length, base to tip (not the distance from the hinge: the
    # wing does not start at the hinge). Used for the aspect ratio, and it is
    # the number that has to agree with the animal's (~2.6 mm).
    length = r_max - r_min
    # Strips for the force integration, and their chords: the sum of the strip
    # areas is the planform *by construction* (`--gate` checks that it is, to a
    # percent, because an integration whose strips do not add up to the wing is
    # an integration of some other wing).
    edges = np.linspace(r_min, r_max, DEFAULTS["n_station"] + 1)
    stations = []
    for i in range(DEFAULTS["n_station"]):
        lo, hi = edges[i], edges[i + 1]
        sel = (r >= lo) & (r < hi) if i < DEFAULTS["n_station"] - 1 else (r >= lo) & (r <= hi)
        a_i = float(weight[sel].sum())
        dr = hi - lo
        stations.append({"r_cm": float(0.5 * (lo + hi)), "chord_cm": float(a_i / dr),
                         "area_cm2": a_i})
    com = R @ np.asarray(m.body_ipos[b], float)
    axis_local = np.asarray(m.jnt_axis[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, f"wing_yaw_{side}")], float)
    # the wing's own long axis, in the wing's frame: from the hinge out to the
    # farthest material (that direction is the span, and the chord is the other)
    span_axis = centroids[r == r_max].mean(axis=0) - hinge
    span_axis = span_axis / max(np.linalg.norm(span_axis), 1e-12)
    # The wing's plane orientation, in the thorax frame. The mesh's own winding
    # is not trusted for the *sign*, so it is fixed below by the rule that a
    # positive angle of attack makes lift point up out of the stroke plane (the
    # same convention the Dickinson et al. coefficients were measured in). The
    # wing's span axis is turned into the thorax frame with the same body
    # rotation the stroke axis and the centre of mass already use.
    normals_thorax = (R @ normals.T).T
    span_axis_thorax = R @ span_axis
    stroke_axis_thorax = R @ axis_local
    # The mesh's own plane normal: informative about the body model's rest pose
    # (it comes out at -z, i.e. the wing at rest is a horizontal membrane), but
    # *not* the plane the wing pitches away from in flight. The angle of attack
    # is measured from the stroke plane: at α = 0 the chord lies along the
    # velocity, and the velocity lies in the stroke plane, so the α = 0 normal
    # is the stroke plane's normal. Using the rest plane instead flew the wing
    # edge-on (n̂·v̂ = 0.994) and pointed the lift at the ground.
    # The rest plane's normal, taken from which way the wing is thinnest (a
    # single triangle's winding would be too fragile to read a sign off, and a
    # closed shell's normals cancel anyway when they are summed). The sign is
    # canonicalised, because a plane has no preferred normal.
    mesh_plane = np.zeros(3)
    mesh_plane[thin] = 1.0
    mesh_plane = R @ mesh_plane
    if mesh_plane[2] < 0:
        mesh_plane = -mesh_plane
    a = span_axis_thorax / max(np.linalg.norm(span_axis_thorax), 1e-12)
    ns = stroke_axis_thorax / max(np.linalg.norm(stroke_axis_thorax), 1e-12)
    plane = ns - (ns @ a) * a
    plane = plane / max(np.linalg.norm(plane), 1e-12)
    if REHEARSAL["lift_from"] == "rest_plane":
        # The wrong plane to pitch the wing away from: the body model's rest
        # plane instead of the stroke plane. Flies the wing edge-on.
        plane = mesh_plane / max(np.linalg.norm(mesh_plane), 1e-12)
    # Which way "up" is: the mesh cannot say (the wing is a closed shell and its
    # winding is not a sign of anything), so it is fixed the same way for both
    # wings by the one convention that matches the animal — a positive angle of
    # attack makes the lift point up out of the stroke plane. Measured this way
    # the two wings' normals come out as mirror images of each other, which is
    # the check that the rule did not quietly pick opposite signs.
    if plane[2] < 0:
        plane = -plane
    return {
        "name": name,
        "area_cm2": planform,
        "span_cm": r_max,                  # reach from the hinge
        "length_cm": length,               # base to tip
        "second_moment_cm4": second,
        "chord_cm": planform / max(length, 1e-12),
        "mass_ug": float(m.body_mass[b] * 1e6),
        "com_cm": float(np.linalg.norm(com)),
        "com_hat": com / max(np.linalg.norm(com), 1e-12),
        "span_axis": [float(v) for v in span_axis],
        "span_axis_thorax": [float(v) for v in span_axis_thorax],
        "stroke_axis_thorax": [float(v) for v in stroke_axis_thorax],
        "plane_normal": [float(v) for v in plane],
        "mesh_plane_normal": [float(v) for v in mesh_plane],
        "hinge_cm": [float(v) for v in m.body_pos[b]],
        "stations": stations,
        "station_area_cm2": float(sum(s["area_cm2"] for s in stations)),
        "r_min_cm": r_min,
        "stroke_axis": stroke_axis_thorax,
    }


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rodrigues(vec, axis, angle_rad: float) -> np.ndarray:
    """`vec` turned about `axis` by `angle_rad`, the right-hand way."""
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    v = np.asarray(vec, float)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v * c + np.cross(a, v) * s + a * (a @ v) * (1.0 - c)


def clean(obj):
    """JSON-able: the report carries numpy vectors now, and a report that cannot
    be written is a report nobody can check."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def mirror_y(vec) -> np.ndarray:
    """The left wing's vector seen as the right wing's: the fly's own mirror."""
    v = np.asarray(vec, float)
    return np.array([v[0], -v[1], v[2]])


class Aero:
    """The quasi-steady force law, and the beat that drives it."""

    def __init__(self, wing: Wing, **kw):
        self.p = dict(DEFAULTS)
        self.p.update({k: v for k, v in kw.items() if v is not None})
        self.wing = wing

    # -- the coefficients -----------------------------------------------------
    @staticmethod
    def coefficients(alpha_deg: float) -> tuple[float, float]:
        """Dickinson, Lehmann & Sane 1999's measured *Drosophila* fits."""
        a = np.radians(alpha_deg)
        cl = 0.225 + 1.58 * np.sin(2.13 * a - np.radians(7.2))
        cd = 1.92 - 1.55 * np.cos(2.13 * a - np.radians(9.8))
        return float(cl), float(cd)

    # -- the beat -------------------------------------------------------------
    def amplitude_deg(self, activation: float) -> float:
        a = float(np.clip(activation, 0.0, 1.0))
        lo = 0.0                                    # a still wing is a folded wing
        hi = min(self.p["amplitude_full_deg"], self.p["amplitude_max_deg"])
        return lo + a * (hi - lo)

    def stroke_deg(self, t_ms: float, amplitude_deg: float) -> float:
        w = 2 * np.pi * self.p["stroke_hz"] * t_ms / 1000.0
        return amplitude_deg * np.sin(w)

    def alpha_deg(self, t_ms: float) -> float:
        w = 2 * np.pi * self.p["stroke_hz"] * t_ms / 1000.0
        return self.p["alpha_mid_deg"] - self.p["alpha_rot_deg"] * np.cos(2 * w)

    # -- the forces, in three dimensions --------------------------------------
    def wing_frame(self, wing: Wing, phi_rad: float, alpha_deg: float):
        """Where one wing's strips are, and which way they are moving and
        pushing, at stroke angle `phi_rad` and angle of attack `alpha_deg`.

        `r̂` is the span direction (the wing swept to `phi_rad` about its stroke
        axis), `v̂` is the direction the strips travel when the stroke angle is
        increasing, and the lift direction is the wing's normal with its
        component along the velocity removed — which for a wing that pitches
        about its span is the stroke plane's normal, and is why the flapping
        force has a fore-aft component at all when the stroke plane is inclined.
        """
        r_hat = rodrigues(wing.span_axis, wing.stroke_axis, phi_rad)
        v_hat = np.cross(wing.stroke_axis, r_hat)
        v_hat = v_hat / max(np.linalg.norm(v_hat), 1e-12)
        n_hat = rodrigues(wing.plane_normal, r_hat, np.radians(alpha_deg))
        lift_hat = n_hat - (n_hat @ v_hat) * v_hat
        return r_hat, v_hat, lift_hat / max(np.linalg.norm(lift_hat), 1e-12)

    def wing_wrench(self, wing: Wing, phi_rad: float, omega_rad_s: float,
                    alpha_deg: float):
        """The force and the moment about the thorax origin that one wing makes
        at one instant of the beat (dyn, dyn·cm), plus the drag magnitude (dyn).

        The moment is taken about the thorax origin, not about the wing's own
        hinge, because that is where the animal's centre of mass is: the hinge
        offset is exactly what turns a left-right force asymmetry into yaw.
        """
        r_hat, v_hat, lift_hat = self.wing_frame(wing, phi_rad, alpha_deg)
        cl, cd = self.coefficients(alpha_deg)
        hinge = np.asarray(wing.hinge_cm, float)
        sgn = 1.0 if omega_rad_s >= 0 else -1.0
        force = np.zeros(3)
        moment = np.zeros(3)
        drag = 0.0
        for s in wing.stations:
            r = s["r_cm"]
            v = abs(omega_rad_s) * r * wing.sin_theta      # cm/s, about the axis
            q = 0.5 * self.p["rho_g_cm3"] * v * v * s["area_cm2"]   # dyn
            f = q * cl * lift_hat - q * cd * sgn * v_hat
            force += f
            moment += np.cross(hinge + r * r_hat, f)
            drag += q * cd
        return force, moment, drag

    def up_direction(self) -> np.ndarray:
        """The pair's common lift direction: the two wings' plane normals are
        mirrors of each other, so this is the direction the *animal's* flight
        force points while the body is held still."""
        n = np.asarray(self.wing.plane_normal, float)
        return n / max(np.linalg.norm(n), 1e-12)

    # -- the scalar strip force: kept for the golden table --------------------
    def strip_forces(self, omega_rad_s: float, alpha_deg: float) -> dict:
        """The forces on the strips for a wing turning at `omega_rad_s` about
        its stroke axis with angle of attack `alpha_deg`.

        The strip's velocity is `ω r sinθ` — its radius about the stroke *axis*;
        the chordwise part sees `v cos α` and the normal part `v sin α` only
        because α is *defined* as the wing's angle to that velocity — so that
        geometry is in the coefficients, not in the velocity.
        Returns the integrated force in the stroke plane: lift (perpendicular to
        the stroke velocity, i.e. what holds the animal up when the stroke plane
        is horizontal) and drag (along the velocity, i.e. what a steering
        asymmetry pushes sideways).
        """
        cl, cd = self.coefficients(alpha_deg)
        rho = self.p["rho_g_cm3"] * 1e-3            # g/cm³ -> kg/cm³? no: keep g/cm³
        rho = self.p["rho_g_cm3"]
        lift = drag = 0.0
        for s in self.wing.stations:
            # about the *stroke axis*, which is what the strip actually turns
            # about: r sinθ, not r. This table fed the golden rows the Swift port
            # is checked against, and the port had the sinθ in it — so the two
            # disagreed by sinθ² = 0.9903 and CI said so, row by row, at the 1 %
            # level. The port was right; this function was the one that was
            # wrong, and `wing_wrench` (which the beats come from) always had it.
            v = abs(omega_rad_s) * s["r_cm"] * self.wing.sin_theta   # cm/s
            q = 0.5 * rho * v * v * s["area_cm2"]   # g·cm/s²  (1 g·cm/s² = 1 dyn)
            lift += q * cl
            drag += q * cd
        return {"lift_dyn": lift, "drag_dyn": drag, "cl": cl, "cd": cd,
                "alpha_deg": alpha_deg, "omega_rad_s": omega_rad_s}

    def inertial_force_dyn(self, amplitude_deg: float) -> float:
        """The force the wing's own mass needs at mid-stroke: `m ω² r`. For a fly
        this is *larger* than the aerodynamic force, which is why the flight
        muscles are as big as they are and why the thorax can be a resonator."""
        w = 2 * np.pi * self.p["stroke_hz"] * np.radians(amplitude_deg)
        r_cm = self.wing.com_cm
        return float(self.wing.mass_ug * 1e-6 * w * w * r_cm)

    def weight_dyn(self) -> float:
        return float(self.p["body_mass_ug"] * 1e-6 * self.p["g_cm_s2"])

    # -- a whole beat ---------------------------------------------------------
    def cycle(self, activation: float, samples: int = 720, steer: float = 0.0,
              right: Wing | None = None) -> dict:
        """One beat at `activation`, with an optional steering asymmetry.

        `steer` adds to the left wing's activation and subtracts from the
        right's, which changes the two stroke *amplitudes* — the asymmetry the
        steering muscles make, and the one a fly uses to yaw.
        """
        return self.cycle_pair(activation + 0.5 * steer,
                               activation - 0.5 * steer, samples=samples, right=right,
                               steer=steer)

    def cycle_pair(self, activation_left: float, activation_right: float,
                   samples: int = 720, right: Wing | None = None,
                   steer: float | None = None) -> dict:
        """One beat with each wing driven *by its own activation*, which is what
        the app actually has: two sets of motor pools, two muscles, two wings.

        Both wings' forces are integrated in the body's own frame, so the report
        contains not just how much force the animal makes but which way it pushes
        and how hard it turns.
        """
        activation = 0.5 * (activation_left + activation_right)
        if steer is None:
            steer = activation_left - activation_right
        amp_l = self.amplitude_deg(activation_left)
        amp_r = self.amplitude_deg(activation_right)
        right = self.wing if right is None else right
        f = self.p["stroke_hz"]
        period = 1000.0 / f
        up_l = self.up_direction()
        up_r = np.asarray(right.plane_normal, float)
        up_r = up_r / max(np.linalg.norm(up_r), 1e-12)
        total = np.zeros(3)
        moment = np.zeros(3)
        lift_l = lift_r = 0.0
        thrust = side = drag_cost = 0.0
        peak = 0.0
        peak_yaw = 0.0
        for i in range(samples):
            t_ms = period * i / samples
            w = 2 * np.pi * f * t_ms / 1000.0
            alpha = self.alpha_deg(t_ms)
            phi_l = np.radians(amp_l) * np.sin(w)
            phi_r = np.radians(amp_r) * np.sin(w)
            om_l = 2 * np.pi * f * np.radians(amp_l) * np.cos(w)
            om_r = 2 * np.pi * f * np.radians(amp_r) * np.cos(w)
            f_l, m_l, d_l = self.wing_wrench(self.wing, phi_l, om_l, alpha)
            f_r, m_r, d_r = self.wing_wrench(right, phi_r, om_r, alpha)
            pair = f_l + f_r
            total += pair / samples
            moment += (m_l + m_r) / samples
            lift_l += (f_l @ up_l) / samples
            lift_r += (f_r @ up_r) / samples
            thrust += pair[0] / samples
            side += pair[1] / samples
            drag_cost += (d_l + d_r) / samples
            peak = max(peak, float(np.linalg.norm(pair)))
            peak_yaw = max(peak_yaw, abs(float(np.cross(
                np.asarray(self.wing.hinge_cm, float), pair)[2])))
        return {
            "activation": activation, "steer": steer,
            "activation_left": activation_left, "activation_right": activation_right,
            "amplitude_deg": 0.5 * (amp_l + amp_r),
            "amplitude_left_deg": amp_l, "amplitude_right_deg": amp_r,
            # the flight force: each wing's own aerodynamic force along its own
            # stroke-plane normal. This is the number to compare with the
            # animal's weight and with Dickinson's coefficients.
            "lift_left_dyn": lift_l, "lift_right_dyn": lift_r,
            "flight_force_dyn": lift_l + lift_r,
            # and the same force resolved in the body's frame, where the animal
            # has to live: what it can hold up depends on its body attitude.
            "vertical_dyn": float(total[2]),
            "thrust_dyn": thrust, "side_dyn": side,
            "force_vector_dyn": [float(v) for v in total],
            "drag_cost_dyn": drag_cost,      # mean of |drag|, both wings: the cost
            "yaw_moment_dyn_cm": float(moment[2]),
            "roll_moment_dyn_cm": float(moment[0]),
            "pitch_moment_dyn_cm": float(moment[1]),
            "peak_force_dyn": peak,
            "peak_yaw_moment_dyn_cm": peak_yaw,
        }

    def attitude_deg(self) -> float:
        """How far the wings' force is from vertical in this rest pose: the
        body pitch the animal has to hold for the wingbeat to carry it. The
        stroke plane is a property of the thorax, so this is a *posture*, and a
        hovering fly holds a nose-up one (BIOLOGICAL DATA, docs/ASSUMPTIONS.md
        #34: 30-60° in measured hovering *Drosophila*)."""
        return float(np.degrees(np.arccos(
            np.clip(abs(self.up_direction()[2]), 0.0, 1.0))))


def load_cord(bin_path: pathlib.Path, meta_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("pool_probe", HERE / "pool_probe.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    return pp, pp.load(bin_path, meta_path)


def group_size(c: dict, name: str) -> int:
    g = c["groups"].get(name)
    return int(g["count"]) if g else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", type=pathlib.Path,
                    default=ROOT / "data/flybody/flybody-main/flybody/fruitfly/assets/fruitfly.xml")
    ap.add_argument("--bin", type=pathlib.Path, default=ROOT / "build" / "flybanc.bin")
    ap.add_argument("--meta", type=pathlib.Path, default=ROOT / "build" / "flybanc_meta.json")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--figure", type=pathlib.Path, default=None)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--golden", type=pathlib.Path, default=None)
    ap.add_argument("--ms", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--gain", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tone", type=float, default=2.5,
                    help="the drive the wing motor pools get from the cord")
    ap.add_argument("--steer-tone", type=float, default=0.9,
                    help="the differential the steering pools get: left gets "
                         "tone + this, right tone - this")
    ap.add_argument("--convention", choices=("measured", "mirrored"),
                    default="measured",
                    help="rehearsal: 'mirrored' uses the wrong stroke-axis "
                         "convention for the right wing and must fail the gate")
    ap.add_argument("--lift-from", choices=("stroke_plane", "rest_plane"),
                    default="stroke_plane",
                    help="rehearsal: 'rest_plane' pitches the wing away from the "
                         "body model's rest plane instead of the stroke plane "
                         "and must fail the gate")
    ap.add_argument("--activation", type=float, default=1.0,
                    help="activation used for the airframe measurements, before "
                         "any connectome is involved")
    args = ap.parse_args()

    REHEARSAL["mirror_convention"] = args.convention == "mirrored"
    REHEARSAL["lift_from"] = args.lift_from
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(args.body))
    if REHEARSAL["mirror_convention"]:
        # The wrong joint convention, for rehearsal — put into the *body model*
        # rather than into this tool, because that is where the convention
        # lives: mirroring one wing's stroke joint instead of leaving the pair
        # antiparallel makes the two wings sweep against each other. (A rehearsal
        # that only flipped a vector inside the tool was a silent no-op: the
        # drag cancels over a cycle whichever way each wing sweeps, so the model
        # was wrong in a way that made no difference to any number. The failure
        # this check guards is in the model, so the rehearsal has to be too.)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wing_yaw_right")
        m.jnt_axis[j] = -np.asarray(m.jnt_axis[j], float)
    wings = {side: Wing(read_wing(m, f"wing_{side}", side)) for side in ("left", "right")}
    out: dict = {"wings": {s: {k: v for k, v in w.__dict__.items()} for s, w in wings.items()}}
    problems = []

    # ---- what the model says the wings are ---------------------------------
    print("the wings, measured from the body model")
    print(f"  {args.body}")
    for side in ("left", "right"):
        w = wings[side]
        print(f"  {side}: area {w.area_cm2 * 100:.3f} mm², span {w.span_cm * 10:.2f} mm, "
              f"mean chord {w.chord_cm * 10:.2f} mm, AR {w.aspect_ratio:.2f}")
        print(f"        mass {w.mass_ug:.2f} ug, hinge->CoM {w.com_cm * 10:.3f} mm, "
              f"∫r²dA {w.r2dA_cm4:.3e} cm⁴, r̂₂ {w.r2_hat * 10:.3f} mm "
              f"({100 * w.r2_hat / w.span_cm:.0f}% of the span)")
        print(f"        stroke axis (thorax frame) {np.round(w.stroke_axis, 4)}"
              f" -> the stroke plane is {np.degrees(np.arccos(abs(w.stroke_axis[2]))):.1f}° "
              f"off the horizontal")
        print(f"        the wing's rest plane normal {np.round(w.mesh_plane_normal, 3)} "
              f"(the body model's rest pose has the wings out flat; the pose that "
              f"matters is the stroke posture, which the fly sets itself)")
    if abs(wings["left"].area_cm2 - wings["right"].area_cm2) > 0.02 * wings["left"].area_cm2:
        problems.append("the two wings are not the same size in the body model")
    # The strips have to add up to the wing, or the force model is integrating
    # some other wing. (This check is here because the first version of the
    # binning dropped everything outside a wrong radius range and quietly
    # integrated a quarter of the wing.)
    for side in ("left", "right"):
        w = wings[side]
        covered = w.__dict__["station_area_cm2"] / w.area_cm2
        print(f"    the strips cover {100 * covered:.2f}% of the {side} wing's area")
        if not 0.98 <= covered <= 1.02:
            problems.append(f"the {side} wing's strips cover {100 * covered:.1f}% "
                            f"of its area: the force model is not integrating "
                            f"this wing")

    # The two wings have to be mirrors, or every yaw moment this tool reports
    # is partly the model talking to itself. Measured, not assumed.
    nl, nr = np.asarray(wings["left"].plane_normal), np.asarray(wings["right"].plane_normal)
    sl, sr = np.asarray(wings["left"].span_axis), np.asarray(wings["right"].span_axis)
    mirrored = np.linalg.norm(nr - np.array([nl[0], -nl[1], nl[2]]))
    print(f"    the two wings' lift directions are mirrors to {mirrored:.2e}, "
          f"and their spans to "
          f"{np.linalg.norm(sr - np.array([sl[0], -sl[1], sl[2]])):.2e}")
    asym = wing_asymmetry(wings)
    print(f"    ... and their *areas* differ by {100 * asym['area_rel']:.3f}%, "
          f"their ∫r²dA by {100 * asym['second_moment_rel']:.3f}%: the body model's "
          f"two wings are not one wing used twice. Each wing is integrated with "
          f"its own strips, and this is the floor under every left-right number "
          f"below (the port carries both tables for the same reason).")
    # 1e-5, not 1e-9: the body model's two wing meshes are mirrors to ~2e-6,
    # which is a property of the model and the floor under every yaw moment this
    # tool can resolve (it is measured against the steering signal below).
    if mirrored > 1e-5 or nl[2] < 0 or nr[2] < 0:
        problems.append("the two wings' lift directions are not mirrors pointing "
                        "up: the pair would make a phantom yaw moment")
    # And the body model's own kinematics have to sweep both wings the same way
    # for the same joint angle — the wings beat in phase; the *halteres* are the
    # pair that beats in antiphase (that is item 7, and it is a different pair).
    import mujoco as _mj
    probe_data = _mj.MjData(m)
    tips = {}
    for side in ("left", "right"):
        j = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_JOINT, f"wing_yaw_{side}")
        gid = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_GEOM, f"wing_{side}_membrane")
        body = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_BODY, f"wing_{side}")
        adr, nv = m.mesh_vertadr[m.geom_dataid[gid]], m.mesh_vertnum[m.geom_dataid[gid]]
        verts = np.array(m.mesh_vert[adr:adr + nv])
        moved = []
        for angle in (0.0, 0.2):
            probe_data.qpos[m.jnt_qposadr[j]] = angle
            _mj.mj_kinematics(m, probe_data)
            Xg = probe_data.geom_xpos[gid]
            Rw = probe_data.geom_xmat[gid].reshape(3, 3)
            world = verts @ Rw.T + Xg
            moved.append(world[np.argmax(np.linalg.norm(world - probe_data.xpos[body], axis=1))])
        tips[side] = moved[1] - moved[0]
        probe_data.qpos[m.jnt_qposadr[j]] = 0.0
    _mj.mj_forward(m, probe_data)
    same_x = np.sign(tips["left"][0]) == np.sign(tips["right"][0])
    print(f"    at +0.2 rad of stroke the two tips move x "
          f"{tips['left'][0]:+.4f} / {tips['right'][0]:+.4f} cm "
          f"({'both the same way: in phase' if same_x else 'OPPOSITE WAYS'})")
    if not same_x:
        problems.append("the two wings sweep in opposite directions at the same "
                        "joint angle: the app would drive them against each other")

    # ---- the beat, and what it makes ---------------------------------------
    aero = {side: Aero(w) for side, w in wings.items()}
    weight = aero["left"].weight_dyn()
    print("")
    print(f"the beat: {aero['left'].p['stroke_hz']:g} Hz, "
          f"alpha {aero['left'].p['alpha_mid_deg']:g} ± "
          f"{aero['left'].p['alpha_rot_deg']:g}°, air "
          f"{aero['left'].p['rho_g_cm3'] * 1000:.3f} kg/m³")
    w = wings["left"]
    print(f"  weight of the animal: {weight:.2f} dyn ({weight * 10:.2f} uN)")
    amp = aero["left"].amplitude_deg(args.activation)
    # The cycle mean of |cos| is 2/π, so the mean strip speed is
    # (2/π)·ωΦ·r̂₂, and the strips' distance from the stroke *axis* is sin θ of
    # their distance from the hinge. The first version of this line was missing
    # the 2/π and reported half the animal's wing speed — and therefore half its
    # Reynolds number.
    v_mean = ((2 / np.pi) * 2 * np.pi * aero["left"].p["stroke_hz"]
              * np.radians(amp) * w.r2_hat * w.sin_theta)
    rec_chord = v_mean * w.chord_cm / 0.15      # ν = 0.15 cm²/s (air, 20 °C)
    rec_len = v_mean * w.length_cm / 0.15
    print(f"  at activation {args.activation:g}: stroke ±{amp:g}°, "
          f"mean wing speed {v_mean * 10:.0f} mm/s")
    print(f"  Reynolds {rec_chord:.0f} on the mean chord, {rec_len:.0f} on the "
          f"wing's length (the animal sits in the 100-200 flapping regime on one "
          f"of those two, depending on whose definition is used)")
    beat = aero["left"].cycle(args.activation, right=wings["right"])
    lift_total = beat["flight_force_dyn"]
    print(f"  cycle-mean flight force (each wing's force along its own stroke "
          f"plane's normal): {beat['lift_left_dyn']:.3f} + "
          f"{beat['lift_right_dyn']:.3f} = {lift_total:.3f} dyn "
          f"-> **lift/weight {lift_total / weight:.2f}**")
    print(f"  the same force in the body's frame: "
          f"{beat['force_vector_dyn'][0]:+.3f} forward, "
          f"{beat['force_vector_dyn'][1]:+.3f} sideways, "
          f"{beat['force_vector_dyn'][2]:+.3f} up")
    print(f"  the wing force is {aero['left'].attitude_deg():.1f}° off vertical "
          f"in this pose: that is the **nose-up attitude the thorax has to hold** "
          f"for the beat to carry the animal (a hovering fly does hold one)")
    print(f"  mean |drag| through the beat, both wings: "
          f"{beat['drag_cost_dyn']:.3f} dyn — this is the cost, not a thrust: the "
          f"*net* drag of a symmetric beat is zero, which is exactly why a fly "
          f"that is not turning does not accelerate")
    if abs(beat["lift_left_dyn"] - beat["lift_right_dyn"]) > 1e-3 * abs(lift_total):
        problems.append(f"the two wings make different flight force at the same "
                        f"activation ({beat['lift_left_dyn']:.4f} vs "
                        f"{beat['lift_right_dyn']:.4f} dyn): they are not mirrors")
    print(f"  a symmetric beat's side force {beat['side_dyn']:+.1e} dyn and yaw "
          f"moment {beat['yaw_moment_dyn_cm']:+.1e} dyn·cm: zero by mirror "
          f"symmetry, and measured here to be zero — which is how we know the two "
          f"wings are wired the right way round")
    inertial = aero["left"].inertial_force_dyn(amp)
    print(f"  the wing's own inertial force at mid-stroke: {inertial:.2f} dyn "
          f"({inertial / weight:.1f}x the animal's weight, "
          f"{inertial / max(lift_total, 1e-9):.1f}x the lift it makes)")
    print(f"    CAVEAT: that is with the body model's wing mass of "
          f"{w.mass_ug:.1f} ug. A *Drosophila* wing weighs ~1 ug (BIOLOGICAL "
          f"DATA), so this number is ~{w.mass_ug:.0f}x too big and the model's "
          f"wing is a lumped placeholder (docs/ASSUMPTIONS.md #35). With a real "
          f"wing mass the inertial force is ~{inertial / w.mass_ug:.2f} dyn — "
          f"comparable with the ~{0.5 * lift_total:.2f} dyn of lift per wing, "
          f"which is the animal's actual situation: the flight muscles have to "
          f"pay for the wing's inertia and the air about equally.")
    out["beat"] = {k: (float(v) if np.isscalar(v) else v) for k, v in beat.items()}
    out["weight_dyn"] = weight
    out["reynolds_chord"] = rec_chord
    out["reynolds_length"] = rec_len
    out["attitude_deg"] = aero["left"].attitude_deg()
    out["mean_wing_speed_cm_s"] = v_mean

    # The angle of attack that makes lift equal weight: the tool *solves* for
    # it rather than asserting a value, and it is the number to compare with the
    # animal's measured mid-stroke α (~45-50°).
    def lift_at(alpha, activation=1.0):
        saved = aero["left"].p["alpha_mid_deg"]
        aero["left"].p["alpha_mid_deg"] = alpha
        b = aero["left"].cycle(activation, right=wings["right"])
        aero["left"].p["alpha_mid_deg"] = saved
        return b["flight_force_dyn"]

    lo, hi = 5.0, 80.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if lift_at(mid) < weight:
            lo = mid
        else:
            hi = mid
    alpha_hover = 0.5 * (lo + hi)
    print(f"  the angle of attack at which the flight force exactly equals the "
          f"animal's weight: **{alpha_hover:.1f}°**, against the ~45-50° "
          f"Dickinson, Lehmann & Sane measured at mid-stroke in the animal's "
          f"own hovering beat — the strongest evidence in this tool that the "
          f"force model is measuring the animal and not a toy")
    out["alpha_hover_deg"] = float(alpha_hover)
    if not 20.0 < alpha_hover < 70.0:
        problems.append(f"the hovering angle of attack is {alpha_hover:.1f}°, "
                        f"which is not an angle a fly wing can be at")
    if lift_total < 0.8 * weight:
        problems.append(f"at full activation the wings make {lift_total:.2f} dyn "
                        f"of lift against a weight of {weight:.2f} dyn: the "
                        f"animal cannot get off the ground")
    if aero["left"].attitude_deg() > 60.0:
        problems.append(f"the wing force is {aero['left'].attitude_deg():.1f}° off "
                        f"vertical when the body is level: no posture a fly can hold "
                        f"would turn that into weight support")

    # The steering asymmetry: an amplitude difference makes a yaw moment
    # (unequal drag) and a roll moment (unequal lift), about the hinges.
    print("")
    print("")
    print("steering: what an amplitude asymmetry makes")
    print(f"  the motors turn the *amplitude* of one wing up and the other down; "
          f"the moment is taken about the thorax origin, so the wings' own hinge "
          f"offset is in it")
    print(f"  {'steer':>6} {'amp L/R (deg)':>15} {'flight force L/R (dyn)':>23} "
          f"{'yaw moment (dyn·cm)':>20} {'roll moment (dyn·cm)':>21}")
    rows = []
    for steer in (-1.0, -0.5, 0.0, 0.5, 1.0):
        b = aero["left"].cycle(args.activation, steer=steer, right=wings["right"])
        rows.append({"steer": steer, "yaw_moment": b["yaw_moment_dyn_cm"],
                     "roll_moment": b["roll_moment_dyn_cm"], **b})
        print(f"  {steer:>6.1f} {b['amplitude_left_deg']:>7.1f}/{b['amplitude_right_deg']:<7.1f} "
              f"{b['lift_left_dyn']:>10.3f}/{b['lift_right_dyn']:<11.3f} "
              f"{b['yaw_moment_dyn_cm']:>+20.5f} {b['roll_moment_dyn_cm']:>+21.5f}")
    out["steering"] = rows
    sym = [r for r in rows if r["steer"] == 0.0][0]
    print(f"  (in N·m, the unit a tether rig reports: the largest yaw moment here "
          f"is {max(abs(r['yaw_moment_dyn_cm']) for r in rows) * 1e-7:.2e} N·m)")
    yaw_pos = [r["yaw_moment"] for r in rows if r["steer"] > 0]
    yaw_neg = [r["yaw_moment"] for r in rows if r["steer"] < 0]
    mean_pos = float(np.mean(yaw_pos))
    mean_neg = float(np.mean(yaw_neg))
    if mean_pos * mean_neg >= 0:
        problems.append(f"a left-right amplitude asymmetry does not reverse the "
                        f"yaw moment ({mean_pos:+.5f} vs {mean_neg:+.5f} dyn·cm): "
                        f"the animal could not steer")
    # And the sign has to be the physical one: the wing that is driven harder
    # sweeps harder and pushes its side of the body forward, so the nose goes
    # the *other* way. A model that got this backwards would fly in circles the
    # wrong way, and it is the kind of sign that is easy to "fix" downstream.
    if mean_pos > 0:
        problems.append(f"driving the left wing harder yaws the animal to the "
                        f"left ({mean_pos:+.5f} dyn·cm): the sign is inverted "
                        f"against the measured geometry")
    # A symmetric beat must not turn the animal. It does not, exactly, because
    # the body model's two wing meshes are mirrors only to ~2e-6 — so the test
    # is not "is it zero" but "is the leak small against the signal a real
    # asymmetry makes", which is the thing that would actually mislead a reader.
    signal = max(abs(r["yaw_moment_dyn_cm"]) for r in rows if r["steer"] != 0.0)
    leak = abs(sym["yaw_moment_dyn_cm"])
    print(f"  the mirror-symmetry floor: a symmetric beat leaks {leak:.2e} dyn·cm "
          f"of yaw, against {signal:.2e} dyn·cm at full asymmetry "
          f"({100 * leak / max(signal, 1e-12):.2f}% — the body model's two wing "
          f"meshes are mirrors to ~2e-6, and that is all this floor is)")
    if leak > 0.02 * signal:
        problems.append(f"a symmetric beat makes {leak:.2e} dyn·cm of yaw moment "
                        f"against a {signal:.2e} dyn·cm signal: too much of the "
                        f"steering is the model talking to itself")
    # A real bound, not a guess: no force can make a moment larger than the
    # force times the wing's own reach from the axis it acts about.
    reach = max(float(np.linalg.norm(np.asarray(wings[s].hinge_cm, float)))
                + wings[s].span_cm for s in ("left", "right"))
    for r in rows:
        bound = r["peak_force_dyn"] * reach
        if abs(r["yaw_moment_dyn_cm"]) > bound:
            problems.append(f"the yaw moment at steer {r['steer']:+.1f} exceeds "
                            f"force x reach: the arithmetic is wrong")


    # ---- the connectome drives the wings -----------------------------------
    probe = None
    if args.probe or args.gate:
        pp, c = load_cord(args.bin, args.meta)
        names = [aero["left"].p["power_left"], aero["left"].p["power_right"],
                 aero["left"].p["steering_left"], aero["left"].p["steering_right"]]
        sizes = {n: group_size(c, n) for n in names}
        print("")
        print("the wing motor pools, in the packed connectome")
        print("  " + ", ".join(f"{n} {sizes[n]}" for n in names))
        if any(s == 0 for s in sizes.values()):
            problems.append("a wing motor pool is not in the binary — the wings "
                            "would have nothing to drive them")
            probe = {"pools": sizes}
        else:
            ids = {n: c["members"](n) for n in names}

            def rates(drive: dict, ms=None, warm=None, seed=None) -> dict:
                net = pp.LIF(c, args.gain, seed=args.seed if seed is None else seed)
                ms = args.ms if ms is None else ms
                warm = args.warmup if warm is None else warm
                hits = {n: 0 for n in names}
                for t in range(warm + ms):
                    for n in names:
                        net.drive(n, drive.get(n, 0.0))
                    net.drive("sensory_vision", 1.0)   # the app's operating point
                    net.step()
                    if t >= warm:
                        for n in names:
                            hits[n] += int(np.isin(net.last, ids[n]).sum())
                return {n: hits[n] / sizes[n] * 1000.0 / ms for n in names}

            tone = args.tone
            steady = {n: tone for n in names}
            base = rates(steady)
            print(f"  with the cord's tone ({tone:g}) on all four pools, "
                  f"{args.ms} ms after {args.warmup} ms of warm-up:")
            for n in names:
                print(f"    {n:<28} {base[n]:>7.2f} Hz")
            # rate -> activation: how hard the thorax is driven. The reference
            # is the pool's own rate at full drive, measured here, not assumed.
            full = rates({n: 4.0 for n in names})
            print(f"  at drive 4.0 (the reference for 'full'): "
                  + ", ".join(f"{n.split('_')[-2]}_{n.split('_')[-1]} {full[n]:.1f} Hz"
                              for n in names))
            # Each pool against *its own* full-drive rate. The two power pools
            # come out 173.7 and 212.3 Hz at full drive — the connectome is not
            # left-right symmetric — so a single shared reference would hold an
            # amplitude asymmetry open and the animal would fly in a slow circle
            # with no command to do so. (This is the same class of finding as
            # the haltere pools' -24.7 Hz offset in item 7: a measured property
            # of the connectome, wired in explicitly rather than cancelled.)
            act_l = base[names[0]] / max(full[names[0]], 1e-9)
            act_r = base[names[1]] / max(full[names[1]], 1e-9)
            beat_app = aero["left"].cycle_pair(min(1.0, act_l), min(1.0, act_r),
                                               right=wings["right"])
            lift_app = beat_app["flight_force_dyn"]
            print(f"  each power pool against its own full-drive rate: "
                  f"left {act_l:.3f}, right {act_r:.3f} (the two pools run at "
                  f"{full[names[0]]:.1f} and {full[names[1]]:.1f} Hz at full "
                  f"drive, so one shared reference would fly the animal in a "
                  f"circle with no command: each pool is calibrated against "
                  f"itself)")
            print(f"  at that operating point: stroke ±{beat_app['amplitude_left_deg']:.1f}"
                  f"/{beat_app['amplitude_right_deg']:.1f}°, flight force "
                  f"{lift_app:.3f} dyn -> lift/weight {lift_app / weight:.2f}, "
                  f"and an uncommanded yaw of "
                  f"{beat_app['yaw_moment_dyn_cm']:+.2e} dyn·cm left over "
                  f"(the map from rate to amplitude is an APPROXIMATION of the "
                  f"thorax, docs/ASSUMPTIONS.md #36)")
            out["probe"] = {"pools": sizes, "rates_tone": base, "rates_full": full,
                            "activation_left": act_l, "activation_right": act_r,
                            "lift_at_tone_dyn": lift_app,
                            "yaw_uncommanded_dyn_cm": beat_app["yaw_moment_dyn_cm"],
                            "lift_over_weight_at_tone": lift_app / weight}

            # the steering asymmetry, through the connectome as well
            diff = {n: tone for n in names}
            diff[names[2]] = tone + args.steer_tone
            diff[names[3]] = max(0.0, tone - args.steer_tone)
            steered = rates(diff)
            l_rate = steered[names[2]] - base[names[2]]
            r_rate = steered[names[3]] - base[names[3]]
            # The steering pool's rate buys stroke *amplitude* on top of the
            # power pool's. One full unit of steering rate is one full unit of
            # activation (APPROXIMATION: the thorax's map, #36), each pool
            # against its own reference, and the activation is clipped to the
            # range a muscle works in.
            act_l_s = float(np.clip(act_l + l_rate / max(full[names[2]], 1e-9), 0, 1))
            act_r_s = float(np.clip(act_r + r_rate / max(full[names[3]], 1e-9), 0, 1))
            steer_act = act_l_s - act_r_s
            b_steer = aero["left"].cycle_pair(act_l_s, act_r_s, right=wings["right"])
            print(f"  the steering pools with a {args.steer_tone:+.2f} differential: "
                  f"left {l_rate:+.2f} Hz ({base[names[2]]:.2f} -> "
                  f"{steered[names[2]]:.2f}), right {r_rate:+.2f} Hz -> stroke "
                  f"{b_steer['amplitude_left_deg']:.1f}° / "
                  f"{b_steer['amplitude_right_deg']:.1f}° -> yaw moment "
                  f"{b_steer['yaw_moment_dyn_cm']:+.5f} dyn·cm, roll "
                  f"{b_steer['roll_moment_dyn_cm']:+.5f} dyn·cm")
            # The *sign* is a physical consequence, not a convention to pick:
            # an amplitude asymmetry makes that wing push forward harder, so the
            # animal's nose goes the other way. Drive the left pool harder and
            # the animal yaws to the right — which is what an already-turned-up
            # left wing does to a fly, and what the app must not "fix" by
            # flipping a sign somewhere downstream.
            print(f"    (+z is up, +x is forward: driving the *left* wing harder "
                  f"yaws the animal's nose to the *right*, because the left side "
                  f"of the body is being pushed forward)")
            out["probe"]["steering"] = {"left_rate_delta": l_rate,
                                        "right_rate_delta": r_rate,
                                        "asymmetry": steer_act,
                                        "yaw_moment": b_steer["yaw_moment_dyn_cm"],
                                        "roll_moment": b_steer["roll_moment_dyn_cm"]}
            if abs(b_steer["yaw_moment_dyn_cm"]) < 1e-5:
                problems.append(f"a differential on the steering pools turns the "
                                f"animal by only "
                                f"{b_steer['yaw_moment_dyn_cm']:+.2e} dyn·cm: the "
                                f"connectome cannot steer it")
            if b_steer["yaw_moment_dyn_cm"] > 0:
                problems.append(f"driving the left steering pool harder yaws the "
                                f"animal to the *left* "
                                f"({b_steer['yaw_moment_dyn_cm']:+.5f} dyn·cm): "
                                f"the wing model's sign is inverted against the "
                                f"measured geometry")

        # ---- the reflex arc the connectome already contains -----------------
        # The animal's yaw reflex is not a controller: the haltere afferents
        # synapse *directly* on the wing steering motor neurons (item 7 measured
        # them: 10 and 8 synapses on `motor_wing_steering_right/left`), so a turn
        # reaches the wings through the cord's own wiring. If that arc is wired
        # the way a reflex needs — the wing that is opened opposes the turn —
        # then no gain in this tool has to be invented, and the yaw damping is
        # the connectome's. One LIF run per condition answers it.
        if args.probe or args.gate:
            rings = []
            for label, dps in (("still", (0.0, 0.0, 0.0)),
                               ("yaw +400", (0.0, 0.0, 400.0)),
                               ("yaw -400", (0.0, 0.0, -400.0))):
                net = pp.LIF(c, args.gain, seed=args.seed)
                # the haltere drives, exactly as `tools/haltere_gyro.py` makes
                # them, from the haltere block in the manifest
                hal = json.loads((ROOT / "world" / "world.json").read_text()).get("haltere", {})
                hl = hal.get("sensitivity_left") or [0.0, 0.0, 0.0]
                hr = hal.get("sensitivity_right") or [0.0, 0.0, 0.0]
                bias = hal.get("bias", 3.0)
                gain = hal.get("gain", 2.0)
                ref = hal.get("reference_dps", 100.0)
                drives = {}
                if max(abs(v) for v in dps) >= hal.get("min_dps", 1.0):
                    drives[hal.get("group_left", "sensory_haltere_left")] = bias + gain * float(
                        np.dot(dps, hl)) / ref
                    drives[hal.get("group_right", "sensory_haltere_right")] = bias + gain * float(
                        np.dot(dps, hr)) / ref
                hits = {k: 0 for k in names + ["sensory_haltere_left", "sensory_haltere_right"]}
                sizes_all = {k: (sizes.get(k) or group_size(c, k)) for k in hits}
                for t_ms in range(args.warmup + args.ms):
                    for k in hits:
                        net.drive(k, drives.get(k, (tone if k in names else 0.0)))
                    net.drive("sensory_vision", 1.0)
                    net.step()
                    if t_ms >= args.warmup:
                        for k in hits:
                            if sizes_all[k] and k in ("motor_wing_steering_left",
                                                      "motor_wing_steering_right"):
                                ids_k = c["members"](k)
                                hits[k] += int(np.isin(net.last, ids_k).sum())
                rings.append((label, {k: hits[k] / max(sizes_all[k], 1) * 1000.0 / args.ms
                                      for k in ("motor_wing_steering_left",
                                                "motor_wing_steering_right")}))
            print("")
            print("the reflex arc, through the connectome's own wiring (no controller):")
            for label, r in rings:
                print(f"  {label:<9} wing steering pools: left {r['motor_wing_steering_left']:.2f} Hz, "
                      f"right {r['motor_wing_steering_right']:.2f} Hz")
            base = rings[0][1]
            yaws = {lab: r for lab, r in rings[1:]}
            for label, r in yaws.items():
                dl = r["motor_wing_steering_left"] - base["motor_wing_steering_left"]
                dr = r["motor_wing_steering_right"] - base["motor_wing_steering_right"]
                print(f"  {label}: the steering pools move {dl:+.2f} / {dr:+.2f} Hz "
                      f"relative to still")
            out["reflex_arc"] = {lab: r for lab, r in rings}
            turn_pos = rings[1][1]["motor_wing_steering_left"] - rings[1][1]["motor_wing_steering_right"]
            turn_neg = rings[2][1]["motor_wing_steering_left"] - rings[2][1]["motor_wing_steering_right"]
            rest_diff = base["motor_wing_steering_left"] - base["motor_wing_steering_right"]
            print(f"  the pools' own left-right difference: still {rest_diff:+.2f} Hz, "
                  f"yaw +400 {turn_pos:+.2f} Hz, yaw -400 {turn_neg:+.2f} Hz")
            if turn_pos == rest_diff and turn_neg == rest_diff:
                problems.append("a yaw of 400 deg/s does not move the wing steering "
                                "pools at all: the reflex arc from the halteres is "
                                "not wired in this connectome slice")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=130)
        # 1: the wing's chord distribution, as measured
        st = wings["left"].stations
        axes[0].bar([s["r_cm"] * 10 for s in st], [s["chord_cm"] * 10 for s in st],
                    width=0.8 * (wings["left"].span_cm * 10 / len(st)))
        axes[0].set_xlabel("distance from the hinge (mm)")
        axes[0].set_ylabel("chord (mm)")
        axes[0].set_title(f"the wing, measured: {wings['left'].area_cm2 * 100:.2f} mm², "
                          f"AR {wings['left'].aspect_ratio:.1f}", fontsize=8)
        # 2: the force through one beat, both wings, in the body's frame
        a = aero["left"]
        period = 1000.0 / a.p["stroke_hz"]
        ts = np.linspace(0, period, 400)
        up_f, fwd_f, drag_f = [], [], []
        for t_ms in ts:
            w_ = 2 * np.pi * a.p["stroke_hz"] * t_ms / 1000.0
            alpha = a.alpha_deg(t_ms)
            amp_ = a.amplitude_deg(1.0)
            phi = np.radians(amp_) * np.sin(w_)
            om = 2 * np.pi * a.p["stroke_hz"] * np.radians(amp_) * np.cos(w_)
            fl, _, dl = a.wing_wrench(a.wing, phi, om, alpha)
            fr, _, dr = a.wing_wrench(wings["right"], phi, om, alpha)
            pair = fl + fr
            up_f.append(pair[2]); fwd_f.append(pair[0]); drag_f.append(dl + dr)
        axes[1].plot(ts * 1000 / period, up_f, label="the wings hold up")
        axes[1].plot(ts * 1000 / period, drag_f, label="|drag| (the cost)")
        axes[1].plot(ts * 1000 / period, fwd_f, label="forward (the stroke plane's tilt)")
        axes[1].axhline(0, lw=0.5, color="k")
        axes[1].axhline(weight, lw=0.7, ls="--", color="grey",
                        label="the animal's weight")
        axes[1].set_xlabel("stroke cycle (%)")
        axes[1].set_ylabel("force, both wings (dyn)")
        axes[1].set_title("one beat, measured in the body's frame", fontsize=8)
        axes[1].legend(fontsize=6)
        # 3: the yaw moment against the asymmetry
        axes[2].plot([r["steer"] for r in rows], [r["yaw_moment"] for r in rows], "o-")
        axes[2].axhline(0, lw=0.5, color="k")
        axes[2].set_xlabel("steering asymmetry")
        axes[2].set_ylabel("yaw moment (dyn·cm)")
        axes[2].set_title("an asymmetrical beat turns the animal", fontsize=8)
        for ax in axes:
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure)
        plt.close(fig)
        print(f"\n  wrote {args.figure}")

    if args.write:
        path = ROOT / "world" / "world.json"
        manifest = json.loads(path.read_text())
        lw = wings["left"]
        rw = wings["right"]
        manifest["wings"] = {
            "area_cm2": lw.area_cm2,
            "span_cm": lw.span_cm,
            "chord_cm": lw.chord_cm,
            "second_moment_cm4": lw.r2dA_cm4,
            "r2_hat_cm": lw.r2_hat,
            "sin_theta": lw.sin_theta,
            "mass_ug": lw.mass_ug,
            "com_cm": lw.com_cm,
            "hinge_left_cm": wings["left"].hinge_cm,
            "hinge_right_cm": wings["right"].hinge_cm,
            # The three vectors the force model needs: which way each wing's
            # strips travel, which way its lift points, and where its strips are.
            "span_axis_left": [float(v) for v in wings["left"].span_axis],
            "span_axis_right": [float(v) for v in wings["right"].span_axis],
            "lift_dir_left": [float(v) for v in wings["left"].plane_normal],
            "lift_dir_right": [float(v) for v in wings["right"].plane_normal],
            "stroke_axis_left": [float(v) for v in wings["left"].stroke_axis],
            "stroke_axis_right": [float(v) for v in wings["right"].stroke_axis],
            # What the wings can do, as measured here — the numbers the app's
            # flight behaviour is allowed to depend on.
            "weight_dyn": weight,
            "flight_force_dyn": lift_total,
            "lift_over_weight": lift_total / weight,
            "attitude_deg": aero["left"].attitude_deg(),
            "alpha_hover_deg": alpha_hover,
            "mean_wing_speed_cm_s": v_mean,
            "reynolds_chord": rec_chord,
            "drag_cost_dyn": beat["drag_cost_dyn"],
            "yaw_moment_full_dyn_cm": max(abs(r["yaw_moment_dyn_cm"])
                                          for r in out["steering"]),
            "stroke_hz": aero["left"].p["stroke_hz"],
            "amplitude_full_deg": aero["left"].p["amplitude_full_deg"],
            "alpha_mid_deg": aero["left"].p["alpha_mid_deg"],
            "alpha_rot_deg": aero["left"].p["alpha_rot_deg"],
            "rho_kg_m3": aero["left"].p["rho_g_cm3"] * 1000,
            "body_mass_ug": aero["left"].p["body_mass_ug"],
            "rate_full_hz": aero["left"].p["rate_full_hz"],
            # The app needs the two power pools' own full-drive rates, because
            # the pools are not symmetric (173.7 vs 212.3 Hz measured): one
            # shared reference would hold an amplitude asymmetry open. They come
            # from the probe, so `--write` says so rather than inventing them.
            "rate_reference_hz": (out.get("probe", {}).get("rates_full") or None),
            "rate_tone_hz": (out.get("probe", {}).get("rates_tone") or None),
            "mass_ug_source": ("ENGINEERING PLACEHOLDER: the body model's own "
                               "wing mass; a real Drosophila wing is ~1 ug "
                               "(docs/ASSUMPTIONS.md #35)"),
            "stations": [{"r_cm": s["r_cm"], "chord_cm": s["chord_cm"],
                          "area_cm2": s["area_cm2"]} for s in lw.stations],
            # The right wing's own strips: the two wings in the body model are
            # *not* identical — their planform areas differ by 6.6e-05 and their
            # ∫r²dA by 7.5e-05 — so a model that integrates one table for both
            # wings is off by ~3e-05 of the flight force on the right wing, and
            # a symmetric beat then leaves that much yaw. Measured, not
            # assumed, and carried rather than shared.
            "stations_right": [{"r_cm": s["r_cm"], "chord_cm": s["chord_cm"],
                                "area_cm2": s["area_cm2"]} for s in rw.stations],
            "wing_asymmetry": wing_asymmetry(wings),
            "power_left": aero["left"].p["power_left"],
            "power_right": aero["left"].p["power_right"],
            "steering_left": aero["left"].p["steering_left"],
            "steering_right": aero["left"].p["steering_right"],
        }
        path.write_text(json.dumps(manifest, indent=1))
        if not out.get("probe"):
            print(f"    NOTE: the reference rates are null because no probe ran: "
                  f"run `--probe --write` together so the app gets the pools' own "
                  f"measured rates (world.json 'rate_reference_hz')")
        print(f"  wrote the wings block into {path}")

    if args.golden:
        # What the Swift port has to reproduce, in three layers, so a port that
        # gets any one of them wrong fails a test that says which: the wing's
        # measured geometry, the force law at a strip, and whole beats driven by
        # the two wings' own activations (which is the shape the app drives them
        # in — two pools, two muscles, no made-up average).
        lw = wings["left"]
        beats = []
        for al, ar in ((1.0, 1.0), (1.0, 0.75), (1.0, 0.5), (1.0, 0.0),
                       (0.75, 0.75), (0.5, 1.0), (0.0, 1.0), (0.4, 0.4), (0.0, 0.0)):
            b = aero["left"].cycle_pair(al, ar, right=wings["right"])
            beats.append({k: b[k] for k in (
                "activation_left", "activation_right", "amplitude_left_deg",
                "amplitude_right_deg", "lift_left_dyn", "lift_right_dyn",
                "flight_force_dyn", "thrust_dyn", "vertical_dyn", "side_dyn",
                "yaw_moment_dyn_cm", "roll_moment_dyn_cm", "pitch_moment_dyn_cm",
                "drag_cost_dyn")})
        strips = []
        for omega in (0.0, 50.0, 300.0, 1697.0, -1697.0):
            for alpha in (0.0, 22.5, 45.0, 67.5, 90.0):
                f = aero["left"].strip_forces(omega, alpha)
                strips.append({"omega_rad_s": omega, "alpha_deg": alpha,
                               "cl": f["cl"], "cd": f["cd"],
                               "lift_dyn": f["lift_dyn"], "drag_dyn": f["drag_dyn"]})
        rw = wings["right"]
        golden = {
            "wing": {
                "name": lw.name,
                "area_cm2": lw.area_cm2, "span_cm": lw.span_cm,
                "length_cm": lw.length_cm, "chord_cm": lw.chord_cm,
                "second_moment_cm4": lw.r2dA_cm4, "r2_hat_cm": lw.r2_hat,
                "mass_ug": lw.mass_ug, "com_cm": lw.com_cm,
                "sin_theta": lw.sin_theta,
                "stroke_axis": [float(v) for v in lw.stroke_axis],
                "span_axis": [float(v) for v in lw.span_axis],
                "lift_dir": [float(v) for v in lw.plane_normal],
                "hinge_cm": lw.hinge_cm,
                "stations": [{"r_cm": s["r_cm"], "chord_cm": s["chord_cm"],
                              "area_cm2": s["area_cm2"]} for s in lw.stations],
                # The right wing as *measured*, not mirrored from the left:
                # the two meshes are mirrors only to ~2e-6, and that error has
                # to stay where the gate can measure it instead of being copied
                # into the port, where it would look like a physics defect.
                "right": {
                    "stroke_axis": [float(v) for v in rw.stroke_axis],
                    "span_axis": [float(v) for v in rw.span_axis],
                    "lift_dir": [float(v) for v in rw.plane_normal],
                    "hinge_cm": rw.hinge_cm,
                    "sin_theta": rw.sin_theta,
                    "area_cm2": rw.area_cm2,
                    "stations": [{"r_cm": s["r_cm"], "chord_cm": s["chord_cm"],
                                  "area_cm2": s["area_cm2"]} for s in rw.stations],
                },
            },
            "law": {"stroke_hz": aero["left"].p["stroke_hz"],
                    "amplitude_full_deg": aero["left"].p["amplitude_full_deg"],
                    "alpha_mid_deg": aero["left"].p["alpha_mid_deg"],
                    "alpha_rot_deg": aero["left"].p["alpha_rot_deg"],
                    "rho_g_cm3": aero["left"].p["rho_g_cm3"],
                    "body_mass_ug": aero["left"].p["body_mass_ug"]},
            "attitude_deg": aero["left"].attitude_deg(),
            "weight_dyn": weight,
            "wing_asymmetry": wing_asymmetry(wings),
            "strip_force": strips,
            "beat": beats,
        }
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        args.golden.write_text(json.dumps(golden, indent=1))
        print(f"  wrote {args.golden}: the wing, {len(strips)} strip-force rows, "
              f"{len(beats)} beats")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(clean(out), indent=1))
        print(f"  wrote {args.json}")

    if args.gate:
        print("")
        for pr in problems:
            print(f"  FAIL {pr}")
        if problems:
            return 1
        print("  ok   the wings are measured, the beat makes lift of the animal's "
              "own weight, and an asymmetry steers it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
