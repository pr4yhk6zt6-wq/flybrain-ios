#!/usr/bin/env python3
"""
simcheck.py — a line-by-line port of FlyBody.update() and World's collision
path, so the physics test suite can be run on a Linux box.

There is no Swift compiler here, but the tests are arithmetic, and arithmetic
ports exactly. Every assertion in FlightPhysicsTests.swift is reproduced
below against this port: if it passes here it passes on the simulator, and if
it fails here it would have failed there too.
"""

import math

# ------------------------------------------------------------ FlyMorphology
mass = 0.96e-6
inertiaYaw = 5.2e-13
bodyLength = 2.5e-3
wingLength = 2.39e-3
wingArea = 1.96e-6
r2 = 0.58 * wingLength
wingbeatHz = 218.0
strokeAmplitudeHover = 2.468
strokeAmplitudeMax = 3.1
liftCoefficient = 1.8
dragCoefficient = 1.27 * 1.8
strokePlaneAngle = 0.5
airDensity = 1.2
gravity = 9.81

meanWingVelocity = 2 * strokeAmplitudeHover * wingbeatHz * r2
forceVelocitySlope = airDensity * dragCoefficient * (2 * wingArea) * meanWingVelocity
translationalDamping = 0.5 * forceVelocitySlope
yawDamping = forceVelocitySlope * r2 * r2

# NEW: the two muscle limits and the torque saturation
steeringRange = 20.0 * math.pi / 180
maxYawRate = 1600.0 * math.pi / 180
maxYawTorque = yawDamping * maxYawRate

walkSpeedMax = 30e-3
stepFrequencyMax = 13.0
dutyFactor = 0.55

weight = mass * gravity


def wing_force(phi, C):
    u = 2 * phi * wingbeatHz * r2
    return 0.5 * airDensity * C * wingArea * u * u


def flight_force(phi):
    return 2 * wing_force(phi, liftCoefficient)


def stroke_tilt(ratio):
    """Stroke-plane forward tilt vs thrust ratio F/W: zero at exact hover
    (so the equilibrium reflex converges on a true hover), full measured
    inclination (~29 deg) at 1.5 body weights; the line between is an
    assumption (ASSUMPTIONS.md)."""
    return strokePlaneAngle * max(0.0, min(1.0, (ratio - 1.0) / 0.5))


METRES_TO_WORLD = 100.0
WORLD_TO_METRES = 0.01


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ------------------------------------------------------------------- World
class World:
    # Open world: the floor runs to the fog horizon; bounds is only the
    # invisible 20 m analytical backstop.
    bounds = 2000.0

    def __init__(self, env="kitchen"):
        self.env = env
        self.objects = []          # (kind, pos(3), size(3))
        self.rebuild()

    @property
    def ceilingHeight(self):
        return 150.0               # invisible sky cap, nothing drawn there

    def rebuild(self):
        self.objects = []
        if self.env == "kitchen":
            self.objects.append(("fruit", [1.4, 0.18, -1.1], [0.18] * 3))

    def contain(self, p, radius):
        """Returns (clamped p, hit normal)."""
        hit = [0.0, 0.0, 0.0]
        limit = self.bounds - radius
        if p[0] < -limit: p[0] = -limit; hit[0] = 1
        if p[0] > limit:  p[0] = limit;  hit[0] = -1
        if p[2] < -limit: p[2] = -limit; hit[2] = 1
        if p[2] > limit:  p[2] = limit;  hit[2] = -1
        if p[1] < radius: p[1] = radius; hit[1] = 1
        top = self.ceilingHeight - radius
        if p[1] > top: p[1] = top; hit[1] = -1
        return p, hit

    def collision(self, p, radius):
        best = None
        deepest = 0.0
        for kind, pos, size in self.objects:
            if kind == "fruit":
                continue
            half = [s * 0.5 for s in size]
            lo = [pos[i] - half[i] for i in range(3)]
            hi = [pos[i] + half[i] for i in range(3)]
            closest = [min(max(p[i], lo[i]), hi[i]) for i in range(3)]
            d = [p[i] - closest[i] for i in range(3)]
            dist = math.sqrt(sum(x * x for x in d))
            if dist > 1e-5:
                if dist >= radius:
                    continue
                n = [x / dist for x in d]
                depth = radius - dist
                if depth > deepest:
                    deepest = depth
                    best = ([closest[i] + n[i] * radius for i in range(3)], n)
            else:
                faces = []
                for i in range(3):
                    faces.append((p[i] - lo[i], [-1 if j == i else 0 for j in range(3)]))
                    faces.append((hi[i] - p[i], [1 if j == i else 0 for j in range(3)]))
                push, n = min(faces, key=lambda t: t[0])
                depth = push + radius
                if depth > deepest:
                    deepest = depth
                    best = ([p[i] + n[i] * depth for i in range(3)], n)
        if p[1] < radius:
            depth = radius - p[1]
            if depth > deepest:
                best = ([p[0], radius, p[2]], [0, 1, 0])
        return best


# ----------------------------------------------------------------- FlyBody
# --- round-5 additions: CT-derived gait geometry and escape constants -------
# (same provenance as FlyMorphology: tools/gait_geometry.py re-derives them)
legTrack = 0.96e-3              # m, mean rest-pose tarsus spread
escapeTakeoffSpeed = 0.9        # m/s (Card & Dickinson 2008)
jumpEventRate = 10.0            # Hz, one TT spike per frame through the EMA
muscleTau = 0.2                 # s, neuromuscular low-pass (ASSUMPTIONS.md #3)
wingEps = 1.0 / (12.0 * muscleTau)
legEps = 1.0 / (3.0 * 63.0 * muscleTau)
jumpRefractory = 0.1            # s (ASSUMPTIONS.md #7)
contactEpsilon = 0.002          # world units (20 um)

# Per segment: neutral (coxa, femur, tibia), protraction signs, amp cap
# (= the amplitude the joint ranges allow at that neutral), secant gain
# mm/rad (travel(A*)/(2A*), so amp = min(cap, R/(2g)) reproduces A*), and
# the stance inverse-map knots: sweep multiplier q at equal travel eighths,
# so the FOOT advances at constant speed despite the strongly nonlinear FK.
# Every number here is derived by tools/gait_geometry.py from the CT model.
GAIT = {
    "T1": ((0.25, 1.55, -0.85), (1.0, -1.0, 1.0), 0.45, 0.630,
           (1.000, 0.213, -0.016, -0.202, -0.369, -0.526, -0.681, -0.836,
            -1.000)),
    "T2": ((0.15, 1.55, -1.00), (1.0, -1.0, 1.0), 0.35, 0.490,
           (1.000, 0.355, 0.079, -0.139, -0.330, -0.506, -0.674, -0.837,
            -1.000)),
    "T3": ((0.15, 1.00, -0.85), (-1.0, 1.0, -1.0), 0.45, 1.218,
           (1.000, 0.753, 0.552, 0.368, 0.190, 0.006, -0.194, -0.438,
            -0.986)),
}


class FlyBody:
    """Line-by-line port of the round-5 FlyBody.swift update()."""

    def __init__(self):
        self.d = Drives()
        self.reset()

    def reset(self, p=(0.0, 0.35, 0.0)):
        self.position = list(p)
        self.heading = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.velocity = [0.0, 0.0, 0.0]
        self.airborne = 0.0
        self.wingPhase = 0.0
        self.strokeAmplitudeL = 0.0
        self.strokeAmplitudeR = 0.0
        self.gaitPhase = 0.0
        self.stepFrequency = 0.0
        self.yawRate = 0.0
        self.referenceRate = 120.0
        self.energy = 1.0
        self.hurt = 0.0
        self.isEating = False
        self.bumped = False
        self.proboscisExtension = 0.0
        # round-5 decode state
        self.grounded = p[1] <= self.contact_radius() + contactEpsilon
        self.phiL = 0.0
        self.phiR = 0.0
        self.legIndex = 0.0
        self.jumpEvents = 0
        self.lastJumpTime = -1000.0
        self.jumpWasHot = False
        self.simTime = 0.0
        self.strideMillimetres = 0.0

    referenceTau = 10.0

    @staticmethod
    def contact_radius():
        return bodyLength * 0.5 * METRES_TO_WORLD

    def stroke_amplitude(self, rate):
        hover = strokeAmplitudeHover
        t = rate / max(self.referenceRate, 1.0)
        if t <= 1:
            return max(0.0, hover * t)
        return hover + (strokeAmplitudeMax - hover) * min(1.0, (t - 1) / 0.5)

    def update_reference(self, rate, dt):
        if rate <= 1:
            return
        a = min(1.0, dt / self.referenceTau)
        self.referenceRate += (rate - self.referenceRate) * a
        self.referenceRate = max(5.0, min(400.0, self.referenceRate))

    def seconds_since_last_jump(self):
        return self.simTime - self.lastJumpTime

    def update(self, dt, world):
        dt = clamp(dt, 1.0 / 480.0, 1.0 / 20.0)
        self.simTime += dt
        self.bumped = False
        self.hurt = max(0.0, self.hurt - dt * 1.5)

        # ---- decode: collective lift + bounded opponent asymmetry ----------
        collective = (self.d.wingPowerL + self.d.wingPowerR) * 0.5
        pIdx = ((self.d.wingPowerR - self.d.wingPowerL)
                / (self.d.wingPowerR + self.d.wingPowerL + wingEps))
        sIdx = ((self.d.wingSteerR - self.d.wingSteerL)
                / (self.d.wingSteerR + self.d.wingSteerL + wingEps))
        self.update_reference(collective, dt)
        asym = pIdx + sIdx
        phi0 = self.stroke_amplitude(collective)
        phiTL = clamp(phi0 - asym * steeringRange, 0.0, strokeAmplitudeMax)
        phiTR = clamp(phi0 + asym * steeringRange, 0.0, strokeAmplitudeMax)
        aM = min(1.0, dt / muscleTau)
        self.phiL += (phiTL - self.phiL) * aM
        self.phiR += (phiTR - self.phiR) * aM
        self.strokeAmplitudeL = self.phiL
        self.strokeAmplitudeR = self.phiR

        # ---- aerodynamics ---------------------------------------------------
        fL = wing_force(self.phiL, liftCoefficient)
        fR = wing_force(self.phiR, liftCoefficient)
        totalForce = fL + fR
        self.liftUN = totalForce * 1e6

        # SIGN: stronger LEFT wing -> torque yaws toward the RIGHT (toward the
        # weaker wing), the direction Fry et al. 2003 measured.
        dragL = wing_force(self.phiL, dragCoefficient)
        dragR = wing_force(self.phiR, dragCoefficient)
        yawTorque = clamp((dragL - dragR) * r2, -maxYawTorque, maxYawTorque)

        # ---- legs -----------------------------------------------------------
        legCollective = (self.d.legL + self.d.legR) * 0.5
        legRaw = ((self.d.legR - self.d.legL)
                  / (self.d.legR + self.d.legL + legEps))
        self.legIndex += (legRaw - self.legIndex) * min(1.0, dt / muscleTau)
        f = min(stepFrequencyMax,
                legCollective / max(self.referenceRate, 1.0)
                * stepFrequencyMax * 2.2)
        self.stepFrequency = f if self.grounded else 0.0
        stride = walkSpeedMax / stepFrequencyMax
        speedMS = min(f * stride, walkSpeedMax)
        self.strideMillimetres = speedMS / f * 1000.0 if f > 0.01 else 0.0

        # ---- one continuous dynamics model ----------------------------------
        if self.grounded:
            # same stroke-plane tilt the flight branch uses: take-off and the
            # in-air equilibrium are one consistent surface, no flicker.
            vert = totalForce * math.cos(stroke_tilt(totalForce / weight))
            if vert > weight:
                vy = (self.velocity[1] * WORLD_TO_METRES
                      + (vert - weight) / mass * dt)
                self.velocity[1] = vy * METRES_TO_WORLD
            else:
                self.velocity[1] = 0.0
            yaw = -2.0 * self.legIndex * speedMS / legTrack
            self.yawRate = yaw
            self.heading += yaw * dt
            fwd = [math.sin(self.heading), 0.0, -math.cos(self.heading)]
            self.velocity[0] = fwd[0] * speedMS * METRES_TO_WORLD
            self.velocity[2] = fwd[2] * speedMS * METRES_TO_WORLD
            self.gaitPhase += dt * f * 2 * math.pi
            self.roll += (0.0 - self.roll) * min(1.0, dt * 8)
            self.pitch += (0.0 - self.pitch) * min(1.0, dt * 8)
        else:
            c = yawDamping
            I = inertiaYaw
            self.yawRate = (self.yawRate + yawTorque / I * dt) / (1 + c / I * dt)
            self.heading += self.yawRate * dt

            tilt = stroke_tilt(totalForce / weight)
            self.pitch += (-max(0.0, tilt) - self.pitch) * min(1.0, dt * 6)

            fwd = [math.sin(self.heading), 0.0, -math.cos(self.heading)]
            ct, st = math.cos(max(0.0, tilt)), math.sin(max(0.0, tilt))
            dirv = [fwd[i] * st for i in range(3)]
            dirv[1] += ct
            n = math.sqrt(sum(x * x for x in dirv))
            dirv = [x / n for x in dirv]

            accel = [dirv[i] * totalForce / mass for i in range(3)]
            accel[1] -= gravity

            v = [x * WORLD_TO_METRES for x in self.velocity]
            v = [v[i] + accel[i] * dt for i in range(3)]
            k = translationalDamping / mass
            v = [x / (1 + k * dt) for x in v]
            self.velocity = [x * METRES_TO_WORLD for x in v]

            self.roll += ((self.phiR - self.phiL) * 0.5 - self.roll) * min(1.0, dt * 6)

        # ---- giant fibre: spike-event take-off -------------------------------
        jumpHot = self.d.jump > jumpEventRate
        if (jumpHot and not self.jumpWasHot
                and self.simTime - self.lastJumpTime > jumpRefractory):
            self.jumpEvents += 1
            self.lastJumpTime = self.simTime
            if self.grounded:
                self.velocity[1] += escapeTakeoffSpeed * METRES_TO_WORLD
                self.grounded = False
        self.jumpWasHot = jumpHot

        self.wingPhase += dt * wingbeatHz * 2 * math.pi
        if self.wingPhase > 2 * math.pi:
            self.wingPhase -= 2 * math.pi * math.floor(self.wingPhase / (2 * math.pi))

        self.position = [self.position[i] + self.velocity[i] * dt for i in range(3)]
        self.resolve_collisions(dt, world)
        self.handle_feeding(dt, world)

        target = 0.0 if self.grounded else 1.0
        self.airborne += (target - self.airborne) * min(1.0, dt * 4)
        self.energy = max(0.0, self.energy - dt * 0.012)

    def resolve_collisions(self, dt, world):
        radius = self.contact_radius()
        self.grounded = False

        p, hit = world.contain(list(self.position), radius)
        if any(h != 0 for h in hit):
            self.position = p
            self.bumped = True
            if hit[1] > 0:
                self.grounded = True
            n = hit
            nl = math.sqrt(sum(x * x for x in n))
            n = [x / nl for x in n]
            into = sum(self.velocity[i] * n[i] for i in range(3))
            if into < 0:
                self.velocity = [self.velocity[i] - n[i] * into for i in range(3)]
            self.velocity = [x * 0.3 for x in self.velocity]
            self.yawRate *= 0.5

        c = world.collision(self.position, radius)
        if c:
            pos, n = c
            self.bumped = True
            self.position = pos
            if n[1] > 0.7:
                self.grounded = True
            into = sum(self.velocity[i] * n[i] for i in range(3))
            if into < 0:
                self.velocity = [self.velocity[i] - n[i] * into * 1.05 for i in range(3)]
                self.velocity = [x * 0.35 for x in self.velocity]
            if n[1] > 0.7:
                self.velocity[1] = max(0.0, self.velocity[1])

        if self.position[1] <= radius + contactEpsilon:
            self.grounded = True

    def handle_feeding(self, dt, world):
        self.isEating = False
        self.proboscisExtension += ((1.0 if self.isEating else 0.0)
                                    - self.proboscisExtension) * min(1.0, dt * 8)

    def speed(self):
        return math.sqrt(sum(x * x for x in self.velocity))

    # ---- gait joint angles (mirror of the leg block in updateJointAngles) ---
    def leg_angles(self):
        angles = {}
        duty = dutyFactor
        stride = self.strideMillimetres
        for side in ("left", "right"):
            for seg, ti in (("T1", 0), ("T2", 1), ("T3", 2)):
                tripod = (ti + (0 if side == "left" else 1)) % 2
                u = self.gaitPhase / (2 * math.pi) + (0.0 if tripod == 0 else 0.5)
                u -= math.floor(u)
                swinging = u > duty
                neutral, signs, cap, gain, knots = GAIT[seg]
                if swinging:
                    # swing: foot is airborne, a plain ramp is fine
                    shape = -1.0 + 2.0 * (u - duty) / (1.0 - duty)
                else:
                    # stance: follow the inverse-map knots so the planted
                    # foot advances at constant speed despite the nonlinear
                    # FK (piecewise linear over equal travel eighths)
                    n = len(knots) - 1
                    p = u / duty * n
                    i = min(int(p), n - 1)
                    shape = knots[i] + (knots[i + 1] - knots[i]) * (p - i)
                amp = min(cap, stride * duty / (2.0 * gain))
                sweep = shape * amp
                angles[f"coxa_{seg}_{side}"] = neutral[0] + signs[0] * sweep
                angles[f"femur_{seg}_{side}"] = neutral[1] + signs[1] * sweep
                angles[f"tibia_{seg}_{side}"] = neutral[2] + signs[2] * sweep
                angles["_swing_" + seg + side] = swinging
                angles["_u_" + seg + side] = u
        return angles


class Drives:
    def __init__(self, **kw):
        for k in ("wingPowerL", "wingPowerR", "wingSteerL", "wingSteerR",
                  "legL", "legR", "jump"):
            setattr(self, k, kw.get(k, 0.0))


def make(env="kitchen"):
    b = FlyBody()
    return World(env), b


# --------------------------------------------------------------- the tests
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("  PASS  " if cond else "  FAIL  ") + name + ("   " + detail if detail else ""))


print("FlightPhysicsTests (ported)")
print()

# 1 hover amplitude
phiDeg = strokeAmplitudeHover * 180 / math.pi
check("testHoverAmplitudeIsBiologicallyPlausible",
      130 < phiDeg < 160, f"{phiDeg:.1f} deg")
check("  ...and equals the solved force balance",
      abs(strokeAmplitudeHover - (weight / (airDensity * liftCoefficient * wingArea
          * (2 * wingbeatHz * r2) ** 2)) ** 0.5) < 0.02)

# 2 hover lift
lift = flight_force(strokeAmplitudeHover)
check("testHoverLiftEqualsBodyWeight", abs(lift / weight - 1) < 0.05,
      f"lift/weight = {lift/weight:.4f}")

# 3 maximum lift
lift = flight_force(strokeAmplitudeMax)
check("testMaximumLiftIsBoundedAndSufficient", 1.2 < lift / weight < 2.2,
      f"lift/weight = {lift/weight:.3f}")

# 4 yaw damping vs measured saccade torque
torque = yawDamping * (1600 * math.pi / 180)
check("testYawDampingReproducesMeasuredSaccadeTorque", 3e-10 < torque < 3e-9,
      f"{torque:.3e} N m  (measured ~1e-9)")

# 5 terminal speed
w, b = make()
b.d = Drives(wingPowerL=400, wingPowerR=400)
for _ in range(60 * 5):
    b.update(1 / 60, w)
check("testTerminalSpeedIsAroundOneMetrePerSecond", b.speed() * 0.01 < 1.5,
      f"{b.speed()*0.01:.3f} m/s")

# 6 yaw rate below the clamp
w, b = make()
b.d = Drives(wingPowerL=300, wingPowerR=300, wingSteerL=300, wingSteerR=0)
for _ in range(60 * 5):
    b.update(1 / 60, w)
dps = abs(b.yawRate) * 180 / math.pi
check("testYawRateStaysBelowTheClamp", dps < 2000, f"{dps:.1f} deg/s")

# 7 stroke amplitude inside the morphological limit
w, b = make()
b.d = Drives(wingPowerL=500, wingPowerR=500, wingSteerL=500, wingSteerR=0)
for _ in range(60 * 3):
    b.update(1 / 60, w)
check("testStrokeAmplitudeStaysInsideTheMorphologicalLimit",
      b.strokeAmplitudeL <= strokeAmplitudeMax + 1e-4
      and b.strokeAmplitudeR <= strokeAmplitudeMax + 1e-4
      and b.strokeAmplitudeR >= 0,
      f"L={b.strokeAmplitudeL*180/math.pi:.1f} deg  "
      f"R={b.strokeAmplitudeR*180/math.pi:.1f} deg")

# 8 steering bias bounded to 20 deg
w, b = make()
b.d = Drives(wingPowerL=300, wingPowerR=300, wingSteerL=300, wingSteerR=0)
for _ in range(60 * 3):
    b.update(1 / 60, w)
bias = abs(b.strokeAmplitudeL - b.strokeAmplitudeR) * 0.5
check("testSteeringBiasIsBoundedToTheMeasuredTwentyDegrees",
      bias <= steeringRange + 1e-3, f"{bias*180/math.pi:.2f} deg (limit 20)")

# 9 cannot escape any environment
ok = True
detail = []
for env in ("kitchen", "garden", "lab"):
    w, b = make(env)
    b.d = Drives(wingPowerL=500, wingPowerR=500, wingSteerL=200, jump=20)
    for _ in range(60 * 10):
        b.update(1 / 60, w)
    p = b.position
    good = (abs(p[0]) <= w.bounds + 0.01 and abs(p[2]) <= w.bounds + 0.01
            and p[1] >= -0.01 and p[1] <= w.ceilingHeight + 0.01
            and not any(math.isnan(x) for x in p))
    ok = ok and good
    detail.append(f"{env}:({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")
check("testFlyCannotEscapeAnyEnvironment", ok, " ".join(detail))

# 10 undriven fly settles
w, b = make()
b.reset((0, 2.0, 0))
b.d = Drives()
for _ in range(60 * 4):
    b.update(1 / 60, w)
check("testUndrivenFlySettlesOnTheFloor",
      b.position[1] < 0.3 and b.position[1] >= -0.01 and abs(b.velocity[1]) < 1.0,
      f"y={b.position[1]:.4f} vy={b.velocity[1]:.3f}")

# 11 walk speed
w, b = make()
b.reset((0, 0.0125, 0))
b.d = Drives(legL=400, legR=400)
for _ in range(60 * 3):
    b.update(1 / 60, w)
mm = b.speed() * 10
check("testWalkSpeedStaysBelowThirtyMillimetresPerSecond", mm < 35, f"{mm:.2f} mm/s")

# 12 large timestep
w, b = make()
b.d = Drives(wingPowerL=400, wingPowerR=400)
for _ in range(120):
    b.update(2.0, w)
check("testLargeTimestepIsClamped",
      not math.isnan(b.speed()) and abs(b.position[0]) <= w.bounds + 0.01,
      f"x={b.position[0]:.3f}")

# 13 containment
w = World()
p, n = w.contain([5000, 5000, 5000], 0.1)
check("testContainClampsAPointFarOutsideTheRoom",
      any(x != 0 for x in n) and abs(p[0]) <= w.bounds and p[1] <= w.ceilingHeight)
p, n = w.contain([0, 1.0, 0], 0.1)
check("testContainLeavesAnInteriorPointAlone",
      all(x == 0 for x in n) and p == [0, 1.0, 0])

# --------------------------------------------------------------------------
# NOTE on two deleted tests. testSustainedLegAsymmetryDoesNotCircleForever and
# testSustainedFlightAsymmetryDoesNotSpinAtTheClamp asserted that a CONSTANT
# left/right drive difference stops turning the animal. That was the adapted
# turn/yaw baseline (turnBias/yawBias), which round 5 removed: it was a body-
# layer hack that deleted the brain's own asymmetry after 1.5 s. A body with a
# genuinely sustained asymmetric command SHOULD keep turning — correcting it
# is the job of the closed-loop brain (optomotor/haltere), checked in
# tools/closed_loop.py T7, not of the plant. What must NOT happen is spinning
# from NOISE (T2 below) and turning the WRONG WAY (T1 below).
# --------------------------------------------------------------------------

# 14 T1 — yaw SIGN: stronger left wing yaws toward the weaker (right) side;
# the walking decode must agree (faster right legs -> turn left).
w, b = make()
b.reset((0.0, 0.5, 0.0))
b.d = Drives(wingPowerL=126, wingPowerR=126)  # 5% headroom: a real hover
for _ in range(60 * 6):                        # command, reference adapts
    b.update(1 / 60, w)
airborne_setup = not b.grounded
b.d = Drives(wingPowerL=164, wingPowerR=88)   # +30%/-30%: left stronger
h0 = b.heading
for _ in range(30):
    b.update(1 / 60, w)
dh_flight = b.heading - h0
check("testYawSignStrongerWingTurnsTowardWeakerSide",
      airborne_setup and dh_flight > 0,
      f"heading {math.degrees(dh_flight):+.1f} deg (right turn expected)")

w, b = make()
b.reset((0.0, 0.0125, 0.0))
b.d = Drives(legL=90, legR=150)              # right legs faster
h0 = b.heading
for _ in range(60):
    b.update(1 / 60, w)
dh_walk = b.heading - h0
check("testYawSignWalkingAgreesWithFlight", dh_walk < 0,
      f"heading {math.degrees(dh_walk):+.1f} deg (left turn expected)")

# 15 T2a — noise floor: iid +-10% left/right wing noise must not spin the
# animal. The opponent decode + muscle low-pass bring spike-counting noise
# (5.2 Hz per single spike in a 12-neuron group) under the 200 deg/s bar.
import random as _random
_rng = _random.Random(11)
w, b = make()
b.reset((0.0, 0.5, 0.0))
b.d = Drives(wingPowerL=126, wingPowerR=126)   # +5% so hover has headroom
for _ in range(60 * 4):
    b.update(1 / 60, w)
sq, nS = 0.0, 0
for _ in range(60 * 3):
    base = b.referenceRate * 1.05
    b.d.wingPowerL = base * (1 + _rng.gauss(0, 0.10))
    b.d.wingPowerR = base * (1 + _rng.gauss(0, 0.10))
    b.update(1 / 60, w)
    sq += b.yawRate * b.yawRate
    nS += 1
rms = math.sqrt(sq / nS) * 180 / math.pi
check("testYawNoiseFloorUnderNoiseyDrive", rms < 200 and not b.grounded,
      f"RMS yaw {rms:.0f} deg/s under +-10% iid noise (bar 200)")

# 16 T2b — a real asymmetric command still yanks: 60% step -> peak > 800
# deg/s, and the spin STOPS when the command goes symmetric again.
w, b = make()
b.reset((0.0, 0.5, 0.0))
b.d = Drives(wingPowerL=126, wingPowerR=126)
for _ in range(60 * 4):
    b.update(1 / 60, w)
base = b.referenceRate * 1.05
peak = 0.0
for _ in range(60):
    b.d.wingPowerL = base * 1.3
    b.d.wingPowerR = base * 0.7
    b.update(1 / 60, w)
    peak = max(peak, abs(b.yawRate))
peak_dps = peak * 180 / math.pi
for _ in range(60):
    b.d.wingPowerL = base
    b.d.wingPowerR = base
    b.update(1 / 60, w)
after = abs(b.yawRate) * 180 / math.pi
check("testSixtyPercentStepYanksAndStops",
      peak_dps > 800 and after < 100,
      f"peak {peak_dps:.0f} deg/s, after symmetric {after:.0f} deg/s")

# 17 T3 — no mode flicker: noisy collective drive near hover, above the
# floor, must not touch down once in 5 s; and a wingless fly on the floor
# must not take off. (The old lift-threshold mode flip switched ~10x/s.)
w, b = make()
b.reset((0.0, 0.5, 0.0))
b.d = Drives(wingPowerL=126, wingPowerR=126)
for _ in range(60 * 4):
    b.update(1 / 60, w)
touchdowns = 0
for _ in range(60 * 5):
    base = b.referenceRate * 1.05
    b.d.wingPowerL = base * (1 + _rng.gauss(0, 0.10))
    b.d.wingPowerR = base * (1 + _rng.gauss(0, 0.10))
    b.update(1 / 60, w)
    if b.grounded:
        touchdowns += 1
check("testNoModeFlickerUnderNoisyHover", touchdowns == 0 and b.position[1] > 0.1,
      f"{touchdowns} contact frames in 5 s, final y {b.position[1]:.2f}")

w, b = make()
b.reset((0.0, 0.0125, 0.0))
b.d = Drives(legL=60, legR=60)                # wings silent
liftoffs = 0
for _ in range(60 * 5):
    b.update(1 / 60, w)
    if not b.grounded:
        liftoffs += 1
check("testGroundedFlyWithoutWingDriveNeverLiftsOff",
      liftoffs == 0 and b.airborne < 0.1,
      f"{liftoffs} airborne frames, airborne {b.airborne:.3f}")

# 18 sustained climb still raises the fly (referenceRate trim, collective only)
w, b = make()
b.reset((0.0, 0.35, 0.0))
b.d = Drives(wingPowerL=240, wingPowerR=240)
for _ in range(60 * 3):
    b.update(1 / 60, w)
y3 = b.position[1]
for _ in range(60 * 2):
    b.update(1 / 60, w)
check("testSustainedClimbCommandRaisesTheFly",
      y3 > 10 and b.position[1] > y3 + 10,
      f"y={y3:.1f} cm at 3 s -> {b.position[1]:.1f} cm at 5 s")

# 18b ...and the equilibrium reflex re-trims afterwards
for _ in range(60 * 45):
    b.update(1 / 60, w)
ratio = b.liftUN * 1e-6 / (mass * gravity)
check("testLiftRetrimsAfterASustainedClimb", abs(ratio - 1) < 0.05,
      f"lift/weight = {ratio:.3f} after 50 s of constant 240 Hz drive")

# 19 T4 — stance foot must not skate: the T3 tarsus tip's body-frame fore-aft
# velocity during stance equals -v (the gait amplitude is derived from the
# body's own stride). Measured through the renderer's FK chain.
import sys as _sys
_sys.path.insert(0, "tools")
from softrender import (axis_angle as _aa, eye as _I4, load_flymodel as _lfm,
                        quat_to_mat as _qm, translate as _tr)
import numpy as _np
_parts, _joints, _V, _I = _lfm("build/flymodel.bin")
_nidx = {p["name"]: i for i, p in enumerate(_parts)}
_lo, _hi = _V[:, :3].min(0), _V[:, :3].max(0)
_scale = 0.25 / float((_hi - _lo).max())


def _assemble(angles):
    M = []
    for p in _parts:
        par = _I4() if p["parent"] < 0 else M[p["parent"]]
        local = _tr(p["pos"]) @ _qm(p["quat"])
        for k in range(p["jcount"]):
            jd = _joints[p["jstart"] + k]
            a = min(max(angles.get(jd["name"], 0.0), jd["lower"]), jd["upper"])
            if a != 0:
                local = local @ _aa(jd["axis"], a)
        M.append(par @ local)
    return M


def _tarsus_x(mm_angles, vert):
    p = _parts[_nidx["tarsus_T3_left"]]
    v = _V[p["vstart"] + vert][:3]          # xyz only (the row is pos+normal)
    M = mm_angles[_nidx["tarsus_T3_left"]]
    return float((M[:3, :3] @ v + M[:3, 3])[0] * _scale * 10)   # mm


w, b = make()
b.reset((0.0, 0.0125, 0.0))
b.d = Drives(legL=120, legR=120)
for _ in range(60 * 3):
    b.update(1 / 60, w)
# fixed tarsus-tip vertex, defined exactly as tools/gait_geometry.py defines
# it: the mesh vertex farthest from the thorax origin in the REST pose
# (part-local coordinates), chosen once, never re-selected per frame.
_p = _parts[_nidx["tarsus_T3_left"]]
_vert = int(_np.argmax(_np.linalg.norm(
    _V[_p["vstart"]:_p["vstart"] + _p["vcount"], :3], axis=1)))
samples = []
prev_x, prev_u = None, None
dt = 1 / 60
for _ in range(60 * 2):
    b.update(dt, w)
    a = b.leg_angles()
    u = a["_u_T3left"]
    x = _tarsus_x(_assemble({k: v for k, v in a.items() if not k.startswith("_")}), _vert)
    # mid-stance only, and only across frames of the SAME stance phase:
    # at 13 Hz / 60 fps a frame can straddle the swing->stance transition,
    # where the finite difference would mix airborne swing with planted
    # stance and say nothing about skating.
    if (prev_x is not None and 0.05 < u < 0.5 and prev_u is not None
            and prev_u < u):
        vfoot = (x - prev_x) / dt                  # mm/s, model +x is forward
        vbody = b.speed() * 10.0                   # world cm/s -> mm/s
        samples.append(vfoot / vbody)
    prev_x, prev_u = x, u
rel = sum(abs(s + 1.0) for s in samples) / len(samples) if samples else 9.9
check("testStanceFootDoesNotSkate", rel < 0.25 and len(samples) > 20,
      f"stance foot v / body v = {sum(samples)/len(samples):+.2f} (target -1.00), "
      f"mean |err| {rel:.0%}, {len(samples)} samples")

# 20 escape: a TT spike event launches the jump once, the refractory swallows
# the rattle, and a mid-air event is counted but not fired.
w, b = make()
b.reset((0.0, 0.0125, 0.0))
b.d = Drives()
b.update(1 / 60, w)
b.d = Drives(jump=31.25)            # one of the two TT neurons spiking
b.update(1 / 60, w)
launched = b.jumpEvents == 1 and b.velocity[1] > 80 and not b.grounded
b.d = Drives(jump=31.25)
for _ in range(2):                  # still hot: no retrigger
    b.update(1 / 60, w)
b.d = Drives()
b.update(1 / 60, w)
b.d = Drives(jump=31.25)            # t=0.067 s: inside the 0.1 s refractory
b.update(1 / 60, w)
refractory_ok = b.jumpEvents == 1
b.d = Drives()
for _ in range(2):                  # cool down, stay airborne (the measured
    b.update(1 / 60, w)             # 0.9 m/s jump lands again at ~0.22 s)
b.d = Drives(jump=31.25)            # t=0.117 s: past the refractory, mid-air
vy_before = b.velocity[1]
b.update(1 / 60, w)
midair_ok = b.jumpEvents == 2 and b.velocity[1] < vy_before + 1.0
check("testEscapeSpikeLaunchesOnceWithRefractory",
      launched and refractory_ok and midair_ok,
      f"events={b.jumpEvents} launch={launched} refractory={refractory_ok} "
      f"midair_counted={midair_ok}")

print()
bad = [r for r in results if not r[1]]
print(f"{len(results) - len(bad)}/{len(results)} checks passed")
if bad:
    print("FAILING: " + ", ".join(r[0] for r in bad))
    import os as _os
    if _os.environ.get("SIMCHECK_STRICT") != "0":
        raise SystemExit(1)
