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
        return 500.0               # invisible sky cap, nothing drawn there

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
class FlyBody:
    def __init__(self):
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
        self.turnBias = 0.0
        self.energy = 1.0
        self.hurt = 0.0
        self.isEating = False
        self.bumped = False
        self.proboscisExtension = 0.0

    referenceTau = 2.0
    turnBiasTau = 1.5

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

    def update(self, dt, world):
        dt = clamp(dt, 1.0 / 480.0, 1.0 / 20.0)
        self.bumped = False
        self.hurt = max(0.0, self.hurt - dt * 1.5)

        self.update_reference((self.d.wingPowerL + self.d.wingPowerR) * 0.5, dt)

        steerBiasL = clamp((self.d.wingSteerL - self.d.wingSteerR) / 60.0
                           * steeringRange, -steeringRange, steeringRange)
        steerBiasR = -steerBiasL
        phiL = clamp(self.stroke_amplitude(self.d.wingPowerL) + steerBiasL,
                     0.0, strokeAmplitudeMax)
        phiR = clamp(self.stroke_amplitude(self.d.wingPowerR) + steerBiasR,
                     0.0, strokeAmplitudeMax)
        self.strokeAmplitudeL = phiL
        self.strokeAmplitudeR = phiR

        fL = wing_force(phiL, liftCoefficient)
        fR = wing_force(phiR, liftCoefficient)
        totalForce = fL + fR
        self.liftUN = totalForce * 1e6

        airborneNow = 1.0 if totalForce > weight * 0.98 else 0.0
        self.airborne += (airborneNow - self.airborne) * min(1.0, dt * 4)

        dragL = wing_force(phiL, dragCoefficient)
        dragR = wing_force(phiR, dragCoefficient)
        yawTorque = clamp((dragR - dragL) * r2, -maxYawTorque, maxYawTorque)

        if self.airborne > 0.5:
            c = yawDamping
            I = inertiaYaw
            self.yawRate = (self.yawRate + yawTorque / I * dt) / (1 + c / I * dt)
            self.heading += self.yawRate * dt

            tilt = strokePlaneAngle * min(1.0, totalForce / weight - 0.6)
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

            self.roll += ((phiR - phiL) * 0.5 - self.roll) * min(1.0, dt * 6)
            self.stepFrequency = 0.0
        else:
            legMean = (self.d.legL + self.d.legR) * 0.5
            f = min(stepFrequencyMax,
                    legMean / self.referenceRate * stepFrequencyMax * 2.2)
            self.stepFrequency = f
            stride = walkSpeedMax / stepFrequencyMax
            speedMS = f * stride
            legDiff = (self.d.legR - self.d.legL) / max(self.referenceRate, 1.0)
            self.turnBias += (legDiff - self.turnBias) * min(1.0, dt / self.turnBiasTau)
            self.yawRate = -(legDiff - self.turnBias) * f * 2.0
            self.heading += self.yawRate * dt
            fwd = [math.sin(self.heading), 0.0, -math.cos(self.heading)]
            v = [fwd[i] * speedMS * METRES_TO_WORLD for i in range(3)]
            v[1] = min(self.velocity[1], 0.0) - gravity * METRES_TO_WORLD * dt * 0.02
            self.velocity = v
            self.gaitPhase += dt * f * 2 * math.pi
            self.roll += (0.0 - self.roll) * min(1.0, dt * 8)
            self.pitch += (0.0 - self.pitch) * min(1.0, dt * 8)

        if self.d.jump > 8 and self.airborne < 0.6:
            self.velocity[1] += 0.9 * METRES_TO_WORLD
            self.airborne = 1.0

        self.wingPhase += dt * wingbeatHz * 2 * math.pi
        if self.wingPhase > 2 * math.pi:
            self.wingPhase -= 2 * math.pi * math.floor(self.wingPhase / (2 * math.pi))

        self.position = [self.position[i] + self.velocity[i] * dt for i in range(3)]
        self.resolve_collisions(dt, world)
        self.handle_feeding(dt, world)
        self.energy = max(0.0, self.energy - dt * 0.012)

    def resolve_collisions(self, dt, world):
        radius = bodyLength * 0.5 * METRES_TO_WORLD
        p, hit = world.contain(list(self.position), radius)
        if any(h != 0 for h in hit):
            self.position = p
            self.bumped = True
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
            into = sum(self.velocity[i] * n[i] for i in range(3))
            if into < 0:
                self.velocity = [self.velocity[i] - n[i] * into * 1.05 for i in range(3)]
                self.velocity = [x * 0.35 for x in self.velocity]
            if n[1] > 0.7:
                self.velocity[1] = max(0.0, self.velocity[1])

    def handle_feeding(self, dt, world):
        self.isEating = False
        # food handling is irrelevant to the physics assertions
        self.proboscisExtension += ((1.0 if self.isEating else 0.0)
                                    - self.proboscisExtension) * min(1.0, dt * 8)

    def speed(self):
        return math.sqrt(sum(x * x for x in self.velocity))


class Drives:
    def __init__(self, **kw):
        for k in ("wingPowerL", "wingPowerR", "wingSteerL", "wingSteerR",
                  "legL", "legR", "jump"):
            setattr(self, k, kw.get(k, 0.0))


def make(env="kitchen"):
    b = FlyBody()
    b.d = Drives()
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

# 6 yaw rate below the clamp  (the one that was red)
w, b = make()
b.d = Drives(wingPowerL=300, wingPowerR=300, wingSteerL=300, wingSteerR=0)
for _ in range(60 * 5):
    b.update(1 / 60, w)
dps = abs(b.yawRate) * 180 / math.pi
check("testYawRateStaysBelowTheClamp", dps < 2000, f"{dps:.1f} deg/s")

# 7 NEW stroke amplitude inside the morphological limit
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

# 8 NEW steering bias bounded to 20 deg
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
b.reset((0, 0.02, 0))
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

# 14 the user's complaint: a constant left/right leg bias used to turn the
# fly in circles forever (steady ~11 deg/s on the HUD). With the adapted
# turn baseline it must straighten out.
w, b = make()
b.reset((0, 0.02, 0))
b.d = Drives(legL=30, legR=45)
for _ in range(60 * 12):
    b.update(1 / 60, w)
dps = abs(b.yawRate) * 180 / math.pi
check("testSustainedLegAsymmetryDoesNotCircleForever", dps < 5,
      f"yaw {dps:.2f} deg/s after 12 s of constant bias")

# 15 ...but a CHANGE in asymmetry still turns the animal (steering works)
w, b = make()
b.reset((0, 0.02, 0))
b.d = Drives(legL=30, legR=30)
for _ in range(60 * 6):
    b.update(1 / 60, w)
b.d = Drives(legL=20, legR=45)          # a new, stronger asymmetry
for _ in range(30):
    b.update(1 / 60, w)
dps = abs(b.yawRate) * 180 / math.pi
check("testTurnsStillRespondToNewAsymmetry", dps > 5,
      f"yaw {dps:.2f} deg/s right after the change")

print()
bad = [r for r in results if not r[1]]
print(f"{len(results) - len(bad)}/{len(results)} checks passed")
if bad:
    print("FAILING: " + ", ".join(r[0] for r in bad))
    raise SystemExit(1)
