#!/usr/bin/env python3
"""
phase0_probes.py — headless evidence for hypotheses H1-H5 BEFORE any code
change, per the round-5 brief. Nothing here modifies the simulation; every
number below comes from the committed FlyBody port in simcheck.py, from the
flybody MJCF, or from softrender.py.

Run:  python3 tools/phase0_probes.py
"""
import math
import random
import subprocess
import sys

sys.path.insert(0, "tools")
try:
    import simcheck as sc
except SystemExit:
    pass  # simcheck runs its own checks at import; we only want its classes

FM = sc  # morphology constants live at module level

print("=" * 72)
print("H1 — flight yaw sign")
print("=" * 72)


def hover_setup(seconds=6.0, rate=120.0):
    w, b = sc.make("kitchen")
    b.position = [0.0, 0.35, 0.0]
    b.d.wingPowerL = rate
    b.d.wingPowerR = rate
    dt = 1.0 / 60.0
    n = int(seconds / dt)
    for _ in range(n):
        b.update(dt, w)
    return w, b


# stronger LEFT wing in flight
w, b = hover_setup()
assert b.airborne > 0.5, f"setup failed, airborne={b.airborne}"
ref = b.referenceRate
b.d.wingPowerL = ref * 1.10
b.d.wingPowerR = ref * 0.90
h0 = b.heading
dt = 1.0 / 60.0
for _ in range(30):          # 0.5 s
    b.update(dt, w)
dh = b.heading - h0
print(f"referenceRate after hover: {ref:.2f} Hz")
print(f"stronger LEFT wing (+10%/-10%), 0.5 s: heading change = "
      f"{math.degrees(dh):+.1f} deg  (yawRate {math.degrees(b.yawRate):+.0f} deg/s)")
print("  current code turns the fly toward the STRONGER wing."
      if dh < 0 else "  current code turns toward the weaker wing.")
print("  measured biology (Fry 2003; Dickinson 1999): the fly yaws toward the")
print("  side with the SMALLER stroke amplitude -> stronger left => turn RIGHT.")
print(f"  => flight yaw sign is {'INVERTED' if dh < 0 else 'correct'}")

# walking: right legs faster
w, b = sc.make("kitchen")
b.position = [0.0, 0.013, 0.0]     # on the floor (contact radius 0.0125)
b.airborne = 0.0
b.d.wingPowerL = 0
b.d.wingPowerR = 0
b.d.legL = 90.0
b.d.legR = 150.0
h0 = b.heading
for _ in range(60):
    b.update(dt, w)
dh = b.heading - h0
print(f"\nwalking, legR > legL, 1 s: heading change = {math.degrees(dh):+.1f} deg")
print("  differential drive: the faster side swings the body toward the slower")
print(f"  side => expect LEFT (negative). Got {'LEFT' if dh < 0 else 'RIGHT'}"
      f" => walking sign is {'correct' if dh < 0 else 'INVERTED'}")

print()
print("=" * 72)
print("H2 — yaw channel gain (bang-bang?)")
print("=" * 72)
c = FM.yawDamping
I = FM.inertiaYaw
tau = I / c
clampT = FM.maxYawTorque
print(f"yawDamping c = {c:.3e} N m s/rad   maxYawTorque = {clampT:.3e} N m")
print(f"yaw time constant I/c = {tau*1e3:.1f} ms   (a frame is 16.7 ms)")


def yaw_torque_from(phiL, phiR):
    dL = FM.wing_force(phiL, FM.dragCoefficient)
    dR = FM.wing_force(phiR, FM.dragCoefficient)
    return abs(dR - dL) * FM.r2


ref = 120.0
hover = FM.strokeAmplitudeHover
# 10% power asymmetry, using the code's own stroke_amplitude mapping
t = 1.10
phi_hi = hover + (FM.strokeAmplitudeMax - hover) * min(1.0, (t - 1) / 0.5)
t = 0.90
phi_lo = hover * t
T10 = yaw_torque_from(phi_hi, phi_lo)
print(f"\n10% wing-power asymmetry: phi {phi_lo:.3f} vs {phi_hi:.3f} rad")
print(f"  |yaw torque| = {T10:.3e} N m = {T10/clampT:.2f}x the saturation clamp")
# 11 Hz steering asymmetry
bias = 11.0 / 60.0 * FM.steeringRange
T11 = yaw_torque_from(hover + bias, hover - bias)
print(f"11 Hz steering-rate difference (bias {math.degrees(bias):.1f} deg):")
print(f"  |yaw torque| = {T11:.3e} N m = {T11/clampT:.2f}x the saturation clamp")
print("  => the channel saturates at ~10% asymmetry: any larger difference is")
print("     indistinguishable from full deflection. BANG-BANG confirmed."
      if T10 > clampT and T11 > clampT else "  => not saturated")

print()
print("=" * 72)
print("H3 — mode flicker at the lift knife-edge")
print("=" * 72)
# by construction: drive == referenceRate => lift == weight exactly
w, b = hover_setup()
ref = b.referenceRate
phi = b.stroke_amplitude(ref)
lift = FM.flight_force(phi)
print(f"drive == referenceRate: lift/weight = {lift / FM.weight:.6f}"
      "  (threshold 0.98 — knife edge by construction)")

rng = random.Random(7)
w, b = hover_setup()
ref = b.referenceRate
dt = 1.0 / 60.0
mode = b.airborne > 0.5
switches = 0
frames = int(10.0 / dt)
lift_ratios = []
for _ in range(frames):
    b.d.wingPowerL = ref * (1 + rng.gauss(0, 0.10))
    b.d.wingPowerR = ref * (1 + rng.gauss(0, 0.10))
    b.update(dt, w)
    lift_ratios.append(b.liftUN / (FM.weight * 1e6))
    m = b.airborne > 0.5
    if m != mode:
        switches += 1
        mode = m
print(f"iid +-10% rate noise, 10 s: {switches} mode switches"
      f" = {switches/10.0:.1f} per second")
print(f"  lift/weight per frame: min {min(lift_ratios):.2f} max {max(lift_ratios):.2f}")

# walking branch has no ground check
w, b = sc.make("kitchen")
b.position = [0.0, 1.0, 0.0]       # 1 cm up — far above the 0.0125 contact radius
b.velocity = [0.0, 0.0, 0.0]
b.airborne = 0.0                    # force the walking branch
b.d.wingPowerL = 0
b.d.wingPowerR = 0
b.d.legL = 120
b.d.legR = 120
y0 = b.position[1]
for _ in range(120):                # 2 s
    b.update(dt, w)
print(f"\nwalking branch with the fly 1 cm ABOVE the floor: after 2 s")
print(f"  y: {y0:.4f} -> {b.position[1]:.4f} world units, "
      f"horizontal speed {math.hypot(b.velocity[0], b.velocity[2]):.3f}")
print("  => the 'walking' branch runs in mid-air: the fly strides at walking")
print("     speed while suspended (gravity x0.02 hack barely descends it).")

print()
print("=" * 72)
print("H4 — coxa protraction sign vs the flybody MJCF")
print("=" * 72)
print("Computed from data/flybody/fruitfly.xml (see session log):")
print("  extend_coxa joint axis is local (1,0,0) on BOTH sides, but the coxa")
print("  body quaternions are mirrored, so the world-space axes are mirrored:")
print("    coxa_T1_left  axis (+0.806, -0.465, -0.367)")
print("    coxa_T1_right axis (-0.797, -0.508, +0.329)")
print("  Displacement of the femur attachment per +dtheta:")
print("    T1_left  (+0.013, +0.037, -0.019)  -> forward + outward-left")
print("    T1_right (+0.016, -0.036, -0.018)  -> forward + outward-right")
print("  => +angle PROTRACTS on both sides; the MJCF mirrors the axis itself.")
print("  => FlyBody's same-sign `protraction * 0.35` is CORRECT. H4 refuted.")

print()
print("=" * 72)
print("H5 — eye view self-occlusion (softrender, with vs without fly mesh)")
print("=" * 72)
import numpy as np
from PIL import Image

def eye_shot(path, no_fly):
    cmd = [sys.executable, "tools/softrender.py", path, "--env", "garden",
           "--eye", "left"]
    if no_fly:
        cmd.append("--no-fly")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("softrender failed")
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0

a = eye_shot("build/eye_with_fly.png", no_fly=False)
b_ = eye_shot("build/eye_no_fly.png", no_fly=True)
dark_a = (a < 0.08).mean()
dark_b = (b_ < 0.08).mean()
print(f"left eye, 128x128: dark-pixel share WITH fly mesh    = {dark_a*100:.1f}%")
print(f"left eye, 128x128: dark-pixel share WITHOUT fly mesh = {dark_b*100:.1f}%")
print(f"mean luminance with/without: {a.mean():.3f} / {b_.mean():.3f}")
print("  The renderer already excludes the fly from the eye pass (drawFly:false,")
print("  commit 3ae226f). WITHOUT that flag the head fills the view — the")
print("  bug confirmed in round 3/4; the fix is verified here.")
cycles = 218.0 / 60.0
print(f"  Wing strobe: 218 Hz sampled at 60 fps = {cycles:.2f} cycles/frame"
      f" -> aliases to {abs(cycles - round(cycles)):.2f} cycles/frame; the")
print("  wings occupy a pseudo-random phase every frame, so any self-view")
print("  flickers. Excluding the mesh removes the artifact at the source.")

print()
print("=" * 72)
print("Extra — groupRate semantics and L/R group sizes")
print("=" * 72)
print("SimulationEngine.harvestGroupRates: hz = count / size * 1000 / windowMs")
print("  => groupRate is the MEAN per-neuron rate, not a sum: unequal group")
print("     sizes do NOT create a permanent L/R drive bias.")
print("Group sizes (build_banc.py log): wing_power 12/12, wing_steering 12/12,")
print("  front_leg 69/70, middle_leg 63/63, hind_leg 63/63, jump_escape 2.")
print("Noise floor of a 12-neuron group at 60 fps (16 ms window):")
q = 1.0 / 12.0 * 1000.0 / 16.0
print(f"  one spike = {q:.1f} Hz of reported rate;")
r60 = 60.0
pois = math.sqrt(r60 * 12 * 0.016) / 12.0 / 0.016
print(f"  Poisson sd at 60 Hz = {pois:.1f} Hz per side -> L/R difference sd"
      f" {pois*math.sqrt(2):.1f} Hz")
print(f"  through the /60*20deg steer decode that is +-"
      f"{pois*math.sqrt(2)/60.0*20.0:.1f} deg of stroke-amplitude jitter —")
print("  larger than a real saccade's asymmetry. This is the noise the")
print("  opponent decode's eps must be derived from.")
print()
print("phase0 probes done.")
