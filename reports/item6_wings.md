# Item 6 — the wings: the force they make, and the fact that an asymmetry turns the animal

*Written with `tools/wing_aero.py`, `build/wing_golden.json` and the runs quoted
below. Where a number is a measurement, the command that produced it is named.
Where it is not measured, it is labelled in `docs/ASSUMPTIONS.md` and the label is
given here too.*

## What was there, and what was not

Both wings were already in the body model: 104 bodies, a membrane and a hinge
plate apiece, three joints each (yaw, roll, pitch), an actuator group on the
thorax — and `build_body.py` drove only the legs. There was no force law, no
beat, and no path for anything the wings might do to reach the animal. The audit
row said PARTIAL and named the gap exactly: *architecture must accept
motor → muscle → aero → body*. That path is what this item builds.

## The wing, measured out of the body model

`python3 tools/wing_aero.py` reads the MJCF with MuJoCo, puts every mesh triangle
into the thorax frame, and measures the wing rather than assuming it:

| | left | right |
|---|---|---|
| planform area | 1.741 mm² | 1.741 mm² |
| reach, hinge → tip | 2.65 mm | 2.65 mm |
| mean chord (A / length) | 0.66 mm | 0.66 mm |
| aspect ratio (length²/A) | 3.96 | 3.96 |
| ∫r²dA about the hinge | 4.164e-4 cm⁴ | 4.164e-4 cm⁴ |
| r̂₂ = √(∫r²dA/A) | 1.547 mm = **58 % of the reach** | 58 % |
| hinge → centre of mass | 1.531 mm | 1.531 mm |
| mass | 8.00 µg | 8.00 µg |
| stroke axis (thorax frame) | (0.7378, 0, 0.6751) | (−0.7378, 0, −0.6751) |
| the stroke plane it implies | 47.5° off the horizontal | 47.5° |
| lift direction (the stroke plane's normal, up) | (0.7283, 0.0980, 0.6782) | (0.7283, −0.0980, 0.6782) |
| strip radius about the axis, sin θ | 0.9951 | 0.9951 |

The wing is binned into 12 strips by measured chord, and the strips' areas sum to
the planform **exactly** (100.00 %, gated): an integration whose strips do not add
up to the wing is an integration of some other wing, which is precisely what the
first version of this tool was doing.

Four defects were found and fixed by measurement, and they are worth writing down
because each one was a plausible-looking number:

1. **A missing geom rotation.** Mesh vertices were taken into the wing's frame
   with the geom's offset but not its rotation, so the wing came out 1.89 mm long
   instead of 2.65 mm — and the strips were then binned over a radius range that
   excluded most of the wing's area. Measuring the wing through MuJoCo's own
   forward kinematics is what caught it.
2. **The second moment counted the shell twice.** The mesh is a closed shell, so
   `Σ |n·ê| A` over its triangles is twice the planform. The planform was halved
   and the second moment was not, which doubled `∫r²dA`, inflated every force by
   2, and read r̂₂ out at **98 % of the span** — a fly's wing is 0.55–0.65 R. With
   it fixed, r̂₂ = 58 %.
3. **The wing was pitched away from the wrong plane.** The angle of attack is
   measured from the *stroke* plane (at α = 0 the chord lies along the velocity,
   which lies in the stroke plane), not from the mesh's own rest plane — in the
   rest pose the wings sit out flat and horizontal while the stroke plane is 47.5°
   off it. Pitching away from the rest plane flew the wing almost edge-on
   (n̂·v̂ = 0.994) and pointed the lift at the ground. Both readings are run as CI
   rehearsals; only the right one passes.
4. **The mean wing speed was missing its 2/π.** The cycle mean of |cos| is 2/π,
   so the mean strip speed is (2/π)·ωΦ·r̂₂ = 1666 mm/s, not 837. The Reynolds
   number was half what it should be: **74 on the mean chord**, 292 on the wing's
   length, which is the animal's flapping regime.

## The beat, and what it makes

The force law is Dickinson, Lehmann & Sane's (1999) measured fits, not a
stand-in: `C_L = 0.225 + 1.58 sin(2.13α − 7.2°)`, `C_D = 1.92 − 1.55 cos(2.13α −
9.8°)`. Each strip at radius `r` moves at `ω r sinθ` and carries `½ρv²A C_L` along
its wing's lift direction and `½ρv²A C_D` against its motion. At full activation
(stroke ±77.5°, 200 Hz, α 45 ± 30°):

| quantity | value |
|---|---|
| the animal's weight | 0.97 dyn (9.66 µN) |
| flight force (each wing's force along its own stroke plane's normal) | 0.485 + 0.485 = **0.969 dyn → lift/weight 1.00** |
| the same force in the body's frame | **+0.712 dyn forward, 0.000 sideways, +0.657 up** |
| the angle of attack at which the flight force equals the weight | **44.9°** — Dickinson et al. measured ~45–50° at mid-stroke in the animal |
| mean \|drag\| through the beat, both wings | 0.809 dyn (the cost; the *net* drag of a symmetric beat is zero) |
| the wing's own inertial force, mid-stroke | 3.54 dyn with the model's wing mass — 3.7× the animal's weight and ~8× too big, because the model's wing is 8 µg and a real one is ~1 µg (ASSUMPTIONS #35). At a real mass it is 0.44 dyn, comparable with the 0.485 dyn of lift it makes, which is the animal's actual situation |
| peak force in the beat / mean | 1.9× (see ASSUMPTIONS #38: the quasi-steady model has no separate unsteady peak) |

### One thing fell out of the geometry, and it closes

The wings' force is **47.3° forward of vertical** while the body is held level —
because the stroke plane belongs to the thorax and this thorax's stroke plane is
tilted. That is not an error in the model, it is a *posture*: rotate the beat's
force by a 47.3° nose-up pitch and it becomes (i) vertical and (ii) **exactly the
animal's weight**. The gate solves for that angle, and the solved value (44.9°)
sits inside the 45–50° Dickinson et al. measured at mid-stroke in a hovering fly.
The geometry, the force and the posture are one fact here, which is what a model
that is not allowed to fake anything is supposed to look like.

## Steering: what an amplitude asymmetry does

`steer` adds to the left wing's activation and subtracts from the right's, which
is the asymmetry the steering muscles make. The moment is taken **about the
thorax origin** — the hinge offset is in it, and that offset is what makes a
left-right force difference a yaw at all:

| steer | stroke L/R | flight force L/R (dyn) | yaw moment (dyn·cm) | roll moment (dyn·cm) |
|---|---|---|---|---|
| −1.0 | 38.8 / 77.5° | 0.121 / 0.485 | **+0.04549** | −0.04159 |
| −0.5 | 58.1 / 77.5° | 0.273 / 0.485 | +0.02438 | −0.02238 |
| 0.0 | 77.5 / 77.5° | 0.485 / 0.485 | −0.00012 | −0.00012 |
| +0.5 | 77.5 / 58.1° | 0.485 / 0.273 | −0.02454 | +0.02224 |
| +1.0 | 77.5 / 38.8° | 0.485 / 0.121 | **−0.04562** | +0.04147 |

* The asymmetry reverses the yaw moment, as it must.
* **The sign is physics, not taste**: the wing that is driven harder sweeps
  harder, so it pushes *its* side of the body forward and the animal's nose goes
  the other way. Driving the left wing harder yaws the animal to the **right**.
  A model that got this backwards would fly in circles the wrong way, and it is
  the kind of sign that gets "fixed" downstream — so it is gated.
* The yaw and the roll of one asymmetry are the same force on the same lever,
  resolved on two axes; their ratio is 1.09, which is the stroke plane's own
  fore-aft/vertical ratio (0.7283/0.6782 = 1.074) to within the strip
  distribution. The gate checks it as a bound (force × reach).
* A *symmetric* beat turns nothing — exactly, by mirror symmetry — and the tool
  measures the floor rather than asserting zero: **1.24e-4 dyn·cm**, 0.27 % of the
  steering signal, which is what the body model's two wing meshes (mirrors to
  ~2e-6) can produce. The gate fails the model at 2 %, so a real mirroring error
  cannot hide under it.
* In the units a tether rig reports, the largest yaw moment here is
  **4.6e-9 N·m** for a full (±100 %) amplitude asymmetry. That is a bigger
  asymmetry than a fly uses to turn; the number is quoted with its asymmetry so
  nobody compares it with a measured turning torque from a different experiment.

## Through the connectome

All four pools are in the packed connectome (`motor_wing_power_left/right` 12
cells each, `motor_wing_steering_left/right` 12 each; the tension pools 6+6 are
there too and are not used yet). Driven at the app's tone (2.5) for 1 s:

| pool | at tone 2.5 | at drive 4.0 ("full") |
|---|---|---|
| `motor_wing_power_left` | 158.67 Hz | 173.7 Hz |
| `motor_wing_power_right` | 182.50 Hz | 212.3 Hz |
| `motor_wing_steering_left` | 109.50 Hz | 105.8 Hz |
| `motor_wing_steering_right` | 130.17 Hz | 114.8 Hz |

**The two power pools are not symmetric** (173.7 vs 212.3 Hz at full drive), so
the app calibrates each pool against *its own* resting and full-drive rates. One
shared reference would hold a ≈20 % amplitude asymmetry open and the animal would
fly in a slow circle with no command to do it. (This is the same class of finding
as item 7's −24.7 Hz haltere-pool offset: measured, wired in explicitly, and
written into `world.json` as `rate_reference_hz`/`rate_tone_hz` rather than
cancelled.) At the app's own operating point the calibration leaves a residual
uncommanded yaw of **−5.4e-3 dyn·cm** — about a third of the signal the steering
pools produce when they are asked for a turn. That residual is a property of this
connectome slice, and it is reported, not trimmed away.

A steering differential of ±0.9 tone units moves the two steering pools by
+13.00 / −2.83 Hz, which the map turns into a stroke of 77.5° / 64.7° (the left
is at its joint limit) and a yaw moment of **−0.0164 dyn·cm**.

### The reflex arc the connectome already contains

Item 7 measured the haltere afferents' *direct* targets: `motor_wing_steering_left`
8 synapses, `motor_wing_steering_right` 10. So the haltere→wing-steering path is
monosynaptic in this connectome and the cord carries it without any controller of
ours. Driving the haltere sensory pools exactly as `tools/haltere_gyro.py` does
and reading the wing steering pools back:

| condition | steering L | steering R | L−R |
|---|---|---|---|
| still | 109.50 Hz | 130.17 Hz | −20.67 |
| yaw +400 °/s | 86.50 Hz | 112.00 Hz | −25.50 |
| yaw −400 °/s | 90.00 Hz | 170.17 Hz | −80.17 |

**The arc exists** — a 400 °/s rotation moves the wing steering pools by 3–40 Hz
without anything of ours in the path, and the gate fails if it stops doing so.
**What is not established:** that the arc *stabilises* the animal. The response is
not antisymmetric in the turn direction (a left yaw moves the difference by
−4.8 Hz, a right yaw by −59.5 Hz), so whether this loop damps a yaw or drives it
is an open question for the closed-loop item, and it is left open on purpose: the
temptation is to flip a sign in Swift until the animal looks stable, which is
exactly the kind of thing this repository is not allowed to do.

## What the app does with it

* `WingAero.swift` — the port, with the wing's measured geometry and the golden
  table (geometry, 25 strip-force rows, 9 beats) generated *from the tool's own
  output* by `tools/wing_golden_block.py` and parsed by `WingAeroTests`. CI
  re-runs the tool and compares the rows, so a drift on either side fails.
* `FlyDynamics` gained an **external wrench** per body (in the body's own frame,
  applied with gravity and the floor in the bias forces) and **hinge targets** for
  the joints whose motion comes from outside the muscle loop.
* `FlyLiveBody` advances the beat at the stroke frequency, holds the two stroke
  joints at the measured kinematics through the joint's own spring, and hands the
  air's answer to the solver as a force and a torque **on the thorax**. Nothing
  in that path sets a position or a velocity of the animal: what leaves it is a
  wrench, and the physics decides where the fly goes.
* `FlyCord` drives the four wing pools from the tone and reads their rates back;
  `WingAero` maps rate → stroke amplitude, each pool against its own references.
* `WorldView` attaches both from `world.json`'s `wings` block and prints a HUD
  line — stroke L/R, lift/weight, yaw moment, drag — because a wing force that is
  not visible on the HUD is a wing force nobody will notice going wrong.

## The gate, and that it has teeth

`python3 tools/wing_aero.py --gate` passes: the strips add up to the wing; the
two wings' lift directions are mirrors pointing up (to 2.2e-6); the body model's
own kinematics sweep both wings the same way at the same joint angle; the two
wings make the same force at the same activation; the flight force is ≥0.8 of the
weight and the hover α solves inside (20°, 70°); a symmetric beat's yaw stays
under 2 % of the steering signal; the asymmetry reverses the yaw with the physical
sign; the attitude is under 60°; the pools exist and a differential moves the
animal; and the reflex arc moves the steering pools.

Two rehearsals are part of the tool, and CI runs both and **requires them to
fail**:

* `--convention mirrored` — the right wing's stroke joint mirrored in the *body
  model*, so the two wings sweep against each other: **1 FAIL** (the in-phase
  check). The first version of this rehearsal only flipped a vector inside the
  tool and passed — because the drag cancels over a cycle whichever way each wing
  sweeps, so the model was wrong in a way that changed no number. The failure this
  check guards is in the model, so the rehearsal had to be in the model too.
* `--lift-from rest_plane` — the wing pitched away from the body model's rest
  plane instead of its stroke plane: **5 FAILs** (lift/weight 0.47, the hover α
  solving to 80°, the two wings no longer matching, the yaw no longer reversing,
  the symmetry floor at 108 % of the signal).

## What is not done

* **The wingbeat's kinematics are prescribed.** The stroke and the flip come from
  the animal's measured beat, held by the joint's own spring; the
  stretch-activated flight muscles and the thorax resonator that *generate* it in
  a fly are not modelled (ASSUMPTIONS #37). Everything the beat then does to the
  air and to the animal is the solver's.
* **No unsteady aerodynamics as such.** The coefficients are the animal's, so the
  measured force is in them, but the rotational peak at the flip and added mass
  are not resolved separately (ASSUMPTIONS #38). The peak-to-mean force ratio
  here is 1.9× against ~2.5–3× in the animal.
* **The tension pools** (6+6 cells) are in the connectome and are not driven or
  read.
* **The wing's own mass is a placeholder** (#35), and it is the one place where
  the *inertial* force — which is comparable with the aerodynamic force in a real
  fly — is wrong by a factor of eight.
* **Whether the haltere→wing-steering arc stabilises or destabilises yaw is not
  established** (above).
* Nothing here flies yet: the wings make weight support at one posture, and the
  posture is the next item's problem (attitude control with the halteres,
  item 8/12).
