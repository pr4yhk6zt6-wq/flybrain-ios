# Item 7 — the halteres, as a gyroscope

**Status: the pair is wired, the geometry is measured from the body model, and a
rotation moves the two afferent populations of the shipped connectome —
measured. What has *not* been shown is that the animal then uses it to walk
straight; that is item 17.**

`docs/AUDIT.md` had this PARTIAL: *"Haltere bodies exist physically; no
haltere-derived angular-rate signal enters `FlyProprioception`."* The bodies
existed and nothing read them. Now the body's own angular velocity — taken into
the body frame, once per cord update — goes through the pair's measured
sensitivity axes onto `sensory_haltere_left` and `sensory_haltere_right`, every
simulated millisecond.

## What a haltere is, and what came from where

A haltere is a hindwing turned into a vibrating structure: it oscillates through
about 90° at wingbeat frequency (200 Hz in *Drosophila*; Lehmann & Dickinson
2005, *J. Exp. Biol.* 208:3075) and, when the body rotates, the Coriolis force
pushes it **out of its stroke plane**, which the campaniform fields at its base
transduce (Pringle 1948, *Phil. Trans. R. Soc. B* 233:347; Nalbach 1993,
*J. Comp. Physiol. A* 173:509, for the kinematics that make the dorsal field the
one that sees it). The force is `F = 2 m (Ω × v)` — **linear in the angular
rate**, which is why a haltere is a rate sensor and not an accelerometer — and
the direction it pushes along is

```
a = v̂ × n̂        v̂: the haltere's velocity, n̂: the normal of its stroke plane
```

which is the quantity each haltere measures, per unit of body rotation.

**Measured, from the body model's own MJCF** (`data/flybody`, loaded with MuJoCo
exactly as `tools/build_body.py` loads it — the model, not the XML text):

| | |
| --- | --- |
| haltere mass | **0.822 µg** each (of a 985 µg animal) |
| hinge → centre of mass | **0.172 mm** (right 0.176) |
| hinge axis, body frame | left (−0.748, −0.081, −0.659), right (0.758, −0.096, 0.645) |
| the two sensitivity axes are plain mirrors | cos(a_left, S·a_right) = **1.0000** |
| stroke: 90° at 200 Hz | peak speed of the CoM **17.0 cm/s** |

And what that means for the problem Pringle posed — separating a rotation from
the forces that drive the stroke:

| rotation rate | Coriolis force | as a fraction of the in-plane force |
| --- | --- | --- |
| 50 °/s | 0.244 nN | 0.18 % |
| 100 °/s | 0.488 nN | 0.35 % |
| 400 °/s | 1.951 nN | 1.41 % |
| 1000 °/s | 4.878 nN | 3.54 % |

(the in-plane force, `m ω² r`, is 137.9 nN and comes back on every stroke.) That
is why the organ is a specialised field of campaniform sensilla and not a
generic mechanoreceptor, and it is the number the report should carry.

**From the literature, labelled:** the stroke itself (frequency, arc) and the
relative stroke phase of the two halteres. The model's halteres are **passive
stubs** — one hinge joint with a ±0.2 rad range and no actuator at all — so
there is no stroke to measure, and a hinge axis has no *direction*, so the
geometry cannot say whether the two tips move together or oppositely. That one
degree of freedom is the whole sign structure, so both conventions are run:
`--convention antiphase` (what the animal does) and `--convention mirrored` (the
model's plain-mirror geometry), and the difference is printed rather than
asserted.

## The pair, and why a difference means anything

Fox, Fairhall & Daniel (2010, *Front. Neural Circuits* 4:123) modelled what the
pair is *for*: the Coriolis forces from yaw and roll are **out of phase** between
the left and right halteres, while pitch and the centrifugal force are **in
phase**, so left-minus-right receives yaw and roll and cancels pitch. The model's
own geometry supports it:

| rotation | difference channel (L−R)/2 | common channel (L+R)/2 | leakage |
| --- | --- | --- | --- |
| roll | **−0.569** | −0.002 | 0.3 % |
| pitch | +0.001 | **+0.585** | 0.1 % |
| yaw | **+0.578** | −0.002 | 0.4 % |

Under the opposite stroke convention yaw moves into the *common* channel and
pitch into the difference — the gate fails on that, which is the rehearsal that
matters, since a fly that confused the two would steer into a gust.

## What the connectome does with it

`tools/haltere_gyro.py` drives the two groups on the shipped connectome
(`build/flybanc.bin`, 175,237 cells, gain 12, 1000 ms after 100 ms of warm-up,
seed 42) and counts:

| condition | left Hz | right Hz | difference | common |
| --- | --- | --- | --- | --- |
| still | 23.19 | 47.90 | −24.72 | 35.54 |
| yaw left 100 °/s | 29.88 | 23.57 | +6.31 | 26.73 |
| yaw right 100 °/s | 14.17 | 57.75 | −43.58 | 35.96 |
| **yaw left 400 °/s** | 55.52 | 0.40 | **+55.12** | 27.96 |
| **yaw right 400 °/s** | 0.00 | 86.30 | **−86.30** | 43.15 |
| roll left 400 °/s | 0.00 | 87.95 | −87.95 | 43.98 |
| **pitch up 400 °/s** | 60.59 | 91.33 | −30.74 | **+75.96** |

Read against the still animal, a 400 °/s yaw moves the difference by **+79.8 Hz**,
a 100 °/s yaw by **+31.0 Hz**, and a 400 °/s *pitch* moves the **common** channel
by +40.4 Hz while moving the difference by only −6.0 Hz. The error on the
difference over 20 × 50 ms bins is **±1.3 Hz** (still) and ±1.5 Hz (yaw), so
every one of those is a resolved signal.

**Two things this is not.** First, the two pools are not interchangeable: at
rest, on the *same* drive, the right fires 47.9 Hz and the left 23.2 Hz. That is
a property of their different connectivity (216 and 212 cells, 738 against 647
first targets, only 57 shared) and it reproduces across seeds to a few hertz —
which is exactly why every rotation is measured *against the baseline* rather
than against zero. A gyro that read "how fast is the right haltere pool
firing" would be reading that offset. Second, the populations *beyond* the
afferents do not move with the rotation in this window (the 3,396 cells watched
in `beyond` move by less than 1 Hz with the sign inconsistent) — so this item
does not claim a haltere→wing→steering reflex. What it claims is stated below.

## The pathway, structurally

The reflex literature's shortest circuit is a monosynaptic haltere afferent
input to a wing steering motor neuron (Fayyazuddin & Dickinson 1996,
*J. Neurosci.* 16:5225). Read out of the packed connectome's CSR — the
afferents' own outgoing synapses, not an assumption:

`ascending` 319, `descending` 44, `motor_neck` **31**, `motor_haltere` 18,
`motor_wing_steering_right` **10**, `motor_wing_steering_left` **8**,
`motor_abdomen` 7, `motor_wing_tension_left` 5, `motor_wing_tension_right` 5,
`motor_front_leg_right` 1.

So the pathway the reflex needs exists in the data, one synapse deep, on both
sides. The gate fails if those direct steering or neck targets ever disappear.

## How small a turn the pair can see

A ladder of yaw rates, the drive straight from the pair (exact algebra), the
rate through the connectome:

| yaw | drive difference | rate difference | |
| --- | --- | --- | --- |
| 0 °/s | +0.00 | +0.00 Hz | |
| 5 °/s | +0.12 | +8.18 Hz | above the floor, but *not* monotone |
| **10 °/s** | +0.23 | **+5.43 Hz** | monotone from here up |
| 25 °/s | +0.58 | +9.02 Hz | |
| 50 °/s | +1.16 | +13.91 Hz | |
| 100 °/s | +2.31 | +31.03 Hz | |
| 400 °/s | +9.24 | +79.84 Hz | |

The noise floor is three standard errors of the still animal's own difference,
measured on the same window: **±3.9 Hz**. The **smallest turn this pair resolves
in one second is 10 °/s (0.17 rad/s)**. The 5 °/s rung is printed and marked as
not evidence: it moves the rate more than 10 °/s does, because at that drive
individual cells cross threshold and the population's response is quantised —
the same effect item 11's receptor ladder showed. Gating on the *rate* rather
than on the drive is what makes that visible instead of hidden: the drive is
monotone at every rung by construction, and a gate that only checked the drive
would pass a fly whose receptors cannot resolve the signal.

## Two rehearsals, and both fail as they should

1. **The opposite stroke convention** (`--convention mirrored`): 7 failures,
   starting with *"yaw(z) lands in the wrong channel"* and *"pitch moves the
   difference channel as much as yaw does (+93.0 against −3.7 Hz)"*.
2. **A gyro wired to nothing** (`--drive-gain 0`, i.e. the body turns and no
   current reaches the afferents): 5 failures, including *"a 400 deg/s turn moves
   the difference between the haltere pools by +0.0 Hz"* and the resolution
   check.

## In the app

* `FlyLiveBody.proprioception()` now carries the body's angular velocity in the
  **body frame**, in degrees per second: `quatToMat(rootQuat)ᵀ · ω`, computed
  once per cord update. Nothing else about it is invented.
* `HaltereGyro.swift` holds the pair's geometry; `FlyCord` builds it from
  `world.json`'s `haltere` block (`tools/haltere_gyro.py --write`), so the app
  does not keep a second copy of a measurement the tool makes.
* `FlyCord.update` drives **both** populations every simulated millisecond, with
  the afferents' lag (one first-order pole, 5 ms — assumption #31). The lag
  lives in the cord loop and not in the render path, because a reflex whose
  timing depends on the display's frame rate is not a reflex.
* A build missing either haltere population says so on the HUD
  (`no haltere afferents in this build`) and drives neither — half a gyro is not
  a gyro, and one live pool beside one missing pool would read as a fly that
  cannot feel its own rotation for no visible reason.
* The HUD carries `gyro roll +0 pitch +0 yaw +0 deg/s · halteres 3.00/3.00
  (L-R +0.00)`: the sensed rate and the two drives, because their difference is
  the yaw signal and their sum is the pitch signal.
* `FlyBrainTests/HaltereGyroTests` pins the port to the tool with a 9-row golden
  table (yaw each way, pitch, roll, a rate below the floor, a mixed rotation),
  and a CI step re-derives that table from the tool and fails if the embedded
  numbers have drifted. **It already earned its keep:** the first version of
  `HaltereGyro.swift` typed the right haltere's axis by hand as the plain mirror
  of the left's, and the table failed on the sign of one component — a cross
  product is a *pseudo*-vector, so a mirror returns it negated, and the two axes
  are (+, +, −) and (−, +, +), not negatives of each other.

## What is not done, stated rather than hidden

* **No reflex.** Nothing steers with this. The measured population beyond the
  afferents does not move with the rotation in a one-second window, so "the
  haltere signal reaches the wings" is a *structural* claim (one synapse,
  measured above) and not a rate measurement. Items 6 and 17 are where flight
  and walking are; this item exists so that a turning body is something the
  nervous system can feel.
* **Walking halteres are not modelled.** The haltere stroke here is the flight
  regime. A walking fly's halteres may not oscillate at all (they are free to
  move and the phase relation between the two is much more variable), and a
  non-oscillating haltere is not a Coriolis sensor. This assumption is written
  in `docs/ASSUMPTIONS.md` #30 and not glossed.
* **The drive map is an approximation.** `bias + gain·(Ω·a)/100 °/s` is the
  simplest map with the right linearity and limits (assumption #31), not a
  measured campaniform dose–response.
* **One pole is not a receptor.** The 5 ms lag stands in for a population that
  fires phase-locked to a 200 Hz stroke; the real population's timing carries
  phase information this model throws away.
* **The pools' asymmetry is real and unmodelled.** The left/right offset
  (−24.7 Hz at rest) is a property of these two annotated populations. A real
  fly's haltere afferents are two halves of one bilateral organ; a model whose
  left and right populations are 216 and 212 cells from an EM volume inherits
  whatever the reconstruction happened to contain.
