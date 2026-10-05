# ASSUMPTIONS — FlyBody.swift, round 5

Every number in `FlyBody.swift` is one of three things:

* **measured** — a published *Drosophila melanogaster* value, cited inline
  (mass 0.96 mg, I_yaw 5.2e-13 kg m², wing 2.39 mm / 1.96 mm², r2 = 0.58 R,
  218 Hz wingbeat, C_L 1.8, C_D = 1.27 C_L, stroke plane 29° in forward
  flight, hover amplitude solved from force balance at 141°, 178° morphological
  limit, 1600 deg/s saccade peak, 30 mm/s walk ceiling, 13 Hz step frequency,
  duty 0.55, 0.9 m/s escape take-off, TT-motoneuron identity from BANC v888);
* **derived** — computed from the CT model or from the measured set by a
  script in `tools/` that anyone can re-run (`gait_geometry.py`: gait
  neutrals, protraction signs, amplitudes, secant gains, stance knots, leg
  track; `build_flymodel.py`: every joint range);
* **an assumption** — listed here, numbered, each with its reason. The code
  cites these numbers inline ("ASSUMPTIONS.md #n").

If you add a constant to FlyBody.swift, it lands in one of the three boxes
or the review fails.

---

## The assumptions

**#1 — Rate→amplitude is a straight line.**
`strokeAmplitude(from:)`: below the reference rate, amplitude scales linearly
with collective rate; above it, a second straight segment reaches the 3.1 rad
morphological limit at 1.5× reference. Both endpoints are measured (force
balance at hover; joint limit from the CT model); the shape between them is
assumed because no published rate/amplitude curve exists for the population
level.

**#2 — Stroke-plane tilt is linear in thrust.**
`strokeTilt(for:)` = 29° · clamp((F/W − 1)/0.5, 0, 1). Hover (F = W) holds the
stroke plane horizontal — required so the equilibrium reflex converges on a
true hover — and the measured ~29° forward-flight inclination is reached at
1.5 body weights. The straight line between the two measured endpoints is
the assumption.

**#3 — Neuromuscular low-pass, τ = 0.2 s.**
The wing and leg decodes see a one-pole filter of the motoneuron population
rate. The smallest wing motor group has 12 neurons, so its reported rate
jumps 5.2 Hz per single spike; a 0.2 s window averages ~144 spikes at 60 Hz
firing and brings that counting noise under 10% before it reaches the muscle.
Muscle time constants of this order exist in the literature, but the exact
population-level τ is not published. It is an assumption; T2 (noise floor
< 200 deg/s) is the test that keeps it honest.

**#4 — Equilibrium reflex τ = 10 s, clamped to 5–400 Hz.**
`referenceRate` closes the haltere/visual lift-trim loop at the muscle
(Sherman & Dickinson 2003 show the pathway; BANC contains it) because the
loop's true gain cannot be calibrated from the connectome alone. It reads
only the COLLECTIVE — never a left/right difference. 10 s is long enough
that a seconds-long climb command survives, short enough that lift re-trims
to 1 W after a sustained command (both are tests).

**#5 — Leg drive → step frequency gain ×2.2.**
f = min(13 Hz, collective/reference · 13 · 2.2). The 13 Hz ceiling and the
speed–frequency linearity are measured (Mendes et al. 2013); the ×2.2 that
sets where mid-range drive saturates the frequency is an assumption chosen so
a ~50 Hz leg-motoneuron population (about half the observed walking range)
already walks at the preferred speed.

**#6 — Gait geometry: derived, with reported residual slip.**
Neutrals, protraction signs, amplitude caps, secant gains and the stance
inverse-map knots are re-derivable from the CT model by
`tools/gait_geometry.py` (true FK travel, not a small-angle tangent gain).
At the 30 mm/s ceiling the morphology runs out of range and the planted feet
slip: **T1 55%, T2 73%, T3 14%** of stride. This is reported, not hidden;
T4 (testStanceFootDoesNotSkate / simcheck #19) measures T3, the leg with the
longest reach, and holds it inside 25%. A real fly's middle legs contribute
more lateral than fore-aft stroke; reproducing that needs joints (femur roll,
tarsal chain) this gait does not yet drive.

**#7 — Escape refractory 0.1 s.**
Card & Dickinson (2008) measure the whole take-off in ~14 ms; a tenth of a
second between launches is a conservative floor that keeps a rattling TT
population from machine-gunning the animal. The 10 Hz event threshold itself
is derived: one TT spike in one 60 fps frame reads as 30 Hz raw, and the
engine's 0.35 EMA peaks a single-spike-per-frame burst at ~10.5 Hz.

**#8 — Escape counts in the air, fires only on the ground.**
A spike event mid-flight increments `jumpEvents` (the signal is never
discarded; the HUD shows it) but applies no impulse — there is nothing to
push against. A behavioural assumption about what the giant fibre "means",
not a measurement.

**#9 — Head/abdomen decode divisors and angular limits.**
headYaw = clamp((neck − 20)/60, ±0.35 rad), headPitch = clamp((neck − 25)/90,
±0.3 rad), abdomen = clamp((abdomen − 15)/80, ±0.25 rad). The 20–25° head
yaw and ~15° abdomen excursions are in the measured ranges of the CT joints;
the population-rate scales (the /60, /90, /80) at which those limits are
reached are assumptions — no published rate/angle calibration exists.

**#10 — Contact tolerance 20 µm.**
`contactEpsilon` = 0.002 world units: how close to the floor counts as
touching. A numerical tolerance, not biology.

**#11 — Visual banking is cosmetic.**
Body roll during asymmetric flight ((φR − φL)/2, τ ≈ 1/6 s) and the pitch
during climb are animation state; the physics never reads them. Real flies
do bank into turns, but the law used here is not measured.

**#12 — Collisions stall, they do not bounce.**
On impact the normal component of velocity is removed, the remainder is
scaled by 0.3 and yaw rate by 0.5. Flies stall on surfaces rather than
rebounding; the exact coefficients are assumptions. Nothing can tunnel: the
`contain()` backstop re-clamps every frame.

**#13 — Animation smoothing constants.**
The `min(1, dt·k)` one-pole rates on head (5/4), abdomen (3), airborne (4),
roll/pitch (6/8) and the wing draw offsets (0.02 rad phase, ±0.8 flip, 0.25
roll) affect only what is drawn, never the physics. Assumptions by category.

**#14 — Opponent-decode noise floors are derived.**
`wingEps = 1/(12·τ)` and `legEps = 1/(3·63·τ)` are the rate change ONE spike
makes in the smallest contributing group over the low-pass window (wing
groups have 12 neurons; the leg drive averages three groups whose smallest
has 63). Listed here so nobody "tunes" them: they move only with the group
sizes in the connectome or with #3.

---

## Removed in round 5 (previously magic, now gone)

| Old constant / layer | Status |
|---|---|
| gravity × 0.02 in the "walking" branch | **Removed.** One continuous dynamics model; contact comes from the collision solver, the floor normal cancels gravity exactly. |
| `jump > 8 && airborne < 0.6` escape trigger | **Removed.** Replaced by the spike-event threshold + refractory (#7, #8). |
| yawBias / turnBias adapted baselines | **Removed.** They silently cancelled sustained turn commands (the H2 cover-up). Asymmetry now decodes through the opponent index onto the measured 20° steering authority. |
| Habit layer (walk/stop/groom timers, groomPhase) | **Removed** as scripted behaviour. Walk, stop and grooming must come from the connectome's own activity. |
| Arousal OU process | **Removed** with the habit layer. |
| ×2.0 wing-steer multiplier | **Removed** with the opponent decode; steering maps onto the measured 20° range directly. |
| Tangent gait gains (0.565/0.453/1.253 mm/rad) and their postures | **Replaced** by true-FK secant gains and postures from `gait_geometry.py` (#6); the tangent linearisation over-estimated stride travel ~2× at stride-scale amplitudes. |
