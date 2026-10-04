# World mode — the fly in a body, in a place

World mode closes the loop that the brain view only half-closed. Instead of the
phone camera feeding the photoreceptors, the app renders a small 3-D world from
between the fly's eyes, feeds *that* to the retina, and lets the motor neurons
that come out the other end move the animal.

## The frame

Everything below happens once per displayed frame, inside one Metal command
buffer plus one readback:

1. **Render the world from the fly's two eyes** — one 128×128 offscreen
   texture per eye (`WorldRenderer.drawEye`). The eye cameras copy the
   flybody model: optical axes 67° off the body axis, 140° fields, so the
   pair covers ~274°. Earlier builds rendered ONE cyclopean camera from the
   head midpoint and fed that single image to both optic lobes — the brain
   got no left/right difference to steer with, the fly circled, and the eye
   preview showed a one-eyed animal. Now the left texture feeds the left
   hemisphere's retina cells and the right texture the right's (the split is
   by the side flag in `neuronMeta`, done once at load in `Connectome`), and
   the PiP shows both eyes side by side. The eye pass renders the world
   *without* the fly's own mesh (`encode(..., drawFly: false)`): the eye
   cameras sit inside the head, so drawing the animal put its own head in
   its field of view. A compound eye cannot see the head it grows on, and
   neither FlyVision nor CompoundRay render the observer into the view.
2. **Blur each eye to ommatidial acuity.** A real fly has ~700 facets per
   eye, about 1.5° apart, so a sharp render is far too good. `ommatidiaBlur`
   box-filters 6×6 pixel cells and collapses to a green-weighted luminance.
   It also re-projects each facet from **equirectangular angles through the
   capture frustum** — the mapping CompoundRay uses for compound-eye
   renderings. The raw frustum is rectilinear, so at 140° a direction 60°
   off-axis lands at tan(60°)/tan(70°) ≈ 0.63 of the half-width: the
   periphery is stretched ~2.7× and the view reads as a warped lens. A
   compound eye samples the sphere uniformly, so each facet is now sampled
   at its true angle and the periphery comes out straight.
3. **Step the brain** 4 × 1 ms with that texture as the photoreceptor input —
   the same `sampleRetina` kernel the phone camera used.
4. **Read the motor group rates back.** `reduceGroupSpikes` counts, on the GPU,
   how many members of each named group spiked; only ~35 uints cross the bus.
5. **Integrate the body** from those rates (`FlyBody.update`).
6. **Write the world back into the senses** — smell, taste, touch,
   proprioception, haltere, pain (`FlyBody.writeSensoryDrives`, which feeds the
   `driveGroups` kernel).
7. **Render the third-person view.**

## What drives what

| Body | BANC group(s) | Count |
|---|---|---|
| Flight thrust / lift | `motor_wing_power_left` + `_right` | 12 + 12 |
| Yaw in flight | `motor_wing_steering_right` − `_left`, minus a slowly adapted baseline | 12 + 12 |
| Walking speed | mean of front/middle/hind `motor_*_leg_*` | 391 |
| Turning on foot | right-side leg rate − left-side, minus a slowly adapted baseline | — |
| Walk/stop/groom bouts | not driven — imposed ethology layer (`FlyBody.Habit`) | — |
| Escape jump | `motor_jump_escape` (the giant fibre target) | 2 |
| Head turn | `motor_neck` | 49 |
| Feeding | `motor_proboscis` | 35 |

The adapted baseline (`turnBias`, tau 1.5 s) matters: the two leg populations
never fire perfectly evenly, and raw differential drive turned the animal in
circles at a steady ~11°/s forever. Only *changes* in asymmetry steer now —
the same equilibrium-reflex idea the wing lift loop uses. The flight yaw gets
the same treatment (`yawBias` on the left/right drag difference): without it
a sustained steering asymmetry pinned the fly at the 1600°/s yaw clamp and it
spun like a top, which the first open-world video showed at 1464°/s.

One renderer note: fog is computed **per fragment** from the interpolated
world position. Per-vertex fog was invisible while the floor was a 13 cm
quad, but on the open-world floor all four vertices sit ~20 m away and each
carried fog ≈ 1, interpolating the whole ground into the sky colour — the
frame read as "the map vanished". `tools/softrender.py` reproduces the GPU
pass (near-plane clip, no winding cull, perspective-correct depth) and is the
fastest way to see what the world pass will look like.

## The habits layer (imposed ethology)

A real fly does not walk continuously and it is not always locomoting:

| Behaviour | Measured value | Source |
|---|---|---|
| Walk/stop transitions | Poisson-like; baseline walk-initiation λ₀ ≈ 0.29 s⁻¹ | Demir, Kadakia, Anderson, Clark & Carey 2020, eLife 5:e57524 |
| Stop durations | ~exponential, floor at 300 ms | Demir et al. 2020 |
| Grooming share of waking time | ~13% | Lazopulo & Syed 2018, eLife 7:e34497 |
| Grooming bout structure | 0.15–2 s bouts, ~150 ms leg-sweep cycles, anterior→posterior | Seeds et al. 2014, eLife 3:e02951; Ray et al. 2019, PLOS Comput Biol 15:e1007105 |
| Grooming triggers | dusting / tactile bump strongly evokes it | Seeds et al. 2014 |

Like the tripod gait, a 1 ms LIF connectome does not spontaneously emit this
bout structure, so `FlyBody` imposes it: a three-state machine
(`walk` / `stop` / `groom`) driven by a **deterministic** xorshift32 rng
(reseeded on reset, so the physics tests reproduce exactly). Exponentially
distributed bout lengths — walk ~3 s, stops at the measured λ₀, grooming
~1 s — with landing and bumping evoking grooming, the 50% walk-exit split
tuned so grooming lands at the measured ~13% of active time. While stopped or
grooming, step frequency is gated to zero (the fly stands still); pivoting in
place is *not* gated, because stopped flies still reorient. The grooming pose
rubbing head then abdomen is posed, like the walking joint angles. The HUD
shows the state (`WALK` / `STOP` / `GROOM`).

The same determinism is what makes `testWalkBoutsAlternateWithStops` and
`testGroomingOccupiesAboutThirteenPercentOfActiveTime` assertable in CI.

### Arousal moves on its own

Vigour is not a constant, and it is not the experimenter's Synaptic gain
slider either — that one is the apparatus. The animal's own arousal state
drifts endogenously: spontaneous walking and flight wander through
high/low-vigour states over tens of seconds (Cohn et al. 2019, Cell 176:254),
brain-wide imaging finds arousal-like signals with time constants from under
4 s to over 20 s (Nat Commun 2023, 14:5420), and the walk/stop statistics
only close when a slowly varying internal state modulates the transition
rates (Demir et al. 2020). `FlyBody` therefore carries an
Ornstein–Uhlenbeck `arousal` (mean 1, tau 15 s, bounded [0.5, 1.5]) on the
same deterministic rng. It stretches walk bouts, shortens stops, and scales
walking speed (capped at the measured 30 mm/s). The HUD shows it as
`AROUSAL`, so you can watch the animal get restless and settle on its own.
`testArousalDriftsSpontaneouslyAndStaysBounded` pins the drift, the bounds
and the slowness; `testArousalStretchesWalkBouts` pins the sign of its
effect on bout structure.

## The lift trim, and why its time constant is 10 s

Lift is held against weight by an equilibrium reflex: a reference wing rate
(`referenceRate`) chases the commanded rate, and stroke amplitude is read off
the *ratio*. The time constant of that chase is the whole story:

- **No trim** (the original bug): a saturating motor population pins stroke at
  178°, lift stays above weight forever, and the fly glues itself to the
  ceiling.
- **tau = 2 s** (the second bug, from the user's video): the trim chased the
  fly's *own* climb command so tightly that lift was renormalised within a
  couple of seconds of any sustained power change. The wings visibly beat
  harder while the eye view never rose — the animal could hover at any height
  but could not climb to a new one.
- **tau = 10 s** (current): a seconds-long climb command raises the fly
  (240 Hz drive: 0 → 150 cm in ~5 s), while a stuck saturation is still
  re-trimmed on the ~10 s scale, so the ceiling-glue bug cannot return.
  `testSustainedClimbCommandRaisesTheFly` and
  `testLiftRetrimsAfterASustainedClimb` pin both halves of that behaviour.

Nobody has measured the real trim's time constant; 10 s is the value that
makes both failure modes impossible, and it is labelled as modelled.

Sensory, the other direction:

| World | BANC group | Count |
|---|---|---|
| Vision | `sensory_vision` (L1–L5, R7, R8) | 9,733 |
| Smell, inverse-square from food | `sensory_olfactory`, `sensory_antenna` | 3,000 / 4,230 |
| Taste while feeding | `sensory_gustatory` | 1,585 |
| Ground and bump contact | `sensory_tactile` | 6,698 |
| Gait and wingbeat feedback | `sensory_proprioception` + per-leg | 2,903 |
| Rotation in flight | `sensory_haltere` | — |
| The swatter | `sensory_nociception` | 170 |

## Measured versus modelled

This is the important distinction and it is deliberately kept visible in one
place, at the top of `FlyBody.swift`.

**From the connectome:** which cells are motor neurons, which muscle each one
innervates, which side of the animal it is on, and the entire synaptic path
from photoreceptor → optic lobe → descending neuron → ventral nerve cord →
motor neuron. None of that is invented.

**Invented by us:** the muscle. A connectome is wiring, not force. Turning
"wing power neurons at 120 Hz" into thrust needs a constant, and we chose it.
Likewise the tripod gait is imposed as an oscillator whose frequency follows the
leg motor rate — the central pattern generators are physically present in BANC,
but a 1 ms LIF network does not spontaneously produce clean gait.

All such constants are grouped and labelled in `FlyBody`:
`thrustPerHz`, `yawPerHz`, `walkPerHz`, `turnPerHz`, `liftThresholdHz`.

## Environments

Kitchen (wooden table, dark), garden (grass, bright sky, tall stems) and lab
(white bench, hard light) are switchable live. The ground checker scale differs
per environment on purpose: optic flow is what the visual system keys off, and a
featureless floor gives it nothing to work with.

Every environment is now an OPEN world: a floor that runs to a fog horizon under
an open sky, with loose scenery clustered a couple of metres around the origin.
There are no walls and no ceiling anywhere. The old 12 cm room was a matchbox
for an animal that flies at 0.9 m/s — it crossed the room in a sixth of a
second and spent its whole life colliding, which read as "the fly is broken".
`World.bounds` (20 m) survives only as an invisible analytical backstop so the
integrator cannot wander to infinity; the fog hides it completely. The sky cap
(5 m, also invisible) keeps the chase camera from losing the fly overhead.

## Interaction

- One finger — orbit the chase camera; two-finger pinch — distance; double tap — recentre.
- **Food** drops a crumb somewhere nearby. Nothing teleports the fly; it has to
  find it on the odour gradient.
- **Object** drops a cube from above.
- **Swat** brings a hand down on the fly's position, which drives nociception
  hard and is what the escape pathway is listening for.
- Two PiP windows (brain, fly eye) are independently draggable and their
  positions persist via `@AppStorage`.

## The fly's body

The animal is the Janelia/DeepMind `flybody` CT reconstruction — 41 rigid parts,
76 real hinge joints with anatomical axes and limits, packed by
`tools/build_flymodel.py` into `flymodel.bin` (1.7 MiB). One draw call per part.

Two things about it are worth knowing, because both have caused the fly to go
missing:

**Size.** `FlyModel` normalises the mesh, and it normalises the *assembled* pose
— it walks the parent chain, transforms every vertex, and scales the body length
that comes out to 2.5 mm. It used to measure the AABB of the raw vertex buffer
instead. That is not the animal: each part keeps its vertices in its own frame,
so the raw AABB is the union of 41 unassembled boxes and its longest axis is one
wing held out sideways. Scaling by it produced a 1.43 mm body and a 2.91 mm
wingspan — a fly at 57 % of life size, being flown by physics that assumed
2.5 mm. With the assembled measurement the same model gives a 2.50 mm body,
5.08 mm wingspan and 1.30 mm height, which are the published numbers for
*D. melanogaster*.

**It must be in the bundle.** `flymodel.bin` is a resource, not source, and CI
builds it. If it is absent, the renderer draws a stand-in fly from the same
primitives as the scenery and the HUD says `NO FLY MESH (fallback body)`. It
never again draws an empty room and says nothing. The `.ipa` is checked for
`flymodel.bin` before it is uploaded, so a build without it fails CI instead of
failing on a phone.

## Rendering cost

The whole world is three draw calls — one instanced cube batch, one icosphere
batch, one quad batch — plus a sky triangle, plus one draw per fly part. Instance
buffers are triple-buffered.

## If the fly is not there

1. **Check the HUD.** `NO FLY MESH (fallback body)` means `flymodel.bin` is
   missing from the bundle. Rebuild it: `python3 tools/build_flymodel.py
   --cache data/flybody --out build/flymodel.bin --budget 110000` and copy it
   into `FlyBrain/Resources/`.
2. **Double-tap** to recentre the chase camera, and pinch out if the fly is
   simply very close to the lens.
3. **`python3 tools/softrender.py out.png`** re-renders the frame on the CPU
   using the same instance list and the same shader maths, and prints how many
   pixels the fly covers. If it is there in the PNG and not on the phone, the
   problem is in the asset, not the geometry.

## Camera fixes shipped alongside

The brain view's camera was inverted on both axes: the old code *added* the drag
delta to azimuth and elevation, which rotates the camera with your finger and so
moves the object the opposite way. The correct polarity subtracts — the gesture
grabs the brain, not the lens. `Camera.swift` also adds a **Fly** mode with a
free camera (position + yaw/pitch) you can push forward into the neuropil;
switching modes preserves the exact viewpoint. Gesture handling moved to real
UIKit recognisers in `MetalViewBridge.swift`, because SwiftUI's `DragGesture`
cannot tell one finger from two.
