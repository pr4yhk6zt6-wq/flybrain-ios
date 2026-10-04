# World mode — the fly in a body, in a place

World mode closes the loop that the brain view only half-closed. Instead of the
phone camera feeding the photoreceptors, the app renders a small 3-D world from
between the fly's eyes, feeds *that* to the retina, and lets the motor neurons
that come out the other end move the animal.

## The frame

Everything below happens once per displayed frame, inside one Metal command
buffer plus one readback:

1. **Render the world from the fly's head** into a 128×128 offscreen texture,
   through a 140° frustum (`WorldRenderer.drawEye`).
2. **Blur it to ommatidial acuity.** A real fly has ~700 facets per eye, about
   1.5° apart, so a sharp render is far too good. `ommatidiaBlur` box-filters
   6×6 pixel cells and collapses to a green-weighted luminance.
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
| Yaw in flight | `motor_wing_steering_right` − `_left` | 12 + 12 |
| Walking speed | mean of front/middle/hind `motor_*_leg_*` | 391 |
| Turning on foot | right-side leg rate − left-side | — |
| Escape jump | `motor_jump_escape` (the giant fibre target) | 2 |
| Head turn | `motor_neck` | 49 |
| Feeding | `motor_proboscis` | 35 |

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

## Interaction

- One finger — orbit the chase camera; two-finger pinch — distance; double tap — recentre.
- **Food** drops a crumb somewhere nearby. Nothing teleports the fly; it has to
  find it on the odour gradient.
- **Object** drops a cube from above.
- **Swat** brings a hand down on the fly's position, which drives nociception
  hard and is what the escape pathway is listening for.
- Two PiP windows (brain, fly eye) are independently draggable and their
  positions persist via `@AppStorage`.

## Rendering cost

The whole world is three draw calls — one instanced cube batch, one icosphere
batch, one quad batch — plus a sky triangle. The fly itself is assembled from
those same primitives (thorax, abdomen, head, two eyes, two wings, six legs), so
it costs nothing extra. Instance buffers are triple-buffered.

## Camera fixes shipped alongside

The brain view's camera was inverted on both axes: the old code *added* the drag
delta to azimuth and elevation, which rotates the camera with your finger and so
moves the object the opposite way. The correct polarity subtracts — the gesture
grabs the brain, not the lens. `Camera.swift` also adds a **Fly** mode with a
free camera (position + yaw/pitch) you can push forward into the neuropil;
switching modes preserves the exact viewpoint. Gesture handling moved to real
UIKit recognisers in `MetalViewBridge.swift`, because SwiftUI's `DragGesture`
cannot tell one finger from two.
