# Step 4 — the whole animal, in a world you can look at

Steps 1 to 3 progressively handed the animal over to its own nervous system.
Step 1 measured the body. Step 2 took the leg reflex out of the connectome.
Step 3 closed the loop on **one** leg and found that the animal still stands,
and that the leg resists being pushed.

Step 4 drives all **six** legs, each from its own cord, and then does the thing
the other three could not: it lets you watch.

```
python3 tools/step4_world.py      # ~7 minutes: six cords, a stance, a recording
python3 tools/serve_world.py      # then look at it
```

Everything the write-up quotes is generated:
[`reports/step4_world.md`](../reports/step4_world.md) and
`reports/step4_world.json`.

---

## Whose fly it is

**The body is not ours.** It is the real
[*flybody*](https://www.janelia.org/news/artificial-intelligence-brings-a-virtual-fly-to-life)
model from Janelia and Google DeepMind (Vaxenburg et al.), downloaded into
`data/flybody/` and used unmodified: 85 meshes scanned from a real animal —
head, the red eyes, the ocelli, thorax, seven abdominal segments, both wings
with their translucent membranes, the halteres, the antennae, the proboscis,
and all six legs down to the last tarsal segment and the claw. 272,550
triangles, 67 rigid parts, 109 degrees of freedom, with the masses, joint axes
and joint limits that model ships.

What this repository builds is the **controller**, and only the controller. The
loop is: the body model says where a joint is and how much load a leg carries;
a cord taken out of the BANC connectome decides what the motor pools do about
it; the pools drive the muscles; the muscles move the joints. There is no
trained network, no gait, no schedule, no clock, and nothing that knows where
the animal is supposed to go.

The meshes are packed down from 817,650 stored vertices to 132,506 distinct
points before they are sent to the browser. That is not simplification — the
model stores each triangle's three corners separately, so most of those
vertices are duplicates of each other — and the surface that comes out is the
same surface.

---

## What it does

| | |
|---|---|
| seconds of animal | 7.0 |
| net displacement, start to end | **0.559 units ≙ 2.07 body lengths** |
| length of the path it walked | 0.758 units |
| speed along that path | 0.40 body lengths per second |
| feet on the floor | 5.22 of 6 on average |
| fell over | no |
| the world pushed it at | 2.6 s and 5.6 s, 0.6 × body weight, for 120 ms |

**It stands. It does not walk.**

That is the honest result and it is the whole point of publishing a recording
rather than a claim. Over seven seconds under its own six cords the animal
travels two body lengths, and every millimetre of that is the world pushing it:
it is standing before the first push, it is still standing after the second one,
and in between it is shoved across the floor and stays upright. Remove the
pushes and it stays where it is.

The reason it does not walk is not a mystery and it is stated in
`docs/ASSUMPTIONS.md` as assumption #15: **the six cords are built
independently.** Each is the ≤3-hop subgraph around one leg's sense organs and
motor neurons. An interneuron that belongs to two legs' circuits exists twice,
once in each cord, and the two copies cannot influence each other. A tripod
gait is a *coordination*: it needs something that knows what the other legs are
doing. Here the only things that can carry that information are the floor, the
thorax they all hang from, and the tonic descending drive they share — and they
do not carry enough of it.

Load share and antagonist balance in the stance it found:

| leg | coxa_abduct | femur | tibia | share of body weight |
|---|---:|---:|---:|---:|
| front left | 0.183 | 0.250 | 0.860 | 29.0 % |
| front right | 0.212 | 0.404 | 0.937 | 10.8 % |
| middle left | 0.654 | 0.540 | 0.959 | 6.7 % |
| middle right | 0.621 | 0.171 | 0.827 | 10.4 % |
| hind left | 0.185 | 0.214 | 0.884 | 17.3 % |
| hind right | 0.486 | 0.554 | 0.853 | 25.8 % |

The stance is not found by hand. The cord's commands and the body's answer are
relaxed against each other for the whole animal at once, over four passes,
until every joint's commanded balance and its measured balance agree — which
they have to be solved together for, because six legs hang off one thorax and a
leg that takes more weight changes what every other leg reports. The load
shares above are lopsided (29 % on the left front leg, 6.7 % on the left middle
one) and nothing in this step makes them even. A real fly distributes its
weight; whether that is a property of the body, of the cords, or of the
connections between them is not something this experiment can say yet.

Per-leg contact duty cycle — the fraction of the recording each leg is carrying
load:

| leg | L1 | L2 | L3 |
|---|---:|---:|---:|
| left | 0.37 | 0.79 | 0.97 |
| right | 0.98 | 0.62 | 0.63 |

---

## How to watch it

```
python3 tools/serve_world.py
```

The viewer is a single HTML page with three.js vendored next to it — no build
step, no network access, no account. It loads the animal's geometry once
(6.3 MB) and 700 frames of pose (1.7 MB), and then:

* drag to orbit, scroll to zoom
* play / pause, scrub, and a speed control — the animal is recorded at 100 Hz
  and plays at a quarter speed by default, because at real speed a fly is a
  blur
* a HUD with the time, the distance travelled, the current speed, how many feet
  are on the floor, and the body's height
* the path it took, drawn on the floor, and a camera that can follow it

**The recording is a playback, not a live simulation.** Six cords stepped at
1 ms and physics at 100 µs do not run in real time on hardware anyone can
assume; twenty seconds of animal takes minutes to compute. Everything about
that is in the manifest rather than hidden: the frame rate, the number of
frames, the seed, and the exact command that produced it.

---

## What came out wrong

* **It does not walk.** See above; this is the headline finding and the first
  thing step 5 owes.
* **Three body weights knocks it over.** An earlier run pushed with 3 × the
  animal's weight instead of 0.6 ×. It was thrown 0.7 units — about seven
  millimetres, twenty-five body lengths — landed on its side with no feet on
  the floor, and lay there for three seconds until the second push stood it
  back up. The recording everyone will look at uses the gentler push, and the
  violent one is written down here instead of being quietly dropped.
* **The load share is lopsided and unexplained.** 29 % on one front leg and
  6.7 % on one middle leg is not a stance, it is a lean.
* **The left front leg is off the floor 63 % of the time.** Whatever that leg's
  cord settled into, it is not carrying its share of the standing.
* **The six cords do not share interneurons.** The connectome is one cord; this
  is six. Splitting it is what makes six of them fit in memory at once, and it
  is the honest description of what the code does, but it is a limitation and
  not a finding.

---

## The same world, inside the `.ipa`

The web viewer was never the point; it was the quickest way to look at the
recording. The world is now in the app, behind a **Map** button at the top
right of the brain view:

* `tools/pack_world.py` copies `world.json`, `fly.bin` and `frames.bin`
  (8.16 MB) into `FlyBrain/World/`, which `project.yml` lists as a *folder*
  reference so the three files travel together into the bundle. It refuses to
  run if either binary is a different size from what the manifest claims.
* `FlyBrain/Sources/FlyWorld.swift` reads them and builds one `SCNGeometry`
  per mesh — 85 meshes, 132,506 vertices, 262,180 triangles — re-basing the
  model's global `Int32` indices into each mesh and casting them to `UInt32`
  for `SCNGeometryElement`.
* `FlyBrain/Sources/WorldView.swift` poses them from `frames.bin` every
  display tick, slerping between the two frames either side of the current
  time. Nothing is keyframed.

**The two floating views.** `WorldRig` carries three cameras on one rig that
follows the animal: one you steer by dragging (and pinch to zoom), and two
fixed on the animal's own left and right, each looking back at it. The two
flanking views are `SCNView`s on the *same* `SCNScene` — they are not a second
simulation, they are the same animal seen from somewhere else. Each is a
floating pane you can drag anywhere on screen, and both can be hidden with the
button under the close cross.

The cameras are aimed with a quaternion built from an up vector of **+z**,
not by `SCNLookAtConstraint`, which assumes y is up and would roll every
camera onto its side in a world where the fly walks on the xy plane.

An `.ipa` built without the recording is not a broken app: the world screen
opens, explains which file is missing and tells you the command that makes it.
CI checks the three files are inside `FlyBrain.app` before the `.ipa` is
uploaded.

---

## What this step establishes, and what step 5 owes

**Establishes:** the loop can be closed for the whole animal at once — six
cords, eighteen joints, one body, no clock — and the animal stands on five or
six feet while doing it; the stance can be *solved* for six legs at once rather
than chosen; and a recording of a connectome-driven body under real physics can
be made, packed and looked at without a graphics card.

**Does not establish:** that the fly can walk. It cannot, and the reason is
known.

**Step 5 owes:** one cord for the whole ventral nerve cord, so the legs can
share interneurons and the question of coordination can even be asked; the
tarsus and the claw, which is where a real stance is decided; a muscle with
dynamics rather than a one-pole filter; and an explanation of the lopsided load
share.
