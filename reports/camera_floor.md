# The camera was allowed under the floor

*What `uploads/IMG_2715.png` was: the lens below the floor plane, and how the
range of tilts and distances the app allowed made that reachable. Fixed by a
rule, gated in CI, with a rehearsal that fails on the old rule.*

## What the screenshot measures

The shot is a phone screenshot of the Map screen, 828 × 1792, with the HUD
saying `2.1 s simulated`, `feet 5/6`, and `85 meshes · 262180 triangles` — so the
*model* was running and standing. What looked wrong was the view: the animal's
mass off-centre at the bottom, huge, with the two eye panes across it, a
cream-coloured region on the left with grid lines on it, and a grey region on the
right, also with grid lines, meeting in a near-vertical edge.

The measurable facts, taken from the PNG:

| measurement | value | what it means |
|---|---|---|
| the grid's spacing on the grey region | 54 px, axis-aligned | a face-on grid, i.e. the lens is close to the grid's normal |
| the grid's spacing on the cream region | 54 px, same lines | the *same* grid, so one plane carries both regions |
| the cream colour | `rgb(246,244,242)` | `FlyWorld.dress`'s background/floodlit floor. 0.82 white, fully lit |
| the grey colour | `rgb(157,157,157)` | the *same* floor, unlit — its underside |
| the brown animal's extent | 601 px wide of 828 | the animal at ~1.3 cm, which is *inside* the pinch's range |
| the boundary between them | straight, tilted ~2° | the floor plane's edge-on projection as the lens crosses it |

The two backgrounds the user reported — "sometimes grey, sometimes white" — are
the two sides of one plane. The floor is `SCNFloor` with a 0.82 diffuse and the
scene's lights are all directional (so every horizontal surface is lit the
same): from above it reads 246, from below it is lit by the ambient only and
reads 157. The grid is drawn from `.line` primitives, which SceneKit does not
back-face cull, so it stays visible either way — floating in the sky when the
floor beneath it is not drawn.

## The rule that allowed it

`WorldModel.orbit` clamped the tilt to ±1.45 rad (±83°) and `WorldRig.set`
clamped it again to the same band. **The distance was not part of the clamp.**
The eye's height is `target.z + distance · sin(tilt)` in a world whose floor is
at −0.132 cm and whose rig orbits the animal's centre at about −0.028 cm, so:

| tilt | distance | eye above the floor |
|---|---|---|
| −0.30 rad (−17°) | 0.819 cm (closest) | **−0.19 cm** — under it |
| −1.45 rad (−83°) | 0.819 cm | **−0.92 cm** — far under it |
| −1.45 rad (−83°) | 2.499 cm (furthest) | **−2.38 cm** — far under it |

`tools/audit_view.py --gate --old-tilt` prints the last row and fails, which is
the rehearsal: **the old rule must fail the gate**, and CI enforces that.

## The fix, and it is a band rather than a number

```swift
sin(tilt) >= (floor + margin − target) / distance
```

`WorldRig.elevationBand(floorZ:targetZ:distance:margin:)` returns that
`AngleBand`, and both places that can move the tilt — the drag
(`WorldModel.orbit`) and the pinch (`WorldModel.zoom`) — clamp against it as
well as the renderer. The margin is `WorldRig.eyeMarginCM = 0.05` cm: half a
millimetre, less than the animal's smallest part and more than the depth
precision at these ranges (ASSUMPTIONS #39).

What that gives, at the world's own measured numbers (floor −0.132 cm, target
−0.028 cm):

| distance | lowest tilt the floor allows | eye height then |
|---|---|---|
| 0.819 cm (min) | −3.8° | floor + 0.050 cm |
| 0.999 cm (home) | −3.1° | floor + 0.050 cm |
| 2.499 cm (max) | −1.2° | floor + 0.050 cm |

So the camera can no longer be below the ground the animal is standing on, at
any distance the pinch reaches — and the *tilt it can hold* changes with the
distance, which is the part that was missing.

## The second defect: the roll at the pole

`WorldRig.aim` builds the camera's basis from its position vector,
`x = up × z`, with a fallback to the world +x axis when that cross product
vanishes (a lens looking straight down or straight up). The fallback is a *step*
in the image's right axis — at azimuth 2.2 rad it is 0.59 in that axis, i.e. the
picture rolls as the tilt crosses the pole — which is the other half of "the
camera is far too fast to control" (`uploads/IMG_2714.png`, and the reason the
old fallback was worth removing rather than leaving).

The basis now comes from the two angles the user steers, not from the position:
the camera's right is the orbit's tangent `(−sin az, cos az, 0)`, orthogonalised
against the view direction. That is defined at every tilt including the poles,
and the image's right axis moves by at most 1.75° per degree of tilt —
`MeshPoseTests.testTheCameraBasisIsContinuousThroughThePoles` sweeps the whole
sphere at four azimuths and fails if it moves more.

## The HUD now says where the camera is

A screenshot of a broken view is only worth something if it carries the camera's
own numbers, so one HUD line prints them:

```
cam az +2.20 el +0.42 rad (+24°) 1.000 cm · eye z +0.362 · floor -0.132 · tilt floor -4°
```

`eye z` is the lens's height above the floor and `tilt floor` is how far below
horizontal the floor lets it go *at that distance*. If `eye z` is ever below
`floor`, the rule has a hole in it, and the next screenshot will say so.

## What was *not* established

* The composition of IMG_2715 — which azimuth, exactly which tilt and distance
  the user held — is **not** reproduced here. The numbers above bound what the
  camera was doing (a face-on grid at 54 px puts the lens ~1.3 cm from the
  floor, looking nearly along the vertical, which is *inside* the tilt the old
  clamp allowed); the exact pose is an inference, and the HUD line exists so the
  next one is a measurement instead.
* I could not reproduce the grey panel's *exact* on-screen shape offline. The
  colour and the grid spacing identify it as the floor seen from below with the
  grid floating over it; the shape depends on where the plane's projection
  lands, which is a function of the pose that was not recorded.
* The two eye panes are fixed 134 × 106 pt panes over the main view. At the
  closest distance the animal is larger than the panes and they cross it, which
  is legible in the shot and is *not* fixed here: it is a layout question (does
  a close-up want the panes), not a defect of the animal or the camera.
