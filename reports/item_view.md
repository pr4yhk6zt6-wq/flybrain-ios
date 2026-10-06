# The model was never in pieces: the camera was inside it

IMG_2715, v0.7 (7) on the device. Reported as: *"the model looks like it is not
connected, the fly seems to be missing half of itself, and the Map itself is odd
— sometimes grey, sometimes white."*

## What the screenshot itself settles

The two eye panes in that same screenshot show a fly standing on a grid,
correctly assembled: head, thorax, abdomen, six legs, wings, all attached and
all at sensible angles. The panes' cameras sit at 0.306 cm from the rig origin
with a 50° lens. The main view in the same frame shows a thorax filling the
screen with legs and wings crossing it at odd angles and pieces floating apart.

One scene, one pose, two cameras, two different-looking animals. **So the pose,
the meshes and the part-to-mesh binding are fine, and the main camera is not.**

## The measurement

`tools/audit_view.py` (new) does what the app does — reads `world/world.json`,
`world/fly.bin`, `world/frames.bin`, poses every mesh with the recorded
`(w,x,y,z)` quaternions exactly as `FlyWorld.apply(frame:)` does, puts the
cameras where `WorldRig.set(azimuth:elevation:distance:)` puts them — and reports
what lands inside the frustum. Frame 0, v0.7's own numbers:

| camera | distance | nearest surface | body in frame | body spans | geometry spans |
| --- | --- | --- | --- | --- | --- |
| home | 1.000 cm | 7.8 mm | **100.0 %** | 0.48 × | 1.32 × |
| **closest the pinch can reach** | **0.220 cm** | **0.04 mm** | **29.7 %** | **2.86 ×** | **36.9 ×** |
| pane left eye | 0.306 cm | 1.5 mm | 100.0 % | 0.55 × | 1.30 × |
| pane right eye | 0.306 cm | 1.6 mm | 100.0 % | 0.51 × | 1.09 × |

"Body" is the thorax, head and abdomen and their plates — the parts a person
reads as the animal. At the closest zoom the lens is 2.2 mm from the rig origin,
the body's bounding radius about that point is 1.24 mm, and a 38° lens **on a
phone held upright** sees a window 0.70 mm wide and 2.6 mm tall there. The
thorax alone is wider than the window; the rest of the animal is outside the
frame or behind the camera. That is the screenshot: not a broken model, a camera
three body-lengths inside the animal.

`reports/view/BEFORE_min_0.22cm.png` and `AFTER_min_0.82cm.png` are the same
frame from the before and after cameras: every mesh vertex drawn as a dot,
far-to-near, with the floor grid for scale. The first is the screenshot's view.

## Two more things in that screenshot

**Grey and white.** The floor was `SCNFloor` at white 0.93 with `reflectivity =
0.05` under a background of 0.965 — the same colour as each other. As the camera
orbits, the flat floor fills the screen at some angles and the background at
others, so the screen swaps between grey and white and the ground stops reading
as ground. The floor is now 0.82 with no mirror, the grid is 0.60 for contrast
against it, and the two surfaces are different colours on purpose.

**The legs looked disconnected** because the parts near the camera were cropped
open by the frustum and the parts outside it were not drawn at all; the
screenshot's panes, from further out, show them attached. The physics in the
same screenshot says the same thing — `feet 5/6 · COM z −0.0143 cm · 5 contacts`.

## The fix

The distances are a property of the animal, so they are measured from it rather
than written into the source:

* `tools/measure_view.py` (new) measures the body and the whole-animal bounding
  radii about the rig origin over the recording and writes a `view` block into
  `world.json`:

  | | measured | v0.7 had |
  | --- | --- | --- |
  | body radius (thorax + head + abdomen) | 1.240 mm | — |
  | whole animal radius | 2.926 mm | — |
  | `min_distance_cm` | **0.8185** | 0.2200 |
  | `home_distance_cm` | **0.9996** | 1.0000 |
  | `max_distance_cm` | **2.499** | 3.000 |
  | `pane_distance_cm` | **0.3059** | 0.3060 |

  The rules reproduce the two numbers that were already right — home 0.9996
  against 1.00, panes 0.3059 against 0.306 — which is the check that the rule
  is the right rule: the only camera it moves is the one IMG_2715 says is wrong.
  It has to be the *narrower* half-angle that decides: a phone held upright is
  taller than it is wide, SceneKit's `fieldOfView` is the vertical angle, and a
  rule that only looked at the vertical said "the body fits" while the thorax
  hung off both sides. The audit caught that, which is why the tool and the gate
  agree on the aspect as well as the angle.

* `FlyWorld` now reads those limits out of the manifest (`ViewLimits`), falls
  back to the v0.7 numbers for a world written before the block existed, and
  hands them to `WorldModel`, which no longer carries `minDistance = 0.22`,
  `maxDistance = 3.0` or `homeDistance = 1.0` at all. `WorldRig` takes the two
  fields of view and the pane distance from the same place.

* `WorldModel.render` centres the rig on the **vertex-weighted** mean of the
  posed part origins, not the plain mean: the legs, wings and bristles are many
  parts with small meshes, and counting them equally put the origin about a
  third of a body radius off the animal. `tools/measure_view.py` derives the
  limits about that same point, so the number written and the number used are
  the same point in space.

* `tools/audit_view.py --gate` is a CI step: every camera the app can reach must
  frame at least 99.9 % of the body, the closest may not make the body span more
  than 1.2 screens nor bring a surface within 0.2 mm of the lens, the furthest
  may not shrink the body below 0.08 screens, both panes must frame the body,
  and the home view must hold 90 % of the *whole* animal (bristle tips are
  allowed to crop; the animal is not).

The gate was rehearsed against the shipped numbers before it was relied on:
with the `view` block stripped — which is exactly what a v0.7 asset looks like —
it fails with

```
FAIL min (0.220 cm): only 29.7% of the body is in frame (it spans 2.86 screens)
FAIL min (0.220 cm): the nearest surface is 0.04 mm in front of the lens
```

## Corrections made while measuring

Recorded because both are the kind of mistake that produces a confident wrong
answer:

1. I first read the manifest's mesh table as `[offset, end]` — `thorax` is
   `[70258, 9158]`, which is only computable as `[offset, count]` — and
   concluded the app might be taking backwards slices. `FlyWorld.buildGeometries`
   already reads it as `[offset, count]`; the wrong reading was mine, in a
   scratch script, not in the app. `tools/audit_view.py` reads it as the app
   does and says so in a comment.
2. The first `measure_view.py` used only the vertical half-angle. The audit
   failed the numbers it produced (`min 0.378` still spanned 1.41 screens of
   body), which is how the aspect term got in. A tool and its checker disagreeing
   is the useful case; they were made to agree by fixing the rule, not the gate.
