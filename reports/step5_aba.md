# Step 5 — results

Reference solver `tools/fly_aba.py`, port `FlyBrain/Sources/FlyDynamics.swift`,
asset `build/fly_body.json` (run #6, written by `tools/build_body.py`), MuJoCo
3.14 stepping `data/flybody/.../fruitfly.xml` for comparison.

`python3 tools/fly_aba.py --verify --mjcf .../fruitfly.xml --out reports/step5_aba.json`

## 1. Forward kinematics against `mj_forward`

8 random poses (seeded), every body's world position and orientation.

| | |
|---|---|
| worst position error | **4.441e-16 cm** |
| worst orientation error | **1.998e-15** |

This one is a frame-convention test, not a tolerance question: the body tree,
the joint axes, the order in which a body's fixed quat and its joint rotations
compose (`R = R_parent · quat(body) · Π quat(axis, q)`, right-multiplied in the
order the asset lists them), and the position/rotation split of the free joint.

## 2. Twenty milliseconds of free fall and tumble against MuJoCo

Zero torque, no floor, both sides at 100 µs, gravity, joint springs, the model's
own damping and armature.

| | |
|---|---|
| first step | **1.749e-08 rad / 1.749e-04 rad/s** |
| worst body position over 200 steps | **2.109e-05 cm** |
| worst centre of mass over 200 steps | **7.001e-09 cm** |
| worst joint rate after 200 steps | 3.235e-01 rad/s (from 1 rad/s) |
| measured separation rate | **1.039× per step** |

The joint *rates* at the end of the run are reported rather than asserted, and
the reason is measured rather than assumed. The model's joint inertia is its own
1e-6 g cm² armature at the lightest joint while accelerations reach 6.8e5 rad/s²,
so a light joint's acceleration is a cancellation across the whole chain and
*nobody* reproduces it from published quantities to machine precision: MuJoCo's
own step disagrees with its own `mj_fullM` + `qfrc_smooth` by 4e-2 relative on
`haltere_left` (4.06e3 rad/s² absolute) and by 1.2 relative on `abdomen` if the
damping is folded into the diagonal instead. What is asserted is the first step,
the positions and the centre of mass — what gravity acts on — and those stay
exact. `tmp/fall_ref.py` and `tmp/fall_terms.py` are the measurements behind
that paragraph.

A by-product worth recording: MuJoCo's Euler step does **not** equal
`solve(mj_fullM + dt·diag(dof_damping), qfrc_smooth)`. On a single hinge the
relation `qacc = qfrc_smooth / (M + dt·damping)` holds to 3.6e-12; on the fly it
does not (the abdomen and haltere joints disagree by more than the factor the
damping can explain), so the implicit form is verified in isolation
(`tmp/damp_semantics.py`) and used as the reference only where it is exact.

## 3. The animal standing for 1.5 s on muscles alone

The animal is started at the stance `tools/stand_body.py` measured — by standing
it up *in this solver* — with its own measured muscle tone, held by a stance
servo whose gains are read off the discretisation
(`omega_dt²·I/(dt²·cap)`, `kd_dt·I/(dt·cap)`) and ramped in over 50 ms, at the
model's own timestep.

| | |
|---|---|
| centre of mass z | −0.022892 → **−0.022809 cm** (stance −0.022900) |
| its standard deviation over the last 0.75 s | **2.95e-05 cm** |
| feet carrying load | **6.0 of 6** at every sample (minimum 6) |
| floor force / weight, last half | **1.017** |
| contacts | 6 at every sample |
| worst joint rate | **0.66 rad/s** |
| drift | **0.31 mm** in 1.5 s |
| fell over | no |

n = 1 run, deterministic: fixed step, fixed initial state, so the deviations are
the animal's own micro-motion and not noise. The 0.31 mm of drift is the animal
settling *itself*, not a drift of the solver: the centre of mass moves by 0.0003
cm vertically while it does so.

## Where the stance comes from, and why it had to be re-measured

`tools/build_body.py` reads the stance off **MuJoCo's** settle — but a stance is
only a stance for the floor it was measured on. At the MJCF stance this solver's
floor gives the front-left claw a penetration of 2.6e-03 cm and a load of 0.42 of
the animal's weight, where MuJoCo's soft contact gives it 3.7e-04 cm and 0.23.
The net force is right and the load *distribution* is not, and the difference is
0.2 of a claw's load through the tibia and femur — 0.01 of torque on a joint
whose inertia is the model's own 1e-6 g cm² armature. Measured, from the MJCF
stance the animal pushes itself a millimetre into the air within 20 ms, whatever
the harness does.

`tools/stand_body.py` therefore measures the stance by standing: the animal is
placed at the MJCF stance, held by its calibrated muscle tone plus a stance
servo, and this solver's own dynamics find the pose where the contact forces
balance the weight. The settled pose is averaged over the last 100 ms and written
as `stance_q` (**0.0222 rad from the MJCF pose at the worst joint, 0.0003 cm
horizontally between the two**), together with the excitation that held it
(`stance_excitation`, largest 0.92 of 1, **no joint saturated**). The pose is
stationary to 3e-05 rad and 2.4e-06 cm over that window, the floor carries 1.000
of the weight on six feet, and the residual is reported rather than forced to
zero.

## What was found on the way, because it changes the numbers

* **The contact damper was the instability, and it was found by bisection.**
  A penalty contact's damper is *explicit*, so a coefficient `b` on a joint whose
  inertia is the armature (1e-6 g cm²) grows by `b·dt/m = b·1e2`. Measured on a
  100 ms stand: `b = 0.5` leaves `|qd|` = 900 rad/s and climbing with the animal
  thrown clear of the floor; 0.1 gives 6.6 rad/s; **0.02 and below stand**
  (|qd| ≤ 2.8 rad/s, floor carrying 0.966-0.967 of the weight, centre of mass
  within 2 µm of the stance). The value shipped is 0.005 — zeta ~ 0.2 against the
  leg's own stiffness — and the limit to argue with is `b < m·2/dt`.
* **The harness's gains are derived, not tuned.** Before the servo was a servo
  it was a hand-picked gain, and a hand-picked gain on a joint this light *is*
  the instability: `0.5·(0 - qd)` in the harness grows to |qd| 4e2 in 3 ms. The
  gains are read off the discretisation instead, and with them the animal holds
  its pose to 3e-05 rad over 100 ms.
* **Everything here was measured, not argued, and that is why the earlier
  numbers in this repository are void.** `verify_aba.py` reported 1.44e-01 on
  the mass matrix against `mj_fullM` for a week: that is `dt·dof_damping`, which
  MuJoCo adds to the diagonal *before* the solve and returns through `mj_fullM`
  afterwards — the check now strips it by name (`--no-damping`). The 4.00e-05
  acceleration residual under "armature only" was the 1e-6 armature itself, not
  an error.
* **The muscle table is the file's, not an invention.** `max_torque =
  |gainprm[0]|·max|ctrlrange|` and `passive_stiffness = −biasprm[1]` for the 62
  joints the MJCF gives actuators: femur 0.8–1.6, coxa 1.36, wing pitch 1.0,
  tibia 0.54, head 0.05, with a passive 0.4–0.8 at the leg joints. Against a
  1e-6 g cm² joint these are what make a leg a structure instead of a hinge.
  The other 40 joints have no actuator in the file; their limits are the
  placeholder (`docs/ASSUMPTIONS.md`), marked per joint in the asset.
* **Muscle geometry is the file's too.** `biasprm[1]` makes the control zero the
  stance, and the joint's own `qpos_spring` is the rest angle, so the
  force-length curve is centred there (`optimal_angle` in the asset) instead of
  at 0, and as wide as the joint's anatomical range.
* **Gravity is a force, not a root acceleration.** The two agree for a fixed
  base and every textbook writes it the other way round; for a free root they
  differ, and the difference is invisible in every static test and wrong in
  every falling one.
* **The force-velocity term is per direction.** One shortening curve applied to
  both directions zeroes the torque past `V_MAX` for either sign of command, so
  a joint that is already moving cannot be braked by anything but its damper and
  the animal comes apart in 200 µs. That was a real bug in this file, found by
  `tmp/dt_probe.py` printing the excitation pinned at ±1 while the torque went to
  zero.

## Environment

MuJoCo 3.14, Python 3.12, single thread, Linux x86-64, NumPy 2.3. The CI runs
the same three checks on macOS and fails the build on any disagreement
(`.github/workflows/ios.yml`, the step-5 job).
