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


---

# 2026-10-05 — the port fails its own golden trace (run 54)

For the first time since this file was written, `FlyDynamicsTests` *ran* in CI.
It had been calling `XCTSkip` in every green run, because `tools/pack_world.py`
deleted `FlyBrain/World/fly_body.json` — written by `tools/pack_body.py` four
seconds earlier — before the build. `Executed 7 tests, with 3 tests skipped`.

With the asset in the bundle the suite runs, and it reports:

| | Swift | reference | tolerance |
|---|---|---|---|
| worst joint angle | 0.1599 | — | 1e-7 |
| worst joint rate | 19.9966 | — | 1e-7 |
| root velocity | 23.1494 | — | 1e-7 |
| root angular velocity | 150.9716 | — | 1e-7 |
| root position | 0.5806 | — | 1e-7 |
| root quaternion | 0.9168 | — | 1e-7 |

`testTheAnimalStandsWithMuscleToneOnly` passes (6.05 s) — the static terms are
consistent — and `testTheMuscleModelIsForceBasedAndBraked` fails too, which was
a real bug and is fixed in the same commit: the lengthening branch of the
force-velocity term was `(1 + s)/(1 - 2s)`, a curve with a pole at s = 0.5 that
is negative past it. Clipped at zero it removed the braking torque exactly
where it was needed; `hill_velocity_factor` replaces it in both languages
(monotone, no pole, 1.8 saturation).

The golden trace itself is not the problem: it is reproducible to 1 ULP (a
one-ULP change in one torque moves the 500-step state by 2.2e-16 in q and
5.7e-14 in qd — measured again here, limits on *and* off), and the run is
limit-heavy rather than chaotic (4,044 hard-stop rate-kills; the final state
has joints pinned at their stops).

Diagnostics added with this commit so the next run localises it instead of only
failing:

* the test replays against the reference's 20 recorded samples and reports the
  **first** step where the two trajectories part, with the joint and both
  values;
* the reference now records its **hard-stop count** (4,044) and the port's is
  compared with it exactly — a structural fingerprint that does not depend on
  any tolerance;
* `FlyDynamics.limitStops` / `jointsAtLimit()` expose the port's own clamping.

### Resolved: two bugs, both settled against MuJoCo (2026-10-05)

The trace reproduces **exactly** now: all six final metrics are `0.000000`
against a tolerance of `1e-07`, the hard-stop count is 4,044 in both, and no
traced step diverges by more than `1e-9`. Two consecutive runs are
byte-identical. Both bugs were in the port, and the reference was used to
*find* them but not to *judge* them — that is what `tools/judge_port.py` is
for.

**Bug 1 — `Mat3()` is the identity.** `xform` and `crm` build their 6×6 as four
3×3 blocks, and both wrote `b: Mat3()` for the upper-right block. `Mat3()`
default-constructs the **identity**, so every spatial transform carried a
spurious identity in its top-right corner: `X` mapped a parent's linear
velocity straight into the child's, and `crm`'s motion cross-product lost its
zero block. The fix spells the intent out with a new `Mat3.zero`.

*How it was found.* The joint constants were identical in both languages (102
of 102 joints agree to 7.0e-14 relative over axis, stiffness, spring ref,
damping, armature, mass, COM, body pose, quat and all 36 inertia entries), and
so was every body's world pose (104 of 104 to 3.3e-16), which is why the
divergence had to be inside the recursion and not in the asset or the frames.
Dumping the backward pass body by body showed every body's `X` differing by
exactly 1.0 in one entry — the upper-right block — and nothing else.

**Bug 2 — the articulated-body inertia's rank-one term had the wrong sign.**

    the port:  Ia = IA[i] - (outer(Ui) * (-1 / Di))    =  IA + U U^T / D
    reference: Ia = IA[i] - np.outer(Ui, Ui) / Di      =  IA - U U^T / D

Featherstone 7.42 is the minus sign: the articulated-body inertia *loses* the
rank-one term the joint cannot see. With the plus sign a child's inertia is
*added twice* on its way up the tree.

*How it was found.* With `v[i]` and the bias `c[i]` proven identical body by
body, and `X`, `ci`, `D`, `U`, `IA` and `pA` compared body by body, the
deepest body whose record differed was the abdomen tip (`abdomen_7__abdomen_abduct_7`):
its parent-accumulated inertia was off by 2.44e-07 in the bottom-right block,
which is 2 · (U Uᵀ / D) for its child — i.e. exactly the difference between
adding and subtracting one rank-one term.

**The judge, not the reference.** The reference solver is verified against
MuJoCo (`--verify`: FK 4.4e-16, 20 ms free tumble, 1.5 s standing), so a
disagreement between the two did not say which was wrong. `tools/judge_port.py`
hands one state to all three implementations and reports who is closer. The
comparison is on the **joint rates after one step**, not on `qacc`: MuJoCo
handles joint damping implicitly inside its integrator, so its own `d.qacc` is
off by tens of percent on a step that is right to 2.5e-6 relative.

| one step, ten random states (joints moving, root moving, gravity on) | before | after |
| --- | --- | --- |
| port vs reference, worst over 10 states | 8.4e-01 rad/s | **1.6e-15 rad/s** |
| port vs MuJoCo | 2.6e+00 rad/s | **8.7e-04 rad/s** |
| reference vs MuJoCo | 8.7e-04 rad/s | 8.7e-04 rad/s |

After the fix the port's distance to MuJoCo *is* the reference's, digit for
digit, on all ten states — the two solvers are now the same solver. (The
harness clips every state inside every joint's range first: the port's hard
stops are a clamp, MuJoCo's are soft constraints, and comparing them on a state
whose joints are already outside their range measures the convention, not the
physics.)

`local/main.swift` keeps the modes that found this — `facts`, `kin`, `pass`,
`judge` — and `tools/fly_aba.py` now keeps the backward pass's working values
(`_X`, `_ci`, `_D`, `_u`, `_U`, `_IA_used`, `_pA_used`) so the next
disagreement can be read field by field instead of guessed at.

**Bug 3 — the animal was standing in the air.** With the dynamics fixed, the
golden trace reproduced exactly and the port's one-step rates matched the
reference to 1.6e-15 rad/s — and the standing test still failed, with
`com_z` **+0.038** (the reference: -0.023) and one foot on the floor out of six.
The port's initial state had the root at `z = 0`; the reference's `reset()` puts
it at `stance_root_z` (-0.004866 cm), the height the stance was measured at
(`tools/stand_body.py` measures the hold torques at that compression).

*How it was found.* The same state handed to both solvers:

| initial root z | contacts, port | contacts, reference | worst rate, port | worst rate, reference |
| --- | --- | --- | --- | --- |
| `0` (the port's initialiser) | **0** | **0** | 0.1482 | 0.1482 |
| `stance_root_z` (the reference's `reset`) | **4** | **4** | 0.2270 | 0.2270 |

The two agreed perfectly in *both* states — the only difference was which state
the port chose to start in. Forty-nine microns of clearance is enough that the
floor never touches the feet, so the servo that holds the stance holds it
against nothing and the animal launches itself within 2 ms. With `init` starting
at `stance_root_z`, running the standing test's exact scenario on both solvers
gives **identical** numbers: mean `com_z` -0.027384, first -0.026137, last
-0.027331, six feet down at every step, worst joint rate 2.644 rad/s.

Status of item 16 (`docs/AUDIT.md`): **DONE** — the trace is reproduced, the
port is verified against MuJoCo in its own right, and the animal stands where
the reference stands.
