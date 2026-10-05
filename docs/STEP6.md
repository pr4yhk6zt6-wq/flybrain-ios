# Step 6 — the nerve cord drives the muscles

Step 5 ended with an animal that stands: the solver matches MuJoCo, the floor
model is fitted, and the muscles hold the posture they were measured holding.
The animal does not yet *do* anything, because nothing is telling the muscles to
change. That is this step, and it is the one the whole project is for: the Map
screen should show behaviour that the connectome produced, not a gait someone
wrote.

    proprioception ──> sensory pools ──> the cord ──> motor pools
         ^                                               │
         │                                               v
      the floor <── contact / joint state <── muscles <── excitation

## What exists already

| piece | where | state |
|---|---|---|
| the cord | `FlyBrain/Sources/SimulationEngine.swift` (Metal LIF, 1 ms) | running: 175,237 neurons, 2,171,713 edges, real BANC v888 |
| named populations | `Connectome.Group`, `SimulationEngine.groupRate(_:)` | already read off the GPU per named group |
| the sense organs | `docs/STEP3.md`, `tools/step3_closedloop.py` | chordotonal on `tibia_<leg>`, campaniform on the feet, measured in Python |
| the pools and their wiring | `tools/motor_pools.py`, `tools/step2_reflex.py` | measured: which motor neurons innervate which muscle, and what the organ does to them |
| muscles | `FlyDynamics.muscleTorque`, `FlyLiveBody.drive` | the hook is there and unused: `drive(proprioception) -> excitation offset per joint` |

## The pool table, as measured

`tools/motor_pools.py` holds the muscle sets step 2 reflexed through and step 3
drove the animal with. `build_banc.py` now emits, for each of those sets on each
leg and side, one **named group** of motor neurons, named
`pool:<leg>_<side>:<muscle_set>` — so the app asks the question it actually has
("how fast is the pool that extends this leg's tibia firing?") through the group
machinery it already has, with no new Metal code and no second copy of the
wiring on the phone.

Measured over BANC v888 (`tmp/pool_check.py`):

* **60 pools** — 10 muscle sets × 6 legs. **None empty.**
* **371 motor neurons**, each in exactly one pool on its own leg (no overlaps).
* Of those, **351 are in the thoracic neuromeres** and **20 have no neuromere
  assigned** in the meta table (they carry a leg effector part and a muscle
  name, so they are in the pool; the predicate does not filter on neuromere).
* The T1–T3 neuromeres hold **369** motor neurons labelled front/middle/hind leg
  effectors, so the pools cover **351 of them**, and **18 are in no pool** —
  they innervate muscles this table does not name.
* Per leg the pools hold, e.g. front-left: 17 tibia flexor, 2 tibia extensor,
  11 trochanter flexor, 8 trochanter extensor, 2 + 4 coxa rotators, 6 femur
  reductor, 8 long tendon, 3 tarsus depressor, 1 tarsus levator.

The excitatory/inhibitory sign is **not** in this table. The antagonist pair for
a joint is two pools, and which of them is "further flexion" is measured on the
assembled animal (step 3), because the MJCF's joint axes are whatever the CT
scan produced.

## The design

1. **Sensory drive.** The chordotonal organ on each tibia reports joint angle as
   `tone·(1 − polarity·κ·(q − q_ref))`; the campaniform sensilla under each foot
   report load as `tone·(leg normal force / reference)`. Both go into the engine
   as per-group drive (`setGroupDrive`), which is the same path the retina uses
   for the eyes — the cord receives them through named populations, not through
   a controller's variables.
2. **The cord steps.** 1 ms per step, as it already does.
3. **Read the pools.** After each cord step, `groupRate("pool:...")` per pool.
4. **Muscle activation.** Each pool's rate is low-passed with the muscle time
   constant (`tau_mus` 60 ms) — that filter is the muscle, not a controller
   smoothing — and the antagonist balance becomes one joint's excitation:
   `exc = tone + gain·(pool_up − pool_down)` clipped to the muscle's range.
5. **The body steps.** `FlyLiveBody` advances the physics; the excitation is
   held for the substeps in one cord step (10 substeps at dt = 1e-4 s).

Two things this must **not** become, both from the project's own rules: there is
no gait table, no phase clock, no `if (leg == .frontLeft) { lift }`, and no
replacement of the cord by a controller. If a leg moves, it is because a motor
pool spiked.

### The one engineering compromise, declared

A cord step per physics step would need a GPU readback per 1 ms. On a phone that
stalls the pipeline, so the cord runs in **bursts** (K ms at a time) and the
muscles hold the excitation between bursts. The muscle's own 60 ms low-pass
means the closed loop's bandwidth is ~2.7 Hz, so a burst of a few milliseconds
is far above what the loop can see — but the number will be reported as a
measured real-time factor, not hidden.

## Acceptance tests for this step

* `tools/verify_muscles.py` (new): with the cord running and a leg loaded, the
  pools that the literature says should answer do answer — the resistance
  reflex at the femur–tibia joint, as step 2 measured it in the connectome, now
  measured **through the app's own engine and app's own pool groups**.
* The excitation that reaches each muscle is produced by pool spikes: a run with
  the cord's motor pools silenced must produce no behaviour beyond the tone.
* Nothing in the app decides a leg's state: the test asserts that the only path
  from the cord to the body is `FlyLiveBody.drive`.
* The step is not accepted on a video. It is accepted on a trace: pool rates,
  joint angles, foot loads, and the delay between a load change and the muscle's
  answer.

## Order of work

1. `tools/build_pools.py`-equivalent data in the bundle (done: the groups are in
   `build_banc.py`) + the pool names and the joint map in Swift, checked against
   the meta JSON by CI.
2. `FlyCord.swift`: proprioception → drive groups → step → pool rates → muscle
   activation. Unit-tested against a Python replay of the same 100 ms.
3. Wire `FlyLiveBody.drive`; the HUD gains a motor-pool readout (the debug views
   come before the polish, per the project's priority order).
4. Measure: a leg loaded and unloaded, and the reflex's latency and sign.
