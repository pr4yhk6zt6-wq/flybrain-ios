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

---

# The switch, as built

*Added with the commit that wires the loop on the phone. The design above is the
first piece (the pools, in the connectome the app reads); this is what the app
does with them.*

## One name per thing, written once

`tools/motor_pools.py` now owns both ends of the loop, because the app has to ask
for both by name:

| what | who writes it | what it is called |
|---|---|---|
| a muscle pool | `build_banc.py` → `pool:T1_left:tibia_flexor` | the group the connectome answers `groupRate()` for |
| a sense organ | `build_banc.py` → `organ:T1_left:chordotonal` | the group the body drives with `setGroupDrive()` |
| the same pool, in the body | `build_body.py` → `legs["T1_left"].pools[...]` | the same string, from the same function |

`tools/build_body.py` and `tools/step3_closedloop.py` now read the pool table out
of `motor_pools.py` instead of keeping copies, and `step2_reflex.py` reads the
organ table from there too. The first version of this work had the body asset
naming pools `motor_front_leg_left_tibia_flexor` while the connectome named them
`pool:T1_left:tibia_flexor` — a loop that reads silence, and silence looks exactly
like an animal doing nothing. `tools/verify_loop.py` exists so that cannot happen
again: it checks every name the app will ask for against the built
`flybanc_meta.json`, that none of those groups is empty, and that every pool lands
on a hinge the body asset actually has. It already caught one bug in this port.

Measured, on the built files (`python3 tools/verify_loop.py`):

```
the loop: 6 legs · 42 pools placed · 0 pools the body could not place · 12 organs
the connectome names 60 pools and 18 organs; the body asks for 42 and 12
```

The 18 pools the body does not ask for are the tarsus and long-tendon sets: they
are in the connectome, and step 3's mapper places muscles on the three proximal
joints a leg actually has (docs/ASSUMPTIONS.md #9).

## What the app does now (FlyBrain/Sources/FlyCord.swift)

Once per millisecond of simulated time, and not once per physics substep:

1. **body → organ.** The chordotonal organ is told where the knee is —
   `x = (angle − rest) / halfSpan`, drive `= tone × (1 − polarity × κ × x)`, so
   flexion (which stretches it) silences it. The campaniform sensilla are told
   what the leg is carrying, in units of the force it carried standing. Both
   numbers come out of the body asset (the joint's own rest angle and range) or
   out of the solver's own state — never out of a constant in the app.
2. **the organs drive the connectome**, by group, through the same
   `SimulationEngine.setGroupDrive()` the eye already used.
3. **cord → muscle.** Each pool's rate is filtered with the muscle's 60 ms time
   constant, and the balance of each joint's antagonists,
   `b = up / (up + down)`, becomes an excitation offset
   `K × (b − b_ref)`, clipped to the muscle's range. Because it is a *ratio*, the
   synaptic scale N\* is not in the loop at all (assumption #8).
4. **`b_ref` is measured on the device**, not chosen: the first 300 ms run with
   the organs clamped at their standing value, exactly as step 3 measured the
   reference it reported (`mode="clamped"`). The HUD says `measuring the stance`
   until it is done.

Nothing decides anything: there is no gait, no state machine and no schedule in
the file. The pools are the animal's own (BANC's motor neurons, by muscle), the
sign of every pool is the one measured on the assembled animal
(`Body.measure_directions`), and if the cord is silent the animal holds the tone
its own actuators were measured holding.

## The loop's shape, as the app builds it

`FlyCordTests` (in CI, on the simulator):

* every pool the loop drives is placed on the hinge the asset names it on, and
  the loop commands exactly three joints per leg;
* the connectome is asked for every name the loop uses, and a name it does not
  have is **reported**, not read as a silent pool;
* with both antagonists of every joint equally active, the balance is the middle
  and the command is exactly the stance — nothing moves;
* with one antagonist firing alone, the command moves the joint one way, and
  with the other, the same amount the other way: the loop is symmetric about the
  stance it measured;
* with every pool silent the command is exactly zero — the loop does not invent
  a behaviour out of silence;
* the chordotonal organ reports the joint it spans and the campaniform organ
  reports the load, and flipping the published polarity of the chordotonal organ
  reverses what it reports — step 3's control experiment, on the phone's code
  path.

## What is not claimed

* **The animal does not walk**, and this step does not make it walk. What changed
  is that nothing is missing from the loop any more: organs → cord → muscles →
  body → organs runs on the device, with the same names at both ends. Whether a
  gait comes out of it is step 7, and it is the next measurement.
* **The reflex through the app's own engine has not been measured yet.** The
  Python side measured it (step 3: ρ = +0.125 and +0.186, resolved; −0.073 and
  −0.087 when the polarity is reversed). Repeating that *through the app's engine
  and the app's pool groups* needs a device or a longer CI run, and it is the
  first item of the acceptance list above that is still open.
* The cord's real-time factor is whatever the device manages. It is measured and
  shown (`cord × real time` on the Body screen), never assumed: a phone that
  cannot run 1 ms of connectome per millisecond runs the loop slower, and says so.
