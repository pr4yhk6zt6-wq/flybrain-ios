# ASSUMPTIONS

Every constant in this repository is one of two things:

* **measured** — published, or read out of a source file at build time by a
  script anyone can re-run. Cited where it is used.
* **an assumption** — listed here, numbered, with a reason and with the test
  that will catch it if it is wrong.

There is no third category. In particular there is no category for "it made the
walk look better", which is what the deleted body model was full of.

Step 1 introduces no constants of its own: everything `tools/step1_anatomy.py`
prints is read out of the MJCF or out of BANC, and everything `tools/step1_sim.py`
does is MuJoCo integrating the model as shipped.

**Current total: 22.** Six arrive with step 2, the first step that simulates
the nerve cord rather than measuring it. Eight more arrive with step 3, the
first step that joins that cord to a body — which means joining two vocabularies
(BANC's muscles, flybody's joints) and inventing a transducer between what the
body does and what the organ reports. Three more arrive with step 4, which
drives six legs at once and then has to admit what it had to leave out to do
it.

| # | assumption | value | why | test that catches it |
|---|---|---|---|---|
| 1 | synaptic scale | N\* = 12 partners at 100 Hz reach threshold | the raw weights are synapse counts, which have no units; one circuit-wide convention beats a per-connection fudge | every result is reported across a sweep of N\* (6/12/24/48). **Test result: the tibia reflex direction flips between N\* = 12 and N\* = 24, so that direction is not claimed.** |
| 2 | transmitter sign | `neurotransmitter_predicted`: ACh +1; GABA, glutamate, histamine −1; unknown 0 | insect convention (ACh excitatory, GABA/glutamate inhibitory); the column is a *prediction* from BANC's classifier, not anatomy | the sign-shuffle control (permute transmitters within cell class). **Test result: the shuffle takes the flexor pool from 2.44 to 23.78 Hz — the sign table is load-bearing, so this assumption is the largest weakness in step 2.** |
| 3 | cell model | conductance-based LIF; E_exc = +4, E_inh = −0.25, τ_m = 20 ms, τ_syn = 5 ms, t_ref = 2 ms, noise 0.015, threshold 1, rest 0 | additive LIF let inhibition reach −1.4× threshold, which inhibitory receptors physically cannot do; the values are the usual normalised units | report the all-silent and all-saturated boundaries. **Test result: silent for N\* ≥ 24, saturated for N\* ≤ 6, physiologically plausible at N\* ≈ 12.** |
| 4 | axonal delay | soma-to-soma distance / 0.25 m/s, clipped to 1–8 ms | the distances are measured from `root_position_nm`; the speed is a stated conduction velocity | median 1 ms, p90 2 ms — sub-millisecond differences do not change the sign of the readouts, and the tool prints the distribution |
| 5 | descending tone | a standing fly's brain drives its descending neurons at 2.5 (threshold units); a sensory population under test gets +3.0 | the descending neurons are the only cells that carry the brain's state into this subgraph, and a standing fly's cord is not silent | sweeping the tone from 1.0 to 3.0. **Test result: below 2.0 the motor pools are silent, above 3.0 they saturate; 2.5 is the first value where both tibia antagonists are active together, as they are in a standing leg.** |
| 6 | scope of the circuit | the ≤3-hop neighbourhood of the leg's organs and motor neurons: 38,181 of 188,508 cells | the full connectome would make the sign of every 4-hop path meaningless | a cell that matters to the reflex but sits 4+ hops away is absent. Not yet measured; this is the known hole. |
| 7 | muscle activation | τ = 60 ms, a first-order filter on each pool's spike rate | a joint cannot follow individual spikes, and insect skeletal muscle fuses twitches over tens of milliseconds | swept at 20 and 200 ms, with the jitter of the commanded angle reported at each — the tremor is the assumption made visible |
| 8 | the antagonist command law | `ctrl = q_rest + (joint range) × (b − b_ref)`, where `b = ā_raiser / (ā_raiser + ā_lowerer)` | the joint's own anatomical range is the only scale the body model supplies, and a ratio does not care how *large* the rates are — so assumption #1, the synaptic scale, drops out of the loop entirely | the N* sweep: neither the standing pose nor the reflex moves with N*. And the standing test: at the measured `b_ref`, the joint sits at the pose step 1 measured |
| 9 | muscle → joint | `POOL_ACTION` in `tools/step3_closedloop.py`; the trochanter is fused to the femur in *Drosophila*, so its muscles act on the coxa–femur joint | BANC names muscles, flybody names joints, and the only honest thing between them is the muscle's anatomical action | each joint's flexion and protraction direction is **measured on the assembled animal**, and every pool is either placed on a joint or listed under `pools_not_driven` |
| 10 | proprioceptive tone | 2.5, the same number as the descending tone | a loaded leg's proprioceptors are tonically active, and re-using the brain's tone means this step adds no new magnitude of its own | the organ gain κ is swept, which spans the same range a change in the tone's modulation depth would |
| 11 | chordotonal transduction | `drive = tone × (1 − polarity × κ × x)`, floored at zero, with `x` the knee's angle away from standing over half its range; κ = 1; polarity = flexion stretches the organ | the polarity is published (FeCO is stretched by flexion, relaxed by extension); κ is genuinely unknown, because a real FeCO is range-fractionated over a working range narrower than the joint's whole range | swept at κ = 1 / 4 / 16, and the polarity is run both ways. **Test result: see `docs/STEP3.md` §5.** |
| 12 | campaniform transduction | `drive = tone × (this leg's normal ground-reaction force ÷ the force it carries standing)` | that is what a strain gauge is; there is no free gain in it | the load conditions (0 and 2 body weights) test it directly, and the measured force is reported at every condition |
| 13 | which organ sees what | chordotonal ← femur–tibia angle; campaniform ← this leg's ground reaction force | the FeCO spans that joint; campaniform sensilla report cuticular strain from load | single-channel conditions run each on its own, and both are reported |
| 14 | scope of the body | one front leg's three proximal joints are driven by the cord; the other five legs, the tarsus and the claw stay at the rest command step 1 used | the loop has to close somewhere, and a leg is the natural unit | the standing test, and the pools that are left out are listed rather than dropped |
| 15 | the six cords are built independently | step 4 builds one cord per leg: each leg's ≤3-hop subgraph is extracted and stepped separately, so an interneuron that belongs to two legs' circuits exists twice, once in each cord | the ventral nerve cord is one cord; splitting it by leg is how six of them fit in memory at once, and it is the honest description of what this does | step 3's N* sweep shows the readout does not depend on the synaptic scale, and the standing test shows six independent cords can still agree on a thorax. **The consequence is not tested away: six cords that cannot talk to each other is the reason the animal does not walk.** |
| 16 | what pushes the animal | the world applies an external force to the thorax, at a time and for a duration stated in the JSON; nothing in the controller knows the time | a stimulus belongs to the world, like gravity and the floor do; a behaviour is not something you schedule | the pushes are listed in `reports/step4_world.json` under `model.nudges_ms`, and the viewer's HUD is on the same clock |
| 17 | what the viewer shows | `world/` is a playback of a simulation that has already been run, not a live one | six cords at 1 ms and physics at 100 µs do not run in real time on hardware this repository can assume; a recording is honest about being a recording | the frame rate, the number of frames and the command that made them are all in `world/world.json` |
| 18 | what the app's world is | `FlyBrain/Sources/FlyWorld.swift` reads the *same* `world.json`, `fly.bin` and `frames.bin` the web viewer serves, and poses the *same* 85 meshes; the only thing it invents is the pose between two frames, which is a slerp of the recorded quaternions | it is a viewer, not a second model — if it drew anything of its own, the numbers in `reports/step4_world.json` would no longer describe what you are looking at | `tools/pack_world.py` refuses to ship unless the two binaries are exactly the size the manifest says, and `ios.yml` step "The .ipa must contain its data" fails the build if the three files are not inside `FlyBrain.app`. A disagreement shows as a load error with a message, never as a different animal |
| 19 | the two floating panes | `WorldRig` puts two more cameras in the *same* scene, on the animal's own left and right (+y and −y, because the fly faces +x), and points each back at it with a quaternion computed from an up vector of **+z**, not SceneKit's default +y | the body model walks on the xy plane with z as the sky, so SceneKit's look-at constraint — which assumes y is up — would roll every camera onto its side | the main view and both panes are `SCNView`s on one `SCNScene`: they cannot disagree, because there is only one animal in it. The up vector is a constant in one place, `WorldRig.up` |
| 20 | the cord's update interval | the cord is asked once per **millisecond of simulated time**, and the excitation it returns is held between updates | 1 ms is the connectome's own timestep and the interval step 3 and step 4 ran the cord at; a real ventral nerve cord does not re-decide at the physics' 100 µs, and the muscle fuses twitches over tens of ms (#7) | the interval is a property of `FlyLiveBody.cordIntervalMs` (default 1.0), the HUD reports `cordUpdates`, and the body's own real-time factor is reported beside it — so a device that cannot keep up shows a slower cord, never a coarser one |
| 21 | the excitation gain `K` | `excitation = stance + K × (b − b_ref)`, K = **0.5** | in the Python experiments the command was a joint *angle* and the joint's own range did the scaling; here the actuator is a force pair in [-1, 1], so the dimensionless balance has to be turned into an excitation. 0.5 means a pool pair swinging from all one side to all the other moves the excitation by the full range, and half of that from the measured stance | the closed-loop tests: with both antagonists equally active the command is exactly the stance and reaches both directions symmetrically from it (`FlyCordTests`), and with every pool silent the command is exactly zero, so the animal holds its tone rather than drifting |
| 22 | the loop's ends are named, not assumed | every pool and organ the cord drives is asked of the connectome **by the name `tools/motor_pools.py` gave it**, and a name the connectome does not have is reported, never read as silence | the body asset and the connectome are written by different tools; a name that means one thing in one file and something else in the other makes the cord read 0 Hz, and 0 Hz is what a quiet pool reads as — the failure would look like a plausible animal doing nothing | `tools/verify_loop.py` (run in CI, on the files the app ships) checks every pool and organ group against the built `flybanc_meta.json`, that none is empty, and that every pool lands on a hinge the asset has. It caught the first version of this code, which resolved pools through the wrong joint name and would have driven nothing |

**Not modelled at all, and therefore not claimed:** gap junctions (electrical
synapses), neuromodulation, synaptic plasticity, and any per-connection weight
that is not the synapse count. Step 4 additionally does not model the
connections *between* the legs: each leg's cord is separate, so whatever
coordination a real tripod gait needs has to come out of the body, the floor
and the shared descending tone — and, as the recording shows, it does not. Muscle *dynamics* is modelled as the one-pole
filter of assumption #7 and nothing more — no force–length, no force–velocity,
no series elasticity. The body's actuators are the affine position actuators
`flybody` ships, so the model inherits their stiffness as if it were muscle.

---

## The two source files

| file | what it fixes | licence |
|---|---|---|
| `flybody/fruitfly/assets/fruitfruit.xml` | the body: 67 parts, 102 DOF, masses, joint axes and limits, the 8 adhesion actuators | Apache-2.0 |
| `banc_888_meta.feather` | the nerves: which cell is a motor neuron, which muscle it innervates, which side it is on, where the soma sits, and the predicted transmitter that assumption #2 rests on | CC BY 4.0 |
| `connections_princeton.csv.gz` | the synapses: who contacts whom, and how many release sites | CC BY 4.0 |

## Removed, and why

The whole previous stack was deleted in the commit that created this file. The
constants it carried — `thrustPerHz`, `yawPerHz`, `walkPerHz`, `turnPerHz`,
`liftThresholdHz`, the arousal time constant, the habit-layer bout rates, the
tangent gait gains — were gains chosen so that a hand-imposed oscillator would
produce motion that looked like walking. They are preserved in the
`archive/round5-world` branch. None of them are coming back; behaviour in this
repository is produced by circuits, not by a schedule.
