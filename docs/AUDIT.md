# AUDIT — where this repository stands against the brief

Read this before changing anything. Nothing here was rebuilt from scratch; the
inventory below is of the tree as it stood at `3353a74`, reconciled with the
fixes that landed on `main` while it was written (`7b05f8b`…`2beb443`: the
part-axis fix, the no-recording policy, the two-bar UI, the step-6 motor-pool
work) and with the scale fix in the commit that carries this file. Every claim
names the file or report it comes from. Status values:

* **DONE** — the link exists, is tested, and no scripted path bypasses it.
* **PARTIAL** — exists, but with a named gap.
* **MISSING** — not present; the brief requires it.
* **NEXT** — the gap this audit puts at the top of the queue.

---

## 0. Why the phone showed what it showed

The two screenshots that prompted this audit are from **build `575faf1`
(step 4)**, the last run that went green. Three facts pin that down:

1. `acbdd33` (step 5) failed CI at `Build for the simulator`:
   `WorldView.swift:216: invalid redeclaration of 'body'` — fixed at HEAD by
   renaming the stored property to `animal`.
2. The screenshot's HUD text ("the fly, in its own world", "moved 0.00 units")
   and its **scrub slider + 0.1×/0.25× speed buttons** only exist in the step-4
   recording playback, which `acbdd33` deleted.
3. The "Map" pill drawn *on top of* the ≈power readout is the old pair of
   absolute overlays that HEAD merged into one `HStack`.

So the recording ("a wiggling video") and the HUD collision were already being
fixed when the screenshots were taken — but the build never shipped, and one
worse bug was still hiding underneath. Two more bugs were fixed in the
meantime on `main` and are kept here: **the part axis** (`3047e92` — the
floor is recorded as part 0, every renderer skipped it and then counted for
itself, so each mesh wore the pose of the part before it; the manifest now
*writes* `part` and CI asserts it) and **the recording policy** (`b20c200` —
the .ipa may not carry `frames.bin` at all, asserted in CI).

### The bug that would have shipped with step 5 anyway

**`geom_size` was being used as a scene-graph scale.** For mesh geoms MuJoCo's
`geom_size` is the axis-aligned bounding box *half-extent* — measured off the
compiled model, `thorax = [0.0439, 0.0584, 0.0601]` cm against a raw mesh AABB
of `[0.0439, 0.0527, 0.0580]`. Both `FlyWorld.apply(frame:)` and the three.js
viewer multiplied every part's vertices by that number, shrinking each part to
4–9 % of its true size: a scatter of slivers and dots where the animal should
be. `apply(live:)` never reset the scale, so the live-solver path inherited the
same wrong scale from `init`'s `apply(frame: 0)`.

![projection of frame 77 at scale 1 (left) and scale = geom_size (right)](img/audit_scale_projection.png)

The figure projects `frames.bin` frame 77 through the app's own camera
(azimuth 2.2, elevation 0.42, distance 0.6, 38° fov, 755×1572): left is
correct (a fly), right is what the app drew and what the screenshot shows.
The fix is to scale by **1** — the vertices packed by `tools/step4_world.py`
are already in model centimetres, which is also what `geom_size` and
`geom_rbound` are computed from.

**Where the "twenty times life size" reading came from, and why it is the
wrong file.** The `.obj` assets under `data/flybody/.../assets/` really are
ten times the compiled model — the thorax measures 1.16 units *there*. But
`tools/step4_world.py` packs `model.mesh_vert`, which MuJoCo has already
scaled at compile time: the thorax in `fly.bin` is 0.116 units, life size,
and `geom_rbound` (0.0946) matches the raw half-diagonal (0.0898), so nothing
multiplies the vertices again at render time. Measured on the file the app
actually draws: thorax extent `[0.0439, 0.105, 0.116]` in `fly.bin` against
`geom_size = [0.0439, 0.0584, 0.0601]` — `geom_size` is the half-extent, and
multiplying by it is what produced the sliver screenshot.

---

## 1–28, item by item

| # | Brief item | Status | Evidence / gap |
|---|---|---|---|
| 1 | Emergence, not scripted behaviour | **DONE** (app), **NEXT** (loop open) | No `WalkController`/`if foodDetected` anywhere; the old ethology layer lives only on `archive/round5-world`. `FlyLiveBody.drive` is the sole behaviour entry point — and **nothing sets it yet**, so the animal stands on muscle tone alone. Closing that loop is the top biological task. |
| 2 | Species, adult anatomy | **DONE** | Unmodified Janelia/DeepMind flybody: 67 parts, 102 articulated DOF, 78 actuators, 0.985 mg, 2.97 mm — asserted in CI (`Step 1 — assert the anatomy`). |
| 3 | Priority order (nervous system → … → graphics) | **DONE** | CI gates run anatomy → reflex → closed loop → world → solver before the app builds; graphics deliberately minimal. |
| 4 | Lightweight anatomically structured 3-D model | **DONE** | 85 scanned meshes (head, both eyes, antennae, proboscis, 3 thoracic segments, both wings, both halteres, A1–A6+, 6 legs) packed by `tools/step4_world.py`. Scale bug fixed in this commit. |
| 5 | Leg segments, joints, contact, adhesion | **DONE** | coxa→trochanter→femur→tibia→tarsus→tarsomeres→claw in the asset; per-segment mass/inertia/joint limits from the MJCF; 74 penalty contacts + 8 adhesion actuators; `AnatomyTests` pins all of it. |
| 6 | Wings as actuatable structures, aero hooks | **PARTIAL** | Both wings have joints, muscles and mesh; `FlyDynamics` carries wing-joint torque like any joint. **Gap:** no aerodynamic force model and no wing-beat state — architecture must accept motor→muscle→aero→body when flight is attempted. |
| 7 | Halteres as gyro sensors | **MISSING** | Haltere bodies exist physically; no haltere-derived angular-rate signal enters `FlyProprioception`. Round-5 code on the archive branch had lateralised halteres; port it after the leg loop closes. |
| 8 | Connectome, multi-resolution, sparse/GPU | **PARTIAL** | BANC v888 in-app: 175,237 neurons, 2.17 M synapses, CSR columns, `bytesNoCopy` mmap, zero per-neuron objects, GPU LIF (`SimulationEngine`, `LIF.metal`). **Gap:** only one resolution exists (~2 M); the brief wants small/debug→2 M→10 M→50 M→100 M+ under one format — the sectioned `flybanc.bin` v2 format is the seam, resolution tiers not yet built. |
| 9 | Brain → VNC → motor neurons separation | **PARTIAL** | The data model has it: BANC carries brain and cord in one animal, 532 MNs annotated to muscles, `NeuronInfo.isVNC`, system ranges for T1/T2/T3/abdomen. **Gap:** in-app the brain sim and the body are separate screens with no descending pathway wired — `drive` is unset (item 1). |
| 10 | Vision as signals, not booleans | **DONE** | `CameraFeed` samples the phone camera into 10,647 photoreceptor UVs per eye, with measured LMC gain 8–10× and ~100 ms adaptation (`SimParams`, ASSUMPTIONS #15); no `enemyDetected` anywhere. Optic-flow/motion readouts beyond the LIF substrate: not yet (the network itself does the motion work). |
| 11 | Odor as a spatial/temporal field | **MISSING** | No odor field in `FlyWorld`; olfactory receptor counts exist in the connectome (3,011) but no plume. Queue after mechanosensory loop: a scalar advection field + ORN drive. |
| 12 | Touch / mechanosensation | **PARTIAL** | Measured in physics: `footForce` per leg, contact count, joint rates (`FlyProprioception`). **Gap:** those signals do not yet enter any neuron — they are the input side of the open loop in item 1. |
| 13 | Proprioception | **PARTIAL** | `FlyProprioception` exposes joint angle, rate, leg load, body height, speed, feetDown — biologically named (chordotonal, campaniform) in docs/STEP3. Same gap: not consumed by neurons in-app yet. |
| 14 | Motor neurons → muscles → physics, no direct commands | **DONE** (path), **NEXT** (source) | The only path to the body is `posture + drive → excitation → muscleTorque → ABA solver → contacts`; high-level code never sets `q`, `position` or `velocity`. The drive *source* must be the cord (item 1). |
| 15 | Force-based muscle model | **DONE** | Hill force-length, per-direction velocity term (shortening weakens, lengthening loads ≤1.8), antagonist pairs, measured `hold_torque` inversion, optional fatigue hooks not yet — `FlyDynamics.muscleTorque`, verified in `FlyDynamicsTests`. |
| 16 | Physics is the authority | **DONE** | Featherstone ABA verified against MuJoCo to 4.4e-16 (FK) and 2.1e-5 cm (free fall); penalty floor fitted (K=150); golden-trace test in CI; the solver runs on-device (`FlyLiveBody`). |
| 17 | Walking emerges from neural activity | **MISSING** (by design) | No walk controller exists — correct per the brief; step 5's report says plainly: six cords cannot coordinate, tripod gait is owed. Offline, `tools/step4_world.py` showed cord-driven stance + nudge responses recorded in `behaviour.json`. |
| 18 | Minimal environment | **PARTIAL** | Floor + grid + lights + walls-not-yet; no obstacles, odor sources, wind, surface types. The world exists to stimulate; keep it deliberately spare. |
| 19 | Emergent behaviours | **PARTIAL** | Standing + postural stability emerge from tone + physics (`testTheAnimalStandsWithMuscleToneOnly`). Reflex arc validated offline (step 2/3: ρ = +0.125/+0.186, polarity control reverses it). Walking/grooming/feeding/escape all follow items 9→17. |
| 20 | iPhone optimisation | **DONE** (brain), **PARTIAL** (body) | GPU LIF, packed CSR, zero-copy connectome, camera-sampled retina; body solver is CPU scalar Swift at 100 µs substeps with adaptive throughput (`realtime` readout). A GPU/Metal port of the ABA solve is possible later; not yet needed — device runs at a measured fraction of real time and reports it. |
| 21 | Graphics budget | **DONE** | Point-cloud brain, SceneKit flat lambert body, no shadows/post-effects; perf beats pixels in every trade so far. |
| 22 | Debug / observability | **PARTIAL** | Brain: FPS/spikes/rate/power, voltage/spiking/regions render modes, tap-to-inspect, stimulate. Body: feet-down, contacts, COM height, substep/RTF readouts. **Gap:** sensory→MN→muscle pipeline view on the body screen (wire it with item 1). |
| 23 | Biological data policy | **DONE** | `docs/ASSUMPTIONS.md` is the registry (every constant: source + test); CI reproduces published checks (Azevedo 2020 monosynaptic absence, Phelps 2021 campaniform presence, corrected-sign guard for 7b22dd1). Placeholders are named `placeholder` in the asset. |
| 24 | Modular, headless-capable architecture | **DONE** | Connectome / SimulationEngine / Renderer / FlyDynamics / FlyLiveBody / FlyWorld / WorldView are separate files with narrow seams; Python tools run the whole science pipeline headless; brain sim runs with no body and vice versa. |
| 25 | Incremental phases | **ON TRACK** | Phases 1–5 of the 12-phase plan complete and measured (reports/step1–5); current position ≈ phase 6–7 (VNC pathways, walking circuitry). |
| 26 | No fake intelligence | **DONE** | No LLM, no RL agent, no behaviour tree, no scripted ethology in `FlyBrain/Sources`. Debug-only utilities (stimulate, flat retinal drive) are labelled as such. |
| 27 | Acceptance test (closed loop) | **PARTIAL** | Offline: steps 2–4 close the loop in Python (organ→pool→muscle→joint→organ). In-app: body→sensors computed, sensors→neurons **open**, neurons→muscles open (`drive` unset). The one broken link is exactly item 1; until it closes, the app must not be claimed as emergent behaviour. |
| 28 | Judged by biology, not pixels | **DONE** (discipline) | Every shipped stage has a report with numbers; the scale bug above was found by measurement (projection test), not by looks. |

---

## Delivery order (what follows this commit)

1. **This commit** — scale bug (both renderers), HUD restructure so readouts
   and buttons can never collide, control panel scrollable within its card,
   default camera distance fits the real-size fly. Then CI green → new .ipa.
2. **The closed loop in-app** — a cord module: BANC leg-circuit pools bundled
   as a small asset, LIF pools on the body side, `proprioception → pools →
   motor pool rates → drive offsets → muscles`. Test: the animal still stands,
   and a pushed leg resists (the step-3 result, now on the phone).
3. **VNC pathways & descending drive** — brain screen's sim and the body share
   one clock; descending neurons become the only source of `drive`.
4. **Walking circuits** — coordinate the six cords (tonic drive + load-
   dependent reflexes first, gait second), still with no `WalkController`.
5. **Environment + olfaction + wind** — a plume field and a light the retina
   can actually see.
6. **Haltere gyro + wing aero hooks** — body rotation → haltere signal → cord;
   wing joints → force model interface.
7. **Connectome resolution tiers** — debug/small/medium packs in the same v2
   section format; benchmark on device (`verify_banc.py` reference).
8. **Observability** — per-pool rates, MN activity, muscle excitation in the
   body HUD, mirroring the brain screen's inspector.
