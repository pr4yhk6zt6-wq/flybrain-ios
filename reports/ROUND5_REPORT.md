# Round 5 — hypothesis → evidence → verdict → change

Companion to commits 87f3e5e…(phase 0+1), 81ab334 (phase 2), and this one
(phase 3). Every "confirmed" below was measured BEFORE the corresponding
change was made, with the probe named in the evidence column. Nothing was
tuned to make the video look good; where a check fails, it is reported.

## Phase 0 — hypotheses

| # | Hypothesis | Evidence (probe) | Verdict | Change made |
|---|------------|------------------|---------|-------------|
| H1 | Flight yaw sign inverted: stronger LEFT wing turned the fly LEFT; real flies turn toward the weaker wing (Fry et al. 2003) | `tools/phase0_probes.py` torque-sign print + new simcheck T1 (`testYawSignStrongerWingTurnsTowardWeakerSide`) | **CONFIRMED** | Flight yaw torque now `(dragL − dragR)·r2`: stronger left wing yaws right. Walking decode already agreed; T1 asserts both now agree. |
| H2 | Yaw gain bang-bang: `yawDamping`/`maxYawTorque` meant a ~10% wing-power difference demanded 2.84× the maximum torque, pinning the animal at the 1600 deg/s clamp | `phase0_probes.py` clamp-ratio print: 2.84×/1.00× at 10% asymmetry | **CONFIRMED** | Opponent decode `(R−L)/(R+L+ε)` with ε from the one-spike noise floor of each motor group over the 0.2 s muscle low-pass; asymmetry maps onto the measured 20° steering authority. T2a: ±10% iid noise → 97 deg/s RMS (bar 200); T2b: 60% step → 1600 deg/s peak that stops when the command stops. |
| H3 | Mode flicker: lift≈1.0 W by construction vs the 0.98 W airborne threshold switched flight↔walk ~10×/s under 10% rate noise | `phase0_probes.py` switch-count: 10.7 switches/s | **CONFIRMED** | One continuous dynamics model: flight forces always apply; contact comes from the collision solver only; take-off shares the flight branch's stroke-plane tilt surface. T3: 0 contact frames in 5 s of noisy hover. The 0.02-gravity walking hack is gone. |
| H4 | Coxa sign wrong for both sides (protraction·0.35 same sign) | FK probe of the MJCF axes + per-segment tip signs | **REFUTED** (the walking-reversal complaint came from per-segment TIP SIGNS differing — driving all segments with one sign was the real bug) | Per-segment protraction signs derived in `tools/gait_geometry.py`. |
| H5 | Eye renders include the fly's own body/wings (phone screenshot artifact) | `tools/softrender.py --eye`: renders with and without the fly mesh are pixel-identical (diff 0.0) — artifact NOT reproduced in the probe; `drawFly:false` verified in code | **NOT REPRODUCED** | Hardened anyway in phase 2: explicit `RenderMask` culling, cameras at 1.5× the CT head half-width (outside the head volume), horizon rolls with the body. |
| — | `groupRate` sum-vs-mean bias with unequal L/R groups | Engine read: mean + 0.35 EMA; 1 spike in a 12-neuron group = 5.2 Hz | mean (no bias), but the **5.2 Hz spike step is real** → it set the opponent ε floors | ε derived, not tuned (ASSUMPTIONS.md #14). |

## Phase 1 — body (FlyBody.swift, mirrored 1:1 in tools/simcheck.py, 25/25)

| Item | Outcome |
|---|---|
| Habit layer, arousal OU, yawBias/turnBias | **DELETED** as scripted behaviour. Their four tests were replaced by an explanation block in FlightPhysicsTests.swift, not silently dropped. |
| Escape | `jump>8 && airborne<0.6` → TT spike-event threshold (10 Hz = one spike/frame through the EMA) + 0.1 s refractory; counted in the air, fired only on the ground; HUD shows it. Verified: launch +90 cm/s, refractory swallows the rattle, mid-air event counted without impulse. |
| Gait | Re-derived from the CT model by `tools/gait_geometry.py` with TRUE FK travel (the old small-angle tangent gains over-estimated travel ~2×). Stance inverse-map knots give constant stance-foot speed. **Residual slip at 30 mm/s, reported not hidden: T1 55%, T2 73%, T3 14%** (ASSUMPTIONS.md #6). T4 (T3, longest reach): −0.87 vs −1.00, inside the 25% bar. |
| Stroke tilt | `strokeTilt = 29°·clamp((F/W−1)/0.5)` — zero at hover so the equilibrium reflex converges on a true hover (the old formula made exact hover sink 2%). |

## Phase 2 — sensors (commit 81ab334, CI green)

| Item | Outcome |
|---|---|
| Halteres | Signed + lateralised (`sensory_haltere_left/right`, 216/212 cells): rectified yaw rate toward each side / measured 1600 deg/s peak (Dickinson 1999). |
| Wing sensors | `sensory_wing_left/right` (272/435 cells), each reading its own stroke amplitude (was L only). |
| Eye pass | `RenderMask` culling; cameras outside the CT head AABB; up vector rolls with the body. |
| Retinal adaptation | `1−exp(−dt/τ)`, τ = 0.1 s (Laughlin & Hardie 1978; Curr Biol 2023) — frame-rate independent. Sliders removed; contrastGain fixed at the measured 8–10× LMC:photoreceptor ratio (ASSUMPTIONS.md #15). |

## Phase 3 — connectome probes (tools/phase3_probes.py, report-only)

| Test | Result |
|---|---|
| T5 no stimulus → small asymmetry | **PASS** — wing L/R 0.0/0.0 Hz, RMS asymmetry 0.0 (bar 5.0). |
| T6 looming > receding & static for LC4 / LPLC2 / DNp01 | **FAILS — REPORTED, NOT TUNED.** The stimulus demonstrably reaches the network (left-eye vision rate 30.6 vs right 23.9 Hz; scores use the post-onset window), but LC4 (3.0–3.7 Hz), LPLC2 (48.5–52.7) and DNp01 (121–150) show no looming preference in the BANC wiring simulated with generic LIF at gain 12. Candidate causes to investigate, none of them "adjust the gain": (a) looming selectivity in real flies is computed by direction-selective elementary motion detectors whose delay structure (Reichardt-type, ~ms inter-ommatidial delays) our 1–4 ms conduction delays + point-sampled retina may not reproduce; (b) DNp01 is 2 cells — no statistics; (c) the disc is a luminance edge stimulus; real LC4 preference is measured with dark looming on the CONTRALATERAL field geometry our per-hemisphere SVD flattens. |
| T7 optomotor sweep → asymmetric steering | **RESPONDS** (wing power 209.8 vs 255.2 Hz, steering 89.0 vs 153.1 Hz). **Sign not calibrated headlessly**: the per-hemisphere SVD defines each u-axis only up to a sign, so "slip-reducing" cannot be asserted from this probe — the closed-loop in-app observation (world slips → animal reduces slip) is the calibrated test. Investigated as instructed rather than flipped. |

## Hardware-unverified list

1. **All Swift behaviour on a real device** — CI compiles and runs XCTest on a
   simulator; Metal timing, the 60 fps eye pass, and thermal behaviour are
   unverified on hardware.
2. **The eye-pass artifact from the phone screenshot** — could not be
   reproduced headlessly (H5); the mask/culling/camera fixes are preventive.
3. **Closed-loop optomotor sign (T7)** — needs the animal flying in the app
   with visible world slip.
4. **Looming escape through the full sensorimotor path** (real looming object
   in the world → retina → LC4/LPLC2/DNp01 → giant fibre → jump) — the body
   half (spike-event take-off) is tested; the connectome half is the failing
   T6 above.
5. **BANC group side-annotations** for the 11 haltere / 45 wing cells with
   `side = None` — they are simply not driven.
