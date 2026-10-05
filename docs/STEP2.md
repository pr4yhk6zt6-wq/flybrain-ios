# Step 2 — the leg reflex, taken out of the connectome

Step 1 gave us a body and a nerve cord. Step 2 asks the first question that
requires both: **does the connectome contain the reflex a fly stands on?**

Everything below is produced by `tools/step2_reflex.py` from BANC v888 and
written to `reports/step2_reflex.json`. The human-readable table is
`reports/step2_reflex.md`, regenerated on every run.

## The question, stated so that it can fail

> When the front-left leg's femoral chordotonal organ (FeCO) reports that the
> femur–tibia joint is moving, which motor pool does the connectome tell the
> leg to fire — and does the wiring agree with the published physiology?

The previous rounds of this project were lost to probes that failed and were
"fixed" with a slider. So this one is written to report a *negative* result just
as loudly, and it did produce one (see §4).

## 1. The organ's own contacts

Ranked by synapse count, the front-left chordotonal cells' strongest targets:

| synapses | target | transmitter | sign |
|---:|---|---|---|
| 571 | `IN13B001` | GABA | −1 |
| 279 | `IN21A009` | glutamate | −1 |
| 253 | `pleural_remotor_and_abductor` (motor) | acetylcholine | +1 |
| 240 | `IN14A001` | GABA | −1 |
| 240 | `ANXXX041` | GABA | −1 |
| 239 | `IN16B038` | glutamate | −1 |
| 232 | `IN08A006` | GABA | −1 |
| 210 | `sternal_anterior_rotator` (motor) | GABA | −1 |

**`IN13B001` is a 13B-lineage interneuron.** The *Drosophila* literature names
the 13Bα population as the FeCO's first central partner — it encodes tibia
position and its activation produces tibia flexion (Agrawal et al. 2020, eLife
9:e60299; Phelps et al. 2021). The connectome was not told to agree with the
paper. It was asked a question, and the answer came back with the same cell
type on top, by a factor of two over the next target.

The same run finds **09Aα** (19 cells projecting to the flexor, 0 to the
extensor) and **10Bα** (6 cells receiving directly, 0 projecting to either pool)
— the other two populations the same papers describe.

## 2. The path is disynaptic, exactly as published

Counted over the whole connectome, every one of the 38 chordotonal cells reaches
the tibia flexor pool, and the shortest routes are:

| hops to the tibia flexor | chordotonal cells |
|---|---|
| 1 (monosynaptic) | **0** |
| 2 (one interneuron) | 36 |
| 3 | 2 |

The FeCO reflex in *Drosophila* is known **not** to be monosynaptic: it engages
"distributed parallel and polysynaptic pathways", and "individual polysynaptic
pathways either support or oppose the overall reflex output" (Agrawal et al.
2020; Phelps et al. 2021 commentary, *Current Biology*). BANC agrees, cell by
cell. The sign of the reflex is therefore decided by the interneurons in the
middle, which makes those interneurons the thing to measure.

## 3. What the interneurons do, weighted by synapses on both legs of the path

| pool | motor neurons | interneurons on the path | arcs | excitatory weight | inhibitory weight | net | net per motor neuron |
|---|---:|---:|---:|---:|---:|---:|---:|
| tibia flexor | 17 | 30 | 53 | 18,947 | 5,043 | **+13,904** | +818 |
| tibia extensor | 2 | 1 | 1 | 9 | 0 | **+9** | +4 |
| trochanter flexor | 11 | 24 | 43 | 17,886 | 3,152 | +14,734 | +1,339 |
| femur reductor | 6 | 16 | 30 | 17,283 | 58 | +17,225 | +2,871 |
| coxa rotator post | 4 | 8 | 10 | 4,291 | 9 | +4,282 | +1,070 |

Weights are path products (synapses organ→interneuron × synapses
interneuron→motor neuron), signed by the interneuron's transmitter — a measure
of how hard each path pushes, not a physical current.

The asymmetry is the finding: **the chordotonal organ has a large disynaptic
route to the flexor pool and essentially no route at all to the extensor**
(one interneuron, weight +9). In the classical insect resistance reflex the
extensor is the muscle that pushes back. Here the connectome says the FeCO
talks to the flexor.

## 4. The simulation, and the result that failed

The project's LIF (1 ms, τ_m 20 ms, τ_syn 5 ms, delays from soma distance at
0.25 m/s, weights `sign(NT)·log1p(synapses)` normalised by the median
postsynaptic drive) over the extracted pathway subgraph — 38,181 neurons,
1,375,595 edges — with each organ population driven hard from t = 100 ms:

| condition | flexor | extensor | coxa rotator post | trochanter extensor |
|---|---:|---:|---:|---:|
| baseline | 0.00 Hz | 0.00 Hz | 0.00 Hz | 0.00 Hz |
| **chordotonal driven** | 0.00 Hz | 0.00 Hz | **21.6 Hz** | 3.3 Hz |
| campaniform driven | 0.00 Hz | 0.00 Hz | 0.00 Hz | 0.00 Hz |
| hairplate driven | 0.00 Hz | 0.00 Hz | 0.00 Hz | 0.00 Hz |

Subthreshold readout (mean membrane potential over the second half, in units of
the 1.0 threshold):

| pool | baseline | chordotonal driven | hairplate driven | sign-shuffled control |
|---|---:|---:|---:|---:|
| tibia flexor | 0.0000 | −0.0200 | +0.0030 | −0.0066 |
| tibia extensor | −0.0005 | −0.0138 | **+0.2951** | — |

**Three things to say honestly.**

1. **Driving the chordotonal organ does not bring the tibia pools to spike
   threshold**, and its subthreshold effect is weakly *negative* — the opposite
   sign to the static weight analysis in §3. Reported, not tuned. The likely
   cause is visible in §1: the single strongest target is an *inhibitory* 13B
   interneuron, and a lumped chordotonal population drives the inhibition
   harder than the excitation. A real FeCO is not lumped — it has three
   functionally distinct subtypes (claw = position, hook = movement direction,
   club = vibration; Mamiya et al. 2018), and a real reflex stimulates one
   subpopulation at a time. BANC's `cell_function_detailed` does not separate
   them, so the next step has to do it by the cells' own projection patterns.

2. **The machinery works**, which is why the negative result means something:
   the identical drive produces 21.6 Hz in the coxa rotator motor pool and a
   clear subthreshold depolarisation of the tibia extensor from the *hair
   plate* population (+0.295 of threshold) — a population-specific effect, not
   noise.

3. **The network's silence is not a tuning problem.** Every neuron has the same
   leak and the same threshold; there is no facilitation, no depression, no
   intrinsic excitability. A local VNC circuit that integrates proprioceptive
   evidence over hundreds of milliseconds is built out of those properties, and
   they are missing — deliberately, because each one is a mechanism that has to
   be cited and tested rather than switched on because the output looked right.

## What step 2 has established, and what it has not

**Established, from the file:**

- the leg's chordotonal organ contacts the cell types the literature names for
  it, with the 13B lineage first by synapse count;
- its route to the tibia motor pools is disynaptic, never monosynaptic;
- the route is strong to the flexor and almost absent to the extensor;
- every one of the 69 motor neurons of the leg is reachable from the leg's own
  proprioceptors within three hops (`tools/step2_legcircuit.py`), 46 of them
  monosynaptically.

**Not yet established:** that the loop, simulated as it stands, produces a
standing or walking leg. It does not — yet — and the missing pieces are
identified rather than papered over:

1. split the chordotonal population by subtype instead of driving it lumped;
2. add the campaniform (load) population as the second half of the loop;
3. give the local circuit the membrane mechanisms a real one integrates with,
   each cited, each with a test;
4. only then close the loop onto the body — the reflex driving the 70 joint
   actuators of the flybody model, with the standing test of Step 1 as the
   acceptance criterion.
