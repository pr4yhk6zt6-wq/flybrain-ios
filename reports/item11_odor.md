# Item 11 — odor as a spatial and temporal field

**Status: the last system the brief had no code for now exists, is in the phone,
and the animal's antennae respond to it — measured. What the smell does *beyond*
the receptor population is not established, and that is written below rather than
rounded up.**

`docs/AUDIT.md` had item 11 as **MISSING**: *"No odor field in `FlyWorld`;
olfactory receptor counts exist in the connectome (3,011) but no plume."* The
field exists, the receptors are driven by it, and the drive is measured on the
same LIF the app runs.

## The field

`tools/odor_field.py` defines it; `FlyBrain/Sources/OdorField.swift` is the same
field in the language the phone runs. A **kinematic** plume with
literature-shaped statistics:

```
mean concentration   c̄(p) = C0 · s0/(s0+s) · exp(−r²/2σ(s)²) · exp(−s/Λ)
plume widening       σ(s) = σ0 + α·s
intermittency        phase = mod(f·t − k·s/2π, 1)
                     c(p,t) = c̄ · (2/D) · ½(1 − cos(2π·phase/D))   if phase < D
                     c(p,t) = 0                                     otherwise
receptor drive       current = baseline_drive · c / (1 + c)
```

`s` is distance downwind of the source, `r` the crosswind distance; upwind of
the source the concentration is exactly zero. Every constant is a labelled
ENGINEERING PLACEHOLDER or APPROXIMATION in `docs/ASSUMPTIONS.md` #26–#29.
Nothing here is measured anatomy, and the field is not a solution of the
advection–diffusion equation.

Why intermittent rather than smooth: an antenna in a real plume spends most of
its time in **clean air** and meets odor as discrete packets, and that is the
property insect odor navigation is built on (Murlis, Elkinton & Cardé 1992,
*Annu. Rev. Entomol.* 37:505; Celani et al. 2014, *PNAS* 111:13013). A sinusoid
around the mean cannot produce it — it is above any threshold more than half the
time — so the modulation is a duty cycle.

| measured, at the animal's own antennae (2.5 mm downwind of the source) | |
| --- | --- |
| time-mean concentration | 0.136 |
| filament peak | 1.359 |
| peak / mean | **10.0** (the 2/D of a 20 % duty) |
| inside a filament | **20 %** of the time |
| in clean air (<10 % of the mean) | **81 %** of the time |

The falloff with distance is monotone and unchanged from the first measurement
of it — 0.473 / 0.304 / 0.168 / 0.079 / 0.030 / 0.008 / 0.001 at 0.05 … 3.2 cm
downwind, exactly 0.0000 upwind — and `docs/img/odor_field.png` draws all three
of those things.

## Where the animal stands

The source is **2.5 mm in front of the animal** and the air blows *onto* it
(`wind_cms` = −x), so the plume passes over the animal's head and walking upwind
(+x) is walking towards the food. The antennae are where the body model's own
head geom is — the point `WorldModel.sampleOdor()` reads on the phone — which
puts them 2.5 mm downwind of the source: in the plume.

## Three corrections, because the first version of this item was wrong

These are the useful part of the item, so they are not buried:

1. **The window was 120 ms.** At 3.5 Hz a gust cycle is 286 ms, so that window
   was *less than one cycle*, and it sat inside the LIF's start-up transient. It
   reported "no odor: **0.00 Hz**" for a group that actually idles at **20.98
   Hz**, and every rate in the table it produced was a transient. The window is
   now 860 ms — three gust cycles after 100 ms of warm-up — and the clean-air
   rate is printed first, so every other number is a difference from it.
2. **The "antenna" was a point the app never samples.** The probe used a point
   3 mm downwind of the source; the app samples the body model's head geom. The
   animal's own point is now the condition the gate judges, and the other
   distances are printed as what it would smell if it walked.
3. **The operating point was below the receptors' threshold.** With
   `baseline_drive` 2.6 the current at the animal's own antennae peaked at 1.37,
   and the receptors leave their clean-air rate at ~2–3: the app shipped a fly
   that could not smell its own food, and the gate passed because its "strong"
   condition multiplied the drive by five — a condition the app never
   experiences. `baseline_drive` is now 45, set from the measured ladder below,
   and the gate judges the animal's own point.

The measurement is now the claim, in the order it was made:

## The drive, at last measured honestly

3,000 ORNs (`sensory_olfactory`, BANC's own `cell_function == "olfactory"`),
860 ms after 100 ms of warm-up, gain 12 (the app's default), 175,237 cells. The
cells the receptors **actually synapse onto** are read out of the packed
connectome, not assumed: 26,359 synapses onto 881 cells, **615 of which are not
labelled olfactory**.

| condition | drive peak | ORN Hz | first targets Hz | ascending Hz | descending Hz |
| --- | --- | --- | --- | --- | --- |
| no odor at all (drive 0) | 0.00 | 20.98 | 232.19 | 20.42 | 32.88 |
| 4 mm upwind of the source | 0.00 | 20.98 | 232.19 | 20.42 | 32.88 |
| 2.5 cm downwind | 1.07 | 21.35 | 232.41 | 20.49 | 32.18 |
| 1.2 cm downwind | 5.77 | 28.43 | 247.98 | 20.21 | 30.92 |
| 1 cm downwind | 7.71 | 30.97 | 248.44 | 21.58 | 30.93 |
| 3 mm downwind | 23.66 | 51.29 | 251.53 | 20.93 | 32.77 |
| **the antennae (2.5 mm) — the app's own point** | **25.92** | **53.54** | **251.78** | 20.57 | 30.98 |

A second seed (7) reproduces it: no odor 20.23 / 225.08, the antennae 53.50 /
251.68, 1.2 cm 28.39 / 247.84. The receptor response is **+155 %** at the
animal's own point, and the cells the receptors project to rise by **+8.4 %**
(seed 7: +11.8 %).

**What this does and does not say.** The smell is a sense at the antenna: 21 Hz
in clean air, 54 Hz in its own plume, and the cells the receptors synapse onto
follow. The ascending and descending populations, over three gust cycles, do
**not** move by more than their own variability (−0.4 … +1.6 Hz on 2,366 and
1,316 cells, with the sign inconsistent between seeds), and the leg motor pool
one would want to point at has **11 cells** in it, where a rate is a handful of
spikes and its change is not evidence of anything. So: *the smell reaches the
receptors and their first targets; the path from there to the legs is not
demonstrated by this measurement.* Item 17 is where walking is, and this item
exists so that there is something to smell when it gets there.

## The threshold and the range — measured, not chosen

A ladder of currents at the antennae's own trace (500 ms per rung; the criterion
is the 20 % the gate uses, because a smaller one is inside the network's own
variability):

| peak drive | 1.30 | 2.07 | 2.85 | 3.89 | 5.70 | 10.37 | 25.92 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ORN Hz | 20.19 | 24.66 | **25.41** | 26.95 | 27.94 | 32.98 | 47.34 |

The receptors leave their clean-air rate between 2.07 and 2.85 (seed 7 puts it at
2.07, seed 42 at 2.85 — the ladder's step is 0.39, so the honest statement is
*about 2–3*). The animal's own filament peak is a current of 25.9, i.e. **9–12×
the threshold**. From that threshold, the distance at which a filament peak still
crosses it is **1.72 cm** (1.97 cm at seed 7's lower threshold): the animal
smells the patch it is standing on, and loses it about 2 cm downwind.

`baseline_drive` = 45 is the number that puts the animal's own antennae 9× above
the threshold with that range. It is an ENGINEERING PLACEHOLDER (assumption #34)
— the threshold and the range it produces are measured, by this tool, on every
run, and the gate fails if the range drops under 1 cm.

## In the app

* `WorldModel.sampleOdor()` reads the field at the head geom of the posed body,
  once per frame, at `live.simulatedMS`; `FlyCord.update` writes it into the
  connectome every simulated millisecond, on `sensory_olfactory`, like the tone
  and the organ drives.
* The source is a disc on the floor 2.5 mm in front of the animal, so the smell
  is a place rather than a number, and the HUD carries
  `odor %.3f at the antennae · drive %.2f`, which rises and falls as filaments
  pass.
* A body file whose head mesh has another name is handled by trying the names
  flybody uses (`head`, `head_body`, `head_red`) and **saying so** when none is
  present — a missing head would otherwise read as drive 0.00, which is exactly
  what clean air reads as.
* `docs/ASSUMPTIONS.md` #26–#29 carry the constants and their sources; the
  `odor` block of `world/world.json` is the parameter set the app reads, so the
  app does not keep a second copy of the field.

## Two rehearsals that changed the code

1. The dose–response check was written backwards (`b <= a`) and failed a run
   whose dose–response was correct. It failed loudly, in one run.
2. The gate's first version passed a fly that could not smell: rehearsing it with
   a drive below threshold left the receptors unchanged and moved nothing, and it
   still printed `ok`. It now requires the animal's own point to raise the
   receptor rate by ≥20 % and the first-target population by ≥5 %, and the
   measured range to be at least 1 cm — and the "strong ×5" condition that used
   to carry the claim is gone, because the app never experiences it.

The Swift port is pinned to the tool by 16 golden samples in
`FlyBrainTests/OdorFieldTests` (a filament peak, clean air exactly 0, the
crosswind falloff, the far field, and a point upwind of the source), and a CI
step re-derives them from the tool and fails if the embedded numbers have drifted
from what that run measures.

## What is not done, stated rather than hidden

* **One antenna, not two.** BANC annotates no side for these 3,000 cells (0 of
  3,000 carry a left/right bit), so the drive is a single channel; a fly
  comparing its two antennae — the mechanism behind tropotaxis — cannot be built
  from this annotation. A lateralised group would need a geometric split with its
  own measurement and its own gate.
* **No odor-guided behaviour.** Nothing walks the animal towards the source;
  items 17/19 are where behaviour comes from.
* **The field is kinematic** — deterministic, monotone downwind, intermittent;
  not a solution of the advection–diffusion equation, and it does not claim to
  be.
* **Sampling cadence.** The field is sampled once per frame (~30–60 Hz) and held
  between frames while filaments pass at 3.5 Hz: about ten samples per filament,
  so the train is resolved but its edges are quantised to a frame. The cord
  applies the value every simulated millisecond, so the connectome's odor input
  is a staircase at frame rate, not at 1 kHz.
* **The receptor response is a rate, not a code.** Nothing here claims the fly
  uses the timing of the filaments — only that its receptors see them.
