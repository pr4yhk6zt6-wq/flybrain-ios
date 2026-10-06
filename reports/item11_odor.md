# Item 11 — odor as a spatial and temporal field

**Status: the field exists, the antenna is wired to it, and the smell reaches the
brain's motor pools — measured.** `docs/AUDIT.md` had this row as the last
**MISSING** system in the brief: *"No odor field in `FlyWorld`; olfactory
receptor counts exist in the connectome (3,011) but no plume."* Both halves of
that sentence are now false.

## The field

`tools/odor_field.py` (new) defines it; `FlyBrain/Sources/OdorField.swift` is the
same field in the language the phone runs. It is a **kinematic plume** with
literature-shaped statistics:

```
mean concentration   c̄(p) = C0 · s0/(s0+s) · exp(−r²/2σ(s)²) · exp(−s/Λ)
plume widening       σ(s) = σ0 + α·s
filaments            c(p,t) = c̄ · (2/D) · w(phase),  phase = mod(f·t − k·s/2π, 1)
                     w = ½(1 − cos(2π·phase/D)) inside a filament, 0 outside
receptor drive       current = 2.6 · c / (1 + c)
```

where `s` is distance downwind of the source and `r` crosswind distance. Upwind
of the source, zero — a plume is advected, so nothing arrives from behind.

Why intermittent rather than smooth: an antenna in a real plume spends most of
its time in **clean air** and meets odor as discrete packets, and that is the
property insect odor navigation is built on (Murlis, Elkinton & Cardé 1992,
*Annu. Rev. Entomol.* 37:505; Celani et al. 2014, *PNAS* 111:13013). A sinusoid
around the mean cannot produce it — it is above any threshold more than half the
time — which is why the modulation here is a duty cycle. Every constant is
labelled in `docs/ASSUMPTIONS.md` #26–#29 as an ENGINEERING PLACEHOLDER or an
APPROXIMATION; nothing here is measured anatomy.

| | measured | 
| --- | --- |
| time-mean falls with distance | 0.473 → 0.304 → 0.168 → 0.079 → 0.030 → 0.008 → 0.001 at 0.05 … 3.2 cm |
| upwind of the source | 0.0000 (exactly) |
| an antenna standing 3 mm downwind | inside a filament **20 %** of the time, in clean air (<10 % of the mean) **81 %** |
| peak / mean at that point | **10.0** (the 2/D of a 20 % duty cycle) |

`docs/img/odor_field.png` is the same three things drawn: the falloff, the
filament train at a standing antenna, and the crosswind profile.

## The drive reaches the brain

The connectome has carried 3,000 olfactory receptor neurons as
`sensory_olfactory` (BANC's own `cell_function == "olfactory"` — 3,000 after the
min-synapse filter, not the 3,011 of the raw annotation) since the first
packing, and nothing had ever driven them. Now the field does, at the antenna,
through the same `setGroupDrive` path as every other input. 120 ms of the same
LIF the app runs, same seed, vision at the app's operating point, five
concentrations:

| condition | peak drive | ORN rate | ascending | descending | `T1_left:trochanter_flexor` |
| --- | --- | --- | --- | --- | --- |
| no odor | 0.00 | 0.00 Hz | 0.45 Hz | 1.56 Hz | 0.76 Hz |
| far (c × 0.05) | 0.07 | 0.00 Hz | 0.45 | 1.56 | 0.76 |
| edge (× 0.25) | 0.34 | 0.00 Hz | 0.45 | 1.56 | 0.76 |
| in it (× 1.0) | 1.37 | 0.00 Hz | 0.45 | 1.56 | 0.76 |
| **strong (× 5.0)** | **6.83** | **27.30 Hz** | **2.03** | **4.51** | **1.52** |

The receptors are threshold devices: below the LIF's firing threshold the drive
is a drive the cell cannot answer (the same reason assumption #28 sets the
baseline in the app's tone units rather than inventing a new scale). Above it,
the odor does not stop at the antenna — it moves the ascending projection, the
descending population, and a leg motor pool, which is the substrate any
odor-guided behaviour would be built on. Nothing about behaviour is claimed
here: item 17 is where walking is; this is that the animal can *smell*.

## In the app

The same field is in the phone build, and the animal samples it at its own
antennae:

* `OdorField.swift` is the port. It is pinned to the Python tool by
  `FlyBrainTests/OdorFieldTests` — 13 samples chosen to cover a filament peak, a
  zero (clean air), the crosswind falloff, the far end of the plume and a point
  upwind of the source, compared at 1e-9. A CI step re-derives those samples from
  the tool and fails if the numbers embedded in the test have drifted from what
  this run measures, so the tool and the phone cannot silently part ways.
* `WorldModel.sampleOdor()` samples the field where the **head** is — the body
  model's own `head` geom, looked up by name in the posed mesh table — once per
  frame, and leaves the current on `FlyCord.odorDrive`, which writes it into the
  connectome every simulated millisecond like the tone and the organ drives.
* The source is drawn: a disc on the floor at the source, 3 mm upwind of where
  the animal starts, so the smell is a place rather than a number. The plume
  blows in +x at 0.30 cm/s. The HUD carries `odor %.3f at the antennae · drive
  %.2f`, which rises and falls as filaments pass.
* `FlyCord` resolves the group name like every other name it depends on, so a
  build without `sensory_olfactory` says `no antenna in this build` instead of
  showing a zero that reads as clean air.

## Two rehearsals, and what they changed

Both are the kind that would have produced a confident wrong answer:

1. **The monotonicity check was written backwards** — `b <= a` demanded that the
   ORN rate *fall* with concentration, and it failed a run whose dose–response
   was correct. It failed loudly, which is why it was found in one run.
2. **The gate passed a dead animal.** Rehearsing it with `baseline_drive` below
   the LIF's threshold (0.6 instead of 2.6) left the receptors at zero and moved
   nothing outside the antenna — and the gate still printed `ok`, because it only
   checked that the receptors fired. It now requires that the odor move at least
   one group beyond the antenna by more than 0.5 Hz, and that the strongest
   condition reach 5 Hz on the receptors. Rehearsed again: exit 1.

A third correction is in the tool's own history: the walk used to start 1 cm
upwind of the source and step 1.2 mm over the run, so the antenna never reached
the odor and every rate was zero. The peak drive is now printed with every
condition so that failure cannot be silent again.

## What is not done, and is stated rather than hidden

* **One antenna, not two.** BANC annotates no side for these 3,000 cells (0 of
  3,000 carry a left/right bit), so the drive is a single channel and a fly
  comparing its two antennae — the mechanism behind tropotaxis — cannot be built
  from this annotation. A lateralised group would need a geometric split with its
  own measurement and its own gate; it is not claimed here.
* **No odor-guided behaviour.** Nothing directs the animal toward the source.
  Items 17/19 are where behaviour comes from, and this item exists so that there
  is something to smell when they get there.
* **The field is kinematic.** It is deterministic, monotone downwind and
  intermittent, which is what the brief asks of an environment. It is not a
  solution of the advection–diffusion equation and does not claim to be.
* **Sampling cadence.** The field is sampled once per frame (~30–60 Hz) and held
  between frames, while filaments pass at 3.5 Hz — about ten samples per
  filament, so the train is resolved but its edges are quantised to a frame. The
  cord applies the value every simulated millisecond, so the connectome's input
  is a staircase at frame rate, not at 1 kHz.
