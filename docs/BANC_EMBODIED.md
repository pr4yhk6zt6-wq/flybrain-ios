# Switching to BANC v888 — the brain *and* the nerve cord

Status: **data layer complete and validated.** The virtual world is built on top
of this; this document covers the connectome swap that makes an embodied fly
possible at all.

## Why swap

FAFB v783 is a *brain* dataset. It has 110 brain motor neurons, no leg motor
neurons, no wing motor neurons, and no ventral nerve cord. So "descending neuron
fires → leg steps" could only ever be invented. To get a fly that genuinely moves
because its brain told it to, the dataset has to contain the cord.

**BANC v888** (Bates et al., *Nature* 2026) is the first synapse-resolution
connectome spanning brain *and* ventral nerve cord in one animal — the same
female fly, so no cross-specimen mapping is needed.

| | FAFB v783 | BANC v888 |
|---|---|---|
| Coverage | brain only | brain **+ ventral nerve cord** |
| Leg motor neurons | **0** | **391** |
| Wing motor neurons | **0** | **60** (24 power, 24 steering, 12 tension) |
| Proprioceptors | 0 | **2,903** |
| Descending | 1,305 | 1,316 |
| Ascending (cord → brain) | — | **2,366** |
| VNC intrinsic | — | 12,866 |
| Licence | CC BY-**NC** 4.0 | CC BY 4.0 |

BANC is also CC BY rather than CC BY-NC, which lifts the non-commercial
restriction the FAFB build carried.

## Sources — both public, no authentication

```
https://storage.googleapis.com/flywire-data/codex/data/banc/888/
    neurons.csv.gz                  2.8 MB
    connections_princeton.csv.gz     28 MB   pre, post, neuropil, syn_count, nt
https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/
    compiled_data/banc_888/banc_888_meta.feather   55 MB
```

The meta feather is the important one. Beyond positions it carries
`cell_function` (`leg_motor`, `wing_power`, `wing_steering`, `proprioception`,
`auditory`, …), `body_part_effector` (`front_leg`, `wing`, `haltere`, `neck`, …),
`body_part_sensory`, `neuromere` (T1/T2/T3/A1–A9) and `side`. Those columns are
precisely the wiring diagram between a connectome and a body.

Note BANC's `connections.csv.gz` does not exist under that name — the Codex
export calls it **`connections_princeton.csv.gz`**.

## What the build produces

```
175,237 neurons · 2,171,713 edges (≥3 synapses) · 19.04 MiB
```

Same zero-copy section layout as before, with one new section — `groupIndices` —
holding the named neuron sets the body reads:

| group | n | group | n |
|---|---|---|---|
| `motor_front_leg_{l,r}` | 69 / 70 | `sensory_vision` | 9,733 |
| `motor_middle_leg_{l,r}` | 63 / 63 | `sensory_proprioception` | 2,903 |
| `motor_hind_leg_{l,r}` | 63 / 63 | `sensory_tactile` | 6,698 |
| `motor_wing_power_{l,r}` | 12 / 12 | `sensory_antenna` | 4,230 |
| `motor_wing_steering_{l,r}` | 12 / 12 | `sensory_olfactory` | 3,000 |
| `motor_wing_tension_{l,r}` | 6 / 6 | `sensory_gustatory` | 1,585 |
| `motor_neck` | 49 | `sensory_auditory` | 1,187 |
| `motor_haltere` | 25 | `sensory_nociception` | 170 |
| `motor_jump_escape` | 2 | `descending` | 1,316 |
| `motor_proboscis` | 35 | `ascending` | 2,366 |

Motor groups are split by side, which is what makes **turning** possible: drive
the left wing steering group harder than the right and the fly yaws, exactly as a
real fly does.

## Four decisions worth recording

**Vision comes from cell types, not body-part tags.** BANC is missing most of the
lamina and retina proper, so only ~1,360 cells carry a `retina`/`lamina` body-part
tag — too few to sample a camera into. The lamina monopolars and inner
photoreceptors are present as *cell types* inside the optic lobe, so
`sensory_vision` is `cell_type ∈ {L1…L5, R7, R8}` plus the tagged cells: **9,733**
input neurons, and the same input layer the FAFB build used.

**Unplaceable neurons are imputed, not dropped.** 18,708 of 175,401 neurons have
no representative point, and they are concentrated in the optic lobe and heavily
connected — dropping them cost **over a million edges**. 98% carry a region label,
so they are placed at that region's centroid with a little jitter and flagged in
the metadata (bit 7). Wrong in detail, right in aggregate, and the wiring survives.

**865,648 edges are still dropped**, because 14,620 of the 153,962 endpoints in
the edge list have no metadata row at all — unannotated segments with no position,
no transmitter and no class. There is nothing to simulate for them.

**Conduction is 0.25 m/s, not 0.5.** BANC's real extent is 877 × 1,068 × 315 µm.
At 0.5 m/s every delay collapses to 1–2 ms. Fly axons are thin and unmyelinated
(0.1–0.5 m/s measured), so 0.25 m/s gives a 1–4 ms spread that is both defensible
and visible. Note how much more compact this is than FAFB's apparent 11 mm z-range,
which was an artefact of the sectioning axis.

Histamine is now in the transmitter table and signed **inhibitory** — it is the
photoreceptor transmitter and hyperpolarises its targets.

## The result that matters

14/14 structural checks pass, and the reference simulation closes the loop:

```
gain 12, visual drive 1.5, 250 ms

population        18.0 Hz
vision            28.1 Hz   <- driven
descending        30.1 Hz   <- brain output to the cord
ascending         19.1 Hz   <- cord feedback to the brain
leg motor         32.6 Hz   <- REAL leg motor neurons
wing power       119.5 Hz   <- real flies beat wings at ~200 Hz
wing steering    119.5 Hz
```

**Light hits the photoreceptors, and 391 real leg motor neurons and 60 real wing
motor neurons fire, through measured wiring, in a single animal's nervous system.**
No invented interface anywhere in that chain.

Gain sweep: 8 → 4.6 Hz (sparse), **12 → 18.0 Hz (chosen)**, 16 → 23.3 Hz,
20 → 27.3 Hz. The cliff is softer than FAFB's because the cord adds recurrent
loops that are not purely visual.

## Reproducing

```bash
bash tools/download_banc.sh
python3 tools/build_banc.py          # ~5 s  -> build/flybanc.bin
python3 tools/verify_banc.py --ms 300 --gain 12
```

## Attribution

BANC v888 — Bates, Phelps, Kim, Yang et al. (2026), *Distributed control circuits
across a brain-and-cord connectome*, Nature. doi:10.1038/s41586-026-10735-w.
Data CC BY 4.0, BANC/FlyWire Consortium, Lee Lab (Harvard) and Zetta.ai.
