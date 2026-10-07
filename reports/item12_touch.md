# Item 12 — touch: the tarsal hairs as a sense organ of the loop

Status: **DONE** (off-device; the on-device run is the next build).
Sources: BANC v888 (`data/banc`, fetched by `tools/download_banc.sh`), the body
asset (`build/fly_body.json`), the shipped connectome (`build/flybanc.bin`).
Tools: `tools/tactile_probe.py` (new, CI-gated), `tools/verify_loop.py`,
`tools/motor_pools.py`, `tools/build_banc.py`, `tools/build_body.py`,
`tools/pool_probe.py`. App: `FlyCord.swift` (`case "tactile"`), `FlyCordTests`.

## What the gap was

`docs/AUDIT.md` row 12 said it in one sentence: the tactile cells were in the
connectome (6,698 of them, `sensory_tactile`) and **nothing drove them**. The
campaniform organ carries the load a leg is under, and load is the same number
whether the foot has just landed or has been standing for a second. A hair is
not a gauge: it reports that the tarsus is *in contact*, it bursts when contact
starts, and it adapts while it lasts.

## The organ (assumption #40)

One group per leg, named by the same function that names the pools:

| group | cells | outgoing edges | release sites | of them, to motor neurons |
|---|---|---|---|---|
| `organ:T1_left:tactile` | 514 | 6,134 | 1,253.9 | 18.8 |
| `organ:T1_right:tactile` | 481 | 4,343 | 951.4 | 11.9 |
| `organ:T2_left:tactile` | 484 | 7,868 | 1,808.9 | 10.0 |
| `organ:T2_right:tactile` | 494 | 7,438 | 1,691.2 | 6.7 |
| `organ:T3_left:tactile` | 564 | 5,166 | 1,109.1 | 10.4 |
| `organ:T3_right:tactile` | 604 | 5,744 | 1,266.8 | 13.2 |

The cells are BANC's own call — `cell_function == "tactile"`, the right side, and
the leg's own `body_part_sensory` tag. The census behind the counts: 3,011
bristle neurons, 114 the mechanosensory neuron of a tarsal taste hair, 38 orphan
neurons, 20 unclassed, 1 hair-plate neuron; all 3,184 are `super_class ==
"sensory"` and sit in the ventral nerve cord; 43 carry no side and are therefore
in no organ, which `tools/verify_loop.py` reports rather than dropping quietly.

Two decisions are worth their lines:

* **The leg tag is required.** The three proprioceptors accept a cell tagged to
  this leg *or* sitting in this leg's neuromere. A thoracic neuromere also holds
  the wing's and the haltere's bristles (290 wing-margin cells alone), so the
  same fallback would put another appendage's sensilla in a leg's touch organ.
  On BANC v888 the fallback happens to add nothing (every leg bristle carries
  its tag) — measured, and the predicate does not depend on it.
* **Touch is not a row of `ORGANS` for step 2.** `tools/step2_reflex.py` drives
  every cell of every organ it is handed and reads what the pools do; its
  numbers are this project's reflex baseline. Admitting 3,184 more driven
  afferents would move a baseline rather than describe touch, so step 2 reads
  the `PROPRIOCEPTORS` view of the same table and item 12 measures its own
  pathway (`tools/tactile_probe.py`).

## Does touch reach the leg's motor pools?

Where the leg tactile cells send, measured on the packed connectome (37,253
outgoing edges): **80 % to ventral-nerve-cord intrinsic neurons** (29,781
release sites), 17 % ascending (6,275), 343 to motor neurons and 219 to
descending; the motor pools are reached the way a reflex is — through the
cord's own interneurons, with a small direct arc.

Like for like, on one leg (T2 left), each of its three organs driven at the
cord's tone for 400 ms with the brain's tone on the descending population, as
the app drives them:

| organ | the leg's own pools | the other side | own/other |
|---|---|---|---|
| chordotonal | 2,510.0 Hz | 943.3 Hz | 2.66 |
| campaniform | 1,773.3 Hz | 1,086.7 Hz | 1.63 |
| **tactile** | **2,486.7 Hz** | **713.3 Hz** | **3.49** |

The touch organ is as strong on its own leg as the femoral chordotonal organ and
the most leg-selective of the three. (Rates are spikes per pool per phase summed
over the leg's ten muscle pools, scaled to Hz; a pool is many cells, so these
compare organs and legs, not cells.)

## What a footfall does (assumptions #41, #42)

One contact cycle — 300 ms on the floor, 100 ms in the air — driven through the
organ's own law (contact threshold 5 % of the standing load, hysteresis 0.5,
phasic peak 2.0×, adaptation 30 ms), leg T2 left:

| pool | contact | in the air | change |
|---|---|---|---|
| trochanter extensor | 382.5 Hz | 10.0 Hz | −372.5 |
| tibia flexor | 242.5 Hz | 47.5 Hz | −195.0 |
| coxa rotator ant | 550.0 Hz | 385.0 Hz | −165.0 |
| tibia extensor | 512.5 Hz | 385.0 Hz | −127.5 |
| trochanter flexor | 672.5 Hz | 607.5 Hz | −65.0 |
| femur reductor | 72.5 Hz | 25.0 Hz | −47.5 |

The leg's own pools lose 10–97 % of their rate when the foot leaves the floor,
most of all the trochanter extensor — the muscle that carries the body while the
foot is down. The other side's pools change too (the cord is not six isolated
legs), but by less and without the same ordering.

**Standing is a fixed point.** A foot in contact holds the adapted response
(h = 1), so the organ drives at exactly the tone the calibration phase clamps
every organ at: a standing animal is the same animal with the touch channel on
or off. `FlyCordTests.testTheTouchOrganReportsContactAndNotLoad` asserts it to
0.1 %, then lifts a leg (the drive must fall below 10 % of the tone within
100 ms) and lands it (the first millisecond above 1.5× the tone, adapting back
within 300 ms). `testTheTouchOrganDoesNotChatterAtTheContactThreshold` holds a
foot on the threshold for 200 ms and requires *no* onset — and re-runs the same
sequence with the hysteresis removed, where it must chatter, or the first
assertion would prove nothing.

## What this is not

* **Not an avoidance reflex.** Stimulating a few bristles on a *Drosophila* leg
  makes the animal move that leg away from the stimulus (Tuthill & Wilson 2016).
  This body can be in contact with the floor and nothing else: there is no
  obstacle, no probe, no geometry to touch, so there is nothing to move away
  from. The organ implemented here is the contact sense, and the direction of
  the reflex is a question for the environment item, not a constant to invent.
* **Not the antennae's, the proboscis's, the wings' or the abdomen's tactile
  cells** (6,698 in total; the legs are 3,184 and the eye's interommatidial
  bristles are 1,209 and are not touch in this sense). They are in the
  connectome, they are in no organ group, and `tools/verify_loop.py` prints the
  count of cells that no organ claims.
* **Not measured on hardware yet.** Everything above is the connectome and the
  reference LIF off-device. The device run that closes the loop is the next
  build; the HUD carries the channel's own input (`touch n/6` feet in contact)
  so the two failure modes — the channel is off, the animal is in the air —
  cannot be confused on the screen.

## Reproduce

```
bash tools/download_banc.sh
python3 tools/build_banc.py --out build/flybanc.bin
python3 tools/build_body.py --flybody data/flybody --out build/fly_body.json \
                            --meshes build/fly_meshes.bin
python3 tools/verify_loop.py
python3 tools/tactile_probe.py --gate
```
