# Item 4 — the walking circuits: what the dataset decides first

Item 4 asks for six legs that coordinate, with a gait that emerges rather than
being scheduled. Whether that is possible is not a design question — it is a
measurement of BANC v888 as packed into `flybanc.bin`, and it has now been made
three ways. Two of the three are in `tools/coupling_probe.py`, the third in
`tools/gait_probe.py`; all three are reproducible off-device in seconds.

## 1. The sense organs cannot couple the legs (structural)

Every outgoing edge of one leg's organ cells (`organ:<leg>:<channel>`, three
channels per leg), walked off the packed CSR:

| | own leg | other legs | ratio |
| --- | --- | --- | --- |
| one hop | 296.7 synapses | 3.5 synapses | **85:1** |
| two hops | 57.5% of that leg's pool MNs | 2.7% | **21:1** |

A leg's reflex route — chordotonal/campaniform/hairplate → pool → muscle → joint
— is essentially private to the leg it serves. The closed loop of steps 2–3 is
therefore a *local stabiliser*: it can hold a leg and resist a push on it, and it
cannot, by itself, tell one leg what another leg is doing. That is consistent
with the loop's own measurements (a resistance reflex that reverses under load)
and it is the reason a gait cannot be expected to fall out of the organ→pool
loop however its gains are tuned.

## 2. The cord can couple them (structural)

7,033 cells in the binary reach any pool motor neuron at all. Of those:

| onto how many legs | cells | share of premotor cells |
| --- | --- | --- |
| 1 leg | 6,245 | 88.8% |
| 2 legs | 483 | 6.9% |
| 3 legs | 208 | 3.0% |
| 4 legs | 45 | 0.6% |
| 5 legs | 26 | 0.4% |
| 6 legs | 26 | 0.4% |

788 cells drive more than one leg's pools, carrying **69,543 synapses** into pool
motor neurons. Adjacent pairs share the most (T2_right↔T3_right 246,
T1_right↔T2_right 228, T2_left↔T3_left 227) and the six diagonal pairs a tripod
would need share 592 cells between them. The substrate for coordination exists.

It now has a name in the binary: `premotor:multileg`, computed by
`tools/build_banc.py` from the graph (not the neuron table), **788 cells**, and
required by CI (`tools/coupling_probe.py --structural-only --require-coupling`,
which also fails if the *named* group is missing). The packer's forward walk and
the probe's reverse walk agree on 788 independently.

## 3. The cord does not, by itself, walk (dynamic)

`tools/gait_probe.py` runs the LIF at the app's operating point (gain 12, the
tone of assumption #5 on the descending group and on every organ channel, retina
flat at 1.0), bins every pool group's per-millisecond spike counts into 20–25 ms
windows, and asks the one question a tripod gait predicts: do the legs of one
tripod co-vary more than legs across tripods?

    tripod index = mean corr(within {T1L,T2R,T3L} and {T1R,T2L,T3R})
                 − mean corr(across)

| condition | within | across | index |
| --- | --- | --- | --- |
| 400 ms, app's point (gain 12) | +0.373 | +0.379 | −0.006 |
| 400 ms, `premotor:multileg` driven at 5 | +0.327 | +0.451 | −0.124 |
| 400 ms, organs silent (cord alone) | +0.530 | +0.539 | −0.009 |
| 2000 ms, app's point | +0.060 | +0.133 | −0.073 |
| 400 ms, gain 2 (sparsest: 11–28 Hz) | +0.475 | +0.425 | +0.050 |
| 400 ms, gain 4 | +0.015 | +0.100 | −0.085 |
| 400 ms, gain 6 | +0.344 | +0.451 | −0.107 |
| 400 ms, gain 8 | +0.282 | +0.346 | −0.063 |
| 400 ms, descending tone only (no organs, no retina) | +0.486 | +0.447 | +0.038 |
| 2000 ms, descending tone only | +0.160 | +0.163 | −0.003 |
| 400 ms, descending tone only at 8 | +0.136 | +0.019 | +0.117 |

Eleven conditions, and the index has no sign: it wanders in ±0.12 with the
correlations always positive (+0.02 to +0.54). The six legs' pool rates move
together, whatever is driving them and however hard the network is pushed — a
common chaotic mode, with the 788 multi-leg cells (11% of the premotor
population) too few to organise it. Even with the descending tone as the only
structured input, the cord produces no tripod.

**So the honest statement for item 4 is: this cord, driven uniformly, does not
walk, and no amount of tuning the existing loop will make it.** The two things
that can change are (a) the drive — a real fly's descending command is patterned,
not a constant tone, and a real fly's legs are in different states at every
instant — and (b) the operating point: the pools are firing at 30–65 Hz here,
which is not the sparse regime a walking VNC runs in.

## What follows

* The substrate is measured, named, packed and gated (done in this commit).
* The mechanism for the step rhythm is *not* in the cord's rates, so if the gait
  is to be built it needs a declared mechanism — a local oscillator per leg with
  its coupling as a parameter, labelled PLACEHOLDER per the brief, with
  `tools/gait_probe.py` as the instrument that measures what it changes and
  `coupling_probe.py`'s numbers as the target it is trying to reproduce from the
  dataset (592 shared cells between the tripods).
* The other lever — a sparser operating point where the cord's own structure is
  not buried by saturation — is a measurement to make *before* committing to the
  oscillator, and it is cheap: the same tool at other gains and tones.

This is written down before any gait code exists, because the alternative is a
gait that looks right on the screen and cannot be traced to the connectome.

## 4. The loop, with the body answering the cord (dynamic, in the loop)

The probes above run the cord as a *rate source*: drives in, rates out. In the
animal the rates go to muscles, the muscles move a body, and the body's joints
and foot forces come back as organ drives. `tools/walk_loop.py` is that loop,
headless, using the same assets and the same laws the app runs (`FlyCord`'s
balance and muscle filter, `fly_aba`'s muscles and articulated body, and the
organ laws #11/#12 fed from the body's own state).

| run, 1 s each (300 ms of it calibration) | tone only | cord, organs frozen | cord, closed loop | closed calibration |
| --- | --- | --- | --- | --- |
| pools firing | — | 51–61 Hz | 51–61 Hz | 51–61 Hz |
| legs that left the floor | 0/6 | 5/6 | 5/6 | 6/6 |
| foot-load, of the stance | 1.00 each | 0.35–2.36 | 0.30–2.42 | 0.38–1.71 |
| COM z | −0.0227 → −0.0233 cm | → −0.0419 | → −0.0481 | → −0.0321 |
| horizontal drift | 0.0017 cm | 0.0092 | 0.0575 | 0.1576 |
| foot-load tripod index | — | +0.131 | +0.114 | +0.265 |

Read honestly:

* **With muscle tone alone the animal stands**, and stays where it is put. That
  is the baseline the rest of the table is measured against.
* **With the cord attached, the pose distorts.** The pools fire hard (51–61 Hz),
  the command does not saturate (|offset| ≤ 0.50 — the command gain of assumption
  #24 is 0.5; note that the *LIF's* gain, 12, is a different quantity entirely,
  and passing it here thrashes the body, which is what the first version of this
  tool did by mistake), and yet the animal presses into the floor and leans on
  two legs while lifting the others.
* **The sensory feedback is not what does it**: freezing every organ at its
  standing value gives the same picture, so the cord's *resting* command is
  already biased. The feedback is not the problem.
* **Why the bias exists is measurable**: the reference balance is calibrated with
  the organs clamped at tone (assumption #10's method) and the loop then runs
  with the organs live, so the network is calibrated in one operating point and
  run in another. Measuring the reference in the regime the loop runs in halves
  the sinking (COM −0.032 vs −0.048 cm) and pulls the loads together (0.38–1.71
  vs 0.30–2.42) — with a cost in drift, because a setpoint that follows the
  standing pose is a setpoint that lets the animal wander.
* **There is an alternation signal and it is not a gait.** The tripod index comes
  out positive in both closed runs, but from *across*-tripod anti-correlation
  (−0.24 to −0.30): the two tripods move against each other while neither moves
  coherently with itself. Six legs dragging themselves into the floor in two
  groups is not a walk, and calling it one would be the exact mistake this
  project's rules exist to prevent.

**What this means for the phone.** The device build behind `IMG_2713` printed
`0/42 pools firing` and stood still *because its pools were silent*. Once the
readout is fixed and they fire, this loop is what the app will do: not a stand,
and not a walk — a distorted, restless stance. That is a known, measured
consequence of fixing the readout, not a new failure, and it moves item 4 to the
critical path:

1. the operating point (tone, LIF gain, command gain) has to be re-measured in
   the loop as one of the two ends of a trade — pool rates high enough for the
   cord to sense its own state, low enough for the balance to carry information;
2. the reference balance has to be measured in the regime the loop runs in, in
   both the tool and `FlyCord.swift`;
3. and only then is a gait's *absence* or *presence* a statement about the cord
   rather than about the setup.


## 5. Two levers, measured (the operating point, and the command)

The loop above says the cord as it stands distorts the stance. Two knobs can
change that, and both were measured in the loop (700 ms, closed calibration,
the app's LIF gain 12):

| condition | pools in the loop | loads, of stance | legs lifting | knee motion |
| --- | --- | --- | --- | --- |
| tone 2.5, command gain 0.5 (the app) | 61 Hz | 0.30–2.42 | 5/6 | 0.20–0.46 rad |
| tone 2.5, command gain **0.1** | 67 Hz | **0.52–1.36** | 3/6, 0–2 steps | 0.06–0.14 rad |
| tone 5.0, command gain 0.5 | 85 Hz | 0.12–3.89 | 5/6 | 0.20–0.49 rad |
| tone 1.0, command gain 0.5 | 56 Hz | 0.54–2.64 | 6/6 | 0.20–0.46 rad |

The trade is exact and it is between the two things item 4 needs at once:

* the command gain sets how much the cord can *move* the body, and how much it
  *disturbs* the stance — at 0.1 the animal stands almost evenly (loads 0.52–1.36,
  knee motion 0.06–0.14 rad) and at 0.5 it distorts (0.30–2.42, up to 0.46 rad);
* the tone sets how hard the network runs, and a higher tone makes the distortion
  worse (tone 5.0: loads 0.12–3.89) while a lower one does not fix it (tone 1.0
  still 0.54–2.64), because the bias is in the *balance*, not in the scale.

So a constant tone cannot be both: it is a hold, and a hold at a gain that can
move a leg is a hold that can also lean on it. What a walking insect has, and
what this loop does not, is a *patterned* descending command. `walk_loop.py` can
now inject one (`--desc-pattern HZ:AMP`, the tone plus AMP·sin(2π·HZ·t) on the
descending population), and the question it answers is the one this whole item
turns on: does the cord's own wiring turn a modulated command into alternating
tripods, or do all six legs simply follow the drive in phase? A tripod index that
grows with the modulation means the connectivity is doing the coordination and
the pattern is a command; a tripod index that stays at zero means the six legs
are six copies of the drive, and item 4 needs the coupling to be built rather
than found.
