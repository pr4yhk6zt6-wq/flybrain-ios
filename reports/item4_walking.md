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
