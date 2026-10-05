# Step 2 — the leg reflex, out of the connectome

Step 1 gave us a body and a nerve cord. Step 2 asks the first question that needs
both: **does the connectome contain the reflex a fly stands on?**

Everything here comes from `tools/step2_reflex.py` reading BANC v888; the raw
numbers are in `reports/step2_reflex.json`, regenerated on every run.

Reproduce:

```
python3 tools/step2_reflex.py --leg front_leg --side left --ms 700 --desc 2.5 --stim 3.0
```

## 0. Four bugs, three of them in the previous version of this tool

The previous run of this step reported a disynaptic weight of **+13,904** to the
tibia flexor and called the reflex excitatory. That number was wrong, and the
reason it was wrong is worth keeping:

1. `disynaptic()` called `np.flatnonzero()` on an array that already held neuron
   *indices*. NumPy read the indices as a boolean mask, so the table was computed
   against the first 17 neurons of the dataset instead of the 17 flexor motor
   neurons. **Corrected: the flexor pool receives −33,347, the extensor −8,091.**
2. `root_position_nm` is `"x, y, z"`, not `"[x y z]"`. The parser tested for
   brackets, never matched, and set every coordinate — and therefore every axonal
   delay — to zero.
3. `simulate()` normalised the weights by the median postsynaptic drive. That
   cancelled whatever scale the caller set, which is why every value of the
   synaptic scale (K) produced byte-identical output.
4. In the conductance version, the refractory counter was set but never
   decremented, so every neuron fired **once per run** — which reads as a cell
   rate of exactly 1/(run length), and 1.25 Hz for the descending neurons was the
   tell.

## 1. Two mechanisms that were missing, and are now assumptions

* **A synaptic scale set by a stated convention.** The old rule left one spike
  worth ~0.1 of threshold, so nothing survived the first synapse. Now: *N\*
  presynaptic partners firing together at 100 Hz should drive an average neuron
  to threshold*, with N\* = 12 (assumption #1) and every result also reported
  across a sweep of N\*.
* **Conductance-based synapses.** An additive LIF let inhibition push the tibia
  flexor to −1.4× threshold. Real inhibitory receptors have a reversal potential.
  Excitatory and inhibitory spikes now add conductance, and the membrane equation
  is `dV/dt = (−V + g_e(E_e−V) + g_i(E_i−V))/τ_m` (assumption #3).

## 2. What the organ touches

The front-left chordotonal organ is 38 cells, **35 cholinergic / 3 GABAergic** —
an excitatory sense organ, so the interneurons it contacts are depolarised by it,
and the sign of those interneurons is the sign the motor pool sees.

| synapses | target | class | transmitter | sign |
|---:|---|---|---|---:|
| 571 | `IN13B001` | intrinsic | GABA | − |
| 279 | `IN21A009` | intrinsic | glutamate | − |
| 253 | `pleural_remotor_and_abductor` | motor | ACh | + |
| 240 | `IN14A001` | intrinsic | ACh | + |
| 239 | `IN16B038` | intrinsic | GABA | − |
| 232 | `IN08A006` | intrinsic | ACh | + |

The strongest target is a **13B-lineage interneuron** — the cell population the
FeCO literature names as the organ's first central partner (Agrawal et al. 2020;
Mamiya et al. 2018). That is a match against published anatomy, not a tuned one.

### Direct (monosynaptic) paths to the tibia motor pools

| organ | → tibia flexor | → tibia extensor | → trochanter flexor |
|---|---:|---:|---:|
| chordotonal | **0 arcs** | 0 arcs | 23 arcs / 119 syn |
| campaniform | 1 arc / 3 syn | 0 | 19 arcs / 117 syn |
| hairplate | 8 arcs / 29 syn | 5 arcs / 27 syn | 0 |

Both of these are the published results, reproduced in the connectome rather than
assumed:

* Azevedo et al. 2020 report the FeCO's signal reaching the tibia motor neurons
  **through distributed polysynaptic pathways** — and here the femoral chordotonal
  organ has *no* direct connection to either tibia motor pool.
* Phelps et al. 2021 report trochanter campaniform sensilla connecting **directly
  to flexor tibiae motor neurons** — and here that is exactly one arc, 3 synapses,
  to the flexor and nothing to the extensor.

## 3. The disynaptic table, corrected

Weight = synapses from the organ into the interneuron × signed synapses from the
interneuron onto the pool, summed over every interneuron in the path.

| pool | motor neurons | reached | interneurons | arcs | excitatory | inhibitory | net | net per motor neuron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tibia flexor | 17 | 16 | 58 | 162 | 12,909 | −46,256 | **−33,347** | −1,962 |
| tibia extensor | 2 | 2 | 38 | 54 | 4,010 | −12,101 | **−8,091** | −4,046 |
| trochanter flexor | 11 | 11 | 104 | 398 | 22,634 | −282,185 | −259,551 | −23,596 |
| coxa rotator post | 4 | 4 | 108 | 242 | — | — | −115,550 | −28,888 |

The strongest single contributor is `IN21A006` (glutamatergic, inhibitory; 88
synapses from the organ, 56 + 38 + 37 onto flexor motor neurons), then
`IN03B019` (GABA, 145 from the organ).

Two readings are possible and they disagree, so both are reported: in **total**
weight the flexor receives 4.1× the extensor; **per motor neuron** the extensor
receives 2.06× the flexor, because the extensor pool is 2 cells. A connectome
weight ratio cannot decide the reflex on its own — which is the whole reason the
rest of this step is a simulation.

## 4. The organ is not one population

Clustering the 38 cells by their *target profile* (correlation distance on the
normalised distribution of their synapses over downstream cells, k = 3):

| cluster | cells | types | strongest targets |
|---|---|---|---|
| cluster1 | 8 | SNpp52 (5), CoHP8 (3) | `IN13B005`, `IN13A061`, `IN13A047` |
| cluster2 | 4 | SNpp45 (4) | `IN19A061`, `IN13B065`, `IN19A021` |
| cluster3 | 26 | SNpp45 (16), SNpp52 (6), CoHP8 (2), CS (2) | `IN13B001` (565), `ANXXX041` (446) |

This is the connectome's own version of the FeCO's claw / hook / club split
(Mamiya et al. 2018: 152 cells; our 38 front-left cells, 3 clusters). It is not
the same partition and should not be claimed as it — but it is derived from the
wiring, not from a label.

## 5. The experiment: a standing fly

The subgraph is every cell within 3 hops of the leg's sense organs *and* within
3 hops of its motor neurons: **38,181 neurons, 1,375,595 edges**, of which 1,177
are descending (brain → cord). **Standing** = the descending neurons driven
tonically at 2.5 (threshold units; the fly's brain maintains tone; assumption #5).
Delays are soma-to-soma distances over 0.25 m/s (assumption #4).

Standing state, motor pools (Hz):

| pool | Hz | pool | Hz |
|---|---:|---|---:|
| tibia flexor | 2.69 | tibia extensor | 14.29 |
| trochanter flexor | 26.88 | trochanter extensor | 21.25 |
| femur reductor | 40.24 | coxa rotator anterior | 120.71 |
| coxa rotator posterior | 17.86 | long tendon / tarsus | 0.00 |

A standing insect holds both antagonists of the femur–tibia joint active, and
that is what the network does here without anyone setting a rate. The tarsus
pools are silent because they are not in the standing reflex path.

## 6. What the sense organs do to the motor pools

Change from the standing state. **flex−ext** is flexor minus extensor; positive
means the condition favours the flexor, i.e. flexion.

| condition | Δflex Hz | Δext Hz | flex−ext Hz | Δg flexor | Δg extensor | flex−ext Δg |
|---|---:|---:|---:|---:|---:|---:|
| + cluster1 (8 cells) | −0.25 | −0.71 | **+0.46** | −0.0156 | −0.0077 | −0.0080 |
| + cluster2 (4 cells) | −0.08 | +0.71 | −0.80 | +0.0004 | +0.0022 | −0.0018 |
| + cluster3 (26 cells) | 0.00 | 0.00 | 0.00 | −0.0179 | −0.0248 | +0.0069 |
| + lineage 13B (24 cells) | −0.08 | −0.71 | **+0.63** | +0.0050 | +0.0149 | −0.0099 |
| + lineage 09A (7 cells) | 0.00 | −0.71 | +0.71 | +0.0046 | +0.0086 | −0.0040 |
| + lineage 10B (6 cells) | 0.00 | 0.00 | 0.00 | −0.0025 | −0.0005 | −0.0021 |
| + lineage 13A (51 cells) | −1.09 | −6.43 | +5.34 | −0.0490 | −0.0426 | −0.0064 |
| **release** (organ silenced) | −0.50 | +0.71 | **−1.22** | −0.0362 | −0.0285 | −0.0077 |
| **+ load** (campaniform) | −0.08 | −0.71 | **+0.63** | −0.0027 | −0.0109 | **+0.0082** |

`Δg` is the change in net synaptic conductance arriving at a pool: what the
wiring delivers, before any threshold is applied.

**What came out right:**

* **Release.** Silencing the chordotonal organ costs the flexor 0.50 Hz and
  *raises* the extensor: the tonic organ is what holds the flexor up.
* **Load.** The campaniform (load) organ is the one condition whose conductance
  metric favours the flexor (+0.0082) while everything else favours the extensor.
* **13B.** Driving the 13B lineage biases the tibia toward the flexor
  (+0.63 Hz), which is what the optogenetics says (Agrawal et al. 2020: 13Bα
  activation produces tibia flexion).
* **Monosynaptic anatomy** of all three organs (§2).

**What came out wrong, and stays wrong:**

* **09A.** The published result is that 9Aα activation produces **extension**.
  Here driving 09A biases the tibia toward the flexor (+0.71 Hz). At the standard
  operating point this prediction **fails**.
* The direction of the tibia reflex is **not stable across the model's own free
  parameter**. At N\* = 6 the network is saturated (flexor 78 Hz, extensor 97 Hz);
  at N\* = 24 and 48 it is silent (0 Hz), and the Δg ordering there favours the
  extensor. Only N\* ≈ 12 puts it in a regime where the motor pools are doing what
  a standing fly does.
* The two summary metrics disagree: the firing-rate metric favours the flexor for
  13B / cluster1 / campaniform, the conductance metric favours the extensor. The
  extensor pool is **two cells**, so its rate is quantised in whole spikes and
  this disagreement cannot be resolved from the data in front of us.

## 7. The control that matters

Permute the transmitter identities within each cell class (so the total
excitatory/inhibitory proportion per class is preserved, but not *which* cell is
which):

| pool | connectome signs | shuffled signs |
|---|---:|---:|
| tibia flexor | 2.44 Hz | **23.78 Hz** |
| tibia extensor | 13.57 Hz | **103.57 Hz** |
| trochanter flexor | 31.95 Hz | 78.57 Hz |
| coxa rotator posterior | 18.57 Hz | 75.71 Hz |

An order of magnitude. It is not the overall excitation/inhibition ratio that
sets the standing state — that is preserved by the shuffle — it is **which**
cells carry which transmitter. Which makes the sign table load-bearing, and it is
a classifier's prediction, not anatomy: assumption #2. That is the largest single
weakness in this step.

## 8. What step 2 does and does not establish

**Establishes:** the connectome contains a polysynaptic pathway from the leg's
proprioceptors to its motor pools, with the published anatomy of all three organs
intact (no direct chordotonal → tibia motor neuron; one direct campaniform →
flexor; 13B as the organ's first partner); and that tonic organ activity plus
load signalling both act to support the flexor, which is the pair of effects the
next step needs.

**Does not establish:** which way the tibia reflex goes. That depends on a free
scale parameter, on a transmitter prediction, and on a 2-cell motor pool.

**Step 3** is therefore not "add more model". It is: close the loop — feed these
motor pool rates into the step-1 muscles and joints and check whether the standing
spine still holds — and then put the leg under load and look for the switch the
reflex literature predicts, resistance turning into assistance. The two effects
that survived this step (organ tone → flexor support; load → flexor support) are
exactly the two the switch is built from.
