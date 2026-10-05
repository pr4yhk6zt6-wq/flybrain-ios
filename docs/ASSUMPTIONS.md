# ASSUMPTIONS

Every constant in this repository is one of two things:

* **measured** — published, or read out of a source file at build time by a
  script anyone can re-run. Cited where it is used.
* **an assumption** — listed here, numbered, with a reason and with the test
  that will catch it if it is wrong.

There is no third category. In particular there is no category for "it made the
walk look better", which is what the deleted body model was full of.

Step 1 introduces no constants of its own: everything `tools/step1_anatomy.py`
prints is read out of the MJCF or out of BANC, and everything `tools/step1_sim.py`
does is MuJoCo integrating the model as shipped.

**Current total: 6.** All six arrive with step 2, the first step that simulates
the nerve cord rather than measuring it.

| # | assumption | value | why | test that catches it |
|---|---|---|---|---|
| 1 | synaptic scale | N\* = 12 partners at 100 Hz reach threshold | the raw weights are synapse counts, which have no units; one circuit-wide convention beats a per-connection fudge | every result is reported across a sweep of N\* (6/12/24/48). **Test result: the tibia reflex direction flips between N\* = 12 and N\* = 24, so that direction is not claimed.** |
| 2 | transmitter sign | `neurotransmitter_predicted`: ACh +1; GABA, glutamate, histamine −1; unknown 0 | insect convention (ACh excitatory, GABA/glutamate inhibitory); the column is a *prediction* from BANC's classifier, not anatomy | the sign-shuffle control (permute transmitters within cell class). **Test result: the shuffle takes the flexor pool from 2.44 to 23.78 Hz — the sign table is load-bearing, so this assumption is the largest weakness in step 2.** |
| 3 | cell model | conductance-based LIF; E_exc = +4, E_inh = −0.25, τ_m = 20 ms, τ_syn = 5 ms, t_ref = 2 ms, noise 0.015, threshold 1, rest 0 | additive LIF let inhibition reach −1.4× threshold, which inhibitory receptors physically cannot do; the values are the usual normalised units | report the all-silent and all-saturated boundaries. **Test result: silent for N\* ≥ 24, saturated for N\* ≤ 6, physiologically plausible at N\* ≈ 12.** |
| 4 | axonal delay | soma-to-soma distance / 0.25 m/s, clipped to 1–8 ms | the distances are measured from `root_position_nm`; the speed is a stated conduction velocity | median 1 ms, p90 2 ms — sub-millisecond differences do not change the sign of the readouts, and the tool prints the distribution |
| 5 | descending tone | a standing fly's brain drives its descending neurons at 2.5 (threshold units); a sensory population under test gets +3.0 | the descending neurons are the only cells that carry the brain's state into this subgraph, and a standing fly's cord is not silent | sweeping the tone from 1.0 to 3.0. **Test result: below 2.0 the motor pools are silent, above 3.0 they saturate; 2.5 is the first value where both tibia antagonists are active together, as they are in a standing leg.** |
| 6 | scope of the circuit | the ≤3-hop neighbourhood of the leg's organs and motor neurons: 38,181 of 188,508 cells | the full connectome would make the sign of every 4-hop path meaningless | a cell that matters to the reflex but sits 4+ hops away is absent. Not yet measured; this is the known hole. |

**Not modelled at all, and therefore not claimed:** gap junctions (electrical
synapses), neuromodulation, muscle dynamics, synaptic plasticity, and any
per-connection weight that is not the synapse count.

---

## The two source files

| file | what it fixes | licence |
|---|---|---|
| `flybody/fruitfly/assets/fruitfruit.xml` | the body: 67 parts, 102 DOF, masses, joint axes and limits, the 8 adhesion actuators | Apache-2.0 |
| `banc_888_meta.feather` | the nerves: which cell is a motor neuron, which muscle it innervates, which side it is on, where the soma sits, and the predicted transmitter that assumption #2 rests on | CC BY 4.0 |
| `connections_princeton.csv.gz` | the synapses: who contacts whom, and how many release sites | CC BY 4.0 |

## Removed, and why

The whole previous stack was deleted in the commit that created this file. The
constants it carried — `thrustPerHz`, `yawPerHz`, `walkPerHz`, `turnPerHz`,
`liftThresholdHz`, the arousal time constant, the habit-layer bout rates, the
tangent gait gains — were gains chosen so that a hand-imposed oscillator would
produce motion that looked like walking. They are preserved in the
`archive/round5-world` branch. None of them are coming back; behaviour in this
repository is produced by circuits, not by a schedule.
