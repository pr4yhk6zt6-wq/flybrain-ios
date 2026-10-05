# ASSUMPTIONS

Every constant in this repository is one of two things:

* **measured** — published, or read out of a source file at build time by a
  script anyone can re-run. Cited where it is used.
* **an assumption** — listed here, numbered, with a reason and with the test
  that will catch it if it is wrong.

There is no third category. In particular there is no category for "it made the
walk look better", which is what the deleted body model was full of.

**Current total: zero.** Step 1 introduces no constants of its own. Everything
`tools/step1_anatomy.py` prints is read out of the MJCF or out of BANC, and
everything `tools/step1_sim.py` does is MuJoCo integrating the model as shipped.
The first assumption will arrive with the first controller, and it will be
numbered #1 when it does.

---

## The two source files

| file | what it fixes | licence |
|---|---|---|
| `flybody/fruitfly/assets/fruitfruit.xml` | the body: 67 parts, 102 DOF, masses, joint axes and limits, the 8 adhesion actuators | Apache-2.0 |
| `banc_888_meta.feather` | the nerves: which cell is a motor neuron, which muscle it innervates, which side it is on | CC BY 4.0 |

## Removed, and why

The whole previous stack was deleted in the commit that created this file. The
constants it carried — `thrustPerHz`, `yawPerHz`, `walkPerHz`, `turnPerHz`,
`liftThresholdHz`, the arousal time constant, the habit-layer bout rates, the
tangent gait gains — were gains chosen so that a hand-imposed oscillator would
produce motion that looked like walking. They are preserved in the
`archive/round5-world` branch. None of them are coming back; behaviour in this
repository is produced by circuits, not by a schedule.
