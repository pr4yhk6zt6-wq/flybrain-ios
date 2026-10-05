# Step 1 — a fly with a real brain and a real nerve cord, that really moves

## The goal

One animal, built in this order, with nothing skipped:

```
    brain        ─┐
                  ├─  the 532 motor neurons that exist in BANC v888
    nerve cord   ─┘        │
                           ▼
                       muscles              the 70 joint actuators + 8 adhesion
                           │               actuators of the flybody model
                           ▼
                       joints              102 degrees of freedom, anatomical
                           │               axes and limits from the CT scan
                           ▼
                        body               a floor, friction, gravity, contact
                           │
                           ▼
    sensors      ◀────  proprioception, tactile, vision, halteres, smell
        │                  (6,699 tactile · 2,903 proprioceptive · 9,733 visual …)
        └──────────────▶  back into the cord and the brain
```

The loop closes. No stage is replaced by a script.

## What "reference" means for this repository

The Janelia news article this work follows describes the same animal and the
same body model, and solves the control problem with a network trained on
recordings of real flies:

> "You take real fly data – how real flies fly and how real flies walk – and
> train the network to mimic these motions … It is like a little brain that
> controls the fly's movements." — Roman Vaxenburg

and it ends by naming the step we are taking:

> "They also want to be able to use a real neural network, like the fruit fly
> ventral nerve cord connectome, to power the model."

So the body model, the physics, and the anatomy come from that work directly.
The controller does not: it is the connectome, with the local circuits the
ventral nerve cord actually contains.

## The pieces, and where each one stands

| # | piece | where it comes from | state |
|---|---|---|---|
| 1 | the body | `flybody` MJCF, read by MuJoCo | **done, measured** |
| 2 | the nerve structure | BANC v888 meta, muscle-level annotations | **done, measured** |
| 3 | standing under gravity | MuJoCo + the 8 adhesion actuators | **done, measured** |
| 4 | proprioceptive feedback | BANC `proprioception` (2,903) + chordotonal/campaniform detail | **extracted and measured — [`STEP2.md`](STEP2.md)** |
| 5 | leg motor → muscle → joint | `MUSCLE_TO_ACTUATOR` in `tools/step1_anatomy.py` | **mapped; drive not yet closed** |
| 6 | the walking rhythm | thoracic circuits + sensory entrainment | after 4–5; blocked on the reflex loop |
| 7 | descending commands | BANC `descending` (1,316) — walk, stop, turn | after 6 |
| 8 | escape | giant fibre → TT motor neurons → take-off | after 6 |
| 9 | flight | wing power/steering pools + halteres | after 6 |
| 10 | vision | the fly's two eyes → the optic lobe | last |

## Rules this step holds itself to

1. **Nothing is scheduled.** If the animal does not walk, the answer is a
   missing circuit, not a timer. The deleted `Habit` state machine
   (`walk`/`stop`/`groom` with exponential bout lengths) is exactly what is not
   allowed to come back.
2. **Every claim is a number in a report.** `reports/step1_anatomy.md` is
   generated on every build and CI fails if the anatomy stops matching the
   published animal — the wrong part count, the wrong mass, the wrong length.
3. **Negative results are published.** The previous round found that a looming
   stimulus did not produce looming selectivity in LC4/LPLC2/DNp01 and wrote
   that down instead of tuning the gain. That habit stays.
4. **A gait is not a pattern generator with a nice name.** Real insect walking
   is produced by thoracic circuits under continuous sensory entrainment:
   proprioceptors report joint angle and load, and those signals reset and
   entrain the rhythm. A model that emits tripod phases from a clock is a
   *timing* model, and it is not this.

## The two honest obstacles, stated up front

These are the reasons the previous round failed, and they have not gone away:

**A 1 ms LIF network over BANC does not spontaneously produce a clean gait.**
That is a fact about the model class, not about the fly. Real thoracic circuits
produce the rhythm through intrinsic membrane properties, synaptic depression
and facilitation, and gap junctions — none of which a uniform LIF without
adaptation or short-term plasticity has. So the plan is to use the connectome's
**structure** (which cell is a motor neuron, which muscle it hits, which
proprioceptor reports which joint, which premotor cells are interposed) and to
give the neurons the dynamics that make a real cord oscillate, with each added
mechanism cited and tested. The alternative — a clock with the
connectome's names painted on it — is what was deleted.

**Nobody has calibrated motor neuron rate → muscle force for *Drosophila*
walking.** It is unmeasured, so it will be assumption #1 in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md), with its endpoints taken from published
forces where they exist and its test named before it is written.
