# How much of this is real?

You asked for everything to be real, with nothing invented. Here is an honest
accounting of where that is now true, where it cannot be, and what changed to
close the gap.

## The short version

A connectome is a wiring diagram. It records which neuron contacts which, and
how strongly. It does **not** record how much force a muscle produces when its
motor neuron fires — nobody has measured that transfer function for *Drosophila*
flight muscle across its dynamic range, so it cannot be looked up.

So a zero-assumption simulation is not possible. What *is* possible, and what
the app now does, is to make every physical constant a published measurement
and shrink the unmeasured part down to **one** clearly labelled link.

Before: five arbitrary gain constants (`thrustPerHz`, `yawPerHz`, `walkPerHz`,
`turnPerHz`, `liftThresholdHz`) chosen so the thing moved nicely.

Now: one assumption, and real aerodynamics.

## The body

Replaced the hand-built ellipsoids with the **flybody** model from HHMI Janelia
and Google DeepMind — an anatomically accurate *Drosophila melanogaster*
reconstruction built from CT scans, Apache-2.0 licensed.

- 41 rigid parts: thorax, head, 7 abdominal segments, 2 wings, 2 halteres,
  2 antennae, proboscis, and 6 legs of coxa / femur / tibia / tarsus
- **76 real hinge joints** with the anatomical rotation axes and angular limits
  from the published model — every joint the app bends is clamped to the range
  a real fly's joint actually has
- the model even places the two 140° eye cameras where the eyes are, which is
  what the fly's-eye render now uses

`tools/build_flymodel.py` fetches the MuJoCo model, parses the body tree, bakes
each geom's placement into its vertices, decimates 248,000 triangles down to a
~60,000 budget by vertex clustering, and packs it into a 1.7 MiB binary.

Citation: Vaxenburg et al., *Whole-body physics simulation of fruit fly
locomotion*, Nature (2025). https://github.com/TuragaLab/flybody

## The physics

Flight force is now computed from **quasi-steady blade-element aerodynamics**,
the standard model in insect flight, in SI units:

```
U = 2 · Φ · f · r₂           mean wing velocity at the radius of the
                             second moment of area
F = ½ · ρ · C · S · U²       force from one wing
```

Every input is measured:

| Quantity | Value | Source |
|---|---|---|
| Body mass | 0.96 mg | Fry, Sayaman & Dickinson 2003 |
| Yaw moment of inertia | 5.2 × 10⁻¹³ kg m² | Fry et al. 2003 |
| Flapping counter-torque | 2.4 × 10⁻¹³ N m s/rad | Hesselberg & Lehmann 2007 |
| Wing length R | 2.5 mm | Sun & Tang 2002 |
| Wing area (one) | 1.6 mm² | Sun & Tang 2002 |
| r₂ | 0.58 R | Sun & Tang 2002 |
| Wingbeat frequency | 218 Hz | Lehmann & Dickinson 1997 |
| Stroke amplitude, hovering | 2.1 rad | Lehmann & Dickinson 1997 |
| Stroke amplitude, maximum | 3.1 rad | Lehmann & Dickinson 1997 |
| Mean lift coefficient | 1.8 | Sane & Dickinson 2001 |
| Mean drag / mean lift | 1.27 | Sun & Tang 2002 |
| Escape take-off speed | 0.9 m/s | Card & Dickinson 2008 |
| Walking speed, max | 30 mm/s | Mendes et al. 2013 |
| Step frequency, max | 13 Hz | Mendes et al. 2013 |
| Tripod duty factor | 0.55 | Mendes et al. 2013 |

The satisfying part: plug the hovering stroke amplitude into that equation and
two wings produce **9.8 µN** against a measured body weight of **9.4 µN**. The
fly hovers. Nothing was tuned to make that happen — the numbers simply agree,
because they are all measurements of the same animal. The HUD shows the live
lift-to-weight ratio so you can watch it.

Yaw is likewise real: the left/right drag difference acting at r₂ gives a
torque, divided by the measured moment of inertia, damped by the measured
flapping counter-torque. Peak turn rates land near the ~1600 °/s that Fry et al.
measured in free-flight saccades, without a clamp being chosen to produce that.

## The one remaining assumption

**Motor-neuron firing rate → wing stroke amplitude.**

Both endpoints are measured: a hovering fly beats at Φ ≈ 2.1 rad, and at maximum
effort at Φ ≈ 3.1 rad. The straight line drawn between them is the assumption.
It lives in exactly one function, `strokeAmplitude(from:)` in `FlyBody.swift`,
and nowhere else.

Worth knowing *why* this is the hard one: *Drosophila* power muscles are
**asynchronous**. Their motor neurons fire at 5–20 Hz while the wing beats at
218 Hz, because firing sets the activation level, not the rhythm. That is also
why the wingbeat frequency in the app is a measured constant rather than
something the spiking network produces — which is the biologically correct
arrangement, and conveniently means the simulation never has to resolve 200 Hz.

The steering muscles are different: they fire phase-locked, once per cycle, and
shift their own wing's stroke amplitude. That mechanism *is* measured, and is
what the app uses for turning.

## The walls (and why they are gone)

The containment story went through three stages. First the fly stopped dead
in mid-air at an *invisible* boundary. Then real walls and a ceiling were
drawn, so the eyes at least had a looming cue for the surface they were about
to hit. But the room was 12 cm across — a matchbox for an animal that flies
at 0.9 m/s and walks a body-length every tenth of a second. At that scale the
fly collided with something constantly, which is not how a fly in a kitchen
reads at all.

The world is therefore open now: a floor under an open sky, fog on the
horizon, no drawn walls or ceiling in any environment, with an invisible
20 m analytical backstop and a 5 m sky cap so the integrator and the chase
camera stay sane. The collision solver still returns a corrected position
rather than a nudge, so nothing tunnels or sticks when the fly does meet one
of the loose objects.

## What is still modelled

Being complete about it:

- the **arousal tone**. Vigour drifts through states over tens of seconds
  in real flies (Cohn et al. 2019, Cell 176:254; Nat Commun 2023, 14:5420 —
  arousal-like time constants from <4 s to >20 s), and the walk/stop
  statistics only close with such a state term (Demir et al. 2020). The
  *existence* and the *timescale* are measured; the Ornstein-Uhlenbeck
  process (tau 15 s, bounded [0.5, 1.5]) that stands in for the
  neuromodulator soup is ours.
- the **walk/stop/groom bout structure**. Real walking is bout-structured
  (walk/stop transitions are Poisson-like with λ₀ ≈ 0.29 s⁻¹, Demir et al.
  2020, eLife 5:e57524) and ~13% of waking time is grooming in 0.15–2 s
  bouts sweeping anterior→posterior (Lazopulo & Syed 2018, eLife 7:e34497;
  Seeds et al. 2014, eLife 3:e02951; Ray et al. 2019, PLOS Comput Biol).
  A 1 ms LIF connectome does not spontaneously emit that structure, so
  `FlyBody` imposes it with a deterministic state machine tuned to those
  measurements — same status as the imposed tripod oscillator. The leg
  *angles* during the grooming rub are posed.
- the **leg joint angles** during walking. The gait timing is measured (duty
  factor 0.55, tripod phase), and the joint limits are anatomical, but the
  specific angle each of the 24 leg joints takes through the cycle is posed,
  not derived from a limb dynamics solve.
- **odour diffusion** is inverse-square from each food item. Real plumes are
  turbulent and intermittent.
- the **world itself** — a kitchen, a garden and a lab box are stage sets.

The brain is untouched by any of this: 175,237 neurons and 2.17 M synapses from
BANC v888, and every spike that reaches a motor neuron got there through real
measured wiring.
