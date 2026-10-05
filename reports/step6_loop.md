# Step 6 — the loop, closed on the phone

**Status: the loop is wired; the reflex through the app's own engine is not yet
measured on a device.** Nothing here claims the animal walks.

## What is in the build (commit "step 6: close the loop")

| file | what changed |
|---|---|
| `tools/motor_pools.py` | one table for pools *and* organs: `POOLS`, `POOL_JOINT`, `ORGANS`, `pool_slug()`, `organ_slug()`, `pool_groups()`, `organ_groups()` |
| `tools/build_banc.py` | emits `organ:<seg>_<side>:<organ>` groups beside the pool groups; reads `cell_function_detailed` |
| `tools/build_body.py` | names pools and organs through `motor_pools.py`; its pool table is step 3's mapper's output, unchanged |
| `tools/step2_reflex.py`, `tools/step3_closedloop.py` | read the shared tables instead of their own copies |
| `tools/verify_loop.py` *(new)* | both ends of the loop exist in the shipped files, by the same names, and no group is empty |
| `FlyBrain/Sources/FlyCord.swift` *(new)* | organ → cord → muscle, with `b_ref` measured on the device |
| `FlyBrain/Sources/FlyLiveBody.swift` | the cord is asked once per **simulated millisecond**; the excitation is held between updates |
| `FlyBrain/Sources/BrainEngine.swift` | `groupNames`, and `pump(count:)` — the connectome, headless |
| `FlyBrain/Sources/WorldView.swift` | the Body screen builds the cord from the loaded connectome, pumps it, and reports it |
| `FlyBrainTests/FlyCordTests.swift` *(new)* | the loop's shape, its balance, its silence, its organs |

## Measured on the built files

```
$ python3 tools/build_banc.py --raw data/banc --out build/flybanc.bin --min-syn 3
    organ:T1_left:chordotonal         38
    organ:T1_left:campaniform         93
    organ:T1_left:hairplate           53
    pool:T1_left:tibia_flexor         17
    pool:T1_left:tibia_extensor        2
$ python3 tools/verify_loop.py
  ok   six legs
  ok   three joints per leg, all of them hinges
  ok   every pool the asset places exists in the connectome
  ok   every organ the asset names exists in the connectome
  ok   no group the loop needs is empty
  ok   the connectome carries one group per pool, per leg and side
  ok   the connectome carries both organs for every leg
  ok   nothing the body could not place was dropped
  the loop: 6 legs · 42 pools placed · 0 pools the body could not place · 12 organs
  the connectome names 60 pools and 18 organs; the body asks for 42 and 12
```

## The bug this step found

The body asset named its pools `motor_front_leg_left_tibia_flexor`; the
connectome names them `pool:T1_left:tibia_flexor`. `FlyLiveBody.drive` would have
been handed a cord whose every `groupRate()` call returned 0 Hz — an animal that
looks like it is doing nothing, with nothing in the logs to say why. Both names
now come from one function, and `verify_loop.py` fails the build if they ever
disagree again. This is assumption #22.

## Still open in this step

1. The resistance reflex measured *through the app's engine* (pool rates, joint
   angle, delay) — step 3 did it in Python with its own cord; this needs the
   phone's engine, on a device or a long CI run. This is the acceptance test
   `verify_muscles.py` in docs/STEP6.md.
2. The real-time factor of the phone's cord, measured (the HUD reports it; no
   measurement has been taken on hardware).
3. Whether the loop walks. It does not yet, and that is the next measurement.

## The second bug, found while checking the .ipa

Run 49 (`4954e15`) was the first green build since step 5, and it produced an
`.ipa`. Opening it showed two things the CI gate did not:

```
  YES world.json      YES fly.bin      no frames.bin      YES flybanc.bin
  no  fly_body.json                    no fly_golden.json
```

**The app shipped without its skeleton.** `tools/pack_body.py` writes
`FlyBrain/World/fly_body.json`; `tools/pack_world.py` then deleted every file in
that folder that was not `world.json`/`fly.bin` — including that one, 4.3
seconds later, in every CI run:

```
14:39:23  -> FlyBrain/World/fly_body.json (0.23 MB)
14:39:28  removed stale fly_body.json
14:39:35  xcodegen / build
```

So the Body screen on the phone had no body asset to load, and — worse, because
it is silent — the entire solver suite **skipped itself** in CI. The log from run
49 says it plainly:

```
Executed 7 tests, with 3 tests skipped and 0 failures
  testGoldenTraceIsReproduced ................... skipped
  testTheAnimalStandsWithMuscleToneOnly ......... skipped
  testTheMuscleModelIsForceBasedAndBraked ....... skipped
```

The three tests that pin the Swift solver to the MuJoCo-verified Python one had
never run in CI. `XCTSkip` is the right default for a developer without the
41 MB model; in CI it is a green tick over an empty seat.

Three fixes, all of them gates rather than promises:

1. `pack_world.py` now deletes **only the recording it owns** and leaves other
   packers' files alone.
2. The `.ipa` gate demands `World/fly_body.json` and `World/fly_golden.json` by
   name, beside the connectome and the world.
3. `tools/check_tests.py` (new, run in CI on the `xcodebuild test` log) fails the
   build if any test is skipped, and if any test in `REQUIRED` did not run. It
   was checked against run 49's own log: it fails on it, and passes on a log
   where the suite ran.

## The tone now arrives where the reference sends it (delivery order #3)

The cord had a hole in it that no measurement had reached: the loop's tone was
being put on the **sense organs** and on nothing else. The organs are not the
brain. In `tools/step3_closedloop.py` the tonic drive is injected into
`net.desc_idx` — the descending neurons, the only population that carries the
brain's state into the leg circuit — on every millisecond of every mode
(`drivers = [(self.desc, self.args.desc, 0, ms)]`), and the organs get their
*own* drive on top of it. The app had the second half only, which is why the
animal on the phone stood on six disconnected legs with no brain input at all.

`FlyCord.update` now injects `settings.tone` into `settings.descendingGroup`
(`"descending"`, the connectome's own name) before anything else it does that
millisecond, in both phases, and `FlyCordTests`
`.testTheDescendingNeuronsCarryTheBrainsToneIntoTheCord` pins it: the first
millisecond, the last millisecond of calibration, and the fact that a
connectome without that group is reported in `missingGroups` instead of reading
as 0 Hz. A missing descending population is not a missing pool — the test says
so — but a cord that is standing on its organs alone has to say so out loud.

**Is it a pathway or a label?** That is a question about the artifact the app
carries, so it is answered there. `tools/verify_descending.py` (new, run in CI
beside `verify_loop.py`) reads `build/flybanc.bin` the way `Connectome.swift`
does and counts:

| | measured |
|---|---|
| descending cells | 1,316 |
| synapses out of them | 89,557 (68.1 per cell) |
| **into the 371 pool motor neurons, directly** | **3,283** |
| per leg, direct | 52–59 (T1_left 56, T1_right 59, T2_left 52, T2_right 55, T3_left 52, T3_right 54) |
| reachable within two hops | 367/371 pool motor neurons (98.9%) |
| per leg, reachable | T1/T2 62/62, T3 59/61 |

The tool fails the build if the group is absent, if `FlyCord.swift`'s name for
it has drifted from the binary's, if fewer than 1,000 direct descending →
pool synapses exist, or if any leg's pools fall below 90% two-hop
reachability. The name check is the one that would have caught this whole
section: the app and the binary have to agree about what the group is called.

**What is not claimed.** That the tone *does* anything to the stance through
this path on the phone. The synapse count says the cells are connected, not
that 2.5 threshold units of current on a standing fly holds it up. That is the
acceptance test of item 27, and it needs the device. Until it is measured, the
only thing this section asserts is the wiring — and that the wiring is now the
one the reference uses.
