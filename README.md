# FlyBrain iOS — the whole *Drosophila* connectome, running on an iPhone

A real-time simulation of the complete adult fruit-fly brain (FlyWire FAFB v783 —
**139,255 proofread neurons, 2,700,513 synaptic connections**) as a spiking
leaky-integrate-and-fire network on the iPhone GPU, rendered as a live 3-D point cloud,
driven by the phone's camera as the fly's eyes.

> **Status:** the app builds and an unsigned `.ipa` is produced by CI on every
> commit. Not yet run on physical hardware.
>
> [![Build FlyBrain](https://github.com/pr4yhk6zt6-wq/flybrain-ios/actions/workflows/ios.yml/badge.svg)](https://github.com/pr4yhk6zt6-wq/flybrain-ios/actions/workflows/ios.yml)

## The headline result from Step 1

The whole brain needs **≈ 21 MB of GPU memory**, not the 1–5 GB the project brief
feared. There is no need to crop to the optic lobe, stream from iCloud, or ship a
reduced fallback dataset — the entire connectome goes in the app bundle at roughly
12–15 MB compressed.

| | |
|---|---|
| Neurons | 139,255 (100 % have 3-D positions) |
| Directed connections | 2,700,513 (≥5 synapses, the authors' threshold) |
| Synapses represented | 34,153,566 |
| GPU residency | ~21 MB (CSR, `uint32` indices, `half` weights, `uint8` delays) |

## Roadmap

- [x] **1. Study the FlyWire dataset, choose the subset** → [`docs/STEP1_DATASET.md`](docs/STEP1_DATASET.md)
- [x] **2. Data pipeline: connectome → packed binary** → [`docs/STEP2_PIPELINE.md`](docs/STEP2_PIPELINE.md)
- [x] **3. Metal compute shaders for the LIF simulation** → [`docs/STEP3_ENGINE.md`](docs/STEP3_ENGINE.md)
- [x] **4. 3-D renderer** — one draw call, 139,255 points, additive glow, LOD fade
- [x] **5. Camera → photoreceptor input** — contrast-adapting retinotopic sampling
- [ ] 6. **Test on a physical device** — needs hardware; CI only proves it compiles
- [x] **7. Unsigned `.ipa`** — 10.88 MiB, built by CI on every commit

## Layout

```
FlyBrain/Shaders/LIF.metal       7 compute kernels: integrate, scatter, retina, stats
FlyBrain/Shaders/Render.metal    instanced point cloud + spike arcs
FlyBrain/Sources/                mmap loader, simulation engine, renderer, SwiftUI
project.yml                      XcodeGen spec (the .xcodeproj is generated)
.github/workflows/ios.yml        rebuild data, validate, build, package the .ipa
tools/download_flywire.sh        fetch the 62 MB public Codex dumps, md5-verified
tools/step1_profile_flywire.py   census, neuropil breakdown, GPU budget calculator
tools/build_connectome.py        CSVs -> flybrain.bin (CSR, half weights, uint8 delays)
tools/verify_and_simulate.py     22 structural checks + NumPy reference LIF simulation
docs/STEP1_DATASET.md            Step 1 findings and the decisions they force
docs/STEP2_PIPELINE.md           binary format, weight/delay model, gain calibration
reports/step1_report.json        machine-readable census
reports/step1_neuropils.csv      all 79 neuropils with synapse counts and system labels
reports/step2_simulation.json    validation results + 600 ms rate trace
build/flybrain.bin               21.57 MiB packed connectome (gitignored, reproducible)
data/raw/                        downloaded CSVs (gitignored)
```

## Reproducing Steps 1–2

```bash
pip3 install numpy pandas pyarrow
bash tools/download_flywire.sh           # 62 MB, md5-verified
python3 tools/step1_profile_flywire.py   # census and budgets
python3 tools/build_connectome.py        # -> build/flybrain.bin, 21.57 MiB
python3 tools/verify_and_simulate.py --ms 600 --gain 6.0 --drive 1.5
```

The last command prints 22/22 structural checks and a 600 ms spiking simulation of the
whole brain: **19 Hz population rate, stable, activity propagating from the retina
into the central complex through real FlyWire wiring.**

## Three corrections to the original brief

1. **139,255 neurons, not 166,700.** The larger figure counts every segmented object
   including glia and unproofread fragments; 139,255 is the proofread neuron count and
   the number used in every published analysis of v783.
2. **~34 M synapses, not 125 M.** 125 M counts raw detected clefts volume-wide;
   34.2 M is the number between two proofread neurons at the recommended threshold.
3. **No VNC.** FAFB is a *brain* dataset — it contains no leg or wing motor neurons
   (only 110 brain motor neurons). The 1,305 descending neurons serve as the motor
   readout instead. A nerve cord would require the separate BANC v888 release.

## Getting the app onto a phone

Every push to `main` builds a **10.88 MiB unsigned `.ipa`**. Download it from the
[Actions tab](https://github.com/pr4yhk6zt6-wq/flybrain-ios/actions) → the latest
green run → **Artifacts** → `FlyBrain-unsigned-ipa`. Unzip it; inside is
`FlyBrain-unsigned.ipa`.

It is unsigned on purpose: every sideloading route re-signs with your own identity
anyway.

### Sideloadly (Windows or macOS, any Apple ID)

1. Install [Sideloadly](https://sideloadly.io) and iTunes (Windows only).
2. Plug the iPhone in and trust the computer.
3. Drag `FlyBrain-unsigned.ipa` into Sideloadly, enter your Apple ID, press **Start**.
4. On the phone: **Settings → General → VPN & Device Management** → trust your
   developer certificate.

A free Apple ID gives a **7-day** signature; reinstall weekly. A paid developer
account gives a year.

### AltStore (over Wi-Fi, auto-refreshes)

1. Install [AltStore](https://altstore.io) plus AltServer on a computer.
2. AltServer → **Install AltStore** → pick the device.
3. In AltStore on the phone: **My Apps → + →** choose `FlyBrain-unsigned.ipa`.

AltStore re-signs in the background as long as AltServer is reachable, which avoids
the weekly manual reinstall.

### Xcode (best for development)

```bash
brew install xcodegen
pip3 install numpy pandas pyarrow

bash tools/download_flywire.sh
python3 tools/build_connectome.py
mkdir -p FlyBrain/Resources
cp build/flybrain.bin build/flybrain_meta.json FlyBrain/Resources/

xcodegen generate --spec project.yml
open FlyBrain.xcodeproj
```

Then set your team under **Signing & Capabilities** and press run. `FlyBrain.xcodeproj`
is generated, not committed — `project.yml` is the source of truth.

> The connectome is **not** in the repository. It is rebuilt from the public FlyWire
> CSVs in about ten seconds, and the build is byte-reproducible, so committing a
> 21 MiB artefact would be pointless.

## Device support

| Device | Chip | Expectation |
|---|---|---|
| iPhone 11 and newer | A13+ | the design target — 60 fps, 16 brain-ms per frame |
| iPhone XS / XR | A12 | the deployment floor — 30 fps, fewer steps per frame |
| iPhone X and older | A11 | below the floor; A11 lacks the Metal features used |
| Simulator | — | builds and launches, but the GPU path is slow and the camera is absent |

Memory is not a constraint on anything: the brain is 21.6 MB, the delay ring 13.4 MB,
state ~1.7 MB — **~36 MB of GPU residency** against a 2 GB budget.

The app adapts rather than stutters: if the frame rate holds above 55 fps it
simulates more milliseconds per frame (up to 16), and below 40 fps it backs off. A
slower phone runs the fly's brain in slow motion instead of dropping frames.

Simulation halts on backgrounding — a GPU command buffer submitted after suspension
gets the process killed for background GPU use.

## Data attribution

FlyWire FAFB v783 data © the FlyWire Consortium, released under
**CC BY-NC 4.0 — non-commercial use only**.

- Dorkenwald et al. (2024) *Neuronal wiring diagram of an adult brain*, Nature.
- Schlegel et al. (2024) *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature.
- https://flywire.ai · https://codex.flywire.ai

Conceptual reference: the FlyVision project by Ali Murat Bozaci.
