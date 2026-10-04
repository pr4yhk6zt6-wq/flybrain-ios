# FlyBrain iOS — the whole *Drosophila* connectome, running on an iPhone

A real-time simulation of the complete adult fruit-fly brain (FlyWire FAFB v783 —
**139,255 proofread neurons, 2,700,513 synaptic connections**) as a spiking
leaky-integrate-and-fire network on the iPhone GPU, rendered as a live 3-D point cloud,
driven by the phone's camera as the fly's eyes.

> **Status:** Step 1 of 7 complete — dataset studied, subset chosen, budgets proven.
> See [`docs/STEP1_DATASET.md`](docs/STEP1_DATASET.md).

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

- [x] **1. Study the FlyWire dataset, choose the subset** → `docs/STEP1_DATASET.md`
- [ ] 2. Data pipeline: connectome → packed binary / Metal buffers
- [ ] 3. Metal compute shader for the LIF simulation
- [ ] 4. 3-D renderer (instanced point cloud + spike arcs)
- [ ] 5. Camera → photoreceptor input
- [ ] 6. Test in Simulator, then on device
- [ ] 7. Build and sideload the `.ipa`

## Layout

```
tools/download_flywire.sh        fetch the 62 MB public Codex dumps, md5-verified
tools/step1_profile_flywire.py   census, neuropil breakdown, GPU budget calculator
docs/STEP1_DATASET.md            Step 1 findings and the decisions they force
reports/step1_report.json        machine-readable census
reports/step1_neuropils.csv      all 79 neuropils with synapse counts and system labels
data/raw/                        downloaded CSVs (gitignored)
```

## Reproducing Step 1

```bash
bash tools/download_flywire.sh
pip3 install numpy pandas pyarrow
python3 tools/step1_profile_flywire.py
```

## Three corrections to the original brief

1. **139,255 neurons, not 166,700.** The larger figure counts every segmented object
   including glia and unproofread fragments; 139,255 is the proofread neuron count and
   the number used in every published analysis of v783.
2. **~34 M synapses, not 125 M.** 125 M counts raw detected clefts volume-wide;
   34.2 M is the number between two proofread neurons at the recommended threshold.
3. **No VNC.** FAFB is a *brain* dataset — it contains no leg or wing motor neurons
   (only 110 brain motor neurons). The 1,305 descending neurons serve as the motor
   readout instead. A nerve cord would require the separate BANC v888 release.

## Data attribution

FlyWire FAFB v783 data © the FlyWire Consortium, released under
**CC BY-NC 4.0 — non-commercial use only**.

- Dorkenwald et al. (2024) *Neuronal wiring diagram of an adult brain*, Nature.
- Schlegel et al. (2024) *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature.
- https://flywire.ai · https://codex.flywire.ai

Conceptual reference: the FlyVision project by Ali Murat Bozaci.
