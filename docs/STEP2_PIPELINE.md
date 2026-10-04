# Step 2 — Connectome → packed binary → validated LIF dynamics

Status: **complete**. `tools/build_connectome.py` turns the raw Codex CSVs into a
single 21.57 MiB file; `tools/verify_and_simulate.py` reads it back the way the Swift
loader will, runs 22 structural checks, and simulates the network in NumPy to pin the
parameters down before any Metal is written.

```bash
bash  tools/download_flywire.sh          # 62 MB
python3 tools/build_connectome.py        # ~10 s  -> build/flybrain.bin
python3 tools/verify_and_simulate.py --ms 600 --gain 6.0 --drive 1.5
```

## 2.1 The file format

`flybrain.bin` is little-endian, with a 256-byte header and ten sections, each
**256-byte aligned** so iOS can hand the section straight to
`device.makeBuffer(bytesNoCopy:…)` — no parsing, no copies, no load-time allocation.

```
 off         section        size        contents
 256         rootIDs        1.06 MiB    uint64 x 139,255   FlyWire ids, for the inspector
 1,114,368   positions      0.80 MiB    half3  x 139,255   centred unit cube
 1,949,952   neuronMeta     1.06 MiB    8 B    x 139,255   system/superclass/NT/flags/typeId
 3,064,064   csrRowPtr      0.53 MiB    uint32 x 139,256
 3,621,120   csrColIdx     10.30 MiB    uint32 x 2,700,513
 14,423,296  csrWeight      5.15 MiB    half   x 2,700,513  signed
 19,824,384  csrDelay       2.58 MiB    uint8  x 2,700,513  ms
 22,524,928  retinaIdx      0.04 MiB    uint32 x 10,647
 22,567,680  retinaUV       0.04 MiB    half2  x 10,647     retinotopic, [0,1]^2
 22,610,432  motorIdx       0.01 MiB    uint32 x 1,415
 ---------------------------------------------------------
 total       21.57 MiB   (≈13 MB in the compressed .ipa)
```

Predicted in Step 1: 21.0 MB. Actual: 21.57 MiB. The whole fly brain, in the bundle.

## 2.2 Five modelling decisions, and why

**CSR is row-major by _presynaptic_ neuron.** The LIF kernel's hot loop is spike
*scatter*: when neuron *i* fires, add its weights to its targets. That wants all of
*i*'s outgoing edges contiguous. Max out-degree is 6,399, mean 19.4.

**Neurons are sorted by (system, super_class, root_id).** Every view mode is now a
contiguous index range, so the renderer filters with a `start`/`count` pair instead of
a per-neuron predicate. Verified by the `system ids contiguous` check. Ranges live in
`build/flybrain_meta.json`.

**Weights are signed by the _presynaptic_ neurotransmitter.** FlyWire publishes no
synaptic weights — only synapse counts, which are unsigned. The standard move is to
take the predicted NT of the sending cell: ACh excitatory, GABA inhibitory, and
**glutamate inhibitory** (in *Drosophila* glutamate gates GluCl chloride channels —
getting this backwards is the classic way to make a fly model explode). DA/SER/OCT are
modulatory and get a weak +0.3. Result: **64.5 % excitatory edges**, which is right in
the biological band.

Magnitude is `log1p(syn_count)`, which compresses the 1 … 93,000 synapse-count range
into something a membrane can integrate, then normalised so the *median* total
excitatory drive onto a neuron is 1.0. That leaves the app with a single global gain
knob rather than per-edge magic numbers.

**Delays are synthesised from distance.** FlyWire has no timing data at all. Euclidean
soma-to-soma distance at 0.5 m/s conduction gives min 1 ms, median 3 ms, max 19 ms —
physically motivated, fits in a `uint8`, and costs nothing.

**Positions are converted to nanometres first.** FAFB voxels are **4 × 4 × 40 nm** —
anisotropic. Normalising the raw voxel coordinates would render the brain stretched
10× along z. True extent is 3.25 × 1.57 × 11.14 mm… which is itself suspicious: the
z-axis is the EM sectioning axis and its range reflects the imaged volume, not the
brain. Scaling is isotropic from the longest axis, so proportions are preserved.

## 2.3 Validation — 22/22

```
PASS rootIDs unique            PASS rowPtr monotonic          PASS delay >= 1 (1..19)
PASS positions finite          PASS rowPtr terminates at E    PASS retinaUV in [0,1]
PASS positions in unit cube    PASS colIdx in range           PASS system ids contiguous
PASS weight nonzero, finite    PASS E/I balance 64.5% exc.    ... 22 total
```

Zero edges referenced a non-neuron segment, so no data is silently dropped.

## 2.4 The dynamics, and the gain cliff

The NumPy reference implements exactly what the Metal kernel will:

```
tau_m dV/dt = -(V - V_rest) + R·I      tau_m 20 ms, V_th 1.0, V_reset 0, t_ref 2 ms
I_syn decays with tau_syn = 5 ms
synaptic input lands delay[e] ms after the presynaptic spike  (ring buffer)
```

Sweeping the global gain with a steady retinal drive exposes a sharp percolation
threshold between gain 4 and 6:

| gain | drive | mean rate | never fired | optic | central cx | central other | gnathic | verdict |
|---|---|---|---|---|---|---|---|---|
| 2 | 1.2 | 2.1 Hz | 92 % | 2.9 | 0.0 | 0.0 | 0.0 | activity trapped in the optic lobe |
| 4 | 1.2 | 2.8 Hz | 91 % | 4.1 | 0.0 | 0.0 | 0.0 | still trapped |
| **6** | **1.5** | **19.0 Hz** | **72 %** | **12.5** | **40.6** | **14.1** | **13.6** | **← chosen** |
| 8 | 1.5 | 24.1 Hz | 67 % | 17.7 | 45.6 | 18.6 | 20.0 | healthy |
| 10 | 2.0 | 33.0 Hz | 62 % | 25.2 | 57.0 | 26.2 | 28.2 | hot |
| 14 | 2.0 | 43.6 Hz | 57 % | 34.1 | 70.9 | 36.5 | 38.0 | runaway |
| 20 | 2.5 | 56.5 Hz | 53 % | 46.5 | 81.0 | 49.0 | 49.1 | runaway |

Below gain ≈ 5 the retina fires but nothing escapes the optic lobe. Above ≈ 12 the
recurrent loops self-ignite. **Default: gain 6.0, retinal drive 1.5**, exposed as a
slider (1–20) so the user can watch the brain cross its own percolation threshold —
which is a far better demo than a fixed number.

At the chosen point the network is stable over 600 ms (19 → 20.7 Hz, no drift), 28 %
of neurons participate, and activity reaches the central complex, SEZ and central
brain through real FlyWire wiring. Mushroom body and lateral-olfactory stay silent at
0 Hz — correct, since the only input is visual and those regions are olfactory.

## 2.5 Camera input and motor readout

**Retina = 10,647 cells** (L1–L5 lamina monopolars + R7/R8 inner photoreceptors):
5,486 left, 5,161 right. R1–R6 are not individually reconstructed in FAFB.

Retinotopic UV is derived per hemisphere by SVD on the cell positions — the optic lobe
is a curved sheet, so its two principal axes *are* the screen axes. The right eye's u
is mirrored so both eyes index the same camera image. This is what lets a camera frame
be sampled directly into photoreceptor currents in Step 5.

**Motor readout = 1,415 cells** (1,305 descending + 110 brain motor), per the Step 1
decision to drop the VNC.

## 2.6 Carried into Step 3 (Metal)

The shader must reproduce the table in §2.4. Concretely:
`dt 1 ms · tau_m 20 · tau_syn 5 · V_th 1.0 · V_reset 0 · t_ref 2 · gain 6.0 · noise 0.015`,
decays precomputed on the CPU as `exp(-dt/tau)`, a `MAX_DELAY = 20` ring buffer of
`half` currents (139,255 × 20 × 2 B = **5.6 MB**), and atomic adds for the spike
scatter. Total GPU residency including the ring and double-buffered state stays under
**30 MB**.
