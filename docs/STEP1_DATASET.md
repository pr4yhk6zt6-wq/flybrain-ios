# Step 1 — FlyWire dataset study & subset selection

Status: **complete**. Produced by `tools/step1_profile_flywire.py`; machine-readable
output in `reports/step1_report.json` and `reports/step1_neuropils.csv`.

## 1.1 Which release, and where it actually comes from

| | |
|---|---|
| Dataset | FlyWire **FAFB v783** (Dorkenwald et al., *Nature* 2024; Schlegel et al., *Nature* 2024) |
| Source | `https://storage.googleapis.com/flywire-data/codex/data/fafb/783/` — public GCS bucket, **no auth** |
| Licence | **CC BY-NC 4.0 — non-commercial only.** The app must carry attribution and cannot be sold. |
| Download size | 62 MB total (7 gzipped CSVs) |

`codex.flywire.ai` itself requires a Google sign-in, but the static CSV dumps the
portal serves are world-readable on the bucket above. That is what the pipeline uses,
so the build is reproducible with no credentials. Files and verified md5s:

| File | Size | md5 |
|---|---|---|
| `neurons.csv.gz` | 1.7 MB | `f60333e9e4124160b9b203b1712a6f91` |
| `connections.csv.gz` | 48 MB | `41206440318c77418bfc7cff1cb3e0fa` |
| `classification.csv.gz` | 913 KB | `701c0faf054ceddb460bea4e87d6a624` |
| `coordinates.csv.gz` | 5.1 MB | `f7a8120f8322405fecbeec2ee1d206e7` |
| `consolidated_cell_types.csv.gz` | 881 KB | `18cc628156bede3129c0e19d14cf52d6` |
| `labels.csv.gz` | 4.6 MB | `b3951998eeeda84bb4a2e209b456f683` |
| `names.csv.gz` | 1.2 MB | `6d134fe2712cb81d179d4b033a86fdc5` |

The 10.6 GB Zenodo record (`zenodo.org/records/10676866`) holds the raw per-synapse
table. **We do not need it** — the pair×neuropil edge list in `connections.csv.gz`
carries everything the simulator consumes.

## 1.2 Three findings that change the project plan

### Finding A — the brain is 139,255 neurons, not 166,700

The spec asks for 166,700 neurons. That figure is the *total segmented object* count
in early FlyWire reporting (it includes glia, trachea, and unproofread fragments).
The proofread **neuron** count in the v783 release is **139,255**, and that is the
number every published analysis uses. Our census:

```
neurons.csv.gz rows            139,255
classification.csv.gz rows     139,255
cell-type annotations          138,327  (99.3 % coverage)
3-D position coverage          139,255  (100 %)
neurons appearing in edges     134,181  (5,074 are isolated at the ≥5-synapse threshold)
```

So the app will say **"139,255 neurons — the complete proofread FlyWire v783 brain"**,
which is both honest and still the whole brain.

### Finding B — ~34 M synapses, not 125 M, and the memory budget is trivial

The "~125 M synapses" figure counts every automatically-detected synapse cleft in the
volume. The connectome that is actually *usable* — synapses between two proofread
neurons, at the authors' recommended ≥5-synapse threshold — is:

```
unique directed neuron→neuron pairs   2,700,513
synapses in those pairs              34,153,566
pair × neuropil rows (raw file)       3,869,878
```

Note `connections.csv.gz` **is already thresholded at ≥5 synapses** — our ≥1, ≥2 and ≥5
queries all return the identical 2,700,513 pairs. (≥10 would cut it to 1,066,822.)

This is the single most important result of Step 1. Priced out as Metal buffers
(CSR with `uint32` column indices, `half` weights, `uint8` delays):

| Buffer | Bytes |
|---|---|
| positions (half3) | 0.80 MB |
| LIF state (v, i_syn, refractory) | 1.59 MB |
| CSR row pointers | 0.53 MB |
| CSR column indices (uint32) | 10.30 MB |
| CSR weights (half) | 5.15 MB |
| CSR delays (uint8) | 2.58 MB |
| **Total GPU residency** | **≈ 21.0 MB** |

**The entire fly brain fits in 21 MB of GPU memory.** The spec's 2 GB budget, the
1–5 GB bundle worry, the iCloud streaming plan and the "crop to the optic lobe as a
fallback" contingency are all unnecessary. We ship the whole brain in the app bundle,
and the file will be roughly **12–15 MB compressed**. Region views become a *rendering
filter*, not a different dataset.

### Finding C — positions are seed points, and there are duplicates

`coordinates.csv.gz` has 238,909 rows for 139,255 neurons (median 1, max 177 per
neuron). These are supervoxel seed coordinates in FAFB14.1 raw voxel space
(4 × 4 × 40 nm), **not** soma centroids. Step 2 must deduplicate — take the centroid
of each neuron's points — or the neuron table silently fans out on join. Bounding box:

```
x  87,624 … 902,880      y  51,360 … 442,888      z  640 … 279,120   (voxels)
≈ 3.26 mm × 1.57 mm × 11.1 mm in nm → normalise per-axis before rendering
```

## 1.3 Biology inventory relevant to the app

**Super-classes** (139,255 total): optic 77,873 · central 32,381 · sensory 16,938 ·
visual_projection 7,684 · ascending 1,750 · descending 1,305 · sensory_ascending 612 ·
visual_centrifugal 522 · motor 110 · endocrine 80.

**Flow:** intrinsic 118,464 · afferent 19,300 · efferent 1,491. This maps directly
onto the spec's sensory / interneuron / motor grouping for the compute kernels.

**Neurotransmitters** — this is how edges get their sign, since FlyWire ships no
weights. Excitatory ACH 82,298; inhibitory GABA 16,017 and GLUT 19,605 (glutamate is
inhibitory in *Drosophila* via GluCl); modulatory SER 1,021, DA 584, OCT 72.
So **w = ±syn_count**, scaled, with modulatory types treated as weak excitatory.

**Camera input path.** Sensory visual = 11,426 cells. Photoreceptor types present:
R7 (1,338), R8 (1,357). R1–R6 are **not** individually reconstructed in FAFB — they
terminate in the lamina and the release models them through lamina monopolars instead:
L1 1,775 · L2 1,728 · L3 1,477. The camera will therefore drive **L1/L2/L3 + R7/R8
(~7,675 cells)** as the retina, mapped by their x/y position into a retinotopic grid.
Downstream T4/T5 motion detectors exist but only under subtyped names (T4a–d, T5a–d),
with Mi1 1,581 and Tm3 1,756 available as the classic motion-pathway relays.

**Motor output is thin.** Only 110 motor neurons, all `brain_motor_neuron`; there are
**no leg or wing motor neurons in FAFB** — that is the ventral nerve cord, which this
dataset does not cover. The spec's "legs, wings" and the VNC view mode cannot be built
from v783. Options: (a) drop the VNC view and use the 1,305 descending neurons as the
motor readout — they are literally the brain's output channel to the VNC; or (b) add
the BANC v888 release later, which does include the nerve cord. **Recommending (a)**
for v1, with descending-neuron population rate as the "motor" HUD readout.

## 1.4 Neuropils and view modes

79 neuropils, grouped into systems by dominant-neuropil assignment per neuron:

| System | Neurons | Synapses |
|---|---|---|
| optic lobe | 86,932 | 16,571,056 |
| central brain (other) | 13,187 | 7,135,404 |
| gnathic / SEZ | 8,177 | 4,434,893 |
| olfactory (AL, LH) | 5,420 | 1,917,335 |
| olfactory / lateral assoc. | 5,413 | 1,342,090 |
| central complex | 4,434 | 1,848,643 |
| antennal mechanosensory | 502 | 143,243 |
| other / unassigned | 15,190 | 760,902 |

62 % of the brain is optic lobe — which is exactly why a camera-driven fly brain is
the right demo. These eight groups become the app's view-mode filter, replacing the
spec's "whole / optic lobe / central brain / VNC" (VNC dropped, see above).

## 1.5 Decisions carried into Step 2

1. **Ship the whole brain.** 139,255 neurons / 2,700,513 edges / ~21 MB GPU. No
   subsetting, no streaming, no iCloud, no fallback crop.
2. **Keep the ≥5-synapse edge list as-is** — it is the authors' own recommendation and
   already applied upstream.
3. **Sign edges by presynaptic neurotransmitter**, weight by `log1p(syn_count)` scaled
   so median input drives a cell near threshold.
4. **Deduplicate coordinates to a centroid per neuron**; normalise to a unit cube for
   rendering; keep FAFB voxel coords in metadata for the neuron-tap inspector.
5. **Two CSR matrices** — forward (row = presynaptic, for spike scatter) is what the
   LIF kernel actually wants; build row-major by pre.
6. **Retina = L1/L2/L3 + R7/R8**, retinotopically ordered by position.
   **Motor readout = 1,305 descending neurons.** VNC view mode removed.
7. Delays: FlyWire has no timing data. Synthesise from Euclidean soma distance at
   ~0.5 m/s conduction, quantised to 1 ms steps in a `uint8` (0–255 ms) — physically
   motivated and free.

## 1.6 Reproducing

```bash
bash tools/download_flywire.sh          # 62 MB into data/raw/
python3 tools/step1_profile_flywire.py  # ~13 s, writes reports/
```

Peak RSS ~1.4 GB (the 3.87 M-row connection table dominates). Runs on any machine with
2 GB free.
