# Step 3–5 — Metal engine, renderer, camera input

Status: **code complete and compiling.** Every commit builds on a real macOS
runner: connectome rebuilt from the public CSVs, validated, Xcode project
generated, simulator build, unsigned device build, `.ipa` packaged. Untested on
physical hardware — see §5.

## 3.1 Seven compute kernels (`FlyBrain/Shaders/LIF.metal`)

| Kernel | Dispatch | Job |
|---|---|---|
| `resetCounters` | 1 group | clear per-step spike and stat counters |
| `sampleRetina` | 10,647 | camera texture → photoreceptor current |
| `lifIntegrate` | 139,255 | one membrane step, emit a compacted spike list |
| `buildDispatchArgs` | 1 thread | spike count → indirect dispatch arguments |
| `spikeScatter` | indirect | scatter spikes along the CSR into the delay ring |
| `reduceStats` | 139,255 | threadgroup reduction for the HUD |
| `updateActivityTrace` | 139,255 | decaying afterglow for the renderer |

### Fixed-point atomics, not float

The synaptic accumulator is `atomic_int` in units of 1/65536, not `atomic_float`.
Float atomics require Metal 3 and an A14-class GPU; the deployment floor here is
**iPhone XS (A12)**, so integer atomics are the only portable option. Weights are
bounded by `|w| < 0.53` and fan-in is bounded, so int32 cannot realistically
overflow — the headroom is about four orders of magnitude.

### One threadgroup per firing neuron

Out-degree in FlyWire spans **0 to 6,399** with a mean of 19. A thread-per-neuron
scatter would leave a handful of threads doing 300× the work of the rest while the
entire dispatch waits for them. Instead each firing neuron gets a whole threadgroup
of 128 threads that strides across its edge list, which flattens the tail.

### No mid-frame CPU readback

`buildDispatchArgs` turns the spike count into
`MTLDispatchThreadgroupsIndirectArguments` on the GPU, so the scatter is dispatched
indirectly and the CPU never needs to know how many neurons fired. On iOS this
matters more than on a Mac: there is no cheap way to read back mid-frame, and one
stall would eat the whole 16 ms budget.

### The camera is a contrast detector, not a brightness meter

`sampleRetina` high-pass filters luminance against a slowly adapting mean
(τ ≈ 300 ms) before driving the photoreceptors. This is what real lamina monopolar
cells do — L1 and L2 are the ON and OFF contrast channels. Feeding raw luminance
would mean a brightly lit room is indistinguishable from a strobe.

## 3.2 Rendering (`FlyBrain/Shaders/Render.metal`)

All 139,255 neurons are **one draw call** of point primitives — no geometry, no
instance buffer, no per-neuron CPU work. The vertex shader reads the half3 position,
the membrane voltage and the activity trace, and produces a point size and colour;
the fragment shader carves a disc out of `point_coord` and adds a glow for spiking
cells. Blending is additive, so a densely firing region reads as a glowing mass
rather than a flat wall of dots.

**Colour.** Deep blue at rest → red approaching threshold → white when firing. A
second palette colours by brain region for the anatomy view.

**View modes are index ranges, not predicates.** Because Step 2 sorted neurons by
`(system, super_class, root_id)`, hiding the optic lobe is a bitmask test on one
byte, and the ranges are precomputed in the sidecar JSON.

**LOD.** Points below one pixel are faded rather than shrunk. Shrinking turns the
cloud into aliased noise as you pull back; fading preserves the density gradient and
cuts overdraw at the same time.

## 3.3 Zero-copy loading (`Sources/Connectome.swift`)

`flybrain.bin` is `mmap`ed and each 256-byte-aligned section is wrapped in an
`MTLBuffer`. Sections that also land on a page boundary use
`makeBuffer(bytesNoCopy:)` and are never copied at all; the rest are copied once at
load. Either way the 21.57 MiB arrives as file-backed pages the kernel can evict
under memory pressure, and load time is essentially the cost of one `mmap`.

The loader also exposes a neuron inspector: tap a point and a brute-force ray test
over 139 k positions (~0.4 ms, not worth a spatial index) returns the FlyWire root
ID, cell type, region, neurotransmitter, side and out-degree.

## 3.4 Matching the reference exactly

`SimulationEngine` carries the same constants the NumPy reference settled on —
`dt 1 ms · τ_m 20 · τ_syn 5 · V_th 1.0 · t_ref 2 · gain 6.0 · drive 1.5 · noise 0.015` —
with the decays precomputed on the CPU as `exp(-dt/τ)`. The timestep is adjustable
over the 0.1–5 ms range the brief asks for, and the decays and refractory period are
recomputed when it changes.

Steps per frame adapt to the device: if the frame rate holds above 55 fps the engine
simulates more milliseconds per frame, up to 16; below 40 fps it backs off. A slow
device therefore runs the brain in slow motion rather than dropping frames.

The delay ring is 24 slots × 139,255 × 4 B = **13.4 MB**, one more slot than the
19 ms maximum delay so a write for *t+19* never lands on the slot being read.

**Total GPU residency: ~36 MB** — connectome 21.6 + ring 13.4 + state ~1.7. The
brief's budget was 2 GB.

## 3.5 What CI proves, and what it does not

Green CI means: the connectome rebuilds byte-identically from the public source, 22
structural checks pass, 120 ms of simulation produces healthy dynamics, all seven
Metal kernels and both render pipelines compile for `iphoneos` and
`iphonesimulator`, the Swift builds clean, and a 10.88 MiB `.ipa` is produced.

It does **not** prove the app runs. Nobody has launched it on a phone. Specifically
unverified: achieved frame rate on any device, whether the activity trace reads well
visually, whether the retinotopic mapping produces a sensible response to a real
camera, memory behaviour under iOS pressure, and the gesture feel. Those are Step 6,
and they need hardware.

## 3.6 Three bugs CI caught

1. **XcodeGen 2.46 emits project format 77**, which only Xcode 16 can open — the
   macos-14 image (Xcode 15.4) refused the project outright. Pinned
   `objectVersion: 56`.
2. **`onTapGesture(count:coordinateSpace:perform:)` is iOS 17+.** The deployment
   target is iOS 15, so orbit and tap now share one `DragGesture(minimumDistance: 0)`
   and an interaction that ends having moved under 10 points counts as a tap.
3. **A missing `import simd`** made `length` resolve to `Duration.length`, producing
   the wonderfully unhelpful error *"binary operator '-' cannot be applied to
   operands of type 'SIMD3&lt;Float&gt;' and 'Duration'"*.
