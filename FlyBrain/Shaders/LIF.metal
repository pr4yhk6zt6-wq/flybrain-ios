//
//  LIF.metal
//  Leaky integrate-and-fire simulation of the whole BANC v888 connectome —
//  brain *and* ventral nerve cord from one animal (175,237 neurons, 2,171,713
//  edges, 19 MiB packed as CSR).
//
//  Reproduces, exactly, the NumPy reference in tools/verify_banc.py:
//      tau_m 20 ms · tau_syn 5 ms · V_th 1.0 · V_reset 0 · t_ref 2 ms
//      gain 12.0 · retinal drive 1.5 · noise sigma 0.015 · dt 1 ms
//  Measured at that operating point (reports/banc_simulation.json): 17.3 Hz
//  population rate, 28.6 Hz in the leg motor neurons, 119.0 Hz in the wing
//  motor neurons, 79.7% of cells never firing — which is what a connectome
//  looks like when it is not being told what to do.
//
//  Synaptic current is accumulated in FIXED POINT (int32, 1/65536 units) rather
//  than float. Metal's float atomics need Metal 3 and an A14 or newer GPU; the
//  project targets iPhone XS (A12), so integer atomics it is. Weights are bounded
//  by |w| < 0.53 and fan-in is bounded, so int32 cannot realistically overflow.
//

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- parameters
struct SimParams {
    uint  neuronCount;
    uint  edgeCount;
    uint  ringSlots;        // MAX_DELAY + 1
    uint  step;             // current timestep index

    float decayMembrane;    // exp(-dt/tau_m),  precomputed on the CPU
    float decaySynaptic;    // exp(-dt/tau_syn)
    float vThreshold;
    float vReset;

    float vRest;
    float gain;             // global synaptic gain, the slider (1...20)
    float noiseSigma;
    float externalDrive;    // retinal input amplitude

    int   refractorySteps;
    uint  retinaCount;
    uint  flags;            // bit0 = use camera texture instead of flat drive
    float dtMillis;

    float contrastGain;     // LMC:photoreceptor gain ratio, 8-10x (Laughlin &
                            // Hardie 1978) — a derived constant, not a slider
    float adaptationTau;    // s, mean-luminance adaptation time constant
    float cameraBias;       // baseline drive when the scene is perfectly flat
    float pad1;
};

// Fixed-point scale for the atomic current accumulator.
constant float  kFixedScale    = 65536.0f;
constant float  kFixedScaleInv = 1.0f / 65536.0f;

// ------------------------------------------------------------------- RNG
// xorshift32 — one state word per neuron, advanced in place. Cheap, and good
// enough for membrane noise.
inline uint xorshift32(thread uint &s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

// Two uniforms -> one standard normal, Box-Muller. We only need one of the pair.
inline float gaussian(thread uint &s) {
    float u1 = max(float(xorshift32(s)) * 2.3283064e-10f, 1e-7f);
    float u2 = float(xorshift32(s)) * 2.3283064e-10f;
    return sqrt(-2.0f * log(u1)) * cos(6.2831853f * u2);
}

// ============================================================================
//  Kernel 1 — integrate every neuron for one timestep.
//
//  One thread per neuron. Reads this step's slot of the delay ring, folds it
//  into the decaying synaptic current, integrates the membrane, and emits a
//  spike index into a compacted list so the scatter kernel only touches cells
//  that actually fired.
// ============================================================================
kernel void lifIntegrate(
    device       float           *vMembrane      [[buffer(0)]],
    device       float           *iSynaptic      [[buffer(1)]],
    device       int             *refractory     [[buffer(2)]],
    device       atomic_int      *ring           [[buffer(3)]],   // ringSlots * N
    device       uint            *rngState       [[buffer(4)]],
    device const float           *externalInput  [[buffer(5)]],   // per neuron, from camera
    device       uint            *spikeList      [[buffer(6)]],
    device       atomic_uint     *spikeCount     [[buffer(7)]],
    device       uchar           *spikeFlags     [[buffer(8)]],   // for the renderer
    constant     SimParams       &P              [[buffer(9)]],
    uint                          gid            [[thread_position_in_grid]])
{
    if (gid >= P.neuronCount) return;

    const uint slot = P.step % P.ringSlots;
    const uint ringIdx = slot * P.neuronCount + gid;

    // Consume and clear this slot. relaxed is fine: the scatter kernel for this
    // step has already completed (separate command encoder, barrier between).
    int fixedArrivals = atomic_exchange_explicit(&ring[ringIdx], 0,
                                                 memory_order_relaxed);
    float arrivals = float(fixedArrivals) * kFixedScaleInv;

    float iSyn = iSynaptic[gid] * P.decaySynaptic + arrivals;
    iSynaptic[gid] = iSyn;

    uint rs = rngState[gid];
    float noise = gaussian(rs) * P.noiseSigma;
    rngState[gid] = rs;

    // External input is a current, added raw — one convention for every
    // population the body drives and for the photoreceptors, which is the
    // convention of the reference this kernel reproduces
    // (`tools/verify_banc.py`: `I = I_syn * gain + I_ext + noise`, and
    // `I_ext[vision] = args.drive`). Multiplying it by `externalDrive` here
    // instead made a group's tone arrive 1.5x stronger than the tone the Python
    // experiments were measured at (tools/pool_probe.py's "app" and "app x1.5"
    // rows are that difference). The retinal *amplitude* now scales the camera
    // path where it belongs — see sampleRetina.
    float I = iSyn * P.gain + externalInput[gid] + noise;

    int  refr   = refractory[gid];
    float v     = vMembrane[gid];
    bool  fired = false;

    if (refr > 0) {
        refractory[gid] = refr - 1;
        // Membrane is clamped at reset during the refractory period.
    } else {
        // Exact exponential-Euler step of  tau dV/dt = -(V - V_rest) + I
        v = P.vRest + (v - P.vRest) * P.decayMembrane + I * (1.0f - P.decayMembrane);
        if (v >= P.vThreshold) {
            v = P.vReset;
            refractory[gid] = P.refractorySteps;
            fired = true;
        }
        vMembrane[gid] = v;
    }

    spikeFlags[gid] = fired ? 1 : 0;

    if (fired) {
        uint slotIdx = atomic_fetch_add_explicit(spikeCount, 1u,
                                                 memory_order_relaxed);
        if (slotIdx < P.neuronCount) {
            spikeList[slotIdx] = gid;
        }
    }
}

// ============================================================================
//  Kernel 2 — scatter the spikes along their outgoing edges.
//
//  Dispatched as ONE THREADGROUP PER SPIKING NEURON (indirectly, so the CPU
//  never reads back the spike count). Out-degree ranges from 0 to 6,399 with a
//  mean of 19, so a per-thread-per-neuron mapping would leave a handful of
//  threads doing 300x the work of the rest. Giving each firing neuron a whole
//  threadgroup and striding across its edges flattens that.
// ============================================================================
kernel void spikeScatter(
    device const uint       *spikeList  [[buffer(0)]],
    device const uint       *spikeCount [[buffer(1)]],
    device const uint       *rowPtr     [[buffer(2)]],
    device const uint       *colIdx     [[buffer(3)]],
    device const half       *weights    [[buffer(4)]],
    device const uchar      *delays     [[buffer(5)]],
    device       atomic_int *ring       [[buffer(6)]],
    constant     SimParams  &P          [[buffer(7)]],
    uint                     groupID    [[threadgroup_position_in_grid]],
    uint                     tid        [[thread_position_in_threadgroup]],
    uint                     tgSize     [[threads_per_threadgroup]])
{
    if (groupID >= spikeCount[0]) return;

    const uint pre   = spikeList[groupID];
    const uint begin = rowPtr[pre];
    const uint end   = rowPtr[pre + 1];

    for (uint e = begin + tid; e < end; e += tgSize) {
        const uint  post  = colIdx[e];
        const float w     = float(weights[e]);
        const uint  d     = uint(delays[e]);                 // 1...19 ms
        const uint  slot  = (P.step + d) % P.ringSlots;
        const int   fixed = int(round(w * kFixedScale));

        atomic_fetch_add_explicit(&ring[slot * P.neuronCount + post],
                                  fixed, memory_order_relaxed);
    }
}

// ============================================================================
//  Kernel 3 — fill the indirect dispatch arguments for kernel 2.
//
//  Runs as a single thread. Turns the spike count into an
//  MTLDispatchThreadgroupsIndirectArguments so the scatter never needs a
//  CPU/GPU round trip. This matters more on iOS than on a Mac: there is no
//  unified-memory shortcut for a mid-frame readback, and a stall here would
//  cost the entire 16 ms budget.
// ============================================================================
kernel void buildDispatchArgs(
    device const uint *spikeCount [[buffer(0)]],
    device       uint *args       [[buffer(1)]],   // 3 x uint32
    uint               gid        [[thread_position_in_grid]])
{
    if (gid != 0) return;
    args[0] = max(spikeCount[0], 1u);   // never dispatch zero threadgroups
    args[1] = 1;
    args[2] = 1;
}

// ============================================================================
//  Kernel 4 — map the camera luminance texture onto the photoreceptors.
//
//  Each retina neuron carries a retinotopic UV computed in the build pipeline
//  (SVD of the per-hemisphere point cloud). Sampling the camera there turns a
//  frame into photoreceptor current directly, with no intermediate image.
//
//  Lamina monopolars L1/L2 are the ON/OFF contrast channels in the real fly,
//  so the drive is high-pass filtered against a slowly adapting mean rather
//  than fed raw luminance — a constant bright scene should NOT mean constant
//  maximal drive.
// ============================================================================
kernel void sampleRetina(
    texture2d<float, access::sample>  camera      [[texture(0)]],
    device const uint                *retinaIdx   [[buffer(0)]],
    device const half                *retinaUV    [[buffer(1)]],   // half2 per cell
    device       float               *externalIn  [[buffer(2)]],   // per neuron
    device       float               *adaptation  [[buffer(3)]],   // per retina cell
    constant     SimParams           &P           [[buffer(4)]],
    uint                              gid         [[thread_position_in_grid]])
{
    if (gid >= P.retinaCount) return;

    constexpr sampler s(filter::linear, address::clamp_to_edge);

    const float u = float(retinaUV[gid * 2 + 0]);
    const float v = float(retinaUV[gid * 2 + 1]);
    const float lum = camera.sample(s, float2(u, v)).r;

    // Mean-luminance adaptation with a real time constant instead of a
    // per-step rate, so the filter no longer changes with the simulation
    // step size. tau = 0.1 s: the rapid phase of photoreceptor light
    // adaptation in flies runs ~100 ms and the lamina re-centres within
    // ~200 ms (Laughlin & Hardie 1978, J Comp Physiol 132:139); sensitivity
    // modulation onsets over 200-300 ms (Curr Biol 2023, "Multifaceted
    // luminance gain control beyond photoreceptors in Drosophila"). This
    // high-pass stands in for both.
    const float alpha = 1.0f - exp(-(P.dtMillis * 0.001f) / P.adaptationTau);
    const float a = adaptation[gid];
    const float aNext = a + (lum - a) * alpha;
    adaptation[gid] = aNext;

    // Contrast, not absolute brightness — this is what lamina monopolars
    // encode. The gain is the measured LMC:photoreceptor slope ratio,
    // 8-10x (Laughlin & Hardie 1978).
    const float contrast = (lum - aNext) * P.contrastGain;
    // The clamp is the documented drive range of the contrast channel
    // (docs/ASSUMPTIONS.md #15: the bias is the midpoint of [0, 4]); the
    // amplitude slider scales the *retinal current* and nothing else, so the
    // camera path is unchanged by the fix above while a group drive is now the
    // current it says it is.
    externalIn[retinaIdx[gid]] = clamp(P.cameraBias + contrast, 0.0f, 4.0f)
                                 * P.externalDrive;
}

// ============================================================================
//  Kernel 5 — decay the per-neuron activity trace used for rendering.
//
//  A spike is one timestep long, but at 60 fps with a 1 ms timestep the renderer
//  only sees every ~16th state. Without a decaying trace most spikes would never
//  be drawn. This gives each neuron a visible afterglow.
// ============================================================================
kernel void updateActivityTrace(
    device const uchar     *spikeFlags [[buffer(0)]],
    device       half      *activity   [[buffer(1)]],
    constant     SimParams &P          [[buffer(2)]],
    uint                    gid        [[thread_position_in_grid]])
{
    if (gid >= P.neuronCount) return;
    float a = float(activity[gid]) * 0.92f;
    if (spikeFlags[gid] != 0) a = 1.0f;
    activity[gid] = half(a);
}

// ============================================================================
//  Kernel 6 — population statistics for the HUD, computed on the GPU.
//
//  Threadgroup reduction over spike flags, one atomic add per threadgroup
//  instead of one per neuron.
// ============================================================================
kernel void reduceStats(
    device const uchar      *spikeFlags [[buffer(0)]],
    device const uchar      *neuronMeta [[buffer(1)]],   // 8 B stride, byte 0 = system
    device       atomic_uint *stats     [[buffer(2)]],   // [0] total, [1..10] per system
    constant     SimParams  &P          [[buffer(3)]],
    threadgroup  uint       *scratch    [[threadgroup(0)]],
    uint                     gid        [[thread_position_in_grid]],
    uint                     tid        [[thread_position_in_threadgroup]],
    uint                     tgSize     [[threads_per_threadgroup]])
{
    uint fired = (gid < P.neuronCount && spikeFlags[gid] != 0) ? 1u : 0u;
    scratch[tid] = fired;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && scratch[0] > 0) {
        atomic_fetch_add_explicit(&stats[0], scratch[0], memory_order_relaxed);
    }

    // Per-system counters: rare enough (one in ~50 neurons fires) that a direct
    // atomic is cheaper than ten more reductions.
    if (fired != 0) {
        uint system = uint(neuronMeta[gid * 8 + 0]);
        if (system < 10) {
            atomic_fetch_add_explicit(&stats[1 + system], 1u, memory_order_relaxed);
        }
    }
}

// ============================================================================
//  Kernel 7 — reset the per-step counters.
// ============================================================================
kernel void resetCounters(
    device uint   *spikeCount  [[buffer(0)]],
    device uint   *stats       [[buffer(1)]],
    device uint   *groupCounts [[buffer(2)]],
    constant uint &groupCount  [[buffer(3)]],
    constant uint &zeroGroups  [[buffer(4)]],
    uint           gid         [[thread_position_in_grid]])
{
    if (gid == 0) spikeCount[0] = 0;
    if (gid < 11) stats[gid] = 0;
    // The group tallies are the one counter that accumulates across a whole
    // frame, so it is cleared once — by the GPU, in the same command buffer
    // where the steps below accumulate it. Clearing it from the CPU with
    // memset() happened at *encode* time, which is not the time the GPU reaches
    // these kernels: a device that keeps its queue a few frames deep is a
    // device where the clear and the accumulation can swap places, and then a
    // frame's tallies are thrown away before anything reads them.
    if (zeroGroups != 0 && gid < groupCount) groupCounts[gid] = 0;
}

// ===========================================================================
// Named body groups.
//
// World mode needs two things every frame: the firing rate of each motor
// group (to drive wings and legs), and a way to push sensory drive into a
// named population (smell, touch, pain). Doing either on the CPU would mean
// reading 175k spike flags back across the bus every frame.
//
// Instead both run on the GPU against the concatenated `groupIndices` array
// from the connectome, and only ~35 uints ever cross back.
// ===========================================================================

struct GroupRange {
    uint start;
    uint count;
};

/// One thread per neuron in the union of all groups. Counts how many members
/// of each group spiked this step.
kernel void reduceGroupSpikes(
    device const uint        *groupIndices [[buffer(0)]],
    device const GroupRange  *ranges       [[buffer(1)]],
    device const uchar       *spikeFlags   [[buffer(2)]],
    device       atomic_uint *counts       [[buffer(3)]],
    constant     uint        &totalIndices [[buffer(4)]],
    constant     uint        &groupCount   [[buffer(5)]],
    uint                      gid          [[thread_position_in_grid]])
{
    if (gid >= totalIndices) return;
    if (spikeFlags[groupIndices[gid]] == 0) return;

    // Which group does this slot belong to? The ranges are sorted and
    // contiguous, so a binary search over ~35 entries is 6 steps.
    uint lo = 0, hi = groupCount;
    while (lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if (ranges[mid].start <= gid) lo = mid; else hi = mid;
    }
    if (gid >= ranges[lo].start + ranges[lo].count) return;

    atomic_fetch_add_explicit(&counts[lo], 1u, memory_order_relaxed);
}

/// Writes a per-group scalar into the external input array. A negative drive
/// means "leave this group alone" — used for vision, which the retina kernel
/// owns, and for groups the body is not currently driving.
kernel void driveGroups(
    device const uint       *groupIndices  [[buffer(0)]],
    device const GroupRange *ranges        [[buffer(1)]],
    device const float      *drives        [[buffer(2)]],
    device       float      *externalInput [[buffer(3)]],
    constant     uint       &totalIndices  [[buffer(4)]],
    constant     uint       &groupCount    [[buffer(5)]],
    uint                     gid           [[thread_position_in_grid]])
{
    if (gid >= totalIndices) return;

    uint lo = 0, hi = groupCount;
    while (lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if (ranges[mid].start <= gid) lo = mid; else hi = mid;
    }
    if (gid >= ranges[lo].start + ranges[lo].count) return;

    float d = drives[lo];
    if (d < 0.0f) return;
    externalInput[groupIndices[gid]] = d;
}
