//
//  Render.metal
//  Instanced point-cloud rendering of 139,255 neurons, coloured by membrane
//  potential and spike afterglow.
//
//  There is no geometry: each neuron is a single point primitive whose size is
//  set in the vertex shader and whose disc is carved out in the fragment shader.
//  That keeps the whole brain at one draw call.
//

#include <metal_stdlib>
using namespace metal;

struct RenderParams {
    float4x4 viewProjection;
    float4x4 model;

    float    pointScale;        // base size in pixels at unit distance
    float    voltageFloor;      // V mapped to "cold"
    float    voltageCeiling;    // V mapped to "hot"
    float    glowStrength;

    uint     neuronCount;
    uint     systemMask;        // bit per system; 0 in the mask = hidden
    uint     renderMode;        // 0 all, 1 spiking only, 2 by system colour
    float    alphaScale;

    float    lodThreshold;      // below this screen size, fade instead of draw
    float    time;
    uint     highlightIndex;    // selected neuron, or 0xFFFFFFFF
    uint     pad;
};

struct PointOut {
    float4 position [[position]];
    float  pointSize [[point_size]];
    half4  colour;
    half   glow;
};

// FlyWire super-class palette, used in "colour by system" mode.
constant half3 kSystemColours[10] = {
    half3(0.25h, 0.60h, 1.00h),   // 0 optic lobe        blue
    half3(1.00h, 0.78h, 0.20h),   // 1 central complex   amber
    half3(0.35h, 0.90h, 0.55h),   // 2 olfactory         green
    half3(0.95h, 0.45h, 0.85h),   // 3 learning/memory   magenta
    half3(1.00h, 0.55h, 0.30h),   // 4 gnathic           orange
    half3(0.60h, 0.95h, 0.95h),   // 5 antennal mech.    cyan
    half3(0.55h, 0.80h, 0.40h),   // 6 olfactory lateral olive
    half3(0.70h, 0.70h, 0.85h),   // 7 central other     lavender
    half3(0.50h, 0.50h, 0.55h),   // 8 other             grey
    half3(0.35h, 0.35h, 0.40h),   // 9 unassigned        dark grey
};

// Resting -> firing ramp: deep blue at rest, white-hot at threshold.
inline half3 voltageColour(float t) {
    t = saturate(t);
    const half3 cold = half3(0.08h, 0.18h, 0.75h);
    const half3 mid  = half3(0.85h, 0.25h, 0.35h);
    const half3 hot  = half3(1.00h, 0.98h, 0.90h);
    return (t < 0.5f) ? mix(cold, mid, half(t * 2.0f))
                      : mix(mid,  hot, half((t - 0.5f) * 2.0f));
}

vertex PointOut neuronVertex(
    uint                        vid       [[vertex_id]],
    device const half          *positions [[buffer(0)]],   // half3 per neuron
    device const float         *voltage   [[buffer(1)]],
    device const half          *activity  [[buffer(2)]],
    device const uchar         *meta      [[buffer(3)]],   // 8 B stride
    constant     RenderParams  &P         [[buffer(4)]])
{
    PointOut out;

    const float3 p = float3(float(positions[vid * 3 + 0]),
                            float(positions[vid * 3 + 1]),
                            float(positions[vid * 3 + 2]));

    const float4 world = P.model * float4(p, 1.0f);
    const float4 clip  = P.viewProjection * world;
    out.position = clip;

    const uint system = uint(meta[vid * 8 + 0]);
    const float act   = float(activity[vid]);
    const float v     = voltage[vid];

    // Hidden by the view-mode filter, or not spiking in spikes-only mode:
    // collapse to a degenerate point rather than branching in the fragment stage.
    const bool systemVisible = (P.systemMask & (1u << system)) != 0u;
    const bool spikeVisible  = (P.renderMode != 1u) || (act > 0.05f);
    if (!systemVisible || !spikeVisible) {
        out.position  = float4(0.0f, 0.0f, -10.0f, 1.0f);
        out.pointSize = 0.0f;
        out.colour    = half4(0.0h);
        out.glow      = 0.0h;
        return out;
    }

    // Perspective-correct point size, with a LOD fade for points below a pixel.
    const float w    = max(clip.w, 1e-4f);
    float size       = P.pointScale / w;
    const float fade = saturate(size / max(P.lodThreshold, 1e-3f));
    size             = max(size, P.lodThreshold);
    out.pointSize    = clamp(size, 1.0f, 64.0f);

    const float t = saturate((v - P.voltageFloor) /
                             max(P.voltageCeiling - P.voltageFloor, 1e-4f));

    half3 base = (P.renderMode == 2u) ? kSystemColours[min(system, 9u)]
                                      : voltageColour(t);
    // Spiking neurons blow out towards white regardless of the palette.
    base = mix(base, half3(1.0h, 0.95h, 0.85h), half(act * P.glowStrength));

    half alpha = half(P.alphaScale * fade * (0.35f + 0.65f * act));

    if (vid == P.highlightIndex) {
        base  = half3(0.2h, 1.0h, 0.4h);
        alpha = 1.0h;
        out.pointSize = max(out.pointSize, 14.0f);
    }

    out.colour = half4(base, alpha);
    out.glow   = half(act);
    return out;
}

fragment half4 neuronFragment(PointOut in            [[stage_in]],
                              float2   pointCoord    [[point_coord]])
{
    // Round sprite with a soft edge; spiking cells get a wider halo.
    const float d = length(pointCoord - float2(0.5f));
    if (d > 0.5f) discard_fragment();

    const float core = 1.0f - smoothstep(0.0f, 0.5f, d);
    const float halo = exp(-d * 6.0f) * float(in.glow);

    half4 c = in.colour;
    c.a *= half(core + halo);
    c.rgb *= half(1.0f + halo * 1.5f);
    return c;
}

// ============================================================================
//  Spike arcs — bright lines flashing along edges that just carried a spike.
//
//  Drawing all 2.7 M edges is pointless (and 20 MB of vertex fetch per frame),
//  so the CPU/GPU picks a bounded sample of currently-active edges and this
//  shader draws those as lines with an animated travelling pulse.
// ============================================================================
struct ArcVertex {
    float4 position [[position]];
    half4  colour;
};

vertex ArcVertex spikeArcVertex(
    uint                        vid        [[vertex_id]],
    device const half          *positions  [[buffer(0)]],
    device const uint          *arcPairs   [[buffer(1)]],   // (pre, post) per arc
    device const half          *arcAge     [[buffer(2)]],   // 0..1 per arc
    constant     RenderParams  &P          [[buffer(3)]])
{
    const uint arc = vid / 2u;
    const uint end = vid % 2u;
    const uint n   = arcPairs[arc * 2u + end];

    const float3 p = float3(float(positions[n * 3 + 0]),
                            float(positions[n * 3 + 1]),
                            float(positions[n * 3 + 2]));

    ArcVertex out;
    out.position = P.viewProjection * P.model * float4(p, 1.0f);

    const float age = float(arcAge[arc]);
    const half3 c = mix(half3(1.0h, 0.85h, 0.35h), half3(0.3h, 0.6h, 1.0h), half(age));
    // Head of the arc is brighter than the tail.
    out.colour = half4(c, half((1.0f - age) * (end == 1u ? 0.9f : 0.35f)));
    return out;
}

fragment half4 spikeArcFragment(ArcVertex in [[stage_in]]) {
    return in.colour;
}
