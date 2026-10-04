//
//  World.metal
//  Instanced rendering of the fly's world. Everything in the scene — scenery,
//  food, the swatter, and the fly's own body segments — is one of three unit
//  primitives (cube, icosphere, quad) drawn with a per-instance model matrix
//  and colour, so the whole world is three draw calls.
//

#include <metal_stdlib>
using namespace metal;

struct WorldParams {
    float4x4 viewProj;
    float3   cameraPos;
    float    time;
    float3   sunDir;
    float    checker;
    float3   groundColour;
    float    fogDensity;
    float3   skyColour;
    float    pad;
};

struct Instance {
    float4x4 model;
    float4   colour;      // rgb + emissive amount
    float4   params;      // x: checker scale (0 = flat)
};

struct VSIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

struct VSOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float4 colour;
    float  checker;
};

vertex VSOut worldVertex(VSIn in [[stage_in]],
                         uint iid [[instance_id]],
                         const device Instance *instances [[buffer(1)]],
                         constant WorldParams &p [[buffer(2)]])
{
    Instance inst = instances[iid];
    float4 wp = inst.model * float4(in.position, 1.0);

    VSOut out;
    out.position = p.viewProj * wp;
    out.worldPos = wp.xyz;
    out.normal = normalize((inst.model * float4(in.normal, 0.0)).xyz);
    out.colour = inst.colour;
    out.checker = inst.params.x;
    return out;
}

fragment float4 worldFragment(VSOut in [[stage_in]],
                              constant WorldParams &p [[buffer(2)]])
{
    float3 n = normalize(in.normal);
    float3 base = in.colour.rgb;

    // A checkerboard on the ground gives the optic flow something to bite on.
    if (in.checker > 0.0) {
        float2 c = floor(in.worldPos.xz * in.checker);
        float f = fmod(c.x + c.y + 2.0, 2.0);
        base *= mix(0.74, 1.0, f);
    }

    float ndl = max(dot(n, normalize(p.sunDir)), 0.0);
    float3 ambient = mix(p.skyColour * 0.35, float3(0.25), 0.4);
    float3 lit = base * (ambient + ndl * 0.85);

    // Rim light so silhouettes read against the sky.
    float3 v = normalize(p.cameraPos - in.worldPos);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    lit += base * rim * 0.25;

    lit += base * in.colour.a;          // emissive

    // Fog PER FRAGMENT, from the interpolated world position. This used to
    // be per-vertex, which was fine while the floor was a 13 cm quad but
    // made the open-world floor invisible: all four of its vertices sit
    // ~20 m away, every vertex fogged to ~1.0, and the interpolation painted
    // the whole ground in sky colour — the "the map vanished" bug.
    float dist = length(in.worldPos - p.cameraPos);
    float fog = 1.0 - exp(-dist * p.fogDensity);
    lit = mix(lit, p.skyColour, saturate(fog));
    return float4(lit, 1.0);
}

// ---------------------------------------------------------------------------
// Sky: a full-screen triangle with a vertical gradient, drawn first.
// ---------------------------------------------------------------------------

struct SkyOut {
    float4 position [[position]];
    float2 uv;
};

vertex SkyOut skyVertex(uint vid [[vertex_id]])
{
    float2 pos[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
    SkyOut out;
    out.position = float4(pos[vid], 1.0, 1.0);
    out.uv = pos[vid] * 0.5 + 0.5;
    return out;
}

fragment float4 skyFragment(SkyOut in [[stage_in]],
                            constant WorldParams &p [[buffer(2)]])
{
    float t = saturate(in.uv.y);
    float3 c = mix(p.skyColour * 0.55, p.skyColour * 1.15, t);
    return float4(c, 1.0);
}

// ---------------------------------------------------------------------------
// Ommatidial resample: turn the 128x128 fly-eye render into the blurred,
// low-acuity image a compound eye actually delivers. ~700 ommatidia per eye
// means roughly 1.5 degrees between facets, so a sharp render is far too good.
// Box-filter and desaturate before the retina kernel reads it.
// ---------------------------------------------------------------------------

kernel void ommatidiaBlur(texture2d<float, access::read>  src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          constant uint &facetSize [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    uint f = max(facetSize, 1u);
    uint2 cell = (gid / f) * f;
    uint2 limit = uint2(src.get_width() - 1, src.get_height() - 1);
    float3 sum = float3(0.0);
    float  n = 0.0;
    for (uint y = 0; y < f; ++y) {
        for (uint x = 0; x < f; ++x) {
            sum += src.read(min(cell + uint2(x, y), limit)).rgb;
            n += 1.0;
        }
    }
    float3 avg = sum / max(n, 1.0);

    // Fly photoreceptors are green-dominant with a strong UV channel we do not
    // have; approximate with a green-weighted luminance.
    float lum = dot(avg, float3(0.18, 0.68, 0.14));
    dst.write(float4(lum, lum, lum, 1.0), gid);
}

// ---------------------------------------------------------------------------
// Blit: show the eye texture in the picture-in-picture, with a facet lattice
// over it so it reads as a compound eye rather than a webcam.
// ---------------------------------------------------------------------------

struct BlitOut {
    float4 position [[position]];
    float2 uv;
};

vertex BlitOut blitVertex(uint vid [[vertex_id]])
{
    float2 pos[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
    BlitOut out;
    out.position = float4(pos[vid], 0.0, 1.0);
    out.uv = float2(pos[vid].x * 0.5 + 0.5, 1.0 - (pos[vid].y * 0.5 + 0.5));
    return out;
}

fragment float4 blitFragment(BlitOut in [[stage_in]],
                             texture2d<float> src [[texture(0)]],
                             constant float &facets [[buffer(0)]])
{
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float lum = src.sample(s, in.uv).r;

    // Hexagonal facet lattice, drawn as a darkening at the cell borders.
    float2 g = in.uv * facets;
    g.x += fmod(floor(g.y), 2.0) * 0.5;
    float2 f = fract(g) - 0.5;
    float d = max(abs(f.x) * 1.15 + abs(f.y) * 0.6, abs(f.y) * 1.2);
    float border = smoothstep(0.42, 0.5, d);

    float3 c = float3(lum * 0.35, lum, lum * 0.45);   // the usual false green
    c *= (1.0 - border * 0.55);

    float2 r = in.uv - 0.5;
    c *= 1.0 - saturate(dot(r, r) * 1.4);             // edge of the visual field
    return float4(c, 1.0);
}

// ---------------------------------------------------------------------------
// Two-eye blit: the left compound eye in the left half of the preview, the
// right in the right half, each with its own facet lattice and field-edge
// falloff, divided by a dark seam. This is what the fly actually sees — two
// eyes, not one.
// ---------------------------------------------------------------------------

fragment float4 eyePairFragment(BlitOut in [[stage_in]],
                                texture2d<float> leftEye  [[texture(0)]],
                                texture2d<float> rightEye [[texture(1)]],
                                constant float &facets [[buffer(0)]])
{
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float side = step(0.5, in.uv.x);
    float2 uv = float2(fract(in.uv.x * 2.0), in.uv.y);
    float lum = mix(leftEye.sample(s, uv).r, rightEye.sample(s, uv).r, side);

    float2 g = uv * facets;
    g.x += fmod(floor(g.y), 2.0) * 0.5;
    float2 f = fract(g) - 0.5;
    float d = max(abs(f.x) * 1.15 + abs(f.y) * 0.6, abs(f.y) * 1.2);
    float border = smoothstep(0.42, 0.5, d);

    float3 c = float3(lum * 0.35, lum, lum * 0.45);
    c *= (1.0 - border * 0.55);

    float2 r = uv - 0.5;
    c *= 1.0 - saturate(dot(r, r) * 1.4);             // edge of each eye's field

    float seam = 1.0 - smoothstep(0.0, 0.015, abs(in.uv.x - 0.5));
    c *= 1.0 - seam * 0.85;                           // the divide between eyes
    return float4(c, 1.0);
}
