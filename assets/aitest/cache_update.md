#version 460
#extension GL_GOOGLE_include_directive : enable
#include "common.glsl"
layout(local_size_x = 64) in;

// Surface-cache update (the Lumen "surface cache + radiosity" idea):
// a rotating slice of cache texels is relit every frame.
//   direct   : ReSTIR DI in cache space - candidates come from the world light grid, the reservoir of the
//              previous update of the SAME texel is the temporal history (no reprojection needed).
//   indirect : cosine rays that read the cache at the hit point -> multi-bounce for free (radiosity feedback).
// Result is irradiance/pi, blended over time. Pixels later look this up instead of doing light sampling.
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= U.misc4.x) return;
    uint entries = U.misc4.z;
    if (entries == 0u) return;
    uint e = (U.misc4.y + g) % entries;
    uint side = e & 1u;
    uint texel = e >> 1u;

    // texel -> triangle: largest triangle whose first texel is <= texel
    uint lo = 0u, hi = U.misc4.w - 1u;
    while (lo < hi) {
        uint mid = (lo + hi + 1u) >> 1u;
        if (cacheOffsets[mid] <= texel) lo = mid; else hi = mid - 1u;
    }
    uint tri = lo;
    Tri t = tris[tri];
    uint k = t.cache.y;
    if (k == 0u) return;

    // texel -> grid cell (i,j) in the triangular layout, cell centre -> barycentrics
    uint local = texel - cacheOffsets[tri];
    uint j = 0u;
    while (local >= k - j) { local -= k - j; ++j; }
    uint i = local;
    vec2 b = (vec2(float(i), float(j)) + 0.5) / float(k);

    vec3 ng = triNormal(t);
    vec3 n = side == 0u ? ng : -ng;
    vec3 p = triPoint(t, b);
    uint s = seedFor(uvec2(e & 0xFFFFu, e >> 16u), U.misc0.z, 9u);

    // ---- direct light: ReSTIR DI per texel
    const uint MC = 4u;
    float wsum = 0.0, selP = 0.0, M = float(MC);
    uint  selL = NO_LIGHT;
    vec2  selU = vec2(0.0);
    for (uint c = 0u; c < MC; ++c) {
        uint li;
        float P = pickLightWorld(p, s, li);
        if (P <= 0.0) continue;
        vec2 u = sampleU(s);
        LightEval ev = evalLight(li, u, p, n);
        float ph = luminance(ev.f);
        resUpdate(wsum, selL, selU, selP, li, u, ph, ph / (P * ev.pdf), s);
    }
    Res prev = unpackRes(cacheRes[e]);
    if (prev.light != NO_LIGHT && prev.light < U.misc1.x && prev.M > 0.0) {
        prev.M = min(prev.M, 20.0 * float(MC));
        float ph = targetPdf(p, n, prev.light, prev.u);
        resUpdate(wsum, selL, selU, selP, prev.light, prev.u, ph, ph * prev.W * prev.M, s);
        M += prev.M;
    }
    Res cur;
    cur.light = selL; cur.u = selU; cur.M = M;
    cur.W = selP > 0.0 ? wsum / (M * selP) : 0.0;

    vec3 Ed = vec3(0.0);
    if (cur.W > 0.0) {
        LightEval ev = evalLight(cur.light, cur.u, p, n);
        if (occluded(p + n * RAY_EPS, ev.wi, ev.dist - 2.0 * RAY_EPS)) cur.W = 0.0;
        else Ed = ev.f * cur.W * (1.0 / PI);
    }
    cacheRes[e] = packRes(cur);

    // ---- indirect light: rays read the cache at their hit points
    vec3 Ei = vec3(0.0);
    uint R = max(U.misc3.z, 1u);
    for (uint r = 0u; r < R; ++r) {
        vec3 dir = cosineSample(n, vec2(rnd(s), rnd(s)));
        Hit h;
        if (traceClosest(p + n * RAY_EPS, dir, 1e4, h)) {
            Tri t2 = tris[h.prim];
            bool back = dot(triNormal(t2), dir) > 0.0;
            Ei += triAlbedo(t2, h.bary, 4.0) * cacheLookup(h.prim, h.bary, back);   // emitters have albedo 0: their light is the direct term
        }
    }
    Ei /= float(R);

    vec3 sampleE = Ed + Ei;
    float l = luminance(sampleE);
    if (l > 20.0) sampleE *= 20.0 / l;                         // firefly clamp
    vec4 old = cacheIrr[e];
    float count = min(old.a + 1.0, U.params3.y);
    cacheIrr[e] = vec4(mix(old.rgb, sampleE, 1.0 / count), count);
}
