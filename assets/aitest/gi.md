#version 460
#extension GL_GOOGLE_include_directive : enable
#include "common.glsl"
layout(local_size_x = 8, local_size_y = 8) in;

// Next-event estimation at a secondary vertex (path mode): 8-candidate RIS over the world light grid
// (or the global power table when sampling mode is "global") + 1 shadow ray.
vec3 nee(vec3 p, vec3 n, vec3 albedo, inout uint s) {
    const uint K = 8u;
    float wsum = 0.0, selP = 0.0;
    uint  selL = NO_LIGHT;
    vec2  selU = vec2(0.0);
    for (uint i = 0u; i < K; ++i) {
        uint li;
        float P = pickLightWorld(p, s, li);
        if (P <= 0.0) continue;
        vec2 u = sampleU(s);
        LightEval e = evalLight(li, u, p, n);
        float ph = luminance(e.f);
        resUpdate(wsum, selL, selU, selP, li, u, ph, ph / (P * e.pdf), s);
    }
    if (selP <= 0.0) return vec3(0.0);
    LightEval e = evalLight(selL, selU, p, n);
    if (occluded(p + n * RAY_EPS, e.wi, e.dist - 2.0 * RAY_EPS)) return vec3(0.0);
    return albedo * (1.0 / PI) * e.f * (wsum / (float(K) * selP));
}

// Screen-space trace (mode 2): march the ray through the depth buffer and, on a hit, reuse LAST FRAME's lit
// image at that surface as radiance. Cheap and exact for what is on screen; misses fall back to RT + cache.
bool screenTrace(vec3 p, vec3 n, vec3 dir, inout uint s, out vec3 radiance) {
    radiance = vec3(0.0);
    if (U.misc0.z == 0u) return false;
    const int STEPS = 16;
    const float minStep = 0.05;
    float maxDist = U.params3.x;
    float growth = pow(maxDist / minStep, 1.0 / float(STEPS));
    float jitter = rnd(s);
    vec3 o = p + n * RAY_EPS;
    float RR = dot(U.camRight.xyz, U.camRight.xyz), UU = dot(U.camUp.xyz, U.camUp.xyz);
    for (int i = 0; i < STEPS; ++i) {
        float t = minStep * pow(growth, float(i) + jitter);
        vec3 v = o + dir * t - U.camPos.xyz;
        float z = dot(v, U.camFwd.xyz);
        if (z <= 1e-3) break;
        vec2 ndc = vec2(dot(v, U.camRight.xyz) / (z * RR), dot(v, U.camUp.xyz) / (z * UU));
        if (abs(ndc.x) >= 1.0 || abs(ndc.y) >= 1.0) break;               // left the screen -> RT
        ivec2 sp = ivec2((ndc * vec2(0.5, -0.5) + 0.5) * vec2(U.misc0.xy));
        vec4 nd = imageLoad(gND, sp);
        if (nd.w <= 0.0) continue;
        float rayDist = length(v);
        float thick = max(0.1, 0.08 * nd.w);
        if (rayDist > nd.w && rayDist < nd.w + thick && dot(nd.xyz, dir) < 0.0) {
            if (imageLoad(gAlbedo, sp).a > 0.5) continue;                // emitters are lights, not reflectors
            vec3 hp = worldPos(uvec2(sp), nd.w);
            ivec2 pp;
            if (!reproject(hp, pp) || !historyValid(pp, hp, nd.xyz)) return false;
            radiance = imageLoad(litPrev, pp).rgb;
            return true;
        }
    }
    return false;
}

// Pass 8: indirect light, three modes:
//   0 path   : cosine path, up to giBounces bounces, NEE at every vertex (light grid sampling)
//   1 cache  : ONE ray per pixel; radiance at the hit = albedo * surface cache (all bounces + all lights already inside)
//   2 screen : like 1, but the ray is first marched in screen space and reuses last frame's lit image
// giRate 2 traces one pixel per 2x2 quad per frame and carries the rest forward by reprojection.
// Output is demodulated (no primary albedo) and temporally accumulated.
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(px, U.misc0.xy))) return;
    ivec2 ip = ivec2(px);

    vec4 nd = imageLoad(gND, ip);
    if (nd.w <= 0.0) { imageStore(giCur, ip, vec4(0.0)); return; }

    vec3 n = nd.xyz;
    vec3 p = worldPos(px, nd.w);
    uint s = seedFor(px, U.misc0.z, 2u);

    ivec2 pp;
    bool hv = U.misc0.z > 0u && reproject(p, pp) && historyValid(pp, p, n);
    bool traceNow = U.misc3.y <= 1u || ((px.x & 1u) + 2u * (px.y & 1u)) == (U.misc0.z & 3u);
    if (!traceNow && hv) { imageStore(giCur, ip, imageLoad(giPrev, pp)); return; }

    vec3 Li = vec3(0.0);
    if (U.misc3.x == 0u) {
        vec3 throughput = vec3(1.0);
        vec3 pos = p, nrm = n;
        for (uint b = 0u; b < U.misc1.z; ++b) {
            vec3 dir = cosineSample(nrm, vec2(rnd(s), rnd(s)));
            Hit h;
            if (!traceClosest(pos + nrm * RAY_EPS, dir, 1e4, h)) break;
            Tri  t  = tris[h.prim];
            vec3 ng = triNormal(t);
            if (dot(ng, dir) > 0.0) ng = -ng;
            vec3 hp  = triPoint(t, h.bary);
            vec3 alb = triAlbedo(t, h.bary, 4.0);                // blurry mip: bounce light only needs the average colour
            Li += throughput * nee(hp, ng, alb, s);
            throughput *= alb;
            pos = hp; nrm = ng;
            if (luminance(throughput) < 1e-3) break;
        }
    } else {
        vec3 dir = cosineSample(n, vec2(rnd(s), rnd(s)));
        bool done = false;
        if (U.misc3.x == 2u) {
            vec3 rad;
            if (screenTrace(p, n, dir, s, rad)) { Li = rad; done = true; }
        }
        if (!done) {
            Hit h;
            if (traceClosest(p + n * RAY_EPS, dir, 1e4, h)) {
                Tri t = tris[h.prim];
                bool back = dot(triNormal(t), dir) > 0.0;
                Li = triAlbedo(t, h.bary, 4.0) * cacheLookup(h.prim, h.bary, back);
            }
        }
    }

    float l = luminance(Li);
    if (l > U.params.w) Li *= U.params.w / l;                 // firefly clamp

    vec4 outv = vec4(Li, 1.0);
    if (hv) {
        vec4  hist  = imageLoad(giPrev, pp);
        float count = min(hist.a + 1.0, U.params.y);
        outv = vec4(mix(hist.rgb, Li, 1.0 / count), count);
    }
    imageStore(giCur, ip, outv);
}
