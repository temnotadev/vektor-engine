#version 460
#extension GL_GOOGLE_include_directive : enable
#include "common.glsl"
layout(local_size_x = 8, local_size_y = 8) in;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Pass 7: edge-aware 5x5 filter on accumulated direct + indirect light,
// re-modulate by albedo, tonemap.
//
// Denoise strength (script: "denoise <0..4>", U.params3.z) scales the tap spacing of the 5x5 kernel (a-trous
// style, so a wide blur stays cheap). The spacing also grows where the temporal history is short (fresh
// disocclusions, camera cuts, moving objects) because that is where the remaining noise lives; fully
// accumulated pixels keep the tight kernel and stay sharp. Taps are rejected across normal / plane-distance
// edges and across emitters (their demodulated light is 0, which used to leak dark halos into the ceiling).
void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(px, U.misc0.xy))) return;
    ivec2 ip = ivec2(px);

    vec4 nd  = imageLoad(gND, ip);
    vec4 alb = imageLoad(gAlbedo, ip);
    vec3 color = vec3(0.0), lit = vec3(0.0);

    if (nd.w > 0.0 && alb.a > 0.5) {
        color = alb.rgb;                                       // looking straight at an emitter
    } else if (nd.w > 0.0) {
        vec3 n = nd.xyz;
        vec3 p = worldPos(px, nd.w);
        float strength = U.params3.z;
        vec3 direct = imageLoad(dirCur, ip).rgb, gi = imageLoad(giCur, ip).rgb;
        if (strength > 0.0) {
            float history = min(imageLoad(giCur, ip).a, imageLoad(dirCur, ip).a);   // frames accumulated so far
            float young = 1.0 - clamp(history / 12.0, 0.0, 1.0);                   // 1 = brand new, 0 = converged
            int stride = clamp(int(floor(strength * (1.0 + 2.0 * young) + 0.5)), 1, 6);
            direct = vec3(0.0); gi = vec3(0.0);
            float wSum = 0.0;
            for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                ivec2 q = ip + ivec2(dx, dy) * stride;
                if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, ivec2(U.misc0.xy)))) continue;
                vec4 nq = imageLoad(gND, q);
                if (nq.w <= 0.0) continue;
                if (imageLoad(gAlbedo, q).a > 0.5) continue;                       // emitter pixel: no demodulated light
                float planeDist = abs(dot(n, worldPos(uvec2(q), nq.w) - p));
                float w = pow(max(dot(n, nq.xyz), 0.0), 32.0)
                        * exp(-planeDist / (0.02 * nd.w + 1e-3))
                        * exp(-0.15 * float(dx * dx + dy * dy));                   // gentle gaussian falloff
                direct += imageLoad(dirCur, q).rgb * w;
                gi     += imageLoad(giCur,  q).rgb * w;
                wSum   += w;
            }
            if (wSum > 0.0) { direct /= wSum; gi /= wSum; }
            else { direct = imageLoad(dirCur, ip).rgb; gi = imageLoad(giCur, ip).rgb; }
        }

        lit = alb.rgb * (direct + gi);
        uint mode = U.misc1.y;                                 // 0 full, 1 direct, 2 indirect
        if (mode != 2u) color += alb.rgb * direct;
        if (mode != 1u) color += alb.rgb * gi;
    }
    imageStore(litCur, ip, vec4(lit, 1.0));                    // full HDR lit image, read next frame by screen traces

    color = aces(color * U.params.x);
    color = pow(color, vec3(1.0 / 2.2));
    imageStore(finalImg, ip, vec4(color, 1.0));
}
