// common.glsl - shared by every pass.
// Requires: #version 460 and GL_GOOGLE_include_directive in the including file.
#extension GL_EXT_ray_query : require

const float PI       = 3.14159265359;
const uint  NO_LIGHT = 0xFFFFFFFFu;
const float RAY_EPS  = 2e-3;   // scene is ~5 units wide

// ---------------------------------------------------------------- resources
layout(set = 0, binding = 0) uniform accelerationStructureEXT gTLAS;

layout(set = 0, binding = 1, std140) uniform FrameUBO {
    vec4  camPos, camFwd, camRight, camUp;          // right/up are pre-scaled by tan(fov/2)(*aspect)
    vec4  prevPos, prevFwd, prevRight, prevUp;
    uvec4 misc0;   // x=width y=height z=frame w=flags (bit0 temporal, bit1 spatial)
    uvec4 misc1;   // x=numLights y=debugMode z=giBounces w=diCandidates
    vec4  params;  // x=exposure y=giMaxHistory z=spatialRadius w=fireflyClamp
    uvec4 misc2;   // x=tilesX y=tileMode(0 global,1 per-tile lists) z,w unused
    vec4  params2; // x=1/totalLightPower y=tileCutoff z=directMaxHistory w=pixelSpread
    vec4  gridMin;     // world light grid: min corner
    vec4  gridInvCell; // world light grid: 1 / cell size per axis
    uvec4 gridDim;     // world light grid: cells per axis
    uvec4 misc3;   // x=giMode (0 path, 1 cache, 2 cache+screen) y=giRate z=cache radiosity rays w=unused
    uvec4 misc4;   // x=cacheBudget y=cacheCursor z=cacheEntries (texels*2 sides) w=numTris
    vec4  params3; // x=screenTraceDist y=cacheMaxHistory z=denoiseStrength (0 = off)
} U;

struct Tri {
    vec4 p0; vec4 p1; vec4 p2; vec4 albedo; vec4 emission;
    vec4 uv01;   // uv0.xy, uv1.xy
    vec4 uv2m;   // uv2.xy, texture layer (<0 = none), uv-per-world-unit footprint
    uvec4 cache; // x = first surface-cache texel, y = grid size k (0 = not cached)
};
layout(set = 0, binding = 2, std430) readonly buffer TriBuf   { Tri  tris[];     };

const uint LIGHT_TRI = 0u, LIGHT_POINT = 1u, LIGHT_SPOT = 2u, LIGHT_DIR = 3u;
const uint TILE_SIZE = 16u;
const uint TILE_MAX  = 256u;     // max lights kept per 16x16 tile (must match tileBytes in main.cpp)

// 64 bytes. TRI: pos=centroid, dir=normal, radius=sqrt(area/pi), tri=index, emission read from the triangle.
// POINT/SPOT: sphere light (soft shadows), color = intensity (W/sr). SPOT: dir = travel direction.
// DIR: dir = travel direction, color = illuminance, cosOuter = cos(angular radius).
struct Light {
    vec3 pos; float radius;
    vec3 dir; uint  type;
    vec3 color; float range;
    float cosOuter; float cosInner; uint tri; float power;   // power: max intensity-like value for sampling/bounds
};
layout(set = 0, binding = 3, std430) readonly buffer LightBuf { Light lights[]; };

layout(set = 0, binding = 4,  rgba32f)  uniform image2D  gND;        // normal.xyz, hit distance (<=0 = miss)
layout(set = 0, binding = 5,  rgba32f)  uniform image2D  gNDPrev;
layout(set = 0, binding = 6,  rgba16f)  uniform image2D  gAlbedo;    // albedo (a=0) or emission (a=1)
layout(set = 0, binding = 7,  rgba32ui) uniform uimage2D resInit;    // ReSTIR DI: initial candidates
layout(set = 0, binding = 8,  rgba32ui) uniform uimage2D resTCur;    //           temporal output (history)
layout(set = 0, binding = 9,  rgba32ui) uniform uimage2D resTPrev;
layout(set = 0, binding = 10, rgba32ui) uniform uimage2D resSpatial; //           spatial output (shaded)
layout(set = 0, binding = 11, rgba16f)  uniform image2D  giCur;      // demodulated indirect radiance, a = history count
layout(set = 0, binding = 12, rgba16f)  uniform image2D  giPrev;
layout(set = 0, binding = 13, rgba8)    uniform image2D  finalImg;
layout(set = 0, binding = 18) uniform sampler2DArray gTex;   // script textures: 512x512 sRGB layers + mips
layout(set = 0, binding = 16, rgba16f)  uniform image2D  dirCur;     // demodulated direct light, a = history count
layout(set = 0, binding = 17, rgba16f)  uniform image2D  dirPrev;

layout(set = 0, binding = 23, rgba16f)  uniform image2D  litCur;     // lit HDR radiance (direct + GI), reused by screen traces
layout(set = 0, binding = 24, rgba16f)  uniform image2D  litPrev;

// Surface cache (Lumen-style): per-triangle texel grid storing irradiance/pi for both sides of every surface.
layout(set = 0, binding = 19, std430) buffer CacheIrrBuf { vec4  cacheIrr[]; };   // rgb = irradiance/pi, a = history count
layout(set = 0, binding = 20, std430) buffer CacheResBuf { uvec4 cacheRes[]; };   // ReSTIR DI reservoir per texel
layout(set = 0, binding = 21, std430) readonly buffer CacheOffBuf { uint cacheOffsets[]; };

const uint CELL_MAX = 64u;
struct Cell { uint count; float totalW; uint pad0; uint pad1; uint idx[CELL_MAX]; float cdf[CELL_MAX]; };
layout(set = 0, binding = 22, std430) buffer CellBuf { Cell cells[]; };           // world-space light grid lists

struct AliasEntry { float prob; uint alias; };
layout(set = 0, binding = 14, std430) readonly buffer AliasBuf { AliasEntry aliasTab[]; };

struct Tile { uint count; float totalW; uint pad0; uint pad1; uint idx[TILE_MAX]; float cdf[TILE_MAX]; };
layout(set = 0, binding = 15, std430) buffer TileBuf { Tile tiles[]; };

// Light BVH (RTXDI/Moana-style light cluster tree): binary tree over all lights, built CPU-side in
// world.hpp::buildLightBVH() whenever the light set changes. rightOrLeaf == NO_LIGHT marks a leaf
// (leftOrLight is then the light index); otherwise left/right are child node indices. O(log N) stochastic
// descent (pickLightBVH below) replaces the brute-force per-tile/per-cell scan for scenes with many lights.
struct BVHNode { vec3 bmin; float power; vec3 bmax; uint leftOrLight; uint rightOrLeaf; uint pad0, pad1, pad2; };
layout(set = 0, binding = 25, std430) readonly buffer BVHBuf { BVHNode bvhNodes[]; };

// ---------------------------------------------------------------- RNG (PCG)
uint pcgHash(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
uint seedFor(uvec2 p, uint frame, uint salt) {
    return pcgHash(p.x + pcgHash(p.y + pcgHash(frame * 2654435761u + salt)));
}
float rnd(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return float(((w >> 22u) ^ w) >> 8u) * (1.0 / 16777216.0);
}

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// ---------------------------------------------------------------- camera
vec3 primaryDir(uvec2 px) {
    vec2 ndc = vec2(2.0 * (float(px.x) + 0.5) / float(U.misc0.x) - 1.0,
                    1.0 - 2.0 * (float(px.y) + 0.5) / float(U.misc0.y));
    return normalize(U.camFwd.xyz + ndc.x * U.camRight.xyz + ndc.y * U.camUp.xyz);
}
vec3 worldPos(uvec2 px, float t) { return U.camPos.xyz + primaryDir(px) * t; }

// Project a world position into the previous frame's pixel grid.
bool reproject(vec3 p, out ivec2 pp) {
    pp = ivec2(-1);
    vec3  v = p - U.prevPos.xyz;
    float z = dot(v, U.prevFwd.xyz);
    if (z <= 1e-4) return false;
    vec2 ndc = vec2(dot(v, U.prevRight.xyz) / (z * dot(U.prevRight.xyz, U.prevRight.xyz)),
                    dot(v, U.prevUp.xyz)    / (z * dot(U.prevUp.xyz,    U.prevUp.xyz)));
    vec2 uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    pp = ivec2(floor(uv * vec2(U.misc0.xy)));
    return all(greaterThanEqual(pp, ivec2(0))) && all(lessThan(pp, ivec2(U.misc0.xy)));
}

// Geometry test against last frame's G-buffer.
bool historyValid(ivec2 pp, vec3 p, vec3 n) {
    vec4 pnd = imageLoad(gNDPrev, pp);
    if (pnd.w <= 0.0) return false;
    vec2 ndc = vec2(2.0 * (float(pp.x) + 0.5) / float(U.misc0.x) - 1.0,
                    1.0 - 2.0 * (float(pp.y) + 0.5) / float(U.misc0.y));
    vec3 pd = normalize(U.prevFwd.xyz + ndc.x * U.prevRight.xyz + ndc.y * U.prevUp.xyz);
    vec3 pPrev = U.prevPos.xyz + pd * pnd.w;                 // previous surface point
    float dist = length(p - U.prevPos.xyz);
    return abs(dot(p - pPrev, n)) < 0.02 * dist && dot(pnd.xyz, n) > 0.9;
}

// ---------------------------------------------------------------- ray queries
struct Hit { float t; uint prim; vec2 bary; };

bool traceClosest(vec3 o, vec3 d, float tmax, out Hit h) {
    h = Hit(0.0, 0u, vec2(0.0));
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, gTLAS, gl_RayFlagsOpaqueEXT, 0xFF, o, 1e-4, d, tmax);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) return false;
    h.t    = rayQueryGetIntersectionTEXT(rq, true);
    h.prim = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
    h.bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    return true;
}

// Any-hit visibility: everything is opaque, so terminate on first hit.
bool occluded(vec3 o, vec3 d, float tmax) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, gTLAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                          0xFF, o, 1e-4, d, tmax);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// ---------------------------------------------------------------- triangles / lights
vec3  triNormal(Tri t) { return normalize(cross(t.p1.xyz - t.p0.xyz, t.p2.xyz - t.p0.xyz)); }
float triArea(Tri t)   { return 0.5 * length(cross(t.p1.xyz - t.p0.xyz, t.p2.xyz - t.p0.xyz)); }
vec3  triPoint(Tri t, vec2 b) { return t.p0.xyz * (1.0 - b.x - b.y) + t.p1.xyz * b.x + t.p2.xyz * b.y; }

// Albedo of a hit, tinted by the object colour. `lod` picks the mip level (no derivatives in compute).
vec3 triAlbedo(Tri t, vec2 b, float lod) {
    int layer = int(round(t.uv2m.z));
    if (layer < 0) return t.albedo.rgb;
    vec2 uv = t.uv01.xy * (1.0 - b.x - b.y) + t.uv01.zw * b.x + t.uv2m.xy * b.y;
    return t.albedo.rgb * textureLod(gTex, vec3(uv, float(layer)), lod).rgb;
}
// Mip level from the pixel footprint on the surface (params2.w = pixel angular spread).
float texLod(Tri t, float dist) {
    float texels = dist * U.params2.w * t.uv2m.w * 512.0;
    return max(log2(max(texels, 1.0)), 0.0);
}
vec2  sampleBary(vec2 u) { float su = sqrt(u.x); return vec2(su * (1.0 - u.y), su * u.y); }

vec2 sampleU(inout uint s) {                                  // 2D sample, quantised like the reservoir
    return unpackUnorm2x16(packUnorm2x16(vec2(rnd(s), rnd(s))));
}

// ---------------------------------------------------------------- light evaluation
// One function for every light type. A "sample" is (light index, u) and is independent of the
// shading point, so reservoirs can be re-evaluated at other pixels.
//   f   = unshadowed radiance contribution (no BRDF) expressed in the light's own measure
//         (area measure for TRI/POINT/SPOT, solid angle for DIR)
//   pdf = pdf of u in that same measure
struct LightEval { vec3 f; vec3 wi; float dist; float pdf; };

LightEval evalLight(uint li, vec2 u, vec3 p, vec3 n) {
    Light L = lights[li];
    LightEval e;
    e.f = vec3(0.0); e.wi = vec3(0.0, 1.0, 0.0); e.dist = 1.0; e.pdf = 1.0;

    if (L.type == LIGHT_DIR) {
        vec3  toL   = -L.dir;
        float omega = 6.28318530718 * (1.0 - L.cosOuter);
        float ct = 1.0 - u.x * (1.0 - L.cosOuter);
        float st = sqrt(max(0.0, 1.0 - ct * ct));
        float ph = 6.28318530718 * u.y;
        vec3  t  = normalize(abs(toL.x) > 0.9 ? cross(toL, vec3(0, 1, 0)) : cross(toL, vec3(1, 0, 0)));
        vec3  b  = cross(toL, t);
        e.wi   = normalize(t * (st * cos(ph)) + b * (st * sin(ph)) + toL * ct);
        e.dist = 1e4;
        e.pdf  = 1.0 / omega;
        e.f    = L.color / omega * max(dot(n, e.wi), 0.0);
        return e;
    }

    vec3 x, nl, radiance;
    bool omni = false;
    if (L.type == LIGHT_TRI) {
        Tri t    = tris[L.tri];
        x        = triPoint(t, sampleBary(u));
        nl       = triNormal(t);
        radiance = t.emission.rgb;
        e.pdf    = 1.0 / triArea(t);
    } else {                                                  // POINT/SPOT: omnidirectional intensity, position jittered on a small sphere
        float z = 1.0 - 2.0 * u.x;                            // (soft shadows only; no area pdf or facing term, so no wasted samples)
        float r = sqrt(max(0.0, 1.0 - z * z));
        float ph = 6.28318530718 * u.y;
        nl       = vec3(r * cos(ph), r * sin(ph), z);
        x        = L.pos + L.radius * nl;
        radiance = L.color;                                   // intensity in W/sr
        e.pdf    = 1.0;
        omni     = true;
    }

    vec3  d  = x - p;
    float d2 = max(dot(d, d), 1e-6);
    e.dist = sqrt(d2);
    e.wi   = d / e.dist;
    float cs = max(dot(n, e.wi), 0.0);
    float cl = omni ? 1.0 : max(dot(nl, -e.wi), 0.0);

    float atten = 1.0;
    if (L.type != LIGHT_TRI) {
        float rr = d2 / (L.range * L.range);
        atten = clamp(1.0 - rr * rr, 0.0, 1.0);
        atten *= atten;                                       // smooth range window
        if (L.type == LIGHT_SPOT)
            atten *= smoothstep(L.cosOuter, L.cosInner, dot(L.dir, normalize(p - L.pos)));
    }
    e.f = radiance * (cs * cl / d2) * atten;
    return e;
}

float targetPdf(vec3 p, vec3 n, uint li, vec2 u) { return luminance(evalLight(li, u, p, n).f); }

// Conservative upper bound of a light's contribution anywhere inside an AABB (screen tile or world cell).
// 0 means "cannot affect this volume" (out of range, behind an emitter, outside a spot cone).
float lightBound(Light L, vec3 bmin, vec3 bmax) {
    if (L.type == LIGHT_DIR) return L.power;
    vec3 c = 0.5 * (bmin + bmax), h = 0.5 * (bmax - bmin);
    vec3 dv = L.pos - clamp(L.pos, bmin, bmax);
    float d2 = max(dot(dv, dv), 1e-4);                       // distance² from light to the AABB
    if (d2 > L.range * L.range) return 0.0;
    if (L.type == LIGHT_TRI) {
        if (dot(L.dir, c - L.pos) + dot(abs(L.dir), h) <= 0.0) return 0.0;
    } else if (L.type == LIGHT_SPOT) {
        vec3 v = c - L.pos;
        float dist = length(v), rb = length(h);
        if (dist > rb) {
            float ang = acos(clamp(dot(v / dist, L.dir), -1.0, 1.0));
            if (ang > acos(L.cosOuter) + asin(min(1.0, rb / dist))) return 0.0;
        }
    }
    return L.power / max(d2, L.radius * L.radius);
}

// ---------------------------------------------------------------- light selection
// Global: alias table over light power (O(1), no spatial awareness).
float pickLightGlobal(inout uint s, out uint li) {
    uint n = U.misc1.x;
    li = NO_LIGHT;
    if (n == 0u) return 0.0;                                  // scene without lights
    uint i = min(uint(rnd(s) * float(n)), n - 1u);
    AliasEntry a = aliasTab[i];
    li = rnd(s) < a.prob ? i : a.alias;
    return lights[li].power * U.params2.x;                    // selection probability
}

// Per-tile: binary search in the tile's contribution CDF built by tile_lights.comp.
float pickLightTile(uint tile, inout uint s, out uint li) {
    li = NO_LIGHT;
    uint cnt = tiles[tile].count;
    if (cnt == 0u) return 0.0;
    float x = rnd(s) * tiles[tile].totalW;
    uint lo = 0u, hi = cnt - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (tiles[tile].cdf[mid] > x) hi = mid; else lo = mid + 1u;
    }
    li = tiles[tile].idx[lo];
    float prev = lo > 0u ? tiles[tile].cdf[lo - 1u] : 0.0;
    return (tiles[tile].cdf[lo] - prev) / tiles[tile].totalW;
}

// Light BVH: stochastic descent from the root, picking the child whose conservative contribution
// bound (lightBound-style, but over the node's AABB+power rather than a single light) is larger with
// higher probability. Unbiased: the returned value is the exact product of the per-level selection
// probabilities, i.e. the true pdf of picking light `li` under this tree (Monte-Carlo tree importance
// sampling, same principle RTXDI/Moana light clusters and UE5's light-grid clustering build on).
float nodeBound(BVHNode nd, vec3 p) {
    vec3 c = clamp(p, nd.bmin, nd.bmax);
    float d2 = max(dot(p - c, p - c), 1e-4);          // avoid a singularity when p is inside the box
    return nd.power / d2;
}
float pickLightBVH(vec3 p, inout uint s, out uint li) {
    li = NO_LIGHT;
    uint N = U.misc1.x, nDir = U.misc2.z;          // directional lights are lights[0..nDir)
    float pDir = nDir == 0u ? 0.0 : (nDir >= N ? 1.0 : 0.5);

    if (pDir > 0.0 && rnd(s) < pDir) {             // sun branch: not part of the tree
        li = min(uint(rnd(s) * float(nDir)), nDir - 1u);
        return pDir / float(nDir);
    }
    if (nDir >= N) return 0.0;                     // only suns, nothing to descend

    uint node = 0u;
    float pdf = 1.0 - pDir;
    for (;;) {
        BVHNode nd = bvhNodes[node];
        if (nd.rightOrLeaf == NO_LIGHT) { li = nd.leftOrLight; return pdf; }
        BVHNode lc = bvhNodes[nd.leftOrLight], rc = bvhNodes[nd.rightOrLeaf];
        float wl = nodeBound(lc, p), wr = nodeBound(rc, p);
        float sum = wl + wr;
        if (sum <= 0.0) { wl = lc.power; wr = rc.power; sum = wl + wr; }
        if (sum <= 0.0) return 0.0;
        float x = rnd(s) * sum;
        if (x < wl) { node = nd.leftOrLight; pdf *= wl / sum; }
        else        { node = nd.rightOrLeaf; pdf *= wr / sum; }
    }
}

uint  tileOf(uvec2 px) { return (px.y / TILE_SIZE) * U.misc2.x + px.x / TILE_SIZE; }
// Light sampling mode: 0 = global alias table, 1 = per-tile/per-cell CDF, 2 = light BVH.
float pickLight(uint tile, vec3 p, inout uint s, out uint li) {
    return U.misc2.y == 2u ? pickLightBVH(p, s, li)
         : U.misc2.y == 1u ? pickLightTile(tile, s, li)
         : pickLightGlobal(s, li);
}

// World-space version of the tile lists: a 3D grid over the scene, each cell with its own contribution CDF.
// Used wherever there is no screen tile: GI hit points and surface-cache texels.
uint cellOf(vec3 p) {
    ivec3 c = clamp(ivec3(floor((p - U.gridMin.xyz) * U.gridInvCell.xyz)), ivec3(0), ivec3(U.gridDim.xyz) - 1);
    return (uint(c.z) * U.gridDim.y + uint(c.y)) * U.gridDim.x + uint(c.x);
}
float pickLightCell(uint cell, inout uint s, out uint li) {
    li = NO_LIGHT;
    uint cnt = cells[cell].count;
    if (cnt == 0u) return 0.0;
    float x = rnd(s) * cells[cell].totalW;
    uint lo = 0u, hi = cnt - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (cells[cell].cdf[mid] > x) hi = mid; else lo = mid + 1u;
    }
    li = cells[cell].idx[lo];
    float prev = lo > 0u ? cells[cell].cdf[lo - 1u] : 0.0;
    return (cells[cell].cdf[lo] - prev) / cells[cell].totalW;
}
float pickLightWorld(vec3 p, inout uint s, out uint li) {
    return U.misc2.y == 2u ? pickLightBVH(p, s, li)
         : U.misc2.y == 1u ? pickLightCell(cellOf(p), s, li)
         : pickLightGlobal(s, li);
}

vec3 cosineSample(vec3 n, vec2 u) {
    float r = sqrt(u.x), phi = 6.28318530718 * u.y;
    vec3 t = normalize(abs(n.x) > 0.9 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0)));
    vec3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u.x)));
}

// Irradiance/pi cached at a surface point (add albedo to get outgoing radiance). backSide = hit from behind the winding normal.
vec3 cacheLookup(uint tri, vec2 b, bool backSide) {
    uint off = tris[tri].cache.x, k = tris[tri].cache.y;
    if (k == 0u) return vec3(0.0);
    uint i = min(uint(b.x * float(k)), k - 1u);
    uint j = min(uint(b.y * float(k)), k - 1u);
    if (i + j > k - 1u) i = k - 1u - j;
    uint cell = j * k - (j * (j - 1u)) / 2u + i;      // triangular layout: row j holds k-j cells
    return cacheIrr[(off + cell) * 2u + (backSide ? 1u : 0u)].rgb;
}

// ---------------------------------------------------------------- reservoirs (16 bytes)
struct Res { uint light; vec2 u; float W; float M; };

uvec4 packRes(Res r) {
    return uvec4(r.light, packUnorm2x16(r.u), floatBitsToUint(r.W), floatBitsToUint(r.M));
}
Res unpackRes(uvec4 v) {
    Res r;
    r.light = v.x;
    r.u     = unpackUnorm2x16(v.y);
    r.W    = uintBitsToFloat(v.z);
    r.M    = uintBitsToFloat(v.w);
    return r;
}
Res emptyRes() { return Res(NO_LIGHT, vec2(0.0), 0.0, 0.0); }

// Weighted reservoir update (selection part only; caller tracks wsum / M).
void resUpdate(inout float wsum, inout uint selTri, inout vec2 selB, inout float selP,
               uint li, vec2 u, float ph, float w, inout uint s) {
    wsum += w;
    if (w > 0.0 && rnd(s) * wsum < w) { selTri = li; selB = u; selP = ph; }
}

// common.glsl, after resUpdate
bool badf(float x) { return isnan(x) || isinf(x); }
