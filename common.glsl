

#extension GL_EXT_ray_query : require

const float PI       = 3.14159265359;
const uint  NO_LIGHT = 0xFFFFFFFFu;
const float RAY_EPS  = 2e-3;   


layout(set = 0, binding = 0) uniform accelerationStructureEXT gTLAS;

layout(set = 0, binding = 1, std140) uniform FrameUBO {
    vec4  camPos, camFwd, camRight, camUp;          
    vec4  prevPos, prevFwd, prevRight, prevUp;
    uvec4 misc0;   
    uvec4 misc1;   
    vec4  params;  
    uvec4 misc2;   
    vec4  params2; 
    vec4  gridMin;     
    vec4  gridInvCell; 
    uvec4 gridDim;     
    uvec4 misc3;   
    uvec4 misc4;   
        vec4  params3; 
    vec4  skyZenith;   
    vec4  skyHorizon;  
        vec4  jitter;      
    vec4  ssrParams;   // x = ssrOpacity (0..1), y = screen-edge fade width (UV), z = temporal history weight (0 = off), w = reserved
} U;

struct Tri {
    vec4 p0; vec4 p1; vec4 p2; vec4 albedo; vec4 emission;
    vec4 uv01;   
    vec4 uv2m;   
    uvec4 cache; 
    vec4 material;
};
layout(set = 0, binding = 2, std430) readonly buffer TriBuf   { Tri  tris[];     };

const uint LIGHT_TRI = 0u, LIGHT_POINT = 1u, LIGHT_SPOT = 2u, LIGHT_DIR = 3u;
const uint TILE_SIZE = 16u;
const uint TILE_MAX  = 256u;     




struct Light {
    vec3 pos; float radius;
    vec3 dir; uint  type;
    vec3 color; float range;
    float cosOuter; float cosInner; uint tri; float power;   
    uint strictAttenuation; uint pad0; uint pad1; uint pad2;
};
layout(set = 0, binding = 3, std430) readonly buffer LightBuf { Light lights[]; };

layout(set = 0, binding = 4,  rgba32f)  uniform image2D  gND;        
layout(set = 0, binding = 5,  rgba32f)  uniform image2D  gNDPrev;
layout(set = 0, binding = 6,  rgba16f)  uniform image2D  gAlbedo;    
layout(set = 0, binding = 7,  rgba32ui) uniform uimage2D resInit;    
layout(set = 0, binding = 8,  rgba32ui) uniform uimage2D resTCur;    
layout(set = 0, binding = 9,  rgba32ui) uniform uimage2D resTPrev;
layout(set = 0, binding = 10, rgba32ui) uniform uimage2D resSpatial; 
layout(set = 0, binding = 11, rgba16f)  uniform image2D  giCur;      
layout(set = 0, binding = 12, rgba16f)  uniform image2D  giPrev;
layout(set = 0, binding = 13, rgba8)    uniform image2D  finalImg;
layout(set = 0, binding = 18) uniform sampler2DArray gTex;   
layout(set = 0, binding = 16, rgba16f)  uniform image2D  dirCur;     
layout(set = 0, binding = 17, rgba16f)  uniform image2D  dirPrev;

layout(set = 0, binding = 23, rgba16f)  uniform image2D  litCur;     
layout(set = 0, binding = 24, rgba16f)  uniform image2D  litPrev;


layout(set = 0, binding = 19, std430) buffer CacheIrrBuf { vec4  cacheIrr[]; };   
layout(set = 0, binding = 20, std430) buffer CacheResBuf { uvec4 cacheRes[]; };   
layout(set = 0, binding = 21, std430) readonly buffer CacheOffBuf { uint cacheOffsets[]; };

const uint CELL_MAX = 64u;
struct Cell { uint count; float totalW; uint pad0; uint pad1; uint idx[CELL_MAX]; float cdf[CELL_MAX]; };
layout(set = 0, binding = 22, std430) buffer CellBuf { Cell cells[]; };           

struct AliasEntry { float prob; uint alias; };
layout(set = 0, binding = 14, std430) readonly buffer AliasBuf { AliasEntry aliasTab[]; };

struct Tile { uint count; float totalW; uint pad0; uint pad1; uint idx[TILE_MAX]; float cdf[TILE_MAX]; };
layout(set = 0, binding = 15, std430) buffer TileBuf { Tile tiles[]; };





struct BVHNode { vec3 bmin; float power; vec3 bmax; uint leftOrLight; uint rightOrLeaf; uint pad0, pad1, pad2; };
layout(set = 0, binding = 25, std430) readonly buffer BVHBuf { BVHNode bvhNodes[]; };
layout(set = 0, binding = 35, rgba16f) uniform image2D gMaterial;
layout(set = 0, binding = 36, rgba16f) uniform image2D directSpec;

// Half-resolution screen-space reflection target (single-level storage image),
// written by ssr_half.comp and bilateral-upsampled in composite.comp.
layout(set = 0, binding = 45, rgba16f) uniform image2D uSSRHalf;
// SSR blur chain: ssr_half.comp traces into uSSRRaw, ssr_blur.comp blurs
// uSSRRaw -> uSSRTmp (horizontal) -> uSSRHalf (vertical).
// All SSR buffers share one contract: PREMULTIPLIED coverage.
//   rgb = reflected HDR colour * coverage, a = coverage (= hit mask * screen-edge fade, 0..1).
// Premultiplied data blurs/accumulates linearly (misses don't darken hits); composite resolves rgb / a
// and applies Fresnel / roughness / ssrOpacity at full resolution.
layout(set = 0, binding = 46, rgba16f) uniform image2D uSSRRaw;
layout(set = 0, binding = 47, rgba16f) uniform image2D uSSRTmp;
// SSR temporal history (ping-ponged with the frame parity exactly like gND / gNDPrev).
// ssr_blur.comp (pass 2) resolves uSSRHalf + uSSRHistPrev -> uSSRHistCur; composite.comp reads uSSRHistCur.
layout(set = 0, binding = 48, rgba16f) uniform image2D uSSRHistCur;
layout(set = 0, binding = 49, rgba16f) uniform image2D uSSRHistPrev;


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

vec3 skyRadiance(vec3 dir) {
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(U.skyHorizon.xyz, U.skyZenith.xyz, t);
}


vec3 primaryDir(uvec2 px) {
    vec2 samplePx = vec2(px) + vec2(0.5) + U.jitter.xy;
    vec2 ndc = vec2(2.0 * samplePx.x / float(U.misc0.x) - 1.0,
                    1.0 - 2.0 * samplePx.y / float(U.misc0.y));
    return normalize(U.camFwd.xyz + ndc.x * U.camRight.xyz + ndc.y * U.camUp.xyz);
}
vec3 worldPos(uvec2 px, float t) { return U.camPos.xyz + primaryDir(px) * t; }


bool reproject(vec3 p, out ivec2 pp) {
    pp = ivec2(-1);
    vec3  v = p - U.prevPos.xyz;
    float z = dot(v, U.prevFwd.xyz);
    if (z <= 1e-4) return false;
    vec2 ndc = vec2(dot(v, U.prevRight.xyz) / (z * dot(U.prevRight.xyz, U.prevRight.xyz)),
                    dot(v, U.prevUp.xyz)    / (z * dot(U.prevUp.xyz,    U.prevUp.xyz)));
    vec2 uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    pp = ivec2(floor(uv * vec2(U.misc0.xy) - U.jitter.zw));
    return all(greaterThanEqual(pp, ivec2(0))) && all(lessThan(pp, ivec2(U.misc0.xy)));
}


bool historyValid(ivec2 pp, vec3 p, vec3 n) {
    vec4 pnd = imageLoad(gNDPrev, pp);
    if (pnd.w <= 0.0) return false;
    vec2 previousPx = vec2(pp) + vec2(0.5) + U.jitter.zw;
    vec2 ndc = vec2(2.0 * previousPx.x / float(U.misc0.x) - 1.0,
                    1.0 - 2.0 * previousPx.y / float(U.misc0.y));
    vec3 pd = normalize(U.prevFwd.xyz + ndc.x * U.prevRight.xyz + ndc.y * U.prevUp.xyz);
    vec3 pPrev = U.prevPos.xyz + pd * pnd.w;                 
    float dist = length(p - U.prevPos.xyz);
    float depthTolerance = max(4.0 * RAY_EPS, dist * U.params2.w);
    return abs(dot(p - pPrev, n)) < depthTolerance && dot(pnd.xyz, n) > 0.9;
}

vec3 previousWorldPos(ivec2 px, float t) {
    vec2 samplePx = vec2(px) + vec2(0.5) + U.jitter.zw;
    vec2 ndc = vec2(2.0 * samplePx.x / float(U.misc0.x) - 1.0,
                    1.0 - 2.0 * samplePx.y / float(U.misc0.y));
    vec3 dir = normalize(U.prevFwd.xyz + ndc.x * U.prevRight.xyz + ndc.y * U.prevUp.xyz);
    return U.prevPos.xyz + dir * t;
}


struct Hit { float t; uint prim; vec2 bary; };

// Instance masks: 0x01 = full-detail geometry (primary rays), 0x02 = simplified RT proxy (secondary rays).
// instanceCustomIndex of the proxy instance = index of its first triangle in the tris[] buffer.
bool traceClosestMask(vec3 o, vec3 d, float tmax, uint mask, out Hit h) {
    h = Hit(0.0, 0u, vec2(0.0));
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, gTLAS, gl_RayFlagsOpaqueEXT, mask, o, 1e-4, d, tmax);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) return false;
    h.t    = rayQueryGetIntersectionTEXT(rq, true);
    h.prim = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true))
           + uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
    h.bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    return true;
}

// primary visibility: full-detail geometry
bool tracePrimary(vec3 o, vec3 d, float tmax, out Hit h) { return traceClosestMask(o, d, tmax, 0x01u, h); }
// everything else (GI, reflections, DI hits...): simplified proxy geometry
bool traceClosest(vec3 o, vec3 d, float tmax, out Hit h) { return traceClosestMask(o, d, tmax, 0x02u, h); }


bool occluded(vec3 o, vec3 d, float tmax) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, gTLAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                          0x02, o, 1e-4, d, tmax);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}


vec3  triNormal(Tri t) { return normalize(cross(t.p1.xyz - t.p0.xyz, t.p2.xyz - t.p0.xyz)); }
float triArea(Tri t)   { return 0.5 * length(cross(t.p1.xyz - t.p0.xyz, t.p2.xyz - t.p0.xyz)); }
vec3  triPoint(Tri t, vec2 b) { return t.p0.xyz * (1.0 - b.x - b.y) + t.p1.xyz * b.x + t.p2.xyz * b.y; }


vec3 triAlbedo(Tri t, vec2 b, float lod) {
    int layer = int(round(t.uv2m.z));
    if (layer < 0) return t.albedo.rgb;
    vec2 uv = t.uv01.xy * (1.0 - b.x - b.y) + t.uv01.zw * b.x + t.uv2m.xy * b.y;
    return t.albedo.rgb * textureLod(gTex, vec3(uv, float(layer)), lod).rgb;
}

vec3 triMappedNormal(Tri t, vec2 b) {
    vec3 n = triNormal(t);
    int layer = int(round(t.material.x));
    if (layer < 0) return n;
    vec2 uv0 = t.uv01.xy, uv1 = t.uv01.zw, uv2 = t.uv2m.xy;
    vec2 uv = uv0 * (1.0 - b.x - b.y) + uv1 * b.x + uv2 * b.y;
    vec3 e1 = t.p1.xyz - t.p0.xyz, e2 = t.p2.xyz - t.p0.xyz;
    vec2 d1 = uv1 - uv0, d2 = uv2 - uv0;
    float det = d1.x * d2.y - d1.y * d2.x;
    if (abs(det) < 1e-8) return n;
    vec3 tangent = normalize((e1 * d2.y - e2 * d1.y) / det);
    vec3 bitangent = normalize((e2 * d1.x - e1 * d2.x) / det);
    vec3 mapN = pow(max(textureLod(gTex, vec3(uv, float(layer)), 0.0).rgb, vec3(0.0)), vec3(1.0 / 2.2)) * 2.0 - 1.0;
    return normalize(tangent * mapN.x + bitangent * mapN.y + n * mapN.z);
}

float texLod(Tri t, float dist) {
    float texels = dist * U.params2.w * t.uv2m.w * 512.0;
    return max(log2(max(texels, 1.0)), 0.0);
}
vec2  sampleBary(vec2 u) { float su = sqrt(u.x); return vec2(su * (1.0 - u.y), su * u.y); }

vec2 sampleU(inout uint s) {                                  
    return unpackUnorm2x16(packUnorm2x16(vec2(rnd(s), rnd(s))));
}







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
    } else {                                                  
        float z = 1.0 - 2.0 * u.x;                            
        float r = sqrt(max(0.0, 1.0 - z * z));
        float ph = 6.28318530718 * u.y;
        nl       = vec3(r * cos(ph), r * sin(ph), z);
        x        = L.pos + L.radius * nl;
        radiance = L.color;                                   
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
        if (L.type == LIGHT_POINT && L.strictAttenuation != 0u) {
            vec3 centerDelta = p - L.pos;
            atten = dot(centerDelta, centerDelta) < L.range * L.range ? 1.0 : 0.0;
        } else {
            float rr = d2 / (L.range * L.range);
            atten = clamp(1.0 - rr * rr, 0.0, 1.0);
            atten *= atten;                                       
        }
        if (L.type == LIGHT_SPOT)
            atten *= smoothstep(L.cosOuter, L.cosInner, dot(L.dir, normalize(p - L.pos)));
    }
    e.f = radiance * (cs * cl / d2) * atten;
    return e;
}

float targetPdf(vec3 p, vec3 n, uint li, vec2 u) { return luminance(evalLight(li, u, p, n).f); }



float lightBound(Light L, vec3 bmin, vec3 bmax) {
    if (L.type == LIGHT_DIR) return L.power;
    vec3 c = 0.5 * (bmin + bmax), h = 0.5 * (bmax - bmin);
    vec3 dv = L.pos - clamp(L.pos, bmin, bmax);
    float d2 = max(dot(dv, dv), 1e-4);                       
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



float pickLightGlobal(inout uint s, out uint li) {
    uint n = U.misc1.x;
    li = NO_LIGHT;
    if (n == 0u) return 0.0;                                  
    uint i = min(uint(rnd(s) * float(n)), n - 1u);
    AliasEntry a = aliasTab[i];
    li = rnd(s) < a.prob ? i : a.alias;
    return lights[li].power * U.params2.x;                    
}


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






float nodeBound(BVHNode nd, vec3 p) {
    vec3 c = clamp(p, nd.bmin, nd.bmax);
    float d2 = max(dot(p - c, p - c), 1e-4);          
    return nd.power / d2;
}
float pickLightBVH(vec3 p, inout uint s, out uint li) {
    li = NO_LIGHT;
    uint N = U.misc1.x, nDir = U.misc2.z;          
    float pDir = nDir == 0u ? 0.0 : (nDir >= N ? 1.0 : 0.5);

    if (pDir > 0.0 && rnd(s) < pDir) {             
        li = min(uint(rnd(s) * float(nDir)), nDir - 1u);
        return pDir / float(nDir);
    }
    if (nDir >= N) return 0.0;                     

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

float pickLight(uint tile, vec3 p, inout uint s, out uint li) {
    return U.misc2.y == 2u ? pickLightBVH(p, s, li)
         : U.misc2.y == 1u ? pickLightTile(tile, s, li)
         : pickLightGlobal(s, li);
}



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


vec3 cacheLookup(uint tri, vec2 b, bool backSide) {
    uint off = tris[tri].cache.x, k = tris[tri].cache.y;
    if (k == 0u) return vec3(0.0);
    uint i = min(uint(b.x * float(k)), k - 1u);
    uint j = min(uint(b.y * float(k)), k - 1u);
    if (i + j > k - 1u) i = k - 1u - j;
    uint cell = j * k - (j * (j - 1u)) / 2u + i;      
    return cacheIrr[(off + cell) * 2u + (backSide ? 1u : 0u)].rgb;
}


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


void resUpdate(inout float wsum, inout uint selTri, inout vec2 selB, inout float selP,
               uint li, vec2 u, float ph, float w, inout uint s) {
    wsum += w;
    if (w > 0.0 && rnd(s) * wsum < w) { selTri = li; selB = u; selP = ph; }
}

float balanceHeuristic(float a, float b) {
    float sum = a + b;
    return sum > 0.0 ? a / sum : 0.0;
}


bool badf(float x) { return isnan(x) || isinf(x); }


// ---------------------------------------------------------------------------
// SSR shared helpers. Everything is prefixed ssr* so it can't collide with other passes.
// ---------------------------------------------------------------------------

// Weight exponent for the normal term of the bilateral filters (higher = stricter).
#define SSR_BILATERAL_NORMAL_POW 2.0   // low on purpose: gND holds normal-MAPPED normals, a strict term makes bumpy floors unfilterable (noisy)

// Interleaved gradient noise (Jimenez). Pass pixel + per-frame offset for temporal variation.
float ssrIGN(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

ivec2 ssrHalfSize() { return ivec2((U.misc0.xy + 1u) / 2u); }

// Full-res pixel that represents half-res texel hp (the pixel ssr_half.comp traces from).
// Half-res texel h therefore "lives" at full-res pixel coordinate 2h+1; every upsample / history
// lookup below is anchored to that so the traced position and the filter position agree.
ivec2 ssrRepPx(ivec2 hp) { return min(hp * 2 + 1, ivec2(U.misc0.xy) - 1); }

struct SsrSurf { vec3 p; vec3 n; float t; };   // world pos, normal, ray distance (t <= 0: sky)

SsrSurf ssrSurfAt(ivec2 fp) {
    SsrSurf s;
    vec4 nd = imageLoad(gND, fp);
    s.t = nd.w;
    s.n = nd.xyz;
    s.p = nd.w > 0.0 ? worldPos(uvec2(fp), nd.w) : vec3(0.0);
    return s;
}

// Edge-stopping weight between a centre surface c and a tap s: linear depth + normal.
// Depth is compared as distance from each surface's tangent plane (symmetric), i.e. the linear
// depth difference corrected for slope, so a floor seen at a grazing angle isn't treated as a
// depth edge, while a real silhouette (large plane distance) or crease (normals differ) is.
// footprintPx = how many full-res pixels of slack the plane distance gets at c's distance.
float ssrBilateralWeight(SsrSurf c, SsrSurf s, float footprintPx) {
    if (s.t <= 0.0) return 0.0;
    vec3  d    = s.p - c.p;
    float pd   = 0.5 * (abs(dot(d, c.n)) + abs(dot(d, s.n)));
    float tol  = max(0.01, c.t * U.params2.w * footprintPx);
    float wz   = exp(-pd / tol);
    float wn   = pow(max(dot(c.n, s.n), 0.0), SSR_BILATERAL_NORMAL_POW);
    return wz * wn;
}
