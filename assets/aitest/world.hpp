// world.hpp - CPU-side scene model + .scr script interpreter (no Vulkan in here).
//
// A .scr file is a list of commands, one per line ('#' starts a comment). Commands outside any
// "on ... end" block run once at startup. See scene.scr for a full example.
//
//   texture <id> <image>                       load PNG/JPG/TGA/BMP (resized to 512x512 + mipmaps)
//   model   <id> <file.obj> [raw]              load OBJ; by default fitted to 1 unit, base on y=0
//   spawn   <name> <cube|quad|sphere|modelId> [pos x y z] [rot x y z] [size x y z | scale s]
//                  [color r g b] [tex id] [uvscale s] [emit r g b] [collide on|off]
//           (name ending in '#' gets a running number: crate# -> crate1, crate2, ...)
//   destroy <name | prefix*>
//   set     <name> pos|rot|size|color|emit|dir ... | scale s | tex id|none | uvscale s | range v | radius v
//   move    <name> dx dy dz       rotate <name> dx dy dz (degrees)
//   light point <name> x y z  r g b  range [radius]
//   light spot  <name> x y z  dx dy dz  r g b  range outerDeg innerDeg [radius]
//   light sun   <name> dx dy dz  r g b  [angularRadiusDeg]
//   camera pos x y z | camera look yawDeg pitchDeg | camera lock on|off
//   var $v value | calc $v a + - * / min max b | rand $v min max | if a < > <= >= == != b ... endif
//   builtin cornell | builtin pointlights [N] | builtin spots | builtin sun
//   area on|off                    emissive triangles emit light (off: only point/spot/sun lights remain)
//   gi path|cache|screen           GI mode: path tracing with NEE | surface cache | surface cache + screen traces
//   gi rate 1|2 | gi cellsize v    GI trace rate (2 = quarter of the pixels per frame), surface-cache texel size
//   denoise on|off|<strength>      spatial denoise of direct + GI in the composite pass (default 1, 0 = off, up to 4)
//   collision camera on|off [radius]   the camera cannot move through solid objects (default off, radius 0.3)
//   collision <name|prefix*> on|off    make objects solid / non-solid (same as: set <name> collide on|off)
//   print <text>
//
//   on start | on key <K> | on every <seconds> | on frame | on collide <A> <B>     ...commands...      end
//
// Colliders are oriented boxes built from each object's mesh bounds * size * rotation (a sphere or a torus
// is therefore approximated by its bounding box). Every object is solid unless spawned with "collide off".
// "on collide A B" fires once when an object matching A starts overlapping one matching B (names may end
// in '*'). Inside the block @a / @b are replaced by the two object names and $hitx $hity $hitz hold the
// midpoint between them, e.g.   on collide crate* ball  /  set @b color 1 0 0  /  destroy @a  /  end
//
// Numbers can be written as $variables. Built-ins: $time $dt $frame $camx $camy $camz $camyaw
// $campitch $objects $lights.
#pragma once

#include <GLFW/glfw3.h>   // only for GLFW_KEY_* constants

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ------------------------------------------------------------------ math
constexpr float kPi = 3.14159265359f;
struct V3 { float x, y, z; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 normalize(V3 a) { float l = length(a); return l > 0 ? a * (1.0f / l) : V3{0, 1, 0}; }
inline float lum(V3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// Euler rotation in degrees, applied X then Y then Z.
inline V3 rotateEuler(V3 p, V3 deg) {
    float rx = deg.x * kPi / 180, ry = deg.y * kPi / 180, rz = deg.z * kPi / 180;
    float cx = std::cos(rx), sx = std::sin(rx), cy = std::cos(ry), sy = std::sin(ry), cz = std::cos(rz), sz = std::sin(rz);
    p = {p.x, p.y * cx - p.z * sx, p.y * sx + p.z * cx};
    p = {p.x * cy + p.z * sy, p.y, -p.x * sy + p.z * cy};
    p = {p.x * cz - p.y * sz, p.x * sz + p.y * cz, p.z};
    return p;
}

// ------------------------------------------------------------------ GPU-facing structs (mirror common.glsl)
constexpr uint32_t MAX_TRIS = 262144, MAX_LIGHTS = 16384;
constexpr uint32_t CACHE_MAX_TEXELS = 262144;     // surface cache: texels per side (each triangle gets k*(k+1)/2)
constexpr int TEX_SIZE = 512, TEX_MIPS = 10, MAX_TEX_LAYERS = 64;

struct GpuTri {                                     // std430, 112 bytes
    float p0[4], p1[4], p2[4], albedo[4], emission[4];
    float uv01[4];                                  // uv0.xy, uv1.xy
    float uv2m[4];                                  // uv2.xy, texture layer (-1 = none), uv-per-world-unit footprint
    uint32_t cache[4];                              // x = first surface-cache texel, y = grid size k (0 = not cached)
};
static_assert(sizeof(GpuTri) == 128, "GpuTri must match the std430 Tri struct in common.glsl");

enum LightType : uint32_t { LIGHT_TRI = 0, LIGHT_POINT = 1, LIGHT_SPOT = 2, LIGHT_DIR = 3 };
struct GpuLight {                                   // std430, 64 bytes
    float pos[3]; float radius;
    float dir[3]; uint32_t type;
    float color[3]; float range;
    float cosOuter, cosInner; uint32_t tri; float power;
};
inline void setF3(float* d, V3 v) { d[0] = v.x; d[1] = v.y; d[2] = v.z; }

// Light BVH node (std430, 48 bytes): a binary tree over the light list, built CPU-side whenever the
// light set changes. Interior nodes bound their subtree (AABB) and carry its total power; the GPU
// descends it stochastically, picking the child proportional to an upper bound of its contribution
// at the shading point (same idea as RTXDI/Moana light-cluster importance sampling). Leaves store one
// light index (right = light index, left = 0xFFFFFFFF marks a leaf).
struct GpuBVHNode {                                 // std430, 48 bytes
    float bmin[3]; float power;                     // power = total power of the subtree
    float bmax[3]; uint32_t leftOrLight;             // leaf: light index. interior: left child index
    uint32_t rightOrLeaf; uint32_t pad0, pad1, pad2; // leaf: rightOrLeaf = 0xFFFFFFFF (marks a leaf)
};
static_assert(sizeof(GpuBVHNode) == 48, "GpuBVHNode must match the std430 BVHNode struct in common.glsl");

inline size_t texLevelOffset(int level) { size_t off = 0; for (int l = 0; l < level; ++l) { size_t s = TEX_SIZE >> l; off += s * s * 4; } return off; }
inline size_t texTotalBytes() { return texLevelOffset(TEX_MIPS); }

// ------------------------------------------------------------------ world model
struct MTri { V3 p[3]; float uv[3][2]; };
struct Mesh { std::string name; std::vector<MTri> tris; };
struct Object {
    std::string name; int mesh = 0;
    V3 pos{0, 0, 0}, rot{0, 0, 0}, size{1, 1, 1}, color{0.73f, 0.73f, 0.73f}, emit{0, 0, 0};
    int tex = -1; float uvscale = 1.0f;
    bool collide = true;                              // solid for the camera and for "on collide" events
};
struct LightDef {
    std::string name; uint32_t type = LIGHT_POINT;
    V3 pos{0, 0, 0}, dir{0, -1, 0}, color{1, 1, 1};
    float radius = 0.03f, range = 5.0f, inner = 20.0f, outer = 35.0f;   // sun: outer = angular radius (deg)
};
struct TexUpload { uint32_t layer; std::vector<uint8_t> data; };   // 10 mip levels, tightly packed

// ---- collision: oriented bounding box (centre, three unit axes, half extents along them)
struct OBB { V3 c{0, 0, 0}; V3 ax[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; float h[3] = {0, 0, 0}; };

// Separating-axis test (3 + 3 face axes, 9 edge cross products). Touching boxes count as overlapping.
inline bool obbOverlap(const OBB& A, const OBB& B) {
    V3 d = B.c - A.c;
    auto separated = [&](V3 L) {
        float l2 = dot(L, L);
        if (l2 < 1e-8f) return false;                 // parallel edges: no usable axis
        L = L * (1.0f / std::sqrt(l2));
        float ra = 0, rb = 0;
        for (int i = 0; i < 3; ++i) { ra += std::fabs(dot(A.ax[i], L)) * A.h[i]; rb += std::fabs(dot(B.ax[i], L)) * B.h[i]; }
        return std::fabs(dot(d, L)) > ra + rb + 1e-5f;
    };
    for (int i = 0; i < 3; ++i) if (separated(A.ax[i]) || separated(B.ax[i])) return false;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) if (separated(cross(A.ax[i], B.ax[j]))) return false;
    return true;
}

// ---- built-in meshes
inline void addFace(Mesh& m, V3 c, V3 n, V3 u, V3 v) {
    V3 P[4] = {c - u * 0.5f - v * 0.5f, c + u * 0.5f - v * 0.5f, c + u * 0.5f + v * 0.5f, c - u * 0.5f + v * 0.5f};
    float UV[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    if (dot(cross(P[1] - P[0], P[2] - P[0]), n) < 0) { std::swap(P[1], P[3]); std::swap(UV[1], UV[3]); }
    for (int k : {1, 2}) {
        MTri t{};
        int idx[3] = {0, k, k + 1};
        for (int i = 0; i < 3; ++i) { t.p[i] = P[idx[i]]; t.uv[i][0] = UV[idx[i]][0]; t.uv[i][1] = UV[idx[i]][1]; }
        m.tris.push_back(t);
    }
}
inline Mesh makeCube() {
    Mesh m; m.name = "cube";
    addFace(m, {0, 0, 0.5f}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0});
    addFace(m, {0, 0, -0.5f}, {0, 0, -1}, {1, 0, 0}, {0, 1, 0});
    addFace(m, {0.5f, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0});
    addFace(m, {-0.5f, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0});
    addFace(m, {0, 0.5f, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1});
    addFace(m, {0, -0.5f, 0}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1});
    return m;
}
inline Mesh makeQuad() {                            // unit square in XZ, normal +Y
    Mesh m; m.name = "quad";
    addFace(m, {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1});
    return m;
}
inline Mesh makeSphere() {
    Mesh m; m.name = "sphere";
    const int lon = 24, lat = 16;
    auto pt = [&](int i, int j) {
        float th = kPi * float(i) / lat, ph = 2 * kPi * float(j) / lon;
        return V3{0.5f * std::sin(th) * std::cos(ph), 0.5f * std::cos(th), 0.5f * std::sin(th) * std::sin(ph)};
    };
    for (int i = 0; i < lat; ++i)
        for (int j = 0; j < lon; ++j) {
            V3 P[4] = {pt(i, j), pt(i, j + 1), pt(i + 1, j + 1), pt(i + 1, j)};
            float UV[4][2] = {{float(j) / lon, float(i) / lat}, {float(j + 1) / lon, float(i) / lat},
                              {float(j + 1) / lon, float(i + 1) / lat}, {float(j) / lon, float(i + 1) / lat}};
            for (int k : {1, 2}) {
                int idx[3] = {0, k, k + 1};
                V3 a = P[idx[0]], b = P[idx[1]], c = P[idx[2]];
                V3 n = cross(b - a, c - a);
                if (length(n) < 1e-9f) continue;                          // pole degenerates
                if (dot(n, a + b + c) < 0) std::swap(idx[1], idx[2]);     // make winding point outward
                MTri t{};
                for (int q = 0; q < 3; ++q) { t.p[q] = P[idx[q]]; t.uv[q][0] = UV[idx[q]][0]; t.uv[q][1] = UV[idx[q]][1]; }
                m.tris.push_back(t);
            }
        }
    return m;
}

// ---- OBJ loader: v / vt / f (any polygon, fan-triangulated, negative indices ok). Normals/materials ignored.
inline bool loadObj(const std::string& path, bool fit, Mesh& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open file"; return false; }
    std::vector<V3> P;
    std::vector<std::array<float, 2>> T;
    std::vector<bool> hasUV;
    out.tris.clear();
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") { V3 v{0, 0, 0}; ss >> v.x >> v.y >> v.z; P.push_back(v); }
        else if (tag == "vt") { std::array<float, 2> t{0, 0}; ss >> t[0] >> t[1]; t[1] = 1.0f - t[1]; T.push_back(t); }
        else if (tag == "f") {
            struct FV { int p, t; };
            std::vector<FV> fv;
            std::string tok;
            bool ok = true;
            while (ss >> tok) {
                int pi = 0, ti = 0;
                size_t s1 = tok.find('/');
                pi = std::atoi(tok.substr(0, s1).c_str());
                if (s1 != std::string::npos) {
                    size_t s2 = tok.find('/', s1 + 1);
                    std::string tpart = tok.substr(s1 + 1, s2 == std::string::npos ? std::string::npos : s2 - s1 - 1);
                    if (!tpart.empty()) ti = std::atoi(tpart.c_str());
                }
                int pIdx = pi > 0 ? pi - 1 : (int)P.size() + pi;
                int tIdx = ti > 0 ? ti - 1 : (ti < 0 ? (int)T.size() + ti : -1);
                if (pIdx < 0 || pIdx >= (int)P.size()) { ok = false; break; }
                if (tIdx >= (int)T.size()) tIdx = -1;
                fv.push_back({pIdx, tIdx});
            }
            if (!ok || fv.size() < 3) continue;
            for (size_t i = 1; i + 1 < fv.size(); ++i) {
                FV idx[3] = {fv[0], fv[i], fv[i + 1]};
                MTri t{};
                bool allUV = true;
                for (int q = 0; q < 3; ++q) {
                    t.p[q] = P[idx[q].p];
                    if (idx[q].t >= 0) { t.uv[q][0] = T[idx[q].t][0]; t.uv[q][1] = T[idx[q].t][1]; } else allUV = false;
                }
                hasUV.push_back(allUV);
                out.tris.push_back(t);
            }
        }
    }
    if (out.tris.empty()) { err = "no faces found"; return false; }

    if (fit) {                                       // unit size, centred on x/z, sitting on y = 0
        V3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (auto& t : out.tris) for (auto& p : t.p) {
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        }
        float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-9f});
        V3 c{(lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f};
        for (auto& t : out.tris) for (auto& p : t.p) p = (p - c) * (1.0f / ext);
    }
    for (size_t i = 0; i < out.tris.size(); ++i) {   // no UVs in the file -> planar projection on the dominant axis
        if (hasUV[i]) continue;
        auto& t = out.tris[i];
        V3 n = cross(t.p[1] - t.p[0], t.p[2] - t.p[0]);
        float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
        for (int q = 0; q < 3; ++q) {
            if (ay >= ax && ay >= az) { t.uv[q][0] = t.p[q].x; t.uv[q][1] = t.p[q].z; }
            else if (ax >= az)        { t.uv[q][0] = t.p[q].z; t.uv[q][1] = t.p[q].y; }
            else                      { t.uv[q][0] = t.p[q].x; t.uv[q][1] = t.p[q].y; }
        }
    }
    return true;
}

// ---- texture loader: any image stb supports -> 512x512 RGBA8 + full mip chain (box filtered)
inline bool loadTextureData(const std::string& path, std::vector<uint8_t>& out) {
    int w, h, n;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (!px) return false;
    out.assign(texTotalBytes(), 0);
    for (int y = 0; y < TEX_SIZE; ++y)
        for (int x = 0; x < TEX_SIZE; ++x) {
            float fx = std::max(0.0f, (x + 0.5f) * w / TEX_SIZE - 0.5f), fy = std::max(0.0f, (y + 0.5f) * h / TEX_SIZE - 0.5f);
            int x0 = std::min((int)fx, w - 1), y0 = std::min((int)fy, h - 1), x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
            float tx = fx - x0, ty = fy - y0;
            for (int c = 0; c < 4; ++c) {
                float a = px[(y0 * w + x0) * 4 + c] * (1 - tx) + px[(y0 * w + x1) * 4 + c] * tx;
                float b = px[(y1 * w + x0) * 4 + c] * (1 - tx) + px[(y1 * w + x1) * 4 + c] * tx;
                out[(size_t(y) * TEX_SIZE + x) * 4 + c] = (uint8_t)std::lround(a * (1 - ty) + b * ty);
            }
        }
    stbi_image_free(px);
    for (int l = 1; l < TEX_MIPS; ++l) {
        size_t s = TEX_SIZE >> l, ps = TEX_SIZE >> (l - 1);
        uint8_t* dst = out.data() + texLevelOffset(l);
        const uint8_t* src = out.data() + texLevelOffset(l - 1);
        for (size_t y = 0; y < s; ++y)
            for (size_t x = 0; x < s; ++x)
                for (int c = 0; c < 4; ++c) {
                    int sum = src[((2 * y) * ps + 2 * x) * 4 + c] + src[((2 * y) * ps + 2 * x + 1) * 4 + c] +
                              src[((2 * y + 1) * ps + 2 * x) * 4 + c] + src[((2 * y + 1) * ps + 2 * x + 1) * 4 + c];
                    dst[(y * s + x) * 4 + c] = (uint8_t)((sum + 2) / 4);
                }
    }
    return true;
}

// ---- Vose alias table (O(1) sampling proportional to weight)
inline void buildAlias(const std::vector<float>& w, std::vector<float>& prob, std::vector<uint32_t>& alias) {
    size_t n = w.size();
    prob.assign(n, 1.0f); alias.assign(n, 0);
    if (n == 0) return;
    double sum = 0;
    for (float x : w) sum += x;
    std::vector<double> p(n);
    std::vector<size_t> small, large;
    for (size_t i = 0; i < n; ++i) { p[i] = sum > 0 ? w[i] * double(n) / sum : 1.0; (p[i] < 1.0 ? small : large).push_back(i); }
    while (!small.empty() && !large.empty()) {
        size_t s = small.back(); small.pop_back();
        size_t l = large.back(); large.pop_back();
        prob[s] = float(p[s]); alias[s] = (uint32_t)l;
        p[l] = p[l] + p[s] - 1.0;
        (p[l] < 1.0 ? small : large).push_back(l);
    }
}

// ---- Light BVH builder (power + spatial importance sampling, RTXDI/Moana-style light cluster tree)
// Binary tree over light indices, built bottom-up via recursive median split on the bounding box's
// largest axis (SAH would pick tighter splits, but at MAX_LIGHTS scale this is cheap and good enough:
// each node still carries an exact AABB + power sum, which is all the GPU traversal needs to be unbiased).
// Output is a flat array of 2*N-1 nodes (N leaves, N-1 interior), root at index 0; empty input -> empty tree.
// ---- Light BVH builder (power + spatial importance sampling, RTXDI/Moana-style light cluster tree)
inline void buildLightBVH(const std::vector<GpuLight>& lights, std::vector<GpuBVHNode>& nodes) {
    nodes.clear();
    size_t n = lights.size();
    if (n == 0) return;

    struct Item { V3 lo, hi; float power; uint32_t idx; };
    std::vector<Item> items(n);
    for (size_t i = 0; i < n; ++i) {
        const GpuLight& L = lights[i];
        V3 p{L.pos[0], L.pos[1], L.pos[2]};
        float r = (L.type == LIGHT_TRI) ? std::max(L.radius, 1e-3f)
                                        : std::min(L.range, 1000.0f);
        items[i].lo    = V3{p.x - r, p.y - r, p.z - r};
        items[i].hi    = V3{p.x + r, p.y + r, p.z + r};
        items[i].power = L.power;
        items[i].idx   = (uint32_t)i;
    }

    nodes.reserve(2 * n - 1);

    std::function<uint32_t(size_t, size_t)> build = [&](size_t lo, size_t hi) -> uint32_t {
        V3 bmin{1e30f, 1e30f, 1e30f}, bmax{-1e30f, -1e30f, -1e30f};
        double power = 0;
        for (size_t i = lo; i < hi; ++i) {
            bmin = {std::min(bmin.x, items[i].lo.x), std::min(bmin.y, items[i].lo.y), std::min(bmin.z, items[i].lo.z)};
            bmax = {std::max(bmax.x, items[i].hi.x), std::max(bmax.y, items[i].hi.y), std::max(bmax.z, items[i].hi.z)};
            power += items[i].power;
        }
        uint32_t ni = (uint32_t)nodes.size();
        nodes.emplace_back();
        GpuBVHNode node{};
        setF3(node.bmin, bmin); setF3(node.bmax, bmax); node.power = float(power);

        if (hi - lo == 1) {
            node.leftOrLight = items[lo].idx; node.rightOrLeaf = 0xFFFFFFFFu;
        } else {
            V3 ext = bmax - bmin;
            int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
            size_t mid = lo + (hi - lo) / 2;
            auto key = [axis](const Item& it) { V3 c = (it.lo + it.hi) * 0.5f; return axis == 0 ? c.x : axis == 1 ? c.y : c.z; };
            std::nth_element(items.begin() + lo, items.begin() + mid, items.begin() + hi,
                              [&](const Item& a, const Item& b) { return key(a) < key(b); });
            uint32_t l = build(lo, mid);
            uint32_t r = build(mid, hi);
            node.leftOrLight = l; node.rightOrLeaf = r;
        }
        nodes[ni] = node;
        return ni;
    };
    build(0, n);
}

// ------------------------------------------------------------------ world
struct World {
    std::vector<Mesh> meshes;
    std::map<std::string, int> meshIndex, modelByKey;
    std::vector<Object> objects;
    std::vector<LightDef> lights;
    std::map<std::string, int> texIds, texByPath;     // texIds: script name -> layer, texByPath: cache
    int texCount = 0;
    std::vector<TexUpload> pendingTex;                // CPU-loaded, waiting for the renderer to upload
    std::map<std::string, float> vars;
    std::vector<std::string> assetDirs;               // extra places to look for assets
    std::mt19937 rng{12345};
    uint32_t spawnCounter = 0;
    bool dirty = true;                                // scene needs re-upload / BLAS rebuild
    bool areaLights = true;                           // false: emissive triangles stop emitting (point/spot/sun only)
    int giMode = 1;                                   // 0 path (NEE), 1 surface cache, 2 surface cache + screen traces
    int giRate = 1;                                   // 1 = trace GI for every pixel each frame, 2 = one pixel per 2x2 quad
    float cacheCellSize = 0.3f;                       // surface cache texel size in world units
    float denoise = 1.0f;                             // composite-pass spatial filter strength (0 = off)
    bool camCollide = false;                          // camera is blocked by solid objects
    bool camLocked = false;                           // camera movement is frozen (scriptable via 'camera lock on|off')
    float camRadius = 0.3f;                           // camera collision sphere radius
    std::map<int, std::pair<V3, V3>> boundsCache;     // mesh index -> local min / max (meshes never change once added)

    // mirrors of runtime state (main.cpp keeps these in sync)
    float time = 0, dt = 0; uint32_t frame = 0;
    V3 camPos{0, 2.5f, 8.5f}; float camYaw = 0, camPitch = 0;   // degrees
    bool camChanged = false;

    World() {
        for (Mesh m : {makeCube(), makeQuad(), makeSphere()}) { meshIndex[m.name] = (int)meshes.size(); meshes.push_back(m); }
    }
    void clearScene() {                               // keeps meshes and texture cache
        objects.clear(); lights.clear(); vars.clear(); texIds.clear(); spawnCounter = 0; dirty = true;
        areaLights = true; giMode = 1; giRate = 1; cacheCellSize = 0.3f;
        denoise = 1.0f; camCollide = false; camLocked = false; camRadius = 0.3f;
        rng.seed(12345);
    }

    const std::pair<V3, V3>& meshBounds(int idx) {
        auto it = boundsCache.find(idx);
        if (it != boundsCache.end()) return it->second;
        V3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (const MTri& t : meshes[idx].tris)
            for (const V3& v : t.p) {
                lo = {std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
                hi = {std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
            }
        if (lo.x > hi.x) { lo = {0, 0, 0}; hi = {0, 0, 0}; }
        return boundsCache[idx] = {lo, hi};
    }
    // World-space collider of an object: same transform as buildScene (scale, then rotate, then translate).
    OBB colliderOf(const Object& o) {
        const auto& bb = meshBounds(o.mesh);
        V3 centre{(bb.first.x + bb.second.x) * 0.5f * o.size.x, (bb.first.y + bb.second.y) * 0.5f * o.size.y,
                  (bb.first.z + bb.second.z) * 0.5f * o.size.z};
        OBB b;
        b.c = rotateEuler(centre, o.rot) + o.pos;
        b.ax[0] = rotateEuler({1, 0, 0}, o.rot); b.ax[1] = rotateEuler({0, 1, 0}, o.rot); b.ax[2] = rotateEuler({0, 0, 1}, o.rot);
        b.h[0] = 0.5f * (bb.second.x - bb.first.x) * std::fabs(o.size.x);
        b.h[1] = 0.5f * (bb.second.y - bb.first.y) * std::fabs(o.size.y);
        b.h[2] = 0.5f * (bb.second.z - bb.first.z) * std::fabs(o.size.z);
        return b;
    }
    // Push a camera sphere (camRadius) out of every solid object. oldPos (last frame) only decides which side to
    // leave through when the centre ends up inside a box or on a zero-thickness quad.
    V3 resolveCamera(const std::vector<OBB>& boxes, V3 oldPos, V3 p) {
        const float r = camRadius;
        for (int iter = 0; iter < 4; ++iter) {
            bool moved = false;
            for (const OBB& b : boxes) {
                V3 d = p - b.c, od = oldPos - b.c;
                float l[3], ol[3], cl[3];
                for (int i = 0; i < 3; ++i) { l[i] = dot(d, b.ax[i]); ol[i] = dot(od, b.ax[i]); cl[i] = std::clamp(l[i], -b.h[i], b.h[i]); }
                V3 closest = b.ax[0] * cl[0] + b.ax[1] * cl[1] + b.ax[2] * cl[2];
                V3 delta = d - closest;                       // closest point on the box -> camera
                float dist = length(delta);
                if (dist >= r) continue;
                if (dist > 1e-5f) p = p + delta * ((r - dist) / dist);
                else {                                        // centre inside / on the box: leave along the shallowest axis
                    int best = 0; float bestPen = 1e30f;
                    for (int i = 0; i < 3; ++i) { float pen = b.h[i] + r - std::fabs(l[i]); if (pen < bestPen) { bestPen = pen; best = i; } }
                    float side = ol[best] < 0 ? -1.0f : 1.0f;
                    p = p + b.ax[best] * (side * (b.h[best] + r) - l[best]);
                }
                moved = true;
            }
            if (!moved) break;
        }
        return p;
    }
    // Move the camera from 'from' to 'to' in sub-steps of at most half the camera radius, so fast movement
    // cannot tunnel through thin geometry (the room walls are zero-thickness quads).
    V3 moveCamera(V3 from, V3 to) {
        if (!camCollide) return to;
        std::vector<OBB> boxes;
        for (const Object& o : objects) if (o.collide) boxes.push_back(colliderOf(o));
        if (boxes.empty()) return to;
        V3 d = to - from;
        int steps = std::min(64, std::max(1, (int)std::ceil(length(d) / (camRadius * 0.5f))));
        V3 p = from;
        for (int i = 1; i <= steps; ++i) {
            V3 next = from + d * (float(i) / float(steps));
            p = resolveCamera(boxes, p, next);
        }
        return p;
    }
    std::string resolve(const std::string& p) const {
        namespace fs = std::filesystem;
        if (fs::exists(p)) return p;
        for (const auto& d : assetDirs) {
            fs::path a = fs::path(d) / p;
            if (fs::exists(a)) return a.string();
            fs::path b = fs::path(d) / fs::path(p).filename();
            if (fs::exists(b)) return b.string();
        }
        return p;
    }
};

// ------------------------------------------------------------------ GPU scene packing
struct Scene {
    std::vector<GpuTri> tris;
    std::vector<float> verts;                         // 9 floats per triangle (BLAS input, non-indexed)
    std::vector<GpuLight> lights;
    bool truncated = false;
    // surface cache layout (one entry per triangle) + world bounds, used for the light grid
    std::vector<uint32_t> cacheOffsets;
    uint32_t cacheTexels = 0;
    uint64_t layoutHash = 1469598103934665603ull;
    float cellSize = 0.3f;
    bool dummy = false;
    V3 bmin{1e30f, 1e30f, 1e30f}, bmax{-1e30f, -1e30f, -1e30f};

    void addTri(V3 a, V3 b, V3 c, V3 albedo, V3 emission, const float uv[3][2], int layer, float footprint) {
        if (tris.size() >= MAX_TRIS) { truncated = true; return; }
        GpuTri t{};
        auto set4 = [](float* d, V3 v) { d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = 0; };
        set4(t.p0, a); set4(t.p1, b); set4(t.p2, c); set4(t.albedo, albedo); set4(t.emission, emission);
        t.uv01[0] = uv[0][0]; t.uv01[1] = uv[0][1]; t.uv01[2] = uv[1][0]; t.uv01[3] = uv[1][1];
        t.uv2m[0] = uv[2][0]; t.uv2m[1] = uv[2][1]; t.uv2m[2] = float(layer); t.uv2m[3] = footprint;
        // ---- surface cache allocation: k x k barycentric grid, triangular layout k(k+1)/2 texels, 0 for black/emissive/tiny tris
        {
            V3 e0 = b - a, e1 = c - b, e2 = a - c;
            float longest = std::max({length(e0), length(e1), length(e2)});
            float area2 = length(cross(b - a, c - a)) * 0.5f;
            uint32_t k = 0, n = 0;
            if (albedo.x + albedo.y + albedo.z > 1e-4f && area2 > 1e-8f) {
                k = (uint32_t)std::min(32.0f, std::max(1.0f, std::ceil(longest / std::max(cellSize, 0.02f))));
                n = k * (k + 1) / 2;
                if (cacheTexels + n > CACHE_MAX_TEXELS) { k = 1; n = 1; }          // out of budget: coarsest cell
                if (cacheTexels + n > CACHE_MAX_TEXELS) { k = 0; n = 0; }          // truly full: not cached
            }
            t.cache[0] = cacheTexels; t.cache[1] = k; t.cache[2] = t.cache[3] = 0;
            cacheOffsets.push_back(cacheTexels);
            cacheTexels += n;
            layoutHash = (layoutHash ^ k) * 1099511628211ull;
            if (!dummy)
                for (V3 v : {a, b, c}) {
                    bmin = {std::min(bmin.x, v.x), std::min(bmin.y, v.y), std::min(bmin.z, v.z)};
                    bmax = {std::max(bmax.x, v.x), std::max(bmax.y, v.y), std::max(bmax.z, v.z)};
                }
        }
        bool emissive = emission.x > 0 || emission.y > 0 || emission.z > 0;
if (emissive && lights.size() < MAX_LIGHTS) {
    V3    cr       = cross(b - a, c - a);
    float area     = 0.5f * length(cr);
    V3    centroid = (a + b + c) * (1.0f / 3.0f);
    float r2       = std::max({length(a - centroid), length(b - centroid), length(c - centroid)});
    GpuLight L{};
    setF3(L.pos, centroid);
    L.radius = std::max(r2, 1e-3f);            // circumradius: actually bounds the triangle
    setF3(L.dir, normalize(cr));
    L.type = LIGHT_TRI; setF3(L.color, emission); L.range = 1e9f;
    L.tri = (uint32_t)tris.size(); L.power = lum(emission) * area;
    if (L.power > 0) lights.push_back(L);
}
        tris.push_back(t);
        for (V3 v : {a, b, c}) { verts.push_back(v.x); verts.push_back(v.y); verts.push_back(v.z); }
    }
};

inline void buildScene(const World& w, Scene& s) {
    for (const Object& o : w.objects) {
        const Mesh& m = w.meshes[o.mesh];
        bool emissive = o.emit.x > 0 || o.emit.y > 0 || o.emit.z > 0;
        bool emitOn = emissive && w.areaLights;        // area lights switched off: panels stay as black geometry
        V3 albedo = emissive ? V3{0, 0, 0} : o.color;   // emitters don't reflect (keeps the GI estimator unbiased)
        int layer = emissive ? -1 : o.tex;
        for (const MTri& mt : m.tris) {
            V3 q[3];
            float uv[3][2];
            for (int i = 0; i < 3; ++i) {
                V3 sp{mt.p[i].x * o.size.x, mt.p[i].y * o.size.y, mt.p[i].z * o.size.z};
                q[i] = rotateEuler(sp, o.rot) + o.pos;
                uv[i][0] = mt.uv[i][0] * o.uvscale; uv[i][1] = mt.uv[i][1] * o.uvscale;
            }
            float worldArea = 0.5f * length(cross(q[1] - q[0], q[2] - q[0]));
            float uvArea = 0.5f * std::fabs((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]));
            float footprint = worldArea > 1e-12f ? std::sqrt(uvArea / worldArea) : 0.0f;
            s.addTri(q[0], q[1], q[2], albedo, emitOn ? o.emit : V3{0, 0, 0}, uv, layer, footprint);
        }
    }
    for (const LightDef& d : w.lights) {
        if (s.lights.size() >= MAX_LIGHTS) { s.truncated = true; break; }
        GpuLight L{};
        setF3(L.pos, d.pos); L.radius = std::max(d.radius, 0.001f); setF3(L.dir, normalize(d.dir));
        L.type = d.type; setF3(L.color, d.color); L.range = std::max(d.range, 0.01f);
        L.power = lum(d.color);
        if (d.type == LIGHT_SPOT) { L.cosOuter = std::cos(d.outer * kPi / 180); L.cosInner = std::cos(d.inner * kPi / 180); }
        if (d.type == LIGHT_DIR)  { L.cosOuter = std::cos(d.outer * kPi / 180); L.range = 1e9f; }
        if (L.power > 0) s.lights.push_back(L);
    }
    if (s.tris.empty()) {                             // BLAS needs at least one primitive; this one is degenerate
        const float uv[3][2] = {{0, 0}, {0, 0}, {0, 0}};
        s.dummy = true;
        s.addTri({0, -1000, 0}, {0, -1000, 0}, {0, -1000, 0}, {0, 0, 0}, {0, 0, 0}, uv, -1, 0);
        s.dummy = false;
    }
    if (s.bmin.x > s.bmax.x) { s.bmin = {-1, -1, -1}; s.bmax = {1, 1, 1}; }   // scene without geometry
}

// ------------------------------------------------------------------ script
inline int keyFromName(std::string n) {
    for (auto& c : n) c = (char)std::toupper((unsigned char)c);
    if (n.size() == 1 && n[0] >= 'A' && n[0] <= 'Z') return GLFW_KEY_A + (n[0] - 'A');
    if (n.size() == 1 && n[0] >= '0' && n[0] <= '9') return GLFW_KEY_0 + (n[0] - '0');
    if (n == "SPACE") return GLFW_KEY_SPACE;
    if (n == "ENTER") return GLFW_KEY_ENTER;
    if (n == "TAB") return GLFW_KEY_TAB;
    if (n == "LEFT") return GLFW_KEY_LEFT;
    if (n == "RIGHT") return GLFW_KEY_RIGHT;
    if (n == "UP") return GLFW_KEY_UP;
    if (n == "DOWN") return GLFW_KEY_DOWN;
    if (n.size() >= 2 && n[0] == 'F') { int k = std::atoi(n.c_str() + 1); if (k >= 1 && k <= 12) return GLFW_KEY_F1 + (k - 1); }
    return -1;
}

class Script {
public:
    struct Line { int no; std::string text; };
    struct Event {
        enum Kind { Start, Key, Every, Frame, Collide } kind = Start;
        int key = -1; float interval = 1.0f, acc = 0.0f;
        std::string a, b;                                       // Collide: name patterns
        std::set<std::pair<std::string, std::string>> active;   // Collide: pairs currently overlapping
        std::vector<Line> lines;
    };

    explicit Script(World& world) : w(world) {}

    bool loadFile(const std::string& path) {
        std::ifstream f(path);
        if (!f) { std::fprintf(stderr, "script: cannot open %s\n", path.c_str()); return false; }
        std::stringstream ss; ss << f.rdbuf();
        name = path;
        w.assetDirs.insert(w.assetDirs.begin(), std::filesystem::path(path).parent_path().string());
        loadText(ss.str(), path);
        return true;
    }

    void loadText(const std::string& text, const std::string& label) {
        name = label;
        events.clear();
        events.push_back(Event{});                    // implicit start block for top-level lines
        Event* cur = &events[0];
        bool inBlock = false;
        std::istringstream in(text);
        std::string raw;
        int no = 0;
        while (std::getline(in, raw)) {
            ++no;
            std::string line = stripComment(raw);
            auto tk = tokenize(line);
            if (tk.empty()) continue;
            if (tk[0] == "on") {
                if (inBlock) err(no, "'on' inside another block (missing 'end'?)");
                Event e;
                std::string kind = tk.size() > 1 ? tk[1] : "";
                if (kind == "start") e.kind = Event::Start;
                else if (kind == "frame") e.kind = Event::Frame;
                else if (kind == "collide" && tk.size() > 3) { e.kind = Event::Collide; e.a = tk[2]; e.b = tk[3]; }
                else if (kind == "every" && tk.size() > 2) { e.kind = Event::Every; e.interval = std::max(0.01f, (float)std::atof(tk[2].c_str())); }
                else if (kind == "key" && tk.size() > 2) {
                    e.kind = Event::Key; e.key = keyFromName(tk[2]);
                    if (e.key < 0) { err(no, "unknown key '" + tk[2] + "'"); continue; }
                } else { err(no, "bad event, expected: on start | on key K | on every S | on frame | on collide A B"); continue; }
                events.push_back(e);
                cur = &events.back();
                inBlock = true;
            } else if (tk[0] == "end" && inBlock) {
                inBlock = false;
                events.push_back(Event{});            // later top-level lines go to a fresh start block
                cur = &events.back();
            } else {
                cur->lines.push_back({no, line});
            }
        }
    }

    void runStart() { for (auto& e : events) if (e.kind == Event::Start) run(e); }
    void fireKey(int key) { for (auto& e : events) if (e.kind == Event::Key && e.key == key) run(e); }
    void update(float dt) {
        for (auto& e : events) {
            if (e.kind == Event::Every) {
                e.acc += dt;
                for (int g = 0; e.acc >= e.interval && g < 8; ++g) { e.acc -= e.interval; run(e); }
                if (e.acc > e.interval) e.acc = 0;
            } else if (e.kind == Event::Frame) run(e);
        }
        updateCollisions();
    }
    std::vector<int> keys() const {
        std::vector<int> k;
        for (auto& e : events) if (e.kind == Event::Key && std::find(k.begin(), k.end(), e.key) == k.end()) k.push_back(e.key);
        return k;
    }
    void execLine(const std::string& s) { auto tk = tokenize(s); if (!tk.empty()) command(tk, 0); }

    std::vector<Event> events;
    std::string name;

private:
    World& w;
    std::string subA, subB;                           // @a / @b substitution while an "on collide" block runs

    static bool nameMatch(const std::string& pat, const std::string& n) {
        if (!pat.empty() && pat.back() == '*') return n.compare(0, pat.size() - 1, pat, 0, pat.size() - 1) == 0;
        return n == pat;
    }
    static void replaceAll(std::string& s, const std::string& from, const std::string& to) {
        for (size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + to.size())) s.replace(p, from.size(), to);
    }

    // Fires "on collide A B" blocks on the frame a pair of solid objects starts overlapping.
    void updateCollisions() {
        bool any = false;
        for (auto& e : events) if (e.kind == Event::Collide) { any = true; break; }
        if (!any) return;
        std::vector<OBB> boxes;
        std::vector<const Object*> objs;
        for (const Object& o : w.objects) if (o.collide) { boxes.push_back(w.colliderOf(o)); objs.push_back(&o); }
        struct Hit { size_t ev; std::string a, b; V3 at; };
        std::vector<Hit> fire;
        for (size_t ei = 0; ei < events.size(); ++ei) {
            Event& e = events[ei];
            if (e.kind != Event::Collide) continue;
            std::set<std::pair<std::string, std::string>> now;
            for (size_t i = 0; i < objs.size(); ++i) {
                if (!nameMatch(e.a, objs[i]->name)) continue;
                for (size_t j = 0; j < objs.size(); ++j) {
                    if (i == j || !nameMatch(e.b, objs[j]->name)) continue;
                    if (!obbOverlap(boxes[i], boxes[j])) continue;
                    auto key = std::make_pair(objs[i]->name, objs[j]->name);
                    now.insert(key);
                    if (!e.active.count(key)) fire.push_back({ei, key.first, key.second, (boxes[i].c + boxes[j].c) * 0.5f});
                }
            }
            e.active = std::move(now);
        }
        for (const Hit& h : fire) {                    // run after detection: the blocks may spawn / destroy objects
            w.vars["$hitx"] = h.at.x; w.vars["$hity"] = h.at.y; w.vars["$hitz"] = h.at.z;
            subA = h.a; subB = h.b;
            run(events[h.ev]);
            subA.clear(); subB.clear();
        }
    }

    // '#' or '//' start a comment only at the beginning of a line or after whitespace, so names like crate# survive.
    static std::string stripComment(const std::string& s) {
        for (size_t i = 0; i < s.size(); ++i) {
            bool boundary = i == 0 || std::isspace((unsigned char)s[i - 1]);
            if (boundary && (s[i] == '#' || (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '/'))) return s.substr(0, i);
        }
        return s;
    }
    static std::vector<std::string> tokenize(const std::string& s) {
        std::vector<std::string> t;
        std::istringstream ss(s);
        std::string x;
        while (ss >> x) t.push_back(x);
        return t;
    }
    void err(int line, const std::string& m) const {
        std::fprintf(stderr, "script %s:%d: %s\n", name.c_str(), line, m.c_str());
    }

    float num(const std::string& t, int ln) {
        if (!t.empty() && t[0] == '$') {
            auto it = w.vars.find(t);
            if (it == w.vars.end()) { err(ln, "unknown variable " + t); return 0; }
            return it->second;
        }
        char* e = nullptr;
        float v = std::strtof(t.c_str(), &e);
        if (e == t.c_str() || *e != '\0') { err(ln, "bad number '" + t + "'"); return 0; }
        return v;
    }
    bool need(const std::vector<std::string>& t, size_t n, int ln, const char* usage) {
        if (t.size() < n) { err(ln, std::string("usage: ") + usage); return false; }
        return true;
    }
    V3 vec3(const std::vector<std::string>& t, size_t i, int ln) { return {num(t[i], ln), num(t[i + 1], ln), num(t[i + 2], ln)}; }

    void refreshBuiltins() {
        w.vars["$time"] = w.time; w.vars["$dt"] = w.dt; w.vars["$frame"] = float(w.frame);
        w.vars["$camx"] = w.camPos.x; w.vars["$camy"] = w.camPos.y; w.vars["$camz"] = w.camPos.z;
        w.vars["$camyaw"] = w.camYaw; w.vars["$campitch"] = w.camPitch;
        w.vars["$objects"] = float(w.objects.size()); w.vars["$lights"] = float(w.lights.size());
    }

    void run(const Event& e) {
        refreshBuiltins();
        int skip = 0;
        for (const Line& L : e.lines) {
            std::string text = L.text;
            if (!subA.empty()) { replaceAll(text, "@a", subA); replaceAll(text, "@b", subB); }
            auto t = tokenize(text);
            if (t.empty()) continue;
            if (skip > 0) { if (t[0] == "if") ++skip; else if (t[0] == "endif") --skip; continue; }
            if (t[0] == "endif") continue;
            if (t[0] == "if") {
                if (!need(t, 4, L.no, "if a <op> b")) { skip = 1; continue; }
                float a = num(t[1], L.no), b = num(t[3], L.no);
                const std::string& op = t[2];
                bool ok = op == "<" ? a < b : op == ">" ? a > b : op == "<=" ? a <= b : op == ">=" ? a >= b : op == "==" ? a == b : op == "!=" ? a != b : false;
                if (!ok) skip = 1;
                continue;
            }
            command(t, L.no);
            if (L.no > 0) refreshBuiltins();          // spawn/camera may have changed built-ins
        }
    }

    static bool isOn(const std::string& s) { return s == "on" || s == "1" || s == "true" || s == "yes"; }

    Object* findObj(const std::string& n) { for (auto& o : w.objects) if (o.name == n) return &o; return nullptr; }
    LightDef* findLight(const std::string& n) { for (auto& l : w.lights) if (l.name == n) return &l; return nullptr; }

    void destroy(const std::string& pat) {
        bool prefix = !pat.empty() && pat.back() == '*';
        std::string base = prefix ? pat.substr(0, pat.size() - 1) : pat;
        auto match = [&](const std::string& n) { return prefix ? n.compare(0, base.size(), base) == 0 : n == base; };
        auto a = std::remove_if(w.objects.begin(), w.objects.end(), [&](const Object& o) { return match(o.name); });
        auto b = std::remove_if(w.lights.begin(), w.lights.end(), [&](const LightDef& l) { return match(l.name); });
        if (a != w.objects.end() || b != w.lights.end()) w.dirty = true;
        w.objects.erase(a, w.objects.end());
        w.lights.erase(b, w.lights.end());
    }

    void command(const std::vector<std::string>& t, int ln) {
        const std::string& c = t[0];
        if (c == "texture") {
            if (!need(t, 3, ln, "texture <id> <image>")) return;
            std::string path = w.resolve(t[2]);
            auto it = w.texByPath.find(path);
            int layer;
            if (it != w.texByPath.end()) layer = it->second;
            else {
                if (w.texCount >= MAX_TEX_LAYERS) { err(ln, "too many textures (max 64)"); return; }
                TexUpload up; up.layer = (uint32_t)w.texCount;
                if (!loadTextureData(path, up.data)) { err(ln, "cannot load image '" + t[2] + "'"); return; }
                layer = w.texCount++;
                w.texByPath[path] = layer;
                w.pendingTex.push_back(std::move(up));
            }
            w.texIds[t[1]] = layer;
        } else if (c == "model") {
            if (!need(t, 3, ln, "model <id> <file.obj> [raw]")) return;
            bool raw = t.size() > 3 && t[3] == "raw";
            std::string path = w.resolve(t[2]);
            std::string key = path + (raw ? "|raw" : "");
            auto it = w.modelByKey.find(key);
            if (it == w.modelByKey.end()) {
                Mesh m; m.name = t[1];
                std::string e;
                if (!loadObj(path, !raw, m, e)) { err(ln, "model '" + t[2] + "': " + e); return; }
                std::printf("model %s: %zu triangles\n", t[1].c_str(), m.tris.size());
                w.modelByKey[key] = (int)w.meshes.size();
                w.meshes.push_back(std::move(m));
                it = w.modelByKey.find(key);
            }
            w.meshIndex[t[1]] = it->second;
        } else if (c == "spawn") {
            if (!need(t, 3, ln, "spawn <name> <mesh> [pos x y z] [rot x y z] [size x y z|scale s] [color r g b] [tex id] [uvscale s] [emit r g b]")) return;
            auto mi = w.meshIndex.find(t[2]);
            if (mi == w.meshIndex.end()) { err(ln, "unknown mesh '" + t[2] + "' (cube, quad, sphere, or a model id)"); return; }
            Object o;
            o.mesh = mi->second;
            o.name = t[1];
            if (!o.name.empty() && o.name.back() == '#') o.name = o.name.substr(0, o.name.size() - 1) + std::to_string(++w.spawnCounter);
            bool colorSet = false;
            for (size_t i = 3; i < t.size();) {
                const std::string& k = t[i];
                if (k == "pos" && i + 3 < t.size()) { o.pos = vec3(t, i + 1, ln); i += 4; }
                else if (k == "rot" && i + 3 < t.size()) { o.rot = vec3(t, i + 1, ln); i += 4; }
                else if (k == "size" && i + 3 < t.size()) { o.size = vec3(t, i + 1, ln); i += 4; }
                else if (k == "scale" && i + 1 < t.size()) { float s = num(t[i + 1], ln); o.size = {s, s, s}; i += 2; }
                else if (k == "color" && i + 3 < t.size()) { o.color = vec3(t, i + 1, ln); colorSet = true; i += 4; }
                else if (k == "emit" && i + 3 < t.size()) { o.emit = vec3(t, i + 1, ln); i += 4; }
                else if (k == "uvscale" && i + 1 < t.size()) { o.uvscale = num(t[i + 1], ln); i += 2; }
                else if (k == "collide" && i + 1 < t.size()) { o.collide = isOn(t[i + 1]); i += 2; }
                else if (k == "tex" && i + 1 < t.size()) {
                    auto ti = w.texIds.find(t[i + 1]);
                    if (ti == w.texIds.end()) err(ln, "unknown texture '" + t[i + 1] + "' (declare it with: texture <id> <file>)");
                    else o.tex = ti->second;
                    i += 2;
                } else { err(ln, "bad spawn option '" + k + "'"); ++i; }
            }
            if (o.tex >= 0 && !colorSet) o.color = {1, 1, 1};
            destroy(o.name);
            w.objects.push_back(o);
            w.dirty = true;
        } else if (c == "destroy") {
            if (!need(t, 2, ln, "destroy <name | prefix*>")) return;
            destroy(t[1]);
        } else if (c == "set") {
            if (!need(t, 4, ln, "set <name> <property> <values>")) return;
            const std::string& p = t[2];
            if (Object* o = findObj(t[1])) {
                if (p == "pos" && t.size() >= 6) o->pos = vec3(t, 3, ln);
                else if (p == "rot" && t.size() >= 6) o->rot = vec3(t, 3, ln);
                else if (p == "size" && t.size() >= 6) o->size = vec3(t, 3, ln);
                else if (p == "scale") { float s = num(t[3], ln); o->size = {s, s, s}; }
                else if (p == "color" && t.size() >= 6) o->color = vec3(t, 3, ln);
                else if (p == "emit" && t.size() >= 6) o->emit = vec3(t, 3, ln);
                else if (p == "uvscale") o->uvscale = num(t[3], ln);
                else if (p == "collide") o->collide = isOn(t[3]);
                else if (p == "tex") {
                    if (t[3] == "none") o->tex = -1;
                    else { auto ti = w.texIds.find(t[3]); if (ti == w.texIds.end()) err(ln, "unknown texture '" + t[3] + "'"); else o->tex = ti->second; }
                } else err(ln, "bad property '" + p + "' for an object");
                w.dirty = true;
            } else if (LightDef* l = findLight(t[1])) {
                if (p == "pos" && t.size() >= 6) l->pos = vec3(t, 3, ln);
                else if (p == "dir" && t.size() >= 6) l->dir = vec3(t, 3, ln);
                else if (p == "color" && t.size() >= 6) l->color = vec3(t, 3, ln);
                else if (p == "range") l->range = num(t[3], ln);
                else if (p == "radius") l->radius = num(t[3], ln);
                else err(ln, "bad property '" + p + "' for a light");
                w.dirty = true;
            } else err(ln, "no object or light named '" + t[1] + "'");
        } else if (c == "move" || c == "rotate") {
            if (!need(t, 5, ln, "move|rotate <name> dx dy dz")) return;
            V3 d = vec3(t, 2, ln);
            if (Object* o = findObj(t[1])) { if (c == "move") o->pos = o->pos + d; else o->rot = o->rot + d; w.dirty = true; }
            else if (LightDef* l = findLight(t[1])) { if (c == "move") l->pos = l->pos + d; else err(ln, "cannot rotate a light, use: set <name> dir x y z"); w.dirty = true; }
            else err(ln, "no object or light named '" + t[1] + "'");
        } else if (c == "light") {
            if (!need(t, 3, ln, "light point|spot|sun <name> ...")) return;
            LightDef l;
            l.name = t[2];
            if (!l.name.empty() && l.name.back() == '#') l.name = l.name.substr(0, l.name.size() - 1) + std::to_string(++w.spawnCounter);
            if (t[1] == "point") {
                if (!need(t, 10, ln, "light point <name> x y z r g b range [radius]")) return;
                l.type = LIGHT_POINT; l.pos = vec3(t, 3, ln); l.color = vec3(t, 6, ln); l.range = num(t[9], ln);
                if (t.size() > 10) l.radius = num(t[10], ln);
            } else if (t[1] == "spot") {
                if (!need(t, 14, ln, "light spot <name> x y z dx dy dz r g b range outerDeg innerDeg [radius]")) return;
                l.type = LIGHT_SPOT; l.pos = vec3(t, 3, ln); l.dir = vec3(t, 6, ln); l.color = vec3(t, 9, ln);
                l.range = num(t[12], ln); l.outer = num(t[13], ln); l.inner = t.size() > 14 ? num(t[14], ln) : l.outer * 0.6f;
                if (t.size() > 15) l.radius = num(t[15], ln);
            } else if (t[1] == "sun") {
                if (!need(t, 9, ln, "light sun <name> dx dy dz r g b [angularRadiusDeg]")) return;
                l.type = LIGHT_DIR; l.dir = vec3(t, 3, ln); l.color = vec3(t, 6, ln); l.outer = t.size() > 9 ? num(t[9], ln) : 0.5f;
            } else { err(ln, "light type must be point, spot or sun"); return; }
            destroy(l.name);
            w.lights.push_back(l);
            w.dirty = true;
        } else if (c == "camera") {
            if (!need(t, 3, ln, "camera pos x y z | camera look yaw pitch | camera lock on|off")) return;
            if (t[1] == "pos" && t.size() >= 5) { w.camPos = vec3(t, 2, ln); w.camChanged = true; }
            else if (t[1] == "look" && t.size() >= 4) { w.camYaw = num(t[2], ln); w.camPitch = num(t[3], ln); w.camChanged = true; }
            else if (t[1] == "lock" && t.size() >= 3) { w.camLocked = isOn(t[2]); }
            else err(ln, "usage: camera pos x y z | camera look yaw pitch | camera lock on|off");
        } else if (c == "var") {
            if (!need(t, 3, ln, "var $name value")) return;
            if (t[1][0] != '$') { err(ln, "variable names start with $"); return; }
            w.vars[t[1]] = num(t[2], ln);
        } else if (c == "calc") {
            if (!need(t, 5, ln, "calc $name a <+ - * / min max> b")) return;
            if (t[1][0] != '$') { err(ln, "variable names start with $"); return; }
            float a = num(t[2], ln), b = num(t[4], ln), r = 0;
            const std::string& op = t[3];
            if (op == "+") r = a + b; else if (op == "-") r = a - b; else if (op == "*") r = a * b;
            else if (op == "/") r = b != 0 ? a / b : 0;
            else if (op == "min") r = std::min(a, b); else if (op == "max") r = std::max(a, b);
            else err(ln, "calc operator must be + - * / min max");
            w.vars[t[1]] = r;
        } else if (c == "rand") {
            if (!need(t, 4, ln, "rand $name min max")) return;
            float lo = num(t[2], ln), hi = num(t[3], ln);
            w.vars[t[1]] = lo + (hi - lo) * std::uniform_real_distribution<float>(0.0f, 1.0f)(w.rng);
        } else if (c == "print") {
            for (size_t i = 1; i < t.size(); ++i) std::printf("%s%s", i > 1 ? " " : "", t[i][0] == '$' ? std::to_string(num(t[i], ln)).c_str() : t[i].c_str());
            std::printf("\n");
        } else if (c == "area") {
            if (!need(t, 2, ln, "area on|off")) return;
            w.areaLights = (t[1] == "on" || t[1] == "1" || t[1] == "true");
            w.dirty = true;
        } else if (c == "denoise") {
            if (!need(t, 2, ln, "denoise on|off|<strength 0..4>")) return;
            if (t[1] == "on") w.denoise = 1.0f;
            else if (t[1] == "off") w.denoise = 0.0f;
            else w.denoise = std::clamp(num(t[1], ln), 0.0f, 4.0f);
        } else if (c == "collision") {
            if (!need(t, 3, ln, "collision camera on|off [radius] | collision <name|prefix*> on|off")) return;
            if (t[1] == "camera") {
                w.camCollide = isOn(t[2]);
                if (t.size() > 3) w.camRadius = std::max(0.01f, num(t[3], ln));
            } else {
                bool on = isOn(t[2]), hit = false;
                for (auto& o : w.objects) if (nameMatch(t[1], o.name)) { o.collide = on; hit = true; }
                if (!hit) err(ln, "no object matches '" + t[1] + "'");
            }
        } else if (c == "gi") {
            if (!need(t, 2, ln, "gi path|cache|screen | gi rate 1|2 | gi cellsize v")) return;
            if (t[1] == "path") w.giMode = 0;
            else if (t[1] == "cache") w.giMode = 1;
            else if (t[1] == "screen") w.giMode = 2;
            else if (t[1] == "rate" && t.size() > 2) w.giRate = num(t[2], ln) >= 2 ? 2 : 1;
            else if (t[1] == "cellsize" && t.size() > 2) { w.cacheCellSize = std::max(0.05f, num(t[2], ln)); w.dirty = true; }
            else err(ln, "usage: gi path|cache|screen | gi rate 1|2 | gi cellsize v");
        } else if (c == "builtin") {
            if (!need(t, 2, ln, "builtin cornell|pointlights [N]|spots|sun")) return;
            builtin(t, ln);
        } else {
            err(ln, "unknown command '" + c + "'");
        }
    }

    void builtin(const std::vector<std::string>& t, int ln) {
        if (t[1] == "cornell") {
            execLine("spawn cornell_floor quad pos 0 0 0 size 5 1 5");
            execLine("spawn cornell_ceiling quad pos 0 5 0 rot 180 0 0 size 5 1 5");
            execLine("spawn cornell_back quad pos 0 2.5 -2.5 rot 90 0 0 size 5 1 5");
            execLine("spawn cornell_left quad pos -2.5 2.5 0 rot 0 0 -90 size 5 1 5 color 0.65 0.06 0.05");
            execLine("spawn cornell_right quad pos 2.5 2.5 0 rot 0 0 90 size 5 1 5 color 0.12 0.45 0.15");
            execLine("spawn cornell_tall cube pos -0.9 1.5 -0.7 rot 0 20 0 size 1.4 3.0 1.4");
            execLine("spawn cornell_short cube pos 0.95 0.7 0.5 rot 0 -17 0 size 1.3 1.4 1.3");
            for (int i = 0; i < 4; ++i)                // 4x4 grid of small emitters, varied colour / intensity
                for (int j = 0; j < 4; ++j) {
                    bool cool = (i + j) & 1;
                    float k = 8.0f + 3.0f * float((i * 5 + j * 3) % 5);
                    char buf[256];
                    std::snprintf(buf, sizeof(buf), "spawn cornell_em_%d_%d quad pos %.2f 4.98 %.2f rot 180 0 0 size 0.44 1 0.44 emit %.3f %.3f %.3f",
                                  i, j, -1.5f + i, -1.5f + j, (cool ? 0.55f : 1.0f) * k, (cool ? 0.75f : 0.72f) * k, (cool ? 1.0f : 0.45f) * k);
                    execLine(buf);
                }
        } else if (t[1] == "pointlights") {
            int n = t.size() > 2 ? (int)num(t[2], ln) : 512;
            std::mt19937 rng(1234);
            std::uniform_real_distribution<float> U(0.0f, 1.0f);
            auto sat = [](float v) { return std::min(1.0f, std::max(0.0f, v)); };
            int placed = 0;
            for (int guard = 0; placed < n && guard < n * 50 + 100; ++guard) {
                float x = -2.2f + 4.4f * U(rng), y = 0.3f + 4.4f * U(rng), z = -2.2f + 4.4f * U(rng);
                auto inCyl = [&](float cx, float cz, float r, float h) { return (x - cx) * (x - cx) + (z - cz) * (z - cz) < r * r && y < h + 0.15f; };
                if (inCyl(-0.9f, -0.7f, 1.05f, 3.0f) || inCyl(0.95f, 0.5f, 0.95f, 1.4f)) continue;
                float h = U(rng);
                V3 col{sat(std::fabs(h * 6 - 3) - 1) * 0.7f + 0.3f, sat(2 - std::fabs(h * 6 - 2)) * 0.7f + 0.3f, sat(2 - std::fabs(h * 6 - 4)) * 0.7f + 0.3f};
                LightDef l;
                l.name = "pl" + std::to_string(placed);
                l.type = LIGHT_POINT; l.pos = {x, y, z}; l.color = col * (8.0f / float(n)); l.radius = 0.03f; l.range = 4.0f;
                w.lights.push_back(l);
                ++placed;
            }
            w.dirty = true;
        } else if (t[1] == "spots") {
            for (int k = 0; k < 8; ++k) {
                float a = 2 * kPi * float(k) / 8.0f;
                LightDef l;
                l.name = "spot" + std::to_string(k);
                l.type = LIGHT_SPOT; l.pos = {1.8f * std::cos(a), 4.85f, 1.8f * std::sin(a)};
                V3 target{0.6f * std::cos(a + 0.8f), 0.0f, 0.6f * std::sin(a + 0.8f)};
                l.dir = target - l.pos; l.color = V3{1.0f, 0.9f, 0.75f} * 4.0f; l.radius = 0.05f; l.range = 8.0f; l.outer = 35; l.inner = 20;
                w.lights.push_back(l);
            }
            w.dirty = true;
        } else if (t[1] == "sun") {
            execLine("light sun sun 0.25 -0.45 -1.0 2.0 1.9 1.7 0.5");
        } else err(ln, "builtin must be cornell, pointlights, spots or sun");
    }
};
