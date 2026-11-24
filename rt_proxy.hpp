// rt_proxy.hpp - simplified ray-tracing proxy geometry for Vektor Engine.
//
// Call appendRtProxy(sc, proxyVerts, params) right after buildScene(). It
//   * leaves sc.tris[0 .. nSrc) and sc.verts untouched (the full-detail geometry used for primary rays),
//   * appends the simplified triangles to sc.tris (indices nSrc .. nSrc+nProxy),
//     with their own radiance-cache slots and cacheOffsets entries,
//   * writes the proxy positions (9 floats / tri) to proxyVerts for the second BLAS.
//
// Needs meshoptimizer >= 0.22 (meshopt_SimplifyErrorAbsolute).
#pragma once
#include "world.hpp"

#include <meshoptimizer.h>

#include <array>
#include <map>
#include <unordered_map>

struct RtProxyParams {
    float targetRatio = 0.25f;   // fraction of triangles to keep per material group (1.0 = no simplification)
    float maxError    = 0.005f;  // ABSOLUTE world-space error limit. Keep small; bias secondary rays by >= this.
};

struct RtProxyStats { uint32_t srcTris = 0, proxyTris = 0; };

namespace rtproxy_detail {

inline V3 v3(const float* p) { return {p[0], p[1], p[2]}; }
inline void set4(float* d, V3 v) { d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = 0; }

using GroupKey = std::array<uint32_t, 10>;

inline GroupKey keyOf(const GpuTri& t) {
    GroupKey k{};
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    k[0] = bits(t.albedo[0]); k[1] = bits(t.albedo[1]); k[2] = bits(t.albedo[2]);
    k[3] = bits(t.uv2m[2]);                                   // texture layer
    k[4] = bits(t.material[0]); k[5] = bits(t.material[1]);   // normal layer, metallic
    k[6] = bits(t.material[2]); k[7] = bits(t.material[3]);   // smoothness, reflectivity / ssr flag
    k[8] = t.cache[2]; k[9] = t.cache[3];                     // wrap / filter mode
    return k;
}

inline bool isEmissive(const GpuTri& t) { return t.emission[0] > 0 || t.emission[1] > 0 || t.emission[2] > 0; }

// Same slot allocation rule as Scene::addTri.
inline void allocCache(Scene& sc, GpuTri& t) {
    V3 a = v3(t.p0), b = v3(t.p1), c = v3(t.p2);
    float longest = std::max({length(b - a), length(c - b), length(a - c)});
    float area2 = length(cross(b - a, c - a)) * 0.5f;
    uint32_t k = 0, n = 0;
    if (t.albedo[0] + t.albedo[1] + t.albedo[2] > 1e-4f && area2 > 1e-8f) {
        k = (uint32_t)std::min(32.0f, std::max(1.0f, std::ceil(longest / std::max(sc.cellSize, 0.02f))));
        n = k * (k + 1) / 2;
        if (sc.cacheTexels + n > CACHE_MAX_TEXELS) { k = 1; n = 1; }
        if (sc.cacheTexels + n > CACHE_MAX_TEXELS) { k = 0; n = 0; }
    }
    t.cache[0] = sc.cacheTexels; t.cache[1] = k;
    sc.cacheOffsets.push_back(sc.cacheTexels);
    sc.cacheTexels += n;
    sc.layoutHash = (sc.layoutHash ^ k) * 1099511628211ull;
}

// Uniform grid over the source triangles of one group (by centroid) for UV transfer.
struct Grid {
    float cell = 1.0f;
    std::unordered_map<uint64_t, std::vector<uint32_t>> map;
    static uint64_t hash(int x, int y, int z) {
        return (uint64_t(uint32_t(x) & 0x1FFFFF) << 42) | (uint64_t(uint32_t(y) & 0x1FFFFF) << 21) | uint64_t(uint32_t(z) & 0x1FFFFF);
    }
    int ci(float v) const { return (int)std::floor(v / cell); }
    void insert(uint32_t id, V3 c) { map[hash(ci(c.x), ci(c.y), ci(c.z))].push_back(id); }
};

// Evaluate the (affine) UV mapping of source triangle s at world point q on its plane.
inline bool uvAt(const GpuTri& s, V3 q, float& u, float& v) {
    V3 p0 = v3(s.p0), e1 = v3(s.p1) - p0, e2 = v3(s.p2) - p0, r = q - p0;
    float d11 = dot(e1, e1), d12 = dot(e1, e2), d22 = dot(e2, e2);
    float det = d11 * d22 - d12 * d12;
    if (std::fabs(det) < 1e-20f) return false;
    float r1 = dot(r, e1), r2 = dot(r, e2);
    float a = (r1 * d22 - r2 * d12) / det, b = (r2 * d11 - r1 * d12) / det;
    float u0 = s.uv01[0], v0 = s.uv01[1], u1 = s.uv01[2], v1 = s.uv01[3], u2 = s.uv2m[0], v2 = s.uv2m[1];
    u = u0 + a * (u1 - u0) + b * (u2 - u0);
    v = v0 + a * (v1 - v0) + b * (v2 - v0);
    return true;
}

} // namespace rtproxy_detail

inline RtProxyStats appendRtProxy(Scene& sc, std::vector<float>& proxyVerts, const RtProxyParams& prm = {}) {
    using namespace rtproxy_detail;
    RtProxyStats st;
    const uint32_t nSrc = (uint32_t)sc.tris.size();
    st.srcTris = nSrc;
    proxyVerts.clear();

    auto pushTri = [&](GpuTri t) {
        if (sc.tris.size() >= size_t(nSrc) + MAX_TRIS) return;   // triBuf holds 2 * MAX_TRIS
        allocCache(sc, t);
        sc.tris.push_back(t);
        const float* p[3] = {t.p0, t.p1, t.p2};
        for (auto* q : p) { proxyVerts.push_back(q[0]); proxyVerts.push_back(q[1]); proxyVerts.push_back(q[2]); }
    };

    // bucket by material; emissive triangles are copied unchanged (light sampling relies on their exact geometry)
    std::map<GroupKey, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < nSrc; ++i) {
        const GpuTri& t = sc.tris[i];
        if (isEmissive(t) || prm.targetRatio >= 0.999f) pushTri(t);
        else groups[keyOf(t)].push_back(i);
    }

    for (auto& [key, ids] : groups) {
        const size_t idxCount = ids.size() * 3;

        std::vector<float> soup(idxCount * 3);
        double edgeSum = 0;
        for (size_t k = 0; k < ids.size(); ++k) {
            const GpuTri& t = sc.tris[ids[k]];
            std::memcpy(&soup[k * 9 + 0], t.p0, 12);
            std::memcpy(&soup[k * 9 + 3], t.p1, 12);
            std::memcpy(&soup[k * 9 + 6], t.p2, 12);
            edgeSum += length(v3(t.p1) - v3(t.p0));
        }

        std::vector<uint32_t> remap(idxCount);
        size_t vcount = meshopt_generateVertexRemap(remap.data(), nullptr, idxCount, soup.data(), idxCount, 12);
        std::vector<uint32_t> idx(idxCount);
        std::vector<float> verts(vcount * 3);
        meshopt_remapIndexBuffer(idx.data(), nullptr, idxCount, remap.data());
        meshopt_remapVertexBuffer(verts.data(), soup.data(), idxCount, 12, remap.data());

        size_t target = std::max<size_t>(size_t(float(idxCount) * prm.targetRatio) / 3 * 3, 3);
        std::vector<uint32_t> simp(idxCount);
        float err = 0.0f;
        size_t simpCount = meshopt_simplify(simp.data(), idx.data(), idxCount, verts.data(), vcount, 12, target,
                                            prm.maxError, meshopt_SimplifyErrorAbsolute, &err);

        // spatial index of the source triangles for UV transfer
        Grid grid;
        grid.cell = std::max(float(edgeSum / double(ids.size())) * 2.0f, 1e-3f);
        for (uint32_t id : ids) {
            const GpuTri& t = sc.tris[id];
            grid.insert(id, (v3(t.p0) + v3(t.p1) + v3(t.p2)) * (1.0f / 3.0f));
        }

        const GpuTri& tmpl = sc.tris[ids[0]];
        for (size_t i = 0; i + 2 < simpCount; i += 3) {
            V3 a = v3(&verts[simp[i] * 3]), b = v3(&verts[simp[i + 1] * 3]), c = v3(&verts[simp[i + 2] * 3]);
            V3 cr = cross(b - a, c - a);
            if (dot(cr, cr) < 1e-14f) continue;
            V3 n = normalize(cr), cen = (a + b + c) * (1.0f / 3.0f);

            // nearest aligned source triangle (same plane, same facing)
            int cx = grid.ci(cen.x), cy = grid.ci(cen.y), cz = grid.ci(cen.z);
            const GpuTri* best = nullptr;
            float bestD = 1e30f;
            for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                auto it = grid.map.find(Grid::hash(cx + dx, cy + dy, cz + dz));
                if (it == grid.map.end()) continue;
                for (uint32_t id : it->second) {
                    const GpuTri& s = sc.tris[id];
                    V3 sn = normalize(cross(v3(s.p1) - v3(s.p0), v3(s.p2) - v3(s.p0)));
                    if (dot(sn, n) < 0.98f) continue;
                    if (std::fabs(dot(sn, cen - v3(s.p0))) > prm.maxError * 4.0f + 1e-4f) continue;
                    V3 sc3 = (v3(s.p0) + v3(s.p1) + v3(s.p2)) * (1.0f / 3.0f);
                    float d = length(sc3 - cen);
                    if (d < bestD) { bestD = d; best = &s; }
                }
            }

            GpuTri t = best ? *best : tmpl;
            set4(t.p0, a); set4(t.p1, b); set4(t.p2, c);
            bool ok = best != nullptr;
            if (ok) {
                float u[3], v[3];
                const V3 pv[3] = {a, b, c};
                for (int k = 0; k < 3 && ok; ++k) ok = uvAt(*best, pv[k], u[k], v[k]);
                if (ok) { t.uv01[0] = u[0]; t.uv01[1] = v[0]; t.uv01[2] = u[1]; t.uv01[3] = v[1]; t.uv2m[0] = u[2]; t.uv2m[1] = v[2]; }
            }
            if (!ok) { t.uv2m[2] = -1.0f; }   // no UV source: fall back to flat albedo (layer < 0 = untextured)
            pushTri(t);
        }
    }

    st.proxyTris = (uint32_t)sc.tris.size() - nSrc;
    sc.layoutHash = (sc.layoutHash ^ st.proxyTris) * 1099511628211ull;
    return st;
}
