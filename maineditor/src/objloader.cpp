#include "objloader.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cfloat>
#include <algorithm>

struct VKey { int v = 0, t = 0, n = 0; bool operator==(const VKey& o) const { return v==o.v && t==o.t && n==o.n; } };
struct VKeyHash {
    size_t operator()(const VKey& k) const {
        return ((size_t)(k.v+7) * 73856093) ^ ((size_t)(k.t+7) * 19349663) ^ ((size_t)(k.n+7) * 83492791);
    }
};

static int parseIdx(const std::string& s) { return s.empty() ? 0 : std::stoi(s); }

bool loadOBJ(const std::string& path, MeshData& out, bool raw) {
    std::ifstream f(path);
    if (!f) return false;
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    std::unordered_map<VKey, uint32_t, VKeyHash> vmap;

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string tag;
        if (!(ls >> tag)) continue;

        if (tag == "v") {
            Vec3 p; ls >> p.x >> p.y >> p.z; positions.push_back(p);
        } else if (tag == "vn") {
            Vec3 n; ls >> n.x >> n.y >> n.z; normals.push_back(n);
        } else if (tag == "vt") {
            Vec2 t; ls >> t.x >> t.y; uvs.push_back(t);
        } else if (tag == "f") {
            std::vector<uint32_t> faceIdx;
            std::string tok;
            while (ls >> tok) {
                std::string parts[3];
                int pi = 0;
                for (char c : tok) {
                    if (c == '/') { if (++pi > 2) pi = 2; }
                    else parts[pi] += c;
                }
                VKey k;
                k.v = parseIdx(parts[0]);
                if (pi >= 1) k.t = parseIdx(parts[1]);
                if (pi >= 2) k.n = parseIdx(parts[2]);

                auto it = vmap.find(k);
                if (it != vmap.end()) { faceIdx.push_back(it->second); continue; }

                Vec3 p = (k.v > 0 && k.v <= (int)positions.size()) ? positions[k.v-1] : Vec3(0,0,0);
                Vec3 n = (k.n > 0 && k.n <= (int)normals.size())   ? normals[k.n-1]   : Vec3(0,1,0);
                Vec2 t = (k.t > 0 && k.t <= (int)uvs.size())       ? uvs[k.t-1]       : Vec2(0,0);

                out.vertices.push_back(p.x); out.vertices.push_back(p.y); out.vertices.push_back(p.z);
                out.vertices.push_back(n.x); out.vertices.push_back(n.y); out.vertices.push_back(n.z);
                out.vertices.push_back(t.x); out.vertices.push_back(t.y);
                uint32_t idx = (uint32_t)(out.vertices.size() / 8 - 1);
                vmap[k] = idx;
                faceIdx.push_back(idx);
            }
            for (size_t i = 1; i + 1 < faceIdx.size(); ++i) {
                out.indices.push_back(faceIdx[0]);
                out.indices.push_back(faceIdx[i]);
                out.indices.push_back(faceIdx[i+1]);
            }
        }
    }

    if (out.vertices.empty() || out.indices.empty()) return false;

    if (!raw) {
        Vec3 mn(FLT_MAX, FLT_MAX, FLT_MAX), mx(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (size_t i = 0; i + 8 <= out.vertices.size(); i += 8) {
            mn.x = std::min(mn.x, out.vertices[i+0]); mx.x = std::max(mx.x, out.vertices[i+0]);
            mn.y = std::min(mn.y, out.vertices[i+1]); mx.y = std::max(mx.y, out.vertices[i+1]);
            mn.z = std::min(mn.z, out.vertices[i+2]); mx.z = std::max(mx.z, out.vertices[i+2]);
        }
        // Match the engine's model fitting: center on X/Z but keep the bottom
        // of the model on the pivot so scale/position look identical in playtest.
        Vec3 c((mn.x + mx.x) * 0.5f, mn.y, (mn.z + mx.z) * 0.5f);
        float md = std::max(std::max(mx.x - mn.x, mx.y - mn.y), mx.z - mn.z);
        if (md < 1e-6f) md = 1.0f;
        float s = 1.0f / md;
        for (size_t i = 0; i + 8 <= out.vertices.size(); i += 8) {
            out.vertices[i+0] = (out.vertices[i+0] - c.x) * s;
            out.vertices[i+1] = (out.vertices[i+1] - c.y) * s;
            out.vertices[i+2] = (out.vertices[i+2] - c.z) * s;
        }
    }
    return true;
}