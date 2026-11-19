#pragma once
#include "glmath.hpp"
#include <vector>
#include <cstdint>

struct MeshData {
    std::vector<float> vertices;      // pos(3) normal(3) uv(2)
    std::vector<uint32_t> indices;
};

inline MeshData makeCube() {
    MeshData m;
    Vec3 A(-0.5f,-0.5f,-0.5f), B(0.5f,-0.5f,-0.5f), C(0.5f,0.5f,-0.5f), D(-0.5f,0.5f,-0.5f);
    Vec3 E(-0.5f,-0.5f, 0.5f), F(0.5f,-0.5f, 0.5f), G(0.5f,0.5f, 0.5f), H(-0.5f,0.5f, 0.5f);
    struct Face { Vec3 n; Vec3 p[4]; };
    Face faces[] = {
        {{0,0,1},  {E,F,G,H}},
        {{0,0,-1}, {B,A,D,C}},
        {{1,0,0},  {F,B,C,G}},
        {{-1,0,0}, {A,E,H,D}},
        {{0,1,0},  {H,G,C,D}},
        {{0,-1,0}, {A,B,F,E}},
    };
    const float uv[4][2] = {{0,0},{1,0},{1,1},{0,1}};
    uint32_t base = 0;
    for (auto& f : faces) {
        for (int i = 0; i < 4; ++i) {
            m.vertices.push_back(f.p[i].x); m.vertices.push_back(f.p[i].y); m.vertices.push_back(f.p[i].z);
            m.vertices.push_back(f.n.x);    m.vertices.push_back(f.n.y);    m.vertices.push_back(f.n.z);
            m.vertices.push_back(uv[i][0]); m.vertices.push_back(uv[i][1]);
        }
        uint32_t idx[6] = {0,1,2, 0,2,3};
        for (int i = 0; i < 6; ++i) m.indices.push_back(base + idx[i]);
        base += 4;
    }
    return m;
}

inline MeshData makeQuad() {
    MeshData m;
    Vec3 p[4] = {
        {-0.5f, 0, -0.5f},
        { 0.5f, 0, -0.5f},
        { 0.5f, 0,  0.5f},
        {-0.5f, 0,  0.5f}
    };
    const float uv[4][2] = {{0,0},{1,0},{1,1},{0,1}};
    for (int i = 0; i < 4; ++i) {
        m.vertices.push_back(p[i].x); m.vertices.push_back(p[i].y); m.vertices.push_back(p[i].z);
        m.vertices.push_back(0); m.vertices.push_back(1); m.vertices.push_back(0);
        m.vertices.push_back(uv[i][0]); m.vertices.push_back(uv[i][1]);
    }
    uint32_t idx[6] = {0,1,2, 0,2,3};
    for (int i = 0; i < 6; ++i) m.indices.push_back(idx[i]);
    return m;
}

inline MeshData makeSphere(int rings = 20, int sectors = 32) {
    MeshData m;
    for (int r = 0; r <= rings; ++r) {
        float v = (float)r / rings;
        float phi = v * 3.14159265f;
        for (int s = 0; s <= sectors; ++s) {
            float u = (float)s / sectors;
            float theta = u * 6.2831853f;
            float x = std::sin(phi) * std::cos(theta);
            float y = std::cos(phi);
            float z = std::sin(phi) * std::sin(theta);
            m.vertices.push_back(0.5f*x); m.vertices.push_back(0.5f*y); m.vertices.push_back(0.5f*z);
            m.vertices.push_back(x);      m.vertices.push_back(y);      m.vertices.push_back(z);
            m.vertices.push_back(u);      m.vertices.push_back(1.0f - v);
        }
    }
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            uint32_t a = r*(sectors+1) + s;
            uint32_t b = a + sectors + 1;
            m.indices.push_back(a);   m.indices.push_back(b);   m.indices.push_back(a+1);
            m.indices.push_back(a+1); m.indices.push_back(b);   m.indices.push_back(b+1);
        }
    }
    return m;
}

// Cone along +Y: base circle at y = 0 (radius 0.5), apex at y = 1.
inline MeshData makeCone(int segments = 28) {
    MeshData m;
    const float r = 0.5f;
    const float h = 1.0f;
    // Side
    for (int i = 0; i < segments; ++i) {
        float a0 = 6.2831853f * i / segments;
        float a1 = 6.2831853f * (i + 1) / segments;
        float c0 = std::cos(a0), s0 = std::sin(a0);
        float c1 = std::cos(a1), s1 = std::sin(a1);
        Vec3 base0(r*c0, 0, r*s0), base1(r*c1, 0, r*s1), apex(0, h, 0);
        Vec3 n0 = normalize(Vec3(c0, r / h, s0));
        Vec3 n1 = normalize(Vec3(c1, r / h, s1));
        uint32_t baseIndex = (uint32_t)(m.vertices.size() / 8);
        const Vec3 p[3]   = {base0, base1, apex};
        const Vec3 n[3]   = {n0, n1, n0 + n1};
        const float uv[3][2] = {{0,0},{1,0},{0.5f,1}};
        for (int k = 0; k < 3; ++k) {
            m.vertices.push_back(p[k].x); m.vertices.push_back(p[k].y); m.vertices.push_back(p[k].z);
            m.vertices.push_back(n[k].x); m.vertices.push_back(n[k].y); m.vertices.push_back(n[k].z);
            m.vertices.push_back(uv[k][0]); m.vertices.push_back(uv[k][1]);
        }
        m.indices.push_back(baseIndex); m.indices.push_back(baseIndex + 1); m.indices.push_back(baseIndex + 2);
    }
    // Base cap
    uint32_t center = (uint32_t)(m.vertices.size() / 8);
    m.vertices.insert(m.vertices.end(), {0,0,0, 0,-1,0, 0.5f,0.5f});
    for (int i = 0; i < segments; ++i) {
        float a0 = 6.2831853f * i / segments;
        float a1 = 6.2831853f * (i + 1) / segments;
        uint32_t idx = (uint32_t)(m.vertices.size() / 8);
        m.vertices.insert(m.vertices.end(), {std::cos(a0)*r,0,std::sin(a0)*r, 0,-1,0, 0,0});
        m.vertices.insert(m.vertices.end(), {std::cos(a1)*r,0,std::sin(a1)*r, 0,-1,0, 0,0});
        m.indices.push_back(center); m.indices.push_back(idx); m.indices.push_back(idx + 1);
    }
    return m;
}

// Cylinder along +Y: base circle at y = 0, top circle at y = 1, radius 0.5.
inline MeshData makeCylinder(int segments = 20) {
    MeshData m;
    const float r = 0.5f;
    for (int i = 0; i < segments; ++i) {
        float a0 = 6.2831853f * i / segments;
        float a1 = 6.2831853f * (i + 1) / segments;
        float c0 = std::cos(a0), s0 = std::sin(a0);
        float c1 = std::cos(a1), s1 = std::sin(a1);
        uint32_t base = (uint32_t)(m.vertices.size() / 8);
        const Vec3 p[4] = {
            {r*c0, 0, r*s0}, {r*c1, 0, r*s1},
            {r*c1, 1, r*s1}, {r*c0, 1, r*s0}
        };
        const Vec3 n[4] = {
            {c0,0,s0}, {c1,0,s1}, {c1,0,s1}, {c0,0,s0}
        };
        const float uv[4][2] = {{0,0},{1,0},{1,1},{0,1}};
        for (int k = 0; k < 4; ++k) {
            m.vertices.push_back(p[k].x); m.vertices.push_back(p[k].y); m.vertices.push_back(p[k].z);
            m.vertices.push_back(n[k].x); m.vertices.push_back(n[k].y); m.vertices.push_back(n[k].z);
            m.vertices.push_back(uv[k][0]); m.vertices.push_back(uv[k][1]);
        }
        m.indices.push_back(base);   m.indices.push_back(base+1); m.indices.push_back(base+2);
        m.indices.push_back(base);   m.indices.push_back(base+2); m.indices.push_back(base+3);
    }
    return m;
}