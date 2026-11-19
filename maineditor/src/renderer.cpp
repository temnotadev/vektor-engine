#include "renderer.hpp"
#include "objloader.hpp"
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>

static const char* kVert = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vNormal;
out vec3 vWorld;
out vec2 vUV;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorld  = world.xyz;
    vNormal = mat3(uModel) * aNormal;
    vUV     = aUV;
    gl_Position = uProj * uView * world;
}
)";

static const char* kFrag = R"(#version 330 core
in vec3 vNormal;
in vec3 vWorld;
in vec2 vUV;

uniform vec3 uColor;
uniform vec3 uCamPos;
uniform sampler2D uTex;
uniform int  uHasTex;
uniform int  uUnlit;
uniform vec2 uUVScale;
uniform sampler2D uNormalTex;
uniform int  uHasNormalTex;
uniform float uMetallic;
uniform float uSmoothness;
uniform float uReflectivity;

#define MAX_LIGHTS 32
uniform int   uNumLights;
uniform int   uLightType[MAX_LIGHTS];
uniform vec3  uLightPos[MAX_LIGHTS];
uniform vec3  uLightDir[MAX_LIGHTS];
uniform vec3  uLightColor[MAX_LIGHTS];
uniform float uLightRange[MAX_LIGHTS];
uniform int   uLightStrictAttenuation[MAX_LIGHTS];
uniform vec2  uLightAngles[MAX_LIGHTS];

out vec4 FragColor;

void main() {
    if (uUnlit == 1) { FragColor = vec4(uColor, 1.0); return; }

    vec4 base = vec4(uColor, 1.0);
    if (uHasTex != 0) base *= texture(uTex, vUV * uUVScale);

    vec3 Ng = normalize(vNormal);
    vec3 N = Ng;
    if (uHasNormalTex != 0) {
        vec2 uv = vUV * uUVScale;
        vec3 dp1 = dFdx(vWorld);
        vec3 dp2 = dFdy(vWorld);
        vec2 duv1 = dFdx(uv);
        vec2 duv2 = dFdy(uv);
        vec3 dp2perp = cross(dp2, Ng);
        vec3 dp1perp = cross(Ng, dp1);
        vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
        vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
        float invmax = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-8));
        mat3 TBN = mat3(T * invmax, B * invmax, Ng);
        vec3 nt = texture(uNormalTex, uv).xyz * 2.0 - 1.0;
        N = normalize(TBN * nt);
    }

    vec3 V = normalize(uCamPos - vWorld);
    if (dot(N, V) < 0.0) N = -N;

    vec3 diffuseColor = base.rgb * (1.0 - 0.7 * uMetallic);
    vec3 col = diffuseColor * (0.55 + 0.35 * max(dot(N, V), 0.0));
    float shininess = mix(4.0, 256.0, uSmoothness * uSmoothness);
    float specStrength = mix(0.04, 1.0, uMetallic) + uReflectivity;

    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= uNumLights) break;
        int t = uLightType[i];
        vec3 L;
        float atten = 1.0;
        if (t == 2) {
            L = normalize(-uLightDir[i]);
        } else {
            vec3 d = uLightPos[i] - vWorld;
            float dist = max(length(d), 1e-4);
            L = d / dist;
            float r = max(uLightRange[i], 1e-3);
            if (t == 0 && uLightStrictAttenuation[i] != 0) {
                atten = dist < r ? 1.0 : 0.0;
            } else {
                atten = clamp(1.0 - dist / r, 0.0, 1.0);
                atten *= atten;
            }
            if (t == 1) {
                float ct = dot(-L, normalize(uLightDir[i]));
                float co = uLightAngles[i].x;
                float ci = uLightAngles[i].y;
                float spot = clamp((ct - co) / max(ci - co, 1e-4), 0.0, 1.0);
                atten *= spot;
            }
        }
        float ndl = max(dot(N, L), 0.0);
        col += diffuseColor * uLightColor[i] * ndl * atten;

        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), shininess) * ndl * atten;
        col += uLightColor[i] * spec * specStrength;
    }

    // Subtle environment sheen so smoothness/reflectivity are visible even without lights.
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
    col += vec3(fresnel * (0.04 + uReflectivity) * (0.35 + 0.65 * uSmoothness) * 0.6);

    FragColor = vec4(col, 1.0);
}
)";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s, sizeof(log), nullptr, log); std::fprintf(stderr, "Shader error:\n%s\n", log); }
    return s;
}

bool Renderer::init() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, kVert);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFrag);
    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);
    GLint ok = 0; glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(program_, sizeof(log), nullptr, log); std::fprintf(stderr, "Link error:\n%s\n", log); }
    glDeleteShader(vs); glDeleteShader(fs);

    ensureMesh("cube",   makeCube());
    ensureMesh("sphere", makeSphere());
    ensureMesh("quad",   makeQuad());
    ensureMesh("$gizmo_cone",     makeCone());
    ensureMesh("$gizmo_cylinder", makeCylinder());

    glGenSamplers(4, &textureSamplers_[0][0]);
    for (int wrap = 0; wrap < 2; ++wrap) {
        for (int filter = 0; filter < 2; ++filter) {
            const GLuint sampler = textureSamplers_[wrap][filter];
            const GLint addressMode = wrap == 0 ? GL_REPEAT : GL_CLAMP_TO_EDGE;
            const GLint minFilter = filter == 0 ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST;
            const GLint magFilter = filter == 0 ? GL_LINEAR : GL_NEAREST;
            glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, addressMode);
            glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, addressMode);
            glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, minFilter);
            glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, magFilter);
        }
    }

    glGenVertexArrays(1, &vaoLine_);
    glGenBuffers(1, &vboLine_);
    glBindVertexArray(vaoLine_);
    glBindBuffer(GL_ARRAY_BUFFER, vboLine_);
    glBufferData(GL_ARRAY_BUFFER, 4096 * 8 * sizeof(float), nullptr, GL_STREAM_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6*sizeof(float)));
    glBindVertexArray(0);

    uploadGrid();
    return ok != 0;
}

void Renderer::uploadMesh(const MeshData& m, GPUMesh& mesh) {
    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glGenBuffers(1, &mesh.ibo);
    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, m.vertices.size()*sizeof(float), m.vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.indices.size()*sizeof(uint32_t), m.indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6*sizeof(float)));
    glBindVertexArray(0);
    mesh.indexCount = (GLuint)m.indices.size();
}

bool Renderer::ensureMesh(const std::string& id, const MeshData& data) {
    if (meshes_.count(id)) return true;
    GPUMesh m;
    uploadMesh(data, m);
    meshes_[id] = m;
    return true;
}

bool Renderer::loadModelOBJ(const std::string& id, const std::string& path, bool raw) {
    MeshData data;
    if (!loadOBJ(path, data, raw)) { std::fprintf(stderr, "[model] failed: %s\n", path.c_str()); return false; }
    deleteFBXModel(id);
    if (auto it = meshes_.find(id); it != meshes_.end() && id != "cube" && id != "sphere" && id != "quad") {
        glDeleteVertexArrays(1, &it->second.vao);
        glDeleteBuffers(1, &it->second.vbo);
        glDeleteBuffers(1, &it->second.ibo);
        meshes_.erase(it);
    }
    GPUMesh m;
    uploadMesh(data, m);
    meshes_[id] = m;
    return true;
}

bool Renderer::loadModelFBX(const std::string& id, const std::string& path) {
    FBXModelData data;
    std::string error;
    if (!loadFBX(path, data, &error)) {
        std::fprintf(stderr, "[fbx] failed to load %s: %s\n", path.c_str(), error.c_str());
        return false;
    }

    deleteFBXModel(id);
    if (auto it = meshes_.find(id); it != meshes_.end()) {
        glDeleteVertexArrays(1, &it->second.vao);
        glDeleteBuffers(1, &it->second.vbo);
        glDeleteBuffers(1, &it->second.ibo);
        meshes_.erase(it);
    }

    std::vector<FBXGPUMeshPart> parts;
    parts.reserve(data.parts.size());
    for (const FBXMeshPart& source : data.parts) {
        FBXGPUMeshPart part;
        part.materialIndex = source.materialIndex;
        uploadMesh(source.mesh, part.mesh);
        parts.push_back(part);
    }
    fbxMeshes_[id] = std::move(parts);
    fbxMaterials_[id] = std::move(data.materials);
    return true;
}

const std::vector<FBXMaterialInfo>* Renderer::fbxMaterials(const std::string& id) const {
    const auto it = fbxMaterials_.find(id);
    return it == fbxMaterials_.end() ? nullptr : &it->second;
}

void Renderer::deleteFBXModel(const std::string& id) {
    const auto it = fbxMeshes_.find(id);
    if (it != fbxMeshes_.end()) {
        for (const FBXGPUMeshPart& part : it->second) {
            glDeleteVertexArrays(1, &part.mesh.vao);
            glDeleteBuffers(1, &part.mesh.vbo);
            glDeleteBuffers(1, &part.mesh.ibo);
        }
        fbxMeshes_.erase(it);
    }
    fbxMaterials_.erase(id);
}

bool Renderer::loadTextureFile(const std::string& id, const std::string& path) {
    Texture t;
    if (!::loadTextureFile(path, t)) {           // <-- :: forces the free function
        std::fprintf(stderr, "[texture] failed: %s\n", path.c_str());
        return false;
    }
    if (auto it = textures_.find(id); it != textures_.end()) {
        if (it->second.id) glDeleteTextures(1, &it->second.id);
        textures_.erase(it);
    }
    textures_[id] = t;
    return true;
}

void Renderer::reloadAssets(const std::unordered_map<std::string, ModelEntry>& models,
                            const std::unordered_map<std::string, std::string>& textures) {
    for (auto it = fbxMeshes_.begin(); it != fbxMeshes_.end(); ) {
        const std::string id = it->first;
        ++it;
        deleteFBXModel(id);
    }
    for (auto it = meshes_.begin(); it != meshes_.end(); ) {
        const auto& id = it->first;
        if (id == "cube" || id == "sphere" || id == "quad" || (!id.empty() && id[0] == '$')) { ++it; continue; }
        glDeleteVertexArrays(1, &it->second.vao);
        glDeleteBuffers(1, &it->second.vbo);
        glDeleteBuffers(1, &it->second.ibo);
        it = meshes_.erase(it);
    }
    for (auto& [k, v] : textures_) if (v.id) glDeleteTextures(1, &v.id);
    textures_.clear();

    for (auto& [id, e] : models) {
        std::string extension = std::filesystem::path(e.path).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".fbx") loadModelFBX(id, e.path);
        else loadModelOBJ(id, e.path, e.raw);
    }
    for (auto& [id, p] : textures) loadTextureFile(id, p);
}

void Renderer::uploadGrid() {
    std::vector<float> v;
    const float E = 10.0f, S = 1.0f;
    for (float x = -E; x <= E; x += S) {
        v.insert(v.end(), { x, 0, -E, 0,1,0, 0,0 });
        v.insert(v.end(), { x, 0,  E, 0,1,0, 0,0 });
        v.insert(v.end(), { -E, 0, x, 0,1,0, 0,0 });
        v.insert(v.end(), {  E, 0, x, 0,1,0, 0,0 });
    }
    glGenVertexArrays(1, &vaoGrid_);
    glGenBuffers(1, &vboGrid_);
    glBindVertexArray(vaoGrid_);
    glBindBuffer(GL_ARRAY_BUFFER, vboGrid_);
    glBufferData(GL_ARRAY_BUFFER, v.size()*sizeof(float), v.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6*sizeof(float)));
    glBindVertexArray(0);
    gridVerts_ = (GLuint)(v.size() / 8);
}

void Renderer::shutdown() {
    if (program_) glDeleteProgram(program_);
    for (auto& [k, m] : meshes_) {
        glDeleteVertexArrays(1, &m.vao);
        glDeleteBuffers(1, &m.vbo);
        glDeleteBuffers(1, &m.ibo);
    }
    meshes_.clear();
    for (auto& [id, parts] : fbxMeshes_) {
        for (const FBXGPUMeshPart& part : parts) {
            glDeleteVertexArrays(1, &part.mesh.vao);
            glDeleteBuffers(1, &part.mesh.vbo);
            glDeleteBuffers(1, &part.mesh.ibo);
        }
    }
    fbxMeshes_.clear();
    fbxMaterials_.clear();
    for (auto& [k, t] : textures_) if (t.id) glDeleteTextures(1, &t.id);
    textures_.clear();
    glDeleteVertexArrays(1, &vaoLine_); glDeleteBuffers(1, &vboLine_);
    glDeleteVertexArrays(1, &vaoGrid_); glDeleteBuffers(1, &vboGrid_);
    glDeleteSamplers(4, &textureSamplers_[0][0]);
    if (fbo_)      glDeleteFramebuffers(1, &fbo_);
    if (fboTex_)   glDeleteTextures(1, &fboTex_);
    if (fboDepth_) glDeleteRenderbuffers(1, &fboDepth_);
}

void Renderer::resizeFBO(int w, int h) {
    if (w <= 0 || h <= 0) return;
    if (w == fboW_ && h == fboH_ && fbo_) return;
    if (fbo_)      { glDeleteFramebuffers(1, &fbo_);   fbo_ = 0; }
    if (fboTex_)   { glDeleteTextures(1, &fboTex_);    fboTex_ = 0; }
    if (fboDepth_) { glDeleteRenderbuffers(1, &fboDepth_); fboDepth_ = 0; }

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

    glGenTextures(1, &fboTex_);
    glBindTexture(GL_TEXTURE_2D, fboTex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTex_, 0);

    glGenRenderbuffers(1, &fboDepth_);
    glBindRenderbuffer(GL_RENDERBUFFER, fboDepth_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fboDepth_);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    fboW_ = w; fboH_ = h;
}

void Renderer::bindFBO()   { glBindFramebuffer(GL_FRAMEBUFFER, fbo_); glViewport(0, 0, fboW_, fboH_); }
void Renderer::unbindFBO() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

void Renderer::uploadLightUniforms() {
    int n = std::min((int)lights_.size(), 32);
    glUniform1i(glGetUniformLocation(program_, "uNumLights"), n);
    if (n == 0) return;

    std::vector<int>   types(n);
    std::vector<int>   strictAttenuation(n);
    std::vector<float> pos(n*3), dir(n*3), color(n*3), range(n), angles(n*2);
    const float D2R = 3.14159265f / 180.0f;
    for (int i = 0; i < n; ++i) {
        const Light& l = lights_[i];
        types[i] = (l.type == LightType::Sun) ? 2 : (l.type == LightType::Spot) ? 1 : 0;
        strictAttenuation[i] = l.type == LightType::Point && l.strictAttenuation ? 1 : 0;
        pos[i*3+0] = l.pos.x; pos[i*3+1] = l.pos.y; pos[i*3+2] = l.pos.z;
        dir[i*3+0] = l.dir.x; dir[i*3+1] = l.dir.y; dir[i*3+2] = l.dir.z;
        const float intensity = std::max(0.0f, l.intensity);
        color[i*3+0] = l.color.x * intensity;
        color[i*3+1] = l.color.y * intensity;
        color[i*3+2] = l.color.z * intensity;
        range[i] = l.range;
        angles[i*2+0] = std::cos(l.outerDeg * D2R);
        angles[i*2+1] = std::cos(l.innerDeg * D2R);
    }
    glUniform1iv(glGetUniformLocation(program_, "uLightType"),  n, types.data());
    glUniform3fv(glGetUniformLocation(program_, "uLightPos"),   n, pos.data());
    glUniform3fv(glGetUniformLocation(program_, "uLightDir"),   n, dir.data());
    glUniform3fv(glGetUniformLocation(program_, "uLightColor"), n, color.data());
    glUniform1fv(glGetUniformLocation(program_, "uLightRange"), n, range.data());
    glUniform1iv(glGetUniformLocation(program_, "uLightStrictAttenuation"), n, strictAttenuation.data());
    glUniform2fv(glGetUniformLocation(program_, "uLightAngles"), n, angles.data());
}

void Renderer::beginFrame(const Mat4& view, const Mat4& proj, const Vec3& camPos) {
    view_ = view; proj_ = proj; camPos_ = camPos;
    glClearColor(0.16f, 0.17f, 0.20f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(program_);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uView"), 1, GL_FALSE, view_.m);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uProj"), 1, GL_FALSE, proj_.m);
    glUniform3f(glGetUniformLocation(program_, "uCamPos"), camPos_.x, camPos_.y, camPos_.z);
    glUniform1i(glGetUniformLocation(program_, "uUnlit"), 0);
    uploadLightUniforms();
}

void Renderer::drawMesh(const std::string& meshId, const Mat4& model, const Vec3& color,
                        const MaterialParams& material, float uvScale,
                        TextureWrapMode wrapMode, TextureFilterMode filterMode) {
    auto it = meshes_.find(meshId);
    if (it == meshes_.end()) return;
    auto& m = it->second;

    glUniform1i(glGetUniformLocation(program_, "uUnlit"), 0);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, model.m);
    glUniform3f(glGetUniformLocation(program_, "uColor"), color.x, color.y, color.z);
    glUniform2f(glGetUniformLocation(program_, "uUVScale"), uvScale, uvScale);
    glUniform1f(glGetUniformLocation(program_, "uMetallic"), material.metallic);
    glUniform1f(glGetUniformLocation(program_, "uSmoothness"), material.smoothness);
    glUniform1f(glGetUniformLocation(program_, "uReflectivity"), material.reflectivity);

    int hasTex = 0;
    GLuint glTexId = 0;
    if (!material.texId.empty()) {
        auto tit = textures_.find(material.texId);
        if (tit != textures_.end() && tit->second.valid()) { hasTex = 1; glTexId = tit->second.id; }
    }
    int hasNormalTex = 0;
    GLuint glNormalTexId = 0;
    if (!material.normalTexId.empty()) {
        auto nit = textures_.find(material.normalTexId);
        if (nit != textures_.end() && nit->second.valid()) { hasNormalTex = 1; glNormalTexId = nit->second.id; }
    }

    const GLuint sampler = textureSampler(wrapMode, filterMode);
    glUniform1i(glGetUniformLocation(program_, "uHasTex"), hasTex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, glTexId);
    glBindSampler(0, sampler);
    glUniform1i(glGetUniformLocation(program_, "uTex"), 0);

    glUniform1i(glGetUniformLocation(program_, "uHasNormalTex"), hasNormalTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, glNormalTexId);
    glBindSampler(1, sampler);
    glUniform1i(glGetUniformLocation(program_, "uNormalTex"), 1);
    glActiveTexture(GL_TEXTURE0);

    glBindVertexArray(m.vao);
    glDrawElements(GL_TRIANGLES, m.indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glBindSampler(0, 0);
    glBindSampler(1, 0);
}

void Renderer::drawFBXModel(const std::string& modelId, const Mat4& model, const Vec3& color,
                            const std::vector<std::string>& materialTexIds, float uvScale,
                            TextureWrapMode wrapMode, TextureFilterMode filterMode,
                            const MaterialParams* materialOverride) {
    const auto modelIt = fbxMeshes_.find(modelId);
    if (modelIt == fbxMeshes_.end()) return;

    glUniform1i(glGetUniformLocation(program_, "uUnlit"), 0);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, model.m);
    glUniform3f(glGetUniformLocation(program_, "uColor"), color.x, color.y, color.z);
    glUniform2f(glGetUniformLocation(program_, "uUVScale"), uvScale, uvScale);
    glUniform1i(glGetUniformLocation(program_, "uTex"), 0);
    glUniform1i(glGetUniformLocation(program_, "uNormalTex"), 1);
    glUniform1f(glGetUniformLocation(program_, "uMetallic"), materialOverride ? materialOverride->metallic : 0.0f);
    glUniform1f(glGetUniformLocation(program_, "uSmoothness"), materialOverride ? materialOverride->smoothness : 0.5f);
    glUniform1f(glGetUniformLocation(program_, "uReflectivity"), materialOverride ? materialOverride->reflectivity : 0.0f);

    const GLuint sampler = textureSampler(wrapMode, filterMode);
    glBindSampler(0, sampler);
    glBindSampler(1, sampler);

    GLuint normalTexId = 0;
    if (materialOverride && !materialOverride->normalTexId.empty()) {
        const auto normalIt = textures_.find(materialOverride->normalTexId);
        if (normalIt != textures_.end() && normalIt->second.valid()) normalTexId = normalIt->second.id;
    }
    glUniform1i(glGetUniformLocation(program_, "uHasNormalTex"), normalTexId != 0 ? 1 : 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normalTexId);
    glActiveTexture(GL_TEXTURE0);

    for (const FBXGPUMeshPart& part : modelIt->second) {
        GLuint textureId = 0;
        std::string texId;
        if (materialOverride && !materialOverride->texId.empty()) texId = materialOverride->texId;
        else if (part.materialIndex < materialTexIds.size()) texId = materialTexIds[part.materialIndex];
        if (!texId.empty()) {
            const auto textureIt = textures_.find(texId);
            if (textureIt != textures_.end() && textureIt->second.valid()) textureId = textureIt->second.id;
        }
        glUniform1i(glGetUniformLocation(program_, "uHasTex"), textureId != 0 ? 1 : 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glBindVertexArray(part.mesh.vao);
        glDrawElements(GL_TRIANGLES, part.mesh.indexCount, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
    glBindSampler(0, 0);
    glBindSampler(1, 0);
}

GLuint Renderer::textureSampler(TextureWrapMode wrapMode, TextureFilterMode filterMode) const {
    const int wrapIndex = wrapMode == TextureWrapMode::Clamp ? 1 : 0;
    const int filterIndex = filterMode == TextureFilterMode::Nearest ? 1 : 0;
    return textureSamplers_[wrapIndex][filterIndex];
}

void Renderer::drawGrid() {
    glUniform1i(glGetUniformLocation(program_, "uUnlit"), 1);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, Mat4::identity().m);
    glUniform3f(glGetUniformLocation(program_, "uColor"), 0.30f, 0.32f, 0.36f);
    glBindVertexArray(vaoGrid_);
    glDrawArrays(GL_LINES, 0, gridVerts_);
    glBindVertexArray(0);
}

void Renderer::drawLineSegments(const std::vector<float>& verts, const Vec3& color) {
    if (verts.empty()) return;
    glUniform1i(glGetUniformLocation(program_, "uUnlit"), 1);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, Mat4::identity().m);
    glUniform3f(glGetUniformLocation(program_, "uColor"), color.x, color.y, color.z);
    glBindVertexArray(vaoLine_);
    glBindBuffer(GL_ARRAY_BUFFER, vboLine_);
    glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(float), verts.data(), GL_STREAM_DRAW);
    glLineWidth(2.0f);
    glDrawArrays(GL_LINES, 0, (GLsizei)(verts.size() / 8));
    glLineWidth(1.0f);
    glBindVertexArray(0);
}

void Renderer::drawWireAABB(const Vec3& mn, const Vec3& mx, const Vec3& color) {
    Vec3 c[8] = {
        {mn.x,mn.y,mn.z},{mx.x,mn.y,mn.z},{mx.x,mn.y,mx.z},{mn.x,mn.y,mx.z},
        {mn.x,mx.y,mn.z},{mx.x,mx.y,mn.z},{mx.x,mx.y,mx.z},{mn.x,mx.y,mx.z}
    };
    int e[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    std::vector<float> v;
    for (auto& ed : e) for (int k = 0; k < 2; ++k) {
        v.push_back(c[ed[k]].x); v.push_back(c[ed[k]].y); v.push_back(c[ed[k]].z);
        v.push_back(0); v.push_back(1); v.push_back(0);
        v.push_back(0); v.push_back(0);
    }
    drawLineSegments(v, color);
}

void Renderer::drawWireSphere(const Vec3& center, float radius, const Vec3& color, int segs) {
    std::vector<float> v;
    for (int i = 0; i < segs; ++i) {
        float a0 = 6.2831853f * i / segs;
        float a1 = 6.2831853f * (i+1) / segs;
        float c0 = std::cos(a0), s0 = std::sin(a0);
        float c1 = std::cos(a1), s1 = std::sin(a1);
        // XZ
        v.insert(v.end(), {center.x+radius*c0, center.y, center.z+radius*s0, 0,1,0, 0,0});
        v.insert(v.end(), {center.x+radius*c1, center.y, center.z+radius*s1, 0,1,0, 0,0});
        // XY
        v.insert(v.end(), {center.x+radius*c0, center.y+radius*s0, center.z, 0,1,0, 0,0});
        v.insert(v.end(), {center.x+radius*c1, center.y+radius*s1, center.z, 0,1,0, 0,0});
        // YZ
        v.insert(v.end(), {center.x, center.y+radius*c0, center.z+radius*s0, 0,1,0, 0,0});
        v.insert(v.end(), {center.x, center.y+radius*c1, center.z+radius*s1, 0,1,0, 0,0});
    }
    drawLineSegments(v, color);
}

void Renderer::drawMeshUnlit(const std::string& meshId, const Mat4& model, const Vec3& color) {
    const auto it = meshes_.find(meshId);
    if (it == meshes_.end()) return;

    glUniform1i(glGetUniformLocation(program_, "uUnlit"), 1);
    glUniformMatrix4fv(glGetUniformLocation(program_, "uModel"), 1, GL_FALSE, model.m);
    glUniform3f(glGetUniformLocation(program_, "uColor"), color.x, color.y, color.z);
    glBindVertexArray(it->second.vao);
    glDrawElements(GL_TRIANGLES, it->second.indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Renderer::setDepthTest(bool enabled) {
    if (enabled) glEnable(GL_DEPTH_TEST);
    else         glDisable(GL_DEPTH_TEST);
}