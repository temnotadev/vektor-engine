#pragma once
#include "gl.hpp"
#include "glmath.hpp"
#include "mesh.hpp"
#include "scene.hpp"
#include "texture.hpp"
#include "fbxloader.hpp"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct GPUMesh {
    GLuint vao = 0, vbo = 0, ibo = 0;
    GLuint indexCount = 0;
};

struct FBXGPUMeshPart {
    GPUMesh mesh;
    unsigned int materialIndex = 0;
};

// PBR-ish parameters passed to the preview shader.
struct MaterialParams {
    std::string texId;
    std::string normalTexId;
    float metallic = 0.0f;
    float smoothness = 0.5f;
    float reflectivity = 0.0f;
};

class Renderer {
public:
    bool init();
    void shutdown();

    void resizeFBO(int w, int h);
    void bindFBO();
    void unbindFBO();
    GLuint fboColorTex() const { return fboTex_; }
    bool   fboReady()   const { return fbo_ != 0 && fboTex_ != 0; }

    bool ensureMesh(const std::string& id, const MeshData& data);
    bool loadModelOBJ(const std::string& id, const std::string& path, bool raw);
    bool loadModelFBX(const std::string& id, const std::string& path);
    bool loadTextureFile(const std::string& id, const std::string& path);
    void reloadAssets(const std::unordered_map<std::string, ModelEntry>&  models,
                      const std::unordered_map<std::string, std::string>& textures);

    bool hasMesh(const std::string& id) const { return meshes_.count(id) > 0; }
    bool hasFBXModel(const std::string& id) const { return fbxMeshes_.count(id) > 0; }
    bool meshBounds(const std::string& id, Vec3& mn, Vec3& mx) const;
    const std::vector<FBXMaterialInfo>* fbxMaterials(const std::string& id) const;
    bool hasTexture(const std::string& id) const { return textures_.count(id) > 0; }
    const std::unordered_map<std::string, GPUMesh>& meshes()   const { return meshes_; }
    const std::unordered_map<std::string, Texture>& textures() const { return textures_; }

    void setLights(const std::vector<Light>& lights) { lights_ = lights; }

    void beginFrame(const Mat4& view, const Mat4& proj, const Vec3& camPos);
    void drawMesh(const std::string& meshId, const Mat4& model, const Vec3& color,
                  const MaterialParams& material, float uvScale,
                  TextureWrapMode wrapMode, TextureFilterMode filterMode);
    void drawFBXModel(const std::string& modelId, const Mat4& model, const Vec3& color,
                      const std::vector<std::string>& materialTexIds, float uvScale,
                      TextureWrapMode wrapMode, TextureFilterMode filterMode,
                      const MaterialParams* materialOverride = nullptr);
    void drawGrid();
    void drawLineSegments(const std::vector<float>& verts, const Vec3& color);
    void drawWireAABB(const Vec3& mn, const Vec3& mx, const Vec3& color);
    void drawWireSphere(const Vec3& center, float radius, const Vec3& color, int segs = 16);
    void drawMeshUnlit(const std::string& meshId, const Mat4& model, const Vec3& color);
    void setDepthTest(bool enabled);

private:
    GLuint program_ = 0;
    std::unordered_map<std::string, GPUMesh> meshes_;
    std::unordered_map<std::string, std::pair<Vec3, Vec3>> meshBounds_;
    std::unordered_map<std::string, Texture> textures_;
    std::unordered_map<std::string, std::vector<FBXGPUMeshPart>> fbxMeshes_;
    std::unordered_map<std::string, std::vector<FBXMaterialInfo>> fbxMaterials_;

    GLuint vaoLine_ = 0, vboLine_ = 0;
    GLuint vaoGrid_ = 0, vboGrid_ = 0; GLuint gridVerts_ = 0;
    GLuint textureSamplers_[2][2] = {};

    GLuint fbo_ = 0, fboTex_ = 0, fboDepth_ = 0;
    int fboW_ = 0, fboH_ = 0;

    Mat4 view_, proj_;
    Vec3 camPos_;
    std::vector<Light> lights_;

    void uploadMesh(const MeshData& m, GPUMesh& mesh);
    void deleteFBXModel(const std::string& id);
    GLuint textureSampler(TextureWrapMode wrapMode, TextureFilterMode filterMode) const;
    void uploadGrid();
    void uploadLightUniforms();
};