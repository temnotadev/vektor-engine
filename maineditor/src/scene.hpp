#pragma once
#include "glmath.hpp"
#include <map>
#include <string>
#include <vector>
#include <unordered_map>

enum class ObjType { Cube = 0, Sphere = 1, Quad = 2 };
enum class LightType { Point = 0, Spot = 1, Sun = 2 };
enum class TextureWrapMode { Repeat = 0, Clamp = 1 };
enum class TextureFilterMode { Linear = 0, Nearest = 1 };

inline const char* objTypeName(ObjType t) {
    switch (t) {
        case ObjType::Sphere: return "sphere";
        case ObjType::Quad:   return "quad";
        default:              return "cube";
    }
}
inline ObjType objTypeFromName(const std::string& s) {
    if (s == "sphere") return ObjType::Sphere;
    if (s == "quad")   return ObjType::Quad;
    return ObjType::Cube;
}

// Mirrors the engine's Material in world.hpp:
//   material <id> [color r g b] [texture id|none] [normal id|none]
//                 [metallic v] [smoothness v] [reflectivity v] [ssr on|off]
struct Material {
    std::string name;            // registry id
    Vec3  color{1, 1, 1};
    std::string texId;           // albedo texture registry id
    std::string normalTexId;     // normal map texture registry id
    float metallic = 0.0f;
    float smoothness = 0.5f;
    float reflectivity = 0.0f;
    bool  ssr = false;
};

struct SceneObject {
    std::string name;
    ObjType     type = ObjType::Cube;
    bool        collisionOnly = false;
    bool        physicsEnabled = false;
    bool        physicsDynamic = false;
    std::string modelId;     // if non-empty, uses model mesh instead of primitive
    std::string materialId;  // material registry id (see Material)
    std::vector<std::string> materialTexIds;
    TextureWrapMode textureWrap = TextureWrapMode::Repeat;
    TextureFilterMode textureFilter = TextureFilterMode::Linear;
    Vec3  pos{0,0,0};
    Vec3  size{1,1,1};
    Vec3  rot{0,0,0};
    Vec3  color{0.8f, 0.8f, 0.8f};
    float uvScale = 1.0f;
    float mass = 1.0f;
    float friction = 0.2f;
    float restitution = 0.0f;
};

struct Light {
    std::string name;
    LightType   type = LightType::Point;
    Vec3  pos{0, 2, 0};
    Vec3  dir{0, -1, 0};
    Vec3  color{1, 1, 1};
    float intensity = 1.0f;   // brightness multiplier applied to color
    float range    = 10.0f;
    float outerDeg = 45.0f;
    float innerDeg = 20.0f;
    bool strictAttenuation = false;
};

struct ModelEntry {
    std::string path;
    bool raw = false;
};

struct ScriptEvent {
    std::string type;
    std::string key;
    std::vector<std::string> commands;
};

struct Scene {
    std::vector<SceneObject> objects;
    std::vector<Light>       lights;
    std::vector<ScriptEvent> scriptEvents;
    std::vector<std::string> scriptGlobals;
    std::unordered_map<std::string, float> scriptVariables;
    std::unordered_map<std::string, std::string> textures; // id -> path
    std::map<std::string, Material> materials;             // id -> material
    std::unordered_map<std::string, ModelEntry>  models;   // id -> path + raw
    bool playerController = false;
    bool playerGravity = true;

    std::string makeUniqueName(const std::string& prefix) const;
    std::string makeUniqueMaterialName(const std::string& prefix) const;

    int addObject(ObjType type, Vec3 pos, Vec3 size, Vec3 color,
                  const std::string& modelId = "");
    int addPointLight(Vec3 pos, Vec3 color, float range);
    int addSpotLight(Vec3 pos, Vec3 dir, Vec3 color, float range,
                     float outerDeg, float innerDeg);

    void removeObject(int idx);
    void removeLight(int idx);

    std::string generateScript() const;
    void        loadScript(const std::string& text);
    void        runKeyEvent(const std::string& key);
    void        runFrameEvents(float dt);
};

inline Mat4 modelMatrix(const SceneObject& o) {
    const float d2r = 3.14159265f / 180.0f;
    return Mat4::translate(o.pos)
         * Mat4::rotateY(o.rot.y*d2r) * Mat4::rotateX(o.rot.x*d2r) * Mat4::rotateZ(o.rot.z*d2r)
         * Mat4::scale(o.size);
}