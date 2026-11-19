#pragma once

#include "mesh.hpp"
#include <string>
#include <vector>

struct FBXMaterialInfo {
    std::string name;
    std::string diffuseTexturePath;
};

struct FBXMeshPart {
    MeshData mesh;
    unsigned int materialIndex = 0;
};

struct FBXModelData {
    std::vector<FBXMaterialInfo> materials;
    std::vector<FBXMeshPart> parts;
};

bool loadFBX(const std::string& path, FBXModelData& out, std::string* error = nullptr);
