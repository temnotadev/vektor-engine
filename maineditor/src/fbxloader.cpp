#include "fbxloader.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cfloat>
#include <filesystem>

static void appendNodeMeshes(const aiNode* node, const aiMatrix4x4& parentTransform,
                             const aiScene* scene, FBXModelData& out) {
    const aiMatrix4x4 transform = parentTransform * node->mTransformation;
    aiMatrix3x3 normalTransform(transform);
    normalTransform.Inverse().Transpose();

    for (unsigned int nodeMesh = 0; nodeMesh < node->mNumMeshes; ++nodeMesh) {
        const aiMesh* source = scene->mMeshes[node->mMeshes[nodeMesh]];
        if (!source->HasPositions()) continue;

        FBXMeshPart part;
        part.materialIndex = source->mMaterialIndex;
        part.mesh.vertices.reserve(source->mNumVertices * 8);
        for (unsigned int i = 0; i < source->mNumVertices; ++i) {
            const aiVector3D position = transform * source->mVertices[i];
            aiVector3D normal = source->HasNormals()
                ? normalTransform * source->mNormals[i]
                : aiVector3D(0.0f, 1.0f, 0.0f);
            normal.Normalize();

            const aiVector3D uv = source->HasTextureCoords(0)
                ? source->mTextureCoords[0][i]
                : aiVector3D(0.0f, 0.0f, 0.0f);
            part.mesh.vertices.insert(part.mesh.vertices.end(), {
                position.x, position.y, position.z,
                normal.x, normal.y, normal.z,
                uv.x, uv.y
            });
        }

        for (unsigned int faceIndex = 0; faceIndex < source->mNumFaces; ++faceIndex) {
            const aiFace& face = source->mFaces[faceIndex];
            for (unsigned int index = 0; index < face.mNumIndices; ++index)
                part.mesh.indices.push_back(face.mIndices[index]);
        }

        if (!part.mesh.indices.empty()) out.parts.push_back(std::move(part));
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
        appendNodeMeshes(node->mChildren[i], transform, scene, out);
}

bool loadFBX(const std::string& path, FBXModelData& out, std::string* error) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality |
        aiProcess_ValidateDataStructure);
    if (!scene || !scene->mRootNode || !scene->HasMeshes()) {
        if (error) *error = scene ? "FBX scene has no mesh geometry." : importer.GetErrorString();
        return false;
    }

    FBXModelData loaded;
    const std::filesystem::path modelDirectory = std::filesystem::path(path).parent_path();
    loaded.materials.reserve(scene->mNumMaterials);
    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        const aiMaterial* source = scene->mMaterials[i];
        aiString materialName;
        source->Get(AI_MATKEY_NAME, materialName);

        FBXMaterialInfo material;
        material.name = materialName.length ? materialName.C_Str() : "Material " + std::to_string(i + 1);
        aiString texturePath;
        if (source->GetTexture(aiTextureType_BASE_COLOR, 0, &texturePath) != AI_SUCCESS)
            source->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath);
        if (texturePath.length && texturePath.C_Str()[0] != '*') {
            std::filesystem::path resolvedPath(texturePath.C_Str());
            if (resolvedPath.is_relative()) resolvedPath = modelDirectory / resolvedPath;
            material.diffuseTexturePath = resolvedPath.lexically_normal().string();
        }
        loaded.materials.push_back(std::move(material));
    }

    appendNodeMeshes(scene->mRootNode, aiMatrix4x4(), scene, loaded);
    if (loaded.parts.empty()) {
        if (error) *error = "FBX scene contains no renderable triangle meshes.";
        return false;
    }
    if (loaded.materials.empty()) loaded.materials.push_back({"Material 1", {}});

    aiVector3D minPoint(FLT_MAX, FLT_MAX, FLT_MAX);
    aiVector3D maxPoint(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const FBXMeshPart& part : loaded.parts) {
        for (size_t i = 0; i + 7 < part.mesh.vertices.size(); i += 8) {
            minPoint.x = std::min(minPoint.x, part.mesh.vertices[i]);
            minPoint.y = std::min(minPoint.y, part.mesh.vertices[i + 1]);
            minPoint.z = std::min(minPoint.z, part.mesh.vertices[i + 2]);
            maxPoint.x = std::max(maxPoint.x, part.mesh.vertices[i]);
            maxPoint.y = std::max(maxPoint.y, part.mesh.vertices[i + 1]);
            maxPoint.z = std::max(maxPoint.z, part.mesh.vertices[i + 2]);
        }
    }
    const aiVector3D center = (minPoint + maxPoint) * 0.5f;
    const float largestDimension = std::max({
        maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z
    });
    const float scale = largestDimension > 1e-6f ? 1.0f / largestDimension : 1.0f;
    for (FBXMeshPart& part : loaded.parts) {
        for (size_t i = 0; i + 7 < part.mesh.vertices.size(); i += 8) {
            part.mesh.vertices[i] = (part.mesh.vertices[i] - center.x) * scale;
            part.mesh.vertices[i + 1] = (part.mesh.vertices[i + 1] - center.y) * scale;
            part.mesh.vertices[i + 2] = (part.mesh.vertices[i + 2] - center.z) * scale;
        }
    }

    out = std::move(loaded);
    return true;
}
