#pragma once
#include "mesh.hpp"
#include <string>

bool loadOBJ(const std::string& path, MeshData& out, bool raw = false);