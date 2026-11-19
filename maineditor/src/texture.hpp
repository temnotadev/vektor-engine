#pragma once
#include "gl.hpp"
#include <string>

struct Texture {
    GLuint id = 0;
    int width = 0, height = 0;
    bool valid() const { return id != 0; }
};

bool loadTextureFile(const std::string& path, Texture& out);