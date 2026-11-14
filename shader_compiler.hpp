#pragma once
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct CompileResult { bool ok = false; std::vector<uint32_t> spv; std::string log; };

inline std::string glslcPath() {
    if (const char* sdk = std::getenv("VULKAN_SDK")) {
        auto p = std::filesystem::path(sdk) / "Bin" / "glslc.exe";
        if (std::filesystem::exists(p)) return p.string();
    }
    return "glslc";   // fall back to PATH
}

// workDir should be next to the exe: a temp dir can contain non-ASCII user names, which std::system mangles.
inline CompileResult compileGlsl(const std::filesystem::path& src, const std::filesystem::path& includeDir,
                                 const std::filesystem::path& workDir, const std::string& id) {
    namespace fs = std::filesystem;
    CompileResult r;
    std::error_code ec;
    fs::create_directories(workDir, ec);
    fs::path spv = workDir / (id + ".spv"), log = workDir / (id + ".log");
    fs::remove(spv, ec);

    std::string inner = "\"" + glslcPath() + "\" --target-env=vulkan1.2 -fshader-stage=compute -O"
                      + " -I \"" + includeDir.string() + "\" \"" + src.string() + "\""
                      + " -o \"" + spv.string() + "\" 2> \"" + log.string() + "\"";
#ifdef _WIN32
    std::string cmd = "\"" + inner + "\"";    // cmd.exe strips one outer pair of quotes
#else
    std::string cmd = inner;
#endif
    int rc = std::system(cmd.c_str());

    { std::ifstream lf(log); std::stringstream ss; ss << lf.rdbuf(); r.log = ss.str(); }
    std::ifstream f(spv, std::ios::binary | std::ios::ate);
    if (rc != 0 || !f) return r;
    size_t sz = (size_t)f.tellg();
    if (sz == 0 || sz % 4) return r;
    r.spv.resize(sz / 4);
    f.seekg(0);
    f.read((char*)r.spv.data(), (std::streamsize)sz);
    r.ok = true;
    return r;
}