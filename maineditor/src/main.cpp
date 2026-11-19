#include "gl.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "glmath.hpp"
#include "scene.hpp"
#include "renderer.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>
#include <iostream>
#include <cctype>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <dwmapi.h>
#include <objbase.h>
#include <shlobj.h>

#ifndef GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <GLFW/glfw3native.h>
#endif

#include "texture.hpp"
#include "stb_image.h"

// -----------------------------------------------------------------------------
// Asset / file helpers
// -----------------------------------------------------------------------------
bool LoadUIAsset(const std::string& filename, GLuint& outTex) {
    Texture tex;
    std::vector<std::string> paths = {
        "assets/" + filename,
        "../assets/" + filename,
        "../../assets/" + filename
    };
    for (const auto& p : paths) {
        if (loadTextureFile(p, tex)) {
            outTex = tex.id;
            return true;
        }
    }
    return false;
}

static void SetWindowIconFromFile(GLFWwindow* win, const std::string& filename) {
    const std::vector<std::string> paths = {
        "assets/" + filename,
        "../assets/" + filename,
        "../../assets/" + filename
    };
    for (const auto& p : paths) {
        int w = 0, h = 0, n = 0;
        unsigned char* data = stbi_load(p.c_str(), &w, &h, &n, 4);
        if (!data) continue;
        GLFWimage image;
        image.width = w;
        image.height = h;
        image.pixels = data;
        glfwSetWindowIcon(win, 1, &image);
        stbi_image_free(data);
        return;
    }
}

static std::filesystem::path FindAssetsDirectory() {
    const std::vector<std::filesystem::path> paths = {
        "assets",
        "../assets",
        "../../assets"
    };
    for (const auto& path : paths) {
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec) && !ec) {
            std::filesystem::path absolutePath = std::filesystem::absolute(path, ec);
            return ec ? path : absolutePath.lexically_normal();
        }
    }
    return {};
}

std::string OpenFileDialog(const char* filter) {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    CHAR szFile[260] = { 0 };
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = filter;
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn) == TRUE) {
        return std::string(ofn.lpstrFile);
    }
#endif
    return "";
}

// Native folder picker used by the startup project dialog.
static std::string BrowseForFolderDialog(const char* title) {
#ifdef _WIN32
    BROWSEINFOA bi = {};
    bi.lpszTitle = title;
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_EDITBOX;
    LPITEMIDLIST pidl = SHBrowseForFolderA(&bi);
    if (!pidl) return "";
    CHAR path[MAX_PATH] = { 0 };
    std::string result;
    if (SHGetPathFromIDListA(pidl, path)) result = path;
    CoTaskMemFree(pidl);
    return result;
#else
    (void)title;
    return "";
#endif
}

static std::string MakeAssetId(const std::string& value) {
    std::string id;
    id.reserve(value.size());
    for (unsigned char c : value)
        id.push_back(std::isalnum(c) || c == '_' || c == '-' ? static_cast<char>(c) : '_');
    return id.empty() ? "asset" : id;
}

static std::string MakeUniqueTextureId(const Scene& scene, const std::string& requestedId) {
    const std::string base = MakeAssetId(requestedId);
    std::string id = base;
    for (unsigned int suffix = 2; scene.textures.count(id); ++suffix)
        id = base + "_" + std::to_string(suffix);
    return id;
}

static ImGuiKey scriptKeyToImGuiKey(std::string key) {
    for (char& c : key) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (key.size() == 1 && key[0] >= 'A' && key[0] <= 'Z')
        return static_cast<ImGuiKey>(ImGuiKey_A + key[0] - 'A');
    if (key.size() == 1 && key[0] >= '0' && key[0] <= '9')
        return static_cast<ImGuiKey>(ImGuiKey_0 + key[0] - '0');
    return ImGuiKey_None;
}

// ---- screen / plane / AABB helpers ----
static Ray screenRay(float mx, float my, int w, int h, const Mat4& view, const Mat4& proj) {
    float nx = (2.0f * mx) / w - 1.0f;
    float ny = 1.0f - (2.0f * my) / h;
    float tanHalfFov = 1.0f / proj.m[5];
    float aspect     = proj.m[5] / proj.m[0];
    float vx = nx * tanHalfFov * aspect;
    float vy = ny * tanHalfFov;
    float vz = -1.0f;
    Vec3 dir(
        view.m[0]*vx + view.m[4]*vy + view.m[8]*vz,
        view.m[1]*vx + view.m[5]*vy + view.m[9]*vz,
        view.m[2]*vx + view.m[6]*vy + view.m[10]*vz);
    Vec3 pos(
        -(view.m[0]*view.m[12] + view.m[1]*view.m[13] + view.m[2]*view.m[14]),
        -(view.m[4]*view.m[12] + view.m[5]*view.m[13] + view.m[6]*view.m[14]),
        -(view.m[8]*view.m[12] + view.m[9]*view.m[13] + view.m[10]*view.m[14]));
    return { pos, normalize(dir) };
}

static bool rayPlaneY(const Ray& r, float py, Vec3& out) {
    if (std::fabs(r.dir.y) < 1e-6f) return false;
    float t = (py - r.origin.y) / r.dir.y;
    if (t < 0) return false;
    out = r.origin + r.dir * t;
    return true;
}

static bool rayAABB(const Ray& r, const Vec3& mn, const Vec3& mx, float& tOut) {
    float tmin = -1e30f, tmax = 1e30f;
    for (int i = 0; i < 3; ++i) {
        float o  = (i==0)?r.origin.x:(i==1)?r.origin.y:r.origin.z;
        float d  = (i==0)?r.dir.x   :(i==1)?r.dir.y   :r.dir.z;
        float lo = (i==0)?mn.x:(i==1)?mn.y:mn.z;
        float hi = (i==0)?mx.x:(i==1)?mx.y:mx.z;
        if (std::fabs(d) < 1e-8f) { if (o < lo || o > hi) return false; }
        else {
            float t1 = (lo - o) / d, t2 = (hi - o) / d;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    tOut = tmin;
    return tmax >= 0;
}

// ---- UE5-Style Free-Fly Camera ----
struct Camera {
    Vec3 pos{4.0f, 4.0f, 6.0f};
    float yaw = -33.7f;
    float pitch = -25.5f;
    float speed = 8.0f;

    Vec3 forward() const {
        float cy = std::cos(yaw * 3.14159265f / 180.0f);
        float sy = std::sin(yaw * 3.14159265f / 180.0f);
        float cp = std::cos(pitch * 3.14159265f / 180.0f);
        float sp = std::sin(pitch * 3.14159265f / 180.0f);
        return Vec3(sy * cp, sp, -cy * cp);
    }
    Vec3 right() const {
        return normalize(cross(forward(), Vec3(0, 1, 0)));
    }
    Vec3 up() const {
        return normalize(cross(right(), forward()));
    }
    Mat4 view() const {
        return Mat4::lookAt(pos, pos + forward(), Vec3(0, 1, 0));
    }
};

enum class PlaceMode { None, Cube, Sphere, Collider, PointLight, SpotLight };
enum class GizmoMode { Move, Scale };
enum class GizmoAxis { None = -1, X = 0, Y = 1, Z = 2, Center = 3 };

struct GizmoDrag {
    bool      active = false;
    GizmoMode mode = GizmoMode::Move;
    int       axis = -1;              // 0..2 axis, 3 center
    Vec3      startPos{0,0,0};
    Vec3      startSize{1,1,1};
    Vec3      planePoint{0,0,0};
    Vec3      planeNormal{0,0,1};
    Vec3      axisDir{1,0,0};
    Vec3      startHit{0,0,0};
    float     startAxisParam = 0.0f;
    ImVec2    startMouse{0,0};
    bool      isLight = false;
};

struct Editor {
    Scene      scene;
    Camera     camera;
    Renderer   renderer;

    PlaceMode   placeMode = PlaceMode::None;
    GizmoMode   gizmoMode = GizmoMode::Move;
    GizmoDrag   gizmoDrag;
    int         hoverAxis = -1;
    int         selectedObj   = -1;
    int         selectedLight = -1;
    std::string selectedMaterialId;

    std::string scriptText;
    bool        scriptRegen = false;

    void regenerate() { scriptText = scene.generateScript(); scriptRegen = true; }
    // Helper to spawn objects directly in front of the camera
    void spawnAtCenter(PlaceMode mode) {
        Vec3 fwd = camera.forward();
        Vec3 spawnPos = camera.pos + fwd * 10.0f;
        if (fwd.y < -0.01f) {
            float t = -camera.pos.y / fwd.y;
            if (t > 0) spawnPos = camera.pos + fwd * t;
        }

        if (mode == PlaceMode::Cube) {
            spawnPos.y += 0.5f;
            selectedObj = scene.addObject(ObjType::Cube, spawnPos, {1,1,1}, {0.8f,0.6f,0.4f});
            selectedLight = -1;
        } else if (mode == PlaceMode::Sphere) {
            spawnPos.y += 0.5f;
            selectedObj = scene.addObject(ObjType::Sphere, spawnPos, {1,1,1}, {0.4f,0.65f,0.9f});
            selectedLight = -1;
        } else if (mode == PlaceMode::Collider) {
            selectedObj = scene.addObject(ObjType::Cube, spawnPos, {1,1,1}, {0.2f,0.8f,1.0f});
            scene.objects[selectedObj].collisionOnly = true;
            scene.objects[selectedObj].physicsEnabled = true;
            selectedLight = -1;
        } else if (mode == PlaceMode::PointLight) {
            spawnPos.y += 2.5f;
            selectedLight = scene.addPointLight(spawnPos, {1,0.95f,0.85f}, 10.0f);
            selectedObj = -1;
        } else if (mode == PlaceMode::SpotLight) {
            spawnPos.y += 3.5f;
            selectedLight = scene.addSpotLight(spawnPos, {0,-1,0}, {1,0.9f,0.7f}, 15.0f, 40.0f, 18.0f);
            selectedObj = -1;
        }
        regenerate();
    }
};

// -----------------------------------------------------------------------------
// Material helpers (mirror the engine's `material` command)
// -----------------------------------------------------------------------------
static std::string MakeUniqueMaterialId(const Scene& scene, const std::string& requestedId) {
    const std::string base = MakeAssetId(requestedId);
    std::string id = base;
    for (unsigned int suffix = 2; scene.materials.count(id); ++suffix)
        id = base + "_" + std::to_string(suffix);
    return id;
}

// Combo over the scene's declared textures. Returns true when the selection changed.
static bool MaterialTextureCombo(const char* label, std::string& textureId, const Scene& scene) {
    bool changed = false;
    const char* preview = textureId.empty() ? "<none>" : textureId.c_str();
    if (ImGui::BeginCombo(label, preview)) {
        if (ImGui::Selectable("<none>", textureId.empty())) {
            textureId.clear();
            changed = true;
        }
        for (const auto& [id, path] : scene.textures) {
            if (ImGui::Selectable(id.c_str(), textureId == id)) {
                textureId = id;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

// Opens the file picker, registers the image, and assigns it to `textureId`.
static bool ImportMaterialTexture(Scene& scene, Renderer& renderer, std::string& textureId,
                                  const std::string& suggestedName) {
    const std::string path =
        OpenFileDialog("Image Files\0*.png;*.jpg;*.jpeg;*.bmp;*.tga\0All Files\0*.*\0");
    if (path.empty()) return false;
    const std::string base = suggestedName.empty()
        ? std::filesystem::path(path).stem().string()
        : suggestedName;
    const std::string id = MakeUniqueTextureId(scene, base);
    if (!renderer.loadTextureFile(id, path)) return false;
    scene.textures[id] = path;
    textureId = id;
    return true;
}

// Inline material inspector. Returns true when anything changed.
static bool DrawMaterialEditor(Editor& ed, Material& material, const std::string& id) {
    bool changed = false;
    ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f), "Material  %s", id.c_str());
    changed |= ImGui::ColorEdit3("Base Color", &material.color.x);

    changed |= MaterialTextureCombo("Albedo Map", material.texId, ed.scene);
    if (ImGui::Button("Import Albedo Map", ImVec2(-1, 22))) {
        if (ImportMaterialTexture(ed.scene, ed.renderer, material.texId, id + "_albedo"))
            changed = true;
    }

    changed |= MaterialTextureCombo("Normal Map", material.normalTexId, ed.scene);
    if (ImGui::Button("Import Normal Map", ImVec2(-1, 22))) {
        if (ImportMaterialTexture(ed.scene, ed.renderer, material.normalTexId, id + "_normal"))
            changed = true;
    }

    changed |= ImGui::SliderFloat("Smoothness", &material.smoothness, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Metallic", &material.metallic, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Reflectivity", &material.reflectivity, 0.0f, 1.0f);
    changed |= ImGui::Checkbox("Screen Space Reflections", &material.ssr);
    return changed;
}

// -----------------------------------------------------------------------------
// 3D Gizmo helpers (translate arrows / scale handles)
// -----------------------------------------------------------------------------
static Vec3 gizmoAxisDir(int axis) {
    if (axis == 0) return Vec3(1, 0, 0);
    if (axis == 1) return Vec3(0, 1, 0);
    return Vec3(0, 0, 1);
}

static Vec3 gizmoAxisColor(int axis) {
    if (axis == 0) return Vec3(0.93f, 0.28f, 0.32f);
    if (axis == 1) return Vec3(0.36f, 0.84f, 0.40f);
    return Vec3(0.28f, 0.56f, 0.98f);
}

// Rotation that maps the mesh +Y axis onto dir.
static Mat4 alignYTo(const Vec3& dirIn) {
    Vec3 y = normalize(dirIn);
    if (length(y) < 1e-6f) y = Vec3(0, 1, 0);
    Vec3 reference = std::fabs(y.y) < 0.99f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 x = normalize(cross(reference, y));
    Vec3 z = cross(x, y);
    Mat4 r;
    r.m[0] = x.x; r.m[1] = x.y; r.m[2]  = x.z; r.m[3]  = 0;
    r.m[4] = y.x; r.m[5] = y.y; r.m[6]  = y.z; r.m[7]  = 0;
    r.m[8] = z.x; r.m[9] = z.y; r.m[10] = z.z; r.m[11] = 0;
    r.m[12] = 0;  r.m[13] = 0;  r.m[14] = 0;   r.m[15] = 1;
    return r;
}

static float gizmoWorldPerPixel(const Camera& camera, const Vec3& origin, const Mat4& proj, int viewportH) {
    const float dist = std::max(0.1f, length(origin - camera.pos));
    const float tanHalfFov = 1.0f / std::max(proj.m[5], 1e-5f);
    return (2.0f * tanHalfFov * dist) / (float)std::max(viewportH, 1);
}

static float gizmoLength(const Camera& camera, const Vec3& origin, const Mat4& proj, int viewportH) {
    return std::clamp(gizmoWorldPerPixel(camera, origin, proj, viewportH) * 92.0f, 0.05f, 2000.0f);
}

static bool projectToScreen(const Vec3& p, const Mat4& view, const Mat4& proj,
                            int w, int h, ImVec2& out) {
    const Vec4 clip = proj * (view * Vec4(p.x, p.y, p.z, 1.0f));
    if (clip.w <= 1e-4f) return false;
    const float inv = 1.0f / clip.w;
    out.x = (clip.x * inv * 0.5f + 0.5f) * (float)w;
    out.y = (1.0f - (clip.y * inv * 0.5f + 0.5f)) * (float)h;
    return true;
}

static float distanceToSegment2D(const ImVec2& p, const ImVec2& a, const ImVec2& b) {
    const float abx = b.x - a.x, aby = b.y - a.y;
    const float len2 = abx*abx + aby*aby;
    float t = len2 > 1e-6f ? ((p.x - a.x)*abx + (p.y - a.y)*aby) / len2 : 0.0f;
    t = std::clamp(t, 0.0f, 1.0f);
    const float cx = a.x + abx*t, cy = a.y + aby*t;
    const float dx = p.x - cx, dy = p.y - cy;
    return std::sqrt(dx*dx + dy*dy);
}

static bool rayPlanePoint(const Ray& r, const Vec3& planePoint, const Vec3& planeNormal, Vec3& out) {
    const float denom = dot(r.dir, planeNormal);
    if (std::fabs(denom) < 1e-6f) return false;
    const float t = dot(planePoint - r.origin, planeNormal) / denom;
    if (t < 0.0f) return false;
    out = r.origin + r.dir * t;
    return true;
}

// Returns axis 0..2, 3 for the center handle, or -1.
static int pickGizmoAxis(const ImVec2& mouse, const Vec3& origin, GizmoMode mode,
                         const Camera& camera, const Mat4& view, const Mat4& proj,
                         int vpW, int vpH) {
    const float len = gizmoLength(camera, origin, proj, vpH);
    ImVec2 center;
    if (!projectToScreen(origin, view, proj, vpW, vpH, center)) return -1;

    const float centerRadius = 10.0f;
    if (std::fabs(mouse.x - center.x) < centerRadius &&
        std::fabs(mouse.y - center.y) < centerRadius)
        return (int)GizmoAxis::Center;

    int best = -1;
    float bestDist = 11.0f;
    for (int axis = 0; axis < 3; ++axis) {
        ImVec2 tip;
        if (!projectToScreen(origin + gizmoAxisDir(axis) * len, view, proj, vpW, vpH, tip))
            continue;
        const float d = distanceToSegment2D(mouse, center, tip);
        if (d < bestDist) { bestDist = d; best = axis; }
    }
    (void)mode;
    return best;
}

static void drawGizmo(Renderer& renderer, const Vec3& origin, GizmoMode mode,
                      int hoverAxis, int dragAxis, const Camera& camera,
                      const Mat4& proj, int vpH) {
    const float wpp = gizmoWorldPerPixel(camera, origin, proj, vpH);
    const float len = gizmoLength(camera, origin, proj, vpH);
    const float headLen = len * 0.22f;
    const float shaftLen = len - headLen;
    const float shaftRadius = std::max(wpp * 1.6f, len * 0.008f);
    const float headRadius = len * 0.055f;
    const float handleSize = std::max(wpp * 7.0f, len * 0.11f);

    renderer.setDepthTest(false);
    for (int axis = 0; axis < 3; ++axis) {
        const Vec3 dir = gizmoAxisDir(axis);
        const bool hot = (hoverAxis == axis || dragAxis == axis);
        Vec3 color = gizmoAxisColor(axis);
        if (hot) color = Vec3(std::min(1.0f, color.x + 0.35f),
                              std::min(1.0f, color.y + 0.35f),
                              std::min(1.0f, color.z + 0.35f));

        const Mat4 shaftModel = Mat4::translate(origin) * alignYTo(dir)
                              * Mat4::scale(Vec3(shaftRadius * 2.0f, shaftLen, shaftRadius * 2.0f));
        renderer.drawMeshUnlit("$gizmo_cylinder", shaftModel, color);

        if (mode == GizmoMode::Move) {
            const Mat4 coneModel = Mat4::translate(origin + dir * shaftLen) * alignYTo(dir)
                                 * Mat4::scale(Vec3(headRadius * 2.0f, headLen, headRadius * 2.0f));
            renderer.drawMeshUnlit("$gizmo_cone", coneModel, color);
        } else {
            const Mat4 handleModel = Mat4::translate(origin + dir * len)
                                   * Mat4::scale(Vec3(handleSize, handleSize, handleSize));
            renderer.drawMeshUnlit("cube", handleModel, color);
        }
    }

    // Center handle (free move / uniform scale)
    const bool centerHot = (hoverAxis == (int)GizmoAxis::Center ||
                            dragAxis == (int)GizmoAxis::Center);
    const float centerSize = std::max(wpp * 4.5f, len * 0.075f);
    const Vec3 centerColor = centerHot ? Vec3(1.0f, 0.88f, 0.32f) : Vec3(0.82f, 0.85f, 0.92f);
    renderer.drawMeshUnlit("cube", Mat4::translate(origin) * Mat4::scale(Vec3(centerSize, centerSize, centerSize)), centerColor);
    renderer.setDepthTest(true);
}

// World-space AABB of an object (accounts for rotation and size).
static void objectWorldAABB(const SceneObject& o, Vec3& mn, Vec3& mx) {
    const Mat4 model = modelMatrix(o);
    mn = Vec3( 1e30f,  1e30f,  1e30f);
    mx = Vec3(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < 8; ++i) {
        const Vec3 corner((i & 1) ? 0.5f : -0.5f,
                          (i & 2) ? 0.5f : -0.5f,
                          (i & 4) ? 0.5f : -0.5f);
        const Vec3 p = transformPoint(model, corner);
        mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
        mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
    }
}

// Draws the object bounds as an oriented box matching its rotation.
static void drawObjectOutline(Renderer& renderer, const SceneObject& o, const Vec3& color) {
    const Mat4 model = modelMatrix(o);
    Vec3 corners[8];
    for (int i = 0; i < 8; ++i) {
        corners[i] = transformPoint(model, Vec3((i & 1) ? 0.5f : -0.5f,
                                                (i & 2) ? 0.5f : -0.5f,
                                                (i & 4) ? 0.5f : -0.5f));
    }
    const int edges[12][2] = {{0,1},{1,3},{3,2},{2,0},{4,5},{5,7},{7,6},{6,4},
                              {0,4},{1,5},{2,6},{3,7}};
    std::vector<float> verts;
    verts.reserve(12 * 2 * 8);
    for (const auto& e : edges) {
        for (int k = 0; k < 2; ++k) {
            const Vec3& p = corners[e[k]];
            verts.insert(verts.end(), {p.x, p.y, p.z, 0, 1, 0, 0, 0});
        }
    }
    renderer.drawLineSegments(verts, color);
}

// -----------------------------------------------------------------------------
// Laura-style theme
// -----------------------------------------------------------------------------
static const ImVec4 kAccent      = ImVec4(0.24f, 0.51f, 0.95f, 1.00f);
static const ImVec4 kAccentHover = ImVec4(0.33f, 0.60f, 1.00f, 1.00f);
static const ImVec4 kTextMuted   = ImVec4(0.55f, 0.55f, 0.55f, 1.00f);

static const ImU32 kAxisXCol = IM_COL32(198, 82, 114, 255);
static const ImU32 kAxisYCol = IM_COL32(122, 192, 74, 255);
static const ImU32 kAxisZCol = IM_COL32(74, 142, 220, 255);

static const ImU32 kRowSelCol  = IM_COL32(60, 60, 60, 255);
static const ImU32 kRowHoverCol = IM_COL32(44, 44, 44, 255);

static void ApplyLauraTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* c = style.Colors;

    c[ImGuiCol_Text]                  = ImVec4(0.84f, 0.84f, 0.84f, 1.00f);
    c[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    c[ImGuiCol_WindowBg]              = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    c[ImGuiCol_ChildBg]               = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    c[ImGuiCol_PopupBg]               = ImVec4(0.16f, 0.16f, 0.16f, 0.99f);
    c[ImGuiCol_Border]                = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    c[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    c[ImGuiCol_FrameBg]               = ImVec4(0.125f, 0.125f, 0.125f, 1.00f);
    c[ImGuiCol_FrameBgHovered]        = ImVec4(0.165f, 0.165f, 0.165f, 1.00f);
    c[ImGuiCol_FrameBgActive]         = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    c[ImGuiCol_TitleBg]               = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
    c[ImGuiCol_TitleBgActive]         = ImVec4(0.13f, 0.13f, 0.13f, 1.00f);
    c[ImGuiCol_MenuBarBg]             = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
    c[ImGuiCol_ScrollbarBg]           = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
    c[ImGuiCol_ScrollbarGrab]         = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.42f, 0.42f, 0.42f, 1.00f);
    c[ImGuiCol_Button]                = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    c[ImGuiCol_ButtonHovered]         = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    c[ImGuiCol_ButtonActive]          = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    c[ImGuiCol_Header]                = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    c[ImGuiCol_HeaderHovered]         = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
    c[ImGuiCol_HeaderActive]          = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    c[ImGuiCol_Separator]             = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    c[ImGuiCol_SeparatorHovered]      = kAccent;
    c[ImGuiCol_SeparatorActive]       = kAccentHover;
    c[ImGuiCol_Tab]                   = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    c[ImGuiCol_TabHovered]            = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    c[ImGuiCol_TabActive]             = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    c[ImGuiCol_TabUnfocused]          = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    c[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    c[ImGuiCol_TextSelectedBg]        = ImVec4(0.24f, 0.51f, 0.95f, 0.35f);
    c[ImGuiCol_DragDropTarget]        = kAccent;
    c[ImGuiCol_CheckMark]             = kAccent;
    c[ImGuiCol_SliderGrab]            = kAccent;
    c[ImGuiCol_SliderGrabActive]       = kAccentHover;
    c[ImGuiCol_NavHighlight]          = kAccent;

    style.WindowRounding    = 0.0f;
    style.ChildRounding     = 0.0f;
    style.FrameRounding     = 2.0f;
    style.PopupRounding     = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.GrabRounding      = 2.0f;
    style.TabRounding       = 0.0f;

    style.WindowBorderSize  = 0.0f;
    style.ChildBorderSize   = 1.0f;
    style.FrameBorderSize   = 1.0f;
    style.TabBorderSize     = 0.0f;
    style.PopupBorderSize   = 1.0f;

    style.WindowPadding     = ImVec2(8, 8);
    style.FramePadding      = ImVec2(7, 4);
    style.ItemSpacing       = ImVec2(7, 5);
    style.ItemInnerSpacing  = ImVec2(5, 4);
    style.CellPadding       = ImVec2(4, 2);
    style.IndentSpacing     = 16.0f;
    style.ScrollbarSize     = 11.0f;
    style.GrabMinSize       = 8.0f;
}

// -----------------------------------------------------------------------------
// Small custom-drawn glyphs
// -----------------------------------------------------------------------------
static void DrawChevron(ImDrawList* dl, ImVec2 center, float size, bool down, ImU32 col) {
    if (down) {
        dl->AddTriangleFilled(ImVec2(center.x - size * 0.5f, center.y - size * 0.28f),
                              ImVec2(center.x + size * 0.5f, center.y - size * 0.28f),
                              ImVec2(center.x, center.y + size * 0.36f), col);
    } else {
        dl->AddTriangleFilled(ImVec2(center.x - size * 0.28f, center.y - size * 0.5f),
                              ImVec2(center.x - size * 0.28f, center.y + size * 0.5f),
                              ImVec2(center.x + size * 0.38f, center.y), col);
    }
}

static void DrawPlayGlyph(ImVec2 pos, float size, ImU32 col) {
    ImGui::GetWindowDrawList()->AddTriangleFilled(
        ImVec2(pos.x + size * 0.30f, pos.y + size * 0.22f),
        ImVec2(pos.x + size * 0.30f, pos.y + size * 0.78f),
        ImVec2(pos.x + size * 0.76f, pos.y + size * 0.50f), col);
}

static void DrawFolderIcon(ImDrawList* dl, ImVec2 pos, float size, ImU32 col) {
    dl->AddRectFilled(ImVec2(pos.x, pos.y + size * 0.2f), ImVec2(pos.x + size * 0.5f, pos.y + size * 0.4f), col, 1.0f);
    dl->AddRectFilled(ImVec2(pos.x, pos.y + size * 0.3f), ImVec2(pos.x + size, pos.y + size * 0.8f), col, 1.5f);
}

static void DrawTrashIcon(ImDrawList* dl, ImVec2 pos, float size, ImU32 col) {
    dl->AddRectFilled(ImVec2(pos.x + size * 0.16f, pos.y + size * 0.12f),
                      ImVec2(pos.x + size * 0.84f, pos.y + size * 0.25f), col, 1.0f);
    dl->AddRectFilled(ImVec2(pos.x + size * 0.38f, pos.y + size * 0.03f),
                      ImVec2(pos.x + size * 0.62f, pos.y + size * 0.12f), col, 1.0f);
    dl->AddRectFilled(ImVec2(pos.x + size * 0.23f, pos.y + size * 0.30f),
                      ImVec2(pos.x + size * 0.77f, pos.y + size * 0.93f), col, 1.0f);
    const ImU32 cut = IM_COL32(52, 52, 52, 255);
    dl->AddLine(ImVec2(pos.x + size * 0.42f, pos.y + size * 0.40f),
                ImVec2(pos.x + size * 0.42f, pos.y + size * 0.82f), cut, 1.2f);
    dl->AddLine(ImVec2(pos.x + size * 0.58f, pos.y + size * 0.40f),
                ImVec2(pos.x + size * 0.58f, pos.y + size * 0.82f), cut, 1.2f);
}

static void DrawComponentGlyph(ImDrawList* dl, ImVec2 c, int kind, ImU32 col) {
    if (kind == 0) { // transform
        dl->AddLine(ImVec2(c.x - 5, c.y), ImVec2(c.x + 5, c.y), col, 1.3f);
        dl->AddLine(ImVec2(c.x, c.y - 5), ImVec2(c.x, c.y + 5), col, 1.3f);
        dl->AddTriangleFilled(ImVec2(c.x + 6, c.y), ImVec2(c.x + 3, c.y - 2), ImVec2(c.x + 3, c.y + 2), col);
        dl->AddTriangleFilled(ImVec2(c.x - 6, c.y), ImVec2(c.x - 3, c.y - 2), ImVec2(c.x - 3, c.y + 2), col);
        dl->AddTriangleFilled(ImVec2(c.x, c.y - 6), ImVec2(c.x - 2, c.y - 3), ImVec2(c.x + 2, c.y - 3), col);
        dl->AddTriangleFilled(ImVec2(c.x, c.y + 6), ImVec2(c.x - 2, c.y + 3), ImVec2(c.x + 2, c.y + 3), col);
    } else if (kind == 1) { // appearance
        dl->AddCircle(c, 5.0f, col, 14, 1.2f);
        dl->AddCircleFilled(ImVec2(c.x + 1.4f, c.y - 1.4f), 1.8f, col);
    } else if (kind == 2) { // material
        dl->AddCircle(c, 5.0f, col, 14, 1.2f);
        dl->AddLine(ImVec2(c.x - 4.3f, c.y + 1.6f), ImVec2(c.x + 4.3f, c.y + 1.6f), col, 1.0f);
    } else if (kind == 3) { // physics
        dl->AddRect(ImVec2(c.x - 4.5f, c.y - 4.5f), ImVec2(c.x + 4.5f, c.y + 4.5f), col, 1.0f, 0, 1.2f);
        dl->AddCircleFilled(c, 1.9f, col);
    } else if (kind == 4) { // light
        dl->AddCircle(c, 3.6f, col, 12, 1.2f);
        for (int i = 0; i < 8; ++i) {
            const float a = (float)i * 3.14159265f * 0.25f;
            dl->AddLine(ImVec2(c.x + std::cos(a) * 5.2f, c.y + std::sin(a) * 5.2f),
                        ImVec2(c.x + std::cos(a) * 6.6f, c.y + std::sin(a) * 6.6f), col, 1.1f);
        }
    } else { // generic
        dl->AddRect(ImVec2(c.x - 4.5f, c.y - 4.5f), ImVec2(c.x + 4.5f, c.y + 4.5f), col, 1.0f, 0, 1.2f);
    }
}

static void DrawAssetGlyph(ImDrawList* dl, ImVec2 c, float size, bool folder,
                           const std::string& ext, ImU32 col) {
    const bool image = (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
                        ext == ".bmp" || ext == ".tga" || ext == ".hdr");
    const bool model = (ext == ".obj" || ext == ".fbx");
    if (folder) {
        dl->AddRectFilled(ImVec2(c.x - size * 0.5f, c.y - size * 0.32f),
                          ImVec2(c.x - size * 0.05f, c.y - size * 0.14f), col, 1.0f);
        dl->AddRectFilled(ImVec2(c.x - size * 0.5f, c.y - size * 0.20f),
                          ImVec2(c.x + size * 0.5f, c.y + size * 0.38f), col, 2.0f);
    } else if (image) {
        dl->AddRect(ImVec2(c.x - size * 0.42f, c.y - size * 0.32f),
                    ImVec2(c.x + size * 0.42f, c.y + size * 0.32f), col, 2.0f, 0, 1.4f);
        dl->AddCircleFilled(ImVec2(c.x + size * 0.15f, c.y - size * 0.12f), size * 0.075f, col);
        dl->AddTriangleFilled(ImVec2(c.x - size * 0.34f, c.y + size * 0.26f),
                              ImVec2(c.x - size * 0.02f, c.y - size * 0.05f),
                              ImVec2(c.x + size * 0.30f, c.y + size * 0.26f), col);
    } else if (model) {
        dl->AddRect(ImVec2(c.x - size * 0.34f, c.y - size * 0.30f),
                    ImVec2(c.x + size * 0.34f, c.y + size * 0.34f), col, 1.5f, 0, 1.4f);
        dl->AddLine(ImVec2(c.x - size * 0.34f, c.y - size * 0.30f), ImVec2(c.x, c.y - size * 0.06f), col, 1.2f);
        dl->AddLine(ImVec2(c.x + size * 0.34f, c.y - size * 0.30f), ImVec2(c.x, c.y - size * 0.06f), col, 1.2f);
        dl->AddLine(ImVec2(c.x, c.y - size * 0.06f), ImVec2(c.x, c.y + size * 0.34f), col, 1.0f);
    } else {
        dl->AddRect(ImVec2(c.x - size * 0.32f, c.y - size * 0.36f),
                    ImVec2(c.x + size * 0.32f, c.y + size * 0.36f), col, 2.0f, 0, 1.3f);
        dl->AddLine(ImVec2(c.x - size * 0.18f, c.y - size * 0.08f),
                    ImVec2(c.x + size * 0.18f, c.y - size * 0.08f), col, 1.2f);
        dl->AddLine(ImVec2(c.x - size * 0.18f, c.y + size * 0.08f),
                    ImVec2(c.x + size * 0.18f, c.y + size * 0.08f), col, 1.2f);
    }
}

static std::string EllipsizeText(const std::string& text, float maxWidth) {
    if (ImGui::CalcTextSize(text.c_str()).x <= maxWidth) return text;
    std::string result = text;
    const char* suffix = "...";
    while (!result.empty() && ImGui::CalcTextSize((result + suffix).c_str()).x > maxWidth)
        result.pop_back();
    return result + suffix;
}

// -----------------------------------------------------------------------------
// Custom dark title bar (Laura-style)
// -----------------------------------------------------------------------------
static const float kTitleBarHeight = 32.0f;
static const ImVec4 kTitleBarBg = ImVec4(0.075f, 0.075f, 0.075f, 1.0f);

static void DrawMinimizeGlyph(ImDrawList* dl, ImVec2 c, ImU32 col) {
    dl->AddLine(ImVec2(c.x - 5.0f, c.y + 0.5f), ImVec2(c.x + 5.0f, c.y + 0.5f), col, 1.0f);
}

static void DrawMaximizeGlyph(ImDrawList* dl, ImVec2 c, ImU32 col, bool restore, ImU32 bg) {
    if (!restore) {
        dl->AddRect(ImVec2(c.x - 5.0f, c.y - 5.0f), ImVec2(c.x + 5.0f, c.y + 5.0f),
                    col, 0.0f, 0, 1.0f);
    } else {
        dl->AddRect(ImVec2(c.x - 1.5f, c.y - 5.5f), ImVec2(c.x + 5.5f, c.y + 1.5f),
                    col, 0.0f, 0, 1.0f);
        dl->AddRectFilled(ImVec2(c.x - 5.0f, c.y - 1.5f), ImVec2(c.x + 2.0f, c.y + 5.5f), bg);
        dl->AddRect(ImVec2(c.x - 5.5f, c.y - 1.0f), ImVec2(c.x + 1.5f, c.y + 5.5f),
                    col, 0.0f, 0, 1.0f);
    }
}

static void DrawCloseGlyph(ImDrawList* dl, ImVec2 c, ImU32 col) {
    dl->AddLine(ImVec2(c.x - 4.5f, c.y - 4.5f), ImVec2(c.x + 4.5f, c.y + 4.5f), col, 1.1f);
    dl->AddLine(ImVec2(c.x + 4.5f, c.y - 4.5f), ImVec2(c.x - 4.5f, c.y + 4.5f), col, 1.1f);
}

static bool TitleBarButton(const char* id, float w, float h, bool closeStyle) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          closeStyle ? ImVec4(0.72f, 0.18f, 0.18f, 1.0f) : ImVec4(1, 1, 1, 0.08f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          closeStyle ? ImVec4(0.60f, 0.12f, 0.12f, 1.0f) : ImVec4(1, 1, 1, 0.14f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
    const bool pressed = ImGui::Button(id, ImVec2(w, h));
    ImGui::PopStyleColor(4);
    return pressed;
}

// Draws the custom window title bar. `drawMenus` is called inside the menu bar
// so callers can inject their own File/View menus; `showPerf` adds the simulate
// button and the FPS readout next to the window controls.
template <typename MenuFn>
static void DrawAppTitleBar(GLFWwindow* win, GLuint iconTex, const std::string& projectName,
                            bool showPerf, MenuFn&& drawMenus) {
    ImGuiIO& io = ImGui::GetIO();
    const float barH = kTitleBarHeight;
    const float padY = std::max(0.0f, (barH - ImGui::GetFontSize()) * 0.5f);

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, barH));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kTitleBarBg);
    ImGui::PushStyleColor(ImGuiCol_MenuBarBg, kTitleBarBg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(9.0f, padY));
    ImGui::Begin("##appTitleBar", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoSavedSettings |
                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_MenuBar);

    const ImVec2 barPos = ImGui::GetWindowPos();
    const float barW = ImGui::GetWindowWidth();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float btnW = 46.0f;
    const bool maximized = glfwGetWindowAttrib(win, GLFW_MAXIMIZED) != 0;

    char fpsLabel[32];
    std::snprintf(fpsLabel, sizeof(fpsLabel), "%.0f FPS", io.Framerate);
    const float fpsW = ImGui::CalcTextSize(fpsLabel).x;

    const float rightClusterW = btnW * 3.0f;
    const float fpsX = barW - rightClusterW - 14.0f - fpsW;
    const float playW = 24.0f;
    const float playX = fpsX - 12.0f - playW;
    const float rightStart = showPerf ? playX : (barW - rightClusterW);

    float menuRight = 0.0f;

    if (ImGui::BeginMenuBar()) {
        // App icon
        const float iconSize = 18.0f;
        const ImVec2 iconMin(barPos.x + 12.0f, barPos.y + (barH - iconSize) * 0.5f);
        if (iconTex) {
            dl->AddImage((ImTextureID)(intptr_t)iconTex, iconMin,
                         ImVec2(iconMin.x + iconSize, iconMin.y + iconSize),
                         ImVec2(0, 0), ImVec2(1, 1), IM_COL32(220, 220, 220, 255));
        } else {
            dl->AddCircleFilled(ImVec2(iconMin.x + iconSize * 0.5f, iconMin.y + iconSize * 0.5f),
                                iconSize * 0.34f, IM_COL32(210, 210, 210, 255), 12);
        }

        ImGui::SetCursorPosX(iconMin.x - barPos.x + iconSize + 14.0f);
        drawMenus();
        menuRight = ImGui::GetCursorScreenPos().x - barPos.x;

        // Centered project folder name
        const float maxLabelW = std::min(barW * 0.40f, rightStart - menuRight - 24.0f);
        if (maxLabelW > 24.0f) {
            const std::string label = EllipsizeText(projectName, maxLabelW);
            const ImVec2 labelSize = ImGui::CalcTextSize(label.c_str());
            float labelX = std::floor((barW - labelSize.x) * 0.5f);
            labelX = std::max(labelX, menuRight + 10.0f);
            ImGui::SetCursorScreenPos(ImVec2(barPos.x + labelX,
                                             barPos.y + (barH - labelSize.y) * 0.5f - padY));
            ImGui::TextColored(ImVec4(0.78f, 0.78f, 0.78f, 1.0f), "%s", label.c_str());
        }

        if (showPerf) {
            // Simulate button
            ImGui::SetCursorScreenPos(ImVec2(barPos.x + playX, barPos.y));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.08f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1, 1, 1, 0.14f));
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
            if (ImGui::Button("##simulate", ImVec2(playW, barH))) {}
            const ImVec2 pmin = ImGui::GetItemRectMin();
            const ImVec2 pmax = ImGui::GetItemRectMax();
            const float glyphSize = 18.0f;
            DrawPlayGlyph(ImVec2((pmin.x + pmax.x - glyphSize) * 0.5f,
                                 (pmin.y + pmax.y - glyphSize) * 0.5f),
                          glyphSize, IM_COL32(220, 220, 220, 255));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Simulate scene");
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(4);

            // FPS readout
            ImGui::SetCursorScreenPos(ImVec2(barPos.x + fpsX,
                                             barPos.y + (barH - ImGui::GetTextLineHeight()) * 0.5f - padY));
            ImGui::TextColored(kTextMuted, "%s", fpsLabel);
        }

        // Window controls
        ImGui::SetCursorScreenPos(ImVec2(barPos.x + barW - btnW * 3.0f, barPos.y));
        if (TitleBarButton("##minimize", btnW, barH, false)) glfwIconifyWindow(win);
        {
            const ImVec2 a = ImGui::GetItemRectMin();
            const ImVec2 b = ImGui::GetItemRectMax();
            DrawMinimizeGlyph(dl, ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f),
                              IM_COL32(205, 205, 205, 255));
        }

        ImGui::SetCursorScreenPos(ImVec2(barPos.x + barW - btnW * 2.0f, barPos.y));
        if (TitleBarButton("##maximize", btnW, barH, false)) {
            if (maximized) glfwRestoreWindow(win);
            else glfwMaximizeWindow(win);
        }
        {
            const ImVec2 a = ImGui::GetItemRectMin();
            const ImVec2 b = ImGui::GetItemRectMax();
            DrawMaximizeGlyph(dl, ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f),
                              IM_COL32(205, 205, 205, 255), maximized,
                              IM_COL32(19, 19, 19, 255));
        }

        ImGui::SetCursorScreenPos(ImVec2(barPos.x + barW - btnW, barPos.y));
        if (TitleBarButton("##close", btnW, barH, true)) glfwSetWindowShouldClose(win, GLFW_TRUE);
        {
            const ImVec2 a = ImGui::GetItemRectMin();
            const ImVec2 b = ImGui::GetItemRectMax();
            DrawCloseGlyph(dl, ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f),
                           IM_COL32(225, 225, 225, 255));
        }

        // Bottom separator line
        dl->AddLine(ImVec2(barPos.x, barPos.y + barH - 0.5f),
                    ImVec2(barPos.x + barW, barPos.y + barH - 0.5f),
                    IM_COL32(0, 0, 0, 255), 1.0f);

        ImGui::EndMenuBar();
    }

    // ---- Window dragging (empty title bar area) ----
    static bool dragging = false;
    static ImVec2 dragStartCursor;
    static ImVec2 dragStartWindow;

    const ImVec2 barMax(barPos.x + barW, barPos.y + barH);
    const ImVec2 mouse = io.MousePos;
    const bool inResizeBorder = mouse.y < barPos.y + 5.0f ||
                                mouse.x < barPos.x + 5.0f ||
                                mouse.x > barMax.x - 5.0f;
    const bool barHovered = ImGui::IsMouseHoveringRect(barPos, barMax, false) &&
                            !ImGui::IsAnyItemHovered() && !inResizeBorder;


    if (!dragging && barHovered && ImGui::IsMouseClicked(0)) {
        if (ImGui::IsMouseDoubleClicked(0)) {
            if (maximized) glfwRestoreWindow(win);
            else glfwMaximizeWindow(win);
        } else if (!maximized) {
            double cx = 0.0, cy = 0.0;
            glfwGetCursorPos(win, &cx, &cy);
            int wx = 0, wy = 0;
            glfwGetWindowPos(win, &wx, &wy);
            dragging = true;
            dragStartCursor = ImVec2((float)(wx + cx), (float)(wy + cy));
            dragStartWindow = ImVec2((float)wx, (float)wy);
        }
    }

    if (dragging) {
        if (!ImGui::IsMouseDown(0)) {
            dragging = false;
        } else {
            double cx = 0.0, cy = 0.0;
            glfwGetCursorPos(win, &cx, &cy);
            int wx = 0, wy = 0;
            glfwGetWindowPos(win, &wx, &wy);
            const float dx = (float)(wx + cx) - dragStartCursor.x;
            const float dy = (float)(wy + cy) - dragStartCursor.y;
            glfwSetWindowPos(win, (int)(dragStartWindow.x + dx),
                             (int)(dragStartWindow.y + dy));
        }
    }

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// Custom resize grip for the undecorated main window (5px border).
static void HandleWindowBorderResize(GLFWwindow* win) {
    static bool active = false;
    static int activeEdge = 0;
    static int startX = 0, startY = 0, startW = 0, startH = 0;
    static double startCursorX = 0.0, startCursorY = 0.0;

    if (glfwGetWindowAttrib(win, GLFW_MAXIMIZED) != 0) {
        active = false;
        return;
    }

    int ww = 0, wh = 0, wx = 0, wy = 0;
    glfwGetWindowSize(win, &ww, &wh);
    glfwGetWindowPos(win, &wx, &wy);
    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(win, &cx, &cy);

    const float edgeSize = 5.0f;
    int edge = 0;
    if (cx >= 0.0 && cy >= 0.0 && cx < (double)ww && cy < (double)wh) {
        if (cx <= edgeSize) edge |= 1;
        if (cx >= (double)ww - edgeSize) edge |= 2;
        if (cy <= edgeSize) edge |= 4;
        if (cy >= (double)wh - edgeSize) edge |= 8;
    }

    if (!active) {
        const bool popupOpen = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId |
                                                     ImGuiPopupFlags_AnyPopupLevel);
        if (edge != 0 && ImGui::IsMouseClicked(0) && !ImGui::IsAnyItemHovered() && !popupOpen) {
            active = true;
            activeEdge = edge;
            startX = wx;
            startY = wy;
            startW = ww;
            startH = wh;
            startCursorX = wx + cx;
            startCursorY = wy + cy;
        }
    }

    if (active) {
        if (!ImGui::IsMouseDown(0)) {
            active = false;
        } else {
            const int dx = (int)std::lround((wx + cx) - startCursorX);
            const int dy = (int)std::lround((wy + cy) - startCursorY);
            const int minW = 720, minH = 480;
            int nx = startX, ny = startY, nw = startW, nh = startH;

            if (activeEdge & 2) nw = startW + dx;
            if (activeEdge & 1) { nw = startW - dx; nx = startX + dx; }
            if (activeEdge & 8) nh = startH + dy;
            if (activeEdge & 4) { nh = startH - dy; ny = startY + dy; }
            if (nw < minW) { if (activeEdge & 1) nx = startX + (startW - minW); nw = minW; }
            if (nh < minH) { if (activeEdge & 4) ny = startY + (startH - minH); nh = minH; }

            glfwSetWindowSize(win, nw, nh);
            glfwSetWindowPos(win, nx, ny);
        }
    }

    if (!active && edge != 0) {
        const bool left = (edge & 1) != 0, right = (edge & 2) != 0;
        const bool top = (edge & 4) != 0, bottom = (edge & 8) != 0;
        ImGuiMouseCursor cursor = ImGuiMouseCursor_ResizeEW;
        if ((left && top) || (right && bottom)) cursor = ImGuiMouseCursor_ResizeNWSE;
        else if ((right && top) || (left && bottom)) cursor = ImGuiMouseCursor_ResizeNESW;
        else if (top || bottom) cursor = ImGuiMouseCursor_ResizeNS;
        ImGui::SetMouseCursor(cursor);
    }
}

// -----------------------------------------------------------------------------
// Panel / component headers
// -----------------------------------------------------------------------------
struct PanelTabDef {
    const char* label;
    GLuint icon;
};

static int PanelHeader(const PanelTabDef* tabs, int count, int active) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 wp = ImGui::GetWindowPos();
    const float ww = ImGui::GetWindowWidth();
    const float h = 26.0f;

    dl->AddRectFilled(wp, ImVec2(wp.x + ww, wp.y + h), IM_COL32(29, 29, 29, 255));

    int clickedTab = -1;
    float x = wp.x;
    for (int i = 0; i < count; ++i) {
        const float textW = ImGui::CalcTextSize(tabs[i].label).x;
        const float tabW = 12.0f + (tabs[i].icon ? 19.0f : 0.0f) + textW + 14.0f;
        const ImVec2 tmin(x, wp.y);
        const ImVec2 tmax(x + tabW, wp.y + h);
        const bool hovered = ImGui::IsMouseHoveringRect(tmin, tmax) && ImGui::IsWindowHovered();
        const bool isActive = (i == active);

        ImU32 bg = isActive ? IM_COL32(46, 46, 46, 255)
                            : (hovered ? IM_COL32(38, 38, 38, 255) : IM_COL32(32, 32, 32, 255));
        dl->AddRectFilled(tmin, tmax, bg);
        if (isActive)
            dl->AddLine(ImVec2(tmin.x, wp.y), ImVec2(tmin.x, wp.y + h),
                        IM_COL32(60, 60, 60, 255), 1.0f);
        if (i + 1 < count)
            dl->AddLine(ImVec2(tmax.x - 0.5f, wp.y + 6), ImVec2(tmax.x - 0.5f, wp.y + h - 6),
                        IM_COL32(22, 22, 22, 255), 1.0f);

        const float iconSize = 13.0f;
        float textX = tmin.x + 12.0f;
        if (tabs[i].icon) {
            const float iy = wp.y + (h - iconSize) * 0.5f;
            dl->AddImage((ImTextureID)(intptr_t)tabs[i].icon,
                         ImVec2(textX, iy), ImVec2(textX + iconSize, iy + iconSize));
            textX += iconSize + 6.0f;
        }
        const ImU32 textCol = isActive ? IM_COL32(224, 224, 224, 255)
                                       : IM_COL32(138, 138, 138, 255);
        dl->AddText(ImVec2(textX, wp.y + (h - ImGui::GetTextLineHeight()) * 0.5f + 1.0f),
                    textCol, tabs[i].label);

        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            clickedTab = i;
        x += tabW;
    }

    dl->AddLine(ImVec2(wp.x, wp.y + h - 0.5f), ImVec2(wp.x + ww, wp.y + h - 0.5f),
                IM_COL32(16, 16, 16, 255), 1.0f);

    ImGui::SetCursorScreenPos(ImVec2(wp.x, wp.y + h));
    return clickedTab;
}

static bool ComponentHeader(const char* label, bool* open, int glyph) {
    ImGui::PushID(label);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = 22.0f;

    ImGui::InvisibleButton("##comp", ImVec2(w, h));
    const bool hovered = ImGui::IsItemHovered();
    if (ImGui::IsItemClicked()) *open = !*open;

    const ImU32 bg = hovered ? IM_COL32(53, 53, 53, 255) : IM_COL32(43, 43, 43, 255);
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), bg, 2.0f);
    dl->AddLine(ImVec2(p.x, p.y + h - 0.5f), ImVec2(p.x + w, p.y + h - 0.5f),
                IM_COL32(24, 24, 24, 255), 1.0f);

    DrawChevron(dl, ImVec2(p.x + 12.0f, p.y + h * 0.5f), 8.0f, *open,
                IM_COL32(175, 175, 175, 255));
    DrawComponentGlyph(dl, ImVec2(p.x + 28.0f, p.y + h * 0.5f), glyph,
                       IM_COL32(195, 195, 195, 255));
    dl->AddText(ImVec2(p.x + 38.0f, p.y + (h - ImGui::GetTextLineHeight()) * 0.5f),
                IM_COL32(222, 222, 222, 255), label);

    ImGui::PopID();
    return *open;
}

// Laura-style XYZ field row: colored letter block + value box.
static bool DrawVector3Row(const char* label, float* v, float speed, const char* fmt) {
    bool changed = false;
    ImGui::PushID(label);
    const float rowH = ImGui::GetFrameHeight();

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(kTextMuted, "%s", label);
    ImGui::SameLine(64.0f);

    const float avail = ImGui::GetContentRegionAvail().x;
    const float gap = 4.0f;
    const float fieldW = (avail - gap * 2.0f) / 3.0f;
    static const ImU32 axisCols[3] = { kAxisXCol, kAxisYCol, kAxisZCol };
    static const char* axisNames[3] = { "X", "Y", "Z" };

    for (int i = 0; i < 3; ++i) {
        ImGui::PushID(i);
        ImDrawList* dl = ImGui::GetWindowDrawList();

        const ImVec2 blockMin = ImGui::GetCursorScreenPos();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        ImGui::InvisibleButton("##axis", ImVec2(16.0f, rowH));
        ImGui::PopStyleVar();
        const ImVec2 blockMax = ImGui::GetItemRectMax();
        dl->AddRectFilled(blockMin, blockMax, axisCols[i], 2.0f, ImDrawFlags_RoundCornersLeft);
        dl->AddText(ImVec2(blockMin.x + 6.0f,
                           blockMin.y + (rowH - ImGui::GetTextLineHeight()) * 0.5f),
                    IM_COL32(245, 245, 245, 240), axisNames[i]);

        ImGui::SameLine(0, 0);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.145f, 0.145f, 0.145f, 1.0f));
        ImGui::SetNextItemWidth(fieldW - 16.0f);
        changed |= ImGui::DragFloat("##v", &v[i], speed, 0.0f, 0.0f, fmt);
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
        ImGui::PopID();

        if (i < 2) ImGui::SameLine(0, gap);
    }

    ImGui::PopID();
    return changed;
}

// Small flat button used for the in-viewport gizmo mode switch.
static bool ViewportModeButton(const char* id, const char* label, bool active) {
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.42f, 0.75f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.48f, 0.84f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.26f, 0.50f, 0.88f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.13f, 0.13f, 0.92f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.22f, 0.22f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.28f, 0.28f, 0.28f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.70f, 0.70f, 1.0f));
    }
    ImGui::PushID(id);
    const bool clicked = ImGui::Button(label, ImVec2(62.0f, 22.0f));
    ImGui::PopID();
    ImGui::PopStyleColor(4);
    return clicked;
}

// -----------------------------------------------------------------------------
// Import helpers
// -----------------------------------------------------------------------------
static std::string ModelIdForPath(Editor& ed, const std::filesystem::path& modelPath) {
    const std::string base = MakeAssetId(modelPath.stem().string());
    std::string id = base;
    for (unsigned int suffix = 2; ed.scene.models.count(id); ++suffix)
        id = base + "_" + std::to_string(suffix);
    return id;
}

static bool LoadModelFile(Editor& ed, const std::string& path) {
    const std::filesystem::path modelPath(path);
    std::string extension = modelPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    const std::string id = ModelIdForPath(ed, modelPath);
    const bool isFBX = extension == ".fbx";
    const bool loaded = isFBX
        ? ed.renderer.loadModelFBX(id, path)
        : extension == ".obj" && ed.renderer.loadModelOBJ(id, path, false);
    if (!loaded) {
        std::fprintf(stderr, "[model] unsupported or failed to load: %s\n", path.c_str());
        return false;
    }

    ed.scene.models[id] = {path, false};

    Vec3 fwd = ed.camera.forward();
    Vec3 spawnPos = ed.camera.pos + fwd * 10.0f;
    if (fwd.y < -0.01f) {
        float t = -ed.camera.pos.y / fwd.y;
        if (t > 0) spawnPos = ed.camera.pos + fwd * t;
    }
    spawnPos.y += 0.5f;

    ed.selectedObj = ed.scene.addObject(ObjType::Cube, spawnPos, {1,1,1}, {0.8f,0.8f,0.8f}, id);
    ed.selectedLight = -1;
    if (isFBX) {
        const std::vector<FBXMaterialInfo>* materials = ed.renderer.fbxMaterials(id);
        if (materials) {
            auto& materialTexIds = ed.scene.objects[ed.selectedObj].materialTexIds;
            materialTexIds.resize(materials->size());
            for (size_t i = 0; i < materials->size(); ++i) {
                const std::string& texturePath = (*materials)[i].diffuseTexturePath;
                if (texturePath.empty()) continue;
                const std::string textureId = MakeUniqueTextureId(ed.scene, id + "_mat" + std::to_string(i + 1));
                if (ed.renderer.loadTextureFile(textureId, texturePath)) {
                    ed.scene.textures[textureId] = texturePath;
                    materialTexIds[i] = textureId;
                }
            }
        }
    }
    ed.regenerate();
    return true;
}

static bool RegisterTextureFile(Editor& ed, const std::string& path) {
    const std::string id = MakeUniqueTextureId(ed.scene,
        std::filesystem::path(path).stem().string());
    if (!ed.renderer.loadTextureFile(id, path)) return false;
    ed.scene.textures[id] = path;
    ed.regenerate();
    return true;
}

static void ImportModelDialog(Editor& ed) {
    const std::string path = OpenFileDialog(
        "Model Files\0*.obj;*.fbx\0OBJ Files\0*.obj\0FBX Files\0*.fbx\0All Files\0*.*\0");
    if (!path.empty()) LoadModelFile(ed, path);
}

static void ImportTextureDialog(Editor& ed) {
    const std::string path = OpenFileDialog(
        "Image Files\0*.png;*.jpg;*.jpeg;*.bmp;*.tga\0All Files\0*.*\0");
    if (!path.empty()) RegisterTextureFile(ed, path);
}

static std::string ObjectKindLabel(const SceneObject& o) {
    if (o.collisionOnly) return "collider";
    if (!o.modelId.empty()) return o.modelId;
    return objTypeName(o.type);
}

// -----------------------------------------------------------------------------
// Inspector
// -----------------------------------------------------------------------------
static void DrawInspectorObject(Editor& ed) {
    auto& o = ed.scene.objects[ed.selectedObj];
    static bool openTransform = true;
    static bool openAppearance = true;
    static bool openMaterial = false;
    static bool openPhysics = false;

    // Name
    ImGui::TextColored(kTextMuted, "Name:");
    ImGui::SameLine(64.0f);
    char nameBuf[160];
    std::snprintf(nameBuf, sizeof(nameBuf), "%s", o.name.c_str());
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##objName", nameBuf, sizeof(nameBuf))) {
        o.name = nameBuf;
        ed.regenerate();
    }
    ImGui::PopStyleColor();

    ImGui::TextColored(kTextMuted, "Type: %s", ObjectKindLabel(o).c_str());
    ImGui::Dummy(ImVec2(0, 3));

    // Transform
    if (ComponentHeader("Transform", &openTransform, 0)) {
        ImGui::Dummy(ImVec2(0, 2));
        bool changed = false;
        changed |= DrawVector3Row("Position", &o.pos.x, 0.02f, "%.3f");
        changed |= DrawVector3Row("Rotation", &o.rot.x, 0.5f, "%.1f");
        changed |= DrawVector3Row("Scale", &o.size.x, 0.02f, "%.3f");
        if (changed) ed.regenerate();
        ImGui::Dummy(ImVec2(0, 4));
    }
    ImGui::Dummy(ImVec2(0, 3));

    // Appearance
    if (ComponentHeader("Appearance", &openAppearance, 1)) {
        ImGui::PushItemWidth(-92.0f);
        if (ImGui::ColorEdit3("Color", &o.color.x)) ed.regenerate();
        ImGui::PopItemWidth();
        ImGui::Dummy(ImVec2(0, 4));
    }
    ImGui::Dummy(ImVec2(0, 3));

    // Material
    if (ComponentHeader("Material", &openMaterial, 2)) {
        const char* currentMaterial = o.materialId.empty() ? "<none>" : o.materialId.c_str();
        ImGui::TextColored(kTextMuted, "Material");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##material", currentMaterial)) {
            if (ImGui::Selectable("<none>", o.materialId.empty())) {
                o.materialId.clear();
                ed.regenerate();
            }
            for (const auto& [id, material] : ed.scene.materials) {
                if (ImGui::Selectable(id.c_str(), o.materialId == id)) {
                    o.materialId = id;
                    ed.selectedMaterialId = id;
                    ed.regenerate();
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::Button("New Material", ImVec2(-1, 22))) {
            const std::string id = MakeUniqueMaterialId(ed.scene, o.name + "_mat");
            Material newMaterial;
            newMaterial.name = id;
            ed.scene.materials[id] = newMaterial;
            o.materialId = id;
            ed.selectedMaterialId = id;
            ed.regenerate();
        }
        if (!o.materialId.empty()) {
            auto materialIt = ed.scene.materials.find(o.materialId);
            if (materialIt != ed.scene.materials.end()) {
                ImGui::Dummy(ImVec2(0, 4));
                ImGui::PushItemWidth(-92.0f);
                if (DrawMaterialEditor(ed, materialIt->second, materialIt->first))
                    ed.regenerate();
                ImGui::PopItemWidth();
            } else {
                ImGui::TextDisabled("Missing material '%s'", o.materialId.c_str());
            }
        }

        const std::vector<FBXMaterialInfo>* fbxMaterials = ed.renderer.fbxMaterials(o.modelId);
        if (fbxMaterials) {
            o.materialTexIds.resize(fbxMaterials->size());
            ImGui::Dummy(ImVec2(0, 4));
            ImGui::TextColored(kTextMuted, "FBX Material Textures");
            for (size_t materialIndex = 0; materialIndex < fbxMaterials->size(); ++materialIndex) {
                const FBXMaterialInfo& fbxMaterial = (*fbxMaterials)[materialIndex];
                ImGui::PushID(static_cast<int>(materialIndex));
                ImGui::TextUnformatted(fbxMaterial.name.c_str());
                const std::string& assignedTexture = o.materialTexIds[materialIndex];
                const char* currentTex = assignedTexture.empty() ? "<none>" : assignedTexture.c_str();
                ImGui::SetNextItemWidth(-1);
                if (ImGui::BeginCombo("##tex", currentTex)) {
                    if (ImGui::Selectable("<none>", assignedTexture.empty())) {
                        o.materialTexIds[materialIndex].clear();
                        ed.regenerate();
                    }
                    for (const auto& [id, texture] : ed.renderer.textures()) {
                        if (ImGui::Selectable(id.c_str(), assignedTexture == id)) {
                            o.materialTexIds[materialIndex] = id;
                            ed.regenerate();
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopID();
            }
        }

        ImGui::PushItemWidth(-112.0f);
        const char* wrapModes[] = {"Repeat", "Clamp to Edge"};
        int wrapMode = o.textureWrap == TextureWrapMode::Clamp ? 1 : 0;
        if (ImGui::Combo("Texture Wrap", &wrapMode, wrapModes, 2)) {
            o.textureWrap = wrapMode == 1 ? TextureWrapMode::Clamp : TextureWrapMode::Repeat;
            ed.regenerate();
        }
        const char* filterModes[] = {"Linear", "Nearest"};
        int filterMode = o.textureFilter == TextureFilterMode::Nearest ? 1 : 0;
        if (ImGui::Combo("Texture Filter", &filterMode, filterModes, 2)) {
            o.textureFilter = filterMode == 1 ? TextureFilterMode::Nearest : TextureFilterMode::Linear;
            ed.regenerate();
        }
        ImGui::PopItemWidth();
        ImGui::Dummy(ImVec2(0, 4));
    }
    ImGui::Dummy(ImVec2(0, 3));

    // Physics
    if (ComponentHeader("Physics", &openPhysics, 3)) {
        ImGui::PushItemWidth(-112.0f);
        bool physicsChanged = ImGui::Checkbox("Physics Body", &o.physicsEnabled);
        if (o.physicsEnabled) {
            const char* bodyTypes[] = {"Static", "Dynamic"};
            int bodyType = o.physicsDynamic ? 1 : 0;
            if (ImGui::Combo("Body Type", &bodyType, bodyTypes, 2)) {
                o.physicsDynamic = bodyType == 1;
                physicsChanged = true;
            }
            if (o.physicsDynamic)
                physicsChanged |= ImGui::DragFloat("Mass", &o.mass, 0.05f, 0.01f, 1000.0f);
            physicsChanged |= ImGui::DragFloat("Friction", &o.friction, 0.01f, 0.0f, 2.0f);
            physicsChanged |= ImGui::DragFloat("Restitution", &o.restitution, 0.01f, 0.0f, 1.0f);
        }
        ImGui::PopItemWidth();
        if (physicsChanged) ed.regenerate();
        ImGui::Dummy(ImVec2(0, 4));
    }
    ImGui::Dummy(ImVec2(0, 4));

    // Add Component
    const float addW = 150.0f;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                         (ImGui::GetContentRegionAvail().x - addW) * 0.5f);
    if (ImGui::Button("Add Component", ImVec2(addW, 24)))
        ImGui::OpenPopup("##addComponent");
    if (ImGui::BeginPopup("##addComponent")) {
        if (ImGui::MenuItem("Material")) {
            const std::string id = MakeUniqueMaterialId(ed.scene, o.name + "_mat");
            Material material;
            material.name = id;
            ed.scene.materials[id] = material;
            o.materialId = id;
            ed.selectedMaterialId = id;
            openMaterial = true;
            ed.regenerate();
        }
        if (ImGui::MenuItem("Physics Body")) {
            o.physicsEnabled = true;
            openPhysics = true;
            ed.regenerate();
        }
        ImGui::EndPopup();
    }

    ImGui::Dummy(ImVec2(0, 8));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.13f, 0.14f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.42f, 0.16f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.50f, 0.19f, 0.21f, 1.0f));
    if (ImGui::Button("Delete Object", ImVec2(-1, 26))) {
        ed.scene.removeObject(ed.selectedObj);
        ed.selectedObj = -1;
        ed.gizmoDrag.active = false;
        ed.hoverAxis = -1;
        ed.regenerate();
    }
    ImGui::PopStyleColor(3);
}

static void DrawInspectorLight(Editor& ed) {
    auto& l = ed.scene.lights[ed.selectedLight];
    static bool openTransform = true;
    static bool openLight = true;

    ImGui::TextColored(kTextMuted, "Name:");
    ImGui::SameLine(64.0f);
    char nameBuf[160];
    std::snprintf(nameBuf, sizeof(nameBuf), "%s", l.name.c_str());
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##lightName", nameBuf, sizeof(nameBuf))) {
        l.name = nameBuf;
        ed.regenerate();
    }
    ImGui::PopStyleColor();

    const char* typeName = (l.type == LightType::Spot) ? "spot" : (l.type == LightType::Sun) ? "sun" : "point";
    ImGui::TextColored(kTextMuted, "Type: %s light", typeName);
    ImGui::Dummy(ImVec2(0, 3));

    if (ComponentHeader("Transform", &openTransform, 0)) {
        ImGui::Dummy(ImVec2(0, 2));
        bool changed = false;
        changed |= DrawVector3Row("Position", &l.pos.x, 0.02f, "%.3f");
        if (l.type == LightType::Spot)
            changed |= DrawVector3Row("Direction", &l.dir.x, 0.02f, "%.3f");
        if (changed) ed.regenerate();
        ImGui::Dummy(ImVec2(0, 4));
    }
    ImGui::Dummy(ImVec2(0, 3));

    if (ComponentHeader("Light", &openLight, 4)) {
        ImGui::PushItemWidth(-112.0f);
        bool lightChanged = ImGui::ColorEdit3("Color", &l.color.x);
        lightChanged |= ImGui::DragFloat("Intensity", &l.intensity, 0.01f, 0.0f, 100.0f, "%.3f");
        const char* rangeLabel = l.type == LightType::Point ? "Radius" : "Range";
        lightChanged |= ImGui::DragFloat(rangeLabel, &l.range, 0.1f, 0.1f, 100.0f);
        if (l.type == LightType::Point)
            lightChanged |= ImGui::Checkbox("Strict Attenuation", &l.strictAttenuation);
        if (l.type == LightType::Spot) {
            lightChanged |= ImGui::DragFloat("Outer Angle", &l.outerDeg, 0.5f, 1.0f, 89.0f);
            lightChanged |= ImGui::DragFloat("Inner Angle", &l.innerDeg, 0.5f, 0.5f, 88.0f);
            if (l.innerDeg >= l.outerDeg) l.innerDeg = l.outerDeg - 0.5f;
        }
        ImGui::PopItemWidth();
        if (lightChanged) ed.regenerate();
        ImGui::Dummy(ImVec2(0, 4));
    }

    ImGui::Dummy(ImVec2(0, 8));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.13f, 0.14f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.42f, 0.16f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.50f, 0.19f, 0.21f, 1.0f));
    if (ImGui::Button("Delete Light", ImVec2(-1, 26))) {
        ed.scene.removeLight(ed.selectedLight);
        ed.selectedLight = -1;
        ed.gizmoDrag.active = false;
        ed.hoverAxis = -1;
        ed.regenerate();
    }
    ImGui::PopStyleColor(3);
}

static void DrawInspectorEmpty(Editor& ed) {
    static bool openScene = true;
    static bool openMaterials = true;

    ImGui::TextColored(kTextMuted, "Select an item to edit its properties.");
    ImGui::Dummy(ImVec2(0, 6));

    if (ComponentHeader("Scene", &openScene, 5)) {
        ImGui::TextColored(kTextMuted, "Name:");
        ImGui::SameLine(64.0f);
        ImGui::TextUnformatted("scene");
        ImGui::TextColored(kTextMuted, "Objects:");
        ImGui::SameLine(64.0f);
        ImGui::TextUnformatted(std::to_string(ed.scene.objects.size()).c_str());
        ImGui::TextColored(kTextMuted, "Lights:");
        ImGui::SameLine(64.0f);
        ImGui::TextUnformatted(std::to_string(ed.scene.lights.size()).c_str());
        ImGui::TextColored(kTextMuted, "Textures:");
        ImGui::SameLine(64.0f);
        ImGui::TextUnformatted(std::to_string(ed.scene.textures.size()).c_str());
        ImGui::Dummy(ImVec2(0, 4));
        ImGui::TextColored(kTextMuted, "Use File or the Assets bar to import models and textures.");
        ImGui::Dummy(ImVec2(0, 4));
    }
    ImGui::Dummy(ImVec2(0, 3));

    if (ComponentHeader("Materials", &openMaterials, 2)) {
        ImGui::BeginChild("##materialList", ImVec2(-1, 96), true);
        if (ed.scene.materials.empty()) {
            ImGui::TextDisabled("No materials yet.");
        } else {
            for (const auto& [id, material] : ed.scene.materials) {
                if (ImGui::Selectable(id.c_str(), ed.selectedMaterialId == id))
                    ed.selectedMaterialId = id;
            }
        }
        ImGui::EndChild();
        if (ImGui::Button("New Material", ImVec2(-1, 22))) {
            const std::string id = ed.scene.makeUniqueMaterialName("material");
            Material material;
            material.name = id;
            ed.scene.materials[id] = material;
            ed.selectedMaterialId = id;
            ed.regenerate();
        }
        auto materialIt = ed.scene.materials.find(ed.selectedMaterialId);
        if (materialIt != ed.scene.materials.end()) {
            const std::string materialId = materialIt->first;
            ImGui::Dummy(ImVec2(0, 4));
            ImGui::PushItemWidth(-92.0f);
            if (DrawMaterialEditor(ed, materialIt->second, materialId))
                ed.regenerate();
            ImGui::PopItemWidth();
            ImGui::Dummy(ImVec2(0, 4));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.13f, 0.14f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.42f, 0.16f, 0.18f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.50f, 0.19f, 0.21f, 1.0f));
            if (ImGui::Button("Delete Material", ImVec2(-1, 24))) {
                for (auto& object : ed.scene.objects)
                    if (object.materialId == materialId) object.materialId.clear();
                ed.scene.materials.erase(materialId);
                ed.selectedMaterialId.clear();
                ed.regenerate();
            }
            ImGui::PopStyleColor(3);
        }
    }
}

static void DrawInspectorBody(Editor& ed) {
    if (ed.selectedObj >= 0 && ed.selectedObj < (int)ed.scene.objects.size())
        DrawInspectorObject(ed);
    else if (ed.selectedLight >= 0 && ed.selectedLight < (int)ed.scene.lights.size())
        DrawInspectorLight(ed);
    else
        DrawInspectorEmpty(ed);
}

// -----------------------------------------------------------------------------
// Scene Hierarchy
// -----------------------------------------------------------------------------
static void DrawHierarchyBody(Editor& ed) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.24f, 0.24f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.27f, 0.27f, 0.27f, 1.0f));
    if (ImGui::Button("##addSceneItem", ImVec2(-1, 20)))
        ImGui::OpenPopup("##addSceneItem");
    ImGui::PopStyleColor(3);
    {
        const ImVec2 amin = ImGui::GetItemRectMin();
        const ImVec2 amax = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddText(
            ImVec2(amin.x + 10.0f, amin.y + (amax.y - amin.y - ImGui::GetTextLineHeight()) * 0.5f),
            IM_COL32(212, 212, 212, 255), "Add");
        DrawChevron(ImGui::GetWindowDrawList(),
                    ImVec2(amax.x - 14.0f, amin.y + 10.0f), 8.0f, true,
                    IM_COL32(170, 170, 170, 255));
    }
    if (ImGui::BeginPopup("##addSceneItem")) {
        if (ImGui::MenuItem("Cube"))        ed.spawnAtCenter(PlaceMode::Cube);
        if (ImGui::MenuItem("Sphere"))      ed.spawnAtCenter(PlaceMode::Sphere);
        if (ImGui::MenuItem("Collider"))    ed.spawnAtCenter(PlaceMode::Collider);
        ImGui::Separator();
        if (ImGui::MenuItem("Point Light")) ed.spawnAtCenter(PlaceMode::PointLight);
        if (ImGui::MenuItem("Spot Light"))  ed.spawnAtCenter(PlaceMode::SpotLight);
        ImGui::EndPopup();
    }

    ImGui::Dummy(ImVec2(0, 2));

    int deleteObj = -1;
    int deleteLight = -1;
    const float rowH = 22.0f;

    auto drawRow = [&](int id, const char* name, const std::string& kind,
                       bool selected, bool isLight, int index) {
        ImGui::PushID(id);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;

        ImGui::InvisibleButton("##row", ImVec2(w, rowH));
        const bool hovered = ImGui::IsItemHovered();
        const bool clicked = ImGui::IsItemClicked();

        if (selected)
            dl->AddRectFilled(p, ImVec2(p.x + w, p.y + rowH), kRowSelCol);
        else if (hovered)
            dl->AddRectFilled(p, ImVec2(p.x + w, p.y + rowH), kRowHoverCol);

        DrawChevron(dl, ImVec2(p.x + 9.0f, p.y + rowH * 0.5f), 8.0f, false,
                    IM_COL32(150, 150, 150, 255));
        dl->AddText(ImVec2(p.x + 20.0f, p.y + (rowH - ImGui::GetTextLineHeight()) * 0.5f),
                    selected ? IM_COL32(238, 238, 238, 255) : IM_COL32(205, 205, 205, 255),
                    name);
        const float nameW = ImGui::CalcTextSize(name).x;
        dl->AddText(ImVec2(p.x + 26.0f + nameW,
                           p.y + (rowH - ImGui::GetTextLineHeight()) * 0.5f),
                    IM_COL32(128, 128, 128, 255), kind.c_str());

        const ImVec2 trashMin(p.x + w - 22.0f, p.y + 4.0f);
        const bool trashHovered = (selected || hovered) &&
            ImGui::IsMouseHoveringRect(trashMin, ImVec2(trashMin.x + 16.0f, trashMin.y + 15.0f));
        if (selected || hovered) {
            if (trashHovered)
                dl->AddRectFilled(ImVec2(trashMin.x - 2, trashMin.y - 2),
                                  ImVec2(trashMin.x + 18, trashMin.y + 17),
                                  IM_COL32(75, 75, 75, 255), 2.0f);
            DrawTrashIcon(dl, ImVec2(trashMin.x + 1.0f, trashMin.y + 1.0f), 12.0f,
                          trashHovered ? IM_COL32(240, 240, 240, 255)
                                       : IM_COL32(155, 155, 155, 255));
        }

        if (clicked) {
            if (trashHovered) {
                if (isLight) deleteLight = index;
                else         deleteObj = index;
            } else {
                if (isLight) { ed.selectedLight = index; ed.selectedObj = -1; }
                else         { ed.selectedObj = index; ed.selectedLight = -1; }
            }
        }
        ImGui::PopID();
    };

    for (int i = 0; i < (int)ed.scene.objects.size(); ++i) {
        const auto& o = ed.scene.objects[i];
        drawRow(i, o.name.c_str(), ObjectKindLabel(o), ed.selectedObj == i, false, i);
    }
    for (int i = 0; i < (int)ed.scene.lights.size(); ++i) {
        const auto& l = ed.scene.lights[i];
        const char* typeName = (l.type == LightType::Spot) ? "spot"
                             : (l.type == LightType::Sun) ? "sun" : "point";
        drawRow(10000 + i, l.name.c_str(), typeName, ed.selectedLight == i, true, i);
    }

    if (ed.scene.objects.empty() && ed.scene.lights.empty())
        ImGui::TextColored(kTextMuted, "  Scene is empty. Use Add.");

    if (deleteObj >= 0) {
        ed.scene.removeObject(deleteObj);
        if (ed.selectedObj == deleteObj) ed.selectedObj = -1;
        else if (ed.selectedObj > deleteObj) ed.selectedObj--;
        ed.gizmoDrag.active = false;
        ed.hoverAxis = -1;
        ed.regenerate();
    }
    if (deleteLight >= 0) {
        ed.scene.removeLight(deleteLight);
        if (ed.selectedLight == deleteLight) ed.selectedLight = -1;
        else if (ed.selectedLight > deleteLight) ed.selectedLight--;
        ed.gizmoDrag.active = false;
        ed.hoverAxis = -1;
        ed.regenerate();
    }
}

// -----------------------------------------------------------------------------
// Asset browser (bottom downbar)
// -----------------------------------------------------------------------------
static void DrawAssetsBody(Editor& ed, const std::filesystem::path& assetRoot,
                           std::filesystem::path& browserPath,
                           std::filesystem::path& selectedPath) {
    // Breadcrumb toolbar
    const bool canUp = !assetRoot.empty() && browserPath != assetRoot &&
                       !browserPath.parent_path().empty();
    if (!canUp) ImGui::BeginDisabled();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 2));
    if (ImGui::Button("Up", ImVec2(38, 20)) && canUp) {
        browserPath = browserPath.parent_path();
        selectedPath.clear();
    }
    ImGui::PopStyleVar();
    if (!canUp) ImGui::EndDisabled();

    ImGui::SameLine(0, 8);
    ImGui::AlignTextToFramePadding();
    const std::string pathLabel = browserPath.empty() ? "assets" : browserPath.string();
    ImGui::TextColored(ImVec4(0.62f, 0.62f, 0.62f, 1.0f), "%s", pathLabel.c_str());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pathLabel.c_str());

    ImGui::Dummy(ImVec2(0, 1));

    // Add strip
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.24f, 0.24f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.27f, 0.27f, 0.27f, 1.0f));
    if (ImGui::Button("##assetAdd", ImVec2(-1, 20)))
        ImGui::OpenPopup("##assetAdd");
    ImGui::PopStyleColor(3);
    {
        const ImVec2 amin = ImGui::GetItemRectMin();
        const ImVec2 amax = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddText(
            ImVec2(amin.x + 10.0f, amin.y + (amax.y - amin.y - ImGui::GetTextLineHeight()) * 0.5f),
            IM_COL32(212, 212, 212, 255), "Add");
        DrawChevron(ImGui::GetWindowDrawList(),
                    ImVec2(amax.x - 14.0f, amin.y + 10.0f), 8.0f, true,
                    IM_COL32(170, 170, 170, 255));
    }
    if (ImGui::BeginPopup("##assetAdd")) {
        if (ImGui::MenuItem("Import Model  (.obj / .fbx)")) ImportModelDialog(ed);
        if (ImGui::MenuItem("Import Texture")) ImportTextureDialog(ed);
        ImGui::EndPopup();
    }

    ImGui::Dummy(ImVec2(0, 2));

    ImGui::BeginChild("##assetTiles", ImVec2(0, 0), false);
    if (browserPath.empty()) {
        ImGui::TextDisabled("Assets folder not found.");
        ImGui::EndChild();
        return;
    }

    std::vector<std::pair<std::filesystem::path, bool>> entries;
    std::error_code ec;
    std::filesystem::directory_iterator it(browserPath, ec);
    const std::filesystem::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        std::error_code typeError;
        const bool isDirectory = it->is_directory(typeError);
        if (typeError) { ec = typeError; break; }
        entries.emplace_back(it->path(), isDirectory);
    }
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
        if (left.second != right.second) return left.second > right.second;
        return left.first.filename().string() < right.first.filename().string();
    });

    if (ec) {
        ImGui::TextDisabled("Unable to read folder: %s", ec.message().c_str());
    } else if (entries.empty()) {
        ImGui::TextDisabled("This folder is empty.");
    } else {
        const float tileW = 104.0f;
        const float tileH = 88.0f;
        const float gap = 6.0f;
        const float availW = ImGui::GetContentRegionAvail().x;
        int perRow = (int)((availW + gap) / (tileW + gap));
        if (perRow < 1) perRow = 1;

        for (int i = 0; i < (int)entries.size(); ++i) {
            const auto& entry = entries[i];
            const std::string name = entry.first.filename().string();
            const bool isDir = entry.second;
            std::string ext = entry.first.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            const bool isSelected = selectedPath == entry.first;

            ImGui::PushID(entry.first.string().c_str());
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 p = ImGui::GetCursorScreenPos();

            ImGui::InvisibleButton("##tile", ImVec2(tileW, tileH));
            const bool hovered = ImGui::IsItemHovered();
            const bool clicked = ImGui::IsItemClicked();
            const bool doubleClicked = hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

            ImU32 bg = isSelected ? IM_COL32(54, 60, 68, 255)
                                  : (hovered ? IM_COL32(47, 47, 47, 255)
                                             : IM_COL32(40, 40, 40, 255));
            dl->AddRectFilled(p, ImVec2(p.x + tileW, p.y + tileH), bg, 2.0f);
            if (isSelected)
                dl->AddRect(p, ImVec2(p.x + tileW, p.y + tileH), ImGui::GetColorU32(kAccent), 2.0f, 0, 1.4f);

            const ImU32 iconCol = isDir ? IM_COL32(205, 178, 120, 255)
                                        : IM_COL32(170, 175, 185, 255);
            DrawAssetGlyph(dl, ImVec2(p.x + tileW * 0.5f, p.y + 32.0f), 34.0f,
                           isDir, ext, iconCol);

            const std::string label = EllipsizeText(name, tileW - 12.0f);
            const float labelW = ImGui::CalcTextSize(label.c_str()).x;
            dl->AddText(ImVec2(p.x + (tileW - labelW) * 0.5f, p.y + tileH - 22.0f),
                        isSelected ? IM_COL32(240, 240, 240, 255) : IM_COL32(200, 200, 200, 255),
                        label.c_str());

            if (hovered) ImGui::SetTooltip("%s", entry.first.string().c_str());

            if (clicked) selectedPath = entry.first;

            if (doubleClicked) {
                if (isDir) {
                    browserPath = entry.first;
                    selectedPath.clear();
                } else if (ext == ".obj" || ext == ".fbx") {
                    LoadModelFile(ed, entry.first.string());
                } else if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
                           ext == ".bmp" || ext == ".tga") {
                    RegisterTextureFile(ed, entry.first.string());
                }
            }

            ImGui::PopID();
            if ((i + 1) % perRow != 0 && i + 1 < (int)entries.size())
                ImGui::SameLine(0, gap);
        }
    }
    ImGui::EndChild();
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------
int main() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(1400, 800, "Vektor Engine", nullptr, nullptr);
    if (!win) return 1;
    glfwSetWindowSizeLimits(win, 720, 480, GLFW_DONT_CARE, GLFW_DONT_CARE);
    SetWindowIconFromFile(win, "logo.png");
#ifdef _WIN32
    {
        HWND hwnd = glfwGetWin32Window(win);
        BOOL useDarkTitle = TRUE;
        DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                              &useDarkTitle, sizeof(useDarkTitle));
    }
#endif
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
#ifdef _WIN32
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
#endif

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return 1;
    glGetError();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImFont* font = io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 15.0f);
    if (font == nullptr) io.Fonts->AddFontDefault();

    ApplyLauraTheme();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    Editor ed;
    ed.renderer.init();

    // UI textures
    GLuint logoTex = 0;
    GLuint playTex = 0;
    GLuint folderTex = 0;
    GLuint boxTex = 0;
    GLuint plusTex = 0;
    GLuint hierarchyTex = 0;
    GLuint settingsTex = 0;
    LoadUIAsset("logo.png", logoTex);
    LoadUIAsset("play.png", playTex);
    LoadUIAsset("folder.png", folderTex);
    LoadUIAsset("box.png", boxTex);
    LoadUIAsset("plus.png", plusTex);
    LoadUIAsset("hirearchy.png", hierarchyTex);
    LoadUIAsset("settings.png", settingsTex);

    std::filesystem::path assetRoot = FindAssetsDirectory();
    std::filesystem::path assetBrowserPath = assetRoot;
    std::filesystem::path selectedAssetPath;

    // --- Startup project dialog state ---
    bool projectChosen = false;
    std::string projectError;
    char projectPathBuf[1024] = {0};
    {
        const std::string initialProject = assetRoot.empty()
            ? std::filesystem::current_path().string()
            : assetRoot.string();
        std::snprintf(projectPathBuf, sizeof(projectPathBuf), "%s", initialProject.c_str());
    }
    bool focusProjectPath = true;

    auto acceptProject = [&]() {
        std::error_code ec;
        std::filesystem::path path(projectPathBuf);
        if (path.empty() || !std::filesystem::is_directory(path, ec) || ec) {
            projectError = "Folder not found. Please check the path.";
            return false;
        }
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        assetRoot = (ec ? path : absolute).lexically_normal();
        assetBrowserPath = assetRoot;
        selectedAssetPath.clear();
        projectError.clear();
        projectChosen = true;
        const std::string title = "Vektor Engine  -  " + assetRoot.filename().string();
        glfwSetWindowTitle(win, title.c_str());
        return true;
    };

    ed.scene.addObject(ObjType::Cube, Vec3(0, 0.5f, 0), {1,1,1}, {0.85f,0.55f,0.35f});
    ed.scene.addObject(ObjType::Sphere, Vec3(2, 0.5f, 0), {1,1,1}, {0.40f,0.65f,0.90f});
    ed.scene.addPointLight(Vec3(-2, 3, 2), {1.0f,0.95f,0.85f}, 12.0f);
    ed.regenerate();

    char scriptBuf[16384] = {0};
    double lastTime = glfwGetTime();
    int rightTab = 0;

    auto applyScript = [&]() {
        ed.scene.loadScript(scriptBuf);
        ed.renderer.reloadAssets(ed.scene.models, ed.scene.textures);
        ed.selectedObj = -1;
        ed.selectedLight = -1;
        ed.gizmoDrag.active = false;
        ed.hoverAxis = -1;
        ed.scriptText = scriptBuf;
    };

    auto focusSelection = [&]() {
        Vec3 target{0,0,0};
        bool hasTarget = false;
        if (ed.selectedObj >= 0 && ed.selectedObj < (int)ed.scene.objects.size()) {
            target = ed.scene.objects[ed.selectedObj].pos;
            hasTarget = true;
        } else if (ed.selectedLight >= 0 && ed.selectedLight < (int)ed.scene.lights.size()) {
            target = ed.scene.lights[ed.selectedLight].pos;
            hasTarget = true;
        }
        if (!hasTarget) return;
        Vec3 dir = normalize(target - ed.camera.pos);
        ed.camera.yaw = std::atan2(dir.x, -dir.z) * 180.0f / 3.14159265f;
        ed.camera.pitch = std::asin(std::clamp(dir.y, -1.0f, 1.0f)) * 180.0f / 3.14159265f;
        ed.camera.pos = target - dir * 5.0f;
    };

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        double now = glfwGetTime();
        float dt = (float)(now - lastTime);
        if (dt > 0.1f) dt = 0.1f;
        lastTime = now;

        int winW, winH;
        glfwGetFramebufferSize(win, &winW, &winH);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (ed.scriptRegen) {
            std::strncpy(scriptBuf, ed.scriptText.c_str(), sizeof(scriptBuf)-1);
            scriptBuf[sizeof(scriptBuf)-1] = '\0';
            ed.scriptRegen = false;
        }

        // ================= STARTUP PROJECT DIALOG =================
        if (!projectChosen) {
            DrawAppTitleBar(win, logoTex, "Open Project", false, []() {});

            ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
                                    ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(520, 0), ImGuiCond_Always);
            ImGui::Begin("Open Project", nullptr,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove);

            ImGui::TextColored(kTextMuted, "Choose the project folder to open.");
            ImGui::Dummy(ImVec2(0, 6));
            ImGui::TextUnformatted("Project folder");
            ImGui::Dummy(ImVec2(0, 2));

            if (focusProjectPath) {
                ImGui::SetKeyboardFocusHere();
                focusProjectPath = false;
            }
            ImGui::SetNextItemWidth(-1);
            bool open = ImGui::InputText("##projectPath", projectPathBuf, sizeof(projectPathBuf),
                                         ImGuiInputTextFlags_EnterReturnsTrue);

            ImGui::Dummy(ImVec2(0, 2));
            if (ImGui::Button("Browse...", ImVec2(96, 26))) {
                const std::string chosen = BrowseForFolderDialog("Select project folder");
                if (!chosen.empty()) {
                    std::snprintf(projectPathBuf, sizeof(projectPathBuf), "%s", chosen.c_str());
                    projectError.clear();
                }
            }
            ImGui::SameLine(0, 6);
            ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kAccentHover);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.18f, 0.42f, 0.80f, 1.0f));
            if (ImGui::Button("Open", ImVec2(96, 26))) open = true;
            ImGui::PopStyleColor(3);
            ImGui::SameLine(0, 6);
            if (ImGui::Button("Cancel", ImVec2(96, 26)))
                glfwSetWindowShouldClose(win, GLFW_TRUE);

            if (!projectError.empty()) {
                ImGui::Dummy(ImVec2(0, 2));
                ImGui::TextColored(ImVec4(0.90f, 0.36f, 0.36f, 1.0f), "%s", projectError.c_str());
            }
            ImGui::Dummy(ImVec2(0, 4));
            ImGui::Separator();
            ImGui::TextColored(kTextMuted, "The Assets panel will show the files in this folder.");
            ImGui::End();

            if (open) acceptProject();

            HandleWindowBorderResize(win);

            ImGui::Render();
            glViewport(0, 0, winW, winH);
            glClearColor(0.09f, 0.09f, 0.09f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(win);
            continue;
        }

        // ================= TITLE BAR =================
        const std::string projectLabel = assetRoot.empty()
            ? std::string("No Project")
            : assetRoot.filename().string();
        DrawAppTitleBar(win, logoTex, projectLabel, true, [&]() {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Import Model...")) ImportModelDialog(ed);
                if (ImGui::MenuItem("Import Texture...")) ImportTextureDialog(ed);
                ImGui::Separator();
                if (ImGui::MenuItem("Apply & Recompile")) applyScript();
                ImGui::Separator();
                if (ImGui::MenuItem("Exit")) glfwSetWindowShouldClose(win, GLFW_TRUE);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                if (ImGui::MenuItem("Reset Camera")) {
                    ed.camera.pos = Vec3(4.0f, 4.0f, 6.0f);
                    ed.camera.yaw = -33.7f;
                    ed.camera.pitch = -25.5f;
                    ed.camera.speed = 8.0f;
                }
                if (ImGui::MenuItem("Focus Selection", "F")) focusSelection();
                ImGui::Separator();
                if (ImGui::MenuItem("Move Gizmo", "1", ed.gizmoMode == GizmoMode::Move))
                    ed.gizmoMode = GizmoMode::Move;
                if (ImGui::MenuItem("Scale Gizmo", "2", ed.gizmoMode == GizmoMode::Scale))
                    ed.gizmoMode = GizmoMode::Scale;
                ImGui::EndMenu();
            }
        });

        ImGui::SetNextWindowPos(ImVec2(0, kTitleBarHeight));
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x,
                                       ImGui::GetIO().DisplaySize.y - kTitleBarHeight));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.09f, 0.09f, 0.09f, 1.0f));
        ImGui::Begin("Host", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();

        // Layout constants
        const float leftW = 240.0f;
        const float rightW = 300.0f;
        const float bottomH = 196.0f;

        // ================= LAYOUT =================
        const ImVec2 root = ImGui::GetCursorScreenPos();
        const float availW = ImGui::GetContentRegionAvail().x;
        const float availH = ImGui::GetContentRegionAvail().y;
        const float topH = std::max(180.0f, availH - bottomH);
        const float bottomActualH = std::max(120.0f, availH - topH);
        const float centerW = std::max(220.0f, availW - leftW - rightW);
        const float bottomW = std::max(260.0f, availW - rightW);

        // --- Scene Hierarchy ---
        ImGui::SetCursorScreenPos(root);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::BeginChild("##panelHierarchy", ImVec2(leftW, topH), true);
        ImGui::PopStyleVar();
        {
            PanelTabDef tabs[] = {{"Scene Hierarchy", hierarchyTex}};
            PanelHeader(tabs, 1, 0);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(7, 7));
            ImGui::BeginChild("##hierpBody", ImVec2(0, 0), false);
            ImGui::PopStyleVar();
            DrawHierarchyBody(ed);
            ImGui::EndChild();
        }
        ImGui::EndChild();

        // --- Viewport ---
        ImGui::SetCursorScreenPos(ImVec2(root.x + leftW, root.y));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::BeginChild("##panelViewport", ImVec2(centerW, topH), true);
        ImGui::PopStyleVar();
        {
            PanelTabDef tabs[] = {{"Viewport", boxTex}};
            PanelHeader(tabs, 1, 0);

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::BeginChild("##vpBody", ImVec2(0, 0), false);
            ImGui::PopStyleVar();

            const ImVec2 vpSize = ImGui::GetContentRegionAvail();
            const int vpW = (int)vpSize.x;
            const int vpH = (int)vpSize.y;

            if (vpW > 4 && vpH > 4) {
                ed.renderer.resizeFBO(vpW, vpH);
                bool hovered = ImGui::IsWindowHovered();
                ImGuiIO& gio = ImGui::GetIO();

                if (hovered && !gio.WantCaptureKeyboard) {
                    std::vector<std::string> triggeredKeys;
                    for (const auto& event : ed.scene.scriptEvents) {
                        if (event.type != "key" ||
                            std::find(triggeredKeys.begin(), triggeredKeys.end(), event.key) != triggeredKeys.end())
                            continue;
                        const ImGuiKey key = scriptKeyToImGuiKey(event.key);
                        if (key != ImGuiKey_None && ImGui::IsKeyPressed(key, false)) {
                            triggeredKeys.push_back(event.key);
                            ed.scene.runKeyEvent(event.key);
                        }
                    }
                }
                ed.scene.runFrameEvents(dt);

                // --- UE5 Style Navigation ---
                if (hovered) {
                    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                        ed.camera.yaw += gio.MouseDelta.x * 0.15f;
                        ed.camera.pitch -= gio.MouseDelta.y * 0.15f;
                        ed.camera.pitch = std::clamp(ed.camera.pitch, -89.0f, 89.0f);
                    }
                    if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                        ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
                        float panSpeed = ed.camera.speed * 0.005f;
                        ed.camera.pos -= ed.camera.right() * (d.x * panSpeed);
                        ed.camera.pos += ed.camera.up() * (d.y * panSpeed);
                        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
                    }
                    if (gio.MouseWheel != 0.0f) {
                        ed.camera.speed *= std::pow(1.2f, gio.MouseWheel);
                        ed.camera.speed = std::clamp(ed.camera.speed, 0.1f, 100.0f);
                    }
                }

                // WASD Fly
                bool rightMouseDown = ImGui::IsMouseDown(ImGuiMouseButton_Right);
                if ((hovered || rightMouseDown) && !gio.WantTextInput) {
                    float moveSpeed = ed.camera.speed * dt;
                    if (ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift))
                        moveSpeed *= 4.0f;
                    if (ImGui::IsKeyDown(ImGuiKey_W)) ed.camera.pos += ed.camera.forward() * moveSpeed;
                    if (ImGui::IsKeyDown(ImGuiKey_S)) ed.camera.pos -= ed.camera.forward() * moveSpeed;
                    if (ImGui::IsKeyDown(ImGuiKey_A)) ed.camera.pos -= ed.camera.right() * moveSpeed;
                    if (ImGui::IsKeyDown(ImGuiKey_D)) ed.camera.pos += ed.camera.right() * moveSpeed;
                    if (ImGui::IsKeyDown(ImGuiKey_E)) ed.camera.pos += ed.camera.up() * moveSpeed;
                    if (ImGui::IsKeyDown(ImGuiKey_Q)) ed.camera.pos -= ed.camera.up() * moveSpeed;
                }

                // F to focus on selected object
                if (!gio.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_F))
                    focusSelection();

                // Gizmo mode shortcuts
                if (!gio.WantCaptureKeyboard && !gio.WantTextInput) {
                    if (ImGui::IsKeyPressed(ImGuiKey_1, false)) ed.gizmoMode = GizmoMode::Move;
                    if (ImGui::IsKeyPressed(ImGuiKey_2, false)) ed.gizmoMode = GizmoMode::Scale;
                }

                float aspect = (float)vpW / (float)vpH;
                Mat4 proj = Mat4::perspective(60.0f * 3.14159265f / 180.0f, aspect, 0.05f, 500.0f);
                Mat4 view = ed.camera.view();
                Vec3 camPos = ed.camera.pos;

                ed.renderer.setLights(ed.scene.lights);
                ed.renderer.bindFBO();
                ed.renderer.beginFrame(view, proj, camPos);
                glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                ed.renderer.drawGrid();
                for (auto& o : ed.scene.objects) {
                    if (o.collisionOnly) {
                        ed.renderer.drawWireAABB(o.pos - o.size*0.5f, o.pos + o.size*0.5f, Vec3(0.2f,0.8f,1.0f));
                        continue;
                    }
                    Mat4 M = modelMatrix(o);
                    const char* prim = o.modelId.empty() ? ((o.type == ObjType::Sphere) ? "sphere" : (o.type == ObjType::Quad) ? "quad" : "cube") : o.modelId.c_str();
                    const Material* material = nullptr;
                    if (!o.materialId.empty()) {
                        auto mit = ed.scene.materials.find(o.materialId);
                        if (mit != ed.scene.materials.end()) material = &mit->second;
                    }
                    Vec3 color = o.color;
                    MaterialParams materialParams;
                    if (material) {
                        color = Vec3(color.x * material->color.x,
                                     color.y * material->color.y,
                                     color.z * material->color.z);
                        materialParams.texId = material->texId;
                        materialParams.normalTexId = material->normalTexId;
                        materialParams.metallic = material->metallic;
                        materialParams.smoothness = material->smoothness;
                        materialParams.reflectivity = material->reflectivity;
                    }
                    if (ed.renderer.hasFBXModel(prim))
                        ed.renderer.drawFBXModel(prim, M, color, o.materialTexIds, o.uvScale,
                                                 o.textureWrap, o.textureFilter,
                                                 material ? &materialParams : nullptr);
                    else
                        ed.renderer.drawMesh(prim, M, color, materialParams, o.uvScale,
                                             o.textureWrap, o.textureFilter);
                }

                // Render Lights and Helpers
                for (auto& l : ed.scene.lights) {
                    if (l.type == LightType::Sun) continue;
                    ed.renderer.drawWireSphere(l.pos, 0.12f, l.color * 1.3f, 12);
                    if (l.type == LightType::Point)
                        ed.renderer.drawWireSphere(l.pos, l.range, l.color * 0.35f, 24);
                    if (l.type == LightType::Spot) {
                        float L = 1.2f;
                        float rad = L * std::tan(l.outerDeg * 3.14159265f / 180.0f);
                        Vec3 d = normalize(l.dir);
                        Vec3 up = std::fabs(d.y) < 0.99f ? Vec3(0,1,0) : Vec3(1,0,0);
                        Vec3 r1 = normalize(cross(d, up));
                        Vec3 r2 = cross(d, r1);
                        Vec3 tip = l.pos + d * L;
                        std::vector<float> segs;
                        const int N = 16;
                        for (int i = 0; i < N; ++i) {
                            float a0 = 6.2831853f * i / N, a1 = 6.2831853f * (i+1) / N;
                            Vec3 p0 = tip + (r1*std::cos(a0) + r2*std::sin(a0)) * rad;
                            Vec3 p1 = tip + (r1*std::cos(a1) + r2*std::sin(a1)) * rad;
                            segs.insert(segs.end(), {p0.x,p0.y,p0.z, 0,1,0, 0,0, p1.x,p1.y,p1.z, 0,1,0, 0,0});
                            segs.insert(segs.end(), {l.pos.x,l.pos.y,l.pos.z, 0,1,0, 0,0, p0.x,p0.y,p0.z, 0,1,0, 0,0});
                        }
                        ed.renderer.drawLineSegments(segs, l.color * 0.9f);
                    }
                }

                if (ed.selectedObj >= 0 && ed.selectedObj < (int)ed.scene.objects.size()) {
                    auto& o = ed.scene.objects[ed.selectedObj];
                    drawObjectOutline(ed.renderer, o, Vec3(1.0f, 0.75f, 0.22f));
                }
                if (ed.selectedLight >= 0 && ed.selectedLight < (int)ed.scene.lights.size()) {
                    auto& l = ed.scene.lights[ed.selectedLight];
                    if (l.type != LightType::Sun)
                        ed.renderer.drawWireSphere(l.pos, 0.18f, Vec3(1.0f, 0.85f, 0.20f), 12);
                }

                // --- 3D transform gizmo ---
                {
                    bool hasGizmo = false;
                    bool gizmoIsLight = false;
                    Vec3 gizmoOrigin{0, 0, 0};
                    if (ed.selectedObj >= 0 && ed.selectedObj < (int)ed.scene.objects.size()) {
                        gizmoOrigin = ed.scene.objects[ed.selectedObj].pos;
                        hasGizmo = true;
                    } else if (ed.selectedLight >= 0 && ed.selectedLight < (int)ed.scene.lights.size()) {
                        gizmoOrigin = ed.scene.lights[ed.selectedLight].pos;
                        hasGizmo = true;
                        gizmoIsLight = true;
                    }
                    if (hasGizmo) {
                        drawGizmo(ed.renderer, gizmoOrigin,
                                  gizmoIsLight ? GizmoMode::Move : ed.gizmoMode,
                                  ed.hoverAxis,
                                  ed.gizmoDrag.active ? ed.gizmoDrag.axis : -1,
                                  ed.camera, proj, vpH);
                    }
                }
                ed.renderer.unbindFBO();

                ImVec2 imgOrigin = ImGui::GetCursorScreenPos();
                if (ed.renderer.fboReady()) {
                    ImGui::Image((ImTextureID)(intptr_t)ed.renderer.fboColorTex(), vpSize,
                                 ImVec2(0,1), ImVec2(1,0));
                } else {
                    ImGui::Dummy(vpSize);
                }

                // Floating gizmo mode switch
                const ImVec2 savedCursor = ImGui::GetCursorPos();
                ImGui::SetCursorScreenPos(ImVec2(imgOrigin.x + 10, imgOrigin.y + 10));
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
                if (ViewportModeButton("##vpMove", "Move", ed.gizmoMode == GizmoMode::Move))
                    ed.gizmoMode = GizmoMode::Move;
                const ImVec2 toolsMin = ImGui::GetItemRectMin();
                ImGui::SameLine(0, 4);
                if (ViewportModeButton("##vpScale", "Scale", ed.gizmoMode == GizmoMode::Scale))
                    ed.gizmoMode = GizmoMode::Scale;
                const ImVec2 toolsMax = ImGui::GetItemRectMax();
                ImGui::PopStyleVar();
                ImGui::SetCursorPos(savedCursor);
                const bool toolbarHovered = ImGui::IsMouseHoveringRect(toolsMin, toolsMax);

                // Viewport hint line
                {
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    const char* hint = "RMB Look    MMB Pan    Scroll Speed    F Focus    Del Delete    Ctrl Snap";
                    dl->AddText(ImVec2(imgOrigin.x + 12.0f, imgOrigin.y + vpSize.y - 19.0f),
                                IM_COL32(128, 132, 140, 210), hint);
                }

                // Viewport interactions
                ImVec2 mouse = ImGui::GetMousePos();
                float lmX = mouse.x - imgOrigin.x;
                float lmY = mouse.y - imgOrigin.y;
                bool inside = hovered && lmX >= 0 && lmY >= 0 && lmX < vpW && lmY < vpH;

                bool hasGizmo = false;
                bool gizmoIsLight = false;
                Vec3 gizmoOrigin{0, 0, 0};
                if (ed.selectedObj >= 0 && ed.selectedObj < (int)ed.scene.objects.size()) {
                    gizmoOrigin = ed.scene.objects[ed.selectedObj].pos;
                    hasGizmo = true;
                } else if (ed.selectedLight >= 0 && ed.selectedLight < (int)ed.scene.lights.size()) {
                    gizmoOrigin = ed.scene.lights[ed.selectedLight].pos;
                    hasGizmo = true;
                    gizmoIsLight = true;
                }
                const GizmoMode effectiveMode = gizmoIsLight ? GizmoMode::Move : ed.gizmoMode;

                if (!ed.gizmoDrag.active) {
                    ed.hoverAxis = (hasGizmo && inside && !toolbarHovered)
                        ? pickGizmoAxis(ImVec2(lmX, lmY), gizmoOrigin, effectiveMode,
                                        ed.camera, view, proj, vpW, vpH)
                        : -1;
                }
                if (inside && ed.hoverAxis >= 0 && !ed.gizmoDrag.active)
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

                if (ed.placeMode == PlaceMode::None && inside && !toolbarHovered &&
                    ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                    !ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                    if (hasGizmo && ed.hoverAxis >= 0) {
                        GizmoDrag d;
                        d.active = true;
                        d.mode = effectiveMode;
                        d.axis = ed.hoverAxis;
                        d.isLight = gizmoIsLight;
                        d.startPos = gizmoOrigin;
                        d.startMouse = mouse;
                        d.planePoint = gizmoOrigin;
                        if (!gizmoIsLight) d.startSize = ed.scene.objects[ed.selectedObj].size;

                        const Ray r = screenRay(lmX, lmY, vpW, vpH, view, proj);
                        if (d.axis >= 0 && d.axis <= 2) {
                            const Vec3 dir = gizmoAxisDir(d.axis);
                            const Vec3 viewDir = normalize(gizmoOrigin - ed.camera.pos);
                            Vec3 n = viewDir - dir * dot(viewDir, dir);
                            if (length(n) < 1e-4f) n = cross(dir, Vec3(0, 1, 0));
                            if (length(n) < 1e-4f) n = Vec3(0, 0, 1);
                            d.planeNormal = normalize(n);
                            d.axisDir = dir;
                            Vec3 hit;
                            d.startAxisParam = rayPlanePoint(r, d.planePoint, d.planeNormal, hit)
                                ? dot(hit - d.planePoint, dir) : 0.0f;
                        } else {
                            d.planeNormal = normalize(ed.camera.pos - gizmoOrigin);
                            Vec3 hit;
                            rayPlanePoint(r, d.planePoint, d.planeNormal, hit);
                            d.startHit = hit;
                        }
                        ed.gizmoDrag = d;
                        ed.hoverAxis = d.axis;
                    } else {
                        Ray r = screenRay(lmX, lmY, vpW, vpH, view, proj);
                        int hitObj = -1, hitLight = -1;
                        float bestT = 1e30f;

                        for (int i = 0; i < (int)ed.scene.objects.size(); ++i) {
                            Vec3 mn, mx;
                            objectWorldAABB(ed.scene.objects[i], mn, mx);
                            float t;
                            if (rayAABB(r, mn, mx, t) && t < bestT) { bestT = t; hitObj = i; hitLight = -1; }
                        }
                        for (int i = 0; i < (int)ed.scene.lights.size(); ++i) {
                            auto& l = ed.scene.lights[i];
                            if (l.type == LightType::Sun) continue;
                            float t;
                            if (rayAABB(r, l.pos - Vec3(0.15f,0.15f,0.15f), l.pos + Vec3(0.15f,0.15f,0.15f), t) && t < bestT) {
                                bestT = t; hitObj = -1; hitLight = i;
                            }
                        }
                        ed.selectedObj = hitObj;
                        ed.selectedLight = hitLight;
                    }
                }

                if (ed.gizmoDrag.active) {
                    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                        const GizmoDrag d = ed.gizmoDrag;
                        const Ray r = screenRay(lmX, lmY, vpW, vpH, view, proj);
                        Vec3 newPos = d.startPos;
                        Vec3 newSize = d.startSize;
                        bool changed = false;
                        const bool snap = ImGui::GetIO().KeyCtrl;

                        if (d.axis >= 0 && d.axis <= 2) {
                            Vec3 hit;
                            if (rayPlanePoint(r, d.planePoint, d.planeNormal, hit)) {
                                float delta = dot(hit - d.planePoint, d.axisDir) - d.startAxisParam;
                                if (snap) delta = std::round(delta / 0.25f) * 0.25f;
                                if (d.mode == GizmoMode::Move) {
                                    newPos = d.startPos + d.axisDir * delta;
                                } else {
                                    float* component = (d.axis == 0) ? &newSize.x : (d.axis == 1) ? &newSize.y : &newSize.z;
                                    *component = std::max(0.01f, *component + delta);
                                }
                                changed = true;
                            }
                        } else {
                            Vec3 hit;
                            if (rayPlanePoint(r, d.planePoint, d.planeNormal, hit)) {
                                if (d.mode == GizmoMode::Move) {
                                    newPos = d.startPos + (hit - d.startHit);
                                } else {
                                    const float dx = mouse.x - d.startMouse.x;
                                    const float dy = mouse.y - d.startMouse.y;
                                    const float factor = std::clamp(1.0f + (dx - dy) * 0.006f, 0.01f, 100.0f);
                                    newSize = Vec3(std::max(0.01f, d.startSize.x * factor),
                                                   std::max(0.01f, d.startSize.y * factor),
                                                   std::max(0.01f, d.startSize.z * factor));
                                    if (snap) {
                                        newSize.x = std::max(0.1f, std::round(newSize.x * 10.0f) / 10.0f);
                                        newSize.y = std::max(0.1f, std::round(newSize.y * 10.0f) / 10.0f);
                                        newSize.z = std::max(0.1f, std::round(newSize.z * 10.0f) / 10.0f);
                                    }
                                }
                                changed = true;
                            }
                        }

                        if (changed) {
                            bool applied = false;
                            if (d.isLight) {
                                if (ed.selectedLight >= 0 && ed.selectedLight < (int)ed.scene.lights.size()) {
                                    ed.scene.lights[ed.selectedLight].pos = newPos;
                                    applied = true;
                                }
                            } else if (ed.selectedObj >= 0 && ed.selectedObj < (int)ed.scene.objects.size()) {
                                auto& o = ed.scene.objects[ed.selectedObj];
                                o.pos = newPos;
                                if (d.mode == GizmoMode::Scale) o.size = newSize;
                                applied = true;
                            }
                            if (applied) ed.regenerate();
                        }
                    } else {
                        ed.gizmoDrag.active = false;
                    }
                }

                if (inside && ImGui::IsKeyPressed(ImGuiKey_Delete)) {
                    if (ed.selectedObj >= 0) { ed.scene.removeObject(ed.selectedObj); ed.selectedObj = -1; ed.regenerate(); }
                    else if (ed.selectedLight >= 0) { ed.scene.removeLight(ed.selectedLight); ed.selectedLight = -1; ed.regenerate(); }
                }
            }

            ImGui::EndChild();
        }
        ImGui::EndChild();

        // --- Assets downbar ---
        ImGui::SetCursorScreenPos(ImVec2(root.x, root.y + topH));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::BeginChild("##panelAssets", ImVec2(bottomW, bottomActualH), true);
        ImGui::PopStyleVar();
        {
            PanelTabDef tabs[] = {{"Assets", folderTex}};
            PanelHeader(tabs, 1, 0);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));
            ImGui::BeginChild("##assetsBody", ImVec2(0, 0), false);
            ImGui::PopStyleVar();
            DrawAssetsBody(ed, assetRoot, assetBrowserPath, selectedAssetPath);
            ImGui::EndChild();
        }
        ImGui::EndChild();

        // --- Inspector / Script ---
        ImGui::SetCursorScreenPos(ImVec2(root.x + availW - rightW, root.y));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::BeginChild("##panelInspector", ImVec2(rightW, availH), true);
        ImGui::PopStyleVar();
        {
            PanelTabDef tabs[] = {{"Inspector", settingsTex}, {"Script", 0}};
            const int clicked = PanelHeader(tabs, 2, rightTab);
            if (clicked >= 0) rightTab = clicked;

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            ImGui::BeginChild("##rightBody", ImVec2(0, 0), false);
            ImGui::PopStyleVar();

            if (rightTab == 0) {
                DrawInspectorBody(ed);
            } else {
                ImGui::TextColored(kTextMuted, "Scene script");
                ImGui::Dummy(ImVec2(0, 2));
                const float scriptH = std::max(80.0f, ImGui::GetContentRegionAvail().y - 44.0f);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.10f, 0.10f, 0.10f, 1.0f));
                ImGui::InputTextMultiline("##script", scriptBuf, sizeof(scriptBuf),
                                          ImVec2(-1, scriptH), ImGuiInputTextFlags_AllowTabInput);
                ImGui::PopStyleColor();
                ImGui::Dummy(ImVec2(0, 4));
                ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kAccentHover);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.18f, 0.42f, 0.80f, 1.0f));
                if (ImGui::Button("Apply & Recompile", ImVec2(-1, 28)))
                    applyScript();
                ImGui::PopStyleColor(3);
            }

            ImGui::EndChild();
        }
        ImGui::EndChild();

        ImGui::End(); // Host

        HandleWindowBorderResize(win);

        ImGui::Render();
        glViewport(0, 0, winW, winH);
        glClearColor(0.09f, 0.09f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
    }

    ed.renderer.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
