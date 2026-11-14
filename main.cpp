// Minimal hardware ray-traced renderer: ReSTIR DI (RTXDI-style, many lights) + path-traced GI,
// driven by .scr scripts (spawn objects, OBJ models, textures, events). See world.hpp / scene.scr.
// Vulkan 1.2 + VK_KHR_ray_query, all passes are compute shaders. Target: RTX 5060 Ti.
//
// Controls: WASD move, Q/E down/up, Shift fast, mouse look, Esc quit, F5 reload script
//           1/2/3 full/direct/indirect   T temporal   Y spatial   G GI bounces   C DI candidates
//           M tile-list vs global light sampling   -/= exposure
//           N GI mode (path / surface cache / screen + sparse BVH fallback)   L area lights   H GI rate
//           B NRD denoising on/off   K camera collision on/off
// Args:     --script file.scr --w N --h N --novsync --validate --cutoff F

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <NRD.h>
#include <NRI.h>
#include <Extensions/NRIHelper.h>
#include <Extensions/NRIWrapperVK.h>
#include <NRDIntegration.h>
#include <NRDIntegration.hpp>

#ifdef RTENGINE_HAS_DLSS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <sl.h>
#include <sl_dlss.h>
#include <sl_helpers_vk.h>
#include <sl_matrix_helpers.h>
#include <sl_security.h>
#ifdef small
#undef small
#endif
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#endif

#include "world.hpp"
#include "shader_compiler.hpp"

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    std::fprintf(stderr, "Vulkan error %d at %s:%d (%s)\n", (int)r_, __FILE__, __LINE__, #x); std::abort(); } } while (0)

// ------------------------------------------------------------------ Vulkan helpers
static VkDevice g_dev;
static VkPhysicalDevice g_pd;
static VkPhysicalDeviceMemoryProperties g_mem;
static VkCommandPool g_pool;
static VkQueue g_queue;

static PFN_vkCreateAccelerationStructureKHR pfnCreateAS;
static PFN_vkGetAccelerationStructureBuildSizesKHR pfnGetASSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR pfnCmdBuildAS;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetASAddr;

struct Buffer { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceAddress addr = 0; void* map = nullptr; };
struct Image  { VkImage img = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; };

static uint32_t findMem(uint32_t bits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0) {
    for (int pass = 0; pass < 2; ++pass) {
        VkMemoryPropertyFlags want = pass == 0 ? (required | preferred) : required;
        for (uint32_t i = 0; i < g_mem.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (g_mem.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    std::fprintf(stderr, "no suitable memory type\n");
    std::abort();
}

static Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible) {
    Buffer b;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(g_dev, &ci, nullptr, &b.buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, b.buf, &mr);
    VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &fi;
    ai.allocationSize = mr.size;
    // Host buffers prefer DEVICE_LOCAL|HOST_VISIBLE (ReBAR) so shader reads stay in VRAM.
    ai.memoryTypeIndex = hostVisible
        ? findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        : findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(g_dev, &ai, nullptr, &b.mem));
    VK(vkBindBufferMemory(g_dev, b.buf, b.mem, 0));

    VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bi.buffer = b.buf;
    b.addr = vkGetBufferDeviceAddress(g_dev, &bi);
    if (hostVisible) VK(vkMapMemory(g_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}

static void destroyBuffer(Buffer& b) {
    vkDestroyBuffer(g_dev, b.buf, nullptr);
    vkFreeMemory(g_dev, b.mem, nullptr);
    b = {};
}

static Image createImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags extra = 0) {
    Image im;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1; ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | extra;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK(vkCreateImage(g_dev, &ci, nullptr, &im.img));

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev, im.img, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(g_dev, &ai, nullptr, &im.mem));
    VK(vkBindImageMemory(g_dev, im.img, im.mem, 0));

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK(vkCreateImageView(g_dev, &vi, nullptr, &im.view));
    return im;
}

static VkCommandBuffer beginOnce() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK(vkAllocateCommandBuffers(g_dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}
static void endOnce(VkCommandBuffer cb) {
    VK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK(vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE));
    VK(vkQueueWaitIdle(g_queue));
    vkFreeCommandBuffers(g_dev, g_pool, 1, &cb);
}

// ---- acceleration structures: created once at full capacity, rebuilt in place when the script changes the scene
struct Accel {
    VkAccelerationStructureKHR as = VK_NULL_HANDLE;
    Buffer storage, scratch;
    VkDeviceAddress scratchAddr = 0;
    VkAccelerationStructureTypeKHR type{};
};

static Accel createAccel(VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR& geom,
                         uint32_t maxPrims, uint32_t scratchAlign) {
    Accel a;
    a.type = type;
    VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = type;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = &geom;
    VkAccelerationStructureBuildSizesInfoKHR si{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pfnGetASSizes(g_dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &maxPrims, &si);

    a.storage = createBuffer(si.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false);
    VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    ci.buffer = a.storage.buf; ci.size = si.accelerationStructureSize; ci.type = type;
    VK(pfnCreateAS(g_dev, &ci, nullptr, &a.as));
    a.scratch = createBuffer(si.buildScratchSize + scratchAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    a.scratchAddr = (a.scratch.addr + scratchAlign - 1) & ~(VkDeviceAddress)(scratchAlign - 1);
    return a;
}

static void recordBuild(VkCommandBuffer cb, const Accel& a, VkAccelerationStructureGeometryKHR geom, uint32_t primCount) {
    if (a.type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)
        geom.geometry.triangles.maxVertex = std::max(primCount * 3u, 1u) - 1u;
    VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = a.type;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = &geom;
    bi.dstAccelerationStructure = a.as;
    bi.scratchData.deviceAddress = a.scratchAddr;
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
    pfnCmdBuildAS(cb, 1, &bi, &pRange);
}

// ---- script textures: one 2D-array image (64 layers of 512x512 sRGB + 10 mips)
struct TexArray { VkImage img = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkSampler sampler = VK_NULL_HANDLE; };

static TexArray createTexArray() {
    TexArray t;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_SRGB;
    ci.extent = {(uint32_t)TEX_SIZE, (uint32_t)TEX_SIZE, 1};
    ci.mipLevels = TEX_MIPS; ci.arrayLayers = MAX_TEX_LAYERS;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK(vkCreateImage(g_dev, &ci, nullptr, &t.img));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev, t.img, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(g_dev, &ai, nullptr, &t.mem));
    VK(vkBindImageMemory(g_dev, t.img, t.mem, 0));

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = t.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; vi.format = ci.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, TEX_MIPS, 0, MAX_TEX_LAYERS};
    VK(vkCreateImageView(g_dev, &vi, nullptr, &t.view));

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod = float(TEX_MIPS);
    VK(vkCreateSampler(g_dev, &si, nullptr, &t.sampler));

    VkCommandBuffer cb = beginOnce();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.img; b.subresourceRange = vi.subresourceRange;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    endOnce(cb);
    return t;
}

static void uploadTexture(const TexArray& t, const TexUpload& up) {
    Buffer st = createBuffer(up.data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    std::memcpy(st.map, up.data.data(), up.data.size());
    VkCommandBuffer cb = beginOnce();
    VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, TEX_MIPS, up.layer, 1};
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.img; b.subresourceRange = rng;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    std::vector<VkBufferImageCopy> regions(TEX_MIPS);
    for (int l = 0; l < TEX_MIPS; ++l) {
        regions[l] = {};
        regions[l].bufferOffset = texLevelOffset(l);
        regions[l].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)l, up.layer, 1};
        regions[l].imageExtent = {(uint32_t)(TEX_SIZE >> l), (uint32_t)(TEX_SIZE >> l), 1};
    }
    vkCmdCopyBufferToImage(cb, st.buf, t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)regions.size(), regions.data());
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    endOnce(cb);
    destroyBuffer(st);
}

static VkShaderModule loadShader(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open shader %s\n", p.string().c_str()); std::abort(); }
    size_t sz = (size_t)f.tellg();
    std::vector<uint32_t> code((sz + 3) / 4);
    f.seekg(0);
    f.read((char*)code.data(), (std::streamsize)sz);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sz; ci.pCode = code.data();
    VkShaderModule m;
    VK(vkCreateShaderModule(g_dev, &ci, nullptr, &m));
    return m;
}

static VkPipeline makeComputePipeline(VkPipelineLayout layout, const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spv.size() * 4; ci.pCode = spv.data();
    VkShaderModule m;
    if (vkCreateShaderModule(g_dev, &ci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    cpci.layout = layout;
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
    vkDestroyShaderModule(g_dev, m, nullptr);
    return p;
}

// ------------------------------------------------------------------ frame data (std140, mirrors FrameUBO)
struct FrameUBO {
    float camPos[4], camFwd[4], camRight[4], camUp[4];
    float prevPos[4], prevFwd[4], prevRight[4], prevUp[4];
    uint32_t misc0[4];
    uint32_t misc1[4];
    float params[4];
    uint32_t misc2[4];
    float params2[4];
    float gridMin[4];
    float gridInvCell[4];
    uint32_t gridDim[4];
    uint32_t misc3[4];
    uint32_t misc4[4];
    float params3[4];
    float skyZenith[4];
    float skyHorizon[4];
    float jitter[4];
};

struct Camera { V3 pos{0, 2.5f, 8.5f}; float yaw = 0, pitch = 0; };   // radians
struct Basis  { V3 fwd, right, up; };
static Basis basisOf(const Camera& c) {
    V3 fwd = {std::sin(c.yaw) * std::cos(c.pitch), std::sin(c.pitch), -std::cos(c.yaw) * std::cos(c.pitch)};
    V3 right = normalize(cross(fwd, V3{0, 1, 0}));
    return {fwd, right, cross(right, fwd)};
}
static void setV(float* d, V3 v) { d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = 0; }
static void setWorldToView(float* m, const Camera& c) {
    Basis b = basisOf(c);
    std::fill(m, m + 16, 0.0f);
    m[0] = b.right.x; m[1] = b.up.x; m[2] = b.fwd.x;
    m[4] = b.right.y; m[5] = b.up.y; m[6] = b.fwd.y;
    m[8] = b.right.z; m[9] = b.up.z; m[10] = b.fwd.z;
    m[12] = -dot(b.right, c.pos); m[13] = -dot(b.up, c.pos); m[14] = -dot(b.fwd, c.pos);
    m[15] = 1.0f;
}
static void setViewToClip(float* m, float tanHalfFov, float aspect) {
    constexpr float nearZ = 0.01f, farZ = 100000.0f;
    std::fill(m, m + 16, 0.0f);
    m[0] = 1.0f / (tanHalfFov * aspect);
    m[5] = 1.0f / tanHalfFov;
    m[10] = farZ / (farZ - nearZ);
    m[11] = 1.0f;
    m[14] = -(nearZ * farZ) / (farZ - nearZ);
}

#ifdef RTENGINE_HAS_DLSS
static sl::Constants makeDLSSConstants(const Camera& camera, float tanHalfFov, float aspect, sl::float2 jitter, bool reset) {
    Basis basis = basisOf(camera);
    sl::Constants constants{};
    constants.cameraPos = {camera.pos.x, camera.pos.y, camera.pos.z};
    constants.cameraFwd = {basis.fwd.x, basis.fwd.y, basis.fwd.z};
    constants.cameraRight = {basis.right.x, basis.right.y, basis.right.z};
    constants.cameraUp = {basis.up.x, basis.up.y, basis.up.z};
    constants.cameraNear = 0.01f;
    constants.cameraFar = 100000.0f;
    constants.cameraFOV = 2.0f * std::atan(tanHalfFov);
    constants.cameraAspectRatio = aspect;
    constants.depthInverted = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eTrue;
    constants.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.mvecScale = {1.0f, 1.0f};
    constants.jitterOffset = jitter;
    float projection[16];
    setViewToClip(projection, tanHalfFov, aspect);
    for (int row = 0; row < 4; ++row)
        constants.cameraViewToClip[row] = sl::float4(projection[row * 4], projection[row * 4 + 1],
                                                     projection[row * 4 + 2], projection[row * 4 + 3]);
    sl::recalculateCameraMatrices(constants);
    return constants;
}

#define RTENGINE_VK12_FEATURES(X) \
    X(samplerMirrorClampToEdge) X(drawIndirectCount) X(storageBuffer8BitAccess) X(uniformAndStorageBuffer8BitAccess) \
    X(storagePushConstant8) X(shaderBufferInt64Atomics) X(shaderSharedInt64Atomics) X(shaderFloat16) X(shaderInt8) \
    X(descriptorIndexing) X(shaderInputAttachmentArrayDynamicIndexing) X(shaderUniformTexelBufferArrayDynamicIndexing) \
    X(shaderStorageTexelBufferArrayDynamicIndexing) X(shaderUniformBufferArrayNonUniformIndexing) \
    X(shaderSampledImageArrayNonUniformIndexing) X(shaderStorageBufferArrayNonUniformIndexing) \
    X(shaderStorageImageArrayNonUniformIndexing) X(shaderInputAttachmentArrayNonUniformIndexing) \
    X(shaderUniformTexelBufferArrayNonUniformIndexing) X(shaderStorageTexelBufferArrayNonUniformIndexing) \
    X(descriptorBindingUniformBufferUpdateAfterBind) X(descriptorBindingSampledImageUpdateAfterBind) \
    X(descriptorBindingStorageImageUpdateAfterBind) X(descriptorBindingStorageBufferUpdateAfterBind) \
    X(descriptorBindingUniformTexelBufferUpdateAfterBind) X(descriptorBindingStorageTexelBufferUpdateAfterBind) \
    X(descriptorBindingUpdateUnusedWhilePending) X(descriptorBindingPartiallyBound) X(descriptorBindingVariableDescriptorCount) \
    X(runtimeDescriptorArray) X(samplerFilterMinmax) X(scalarBlockLayout) X(imagelessFramebuffer) \
    X(uniformBufferStandardLayout) X(shaderSubgroupExtendedTypes) X(separateDepthStencilLayouts) X(hostQueryReset) \
    X(timelineSemaphore) X(bufferDeviceAddress) X(bufferDeviceAddressCaptureReplay) X(bufferDeviceAddressMultiDevice) \
    X(vulkanMemoryModel) X(vulkanMemoryModelDeviceScope) X(vulkanMemoryModelAvailabilityVisibilityChains) \
    X(shaderOutputViewportIndex) X(shaderOutputLayer) X(subgroupBroadcastDynamicId)
#define RTENGINE_VK13_FEATURES(X) \
    X(robustImageAccess) X(inlineUniformBlock) X(descriptorBindingInlineUniformBlockUpdateAfterBind) \
    X(pipelineCreationCacheControl) X(privateData) X(shaderDemoteToHelperInvocation) X(shaderTerminateInvocation) \
    X(subgroupSizeControl) X(computeFullSubgroups) X(synchronization2) X(textureCompressionASTC_HDR) \
    X(shaderZeroInitializeWorkgroupMemory) X(dynamicRendering) X(shaderIntegerDotProduct) X(maintenance4)

static bool hasRequiredVulkanFeatures(const VkPhysicalDeviceVulkan12Features& requested,
                                      const VkPhysicalDeviceVulkan12Features& supported,
                                      const VkPhysicalDeviceVulkan13Features& requested13,
                                      const VkPhysicalDeviceVulkan13Features& supported13) {
#define CHECK12(name) if (requested.name && !supported.name) return false;
    RTENGINE_VK12_FEATURES(CHECK12)
#undef CHECK12
#define CHECK13(name) if (requested13.name && !supported13.name) return false;
    RTENGINE_VK13_FEATURES(CHECK13)
#undef CHECK13
    return true;
}

static void enableRequiredVulkanFeatures(VkPhysicalDeviceVulkan12Features& enabled,
                                         const VkPhysicalDeviceVulkan12Features& requested,
                                         VkPhysicalDeviceVulkan13Features& enabled13,
                                         const VkPhysicalDeviceVulkan13Features& requested13) {
#define ENABLE12(name) if (requested.name) enabled.name = VK_TRUE;
    RTENGINE_VK12_FEATURES(ENABLE12)
#undef ENABLE12
#define ENABLE13(name) if (requested13.name) enabled13.name = VK_TRUE;
    RTENGINE_VK13_FEATURES(ENABLE13)
#undef ENABLE13
}
#undef RTENGINE_VK12_FEATURES
#undef RTENGINE_VK13_FEATURES
#endif

static const char* kDefaultScript =
    "# built-in fallback scene (no scene.scr found)\n"
    "builtin cornell\n"
    "builtin pointlights 512\n"
    "builtin spots\n";

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
    uint32_t W = 1920, H = 1080;
    constexpr int NP = 10;   // compute passes per frame
    constexpr int NSTAT = NP + 1;
    bool vsync = true, validate = false;
    float tileCutoff = 0.002f;
    std::string scriptArg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--w" && i + 1 < argc) W = (uint32_t)std::atoi(argv[++i]);
        else if (a == "--h" && i + 1 < argc) H = (uint32_t)std::atoi(argv[++i]);
        else if (a == "--novsync") vsync = false;
        else if (a == "--validate") validate = true;
        else if (a == "--cutoff" && i + 1 < argc) tileCutoff = (float)std::atof(argv[++i]);
        else if (a == "--script" && i + 1 < argc) scriptArg = argv[++i];
    }
    std::filesystem::path exeDir = std::filesystem::absolute(argv[0]).parent_path();
    std::filesystem::path shaderDir = exeDir / "shaders";
    if (!std::filesystem::exists(shaderDir)) shaderDir = exeDir.parent_path() / "shaders";

    World world;
    Script script(world);
    world.assetDirs.push_back(exeDir.string());
    {
        std::vector<std::filesystem::path> candidates;
        if (!scriptArg.empty()) candidates.push_back(scriptArg);
        candidates.push_back("scene.scr");
        candidates.push_back(exeDir / "scene.scr");
        candidates.push_back(exeDir.parent_path() / "scene.scr");
        candidates.push_back(exeDir.parent_path().parent_path() / "scene.scr");
        bool loaded = false;
        for (auto& c : candidates)
            if (std::filesystem::exists(c)) { std::printf("script: %s\n", c.string().c_str()); loaded = script.loadFile(c.string()); break; }
        if (!loaded) { std::printf("script: no scene.scr found, using the built-in default scene\n"); script.loadText(kDefaultScript, "<built-in>"); }
    }
    script.runStart();

int activeDlssMode = 0;
#ifdef RTENGINE_HAS_DLSS
    bool streamlineInitialized = false;
    bool dlssActive = false;
    sl::FeatureRequirements dlssRequirements{};
    if (world.dlssMode > 0) {
        std::filesystem::path interposerPath = exeDir / "sl.interposer.dll";
        if (!sl::security::verifyEmbeddedSignature(interposerPath.wstring().c_str())) {
            std::fprintf(stderr, "DLSS disabled: Streamline interposer signature verification failed (%s)\n",
                         interposerPath.string().c_str());
        } else {
            sl::Feature feature = sl::kFeatureDLSS;
            const std::wstring pluginPath = exeDir.wstring();
            const wchar_t* pluginDirs[] = {pluginPath.c_str()};
            sl::Preferences preferences{};
            preferences.renderAPI = sl::RenderAPI::eVulkan;
            preferences.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
            preferences.featuresToLoad = &feature;
            preferences.numFeaturesToLoad = 1;
            preferences.pathsToPlugins = pluginDirs;
            preferences.numPathsToPlugins = 1;
            preferences.engineVersion = "0.1.0";
            if (slInit(preferences) != sl::Result::eOk) {
                std::fprintf(stderr, "DLSS disabled: Streamline initialization failed\n");
            } else {
                streamlineInitialized = true;
                if (slGetFeatureRequirements(sl::kFeatureDLSS, dlssRequirements) != sl::Result::eOk ||
                    !(dlssRequirements.flags & sl::FeatureRequirementFlags::eVulkanSupported)) {
                    std::fprintf(stderr, "DLSS disabled: Streamline DLSS Vulkan requirements are unavailable\n");
                    slShutdown();
                    streamlineInitialized = false;
                }
            }
        }
    }
#else
    if (world.dlssMode > 0) {
        std::fprintf(stderr, "DLSS disabled: configure with RTENGINE_ENABLE_DLSS and a Streamline SDK\n");
        world.dlssMode = 0;
    }
#endif

    // ---- window
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow((int)W, (int)H, "rt_renderer", nullptr, nullptr);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported()) glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

    // ---- instance
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "rt_renderer"; app.apiVersion = VK_API_VERSION_1_2;
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> instExts(glfwExts, glfwExts + glfwExtCount);
#ifdef RTENGINE_HAS_DLSS
    auto addUniqueExtension = [](std::vector<const char*>& extensions, const char* name) {
        if (std::none_of(extensions.begin(), extensions.end(), [&](const char* existing) { return std::strcmp(existing, name) == 0; }))
            extensions.push_back(name);
    };
    if (streamlineInitialized)
        for (uint32_t i = 0; i < dlssRequirements.vkNumInstanceExtensions; ++i)
            addUniqueExtension(instExts, dlssRequirements.vkInstanceExtensions[i]);
#endif
    std::vector<const char*> layers;
    if (validate) layers.push_back("VK_LAYER_KHRONOS_validation");
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)instExts.size(); ici.ppEnabledExtensionNames = instExts.data();
    ici.enabledLayerCount = (uint32_t)layers.size(); ici.ppEnabledLayerNames = layers.data();
    VkInstance instance;
    VK(vkCreateInstance(&ici, nullptr, &instance));
    VkSurfaceKHR surface;
    VK(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    // ---- physical device: needs ray query + AS + a graphics/compute/present queue; prefers discrete NVIDIA
    std::vector<const char*> devExts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
#ifdef RTENGINE_HAS_DLSS
    if (streamlineInitialized)
        for (uint32_t i = 0; i < dlssRequirements.vkNumDeviceExtensions; ++i)
            addUniqueExtension(devExts, dlssRequirements.vkDeviceExtensions[i]);
    VkPhysicalDeviceVulkan12Features dlssRequested12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features dlssRequested13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    if (streamlineInitialized) {
        dlssRequested12 = sl::getVkPhysicalDeviceVulkan12Features(dlssRequirements.vkNumFeatures12, dlssRequirements.vkFeatures12);
        dlssRequested13 = sl::getVkPhysicalDeviceVulkan13Features(dlssRequirements.vkNumFeatures13, dlssRequirements.vkFeatures13);
    }
#endif
    uint32_t pdCount = 0;
    vkEnumeratePhysicalDevices(instance, &pdCount, nullptr);
    std::vector<VkPhysicalDevice> pds(pdCount);
    vkEnumeratePhysicalDevices(instance, &pdCount, pds.data());

    uint32_t qf = 0;
    int bestScore = -1;
    for (VkPhysicalDevice pd : pds) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.apiVersion < VK_API_VERSION_1_2) continue;

        uint32_t ec = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, nullptr);
        std::vector<VkExtensionProperties> have(ec);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ec, have.data());
        bool ok = true;
        for (const char* need : devExts) {
            bool found = false;
            for (auto& e : have) if (!std::strcmp(e.extensionName, need)) found = true;
            ok = ok && found;
        }
        if (!ok) continue;

        VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        VkPhysicalDeviceSynchronization2Features sync2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        VkPhysicalDeviceVulkan12Features f12Query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features f13Query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        const bool supportsVulkan13 = props.apiVersion >= VK_API_VERSION_1_3;
        rq.pNext = &as;
        as.pNext = &sync2;
        sync2.pNext = &f12Query;
        f12Query.pNext = supportsVulkan13 ? &f13Query : nullptr;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &rq;
        vkGetPhysicalDeviceFeatures2(pd, &f2);
        if (!rq.rayQuery || !as.accelerationStructure || !sync2.synchronization2 ||
            !f12Query.descriptorIndexing || !f12Query.descriptorBindingPartiallyBound ||
            !f2.features.shaderStorageImageReadWithoutFormat ||
            !f2.features.shaderStorageImageWriteWithoutFormat ||
            !f2.features.shaderStorageImageExtendedFormats) continue;
#ifdef RTENGINE_HAS_DLSS
        if (streamlineInitialized &&
            ((!supportsVulkan13 && dlssRequirements.vkNumFeatures13 != 0) ||
             !hasRequiredVulkanFeatures(dlssRequested12, f12Query, dlssRequested13, f13Query))) continue;
#endif

        uint32_t qc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qc, nullptr);
        std::vector<VkQueueFamilyProperties> qfp(qc);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qc, qfp.data());
        for (uint32_t i = 0; i < qc; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
#ifdef RTENGINE_HAS_DLSS
            const uint32_t neededQueues = 1 + (streamlineInitialized
                ? dlssRequirements.vkNumGraphicsQueuesRequired + dlssRequirements.vkNumComputeQueuesRequired : 0);
#else
            const uint32_t neededQueues = 1;
#endif
            if (present && qfp[i].queueCount >= neededQueues &&
                (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                int score = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 : 0) +
                            (props.vendorID == 0x10DE ? 500 : 0);
                if (score > bestScore) { bestScore = score; g_pd = pd; qf = i; }
                break;
            }
        }
    }
    if (!g_pd) { std::fprintf(stderr, "No Vulkan GPU with VK_KHR_ray_query found.\n"); return 1; }

    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &asProps;
    vkGetPhysicalDeviceProperties2(g_pd, &p2);
    vkGetPhysicalDeviceMemoryProperties(g_pd, &g_mem);
    std::printf("GPU: %s\n", p2.properties.deviceName);
    const float tsPeriod = p2.properties.limits.timestampPeriod;

    uint32_t qc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &qc, qfp.data());
    const uint32_t tsBits = qfp[qf].timestampValidBits;
    const uint64_t tsMask = tsBits >= 64 ? ~0ull : ((1ull << tsBits) - 1);

    // ---- logical device
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.descriptorIndexing = VK_TRUE;
    f12.descriptorBindingPartiallyBound = VK_TRUE;
    VkPhysicalDeviceSynchronization2Features fsync2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    fsync2.synchronization2 = VK_TRUE;
    f12.pNext = &fsync2;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    fsync2.pNext = p2.properties.apiVersion >= VK_API_VERSION_1_3 ? &f13 : nullptr;
#ifdef RTENGINE_HAS_DLSS
    if (streamlineInitialized) enableRequiredVulkanFeatures(f12, dlssRequested12, f13, dlssRequested13);
#endif
    VkPhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.shaderStorageImageExtendedFormats = VK_TRUE;
    enabledFeatures.shaderStorageImageReadWithoutFormat = VK_TRUE;
    enabledFeatures.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR fas{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    fas.accelerationStructure = VK_TRUE; fas.pNext = &f12;
    VkPhysicalDeviceRayQueryFeaturesKHR frq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    frq.rayQuery = VK_TRUE; frq.pNext = &fas;

    const uint32_t appQueueIndex = 0;
#ifdef RTENGINE_HAS_DLSS
    const uint32_t slGraphicsQueueIndex = streamlineInitialized && dlssRequirements.vkNumGraphicsQueuesRequired ? 1u : 0u;
    const uint32_t slComputeQueueIndex = streamlineInitialized && dlssRequirements.vkNumComputeQueuesRequired
        ? 1u + dlssRequirements.vkNumGraphicsQueuesRequired : 0u;
    const uint32_t queueCount = 1u + (streamlineInitialized
        ? dlssRequirements.vkNumGraphicsQueuesRequired + dlssRequirements.vkNumComputeQueuesRequired : 0u);
#else
    const uint32_t queueCount = 1;
#endif
    std::vector<float> queuePriorities(queueCount, 1.0f);
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf; qci.queueCount = queueCount; qci.pQueuePriorities = queuePriorities.data();
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &frq;
    dci.pEnabledFeatures = &enabledFeatures;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)devExts.size(); dci.ppEnabledExtensionNames = devExts.data();
    VK(vkCreateDevice(g_pd, &dci, nullptr, &g_dev));
    vkGetDeviceQueue(g_dev, qf, appQueueIndex, &g_queue);
#ifdef RTENGINE_HAS_DLSS
    if (streamlineInitialized) {
        sl::VulkanInfo streamlineVulkan{};
        streamlineVulkan.device = g_dev;
        streamlineVulkan.instance = instance;
        streamlineVulkan.physicalDevice = g_pd;
        streamlineVulkan.graphicsQueueFamily = qf;
        streamlineVulkan.computeQueueFamily = qf;
        streamlineVulkan.graphicsQueueIndex = slGraphicsQueueIndex;
        streamlineVulkan.computeQueueIndex = slComputeQueueIndex;
        if (slSetVulkanInfo(streamlineVulkan) != sl::Result::eOk) {
            std::fprintf(stderr, "DLSS disabled: Streamline rejected Vulkan device information\n");
            slShutdown();
            streamlineInitialized = false;
        }
    }
#endif

    pfnCreateAS   = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(g_dev, "vkCreateAccelerationStructureKHR");
    pfnGetASSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(g_dev, "vkGetAccelerationStructureBuildSizesKHR");
    pfnCmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(g_dev, "vkCmdBuildAccelerationStructuresKHR");
    pfnGetASAddr  = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(g_dev, "vkGetAccelerationStructureDeviceAddressKHR");

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = qf;
    VK(vkCreateCommandPool(g_dev, &cpi, nullptr, &g_pool));

    // ---- swapchain (we blit our final image into it, so only TRANSFER_DST is needed)
    VkSurfaceCapabilitiesKHR caps;
    VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, surface, &caps));
    uint32_t fc = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_pd, surface, &fc, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fc);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_pd, surface, &fc, fmts.data());
    VkSurfaceFormatKHR sf = fmts[0];
    for (auto& f : fmts)   // UNORM on purpose: composite.comp already applies the gamma curve
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = f; break; }

    uint32_t pmc = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_pd, surface, &pmc, nullptr);
    std::vector<VkPresentModeKHR> pms(pmc);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_pd, surface, &pmc, pms.data());
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync) for (auto m : pms) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) pm = m;

    VkExtent2D ext = caps.currentExtent;
    if (ext.width == 0xFFFFFFFFu) { int fw, fh; glfwGetFramebufferSize(window, &fw, &fh); ext = {(uint32_t)fw, (uint32_t)fh}; }
    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface;
    sci.minImageCount = std::min(caps.minImageCount + 1, caps.maxImageCount ? caps.maxImageCount : 8u);
    sci.imageFormat = sf.format; sci.imageColorSpace = sf.colorSpace; sci.imageExtent = ext;
    sci.imageArrayLayers = 1; sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform; sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = pm; sci.clipped = VK_TRUE;
    VkSwapchainKHR swapchain;
    VK(vkCreateSwapchainKHR(g_dev, &sci, nullptr, &swapchain));
    uint32_t scCount = 0;
    vkGetSwapchainImagesKHR(g_dev, swapchain, &scCount, nullptr);
    std::vector<VkImage> scImages(scCount);
    vkGetSwapchainImagesKHR(g_dev, swapchain, &scCount, scImages.data());

#ifdef RTENGINE_HAS_DLSS
    sl::ViewportHandle dlssViewport(0);
    sl::DLSSOptions dlssOptions{};
    uint32_t renderW = W, renderH = H;
    if (streamlineInitialized && world.dlssMode > 0) {
        sl::AdapterInfo adapter{};
        adapter.vkPhysicalDevice = g_pd;
        sl::Result support = slIsFeatureSupported(sl::kFeatureDLSS, adapter);
        if (support != sl::Result::eOk) {
            std::fprintf(stderr, "DLSS disabled: Streamline reports this GPU/driver is unsupported (%u)\n",
                         static_cast<uint32_t>(support));
        } else {
            switch (world.dlssMode) {
            case 1: dlssOptions.mode = sl::DLSSMode::eMaxQuality; break;
            case 2: dlssOptions.mode = sl::DLSSMode::eBalanced; break;
            case 3: dlssOptions.mode = sl::DLSSMode::eMaxPerformance; break;
            case 4: dlssOptions.mode = sl::DLSSMode::eUltraPerformance; break;
            case 5: dlssOptions.mode = sl::DLSSMode::eUltraQuality; break;
            default: dlssOptions.mode = sl::DLSSMode::eOff; break;
            }
            dlssOptions.outputWidth = W;
            dlssOptions.outputHeight = H;
            dlssOptions.colorBuffersHDR = sl::Boolean::eFalse;
            dlssOptions.useAutoExposure = sl::Boolean::eTrue;
            sl::DLSSOptimalSettings optimal{};
            if (slDLSSGetOptimalSettings(dlssOptions, optimal) == sl::Result::eOk &&
                optimal.optimalRenderWidth > 0 && optimal.optimalRenderHeight > 0 &&
                slDLSSSetOptions(dlssViewport, dlssOptions) == sl::Result::eOk) {
                renderW = optimal.optimalRenderWidth;
                renderH = optimal.optimalRenderHeight;
                dlssActive = true;
                activeDlssMode = world.dlssMode;
                std::printf("DLSS Super Resolution active: render %ux%u -> output %ux%u\n", renderW, renderH, W, H);
            } else {
                std::fprintf(stderr, "DLSS disabled: Streamline could not configure the requested mode\n");
            }
        }
    }
    if (!dlssActive) world.dlssMode = 0;
#else
    uint32_t renderW = W, renderH = H;
#endif

    // ---- scene buffers at full capacity: scripts can add/remove things without recreating anything
    Buffer vbuf = createBuffer(size_t(MAX_TRIS) * 9 * sizeof(float), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, true);
    Buffer triBuf = createBuffer(size_t(MAX_TRIS) * sizeof(GpuTri), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    Buffer lightBuf = createBuffer(size_t(MAX_LIGHTS) * sizeof(GpuLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    Buffer aliasBuf = createBuffer(size_t(MAX_LIGHTS) * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    // light BVH: N leaves + (N-1) interior nodes, worst case (see buildLightBVH / BVHNode in common.glsl)
    Buffer lightBvhBuf = createBuffer(size_t(2 * MAX_LIGHTS - 1) * sizeof(GpuBVHNode), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    Buffer uboBuf = createBuffer(sizeof(FrameUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
    const uint32_t tilesX = (renderW + 15) / 16, tilesY = (renderH + 15) / 16;
    const VkDeviceSize tileBytes = 16 + 256 * 4 * 2;     // count + totalW + 256 indices + 256 CDF entries (see common.glsl)
    Buffer tileBuf = createBuffer(tilesX * tilesY * tileBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);

    // world-space light grid (same idea as the screen tile lists, but for GI hits and surface-cache texels)
    const uint32_t GRID = 24;                           // 24^3 cells, up to 64 lights each (see Cell in common.glsl)
    const VkDeviceSize cellBytes = 16 + 64 * 4 * 2;
    Buffer cellBuf = createBuffer(VkDeviceSize(GRID) * GRID * GRID * cellBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    // surface cache: irradiance + ReSTIR reservoir per texel and side, plus the per-triangle first-texel table
    const VkBufferUsageFlags cacheUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    Buffer cacheIrr = createBuffer(VkDeviceSize(CACHE_MAX_TEXELS) * 2 * 16, cacheUsage, false);
    Buffer cacheRes = createBuffer(VkDeviceSize(CACHE_MAX_TEXELS) * 2 * 16, cacheUsage, false);
    Buffer cacheOffBuf = createBuffer((VkDeviceSize(MAX_TRIS) + 1) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);

    VkAccelerationStructureGeometryKHR blasGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    blasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;                       // no any-hit invocations, ever
    auto& tr = blasGeom.geometry.triangles;
    tr.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tr.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tr.vertexData.deviceAddress = vbuf.addr;
    tr.vertexStride = 3 * sizeof(float);
    tr.maxVertex = MAX_TRIS * 3 - 1;
    tr.indexType = VK_INDEX_TYPE_NONE_KHR;
    Accel blas = createAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, blasGeom, MAX_TRIS,
                             asProps.minAccelerationStructureScratchOffsetAlignment);

    VkAccelerationStructureDeviceAddressInfoKHR dai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    dai.accelerationStructure = blas.as;
    VkAccelerationStructureInstanceKHR inst{};
    inst.transform.matrix[0][0] = inst.transform.matrix[1][1] = inst.transform.matrix[2][2] = 1.0f;
    inst.mask = 0xFF;
    inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst.accelerationStructureReference = pfnGetASAddr(g_dev, &dai);
    Buffer instBuf = createBuffer(sizeof(inst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, true);
    std::memcpy(instBuf.map, &inst, sizeof(inst));

    VkAccelerationStructureGeometryKHR tlasGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.data.deviceAddress = instBuf.addr;
    Accel tlas = createAccel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tlasGeom, 1,
                             asProps.minAccelerationStructureScratchOffsetAlignment);

    TexArray texArray = createTexArray();

    // ---- render targets (all stay in GENERAL; ping-pong pairs are swapped via two descriptor sets)
    const VkFormat F32 = VK_FORMAT_R32G32B32A32_SFLOAT, F16 = VK_FORMAT_R16G16B16A16_SFLOAT,
                   U32 = VK_FORMAT_R32G32B32A32_UINT, R8 = VK_FORMAT_R8G8B8A8_UNORM;
    Image gND[2] = {createImage(renderW, renderH, F32), createImage(renderW, renderH, F32)};
    Image gAlbedo = createImage(renderW, renderH, F16);
    Image resInit = createImage(renderW, renderH, U32);
    Image resT[2] = {createImage(renderW, renderH, U32), createImage(renderW, renderH, U32)};
    Image resSpatial = createImage(renderW, renderH, U32);
    Image gi[2] = {createImage(renderW, renderH, F16), createImage(renderW, renderH, F16)};
    Image dirAcc[2] = {createImage(renderW, renderH, F16), createImage(renderW, renderH, F16)};
    Image lit[2] = {createImage(renderW, renderH, F16), createImage(renderW, renderH, F16)};
    Image nrdRawDI = createImage(renderW, renderH, F16);
    Image nrdRawGI = createImage(renderW, renderH, F16);
    Image nrdDIInput = createImage(renderW, renderH, F16, VK_IMAGE_USAGE_SAMPLED_BIT);
    Image nrdGIInput = createImage(renderW, renderH, F16, VK_IMAGE_USAGE_SAMPLED_BIT);
    Image nrdMotion = createImage(renderW, renderH, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT);
    Image nrdNormalRoughness = createImage(renderW, renderH, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_IMAGE_USAGE_SAMPLED_BIT);
    Image nrdViewZ = createImage(renderW, renderH, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT);
    Image nrdDIOutput = createImage(renderW, renderH, F16, VK_IMAGE_USAGE_SAMPLED_BIT);
    Image nrdGIOutput = createImage(renderW, renderH, F16, VK_IMAGE_USAGE_SAMPLED_BIT);
    Image finalImg = createImage(renderW, renderH, R8, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    Image dlssOutput{};
#ifdef RTENGINE_HAS_DLSS
    if (dlssActive) dlssOutput = createImage(W, H, R8, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
#endif
    std::vector<VkImage> allImages = {gND[0].img, gND[1].img, gAlbedo.img, resInit.img, resT[0].img, resT[1].img,
                                      resSpatial.img, gi[0].img, gi[1].img, dirAcc[0].img, dirAcc[1].img, lit[0].img, lit[1].img,
                                      nrdRawDI.img, nrdRawGI.img, nrdDIInput.img, nrdGIInput.img, nrdMotion.img,
                                      nrdNormalRoughness.img, nrdViewZ.img, nrdDIOutput.img, nrdGIOutput.img, finalImg.img};
#ifdef RTENGINE_HAS_DLSS
    if (dlssActive) allImages.push_back(dlssOutput.img);
#endif
    {
        VkCommandBuffer cb = beginOnce();
        std::vector<VkImageMemoryBarrier> bars;
        for (VkImage im : allImages) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = im; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bars.push_back(b);
        }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             (uint32_t)bars.size(), bars.data());
        VkClearColorValue zero{};
        VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (VkImage im : allImages) vkCmdClearColorImage(cb, im, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rng);
        endOnce(cb);
    }

    // ---- descriptors: binding layout mirrors common.glsl
    const uint32_t NB = 35;
    auto isStorageBuf = [](uint32_t i) { return i == 2 || i == 3 || i == 14 || i == 15 || (i >= 19 && i <= 22) || i == 25; };
    std::array<VkDescriptorSetLayoutBinding, NB> lb{};
    for (uint32_t i = 0; i < NB; ++i) {
        lb[i].binding = i; lb[i].descriptorCount = 1; lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        lb[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                             : i == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                             : i == 18 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                             : isStorageBuf(i) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = (uint32_t)lb.size(); dli.pBindings = lb.data();
    VkDescriptorSetLayout dsl;
    VK(vkCreateDescriptorSetLayout(g_dev, &dli, nullptr, &dsl));

    std::array<VkDescriptorPoolSize, 5> dps = {{{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4},
                                                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 36}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 92},
                                                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 4; dpi.poolSizeCount = (uint32_t)dps.size(); dpi.pPoolSizes = dps.data();
    VkDescriptorPool dpool;
    VK(vkCreateDescriptorPool(g_dev, &dpi, nullptr, &dpool));
    VkDescriptorSet sets[2], compositeSets[2];
    VkDescriptorSetLayout dsls[4] = {dsl, dsl, dsl, dsl};
    VkDescriptorSetAllocateInfo dai2{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai2.descriptorPool = dpool; dai2.descriptorSetCount = 4; dai2.pSetLayouts = dsls;
    VkDescriptorSet allSets[4];
    VK(vkAllocateDescriptorSets(g_dev, &dai2, allSets));
    for (int cur = 0; cur < 2; ++cur) {
        sets[cur] = allSets[cur * 2];
        compositeSets[cur] = allSets[cur * 2 + 1];
    }

    for (int cur = 0; cur < 2; ++cur) {
        int prev = 1 - cur;
        VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        asw.accelerationStructureCount = 1; asw.pAccelerationStructures = &tlas.as;
        VkDescriptorBufferInfo bufInfo[NB] = {};
        bufInfo[1] = {uboBuf.buf, 0, sizeof(FrameUBO)};
        bufInfo[2] = {triBuf.buf, 0, VK_WHOLE_SIZE};
        bufInfo[3] = {lightBuf.buf, 0, VK_WHOLE_SIZE};
        bufInfo[14] = {aliasBuf.buf, 0, VK_WHOLE_SIZE};
        bufInfo[15] = {tileBuf.buf, 0, VK_WHOLE_SIZE};
        bufInfo[19] = {cacheIrr.buf, 0, VK_WHOLE_SIZE};
        bufInfo[20] = {cacheRes.buf, 0, VK_WHOLE_SIZE};
        bufInfo[21] = {cacheOffBuf.buf, 0, VK_WHOLE_SIZE};
        bufInfo[22] = {cellBuf.buf, 0, VK_WHOLE_SIZE};
        bufInfo[25] = {lightBvhBuf.buf, 0, VK_WHOLE_SIZE};
        VkImageView views[14] = {gND[cur].view, gND[prev].view, gAlbedo.view, resInit.view, resT[cur].view, resT[prev].view,
                                 resSpatial.view, gi[cur].view, gi[prev].view, finalImg.view, dirAcc[cur].view, dirAcc[prev].view, lit[cur].view, lit[prev].view};
        VkDescriptorImageInfo imgInfo[14];
        for (int i = 0; i < 14; ++i) imgInfo[i] = {VK_NULL_HANDLE, views[i], VK_IMAGE_LAYOUT_GENERAL};
        VkImageView nrdViews[9] = {nrdRawDI.view, nrdRawGI.view, nrdDIInput.view, nrdGIInput.view, nrdMotion.view,
                                   nrdNormalRoughness.view, nrdViewZ.view, nrdDIOutput.view, nrdGIOutput.view};
        VkDescriptorImageInfo nrdImgInfo[9];
        for (int i = 0; i < 9; ++i) nrdImgInfo[i] = {VK_NULL_HANDLE, nrdViews[i], VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo texInfo{texArray.sampler, texArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        for (int variant = 0; variant < 2; ++variant) {
            if (variant == 1) {
                imgInfo[7].imageView = nrdGIOutput.view;
                imgInfo[10].imageView = nrdDIOutput.view;
            }
            std::vector<VkWriteDescriptorSet> w(NB);
            for (uint32_t i = 0; i < NB; ++i) {
                w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w[i].dstSet = variant == 0 ? sets[cur] : compositeSets[cur];
                w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = lb[i].descriptorType;
                if (i == 0) w[i].pNext = &asw;
                else if (i == 1 || isStorageBuf(i)) w[i].pBufferInfo = &bufInfo[i];
                else if (i == 18) w[i].pImageInfo = &texInfo;
                else if (i >= 26) w[i].pImageInfo = &nrdImgInfo[i - 26];
                else w[i].pImageInfo = &imgInfo[i <= 13 ? i - 4 : (i <= 17 ? i - 6 : i - 11)];
            }
            vkUpdateDescriptorSets(g_dev, (uint32_t)w.size(), w.data(), 0, nullptr);
        }
    }

    // ---- pipelines
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    VkPipelineLayout playout;
    VK(vkCreatePipelineLayout(g_dev, &pli, nullptr, &playout));

    const char* passNames[NSTAT] = {"gbuffer+AS", "light_grid", "tile_lights", "di_initial", "di_temporal", "di_spatial",
                                    "cache_update", "direct", "gi", "NRD", "composite"};
    const char* shaderFiles[NP] = {"gbuffer.comp.spv", "light_grid.comp.spv", "tile_lights.comp.spv", "di_initial.comp.spv", "di_temporal.comp.spv",
                                   "di_spatial.comp.spv", "cache_update.comp.spv", "direct.comp.spv", "gi.comp.spv", "composite.comp.spv"};
    VkPipeline pipes[NP];
    for (int i = 0; i < NP; ++i) {
        VkShaderModule m = loadShader(shaderDir / shaderFiles[i]);
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
        cpci.layout = playout;
        VK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipes[i]));
        vkDestroyShaderModule(g_dev, m, nullptr);
    }
    VkShaderModule nrdPrepModule = loadShader(shaderDir / "nrd_prep.comp.spv");
    VkComputePipelineCreateInfo nrdPrepPipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    nrdPrepPipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                                 nrdPrepModule, "main", nullptr};
    nrdPrepPipelineInfo.layout = playout;
    VkPipeline nrdPrepPipe;
    VK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &nrdPrepPipelineInfo, nullptr, &nrdPrepPipe));
    vkDestroyShaderModule(g_dev, nrdPrepModule, nullptr);

    constexpr nrd::Identifier nrdDIId = 1, nrdGIId = 2;
    const nrd::DenoiserDesc nrdDIDenoiser{nrdDIId, nrd::Denoiser::RELAX_DIFFUSE};
    const nrd::DenoiserDesc nrdGIDenoiser{nrdGIId, nrd::Denoiser::REBLUR_DIFFUSE};
    nrd::InstanceCreationDesc nrdDIInstanceDesc{};
    nrdDIInstanceDesc.denoisers = &nrdDIDenoiser;
    nrdDIInstanceDesc.denoisersNum = 1;
    nrd::InstanceCreationDesc nrdGIInstanceDesc{};
    nrdGIInstanceDesc.denoisers = &nrdGIDenoiser;
    nrdGIInstanceDesc.denoisersNum = 1;
    const nrd::LibraryDesc* nrdLibrary = nrd::GetLibraryDesc();
    nrd::IntegrationCreationDesc nrdIntegrationDesc{};
    std::snprintf(nrdIntegrationDesc.name, sizeof(nrdIntegrationDesc.name), "rtengine");
    nrdIntegrationDesc.resourceWidth = static_cast<uint16_t>(renderW);
    nrdIntegrationDesc.resourceHeight = static_cast<uint16_t>(renderH);
    nrdIntegrationDesc.queuedFrameNum = 1;
    nrdIntegrationDesc.autoWaitForIdle = false;
    nri::QueueFamilyVKDesc nrdQueueFamily{1, nri::QueueType::GRAPHICS, qf};
    nri::DeviceCreationVKDesc nrdDeviceDesc{};
    nrdDeviceDesc.vkBindingOffsets = {
        nrdLibrary->spirvBindingOffsets.samplerOffset,
        nrdLibrary->spirvBindingOffsets.textureOffset,
        nrdLibrary->spirvBindingOffsets.constantBufferOffset,
        nrdLibrary->spirvBindingOffsets.storageTextureAndBufferOffset
    };
    nrdDeviceDesc.vkExtensions.deviceExtensions = devExts.data();
    nrdDeviceDesc.vkExtensions.deviceExtensionNum = static_cast<uint32_t>(devExts.size());
    nrdDeviceDesc.vkInstance = instance;
    nrdDeviceDesc.vkDevice = g_dev;
    nrdDeviceDesc.vkPhysicalDevice = g_pd;
    nrdDeviceDesc.queueFamilies = &nrdQueueFamily;
    nrdDeviceDesc.queueFamilyNum = 1;
    nrdDeviceDesc.minorVersion = 2;
    nrd::Integration nrdDIIntegration, nrdGIIntegration;
    nrd::Result nrdResult = nrdDIIntegration.RecreateVK(nrdIntegrationDesc, nrdDIInstanceDesc, nrdDeviceDesc);
    if (nrdResult != nrd::Result::SUCCESS) {
        std::fprintf(stderr, "NRD DI initialization failed (%u)\n", static_cast<uint32_t>(nrdResult));
        return 1;
    }
    nrdResult = nrdGIIntegration.RecreateVK(nrdIntegrationDesc, nrdGIInstanceDesc, nrdDeviceDesc);
    if (nrdResult != nrd::Result::SUCCESS) {
        std::fprintf(stderr, "NRD GI initialization failed (%u)\n", static_cast<uint32_t>(nrdResult));
        return 1;
    }
    nrd::RelaxSettings nrdDISettings{};
    nrdDISettings.diffuseMaxAccumulatedFrameNum = 20;
    nrdDISettings.diffuseMaxFastAccumulatedFrameNum = 6;
    nrd::ReblurSettings nrdGISettings{};
    nrdGISettings.maxAccumulatedFrameNum = 30;
    nrdGISettings.maxFastAccumulatedFrameNum = 6;
    nrdGISettings.maxStabilizedFrameNum = 30;
    nrdResult = nrdDIIntegration.SetDenoiserSettings(nrdDIId, &nrdDISettings);
    if (nrdResult != nrd::Result::SUCCESS) {
        std::fprintf(stderr, "NRD DI configuration failed (%u)\n", static_cast<uint32_t>(nrdResult));
        return 1;
    }
    nrdResult = nrdGIIntegration.SetDenoiserSettings(nrdGIId, &nrdGISettings);
    if (nrdResult != nrd::Result::SUCCESS) {
        std::fprintf(stderr, "NRD GI configuration failed (%u)\n", static_cast<uint32_t>(nrdResult));
        return 1;
    }
    auto makeNrdResource = [](Image& image, VkFormat format) {
        nrd::Resource resource{};
        resource.vk.image = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image.img));
        resource.vk.format = static_cast<int32_t>(format);
        resource.state = {nri::AccessBits::SHADER_RESOURCE_STORAGE,
                          nri::Layout::SHADER_RESOURCE_STORAGE,
                          nri::StageBits::COMPUTE_SHADER};
        resource.userArg = &image;
        return resource;
    };

    // ---- sync, command buffer, GPU timestamps
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g_pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK(vkAllocateCommandBuffers(g_dev, &cbai, &cmd));
    VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore imageAvail;
    VK(vkCreateSemaphore(g_dev, &semi, nullptr, &imageAvail));
    std::vector<VkSemaphore> renderDone(scCount);
    for (auto& s : renderDone) VK(vkCreateSemaphore(g_dev, &semi, nullptr, &s));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence fence;
    VK(vkCreateFence(g_dev, &fi, nullptr, &fence));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = NSTAT + 1;
    VkQueryPool qpool;
    VK(vkCreateQueryPool(g_dev, &qpi, nullptr, &qpool));

    // ---- main loop state
    Camera cam, lastCam;
    cam.pos = world.camPos;
    cam.yaw = world.camYaw * kPi / 180.0f;
    cam.pitch = std::clamp(world.camPitch * kPi / 180.0f, -1.55f, 1.55f);
    lastCam = cam;
    bool temporal = true, spatial = true;
    uint32_t lightMode = 2;  // 0 = global alias table, 1 = per-tile/cell CDF, 2 = light BVH (default: BVH)
    float exposure = 1.0f;
    uint32_t dbg = 0, giBounces = 2, candIdx = 2;  // 12 candidates by default
    const uint32_t candTable[6] = {1, 4, 8, 12, 16, 32};
    uint32_t frame = 0, numLights = 0, numTris = 0, nDir = 0;
    bool nrdWasEnabled = false;
    uint64_t layoutHash = 0;                             // surface-cache layout: when it changes the cache is cleared
    uint32_t cacheEntries = 0, cacheCursor = 0;          // texels*2 sides, rotating refresh position
    bool cacheClear = true;
    float gridMinV[3] = {-1, -1, -1}, gridInvV[3] = {12, 12, 12};
    double totalPower = 1.0;
    double lastTime = glfwGetTime(), titleTime = lastTime, mx = 0, my = 0;
    bool firstMouse = true;
    std::array<bool, GLFW_KEY_LAST + 1> keyWas{}, edge{};
    double passAcc[NSTAT] = {};
    int accFrames = 0;
    const float tanHalf = std::tan(0.5f * 45.0f * kPi / 180.0f);
    const float aspect = float(W) / float(H);
    const uint32_t gx = (renderW + 7) / 8, gy = (renderH + 7) / 8;
#ifdef RTENGINE_HAS_DLSS
    auto halton = [](uint32_t index, uint32_t base) {
        float value = 0.0f, fraction = 1.0f;
        while (index) {
            fraction /= float(base);
            value += fraction * float(index % base);
            index /= base;
        }
        return value;
    };
    float lastJitterX = 0.0f, lastJitterY = 0.0f;
    sl::FrameToken* dlssFrameToken = nullptr;
#endif
    const std::vector<int> scriptKeys = [&] { return script.keys(); }();
    (void)scriptKeys;

    // Re-uploads the whole scene from the world model (GPU is idle when this runs).
    auto rebuildWorld = [&]() -> bool {
        Scene sc;
        sc.cellSize = world.cacheCellSize;
        buildScene(world, sc);
        if (sc.truncated) std::fprintf(stderr, "warning: scene truncated (max %u triangles / %u lights)\n", MAX_TRIS, MAX_LIGHTS);
        numTris = (uint32_t)sc.tris.size();
        std::memcpy(vbuf.map, sc.verts.data(), sc.verts.size() * sizeof(float));
        std::memcpy(triBuf.map, sc.tris.data(), sc.tris.size() * sizeof(GpuTri));
        uint32_t nl = (uint32_t)sc.lights.size();
        auto mid = std::stable_partition(sc.lights.begin(), sc.lights.end(),
                                 [](const GpuLight& L) { return L.type == LIGHT_DIR; });
nDir = (uint32_t)(mid - sc.lights.begin());
        if (nl) std::memcpy(lightBuf.map, sc.lights.data(), nl * sizeof(GpuLight));
        std::vector<float> power, aprob;
        std::vector<uint32_t> aidx;
        double tp = 0;
        for (auto& L : sc.lights) { power.push_back(L.power); tp += L.power; }
        buildAlias(power, aprob, aidx);
        uint32_t* d = (uint32_t*)aliasBuf.map;
        for (uint32_t i = 0; i < nl; ++i) { std::memcpy(&d[2 * i], &aprob[i], 4); d[2 * i + 1] = aidx[i]; }
        // Light BVH for stochastic importance-weighted traversal (RTXDI/Moana-style light clustering)
        std::vector<GpuBVHNode> bvhNodes;
        std::vector<GpuLight> rest(sc.lights.begin() + nDir, sc.lights.end());
buildLightBVH(rest, bvhNodes);
for (auto& nd : bvhNodes)                         // leaves index into `rest`, shift to the full list
    if (nd.rightOrLeaf == 0xFFFFFFFFu) nd.leftOrLight += nDir;
if (!bvhNodes.empty())
std::printf("BVH: %zu nodes, root power=%.4f, bmin=(%.2f,%.2f,%.2f), bmax=(%.2f,%.2f,%.2f)\n",
        bvhNodes.size(), bvhNodes[0].power,
        bvhNodes[0].bmin[0], bvhNodes[0].bmin[1], bvhNodes[0].bmin[2],
        bvhNodes[0].bmax[0], bvhNodes[0].bmax[1], bvhNodes[0].bmax[2]);
    std::memcpy(lightBvhBuf.map, bvhNodes.data(), bvhNodes.size() * sizeof(GpuBVHNode));
        totalPower = tp > 0 ? tp : 1.0;
        // surface-cache layout + world light grid bounds
        std::memcpy(cacheOffBuf.map, sc.cacheOffsets.data(), sc.cacheOffsets.size() * sizeof(uint32_t));
        cacheEntries = sc.cacheTexels * 2;
        if (sc.layoutHash != layoutHash) { layoutHash = sc.layoutHash; cacheClear = true; }
        const float lo[3] = {sc.bmin.x, sc.bmin.y, sc.bmin.z}, hi[3] = {sc.bmax.x, sc.bmax.y, sc.bmax.z};
        for (int a = 0; a < 3; ++a) {
            float ext = std::max(hi[a] - lo[a], 0.1f);
            gridMinV[a] = lo[a] - ext * 0.01f;
            gridInvV[a] = float(GRID) / (ext * 1.02f);
        }
        bool structural = nl != numLights;
        numLights = nl;
        return structural;
    };
    auto flushTextures = [&]() {
        for (auto& up : world.pendingTex) uploadTexture(texArray, up);
        world.pendingTex.clear();
    };

    struct GpuCustom { VkPipeline pipe = VK_NULL_HANDLE; std::filesystem::file_time_type stamp{}; };
    std::map<std::string, GpuCustom> customGpu;
    double lastShaderPoll = 0;
    auto syncCustomShaders = [&](bool poll) {       // call only while the GPU is idle
        for (auto it = customGpu.begin(); it != customGpu.end();) {          // entries removed by F5 reload
            bool alive = std::any_of(world.shaders.begin(), world.shaders.end(), [&](const CustomShader& s) { return s.id == it->first; });
            if (!alive) { vkDestroyPipeline(g_dev, it->second.pipe, nullptr); it = customGpu.erase(it); } else ++it;
        }
        for (auto& s : world.shaders) {
            GpuCustom& g = customGpu[s.id];
            std::error_code ec;
            auto stamp = std::filesystem::last_write_time(s.path, ec);
            if (poll && !ec && stamp != g.stamp) s.recompile = true;         // file saved: hot reload
            if (!s.recompile) continue;
            s.recompile = false;
            if (!ec) g.stamp = stamp;
            CompileResult r = compileGlsl(s.path, shaderDir, exeDir / "shader_cache", s.id);
            if (!r.ok) { std::fprintf(stderr, "shader '%s' failed to compile:\n%s\n", s.id.c_str(), r.log.c_str()); continue; }   // keep old pipeline
            VkPipeline p = makeComputePipeline(playout, r.spv);
            if (!p) { std::fprintf(stderr, "shader '%s': pipeline creation failed\n", s.id.c_str()); continue; }
            if (g.pipe) vkDestroyPipeline(g_dev, g.pipe, nullptr);
            g.pipe = p;
            std::printf("shader '%s' compiled\n", s.id.c_str());
        }
    };

    flushTextures();
    syncCustomShaders(false);
    std::printf("Controls: WASD/QE move, mouse look, F5 reload script, N GI mode, L area lights, H GI rate (screen fallback 1/8), M light sampling (global/tile/BVH), T/Y reuse, G bounces (path mode), C candidates, 1/2/3 view, -/= exposure\n");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        double now = glfwGetTime();
        float dt = float(now - lastTime);
        lastTime = now;

        // ---- input
        for (int k = GLFW_KEY_SPACE; k <= GLFW_KEY_LAST; ++k) {
            bool down = glfwGetKey(window, k) == GLFW_PRESS;
            edge[k] = down && !keyWas[k];
            keyWas[k] = down;
        }
        double cx, cy;
        glfwGetCursorPos(window, &cx, &cy);
        if (firstMouse) { mx = cx; my = cy; firstMouse = false; }
        auto down = [&](int k) { return glfwGetKey(window, k) == GLFW_PRESS; };
        if (!world.camLocked) {
            cam.yaw += float(cx - mx) * 0.0025f;
            cam.pitch = std::clamp(cam.pitch - float(cy - my) * 0.0025f, -1.55f, 1.55f);
            if (world.playerController) {
                world.updatePlayer(dt, down(GLFW_KEY_W), down(GLFW_KEY_S), down(GLFW_KEY_A),
                                   down(GLFW_KEY_D), edge[GLFW_KEY_SPACE],
                                   down(GLFW_KEY_LEFT_SHIFT), down(GLFW_KEY_E),
                                   down(GLFW_KEY_Q), cam.yaw, cam.pitch);
                cam.pos = world.camPos;
            } else {
                Basis b = basisOf(cam);
                float speed = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 6.0f : 2.0f) * dt;
                const V3 camBefore = cam.pos;
                if (down(GLFW_KEY_W)) cam.pos = cam.pos + b.fwd * speed;
                if (down(GLFW_KEY_S)) cam.pos = cam.pos - b.fwd * speed;
                if (down(GLFW_KEY_D)) cam.pos = cam.pos + b.right * speed;
                if (down(GLFW_KEY_A)) cam.pos = cam.pos - b.right * speed;
                if (down(GLFW_KEY_E)) cam.pos.y += speed;
                if (down(GLFW_KEY_Q)) cam.pos.y -= speed;
                if (world.camCollide) cam.pos = world.moveCamera(camBefore, cam.pos); // script: collision camera on
            }
        }
        mx = cx; my = cy;
        if (edge[GLFW_KEY_1]) dbg = 0;
        if (edge[GLFW_KEY_2]) dbg = 1;
        if (edge[GLFW_KEY_3]) dbg = 2;
        if (edge[GLFW_KEY_T]) temporal = !temporal;
        if (edge[GLFW_KEY_Y]) spatial = !spatial;
        if (edge[GLFW_KEY_G]) giBounces = (giBounces + 1) % 4;
        if (edge[GLFW_KEY_C]) candIdx = (candIdx + 1) % 6;
        if (edge[GLFW_KEY_M]) lightMode = (lightMode + 1) % 3;  // cycle: global -> tile/grid -> BVH -> global
        if (edge[GLFW_KEY_N]) world.giMode = (world.giMode + 1) % 3;                  // path -> surface cache -> cache + screen traces
        if (edge[GLFW_KEY_H]) {
            if (world.giMode == 2) world.giOffscreenRate = world.giOffscreenRate == 1 ? 8 : 1;
            else world.giRate = world.giRate == 1 ? 2 : 1;
        }
        if (edge[GLFW_KEY_L]) { world.areaLights = !world.areaLights; world.dirty = true; }   // emissive triangles on/off
        if (edge[GLFW_KEY_B]) world.denoise = world.denoise > 0.0f ? 0.0f : 1.0f;     // NRD denoising on/off
        if (edge[GLFW_KEY_K]) world.camCollide = !world.camCollide;                   // camera collision on/off
        if (down(GLFW_KEY_EQUAL)) exposure *= std::exp(dt);
        if (down(GLFW_KEY_MINUS)) exposure /= std::exp(dt);

        // ---- wait for previous frame, collect GPU timings
        VK(vkWaitForFences(g_dev, 1, &fence, VK_TRUE, UINT64_MAX));
        if (frame > 0 && tsBits > 0) {
            uint64_t ts[NSTAT + 1];
            if (vkGetQueryPoolResults(g_dev, qpool, 0, NSTAT + 1, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                for (int i = 0; i < NSTAT; ++i) passAcc[i] += double((ts[i + 1] - ts[i]) & tsMask) * tsPeriod * 1e-6;
                ++accFrames;
            }
        }

        // ---- scripts + scene updates (the GPU is idle here, so buffers can be rewritten safely)
        world.time = float(now); world.dt = dt; world.frame = frame;
        world.camPos = cam.pos; world.camYaw = cam.yaw * 180.0f / kPi; world.camPitch = cam.pitch * 180.0f / kPi;
        world.camChanged = false;
        if (edge[GLFW_KEY_F5]) {
            std::printf("reloading script...\n");
            world.clearScene();
            Script fresh(world);
            if (!script.name.empty() && script.name != "<built-in>" && fresh.loadFile(script.name)) { script.events = fresh.events; script.runStart(); }
            else { script.loadText(kDefaultScript, "<built-in>"); script.runStart(); }
            if (world.dlssMode != activeDlssMode) {
                std::fprintf(stderr, "DLSS mode changes require restarting the renderer; keeping the active mode\n");
                world.dlssMode = activeDlssMode;
            }
            frame = 0;
        }
        for (int k : script.keys()) if (k != GLFW_KEY_F5 && edge[k]) script.fireKey(k);
        script.update(dt);
        if (world.camChanged) {                          // script moved the camera: treat as a cut
            cam.pos = world.camPos;
            cam.yaw = world.camYaw * kPi / 180.0f;
            cam.pitch = std::clamp(world.camPitch * kPi / 180.0f, -1.55f, 1.55f);
            lastCam = cam;
            frame = 0;
        }
        flushTextures();
        bool rebuilt = false;
        if (world.dirty) {
            if (rebuildWorld()) frame = 0;               // light list changed -> old reservoirs would point at the wrong lights
            world.dirty = false;
            rebuilt = true;
        }
        const bool pollShaders = now - lastShaderPoll > 0.5;
        if (pollShaders) lastShaderPoll = now;
        syncCustomShaders(pollShaders);

        VK(vkResetFences(g_dev, 1, &fence));
        uint32_t imageIndex;
        VkResult ar = vkAcquireNextImageKHR(g_dev, swapchain, UINT64_MAX, imageAvail, VK_NULL_HANDLE, &imageIndex);
        if (ar == VK_ERROR_OUT_OF_DATE_KHR) break;

        // ---- per-frame uniforms
        Basis cb_ = basisOf(cam), pb = basisOf(lastCam);
        FrameUBO u{};
        setV(u.camPos, cam.pos);   setV(u.camFwd, cb_.fwd);
        setV(u.camRight, cb_.right * (tanHalf * aspect)); setV(u.camUp, cb_.up * tanHalf);
        setV(u.prevPos, lastCam.pos); setV(u.prevFwd, pb.fwd);
        setV(u.prevRight, pb.right * (tanHalf * aspect)); setV(u.prevUp, pb.up * tanHalf);
        u.misc0[0] = renderW; u.misc0[1] = renderH; u.misc0[2] = frame; u.misc0[3] = (temporal ? 1u : 0u) | (spatial ? 2u : 0u);
        u.misc1[0] = numLights; u.misc1[1] = dbg; u.misc1[2] = giBounces; u.misc1[3] = candTable[candIdx];
        u.params[0] = exposure; u.params[1] = 32.0f; u.params[2] = 12.0f; u.params[3] = 10.0f;   // exposure, GI history, spatial radius, firefly clamp
        u.misc2[0] = tilesX; u.misc2[1] = lightMode; u.misc2[2] = nDir;
        u.params2[0] = float(1.0 / totalPower); u.params2[1] = tileCutoff; u.params2[2] = 16.0f;
        u.params2[3] = 2.0f * tanHalf / float(renderH);    // pixel angular spread (texture LOD)
        for (int a = 0; a < 3; ++a) { u.gridMin[a] = gridMinV[a]; u.gridInvCell[a] = gridInvV[a]; u.gridDim[a] = GRID; }
        const uint32_t cacheBudget = std::min(cacheEntries, std::min(131072u, std::max(8192u, cacheEntries / 6u)));   // texels refreshed per frame
        u.misc3[0] = (uint32_t)world.giMode; u.misc3[1] = (uint32_t)world.giRate;
        u.misc3[2] = 2; u.misc3[3] = world.giOffscreenRate;                       // mode, rate, cache rays, screen-mode BVH fallback rate
        u.misc4[0] = cacheBudget; u.misc4[1] = cacheCursor; u.misc4[2] = cacheEntries; u.misc4[3] = numTris;
                const bool nrdEnabled = world.denoise > 0.0f;
        u.params3[0] = 2.5f; u.params3[1] = 12.0f; u.params3[2] = world.denoise;
        setV(u.skyZenith,  world.skyZenith  * world.skyIntensity);
        setV(u.skyHorizon, world.skyHorizon * world.skyIntensity);                                 // screen trace distance, cache history, denoise strength
#ifdef RTENGINE_HAS_DLSS
        float jitterX = 0.0f, jitterY = 0.0f;
        if (dlssActive) {
            jitterX = halton(frame + 1, 2) - 0.5f;
            jitterY = halton(frame + 1, 3) - 0.5f;
        }
        u.jitter[0] = jitterX; u.jitter[1] = jitterY;
        u.jitter[2] = frame == 0 ? jitterX : lastJitterX;
        u.jitter[3] = frame == 0 ? jitterY : lastJitterY;
        if (dlssActive) {
            if (slGetNewFrameToken(dlssFrameToken, &frame) != sl::Result::eOk || !dlssFrameToken) {
                std::fprintf(stderr, "DLSS frame-token creation failed\n");
                return 1;
            }
            sl::Constants constants = makeDLSSConstants(cam, tanHalf, aspect, {jitterX, jitterY}, frame == 0);
            if (slSetConstants(constants, *dlssFrameToken, dlssViewport) != sl::Result::eOk) {
                std::fprintf(stderr, "DLSS camera constants were rejected\n");
                return 1;
            }
        }
#endif
        std::memcpy(uboBuf.map, &u, sizeof(u));

        if (nrdEnabled) {
            nrdDIIntegration.NewFrame();
            nrdGIIntegration.NewFrame();
            nrd::CommonSettings nrdCommon{};
            setViewToClip(nrdCommon.viewToClipMatrix, tanHalf, aspect);
            std::memcpy(nrdCommon.viewToClipMatrixPrev, nrdCommon.viewToClipMatrix, sizeof(nrdCommon.viewToClipMatrix));
            setWorldToView(nrdCommon.worldToViewMatrix, cam);
            setWorldToView(nrdCommon.worldToViewMatrixPrev, lastCam);
            nrdCommon.resourceSize[0] = nrdCommon.resourceSizePrev[0] = static_cast<uint16_t>(renderW);
            nrdCommon.resourceSize[1] = nrdCommon.resourceSizePrev[1] = static_cast<uint16_t>(renderH);
            nrdCommon.rectSize[0] = nrdCommon.rectSizePrev[0] = static_cast<uint16_t>(renderW);
            nrdCommon.rectSize[1] = nrdCommon.rectSizePrev[1] = static_cast<uint16_t>(renderH);
            nrdCommon.motionVectorScale[0] = nrdCommon.motionVectorScale[1] = 1.0f;
            nrdCommon.cameraJitter[0] = u.jitter[0];
            nrdCommon.cameraJitter[1] = u.jitter[1];
            nrdCommon.cameraJitterPrev[0] = u.jitter[2];
            nrdCommon.cameraJitterPrev[1] = u.jitter[3];
            nrdCommon.denoisingRange = 50000.0f;
            nrdCommon.timeDeltaBetweenFrames = dt * 1000.0f;
            nrdCommon.frameIndex = frame;
            nrdCommon.accumulationMode = (!nrdWasEnabled || frame == 0)
                ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
            if (nrdDIIntegration.SetCommonSettings(nrdCommon) != nrd::Result::SUCCESS ||
                nrdGIIntegration.SetCommonSettings(nrdCommon) != nrd::Result::SUCCESS) {
                std::fprintf(stderr, "NRD common settings rejected\n");
                return 1;
            }
        }

        // ---- record
        VK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK(vkBeginCommandBuffer(cmd, &cbi));
        vkCmdResetQueryPool(cmd, qpool, 0, NSTAT + 1);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &sets[frame & 1], 0, nullptr);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);

        if (rebuilt) {                                   // BLAS -> TLAS, in place, ahead of the ray tracing passes
            recordBuild(cmd, blas, blasGeom, numTris);
            VkMemoryBarrier asb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            asb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            asb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 0, 1, &asb, 0, nullptr, 0, nullptr);
            recordBuild(cmd, tlas, tlasGeom, 1);
            asb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &asb, 0, nullptr, 0, nullptr);
        }

        if (cacheClear) {                                // new triangle layout: old cache contents would map to the wrong surfaces
            vkCmdFillBuffer(cmd, cacheIrr.buf, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, cacheRes.buf, 0, VK_WHOLE_SIZE, 0);
            VkMemoryBarrier fb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fb, 0, nullptr, 0, nullptr);
            cacheClear = false;
        }

        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        auto runCustom = [&](int stage) {
    for (auto& s : world.shaders) {
        if (s.stage != stage) continue;
        auto it = customGpu.find(s.id);
        if (it == customGpu.end() || !it->second.pipe) { s.runOnce = 0; continue; }
        if (!s.enabled && s.runOnce == 0) continue;
        if (s.runOnce > 0) --s.runOnce;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, it->second.pipe);
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(s.pc), s.pc);
        if (s.customGroups) vkCmdDispatch(cmd, s.groups[0], s.groups[1], s.groups[2]);
        else vkCmdDispatch(cmd, gx, gy, 1);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
};
        for (int i = 0; i < NP; ++i) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[i]);
        VkDescriptorSet passSet = i == 9 && nrdEnabled ? compositeSets[frame & 1] : sets[frame & 1];
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &passSet, 0, nullptr);
            if (i == 1) { if (rebuilt) vkCmdDispatch(cmd, GRID, GRID, GRID); }          // light_grid: camera independent, only when the scene changed
            else if (i == 2) { if (lightMode == 1) vkCmdDispatch(cmd, tilesX, tilesY, 1); } // tile_lights is only used by tile sampling
            else if (i == 6) { if (world.giMode > 0 && cacheBudget > 0) vkCmdDispatch(cmd, (cacheBudget + 63) / 64, 1, 1); }   // cache_update: a slice per frame
            else vkCmdDispatch(cmd, gx, gy, 1);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        if (i == 0) runCustom(0);          // after gbuffer
        if (i == 8) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 9);
        else if (i == 9) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 11);
        else vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, (uint32_t)i + 1);
        if (i == 8 && dlssActive && !nrdEnabled) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nrdPrepPipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &sets[frame & 1], 0, nullptr);
            vkCmdDispatch(cmd, gx, gy, 1);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
        if (i == 8 && nrdEnabled) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nrdPrepPipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &sets[frame & 1], 0, nullptr);
            vkCmdDispatch(cmd, gx, gy, 1);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);

            nrd::ResourceSnapshot diSnapshot;
            nrd::ResourceSnapshot giSnapshot;
            diSnapshot.restoreInitialState = giSnapshot.restoreInitialState = true;
            nrd::Resource motion = makeNrdResource(nrdMotion, VK_FORMAT_R16G16_SFLOAT);
            nrd::Resource normalRoughness = makeNrdResource(nrdNormalRoughness, VK_FORMAT_A2B10G10R10_UNORM_PACK32);
            nrd::Resource viewZ = makeNrdResource(nrdViewZ, VK_FORMAT_R32_SFLOAT);
            diSnapshot.SetResource(nrd::ResourceType::IN_MV, motion);
            diSnapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, normalRoughness);
            diSnapshot.SetResource(nrd::ResourceType::IN_VIEWZ, viewZ);
            diSnapshot.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, makeNrdResource(nrdDIInput, F16));
            diSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, makeNrdResource(nrdDIOutput, F16));
            giSnapshot.SetResource(nrd::ResourceType::IN_MV, motion);
            giSnapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, normalRoughness);
            giSnapshot.SetResource(nrd::ResourceType::IN_VIEWZ, viewZ);
            giSnapshot.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, makeNrdResource(nrdGIInput, F16));
            giSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, makeNrdResource(nrdGIOutput, F16));

            nri::CommandBufferVKDesc nrdCommandBuffer{};
            nrdCommandBuffer.vkCommandBuffer = cmd;
            nrdCommandBuffer.queueType = nri::QueueType::GRAPHICS;
            nrdDIIntegration.DenoiseVK(&nrdDIId, 1, nrdCommandBuffer, diSnapshot);
            nrdGIIntegration.DenoiseVK(&nrdGIId, 1, nrdCommandBuffer, giSnapshot);
        }
        if (i == 8) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 10);
    }
    runCustom(1);                          // post: after composite, before the blit

#ifdef RTENGINE_HAS_DLSS
        VkImage sourceImage = finalImg.img;
        uint32_t sourceWidth = renderW, sourceHeight = renderH;
        if (dlssActive) {
            sl::Resource colorIn{sl::ResourceType::eTex2d, reinterpret_cast<void*>(finalImg.img),
                                 reinterpret_cast<void*>(finalImg.mem), reinterpret_cast<void*>(finalImg.view),
                                 VK_IMAGE_LAYOUT_GENERAL};
            sl::Resource colorOut{sl::ResourceType::eTex2d, reinterpret_cast<void*>(dlssOutput.img),
                                  reinterpret_cast<void*>(dlssOutput.mem), reinterpret_cast<void*>(dlssOutput.view),
                                  VK_IMAGE_LAYOUT_GENERAL};
            sl::Resource depth{sl::ResourceType::eTex2d, reinterpret_cast<void*>(nrdViewZ.img),
                               reinterpret_cast<void*>(nrdViewZ.mem), reinterpret_cast<void*>(nrdViewZ.view),
                               VK_IMAGE_LAYOUT_GENERAL};
            sl::Resource motion{sl::ResourceType::eTex2d, reinterpret_cast<void*>(nrdMotion.img),
                                reinterpret_cast<void*>(nrdMotion.mem), reinterpret_cast<void*>(nrdMotion.view),
                                VK_IMAGE_LAYOUT_GENERAL};
            auto describeResource = [](sl::Resource& resource, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage) {
                resource.width = width;
                resource.height = height;
                resource.nativeFormat = static_cast<uint32_t>(format);
                resource.mipLevels = 1;
                resource.arrayLayers = 1;
                resource.usage = usage;
            };
            describeResource(colorIn, renderW, renderH, R8,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            describeResource(colorOut, W, H, R8,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            describeResource(depth, renderW, renderH, VK_FORMAT_R32_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            describeResource(motion, renderW, renderH, VK_FORMAT_R16G16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            const sl::Extent inputExtent{0, 0, renderW, renderH};
            const sl::Extent outputExtent{0, 0, W, H};
            sl::ResourceTag dlssTags[] = {
                {&colorIn, sl::kBufferTypeScalingInputColor, sl::eOnlyValidNow, &inputExtent},
                {&colorOut, sl::kBufferTypeScalingOutputColor, sl::eOnlyValidNow, &outputExtent},
                {&depth, sl::kBufferTypeLinearDepth, sl::eOnlyValidNow, &inputExtent},
                {&motion, sl::kBufferTypeMotionVectors, sl::eOnlyValidNow, &inputExtent}
            };
            if (slSetTagForFrame(*dlssFrameToken, dlssViewport, dlssTags, 4,
                                     reinterpret_cast<sl::CommandBuffer*>(cmd)) != sl::Result::eOk) {
                std::fprintf(stderr, "DLSS input tagging failed\n");
                return 1;
            }
            const sl::BaseStructure* dlssInputs[] = {&dlssViewport};
            if (slEvaluateFeature(sl::kFeatureDLSS, *dlssFrameToken, dlssInputs, 1,
                                      reinterpret_cast<sl::CommandBuffer*>(cmd)) != sl::Result::eOk) {
                std::fprintf(stderr, "DLSS evaluation failed\n");
                return 1;
            }
            sourceImage = dlssOutput.img;
            sourceWidth = W;
            sourceHeight = H;
        }
#else
        VkImage sourceImage = finalImg.img;
        uint32_t sourceWidth = renderW, sourceHeight = renderH;
#endif

        auto imgBarrier = [&](VkImage im, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst) {
            VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            ib.oldLayout = from; ib.newLayout = to; ib.srcAccessMask = src; ib.dstAccessMask = dst;
            ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.image = im; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            return ib;
        };
        VkImage sc = scImages[imageIndex];
        VkImageMemoryBarrier toBlit[2] = {
            imgBarrier(sourceImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            imgBarrier(sc, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT)};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 2, toBlit);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {(int32_t)sourceWidth, (int32_t)sourceHeight, 1};
        blit.dstOffsets[1] = {(int32_t)ext.width, (int32_t)ext.height, 1};
        vkCmdBlitImage(cmd, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        VkImageMemoryBarrier fromBlit[2] = {
            imgBarrier(sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            imgBarrier(sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0)};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 2, fromBlit);
        VK(vkEndCommandBuffer(cmd));

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1; si.pWaitSemaphores = &imageAvail; si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = &renderDone[imageIndex];
        VK(vkQueueSubmit(g_queue, 1, &si, fence));

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &renderDone[imageIndex];
        pi.swapchainCount = 1; pi.pSwapchains = &swapchain; pi.pImageIndices = &imageIndex;
        vkQueuePresentKHR(g_queue, &pi);

#ifdef RTENGINE_HAS_DLSS
        lastJitterX = jitterX;
        lastJitterY = jitterY;
#endif
        lastCam = cam;
        ++frame;
        nrdWasEnabled = nrdEnabled;
        if (world.giMode > 0 && cacheEntries > 0) cacheCursor = (cacheCursor + cacheBudget) % cacheEntries;

        // ---- stats (once per second)
        if (now - titleTime > 1.0 && accFrames > 0) {
            double total = 0;
            for (double a : passAcc) total += a;
            char title[300];
            std::snprintf(title, sizeof(title),
                          "rt_renderer | GPU %.2f ms | %zu objects, %u lights (area:%s), sampling:%s | GI:%s rate:%d | temporal:%s spatial:%s | DI cand:%u | bounces:%u | view:%u",
                          total / accFrames, world.objects.size(), numLights, world.areaLights ? "on" : "off",
            (lightMode == 0 ? "global" : lightMode == 1 ? "tile/grid" : "BVH"),
                          world.giMode == 0 ? "path" : world.giMode == 1 ? "surface cache" : "screen+BVH",
                          world.giMode == 2 ? (int)world.giOffscreenRate : world.giRate,
                          temporal ? "on" : "off", spatial ? "on" : "off", candTable[candIdx], giBounces, dbg);
            glfwSetWindowTitle(window, title);
            std::printf("GPU ms (avg of %d):", accFrames);
            for (int i = 0; i < NSTAT; ++i) std::printf("  %s %.2f", passNames[i], passAcc[i] / accFrames);
            std::printf("\n");
            for (double& a : passAcc) a = 0;
            accFrames = 0;
            titleTime = now;
        }
    }

    vkDeviceWaitIdle(g_dev);
#ifdef RTENGINE_HAS_DLSS
    if (streamlineInitialized && slShutdown() != sl::Result::eOk)
        std::fprintf(stderr, "Streamline shutdown reported an error\n");
#endif
    glfwTerminate();
    return 0;
}
