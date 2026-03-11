// benchmark.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Standalone Vulkan compute benchmark for three material-reorder kernels:
//   1. reorder.slang          — baseline bitonic sort (38 barriers)
//   2. reorder_optimized.slang — wave-intrinsic optimized (11 barriers)
//   3. reorder_setassoc.slang  — hash-table counting sort (3 barriers)
//
// Build (Windows):
//   mkdir build && cd build
//   cmake ..
//   cmake --build . --config Release
//   Release\benchmark.exe
// ─────────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <unistd.h>
#endif

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef HAS_RENDERDOC
#  include <renderdoc_app.h>
#endif

// ── Constants ───────────────────────────────────────────────────────────────
static constexpr uint32_t TILE_W          = 16;
static constexpr uint32_t TILE_H          = 16;
static constexpr uint32_t TILE_ELEMENTS   = 256;
static constexpr uint32_t WAVE_SIZE       = 32;
static constexpr uint32_t WARMUP_RUNS     = 10;
static constexpr uint32_t BENCHMARK_RUNS  = 200;

// ── Vulkan check macro ──────────────────────────────────────────────────────
#define VK_CHECK(call)                                                         \
    do {                                                                       \
        VkResult _r = (call);                                                  \
        if (_r != VK_SUCCESS) {                                                \
            fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__,        \
                    __LINE__);                                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ═════════════════════════════════════════════════════════════════════════════
// Tile generation — matches the notebook (seed=42, 6 materials)
// ═════════════════════════════════════════════════════════════════════════════
struct TileData {
    std::vector<uint32_t> data;        // [256] material IDs
    std::vector<uint32_t> material_ids; // the unique materials
};

static TileData generate_tile(uint32_t seed = 42, int num_materials = 6) {
    std::mt19937 rng(seed);

    // Generate unique material IDs
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    std::vector<uint32_t> mats;
    while ((int)mats.size() < num_materials) {
        uint32_t id = dist(rng);
        if (id == 0xFFFFFFFF) continue; // reserved sentinel
        bool dup = false;
        for (auto m : mats) if (m == id) { dup = true; break; }
        if (!dup) mats.push_back(id);
    }

    // Assign each element a random material
    std::vector<uint32_t> tile(TILE_ELEMENTS);
    std::uniform_int_distribution<int> mat_dist(0, num_materials - 1);
    for (uint32_t i = 0; i < TILE_ELEMENTS; i++)
        tile[i] = mats[mat_dist(rng)];

    return {tile, mats};
}

// ═════════════════════════════════════════════════════════════════════════════
// CPU reference reorder — matches the notebook's cpu_reorder()
// ═════════════════════════════════════════════════════════════════════════════
struct ReorderResult {
    std::vector<uint32_t> rx;   // [256] x coords
    std::vector<uint32_t> ry;   // [256] y coords
    uint32_t total_full;
};

static ReorderResult cpu_reorder(const std::vector<uint32_t>& tile) {
    const int N = (int)tile.size();

    // Stable sort indices by material ID
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](int a, int b) { return tile[a] < tile[b]; });

    // Build sorted material array
    std::vector<uint32_t> sorted_mats(N);
    for (int i = 0; i < N; i++) sorted_mats[i] = tile[perm[i]];

    // Identify run boundaries
    std::vector<int> run_starts;
    run_starts.push_back(0);
    for (int i = 1; i < N; i++)
        if (sorted_mats[i] != sorted_mats[i - 1])
            run_starts.push_back(i);

    // Compute run lengths
    std::vector<int> run_lengths;
    for (size_t r = 0; r < run_starts.size(); r++) {
        int end = (r + 1 < run_starts.size()) ? run_starts[r + 1] : N;
        run_lengths.push_back(end - run_starts[r]);
    }

    // Map each element to its run and local rank
    std::vector<int> run_id(N), local_rank(N);
    for (size_t r = 0; r < run_starts.size(); r++) {
        int start = run_starts[r];
        int len   = run_lengths[r];
        for (int j = 0; j < len; j++) {
            run_id[start + j]     = (int)r;
            local_rank[start + j] = j;
        }
    }

    // Compute full-wave count per run
    std::vector<int> full_per_run(run_lengths.size());
    for (size_t r = 0; r < run_lengths.size(); r++)
        full_per_run[r] = (run_lengths[r] / (int)WAVE_SIZE) * (int)WAVE_SIZE;

    // Classify full vs overflow, compute prefix sums
    uint32_t full_cnt = 0, ovfl_cnt = 0;
    std::vector<uint32_t> out_pos(N);
    for (int i = 0; i < N; i++) {
        bool is_full = local_rank[i] < full_per_run[run_id[i]];
        if (is_full) {
            out_pos[i] = full_cnt++;
        } else {
            out_pos[i] = ovfl_cnt++;  // will add total_full later
        }
    }
    uint32_t total_full = full_cnt;
    for (int i = 0; i < N; i++) {
        bool is_full = local_rank[i] < full_per_run[run_id[i]];
        if (!is_full) out_pos[i] += total_full;
    }

    // Scatter
    std::vector<uint32_t> rx(N), ry(N);
    for (int i = 0; i < N; i++) {
        rx[out_pos[i]] = perm[i] % TILE_W;
        ry[out_pos[i]] = perm[i] / TILE_W;
    }

    return {rx, ry, total_full};
}

// ═════════════════════════════════════════════════════════════════════════════
// Get directory containing the running executable
// ═════════════════════════════════════════════════════════════════════════════
static std::string get_exe_directory() {
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string path(buf);
    auto pos = path.find_last_of("\\/");
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
#else
    // Linux: readlink /proc/self/exe
    char buf[4096] = {};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return ".";
    buf[len] = '\0';
    std::string path(buf);
    auto pos = path.find_last_of('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : ".";
#endif
}

// ═════════════════════════════════════════════════════════════════════════════
// File I/O helper
// ═════════════════════════════════════════════════════════════════════════════
static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "Failed to open: %s\n", path.c_str());
        exit(1);
    }
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

// ═════════════════════════════════════════════════════════════════════════════
// Vulkan context
// ═════════════════════════════════════════════════════════════════════════════
struct VulkanContext {
    VkInstance               instance       = VK_NULL_HANDLE;
    VkPhysicalDevice         physDevice     = VK_NULL_HANDLE;
    VkDevice                 device         = VK_NULL_HANDLE;
    VkQueue                  computeQueue   = VK_NULL_HANDLE;
    uint32_t                 queueFamily    = 0;
    VkCommandPool            cmdPool        = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties devProps     = {};
    float                    timestampPeriod = 1.0f; // ns per tick
#ifdef HAS_RENDERDOC
    RENDERDOC_API_1_6_0*     rdoc_api       = nullptr;
#endif
};

// ═════════════════════════════════════════════════════════════════════════════
// RenderDoc initialization
// ═════════════════════════════════════════════════════════════════════════════
#ifdef HAS_RENDERDOC
static RENDERDOC_API_1_6_0* init_renderdoc() {
    RENDERDOC_API_1_6_0* rdoc_api = nullptr;

#ifdef _WIN32
    // Try to load renderdoc.dll — works if app is launched from RenderDoc
    // or if renderdoc.dll is on PATH / next to exe.
    HMODULE mod = GetModuleHandleA("renderdoc.dll");
    if (!mod)
        mod = LoadLibraryA("renderdoc.dll");
    if (!mod) {
        printf("RenderDoc: renderdoc.dll not loaded (launch from RenderDoc to enable capture)\n");
        return nullptr;
    }
    pRENDERDOC_GetAPI RENDERDOC_GetAPI =
        (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
#else
    void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (!mod) {
        printf("RenderDoc: librenderdoc.so not loaded (launch from RenderDoc to enable capture)\n");
        return nullptr;
    }
    pRENDERDOC_GetAPI RENDERDOC_GetAPI =
        (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
#endif

    if (!RENDERDOC_GetAPI) {
        printf("RenderDoc: could not find RENDERDOC_GetAPI\n");
        return nullptr;
    }

    int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
    if (ret != 1 || !rdoc_api) {
        printf("RenderDoc: RENDERDOC_GetAPI failed\n");
        return nullptr;
    }

    printf("RenderDoc: API loaded successfully\n");
    return rdoc_api;
}
#endif

static VulkanContext create_vulkan_context(bool enable_renderdoc) {
    VulkanContext ctx;

#ifdef HAS_RENDERDOC
    // Init RenderDoc BEFORE creating the Vulkan instance so it can hook the API.
    if (enable_renderdoc)
        ctx.rdoc_api = init_renderdoc();
#else
    (void)enable_renderdoc;
#endif

    // Instance
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "reorder_benchmark";
    appInfo.apiVersion       = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instCI = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instCI.pApplicationInfo = &appInfo;

#ifndef NDEBUG
    // Try to enable validation layers in debug builds
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availLayers.data());
    bool hasValidation = false;
    for (auto& l : availLayers)
        if (strcmp(l.layerName, layers[0]) == 0) { hasValidation = true; break; }
    if (hasValidation) {
        instCI.enabledLayerCount   = 1;
        instCI.ppEnabledLayerNames = layers;
    }
#endif

    VK_CHECK(vkCreateInstance(&instCI, nullptr, &ctx.instance));

    // Physical device — prefer discrete GPU
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr);
    if (devCount == 0) {
        fprintf(stderr, "No Vulkan-capable GPU found.\n");
        exit(1);
    }
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, devs.data());

    ctx.physDevice = devs[0]; // default: first
    for (auto& d : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx.physDevice = d;
            break;
        }
    }

    vkGetPhysicalDeviceProperties(ctx.physDevice, &ctx.devProps);
    ctx.timestampPeriod = ctx.devProps.limits.timestampPeriod; // ns per tick
    printf("GPU: %s\n", ctx.devProps.deviceName);
    printf("Timestamp period: %.1f ns/tick\n\n", ctx.timestampPeriod);

    // Compute queue family
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physDevice, &qfCount, qfProps.data());

    ctx.queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx.queueFamily = i;
            // Prefer compute-only queue if available
            if (!(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                break;
        }
    }
    if (ctx.queueFamily == UINT32_MAX) {
        fprintf(stderr, "No compute queue family found.\n");
        exit(1);
    }

    // Check timestamp support
    if (qfProps[ctx.queueFamily].timestampValidBits == 0) {
        fprintf(stderr, "Warning: timestamp queries not supported on this queue.\n");
    }

    // Query subgroup properties for diagnostics
    VkPhysicalDeviceSubgroupProperties subgroupProps = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props2 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(ctx.physDevice, &props2);
    printf("Subgroup size: %u\n", subgroupProps.subgroupSize);
    printf("Subgroup supported stages: 0x%x\n", subgroupProps.supportedStages);
    printf("Subgroup supported operations: 0x%x\n\n", subgroupProps.supportedOperations);

    // Enable Vulkan 1.1+ features needed by wave intrinsics in the shaders.
    // Chain VkPhysicalDeviceVulkan11Features and 12Features to enable
    // subgroup operations and shader int64/float16 if available.
    VkPhysicalDeviceVulkan12Features features12 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};

    VkPhysicalDeviceVulkan11Features features11 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    features11.pNext = &features12;

    // Query what the device actually supports, then request all of it.
    VkPhysicalDeviceFeatures2 features2 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &features11;
    vkGetPhysicalDeviceFeatures2(ctx.physDevice, &features2);

    // Logical device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qCI = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qCI.queueFamilyIndex = ctx.queueFamily;
    qCI.queueCount       = 1;
    qCI.pQueuePriorities = &priority;

    VkDeviceCreateInfo devCI = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos    = &qCI;
    devCI.pNext                = &features2;

    VK_CHECK(vkCreateDevice(ctx.physDevice, &devCI, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.computeQueue);

    // Command pool
    VkCommandPoolCreateInfo poolCI = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCI.queueFamilyIndex = ctx.queueFamily;
    VK_CHECK(vkCreateCommandPool(ctx.device, &poolCI, nullptr, &ctx.cmdPool));

    return ctx;
}

// ═════════════════════════════════════════════════════════════════════════════
// Buffer helpers
// ═════════════════════════════════════════════════════════════════════════════
struct GpuBuffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
};

static uint32_t find_memory_type(const VulkanContext& ctx, uint32_t typeBits,
                                  VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx.physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    fprintf(stderr, "Failed to find suitable memory type.\n");
    exit(1);
}

static GpuBuffer create_buffer(const VulkanContext& ctx, VkDeviceSize size,
                                VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags memProps) {
    GpuBuffer buf;
    buf.size = size;

    VkBufferCreateInfo bCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bCI.size        = size;
    bCI.usage       = usage;
    bCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx.device, &bCI, nullptr, &buf.buffer));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device, buf.buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize  = memReqs.size;
    allocInfo.memoryTypeIndex = find_memory_type(ctx, memReqs.memoryTypeBits, memProps);
    VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &buf.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device, buf.buffer, buf.memory, 0));

    return buf;
}

static void upload_buffer(const VulkanContext& ctx, GpuBuffer& buf,
                           const void* data, VkDeviceSize size) {
    void* mapped;
    VK_CHECK(vkMapMemory(ctx.device, buf.memory, 0, size, 0, &mapped));
    memcpy(mapped, data, size);
    vkUnmapMemory(ctx.device, buf.memory);
}

static void readback_buffer(const VulkanContext& ctx, const GpuBuffer& buf,
                              void* dst, VkDeviceSize size) {
    void* mapped;
    VK_CHECK(vkMapMemory(ctx.device, buf.memory, 0, size, 0, &mapped));
    memcpy(dst, mapped, size);
    vkUnmapMemory(ctx.device, buf.memory);
}

static void destroy_buffer(const VulkanContext& ctx, GpuBuffer& buf) {
    if (buf.buffer) vkDestroyBuffer(ctx.device, buf.buffer, nullptr);
    if (buf.memory) vkFreeMemory(ctx.device, buf.memory, nullptr);
    buf.buffer = VK_NULL_HANDLE;
    buf.memory = VK_NULL_HANDLE;
}

// ═════════════════════════════════════════════════════════════════════════════
// Compute pipeline
// ═════════════════════════════════════════════════════════════════════════════
struct ComputePipeline {
    VkShaderModule       shaderModule   = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout      = VK_NULL_HANDLE;
    VkPipelineLayout     pipelineLayout = VK_NULL_HANDLE;
    VkPipeline           pipeline       = VK_NULL_HANDLE;
    VkDescriptorPool     dsPool         = VK_NULL_HANDLE;
    VkDescriptorSet      dsSet          = VK_NULL_HANDLE;
    std::string          name;
};

// ── Minimal SPIR-V reflection ───────────────────────────────────────────────
// We parse the SPIR-V binary to find descriptor bindings rather than
// vendoring the full spirv-reflect library.  Our shaders are simple:
// each has exactly 4 storage buffer bindings in set 0.
//
// SPIR-V layout:
//   Word 0: magic (0x07230203)
//   Word 3: bound (largest ID + 1)
//   Word 5+: instructions
//
// We look for OpDecorate instructions with Binding decoration to discover
// which SPIR-V IDs map to which binding numbers.

struct SpvBinding {
    uint32_t binding;
    uint32_t set;
};

static std::vector<SpvBinding> reflect_bindings(const std::vector<uint8_t>& spirv) {
    if (spirv.size() < 20 || spirv.size() % 4 != 0) {
        fprintf(stderr, "Invalid SPIR-V binary.\n");
        exit(1);
    }

    const uint32_t* words = reinterpret_cast<const uint32_t*>(spirv.data());
    uint32_t wordCount = (uint32_t)(spirv.size() / 4);

    if (words[0] != 0x07230203) {
        fprintf(stderr, "Bad SPIR-V magic.\n");
        exit(1);
    }

    // First pass: collect binding and descriptor set decorations per ID
    struct IdInfo {
        int32_t binding = -1;
        int32_t set     = -1;
    };
    uint32_t bound = words[3];
    std::vector<IdInfo> ids(bound);

    uint32_t i = 5;
    while (i < wordCount) {
        uint32_t instrLen = words[i] >> 16;
        uint32_t opcode   = words[i] & 0xFFFF;

        if (instrLen == 0) break;

        // OpDecorate = 71
        if (opcode == 71 && instrLen >= 4) {
            uint32_t targetId   = words[i + 1];
            uint32_t decoration = words[i + 2];
            uint32_t value      = words[i + 3];
            if (targetId < bound) {
                if (decoration == 33) // Binding
                    ids[targetId].binding = (int32_t)value;
                else if (decoration == 34) // DescriptorSet
                    ids[targetId].set = (int32_t)value;
            }
        }

        i += instrLen;
    }

    // Collect entries that have both binding and set
    std::vector<SpvBinding> bindings;
    for (uint32_t id = 0; id < bound; id++) {
        if (ids[id].binding >= 0 && ids[id].set >= 0) {
            bindings.push_back({(uint32_t)ids[id].binding, (uint32_t)ids[id].set});
        }
    }

    // Sort by binding number
    std::sort(bindings.begin(), bindings.end(),
              [](const SpvBinding& a, const SpvBinding& b) {
                  return a.binding < b.binding;
              });

    return bindings;
}

// Extract the first entry point name from SPIR-V.
// OpEntryPoint (opcode 15): word layout is [op|len, ExecutionModel, id, name...]
// The name is a nul-terminated UTF-8 string packed into 32-bit words starting at word 3.
static std::string reflect_entry_point(const std::vector<uint8_t>& spirv) {
    const uint32_t* words = reinterpret_cast<const uint32_t*>(spirv.data());
    uint32_t wordCount = (uint32_t)(spirv.size() / 4);

    uint32_t i = 5;
    while (i < wordCount) {
        uint32_t instrLen = words[i] >> 16;
        uint32_t opcode   = words[i] & 0xFFFF;
        if (instrLen == 0) break;

        // OpEntryPoint = 15
        if (opcode == 15 && instrLen >= 4) {
            // Name starts at word i+3, packed as nul-terminated chars
            const char* nameStart = reinterpret_cast<const char*>(&words[i + 3]);
            // Max name length = remaining words * 4
            size_t maxLen = (instrLen - 3) * 4;
            return std::string(nameStart, strnlen(nameStart, maxLen));
        }

        i += instrLen;
    }
    return "main"; // fallback
}

static ComputePipeline create_compute_pipeline(
        const VulkanContext& ctx,
        const std::string& spvPath,
        const std::string& entryPoint,
        const std::string& name) {
    ComputePipeline cp;
    cp.name = name;

    // Load SPIR-V
    auto spirv = read_file(spvPath);

    // Reflect entry point — use whatever slangc actually emitted
    std::string spvEntry = reflect_entry_point(spirv);
    if (spvEntry != entryPoint) {
        printf("[%s] NOTE: SPIR-V entry point is \"%s\" (not \"%s\")\n",
               name.c_str(), spvEntry.c_str(), entryPoint.c_str());
    }

    // Reflect bindings
    auto bindings = reflect_bindings(spirv);
    printf("[%s] Reflected %zu bindings:", name.c_str(), bindings.size());
    for (auto& b : bindings) printf(" set=%u binding=%u", b.set, b.binding);
    printf("\n");

    uint32_t numBindings = (uint32_t)bindings.size();
    if (numBindings < 3 || numBindings > 4) {
        fprintf(stderr, "Expected 3 or 4 bindings, got %u for %s\n",
                numBindings, name.c_str());
        exit(1);
    }

    // Shader module
    VkShaderModuleCreateInfo smCI = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smCI.codeSize = spirv.size();
    smCI.pCode    = reinterpret_cast<const uint32_t*>(spirv.data());
    VK_CHECK(vkCreateShaderModule(ctx.device, &smCI, nullptr, &cp.shaderModule));

    // Descriptor set layout — storage buffers from reflected bindings
    std::vector<VkDescriptorSetLayoutBinding> dsBindings(numBindings);
    for (uint32_t i = 0; i < numBindings; i++) {
        dsBindings[i] = {};
        dsBindings[i].binding         = bindings[i].binding;
        dsBindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        dsBindings[i].descriptorCount = 1;
        dsBindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsLayoutCI =
        {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsLayoutCI.bindingCount = numBindings;
    dsLayoutCI.pBindings    = dsBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dsLayoutCI, nullptr, &cp.dsLayout));

    // Pipeline layout
    VkPipelineLayoutCreateInfo plCI = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts    = &cp.dsLayout;
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &plCI, nullptr, &cp.pipelineLayout));

    // Compute pipeline
    VkComputePipelineCreateInfo cpCI = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpCI.stage.module = cp.shaderModule;
    cpCI.stage.pName  = spvEntry.c_str();
    cpCI.layout       = cp.pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpCI,
                                       nullptr, &cp.pipeline));

    // Descriptor pool + set
    VkDescriptorPoolSize poolSize = {};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = numBindings;

    VkDescriptorPoolCreateInfo dpCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCI.maxSets       = 1;
    dpCI.poolSizeCount = 1;
    dpCI.pPoolSizes    = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &dpCI, nullptr, &cp.dsPool));

    VkDescriptorSetAllocateInfo dsAI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAI.descriptorPool     = cp.dsPool;
    dsAI.descriptorSetCount = 1;
    dsAI.pSetLayouts        = &cp.dsLayout;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dsAI, &cp.dsSet));

    return cp;
}

static void bind_buffers(const VulkanContext& ctx, ComputePipeline& cp,
                          const std::vector<SpvBinding>& bindings,
                          const std::vector<GpuBuffer>& bufs) {
    uint32_t n = (uint32_t)bindings.size();
    std::vector<VkWriteDescriptorSet> writes(n);
    std::vector<VkDescriptorBufferInfo> bufInfos(n);

    for (uint32_t i = 0; i < n; i++) {
        bufInfos[i].buffer = bufs[i].buffer;
        bufInfos[i].offset = 0;
        bufInfos[i].range  = bufs[i].size;

        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet          = cp.dsSet;
        writes[i].dstBinding      = bindings[i].binding;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufInfos[i];
    }

    vkUpdateDescriptorSets(ctx.device, n, writes.data(), 0, nullptr);
}

static void destroy_pipeline(const VulkanContext& ctx, ComputePipeline& cp) {
    if (cp.pipeline)       vkDestroyPipeline(ctx.device, cp.pipeline, nullptr);
    if (cp.pipelineLayout) vkDestroyPipelineLayout(ctx.device, cp.pipelineLayout, nullptr);
    if (cp.dsPool)         vkDestroyDescriptorPool(ctx.device, cp.dsPool, nullptr);
    if (cp.dsLayout)       vkDestroyDescriptorSetLayout(ctx.device, cp.dsLayout, nullptr);
    if (cp.shaderModule)   vkDestroyShaderModule(ctx.device, cp.shaderModule, nullptr);
}

// ═════════════════════════════════════════════════════════════════════════════
// Timestamp query pool
// ═════════════════════════════════════════════════════════════════════════════
static VkQueryPool create_timestamp_pool(const VulkanContext& ctx, uint32_t count) {
    VkQueryPoolCreateInfo qpCI = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpCI.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qpCI.queryCount = count;
    VkQueryPool pool;
    VK_CHECK(vkCreateQueryPool(ctx.device, &qpCI, nullptr, &pool));
    return pool;
}

// ═════════════════════════════════════════════════════════════════════════════
// Verification
// ═════════════════════════════════════════════════════════════════════════════
static bool verify_result(const std::string& name,
                           const std::vector<uint32_t>& tile,
                           const std::vector<uint32_t>& gpu_rxy,
                           uint32_t gpu_total_full,
                           const ReorderResult& cpu_ref) {
    bool ok = true;

    // Unpack helper
    auto unpack_x = [](uint32_t packed) { return packed & 0xFFFFu; };
    auto unpack_y = [](uint32_t packed) { return packed >> 16u; };

    // 1. total_full must be a multiple of 32
    if (gpu_total_full % WAVE_SIZE != 0) {
        printf("  FAIL: total_full=%u is not a multiple of %u\n",
               gpu_total_full, WAVE_SIZE);
        ok = false;
    }

    // 2. total_full must match CPU
    if (gpu_total_full != cpu_ref.total_full) {
        printf("  FAIL: total_full GPU=%u vs CPU=%u\n",
               gpu_total_full, cpu_ref.total_full);
        ok = false;
    }

    // 3. All positions must be in range and unique
    std::vector<bool> seen(TILE_ELEMENTS, false);
    for (uint32_t i = 0; i < TILE_ELEMENTS; i++) {
        uint32_t rx = unpack_x(gpu_rxy[i]);
        uint32_t ry = unpack_y(gpu_rxy[i]);
        if (rx >= TILE_W || ry >= TILE_H) {
            printf("  FAIL: out[%u] = (%u, %u) out of range\n", i, rx, ry);
            ok = false;
            continue;
        }
        uint32_t flat = ry * TILE_W + rx;
        if (seen[flat]) {
            printf("  FAIL: duplicate position (%u, %u) at index %u\n", rx, ry, i);
            ok = false;
        }
        seen[flat] = true;
    }

    // 4. Every 32-block in full section must be uniform material
    for (uint32_t base = 0; base < gpu_total_full; base += WAVE_SIZE) {
        uint32_t first_mat = tile[unpack_y(gpu_rxy[base]) * TILE_W + unpack_x(gpu_rxy[base])];
        for (uint32_t j = 1; j < WAVE_SIZE && base + j < TILE_ELEMENTS; j++) {
            uint32_t idx = base + j;
            uint32_t mat = tile[unpack_y(gpu_rxy[idx]) * TILE_W + unpack_x(gpu_rxy[idx])];
            if (mat != first_mat) {
                printf("  FAIL: full-wave block at %u not uniform "
                       "(mat[%u]=%u != mat[%u]=%u)\n",
                       base, base, first_mat, idx, mat);
                ok = false;
                break;
            }
        }
    }

    // 5. Overflow section must be grouped by material
    {
        std::vector<uint32_t> ovfl_mats;
        for (uint32_t i = gpu_total_full; i < TILE_ELEMENTS; i++) {
            uint32_t mat = tile[unpack_y(gpu_rxy[i]) * TILE_W + unpack_x(gpu_rxy[i])];
            ovfl_mats.push_back(mat);
        }
        for (size_t i = 1; i < ovfl_mats.size(); i++) {
            if (ovfl_mats[i] != ovfl_mats[i - 1]) {
                for (size_t j = 0; j < i - 1; j++) {
                    if (ovfl_mats[j] == ovfl_mats[i]) {
                        printf("  FAIL: overflow not grouped — mat %u "
                               "reappears at overflow[%zu] after gap\n",
                               ovfl_mats[i], i);
                        ok = false;
                        goto done_ovfl_check;
                    }
                }
            }
        }
        done_ovfl_check:;
    }

    return ok;
}

// ═════════════════════════════════════════════════════════════════════════════
// Statistics
// ═════════════════════════════════════════════════════════════════════════════
struct Stats {
    double mean, median, min_val, p5, p95;
};

static Stats compute_stats(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    int n = (int)times.size();
    double sum = 0;
    for (auto t : times) sum += t;

    Stats s;
    s.mean    = sum / n;
    s.median  = (n % 2 == 0) ? (times[n/2 - 1] + times[n/2]) / 2.0 : times[n/2];
    s.min_val = times[0];
    s.p5      = times[(int)(n * 0.05)];
    s.p95     = times[(int)(n * 0.95)];
    return s;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    // ── CLI parsing ─────────────────────────────────────────────────────
    bool capture_mode = false;
    std::string capture_kernel = "";  // empty = capture all kernels
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--capture") == 0) {
            capture_mode = true;
            // Optional: next arg can be a kernel name filter
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                capture_kernel = argv[++i];
            }
        }
    }

    if (capture_mode) {
#ifdef HAS_RENDERDOC
        printf("RenderDoc capture mode enabled");
        if (!capture_kernel.empty())
            printf(" (filter: \"%s\")", capture_kernel.c_str());
        printf("\n");
#else
        fprintf(stderr, "Error: --capture requires building with RenderDoc support.\n");
        fprintf(stderr, "  Install RenderDoc and reconfigure with cmake.\n");
        return 1;
#endif
    }

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Material Reorder — Vulkan Compute Benchmark\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // ── Vulkan init ─────────────────────────────────────────────────────
    auto ctx = create_vulkan_context(capture_mode);

    // ── Generate tile ───────────────────────────────────────────────────
    auto tileData = generate_tile(42, 6);
    printf("Tile: %u elements, %zu unique materials\n",
           TILE_ELEMENTS, tileData.material_ids.size());

    // ── CPU reference ───────────────────────────────────────────────────
    auto cpuRef = cpu_reorder(tileData.data);
    printf("CPU reference: total_full = %u\n\n", cpuRef.total_full);

    // ── GPU buffers ─────────────────────────────────────────────────────
    // Use host-visible + host-coherent for simplicity (small data)
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags memFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    GpuBuffer buf_tile      = create_buffer(ctx, TILE_ELEMENTS * sizeof(uint32_t), usage, memFlags);
    GpuBuffer buf_rxy       = create_buffer(ctx, TILE_ELEMENTS * sizeof(uint32_t), usage, memFlags);
    GpuBuffer buf_totalFull = create_buffer(ctx, sizeof(uint32_t), usage, memFlags);

    // Upload tile data
    upload_buffer(ctx, buf_tile, tileData.data.data(),
                  TILE_ELEMENTS * sizeof(uint32_t));

    // ── Timestamp query pool ────────────────────────────────────────────
    VkQueryPool tsPool = create_timestamp_pool(ctx, 2); // start + end

    // ── Command buffer ──────────────────────────────────────────────────
    VkCommandBufferAllocateInfo cbAI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool        = ctx.cmdPool;
    cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;
    VkCommandBuffer cmdBuf;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cbAI, &cmdBuf));

    // ── Fence for synchronization ───────────────────────────────────────
    VkFenceCreateInfo fenceCI = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(ctx.device, &fenceCI, nullptr, &fence));

    // ── Kernel configs ──────────────────────────────────────────────────
    struct KernelConfig {
        std::string spvPath;
        std::string entryPoint;
        std::string name;
    };

    std::string exeDir = get_exe_directory();
    std::vector<KernelConfig> kernels = {
        {exeDir + "/reorder.spv",           "reorderByMaterial",        "Baseline (bitonic)"},
        {exeDir + "/reorder_optimized.spv", "reorderByMaterialOpt",     "Optimized (wave)"},
        {exeDir + "/reorder_setassoc.spv",  "reorderByMaterialSetAssoc","Set-Assoc (hash)"},
        {exeDir + "/reorder_setassoc_v2.spv","reorderByMaterialSetAssocV2","Set-Assoc v2 (wave-coop)"},
        {exeDir + "/reorder_setassoc_v3.spv","reorderByMaterialWaveMerge","Set-Assoc v3 (wave-merge)"},
        {exeDir + "/reorder_setassoc_v4.spv","reorderByMaterialV4","Set-Assoc v4 (parallel-merge)"},
    };

    // ── Results storage ─────────────────────────────────────────────────
    struct KernelResult {
        std::string name;
        Stats stats;
        bool verified;
    };
    std::vector<KernelResult> results;

    // ── Run each kernel ─────────────────────────────────────────────────
    for (auto& kern : kernels) {
        printf("───────────────────────────────────────────────────────────\n");
        printf("Kernel: %s\n", kern.name.c_str());
        printf("  SPV:   %s\n", kern.spvPath.c_str());
        printf("  Entry: %s\n", kern.entryPoint.c_str());

        // Create pipeline
        auto pipeline = create_compute_pipeline(ctx, kern.spvPath,
                                                 kern.entryPoint, kern.name);

        // Reflect bindings for buffer binding
        auto bindings = reflect_bindings(read_file(kern.spvPath));

        // Bind buffers: binding 0=tileData, 1=reorderedXY, 2=totalFull
        std::vector<GpuBuffer> bufs = {buf_tile, buf_rxy, buf_totalFull};
        bind_buffers(ctx, pipeline, bindings, bufs);

        // Lambda to dispatch once and optionally time
        auto dispatch_once = [&](bool record_time) -> double {
            // Clear output buffers
            uint32_t zeros[TILE_ELEMENTS] = {};
            upload_buffer(ctx, buf_rxy, zeros, TILE_ELEMENTS * sizeof(uint32_t));
            uint32_t zero = 0;
            upload_buffer(ctx, buf_totalFull, &zero, sizeof(uint32_t));

            VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));

            if (record_time) {
                vkCmdResetQueryPool(cmdBuf, tsPool, 0, 2);
                vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   tsPool, 0);
            }

            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline.pipelineLayout, 0, 1,
                                    &pipeline.dsSet, 0, nullptr);
            vkCmdDispatch(cmdBuf, 1, 1, 1);

            if (record_time) {
                vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                   tsPool, 1);
            }

            VK_CHECK(vkEndCommandBuffer(cmdBuf));

            VK_CHECK(vkResetFences(ctx.device, 1, &fence));
            VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers    = &cmdBuf;
            VK_CHECK(vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, fence));
            VK_CHECK(vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX));

            double us = 0;
            if (record_time) {
                uint64_t ts[2];
                VK_CHECK(vkGetQueryPoolResults(
                    ctx.device, tsPool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                us = (double)(ts[1] - ts[0]) * ctx.timestampPeriod / 1000.0;
            }
            return us;
        };

        // Warmup
        printf("  Warmup: %u runs...\n", WARMUP_RUNS);
        for (uint32_t i = 0; i < WARMUP_RUNS; i++)
            dispatch_once(false);

        // Benchmark
        printf("  Benchmark: %u runs...\n", BENCHMARK_RUNS);
        std::vector<double> times;
        times.reserve(BENCHMARK_RUNS);
        for (uint32_t i = 0; i < BENCHMARK_RUNS; i++)
            times.push_back(dispatch_once(true));

        auto stats = compute_stats(times);

        // Readback and verify last run's results
        std::vector<uint32_t> gpu_rxy(TILE_ELEMENTS);
        uint32_t gpu_total_full = 0;
        readback_buffer(ctx, buf_rxy, gpu_rxy.data(), TILE_ELEMENTS * sizeof(uint32_t));
        readback_buffer(ctx, buf_totalFull, &gpu_total_full, sizeof(uint32_t));

        bool verified = verify_result(kern.name, tileData.data, gpu_rxy, gpu_total_full, cpuRef);

        printf("  Verification: %s\n", verified ? "PASS" : "FAIL");
        printf("  GPU total_full = %u\n", gpu_total_full);
        printf("  Timing (us): mean=%.2f  median=%.2f  min=%.2f  p5=%.2f  p95=%.2f\n",
               stats.mean, stats.median, stats.min_val, stats.p5, stats.p95);

        results.push_back({kern.name, stats, verified});

        // ── RenderDoc capture ────────────────────────────────────────
        // Run one extra dispatch wrapped in a capture frame, after timing
        // is done so it doesn't affect benchmark results.
#ifdef HAS_RENDERDOC
        if (ctx.rdoc_api) {
            bool should_capture = capture_kernel.empty() ||
                kern.name.find(capture_kernel) != std::string::npos;
            if (should_capture) {
                printf("  Capturing RenderDoc frame...\n");
                ctx.rdoc_api->StartFrameCapture(
                    RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(ctx.instance), NULL);
                dispatch_once(false);
                ctx.rdoc_api->EndFrameCapture(
                    RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(ctx.instance), NULL);
                printf("  Capture saved.\n");
            }
        }
#endif

        destroy_pipeline(ctx, pipeline);
    }

    // ── Summary table ───────────────────────────────────────────────────
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  Summary  (%u warmup + %u timed runs)\n", WARMUP_RUNS, BENCHMARK_RUNS);
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("%-25s %8s %8s %8s %8s %8s %6s\n",
           "Kernel", "Mean", "Median", "Min", "P5", "P95", "Pass");
    printf("%-25s %8s %8s %8s %8s %8s %6s\n",
           "─────────────────────────", "────────", "────────", "────────",
           "────────", "────────", "──────");

    for (auto& r : results) {
        printf("%-25s %7.2fus %7.2fus %7.2fus %7.2fus %7.2fus %6s\n",
               r.name.c_str(),
               r.stats.mean, r.stats.median, r.stats.min_val,
               r.stats.p5, r.stats.p95,
               r.verified ? "YES" : "NO");
    }

    // Speedup ratios
    if (results.size() >= 2) {
        printf("\n── Speedup vs Baseline (median) ──\n");
        double baseline = results[0].stats.median;
        for (size_t i = 1; i < results.size(); i++) {
            double speedup = baseline / results[i].stats.median;
            printf("  %s: %.2fx\n", results[i].name.c_str(), speedup);
        }
    }

    // ── Cleanup ─────────────────────────────────────────────────────────
    vkDestroyFence(ctx.device, fence, nullptr);
    vkDestroyQueryPool(ctx.device, tsPool, nullptr);
    vkFreeCommandBuffers(ctx.device, ctx.cmdPool, 1, &cmdBuf);
    destroy_buffer(ctx, buf_tile);
    destroy_buffer(ctx, buf_rxy);
    destroy_buffer(ctx, buf_totalFull);
    vkDestroyCommandPool(ctx.device, ctx.cmdPool, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);

    printf("\nDone.\n");
    return 0;
}
