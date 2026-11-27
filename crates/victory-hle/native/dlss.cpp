// Optional NVIDIA DLSS backend using the ABI of VIctory's FidelityFX bridge.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <d3d12.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_d3d.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {
constexpr uint32_t Ok = 0, UnknownType = 2, RuntimeError = 3, NoProvider = 4, MemoryError = 5, Invalid = 6;
constexpr uint64_t CreateType = 0x10000, DispatchType = 0x10001;
constexpr uint64_t ProviderId = 0x444c53530000000d;
constexpr char ProviderName[] = "DLSS 4.5 (preset M)";

struct Header { uint64_t kind; Header* next; };
struct Dimensions { uint32_t width, height; };
struct Create { Header header; uint32_t flags; Dimensions render, output; void* message; };
struct Resource { ID3D12Resource* resource; uint32_t description[8]; uint32_t state; };
struct Dispatch {
    Header header;
    ID3D12GraphicsCommandList* list;
    Resource color, depth, motion, exposure, reactive, transparency, output;
    float jitter_x, jitter_y, motion_x, motion_y;
    Dimensions render, display;
    uint32_t sharpen;
    float sharpness, frame_ms, pre_exposure;
    uint32_t reset;
    float near_plane, far_plane, fov, meters;
    uint32_t flags;
};
struct Versions { Header header; uint64_t type; ID3D12Device* device; uint64_t* count; uint64_t* ids; const char** names; };
struct Provider { Header header; uint64_t id; const char* name; };
struct Ratio { Header header; uint32_t quality; float* ratio; };
struct Resolution { Header header; Dimensions output; uint32_t quality; uint32_t* width; uint32_t* height; };
struct PhaseCount { Header header; uint32_t render, output; int32_t* phases; };
struct Jitter { Header header; int32_t index, phases; float* x; float* y; };
struct Memory { Header header; uint64_t* bytes; };
struct Requirements { Header header; uint64_t required, optional; };
static_assert(sizeof(Create) == 0x30);
static_assert(sizeof(Resource) == 0x30);
static_assert(sizeof(Dispatch) == 0x1b0);

// Retain the features with the console context, including across a change in
// output size while an earlier list might still use the old feature.
struct Feature {
    Dimensions output;
    bool auto_exposure;
    NVSDK_NGX_Handle* handle = nullptr;
    Feature(Dimensions size, bool automatic) : output(size), auto_exposure(automatic) {}
    ~Feature() { if (handle) NVSDK_NGX_D3D12_ReleaseFeature(handle); }
    Feature(Feature const&) = delete;
    Feature(Feature&& other) noexcept : Feature(other.output, other.auto_exposure) {
        handle = other.handle;
        other.handle = nullptr;
    }
};
struct Context {
    Dimensions max_render, max_output;
    uint32_t flags;
    NVSDK_NGX_Parameter* params = nullptr;
    std::vector<Feature> features;
    ~Context() {
        features.clear();
        if (params) NVSDK_NGX_D3D12_DestroyParameters(params);
    }
};
std::mutex ngx_mutex;
ID3D12Device* device_in_use = nullptr;

uint32_t ngx_error(const char* operation, NVSDK_NGX_Result result) {
    std::fprintf(stderr, "VIctory DLSS: %s failed (%#x)\n", operation, static_cast<unsigned>(result));
    return RuntimeError;
}

std::wstring module_directory() {
    HMODULE module = nullptr;
    const auto flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&module_directory), &module)) return {};
    std::wstring path(32768, L'\0');
    const auto n = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (n == 0 || n == path.size()) return {};
    path.resize(n);
    return path.substr(0, path.find_last_of(L"\\/"));
}

uint32_t initialize(ID3D12Device* device) {
    if (!device) return Invalid;
    if (device_in_use) return device == device_in_use ? Ok : Invalid;
    const auto directory = module_directory();
    if (directory.empty() ||
        GetFileAttributesW((directory + L"\\nvngx_dlss.dll").c_str()) == INVALID_FILE_ATTRIBUTES) return NoProvider;
    const wchar_t* paths[] = { directory.c_str() };
    NVSDK_NGX_FeatureCommonInfo info{};
    info.PathListInfo = { paths, 1 };
    const auto init = NVSDK_NGX_D3D12_Init_with_ProjectID(
        "8c87b5fe-2ba7-4a7c-8cdd-faa5f12d5ad6", NVSDK_NGX_ENGINE_TYPE_CUSTOM,
        "VIctory", directory.c_str(), device, &info);
    if (NVSDK_NGX_FAILED(init)) return ngx_error("initialize NGX", init);

    NVSDK_NGX_Parameter* capabilities = nullptr;
    const auto result = NVSDK_NGX_D3D12_GetCapabilityParameters(&capabilities);
    int available = 0;
    if (NVSDK_NGX_SUCCEED(result)) {
        capabilities->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &available);
        NVSDK_NGX_D3D12_DestroyParameters(capabilities);
    }
    if (!available) {
        NVSDK_NGX_D3D12_Shutdown1(device);
        return NVSDK_NGX_FAILED(result) ? ngx_error("query capabilities", result) : NoProvider;
    }
    device_in_use = device;
    return Ok;
}

float halton(int32_t index, int32_t base) {
    float result = 0, fraction = 1;
    for (; index > 0; index /= base) {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(index % base);
    }
    return result;
}

uint32_t create_feature(Context& context, Feature& feature, ID3D12GraphicsCommandList* list) {
    constexpr uint32_t Hdr = 1, DisplayMotion = 2, JitteredMotion = 4, InvertedDepth = 8;
    NVSDK_NGX_DLSS_Create_Params create{};
    create.Feature.InWidth = context.max_render.width;
    create.Feature.InHeight = context.max_render.height;
    create.Feature.InTargetWidth = feature.output.width;
    create.Feature.InTargetHeight = feature.output.height;
    // The title chooses the resolution. Request the 4.5 model for the Series S
    // 720p-to-1080p Quality ratio.
    create.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    const auto flags = context.flags;
    create.InFeatureCreateFlags =
        ((flags & Hdr) ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0) |
        ((flags & DisplayMotion) ? 0 : NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) |
        ((flags & JitteredMotion) ? NVSDK_NGX_DLSS_Feature_Flags_MVJittered : 0) |
        ((flags & InvertedDepth) ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0) |
        (feature.auto_exposure ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0);
    context.params->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                        static_cast<unsigned>(NVSDK_NGX_DLSS_Hint_Render_Preset_M));
    const auto result = NGX_D3D12_CREATE_DLSS_EXT(list, 1, 1, &feature.handle, context.params, &create);
    return NVSDK_NGX_SUCCEED(result) ? Ok : ngx_error("create preset M", result);
}
} // namespace

#define EXPORT extern "C" __declspec(dllexport) uint32_t

EXPORT ffxCreateContext(void** output, Header* header, const void*) {
    if (!output || !header || header->kind != CreateType) return Invalid;
    const auto& create = *reinterpret_cast<Create*>(header);
    if (!create.render.width || !create.render.height || !create.output.width || !create.output.height) return Invalid;
    std::lock_guard lock(ngx_mutex);
    if (!device_in_use) return NoProvider;
    auto* context = new (std::nothrow) Context{create.render, create.output, create.flags};
    if (!context) return MemoryError;
    const auto result = NVSDK_NGX_D3D12_AllocateParameters(&context->params);
    if (NVSDK_NGX_FAILED(result)) {
        delete context;
        return ngx_error("allocate parameters", result);
    }
    *output = context;
    return Ok;
}

EXPORT ffxDestroyContext(void** context, const void*) {
    if (!context || !*context) return Invalid;
    std::lock_guard lock(ngx_mutex);
    delete static_cast<Context*>(*context);
    *context = nullptr;
    return Ok;
}

EXPORT ffxConfigure(void**, Header* header) {
    // FidelityFX tuning constants have no DLSS counterpart.
    return header && header->kind == 0x10007 ? Ok : UnknownType;
}

EXPORT ffxQuery(void** context, Header* header) {
    if (!header) return Invalid;
    switch (header->kind) {
    case 4: {
        auto& q = *reinterpret_cast<Versions*>(header);
        if (!q.count || q.type != CreateType) return Invalid;
        std::lock_guard lock(ngx_mutex);
        uint32_t status;
        try { status = initialize(q.device); }
        catch (const std::bad_alloc&) { status = MemoryError; }
        const auto capacity = *q.count;
        *q.count = status == Ok ? 1 : 0;
        if (status == Ok && capacity > 0) {
            if (q.names) q.names[0] = ProviderName;
            if (q.ids) q.ids[0] = ProviderId;
        }
        return status;
    }
    case 6: {
        if (!context || !*context) return Invalid;
        auto& q = *reinterpret_cast<Provider*>(header);
        q.id = ProviderId;
        q.name = ProviderName;
        return Ok;
    }
    case 0x10002:
    case 0x10003: {
        constexpr float ratios[] = {1.0f, 1.5f, 1.7f, 2.0f, 3.0f};
        if (header->kind == 0x10002) {
            auto& q = *reinterpret_cast<Ratio*>(header);
            if (!q.ratio || q.quality >= 5) return Invalid;
            *q.ratio = ratios[q.quality];
        } else {
            auto& q = *reinterpret_cast<Resolution*>(header);
            if (!q.width || !q.height || q.quality >= 5) return Invalid;
            *q.width = static_cast<uint32_t>(q.output.width / static_cast<double>(ratios[q.quality]));
            *q.height = static_cast<uint32_t>(q.output.height / static_cast<double>(ratios[q.quality]));
        }
        return Ok;
    }
    case 0x10004: {
        const auto& q = *reinterpret_cast<PhaseCount*>(header);
        if (!q.phases || !q.render || !q.output) return Invalid;
        const auto ratio = static_cast<double>(q.output) / q.render;
        const auto phases = 8.0 * ratio * ratio;
        if (phases > std::numeric_limits<int32_t>::max()) return Invalid;
        *q.phases = phases < 1 ? 1 : static_cast<int32_t>(phases);
        return Ok;
    }
    case 0x10005: {
        const auto& q = *reinterpret_cast<Jitter*>(header);
        if (!q.x || !q.y || q.phases <= 0 || q.index < 0) return Invalid;
        const auto index = q.index % q.phases + 1;
        *q.x = halton(index, 2) - 0.5f;
        *q.y = halton(index, 3) - 0.5f;
        return Ok;
    }
    case 0x10008: {
        const auto& q = *reinterpret_cast<Memory*>(header);
        if (!q.bytes || !context || !*context) return Invalid;
        std::lock_guard lock(ngx_mutex);
        unsigned long long bytes = 0;
        unsigned level = 0, dev = 0;
        NVSDK_NGX_Parameter* capabilities = nullptr;
        if (NVSDK_NGX_SUCCEED(NVSDK_NGX_D3D12_GetCapabilityParameters(&capabilities))) {
            NGX_DLSS_GET_STATS_2(capabilities, &bytes, &level, &dev);
            NVSDK_NGX_D3D12_DestroyParameters(capabilities);
        }
        q.bytes[0] = bytes;
        q.bytes[1] = 0; // NGX manages these allocations internally.
        return Ok;
    }
    case 0x1000a: {
        auto& q = *reinterpret_cast<Requirements*>(header);
        q.required = 0x7; // color, depth and motion vectors
        q.optional = 0x8; // exposure, with auto-exposure if absent
        return Ok;
    }
    default:
        return UnknownType;
    }
}

EXPORT ffxDispatch(void** raw_context, Header* header) {
    if (!header || header->kind != DispatchType) return UnknownType;
    if (!raw_context || !*raw_context) return Invalid;
    const auto& d = *reinterpret_cast<Dispatch*>(header);
    if (!d.list || !d.color.resource || !d.depth.resource || !d.motion.resource || !d.output.resource ||
        !d.render.width || !d.render.height) return Invalid;

    std::lock_guard lock(ngx_mutex);
    auto& context = *static_cast<Context*>(*raw_context);
    const auto out = d.display.width && d.display.height ? d.display : context.max_output;
    constexpr uint32_t AutoExposure = 1 << 5;
    const bool auto_exposure = (context.flags & AutoExposure) || !d.exposure.resource;
    if (out.width > context.max_output.width || out.height > context.max_output.height ||
        d.render.width > context.max_render.width || d.render.height > context.max_render.height) return Invalid;
    Feature* feature = nullptr;
    for (auto& candidate : context.features) {
        if (candidate.output.width == out.width && candidate.output.height == out.height &&
            candidate.auto_exposure == auto_exposure) {
            feature = &candidate;
            break;
        }
    }
    const bool first = feature == nullptr;
    if (first) {
        try { context.features.emplace_back(out, auto_exposure); }
        catch (const std::bad_alloc&) { return MemoryError; }
        feature = &context.features.back();
        const auto result = create_feature(context, *feature, d.list);
        if (result != Ok) {
            context.features.pop_back();
            return result;
        }
        std::fprintf(stderr, "VIctory DLSS: created preset M, %ux%u to %ux%u\n",
                     d.render.width, d.render.height, out.width, out.height);
    }
    NVSDK_NGX_D3D12_DLSS_Eval_Params eval{};
    eval.Feature.pInColor = d.color.resource;
    eval.Feature.pInOutput = d.output.resource;
    eval.pInDepth = d.depth.resource;
    eval.pInMotionVectors = d.motion.resource;
    eval.pInExposureTexture = auto_exposure ? nullptr : d.exposure.resource;
    eval.InJitterOffsetX = d.jitter_x;
    eval.InJitterOffsetY = d.jitter_y;
    eval.InMVScaleX = d.motion_x;
    eval.InMVScaleY = d.motion_y;
    eval.InRenderSubrectDimensions = { d.render.width, d.render.height };
    eval.InPreExposure = d.pre_exposure;
    eval.InFrameTimeDeltaInMsec = d.frame_ms;
    eval.InReset = first || d.reset;
    const auto result = NGX_D3D12_EVALUATE_DLSS_EXT(d.list, feature->handle, context.params, &eval);
    return NVSDK_NGX_SUCCEED(result) ? Ok : ngx_error("evaluate preset M", result);
}

