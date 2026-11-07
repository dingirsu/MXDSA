#pragma once

// Global/shared loads, stores, prefetch and cache policies.
#include "ptx/common.cuh"

namespace ptx {

// Read-only global loads with default cache policy and no extra prefetch hint.
// Keep scalar/two-word accesses narrow for index tails and 8-byte scale payloads.
DEVICE int ld_global_nc(const int* p) {
    int value;
    asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(value) : "l"(p));
    return value;
}

DEVICE float ld_global_nc(const float* p) {
    float value;
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(value) : "l"(p));
    return value;
}

DEVICE int2 ld_global_nc(const int2* p) {
    int2 value;
    asm volatile("ld.global.nc.v2.s32 {%0, %1}, [%2];"
                 : "=r"(value.x), "=r"(value.y)
                 : "l"(p));
    return value;
}

DEVICE uint2 ld_global_nc(const uint2* p) {
    uint2 value;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
                 : "=r"(value.x), "=r"(value.y)
                 : "l"(p));
    return value;
}

DEVICE void ld_shared_v2_u32(const void* p, int (&values)[2]) {
    const uint32_t addr = cast_smem_ptr_to_uint(p);
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                 : "=r"(values[0]), "=r"(values[1])
                 : "r"(addr)
                 : "memory");
}

DEVICE uint64_t ld_shared_acquire_u64(const void* p) {
    const uint32_t addr = cast_smem_ptr_to_uint(p);
    uint64_t value;
    asm volatile("ld.acquire.cta.shared.b64 %0, [%1];" : "=l"(value) : "r"(addr) : "memory");
    return value;
}

DEVICE void prefetch_global_l1(const void* p) { asm volatile("prefetch.global.L1 [%0];" ::"l"(p)); }

DEVICE void st_shared(void* p, __int128_t v) {
    asm volatile("st.shared.b128 [%0], %1;" ::"l"(__cvta_generic_to_shared(p)), "q"(v));
}
DEVICE void st_shared(void* p, float4 v) { st_shared(p, *reinterpret_cast<__int128_t*>(&v)); }
enum class CacheHint { EVICT_FIRST, EVICT_NORMAL, EVICT_LAST };
template <CacheHint C> DEVICE constexpr int64_t create_simple_cache_policy() {
    return C == CacheHint::EVICT_FIRST    ? 0x12f0000000000000
           : C == CacheHint::EVICT_NORMAL ? 0x1000000000000000
                                          : 0x14f0000000000000;
}
// st.bulk
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)
DEVICE
void st_bulk(void* dst_ptr, int64_t size) {
    uint32_t dst_addr = ptx::cast_smem_ptr_to_uint(dst_ptr);
    asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;\n" : : "r"(dst_addr), "l"(size) : "memory");
}
// LDG.256 or LDG.256 with non-coherent cache
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
// We use macro instead of function here, since we need a multi-level recursive dispatch based on
// template parameters if using function NC_STR should be either "" or ".nc" L1_CACHE_HINT_STR
// should be either "evict_first", "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_CACHE_HINT_STR should be either "evict_first", "evict_normal", or "evict_last"
// L2_PREFETCH_SIZE_STR should be either "64B", "128B", or "256B"
#define LDG_256(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR,                 \
                L2_PREFETCH_SIZE_STR)                                                              \
    {                                                                                              \
        static_assert(std::is_pointer_v<decltype(global_addr)> ||                                  \
                          std::is_array_v<decltype(global_addr)>,                                  \
                      "`global_addr` must be a pointer");                                          \
        static_assert(std::is_pointer_v<decltype(result)> || std::is_array_v<decltype(result)>,    \
                      "`result` must be a pointer");                                               \
        uint64_t* result_as_uint64_ptr = (uint64_t*)(result);                                      \
        asm volatile("ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR        \
                     ".L2::" L2_PREFETCH_SIZE_STR ".v4.u64 {%0, %1, %2, %3}, [%4];\n"              \
                     : "=l"(result_as_uint64_ptr[0]), "=l"(result_as_uint64_ptr[1]),               \
                       "=l"(result_as_uint64_ptr[2]), "=l"(result_as_uint64_ptr[3])                \
                     : "l"(global_addr));                                                          \
    }

// STG.256
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st)
// L1_CACHE_HINT_STR should be either "evict_first", "evict_normal", "evict_last",
// "evict_unchanged", or "no_allocate" L2_CACHE_HINT_STR should be either "evict_first",
// "evict_normal", or "evict_last"
#define STG_256(global_addr, src, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR)                            \
    {                                                                                              \
        static_assert(std::is_pointer_v<decltype(global_addr)> ||                                  \
                          std::is_array_v<decltype(global_addr)>,                                  \
                      "`global_addr` must be a pointer");                                          \
        static_assert(std::is_pointer_v<decltype(src)> || std::is_array_v<decltype(src)>,          \
                      "`src` must be a pointer");                                                  \
        uint64_t const* src_as_uint64_ptr = (uint64_t const*)(src);                                \
        asm volatile("st.global.L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR                  \
                     ".v4.u64 [%0], {%1, %2, %3, %4};\n"                                           \
                     :                                                                             \
                     : "l"(global_addr), "l"(src_as_uint64_ptr[0]), "l"(src_as_uint64_ptr[1]),     \
                       "l"(src_as_uint64_ptr[2]), "l"(src_as_uint64_ptr[3]));                      \
    }
#define LDG_128(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_PREFETCH_SIZE_STR)              \
    {                                                                                              \
        static_assert(std::is_pointer_v<decltype(global_addr)> ||                                  \
                          std::is_array_v<decltype(global_addr)>,                                  \
                      "`global_addr` must be a pointer");                                          \
        static_assert(std::is_pointer_v<decltype(result)> || std::is_array_v<decltype(result)>,    \
                      "`result` must be a pointer");                                               \
        uint64_t* result_as_uint64_ptr = (uint64_t*)(result);                                      \
        asm volatile("ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR ".L2::" L2_PREFETCH_SIZE_STR     \
                     ".v2.u64 {%0, %1}, [%2];\n"                                                   \
                     : "=l"(result_as_uint64_ptr[0]), "=l"(result_as_uint64_ptr[1])                \
                     : "l"(global_addr));                                                          \
    }

} // namespace ptx
