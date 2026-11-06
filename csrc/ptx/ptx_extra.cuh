#pragma once

#include "ptx/ptx.cuh"

namespace ptx {
// tma gather4
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
// Please pay attention that the coordinates of TMA gather4 are int32, which may lead to overflow
// under some scenarios
DEVICE
void tma_gather4(const void* desc_ptr, transac_bar_t& mbar_ptr, void* smem_ptr, int col_idx,
                 int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = ptx::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = ptx::cast_smem_ptr_to_uint(&mbar_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx), "r"(row_idxs.x), "r"(row_idxs.y),
          "r"(row_idxs.z), "r"(row_idxs.w), "r"(mbar_addr), "l"(cache_hint)
        : "memory");
}

// tma gather4 prefetch
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)
// Please pay attention that the coordinates of TMA gather4 are int32, which may lead to overflow
// under some scenarios
DEVICE
void tma_gather4_prefetch(const void* desc_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [%0, {%1, %2, %3, %4, %5}], %6;\n"
        :
        : "l"(desc_ptr), "r"(col_idx), "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z),
          "r"(row_idxs.w), "l"(cache_hint));
}

// tma gather4 with cta_group::2, allowing for synchronization across CTAs within a pair of CTAs
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
template <bool USE_CTA0_MBAR = false>
DEVICE void tma_gather4_cta_group_2(const void* desc_ptr, transac_bar_t& mbar_ptr, void* smem_ptr,
                                    int col_idx, int4 row_idxs, int64_t cache_hint) {
    uint32_t smem_addr = ptx::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = ptx::cast_smem_ptr_to_uint(&mbar_ptr);
    if constexpr (USE_CTA0_MBAR) {
        mbar_addr &= ptx::Sm100MmaPeerBitMask;
    }
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx), "r"(row_idxs.x), "r"(row_idxs.y),
          "r"(row_idxs.z), "r"(row_idxs.w), "r"(mbar_addr), "l"(cache_hint)
        : "memory");
}

DEVICE
__nv_bfloat162 bfloat162_add(const __nv_bfloat162& a, const __nv_bfloat162& b) {
    __nv_bfloat162 c;
    asm volatile("add.bf16x2 %0, %1, %2;\n"
                 : "=r"(reinterpret_cast<uint32_t&>(c))
                 : "r"(reinterpret_cast<uint32_t const&>(a)),
                   "r"(reinterpret_cast<uint32_t const&>(b)));
    return c;
}

DEVICE
__nv_bfloat162 bfloat162_mul(const __nv_bfloat162& a, const __nv_bfloat162& b) {
    __nv_bfloat162 c;
    asm volatile("mul.bf16x2 %0, %1, %2;\n"
                 : "=r"(reinterpret_cast<uint32_t&>(c))
                 : "r"(reinterpret_cast<uint32_t const&>(a)),
                   "r"(reinterpret_cast<uint32_t const&>(b)));
    return c;
}

// Vectorized addition for float32
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add)
DEVICE
float2 float2_add(const float2& a, const float2& b) {
    float2 c;
    asm volatile("add.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)),
                   "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

// Vectorized multiplication for float32
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul)
DEVICE
float2 float2_mul(const float2& a, const float2& b) {
    float2 c;
    asm volatile("mul.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)),
                   "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

// Vectorized fused addition-multiplication for float32
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)
DEVICE
float2 float2_fma(const float2& a, const float2& b, const float2& c) {
    // return a*b+c
    float2 d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(d))
                 : "l"(reinterpret_cast<uint64_t const&>(a)),
                   "l"(reinterpret_cast<uint64_t const&>(b)),
                   "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

// Vectorized negation for foat32
DEVICE
float2 float2_neg(const float2& a) {
    float2 t = {-1.0f, -1.0f};
    return float2_mul(a, t);
}

DEVICE
uint16_t float2_to_e4m3x2_bits(float2 const& value) {
    uint16_t packed;

    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
                 : "=h"(packed)
                 : "f"(value.y), "f"(value.x));

    return packed;
}

// st.bulk
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)
DEVICE
void st_bulk(void* dst_ptr, int64_t size) {
    uint32_t dst_addr = ptx::cast_smem_ptr_to_uint(dst_ptr);
    asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;\n" : : "r"(dst_addr), "l"(size) : "memory");
}

struct alignas(16) CLCResponseObj {
    // An opaque 16B value
    char opaque[16];
};

struct CLCResult {
    int is_valid;
    int x, y, z;
};

// Issue a CLC try_cancel query
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
DEVICE
void issue_clc_query(transac_bar_t& bar, CLCResponseObj& response_obj) {
    uint32_t response_addr = ptx::cast_smem_ptr_to_uint(response_obj.opaque);
    uint32_t mbarrier_addr = ptx::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n"
        :
        : "r"(response_addr), "r"(mbarrier_addr));
}

// Issue a CLC try_cancel query with .multicast::cluster::all
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
DEVICE
void issue_clc_query_multicast_cluster_all(transac_bar_t& bar, CLCResponseObj& response_obj) {
    uint32_t response_addr = ptx::cast_smem_ptr_to_uint(response_obj.opaque);
    uint32_t mbarrier_addr = ptx::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n"
        :
        : "r"(response_addr), "r"(mbarrier_addr));
}

// Get the result of a CLC query
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel)
// In this function, we separate get_first_ctaid::x/y/z and hope PTXAS's dead code elimination can
// remove unnecessary instructions
template <bool USE_LD_ACQUIRE>
DEVICE CLCResult get_clc_query_response(CLCResponseObj& response_obj) {
    uint32_t response_addr = ptx::cast_smem_ptr_to_uint(&response_obj);
    CLCResult result;
#define EMIT_ASM(LD_MODIFIER)                                                                      \
    asm volatile(                                                                                  \
        "{\n"                                                                                      \
        ".reg .pred p1;\n\t"                                                                       \
        ".reg .b128 clc_result;\n\t"                                                               \
        "ld" LD_MODIFIER ".shared.b128 clc_result, [%4];\n\t"                                      \
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"              \
        "selp.u32 %3, 1, 0, p1;\n\t"                                                               \
        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_result;\n\t"    \
        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_result;\n\t"    \
        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_result;\n\t"    \
        "}\n"                                                                                      \
        : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.is_valid)                    \
        : "r"(response_addr)                                                                       \
        : "memory");
    if constexpr (USE_LD_ACQUIRE) {
        EMIT_ASM(".acquire.cta");
    } else {
        EMIT_ASM("");
    }
    return result;
}

// LDG.256 or LDG.256 with non-coherent cache
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
// We use macro instead of function here, since we need a multi-level recursive dispatch based on
// template parameters if using function NC_STR should be either "" or ".nc" L1_CACHE_HINT_STR
// should be either "evict_first", "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_CACHE_HINT_STR should be either "evict_first", "evict_normal", or "evict_last"
// L2_PREFETCH_SIZE_STR should be either "64B", "128B", or "256B"
#define KU_LDG_256(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR,              \
                   L2_PREFETCH_SIZE_STR)                                                           \
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
#define KU_STG_256(global_addr, src, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR)                         \
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

} // namespace ptx

namespace ptx {

// tcgen05.commit.cta_group::1
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
DEVICE
void umma_arrive_noelect(transac_bar_t& bar) {
    uint32_t bar_intptr = ptx::cast_smem_ptr_to_uint(&bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                 :
                 : "r"(bar_intptr));
}

// tcgen05.commit.cta_group::1, with multicast
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
DEVICE
void umma_arrive_multicast_noelect(transac_bar_t& bar, uint16_t cta_mask) {
    uint32_t bar_intptr = ptx::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :
        : "r"(bar_intptr), "h"(cta_mask));
}

// tcgen05.commit.cta_group::2
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
DEVICE
void umma_arrive_2x1SM_noelect(transac_bar_t& bar) {
    uint32_t bar_intptr = ptx::cast_smem_ptr_to_uint(&bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                 :
                 : "r"(bar_intptr));
}

// tcgen05.commit.cta_group::2, with multicast
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
DEVICE
void umma_arrive_multicast_2x1SM_noelect(transac_bar_t& bar, uint16_t cta_mask) {
    uint32_t bar_intptr = ptx::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        :
        : "r"(bar_intptr), "h"(cta_mask));
}

// tcgen05.fence::before_thread_sync
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

// tcgen05.fence::after_thread_sync
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

} // namespace ptx

#define KU_LDG_128(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_PREFETCH_SIZE_STR)           \
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
