#pragma once

// Tensor/bulk asynchronous copies, gather, prefetch and store completion.
#include "ptx/barrier.cuh"
#include "ptx/memory.cuh"

namespace ptx {

DEVICE void prefetch_tma_descriptor(const void* p) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(p) : "memory");
}
DEVICE void tma_store_arrive() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> DEVICE void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
struct SM90_TMA_STORE_5D {
    DEVICE static void copy(void const* desc_ptr, void const* smem_ptr, int32_t const& crd0,
                            int32_t const& crd1, int32_t const& crd2, int32_t const& crd3,
                            int32_t const& crd4) {
        uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);

        asm volatile(
            "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3),
              "r"(crd4)
            : "memory");
    }

    DEVICE static void copy(void const* desc_ptr, void const* smem_ptr, int32_t const& crd0,
                            int32_t const& crd1, int32_t const& crd2, int32_t const& crd3,
                            int32_t const& crd4, uint64_t cache_hint) {
        uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group.L2::cache_hint "
            "[%0, {%2, %3, %4, %5, %6}], [%1], %7;" ::"l"(gmem_int_desc),
            "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4),
            "l"(cache_hint)
            : "memory");
    }
};
struct SM90_BULK_COPY_G2S {
    DEVICE static void copy(void const* gmem_ptr, uint64_t* mbar_ptr, void* smem_ptr,
                            int32_t load_bytes) {
        uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :
            : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes), "r"(smem_int_mbar)
            : "memory");
    }

    struct PREFETCH {
        DEVICE static void copy(void const* gmem_ptr, int32_t load_bytes) {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
                         :
                         : "l"(gmem_ptr), "r"(load_bytes)
                         : "memory");
        }
    };
};
struct SM100_TMA_2SM_LOAD_5D_NOSPLIT {
    DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
                            void* smem_ptr, int32_t const& crd0, int32_t const& crd1,
                            int32_t const& crd2, int32_t const& crd3, int32_t const& crd4) {
        uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
        // Executed by both CTAs. Set peer bit to 0 so that the
        // transaction bytes will update CTA0's barrier.
        uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask;
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
            :
            : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1),
              "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
            : "memory");
    }
};
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
// Route gather completion to CTA1 in a two-CTA group.
DEVICE void tma_gather4_cta1(const void* desc_ptr, ptx::transac_bar_t& mbar, void* dst, int col,
                             int4 rows, int64_t hint) {
    uint32_t dst_addr = ptx::cast_smem_ptr_to_uint(dst);
    uint32_t bar_addr = (ptx::cast_smem_ptr_to_uint(&mbar) & ptx::Sm100MmaPeerBitMask) |
                        uint32_t(~ptx::Sm100MmaPeerBitMask);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::
            "r"(dst_addr),
        "l"(desc_ptr), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w), "r"(bar_addr),
        "l"(hint)
        : "memory");
}

} // namespace ptx
