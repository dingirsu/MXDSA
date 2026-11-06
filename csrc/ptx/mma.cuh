#pragma once

// Block-scaled MMA, completion commits and tcgen05 thread-sync fences.
#include "ptx/barrier.cuh"

namespace ptx {

DEVICE void mma_mxfp8(uint64_t a, uint64_t b, uint32_t c, uint32_t idesc, uint32_t sa, uint32_t sb,
                      bool accumulate) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; }" ::
            "r"(c),
        "l"(a), "l"(b), "r"(idesc), "r"(uint32_t(accumulate)), "r"(sa), "r"(sb));
}
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
