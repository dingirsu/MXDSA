#pragma once

// Warp/cluster identity, lane election and register allocation.
#include "ptx/common.cuh"

namespace ptx {

DEVICE int canonical_warp_idx_sync() { return __shfl_sync(0xffffffff, threadIdx.x / 32, 0); }
DEVICE int canonical_warp_idx() { return threadIdx.x / 32; }
DEVICE int canonical_warp_group_idx() { return __shfl_sync(0xffffffff, threadIdx.x / 128, 0); }
DEVICE uint32_t elect_one_sync() {
    uint32_t pred = 0, lane = 0;
    asm volatile(
        "{ .reg .b32 rx; .reg .pred px; elect.sync rx|px, %2; @px mov.s32 %1, 1; mov.s32 %0, rx; }"
        : "+r"(lane), "+r"(pred)
        : "r"(0xffffffff));
    return pred;
}
DEVICE dim3 block_id_in_cluster() {
    dim3 d;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(d.x));
    asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(d.y));
    asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(d.z));
    return d;
}
template <int N> DEVICE void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}
template <int N> DEVICE void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}

} // namespace ptx
