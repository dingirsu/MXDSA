#pragma once

// Cluster launch control queries and responses.
#include "ptx/barrier.cuh"

namespace ptx {

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
#undef EMIT_ASM
    return result;
}

} // namespace ptx
