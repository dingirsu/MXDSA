#pragma once

// Transaction/named/cluster barriers and shared-memory proxy fences.
#include "ptx/common.cuh"

namespace ptx {

struct transac_bar_t {
    uint64_t bits;
    DEVICE void init(uint32_t n) const {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(cast_smem_ptr_to_uint(this)),
                     "r"(n)
                     : "memory");
    }
    DEVICE void wait(uint32_t phase) const {
        asm volatile(
            "{ .reg .pred p; wait_loop: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; @p bra done; bra wait_loop; done: }" ::
                "r"(cast_smem_ptr_to_uint(this)),
            "r"(phase), "r"(0x989680)
            : "memory");
    }
    DEVICE void arrive() const {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(cast_smem_ptr_to_uint(this))
                     : "memory");
    }
    DEVICE void arrive(uint32_t cta, uint32_t pred = 1) const {
        if (pred)
            asm volatile(
                "{ .reg .b32 remote; mapa.shared::cluster.u32 remote, %0, %1; mbarrier.arrive.shared::cluster.b64 _, [remote]; }" ::
                    "r"(cast_smem_ptr_to_uint(this)),
                "r"(cta)
                : "memory");
    }
    DEVICE void arrive_and_expect_tx(uint32_t n) const {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                         cast_smem_ptr_to_uint(this)),
                     "r"(n)
                     : "memory");
    }
};
struct NamedBarrier {
    DEVICE static void arrive_and_wait(uint32_t n, uint32_t id) {
        asm volatile("bar.sync %0, %1;" ::"r"(id + 8), "r"(n) : "memory");
    }
};
DEVICE void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
DEVICE void fence_view_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
DEVICE void barrier_cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed;"); }
DEVICE void barrier_cluster_wait_acquire() {
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
}

} // namespace ptx
