#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <algorithm>
#define DEVICE __device__ __forceinline__
#define UNROLL _Pragma("unroll")
#define NO_UNROLL _Pragma("unroll 1")
namespace ptx {
using bf16 = __nv_bfloat16;
struct fp8_e4m3 {
    uint8_t storage;
};
struct fp8_e8m0 {
    uint8_t storage;
    fp8_e8m0() = default;
    DEVICE explicit fp8_e8m0(float x) {
        uint16_t out;
        asm volatile("cvt.rp.satfinite.ue8m0x2.f32 %0, 0.0, %1;" : "=h"(out) : "f"(x));
        storage = uint8_t(out);
    }
    DEVICE operator float() const {
        uint32_t b;
        float f;
        asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(b) : "h"(uint16_t(storage)));
        asm("prmt.b32 %0, %1, %2, %3;" : "=f"(f) : "r"(0), "r"(b), "r"(0x5410));
        return f;
    }
};
template <class T, size_t N, size_t A = 16> struct alignas(A) array_aligned {
    T elems[N];
    DEVICE T* data() { return elems; }
    DEVICE const T* data() const { return elems; }
    DEVICE T& operator[](size_t i) { return elems[i]; }
};
DEVICE uint32_t cast_smem_ptr_to_uint(const void* p) {
    return uint32_t(__cvta_generic_to_shared(p));
}
constexpr uint32_t Sm100MmaPeerBitMask = 0xfeffffff;
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
DEVICE void fence_view_async_tmem_load() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
DEVICE void fence_view_async_tmem_store() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
DEVICE void barrier_cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed;"); }
DEVICE void barrier_cluster_wait_acquire() {
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
}
template <int N> DEVICE void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}
template <int N> DEVICE void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}
DEVICE void prefetch_tma_descriptor(const void* p) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(p) : "memory");
}
DEVICE void tma_store_arrive() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> DEVICE void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
struct Allocator2Sm {
    DEVICE void allocate(int n, uint32_t* p) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                         cast_smem_ptr_to_uint(p)),
                     "r"(n));
    }
    DEVICE void free(uint32_t p, int n) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(p), "r"(n));
    }
    DEVICE void release_allocation_lock() {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
};
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
} // namespace ptx
#include "common/host.h"
#include "ptx/ptx_tmem.cuh"
#include "ptx/ptx_copy.cuh"
#include "ptx/ptx_extra.cuh"
namespace ku = ptx;
using ptx::bf16;
using ptx::fence_barrier_init;
using ptx::fence_view_async_shared;
using ptx::NamedBarrier;
using ptx::transac_bar_t;
