#pragma once

// Shared CUDA declarations, annotations, storage and address helpers.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#define DEVICE __device__ __forceinline__
#define UNROLL _Pragma("unroll")
#define NO_UNROLL _Pragma("unroll 1")

namespace ptx {

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

} // namespace ptx
