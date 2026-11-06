#pragma once

// Packed arithmetic and explicit numeric conversion instructions.
#include "ptx/types.cuh"

namespace ptx {

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

} // namespace ptx
