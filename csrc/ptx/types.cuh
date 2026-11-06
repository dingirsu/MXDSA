#pragma once

// Scalar types and their storage conversions.
#include "ptx/common.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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

} // namespace ptx
