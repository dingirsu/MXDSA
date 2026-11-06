#pragma once
#include "attention/launch.h"

#include <math_constants.h>

#include "common/params.h"
#include "common/utils.h"
#include "attention/common_subroutine.h"

#include "attention/config.h"
#include "common/host.h"
#include "ptx/barrier.cuh"
#include "ptx/clc.cuh"
#include "ptx/math.cuh"
#include "ptx/memory.cuh"
#include "ptx/mma.cuh"
#include "ptx/tma.cuh"
#include "ptx/tmem.cuh"
#include "ptx/types.cuh"
#include "ptx/warp.cuh"

namespace sm100::dual_mxfp8 {

using FwdMode = SparseAttnFwdMode;

DEVICE void copy_k_scale_smem_to_tmem(void* ptr, uint32_t col) {
    uint64_t desc = 0x0000400800000000ull | (ptx::cast_smem_ptr_to_uint(ptr) >> 4);
    ptx::SM100_UTCCP_2x64dp128bitlw0123_2cta::copy(desc, col);
}

DEVICE void store_s_scale_to_tmem(uint32_t tmem_col, int warp_in_warpgroup, uint8_t scale0_bits,
                                  uint8_t scale1_bits) {
    // S is quantized independently for each softmax row.  The O MMA consumes
    // four K=32 SFA bytes from one 32-bit TMEM word, so repeat this row's
    // UE8M0 factor across the word and keep the duplicated 2x2 DP layout.
    const uint32_t packed_scale = uint32_t(scale0_bits) | (uint32_t(scale1_bits) << 8) |
                                  (uint32_t(scale0_bits) << 16) | (uint32_t(scale1_bits) << 24);
    const uint32_t warp_dp_addr = tmem_col + warp_in_warpgroup * 32 * 65536;
    UNROLL
    for (int row_group = 0; row_group < 4; ++row_group) {
        ptx::SM100_TMEM_STORE_32dp32b1x::copy(packed_scale, warp_dp_addr + row_group);
    }
}

template <FwdMode FWD_MODE, int D_QK>
void run_dual_mxfp8_phase1_kernel(
    const std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDualMxfp8DecodeParams,
                             MxFp8SparseAttnFwdParams>& params) {
    if constexpr (!is_decode_v<FWD_MODE>) {
        if (params.topk >= 512) {
            KernelTemplate<FWD_MODE, D_QK, 128>::run(params);
        } else {
            KernelTemplate<FWD_MODE, D_QK, 64>::run(params);
        }
    } else {
        KernelTemplate<FWD_MODE, D_QK>::run(params);
    }
}

} // namespace sm100::dual_mxfp8
