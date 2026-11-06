#pragma once

#include "common/params.h"
#include <type_traits>

namespace sm100::dual_mxfp8 {

template <SparseAttnFwdMode FWD_MODE, int D_QK>
void run_dual_mxfp8_phase1_kernel(
    const std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDualMxfp8DecodeParams,
                             MxFp8SparseAttnFwdParams>& params);

}
