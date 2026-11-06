#pragma once

#include "common/params.h"
#include <type_traits>

namespace mxdsa::attention {

template <SparseAttnFwdMode FWD_MODE, int D_QK, int PREFILL_B_TOPK> struct KernelTemplate;

template <SparseAttnFwdMode FWD_MODE, int D_QK>
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
        KernelTemplate<FWD_MODE, D_QK, 64>::run(params);
    }
}

// The CUDA translation units provide these instantiations for host-only callers.
extern template void run_dual_mxfp8_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(
    const MxFp8SparseAttnFwdParams&);
extern template void run_dual_mxfp8_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(
    const SparseAttnDualMxfp8DecodeParams&);

} // namespace mxdsa::attention
