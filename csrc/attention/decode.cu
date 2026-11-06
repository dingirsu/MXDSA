#include "attention/launch.h"
#include "attention/decode_kernel.cuh"
namespace sm100::dual_mxfp8 {
template void run_dual_mxfp8_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(
    const SparseAttnDualMxfp8DecodeParams&);
}
