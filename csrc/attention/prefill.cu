#include "attention/launch.h"
#include "attention/prefill_kernel.cuh"
namespace mxdsa::attention {
template void
run_dual_mxfp8_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(const MxFp8SparseAttnFwdParams&);
} // namespace mxdsa::attention
