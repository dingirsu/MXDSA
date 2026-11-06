#include "attention/prefill_kernel.cuh"
namespace sm100::dual_mxfp8 {
template void
run_dual_mxfp8_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(const MxFp8SparseAttnFwdParams&);
}
