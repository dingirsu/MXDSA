#pragma once

#include "common/params.h"

namespace mxdsa::split_kv {

template <typename ElementT> void run_flash_mla_combine_kernel(CombineParams& params);

} // namespace mxdsa::split_kv
