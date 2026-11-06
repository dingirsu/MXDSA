#include "ptx/common.cuh"
#include "split_kv/combine.h"

#include <math_constants.h>

#include "common/params.h"
#include "common/utils.h"
#include "common/host.h"

namespace mxdsa::split_kv {

template <typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    flash_fwd_mla_combine_kernel(__grid_constant__ const CombineParams params) {
    // grid_shape: [batch_size*s_q, 1, h_q/BLOCK_SIZE_M]
    // Each CTA gathers the activation of some heads from one batch, do scaling & accumulation, and
    // save the result
    static_assert(NUM_THREADS / 32 == BLOCK_SIZE_M); // The number of warps == block_size_m
    const int batch_s_q_idx = blockIdx.x;
    const int batch_idx = batch_s_q_idx / params.s_q;
    const int s_q_idx = batch_s_q_idx - batch_idx * params.s_q;
    const int h_block_idx = blockIdx.z;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;

    int num_valid_heads = std::min(BLOCK_SIZE_M, params.h_q - BLOCK_SIZE_M * h_block_idx);
    if (warp_idx >= num_valid_heads) {
        return;
    }

    const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
    const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
    const int my_num_splits = end_split_idx - start_split_idx;
    if (my_num_splits == 1) {
        return;
    }

    FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);

    float* gLseAccum = (float*)params.lse_accum + start_split_idx * params.stride_lse_accum_split +
                       s_q_idx * params.stride_lse_accum_s_q + h_block_idx * BLOCK_SIZE_M;
    float* gLse = (float*)params.lse + batch_idx * params.stride_lse_b +
                  s_q_idx * params.stride_lse_s_q + h_block_idx * BLOCK_SIZE_M;

    __shared__ float smem_buf[BLOCK_SIZE_M][MAX_SPLITS];

    // Wait for the previous kernel (the MLA kernel) to finish
    cudaGridDependencySynchronize();

    // Prefetch
    static_assert(HEAD_DIM_V % (32 * 4) == 0);
    constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / (32 * 4);
    float* oaccum_ptr = params.o_accum + start_split_idx * params.stride_o_accum_split +
                        s_q_idx * params.stride_o_accum_s_q +
                        (h_block_idx * BLOCK_SIZE_M + warp_idx) * params.stride_o_accum_h_q;
    float4 datas[ELEMS_PER_THREAD];
    UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        datas[i] =
            *(float4*)(oaccum_ptr + lane_idx * 4 +
                       i * 128); // NOTE We don't use __ldg here since it is incompatible with PDL
    }

    // Warp #i gathers LseAccum for seq #i
    {
        constexpr int NUM_LSE_PER_THREAD = mxdsa::ceil_div(MAX_SPLITS, 32);
        float local_lse[NUM_LSE_PER_THREAD];
        UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i * 32 + lane_idx;
            local_lse[i] = split_idx < my_num_splits
                               ? gLseAccum[(split_idx)*params.stride_lse_accum_split + (warp_idx)]
                               : -INFINITY;
        }

        float max_lse = -INFINITY;
        UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            max_lse = max(max_lse, local_lse[i]);
        UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse; // In case all local LSEs are -inf

        float sum_lse = 0;
        UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
        UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse =
            (sum_lse == 0.f || sum_lse == -INFINITY) ? INFINITY : log2f(sum_lse) + max_lse;
        if (lane_idx == 0)
            gLse[warp_idx] = global_lse / (float)M_LOG2E;

        if (params.attn_sink != nullptr) {
            int q_head_idx = h_block_idx * BLOCK_SIZE_M + warp_idx;
            float attn_sink = __ldg(params.attn_sink + q_head_idx);
            if (global_lse != INFINITY) {
                // If attn_sink is +inf, global_lse will be +inf and scale factors will be
                // exp2f(local_lse - inf) = 0 (since local_lse never becomes +inf) If attn_sink is
                // -inf, this has no effect on global_lse
                global_lse += log2f(1 + exp2f(attn_sink * CUDART_L2E_F - global_lse));
            } else {
                // We have no tokens to attend, so global lse should be attn_sink*CUDART_L2E_F (+inf
                // if it's -inf or +inf)
                global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink * CUDART_L2E_F;
            }
        }
        UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i * 32 + lane_idx;
            smem_buf[warp_idx][split_idx] = exp2f(local_lse[i] - global_lse);
        }
    }

    __syncwarp();

    // Warp #i accumulates activation for seq #i
    {
        float4 result[ELEMS_PER_THREAD];
        UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            result[i] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
        for (int split = 0; split < my_num_splits; ++split) {
            float lse_scale = smem_buf[warp_idx][split];
            // if (lse_scale != 0.f) {
            UNROLL
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                result[i].x += lse_scale * datas[i].x;
                result[i].y += lse_scale * datas[i].y;
                result[i].z += lse_scale * datas[i].z;
                result[i].w += lse_scale * datas[i].w;
                if (split != my_num_splits - 1) {
                    datas[i] = *(float4*)(oaccum_ptr + (split + 1) * params.stride_o_accum_split +
                                          lane_idx * 4 + i * 128);
                }
            }
            // }
        }

        const int h_q_idx = h_block_idx * BLOCK_SIZE_M + warp_idx;
        ElementT* o_ptr = (ElementT*)params.out + batch_idx * params.stride_o_b +
                          s_q_idx * params.stride_o_s_q + h_q_idx * params.stride_o_h_q;

        UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            float4 data = result[i];
            ElementT data_converted[4];
            data_converted[0] = (ElementT)(data.x);
            data_converted[1] = (ElementT)(data.y);
            data_converted[2] = (ElementT)(data.z);
            data_converted[3] = (ElementT)(data.w);
            static_assert(sizeof(ElementT) == 2);
            *(uint64_t*)(o_ptr + lane_idx * 4 + i * 128) = *(uint64_t*)data_converted;
        }
    }
}

template <typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    flash_fwd_mla_combine_boundary_kernel(__grid_constant__ const CombineParams params) {
    static_assert(NUM_THREADS / 32 == BLOCK_SIZE_M);
    const int group = blockIdx.x / params.s_q;
    const int s_q_idx = blockIdx.x - group * params.s_q;
    auto combine_one = [&](const DecodingSchedMeta& meta) {
        const int batch_idx = meta.end_req_idx;
        const int h_block_idx = blockIdx.z;
        const int warp_idx = threadIdx.x / 32;
        const int lane_idx = threadIdx.x % 32;

        int num_valid_heads = std::min(BLOCK_SIZE_M, params.h_q - BLOCK_SIZE_M * h_block_idx);
        if (warp_idx >= num_valid_heads) {
            return;
        }

        const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
        const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
        const int my_num_splits = end_split_idx - start_split_idx;
        if (my_num_splits == 1) {
            return;
        }

        FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);

        float* gLseAccum = (float*)params.lse_accum +
                           start_split_idx * params.stride_lse_accum_split +
                           s_q_idx * params.stride_lse_accum_s_q + h_block_idx * BLOCK_SIZE_M;
        float* gLse = (float*)params.lse + batch_idx * params.stride_lse_b +
                      s_q_idx * params.stride_lse_s_q + h_block_idx * BLOCK_SIZE_M;

        __shared__ float smem_buf[BLOCK_SIZE_M][MAX_SPLITS];

        // Wait for the previous kernel (the MLA kernel) to finish
        cudaGridDependencySynchronize();

        // Prefetch
        static_assert(HEAD_DIM_V % (32 * 4) == 0);
        constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / (32 * 4);
        float* oaccum_ptr = params.o_accum + start_split_idx * params.stride_o_accum_split +
                            s_q_idx * params.stride_o_accum_s_q +
                            (h_block_idx * BLOCK_SIZE_M + warp_idx) * params.stride_o_accum_h_q;
        float4 datas[ELEMS_PER_THREAD];
        UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            datas[i] = *(
                float4*)(oaccum_ptr + lane_idx * 4 +
                         i * 128); // NOTE We don't use __ldg here since it is incompatible with PDL
        }

        if (my_num_splits == 2) {
            // Boundary dispatch requires h_q=64 and a complete head block.
            // One warp coalesces LSE loads/stores and computes each head once.
            if (warp_idx == 0 && lane_idx < BLOCK_SIZE_M) {
                const int weight_head = lane_idx;
                const float lse0 = gLseAccum[(0) * params.stride_lse_accum_split + (weight_head)];
                const float lse1 = gLseAccum[(1) * params.stride_lse_accum_split + (weight_head)];
                float max_lse = max(lse0, lse1);
                max_lse = max_lse == -INFINITY ? 0.0f : max_lse;
                const float sum_lse = exp2f(lse0 - max_lse) + exp2f(lse1 - max_lse);
                float global_lse =
                    (sum_lse == 0.f || sum_lse == -INFINITY) ? INFINITY : log2f(sum_lse) + max_lse;
                gLse[weight_head] = global_lse / (float)M_LOG2E;
                if (params.attn_sink != nullptr) {
                    const float attn_sink =
                        __ldg(params.attn_sink + h_block_idx * BLOCK_SIZE_M + weight_head);
                    if (global_lse != INFINITY)
                        global_lse += log2f(1 + exp2f(attn_sink * CUDART_L2E_F - global_lse));
                    else
                        global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink * CUDART_L2E_F;
                }
                smem_buf[weight_head][0] = exp2f(lse0 - global_lse);
                smem_buf[weight_head][1] = exp2f(lse1 - global_lse);
            }
        } else {
            // Warp #i gathers LseAccum for seq #i
            {
                constexpr int NUM_LSE_PER_THREAD = mxdsa::ceil_div(MAX_SPLITS, 32);
                float local_lse[NUM_LSE_PER_THREAD];
                UNROLL
                for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
                    const int split_idx = i * 32 + lane_idx;
                    local_lse[i] =
                        split_idx < my_num_splits
                            ? gLseAccum[(split_idx)*params.stride_lse_accum_split + (warp_idx)]
                            : -INFINITY;
                }

                float max_lse = -INFINITY;
                UNROLL
                for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
                    max_lse = max(max_lse, local_lse[i]);
                UNROLL
                for (int offset = 16; offset >= 1; offset /= 2)
                    max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
                max_lse = max_lse == -INFINITY ? 0.0f : max_lse; // In case all local LSEs are -inf

                float sum_lse = 0;
                UNROLL
                for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
                    sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
                UNROLL
                for (int offset = 16; offset >= 1; offset /= 2)
                    sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

                float global_lse =
                    (sum_lse == 0.f || sum_lse == -INFINITY) ? INFINITY : log2f(sum_lse) + max_lse;
                if (lane_idx == 0)
                    gLse[warp_idx] = global_lse / (float)M_LOG2E;

                if (params.attn_sink != nullptr) {
                    int q_head_idx = h_block_idx * BLOCK_SIZE_M + warp_idx;
                    float attn_sink = __ldg(params.attn_sink + q_head_idx);
                    if (global_lse != INFINITY) {
                        // If attn_sink is +inf, global_lse will be +inf and scale factors will be
                        // exp2f(local_lse - inf) = 0 (since local_lse never becomes +inf) If
                        // attn_sink is -inf, this has no effect on global_lse
                        global_lse += log2f(1 + exp2f(attn_sink * CUDART_L2E_F - global_lse));
                    } else {
                        // We have no tokens to attend, so global lse should be
                        // attn_sink*CUDART_L2E_F (+inf if it's -inf or +inf)
                        global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink * CUDART_L2E_F;
                    }
                }
                UNROLL
                for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
                    const int split_idx = i * 32 + lane_idx;
                    smem_buf[warp_idx][split_idx] = exp2f(local_lse[i] - global_lse);
                }
            }
        }

        // The two-split path shares weights across warps. Other paths are warp-local.
        if (my_num_splits == 2)
            __syncthreads();
        else
            __syncwarp();

        // Warp #i accumulates activation for seq #i
        {
            float4 result[ELEMS_PER_THREAD];
            UNROLL
            for (int i = 0; i < ELEMS_PER_THREAD; ++i)
                result[i] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
            for (int split = 0; split < my_num_splits; ++split) {
                float lse_scale = smem_buf[warp_idx][split];
                // if (lse_scale != 0.f) {
                UNROLL
                for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                    result[i].x += lse_scale * datas[i].x;
                    result[i].y += lse_scale * datas[i].y;
                    result[i].z += lse_scale * datas[i].z;
                    result[i].w += lse_scale * datas[i].w;
                    if (split != my_num_splits - 1) {
                        datas[i] =
                            *(float4*)(oaccum_ptr + (split + 1) * params.stride_o_accum_split +
                                       lane_idx * 4 + i * 128);
                    }
                }
                // }
            }

            const int h_q_idx = h_block_idx * BLOCK_SIZE_M + warp_idx;
            ElementT* o_ptr = (ElementT*)params.out + batch_idx * params.stride_o_b +
                              s_q_idx * params.stride_o_s_q + h_q_idx * params.stride_o_h_q;

            UNROLL
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                float4 data = result[i];
                ElementT data_converted[4];
                data_converted[0] = (ElementT)(data.x);
                data_converted[1] = (ElementT)(data.y);
                data_converted[2] = (ElementT)(data.z);
                data_converted[3] = (ElementT)(data.w);
                static_assert(sizeof(ElementT) == 2);
                *(uint64_t*)(o_ptr + lane_idx * 4 + i * 128) = *(uint64_t*)data_converted;
            }
        }
    };
    const int part0 = group * 2, part1 = part0 + 1;
    const DecodingSchedMeta meta0 = params.tile_scheduler_metadata_ptr[part0];
    DecodingSchedMeta meta1 = {};
    if (part1 < params.num_sm_parts)
        meta1 = params.tile_scheduler_metadata_ptr[part1];
    const bool valid0 = meta0.begin_req_idx < params.b && meta0.is_last_req_splitted &&
                        !(meta0.begin_req_idx == meta0.end_req_idx && meta0.begin_split_idx != 0);
    const bool valid1 = part1 < params.num_sm_parts && meta1.begin_req_idx < params.b &&
                        meta1.is_last_req_splitted &&
                        !(meta1.begin_req_idx == meta1.end_req_idx && meta1.begin_split_idx != 0);
    if (!valid0 && !valid1)
        return;
    combine_one(valid0 ? meta0 : meta1);
    if (valid0 && valid1) {
        __syncthreads();
        combine_one(meta1);
    }
}

#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)                                               \
    [&] {                                                                                          \
        if (NUM_SPLITS <= 32) {                                                                    \
            constexpr static int NAME = 32;                                                        \
            return __VA_ARGS__();                                                                  \
        } else if (NUM_SPLITS <= 64) {                                                             \
            constexpr static int NAME = 64;                                                        \
            return __VA_ARGS__();                                                                  \
        } else if (NUM_SPLITS <= 96) {                                                             \
            constexpr static int NAME = 96;                                                        \
            return __VA_ARGS__();                                                                  \
        } else if (NUM_SPLITS <= 128) {                                                            \
            constexpr static int NAME = 128;                                                       \
            return __VA_ARGS__();                                                                  \
        } else if (NUM_SPLITS <= 160) {                                                            \
            constexpr static int NAME = 160;                                                       \
            return __VA_ARGS__();                                                                  \
        } else {                                                                                   \
            FLASH_ASSERT(false);                                                                   \
        }                                                                                          \
    }()

template <typename ElementT> void run_original_flash_mla_combine_kernel(CombineParams& params) {
    static constexpr int HEAD_DIM_V =
        512; // Since only this head dimension is supported by Flash MLA
    FLASH_ASSERT(params.d_v == HEAD_DIM_V);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, NUM_SPLITS, [&] {
        constexpr int BLOCK_SIZE_M = 8;
        constexpr int NUM_THREADS = BLOCK_SIZE_M * 32;
        constexpr size_t smem_size = BLOCK_SIZE_M * (NUM_SPLITS + 1) * sizeof(float);
        auto combine_kernel = &flash_fwd_mla_combine_kernel<ElementT, HEAD_DIM_V, BLOCK_SIZE_M,
                                                            NUM_SPLITS, NUM_THREADS>;
        CHECK_CUDA(cudaFuncSetAttribute(combine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
        // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t combine_kernel_config = {
            dim3(params.b * params.s_q, 1, mxdsa::ceil_div(params.h_q, BLOCK_SIZE_M)),
            dim3(NUM_THREADS, 1, 1),
            0,
            params.stream,
            attribute,
            1};
        CHECK_CUDA(cudaLaunchKernelEx(&combine_kernel_config, combine_kernel, params));
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename ElementT> void run_flash_mla_combine_kernel(CombineParams& params) {
    if (params.h_q != 64 || params.tile_scheduler_metadata_ptr == nullptr ||
        params.b <= params.num_sm_parts) {
        run_original_flash_mla_combine_kernel<ElementT>(params);
        return;
    }
    static constexpr int HEAD_DIM_V =
        512; // Since only this head dimension is supported by Flash MLA
    FLASH_ASSERT(params.d_v == HEAD_DIM_V);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, NUM_SPLITS, [&] {
        constexpr int BLOCK_SIZE_M = 8;
        constexpr int NUM_THREADS = BLOCK_SIZE_M * 32;
        constexpr size_t smem_size = BLOCK_SIZE_M * (NUM_SPLITS + 1) * sizeof(float);
        auto combine_kernel =
            &flash_fwd_mla_combine_boundary_kernel<ElementT, HEAD_DIM_V, BLOCK_SIZE_M, NUM_SPLITS,
                                                   NUM_THREADS>;
        CHECK_CUDA(cudaFuncSetAttribute(combine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
        // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t combine_kernel_config = {
            dim3(mxdsa::ceil_div(params.num_sm_parts, 2) * params.s_q, 1,
                 mxdsa::ceil_div(params.h_q, BLOCK_SIZE_M)),
            dim3(NUM_THREADS, 1, 1),
            0,
            params.stream,
            attribute,
            1};
        CHECK_CUDA(cudaLaunchKernelEx(&combine_kernel_config, combine_kernel, params));
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template void run_flash_mla_combine_kernel<__nv_bfloat16>(CombineParams& params);

#ifndef FLASH_MLA_DISABLE_FP16
template void run_flash_mla_combine_kernel<__half>(CombineParams& params);
#endif

} // namespace mxdsa::split_kv
