#pragma once

#include <math_constants.h>

#include "attention/common.cuh"
#include "attention/config.h"
#include "common/host.h"
#include "common/params.h"
#include "common/utils.h"
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

template <SparseAttnFwdMode FWD_MODE, int D_QK, int PREFILL_B_TOPK>
__device__ void KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>::sparse_attn_fwd_kernel_devfunc(
    const ArgT& params, const TmaParams& tma_params)
    requires(IS_DECODE)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int warp_idx = ptx::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = ptx::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int cta_idx = ptx::block_id_in_cluster().x;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan& smem = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && ptx::elect_one_sync()) {
        ptx::prefetch_tma_descriptor(&tma_params.tensor_map_q);
        ptx::prefetch_tma_descriptor(&tma_params.tensor_map_o);
        ptx::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
    } else if (warp_idx == 1 && ptx::elect_one_sync()) {
        UNROLL
        for (int slot = 0; slot < 2; ++slot) {
            smem.bar_sQ_full[slot].init(1);
            smem.bar_raw_indices_full[slot].init(1);
            smem.bar_Q_scale_ready[slot].init(2);
            smem.bar_tQ_empty[slot].init(1);
            smem.bar_tQ_full[slot].init(1);
        }
        smem.bar_tOut_full.init(1);
        smem.bar_tOut_empty.init(256);
        smem.bar_P_empty.init(256);
        UNROLL
        for (int i = 0; i < 2; ++i) {
            smem.bar_S_empty[i].init(1);
            smem.bar_S_O_full[i].init(256);
        }
        smem.bar_li_full.init(128);
        smem.bar_li_empty.init(128);

        ptx::fence_barrier_init();
    } else if (warp_idx == 3 && ptx::elect_one_sync()) {
        UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
            smem.bar_V_full[i].init(1);
            smem.bar_K_full[i].init(1);
            smem.bar_V_empty[i].init(1);
            // CTA0 issues a 2-CTA UTCCP, so its barrier must collect warp10
            // and warp11 from both CTAs before either SMEM source is read.
            smem.bar_K_scale_copy_ready[i].init(4);
            // Publish V token scales directly to softmax: QK's asynchronous
            // completion is not a replacement for this shared-memory handoff.
            // All 128 softmax threads release their reads before stage reuse.
            smem.bar_v_scale_full[i].init(64);
            smem.bar_v_scale_empty[i].init(128);
            // The MMA completion releases this stage's scale source and is
            // also consumed by the softmax workers as QK-ready.
            smem.bar_QK_done[i].init(1);
        }
        UNROLL
        for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
            smem.bar_valid_coord_scales_full[i].init(32);
            smem.bar_valid_coord_scales_empty[i].init(128 + 4 + 64);
        }
        ptx::fence_barrier_init();
    }

    ptx::barrier_cluster_arrive_relaxed();
    ptx::barrier_cluster_wait_acquire();

    if (warp_idx == 2) {
        ptx::Allocator2Sm().allocate(512, smem.tmem_start_addr.data());
        __syncwarp();
        ptx::Allocator2Sm().release_allocation_lock();
    }

    ptx::barrier_cluster_arrive_relaxed();
    ptx::barrier_cluster_wait_acquire();

    struct OuterloopArgs {
        bool outer_loop_phase;
        int batch_idx, s_q_idx;
        int start_block_idx, end_block_idx;
        int topk_length;

        int extra_topk_length, num_orig_kv_blocks;
        bool is_no_split;
        int n_split_idx;
    };

    int decode_q_phase = 0;
    auto run_outer_loop = [&](auto loop_body) -> bool {
        int outer_loop_phase = false;
        int iteration = 0;
        int s_q_idx = blockIdx.x / 2; // adjacent-token pair index
        DecodingSchedMeta sched_meta;
        KU_LDG_256(params.tile_scheduler_metadata_ptr + blockIdx.y, &sched_meta, ".nc",
                   "no_allocate", "evict_normal", "256B");
        if (sched_meta.begin_req_idx >= params.b) {
            return 0;
        }

#pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx;
             ++batch_idx) {
            int topk_length =
                params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ptx::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
            int extra_topk_length = params.extra_topk_length
                                        ? __ldg(params.extra_topk_length + batch_idx)
                                        : params.extra_topk;
            int total_topk_padded = orig_topk_padded + ptx::ceil(extra_topk_length, (int)B_TOPK);
            int start_block_idx =
                batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
            int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx
                                                                    : total_topk_padded / B_TOPK;
            bool is_split =
                batch_idx == sched_meta.begin_req_idx
                    ? sched_meta.is_first_req_splitted
                    : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted
                                                           : false);
            int n_split_idx =
                batch_idx == sched_meta.begin_req_idx
                    ? (__ldg(params.num_splits_ptr + batch_idx) + sched_meta.begin_split_idx)
                    : __ldg(params.num_splits_ptr + batch_idx);

            OuterloopArgs args = {(bool)outer_loop_phase,
                                  batch_idx,
                                  s_q_idx,
                                  start_block_idx,
                                  end_block_idx,
                                  topk_length,

                                  extra_topk_length,
                                  orig_topk_padded / B_TOPK,
                                  !is_split,
                                  n_split_idx};

            decode_q_phase = (iteration / 2) & 1;
            loop_body(args);
            ++iteration;
            outer_loop_phase ^= 1;
        }
        return outer_loop_phase;
    };

    if (warpgroup_idx == 0) {
        ptx::warpgroup_reg_alloc<176>();
        const uint32_t warp_dp_addr = tmem_cols::V_scale + warp_idx * 32 * 65536;
        uint32_t initial_w_words[8];
        UNROLL
        for (int col = 0; col < 8; ++col)
            initial_w_words[col] = warp_idx < 2 ? params.w1[col] : params.w2[col];
        ptx::tmem_st_32dp32bNx<8>(warp_dp_addr, initial_w_words);
        ptx::fence_view_async_tmem_store();

        auto perform_o_copy_out = [&](const OuterloopArgs& args, bool is_last_o) {
            ptx::bf16* o_storage =
                reinterpret_cast<ptx::bf16*>(smem.Q[args.outer_loop_phase].data());
            ptx::bf16* sO_addrs[B_EPI / 8];
            UNROLL
            for (int i = 0; i < B_EPI / 8; ++i) {

                sO_addrs[i] =
                    sm100::k_major_ptr<64, 128>(o_storage, idx_in_warpgroup % 64,
                                              (idx_in_warpgroup / 64) * (O_STAGE_D / 2) + i * 8);
            }

            float* sO_accum_addrs[B_EPI_SPLITKV / 4];
            UNROLL
            for (int i = 0; i < B_EPI_SPLITKV / 4; ++i) {
                sO_accum_addrs[i] =
                    sm100::k_major_ptr<64, 128>((float*)o_storage, idx_in_warpgroup % 64, i * 4) +
                    (idx_in_warpgroup >= 64 ? (H_Q / 2) * B_EPI_SPLITKV : 0);
            }

            smem.bar_li_full.wait(args.outer_loop_phase);
            float output_scale;
            const int head_idx = idx_in_warpgroup % 64;
            const float final_li =
                smem.rowwise_li_buf[head_idx] + smem.rowwise_li_buf[head_idx + 64];
            const float final_mi = smem.final_mi[head_idx];
            smem.bar_li_empty.arrive();
            const float attn_sink =
                params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + head_idx);
            if (args.is_no_split)
                output_scale =
                    __fdividef(1.0f, final_li + exp2f(fmaf(attn_sink, CUDART_L2E_F, -final_mi)));
            else
                output_scale = __fdividef(1.0f, final_li);
            output_scale = final_li == 0.0f ? 0.0f : output_scale;
            if (idx_in_warpgroup < H_Q / 2) {
                const int head_idx = idx_in_warpgroup;
                const int q_token_idx = 2 * args.s_q_idx + cta_idx;
                float cur_lse = fmaf(final_mi, CUDART_LN2_F, logf(final_li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                if (q_token_idx < params.s_q) {
                    if (args.is_no_split) {
                        params.lse[args.batch_idx * params.stride_lse_b +
                                   q_token_idx * params.stride_lse_s_q + head_idx] = cur_lse;
                    } else {
                        const float cur_lse_2base = log2f(final_li) + final_mi;
                        params.lse_accum[args.n_split_idx * params.stride_lse_accum_split +
                                         q_token_idx * params.stride_lse_accum_s_q + head_idx] =
                            cur_lse_2base;
                    }
                }
            }
            float2 output_scale_float2 = float2{output_scale, output_scale};
            smem.bar_tOut_full.wait(args.outer_loop_phase);
            ptx::tcgen05_after_thread_sync();
            if (is_last_o && ptx::elect_one_sync()) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            if (args.is_no_split) {
                UNROLL
                for (int chunk = 0; chunk < 2; ++chunk) {
                    float2 first[B_EPI / 2];
                    nv_bfloat162 first_bf16[B_EPI / 2];
                    ptx::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + chunk * 2 * B_EPI, first);
                    ptx::fence_view_async_tmem_load();
                    UNROLL
                    for (int i = 0; i < B_EPI / 2; ++i) {
                        first_bf16[i] =
                            __float22bfloat162_rn(ptx::float2_mul(first[i], output_scale_float2));
                    }

                    float2 second[B_EPI / 2];
                    ptx::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + (chunk * 2 + 1) * B_EPI, second);
                    ptx::fence_view_async_tmem_load();
                    if (chunk == 1) {
                        smem.bar_tOut_empty.arrive(1u);
                        if (warp_idx == 0 && ptx::elect_one_sync())
                            ptx::tma_store_wait<0>();
                        ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                    }
                    UNROLL
                    for (int i = 0; i < B_EPI / 8; ++i) {
                        nv_bfloat162 second_bf16[4];
                        UNROLL
                        for (int j = 0; j < 4; ++j) {
                            second_bf16[j] = __float22bfloat162_rn(
                                ptx::float2_mul(second[i * 4 + j], output_scale_float2));
                        }
                        ptx::st_shared(sO_addrs[i],
                                       *reinterpret_cast<__int128_t*>(first_bf16 + i * 4));
                        ptx::st_shared(sO_addrs[i] + B_EPI * (H_Q / 2),
                                       *reinterpret_cast<__int128_t*>(second_bf16));
                    }
                    ptx::fence_view_async_shared();
                    ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                    if (warp_idx == 0 && ptx::elect_one_sync() &&
                        (2 * args.s_q_idx + cta_idx < params.s_q)) {
                        const uint64_t cache_hint =
                            ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_FIRST>();
                        ptx::SM90_TMA_STORE_5D::copy(
                            &tma_params.tensor_map_o, o_storage, 0, 0, chunk * 2, 0,
                            (args.batch_idx * params.s_q) + 2 * args.s_q_idx + cta_idx, cache_hint);
                        ptx::tma_store_arrive();
                    }
                }
            } else {
                UNROLL
                for (int k = 0; k < (D_V / 2) / B_EPI_SPLITKV; ++k) {
                    int cur_buf_idx = k % NUM_EPI_SPLITKV_BUFS;
                    if (k == 0) {
                        ptx::tma_store_wait<0>();
                    } else {
                        ptx::tma_store_wait<NUM_EPI_SPLITKV_BUFS - 1>();
                    }
                    ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);

                    float o[B_EPI_SPLITKV];
                    ptx::tmem_ld_32dp32bNx<B_EPI_SPLITKV>(tmem_cols::O + k * B_EPI_SPLITKV, o);
                    ptx::fence_view_async_tmem_load();
                    if (k == (D_V / 2) / B_EPI_SPLITKV - 1) {
                        smem.bar_tOut_empty.arrive(1u);
                    }
                    UNROLL
                    for (int i = 0; i < B_EPI_SPLITKV / 4; ++i) {
                        UNROLL
                        for (int j = 0; j < 4; j += 2) {
                            *(float2*)(o + i * 4 + j) = ptx::float2_mul(
                                float2{o[i * 4 + j], o[i * 4 + j + 1]}, output_scale_float2);
                        }

                        ptx::st_shared(sO_accum_addrs[i] +
                                           cur_buf_idx * ((H_Q / 2) * B_EPI_SPLITKV * 2),
                                       *(__int128_t*)(o + i * 4));
                    }

                    ptx::fence_view_async_shared();
                    ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                    float* cur_buf_base =
                        (float*)o_storage + cur_buf_idx * ((H_Q / 2) * B_EPI_SPLITKV * 2);
                    int q_token_idx = 2 * args.s_q_idx + cta_idx;
                    bool is_real_q_token = q_token_idx < params.s_q;
                    if (warp_idx == 0 && ptx::elect_one_sync() && is_real_q_token) {
                        ptx::SM90_TMA_STORE_5D::copy(&tma_params.tensor_map_o_accum, cur_buf_base,
                                                     0, 0, k * (B_EPI_SPLITKV / 32), q_token_idx,
                                                     args.n_split_idx);
                        ptx::tma_store_arrive();
                    } else if (warp_idx == 1 && ptx::elect_one_sync() && is_real_q_token) {
                        ptx::SM90_TMA_STORE_5D::copy(&tma_params.tensor_map_o_accum,
                                                     cur_buf_base + (H_Q / 2) * B_EPI_SPLITKV, 0, 0,
                                                     k * (B_EPI_SPLITKV / 32) + (D_V / 2) / 32,
                                                     q_token_idx, args.n_split_idx);
                        ptx::tma_store_arrive();
                    }
                }
            }
        };

        OuterloopArgs last_args;
        last_args.batch_idx = -1;

        run_outer_loop([&](const OuterloopArgs& args) {
            smem.bar_tQ_empty[args.outer_loop_phase].wait(decode_q_phase ^ 1);
            ptx::tcgen05_after_thread_sync();
            ptx::tma_store_wait<0>();
            ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
            if (warp_idx == 1 && ptx::elect_one_sync()) {
                const int raw_extra_start = (params.topk + 3) & ~3;
                const bool raw_staged =
                    raw_extra_start + params.extra_topk <= SharedMemoryPlan::RAW_INDEX_CAPACITY &&
                    (params.topk & 3) == 0 && (params.extra_topk & 3) == 0 &&
                    (params.stride_indices_b & 3) == 0 && (params.stride_indices_s_q & 3) == 0 &&
                    (params.stride_extra_indices_b & 3) == 0 &&
                    (params.stride_extra_indices_s_q & 3) == 0 &&
                    (reinterpret_cast<uintptr_t>(params.indices) & 15) == 0 &&
                    (reinterpret_cast<uintptr_t>(params.extra_indices) & 15) == 0;
                if (raw_staged) {
                    int* raw_dst = smem.raw_indices[args.outer_loop_phase].data();
                    const int* raw_main = params.indices +
                                          params.stride_indices_b * args.batch_idx +
                                          params.stride_indices_s_q * args.s_q_idx;
                    const int* raw_extra = params.extra_indices +
                                           params.stride_extra_indices_b * args.batch_idx +
                                           params.stride_extra_indices_s_q * args.s_q_idx;
                    smem.bar_raw_indices_full[args.outer_loop_phase].arrive_and_expect_tx(
                        (params.topk + params.extra_topk) * 4);
                    if (params.topk > 0)
                        ptx::SM90_BULK_COPY_G2S::copy(
                            raw_main,
                            reinterpret_cast<uint64_t*>(
                                &smem.bar_raw_indices_full[args.outer_loop_phase]),
                            raw_dst, params.topk * 4);
                    if (params.extra_topk > 0)
                        ptx::SM90_BULK_COPY_G2S::copy(
                            raw_extra,
                            reinterpret_cast<uint64_t*>(
                                &smem.bar_raw_indices_full[args.outer_loop_phase]),
                            raw_dst + raw_extra_start, params.extra_topk * 4);
                }
            }
            if (warp_idx == 0 && ptx::elect_one_sync()) {

                const int q_batch_base = args.batch_idx * (params.stride_q_b / params.stride_q_s_q);
                ptx::SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
                    &tma_params.tensor_map_q, (uint64_t*)&smem.bar_sQ_full[args.outer_loop_phase],
                    ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_FIRST>(),
                    smem.Q[args.outer_loop_phase].data(), 0, 0, 0, 0,
                    q_batch_base + min(2 * args.s_q_idx + cta_idx, params.s_q - 1));
                if (cta_idx == 0) {
                    smem.bar_sQ_full[args.outer_loop_phase].arrive_and_expect_tx(H_Q * D_Q *
                                                                                 sizeof(fp8_e4m3));
                }
            }
            {
                const int head_idx = lane_idx;
                const int q_token_idx = 2 * args.s_q_idx + cta_idx;
                const int q_load_token_idx = min(q_token_idx, params.s_q - 1);
                const int q_batch_offset = args.batch_idx * params.stride_q_b;
                const uint8_t* q_scale_base = reinterpret_cast<const uint8_t*>(params.q) +
                                              q_batch_offset +
                                              q_load_token_idx * params.stride_q_s_q + D_Q;
                uint4 packed_scale_lo, packed_scale_hi;
                KU_LDG_128(q_scale_base + head_idx * params.stride_q_h_q, &packed_scale_lo, ".nc",
                           "evict_first", "128B");
                KU_LDG_128(q_scale_base + (head_idx + 32) * params.stride_q_h_q, &packed_scale_hi,
                           ".nc", "evict_first", "128B");
                uint32_t q_scale_words[8] = {
                    packed_scale_lo.x, packed_scale_hi.x, packed_scale_lo.x, packed_scale_hi.x,
                    packed_scale_lo.y, packed_scale_hi.y, packed_scale_lo.y, packed_scale_hi.y};
                {
                    q_scale_words[2] = packed_scale_lo.y;
                    q_scale_words[3] = packed_scale_hi.y;
                    q_scale_words[4] = packed_scale_lo.x;
                    q_scale_words[5] = packed_scale_hi.x;
                }
                ptx::tmem_st_32dp32bNx<8>((tmem_cols::Q_scale + args.outer_loop_phase * 8) +
                                              warp_idx * 32 * 65536,
                                          q_scale_words);
                ptx::fence_view_async_tmem_store();
                ptx::tcgen05_before_thread_sync();
                ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                if (warp_idx == 0 && ptx::elect_one_sync()) {
                    if (cta_idx == 0) {
                        smem.bar_Q_scale_ready[args.outer_loop_phase].arrive();
                    } else {
                        smem.bar_Q_scale_ready[args.outer_loop_phase].arrive(uint32_t{0}, 1u);
                    }
                }
            }

            if (warp_idx == 0 && cta_idx == 0 && ptx::elect_one_sync()) {
                smem.bar_sQ_full[args.outer_loop_phase].wait(decode_q_phase);
                ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_tQ_full[args.outer_loop_phase],
                                                         1 | 2);
            }

            if (last_args.batch_idx != -1) {
                ptx::tma_store_wait<0>();
                ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                perform_o_copy_out(last_args, false);
            }
            // The next Q-empty acquisition already waits for the MMA
            // consumer to finish this QK, including its Q-full acquisition.
            // It therefore protects Q-full reuse without a WG0 observer.
            last_args = args;
        });
        if (last_args.batch_idx != -1) {
            ptx::tma_store_wait<0>();
            ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
            perform_o_copy_out(last_args, true);
        }

        if (warp_idx == 0) {
            ptx::Allocator2Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // KV gather and scale staging threads.
        ptx::warpgroup_reg_dealloc<80>();
        ::RingBufferState rs;

        // Decode gathers both V's 64-token half-D tile and QK's native
        // 32-token full-D tile per CTA. Both D256 partials now belong to
        // the same softmax thread, eliminating their shared-memory exchange.
        if (ptx::elect_one_sync()) {
            const int producer_warp = warp_idx - 4;
            run_outer_loop([&](const OuterloopArgs& args) {
                NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx;
                     ++block_idx) {
                    auto [k_buf_idx, k_bar_phase] = rs.template get<NUM_K_BUFS>();
                    auto [index_buf_idx, index_bar_phase] = rs.template get<NUM_INDEX_BUFS>();
                    smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                    if (producer_warp < 2) {
                        smem.bar_QK_done[k_buf_idx].wait(k_bar_phase ^ 1);

                        UNROLL
                        for (int local_row = producer_warp * 16;
                             local_row < producer_warp * 16 + 16; local_row += 4) {
                            int4 coords = *reinterpret_cast<int4*>(
                                smem.index_storage.tma_coord[index_buf_idx] + cta_idx * 32 +
                                local_row);
                            UNROLL
                            for (int col = 0; col < D_K / 128; ++col) {
                                ptx::tma_gather4_cta_group_2<true>(
                                    block_idx >= args.num_orig_kv_blocks
                                        ? &tma_params.tensor_map_extra_kv
                                        : &tma_params.tensor_map_kv,
                                    smem.bar_K_full[k_buf_idx],
                                    smem.K[k_buf_idx].data() + local_row * 128 +
                                        col * 128 * (B_TOPK / 2),
                                    col * 128, coords,
                                    ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_LAST>());
                            }
                        }
                    } else {
                        smem.bar_V_empty[k_buf_idx].wait(k_bar_phase ^ 1);
                        const int row_begin = (producer_warp - 2) * (B_TOPK / 2);
                        UNROLL
                        for (int row = row_begin; row < row_begin + B_TOPK / 2; row += 4) {
                            int4 coords = *reinterpret_cast<int4*>(
                                smem.index_storage.tma_coord[index_buf_idx] + row);
                            UNROLL
                            for (int col = 0; col < (D_K / 128) / 2; ++col) {
                                ptx::tma_gather4_cta1(
                                    block_idx >= args.num_orig_kv_blocks
                                        ? &tma_params.tensor_map_extra_kv
                                        : &tma_params.tensor_map_kv,
                                    smem.bar_V_full[k_buf_idx],
                                    smem.V[k_buf_idx].data() + row * 128 + col * 128 * B_TOPK,
                                    col * 128 + cta_idx * (D_K / 2), coords,
                                    ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_LAST>());
                            }
                        }
                    }
                    smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                    rs.update();
                }
            });
        }
    } else if (warpgroup_idx == 2) {
        ptx::warpgroup_reg_dealloc<80>();
        // Rotate logical roles only within physical warpgroup 2 for decode.
        // Register allocation, lanes, buffers, and synchronization are unchanged.
        const int wg2_physical_warp = warp_idx;
        const int warp_idx = 8 + ((wg2_physical_warp - 8 + 2) & 3);

        ::RingBufferState rs;
        // CTA1 issues SV independently; CTA0 issues QK.
        if (warp_idx == 8 && cta_idx == 1 && ptx::elect_one_sync()) {
            run_outer_loop([&](const OuterloopArgs& args) {
                auto issue_SV = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] =
                        rs.offset_by(rs_offset).template get<NUM_K_BUFS>();
                    auto [s_buf_idx, s_bar_phase] = rs.offset_by(rs_offset).template get<2>();
                    {
                        smem.bar_V_full[k_buf_idx].arrive_and_expect_tx(B_TOPK * D_V *
                                                                        sizeof(fp8_e4m3));
                        smem.bar_V_full[k_buf_idx].wait(k_bar_phase);
                    }
                    smem.bar_S_O_full[s_buf_idx].wait(s_bar_phase);
                    if (k == args.start_block_idx) {
                        smem.bar_tOut_empty.wait(args.outer_loop_phase ^ 1);
                    }
                    ptx::tcgen05_after_thread_sync();
                    sm100::issue_sv<B_TOPK, true>(
                        smem.S[s_buf_idx].data(), smem.V[k_buf_idx].data(), tmem_cols::O,
                        tmem_cols::S_scale + s_buf_idx * tmem_cols::S_scale_stride,
                        tmem_cols::V_scale, k == args.start_block_idx);
                    ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_S_empty[s_buf_idx], 1 | 2);
                    ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_V_empty[k_buf_idx], 1 | 2);
                };

                NO_UNROLL
                for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                    issue_SV(k, 0);
                    rs.update();
                }
                ptx::tcgen05_before_thread_sync();
                ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_tOut_full, 1 | 2);
            });
        }
        if (warp_idx == 8 && cta_idx == 0 && ptx::elect_one_sync()) {

            run_outer_loop([&](const OuterloopArgs& args) {
                smem.bar_tQ_full[args.outer_loop_phase].wait(decode_q_phase);
                smem.bar_Q_scale_ready[args.outer_loop_phase].wait(decode_q_phase);
                ptx::tcgen05_after_thread_sync();

                auto issue_QK = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] =
                        rs.offset_by(rs_offset).template get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).template get<1>();
                    smem.bar_P_empty.wait(bar_phase ^ 1);
                    ptx::transac_bar_t& k_full = smem.bar_K_full[k_buf_idx];
                    k_full.arrive_and_expect_tx(B_TOPK * D_K * sizeof(fp8_e4m3));
                    k_full.wait(k_bar_phase);
                    ptx::tcgen05_after_thread_sync();

                    smem.bar_K_scale_copy_ready[k_buf_idx].wait(k_bar_phase);
                    copy_k_scale_smem_to_tmem(smem.k_scale_mma[k_buf_idx].data(),
                                              tmem_cols::K_scale);
                    ptx::tcgen05_before_thread_sync();
                    sm100::issue_qk<B_TOPK, true>(smem.Q[args.outer_loop_phase].data(),
                                                smem.K[k_buf_idx].data(), tmem_cols::P,
                                                tmem_cols::Q_scale + args.outer_loop_phase * 8,
                                                tmem_cols::K_scale);
                    ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_QK_done[k_buf_idx], 1 | 2);
                };

                NO_UNROLL
                for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                    issue_QK(k, 0);
                    rs.update();
                }
                ptx::tcgen05_before_thread_sync();
                ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_tQ_empty[args.outer_loop_phase],
                                                         1 | 2);
            });
        } else if (warp_idx == 9) {
            // Transform indices into TMA coordinates, scale offsets, and validity masks.
            static_assert(B_TOPK == 64);
            // Each thread is responsible for 2 tokens
            static constexpr int tma_coords_step_per_token = 1;
            int tma_coords_step_per_block = params.stride_kv_block / TMA_K_STRIDE_FOR_DECODING;
            int tma_coords_step_per_extra_block =
                params.stride_extra_kv_block / TMA_K_STRIDE_FOR_DECODING;

            run_outer_loop([&](const OuterloopArgs& args) {
                int* indices = (int*)params.indices + params.stride_indices_b * args.batch_idx +
                               params.stride_indices_s_q * args.s_q_idx;
                int* extra_indices = (int*)params.extra_indices +
                                     params.stride_extra_indices_b * args.batch_idx +
                                     params.stride_extra_indices_s_q * args.s_q_idx;
                const int raw_extra_start = (params.topk + 3) & ~3;
                const bool raw_staged =
                    raw_extra_start + params.extra_topk <= SharedMemoryPlan::RAW_INDEX_CAPACITY &&
                    (params.topk & 3) == 0 && (params.extra_topk & 3) == 0 &&
                    (params.stride_indices_b & 3) == 0 && (params.stride_indices_s_q & 3) == 0 &&
                    (params.stride_extra_indices_b & 3) == 0 &&
                    (params.stride_extra_indices_s_q & 3) == 0 &&
                    (reinterpret_cast<uintptr_t>(params.indices) & 15) == 0 &&
                    (reinterpret_cast<uintptr_t>(params.extra_indices) & 15) == 0;
                const int* raw_src = smem.raw_indices[args.outer_loop_phase].data();
                if (raw_staged)
                    smem.bar_raw_indices_full[args.outer_loop_phase].wait(decode_q_phase);

                struct IsOrigBlock {};
                struct IsExtraBlock {};
                auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
                    auto [index_buf_idx, index_bar_phase] = rs.template get<NUM_INDEX_BUFS>();
                    static constexpr bool IS_EXTRA_BLOCK =
                        std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                    int cur_block_size =
                        IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
                    int cur_tma_coords_step_per_block = IS_EXTRA_BLOCK
                                                            ? tma_coords_step_per_extra_block
                                                            : tma_coords_step_per_block;

                    int abs_pos, my_indices[2];
                    if (!IS_EXTRA_BLOCK) {
                        abs_pos = block_idx * B_TOPK + lane_idx * 2;
                        if (raw_staged) {
                            if (abs_pos + 1 < params.topk) {
                                ptx::ld_shared_v2_u32(raw_src + abs_pos, my_indices);
                            } else {
                                my_indices[0] = my_indices[1] = -1;
                            }
                        } else {
                            *(int2*)my_indices = __ldg((int2*)(indices + abs_pos));
                        }
                    } else {
                        abs_pos = (block_idx - args.num_orig_kv_blocks) * B_TOPK + lane_idx * 2;
                        // An extra row need not be a multiple of the tile
                        // size, or even int2-aligned when extra_topk is odd.
                        if (raw_staged) {
                            if (abs_pos + 1 < params.extra_topk) {
                                ptx::ld_shared_v2_u32(raw_src + raw_extra_start + abs_pos,
                                                      my_indices);
                            } else {
                                my_indices[0] = my_indices[1] = -1;
                            }
                        } else {
                            my_indices[0] = abs_pos < params.extra_topk
                                                ? (raw_staged ? raw_src[raw_extra_start + abs_pos]
                                                              : __ldg(extra_indices + abs_pos))
                                                : -1;
                            my_indices[1] =
                                abs_pos + 1 < params.extra_topk
                                    ? (raw_staged ? raw_src[raw_extra_start + abs_pos + 1]
                                                  : __ldg(extra_indices + abs_pos + 1))
                                    : -1;
                        }
                    }
                    smem.bar_valid_coord_scales_empty[index_buf_idx].wait(index_bar_phase ^ 1);

                    int tma_coords[2];
                    alignas(16) int64_t scale_offsets[2];
                    char valid_mask = 0;
                    UNROLL
                    for (int i = 0; i < 2; ++i) {
                        int block_idx, idx_in_block;
                        block_idx = (unsigned int)my_indices[i] / cur_block_size;
                        idx_in_block = (unsigned int)my_indices[i] % cur_block_size;
                        bool is_token_valid =
                            my_indices[i] != -1 &&
                            (abs_pos + i <
                             (IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length));
                        valid_mask |= is_token_valid << i;
                        tma_coords[i] =
                            is_token_valid ? block_idx * cur_tma_coords_step_per_block +
                                                 idx_in_block * tma_coords_step_per_token
                                           : -1; // If the token is invalid because it topk position
                                                 // exceeds topk_length, we must manually fill
                                                 // tma_coords with -1 to avoid copying-in NaN.
                        scale_offsets[i] =
                            is_token_valid
                                ? static_cast<int64_t>(block_idx) *
                                          (IS_EXTRA_BLOCK ? params.stride_extra_kv_block
                                                          : params.stride_kv_block) +
                                      static_cast<int64_t>(cur_block_size) * D_K +
                                      static_cast<int64_t>(idx_in_block) * K_SCALE_SLOT_BYTES
                                : 0;
                        // Index production runs ahead of the scale reader.
                        // This hint hides its scattered page-tail load latency.
                        if (is_token_valid) {
                            const auto* ptr = reinterpret_cast<const uint8_t*>(
                                                  IS_EXTRA_BLOCK ? params.extra_kv : params.kv) +
                                              scale_offsets[i];
                            ptx::prefetch_global_l1(ptr);
                        }
                    }
                    ptx::st_shared(smem.index_storage.k_scale_offset[index_buf_idx] + lane_idx * 2,
                                   *reinterpret_cast<const __int128_t*>(scale_offsets));
                    valid_mask <<= lane_idx % 4 * 2;
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                    *(int2*)(smem.index_storage.tma_coord[index_buf_idx] + lane_idx * 2) =
                        *(int2*)tma_coords;
                    if (lane_idx % 4 == 0)
                        smem.is_k_valid[index_buf_idx][lane_idx / 4] = valid_mask;

                    smem.bar_valid_coord_scales_full[index_buf_idx].arrive();
                    rs.update();
                };

                NO_UNROLL
                for (int block_idx = args.start_block_idx;
                     block_idx < min(args.num_orig_kv_blocks, args.end_block_idx); ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{});
                }

                NO_UNROLL
                for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks);
                     block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsExtraBlock{});
                }
            });
        } else if (warp_idx == 10 || warp_idx == 11) {
            // Load the two unique UE8M0 words into registers and repeat
            // them in the native QK UTCCP layout. Index storage already
            // contains the page-aware scale offsets, including extra KV.
            const int scale_warp = warp_idx - 10;
            ::RingBufferState scale_rs;
            run_outer_loop([&](const OuterloopArgs& args) {
                NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx;
                     ++block_idx) {
                    auto [k_buf_idx, k_bar_phase] = scale_rs.template get<NUM_K_BUFS>();
                    auto [index_buf_idx, index_bar_phase] = scale_rs.template get<NUM_INDEX_BUFS>();
                    smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                    const int row = scale_warp * (B_TOPK / 2) + lane_idx;
                    const bool token_valid =
                        static_cast<uint8_t>(smem.is_k_valid[index_buf_idx][row / 8]) &
                        (uint8_t(1) << (row % 8));
                    const bool is_extra = block_idx >= args.num_orig_kv_blocks;
                    const uint8_t* scale_base =
                        reinterpret_cast<const uint8_t*>(is_extra ? params.extra_kv : params.kv);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(
                        scale_base +
                        (token_valid ? smem.index_storage.k_scale_offset[index_buf_idx][row] : 0));

                    const uint2 packed =
                        token_valid ? __ldg(reinterpret_cast<const uint2*>(src)) : make_uint2(0, 0);
                    // Load into registers before waiting to overwrite the
                    // stage. Softmax releases it only after QK and V reads.
                    smem.bar_v_scale_empty[k_buf_idx].wait(k_bar_phase ^ 1);
                    uint32_t* native_scale =
                        reinterpret_cast<uint32_t*>(smem.k_scale_mma[k_buf_idx].data() + row * 16);
                    native_scale[0] = native_scale[1] = packed.x;
                    native_scale[2] = native_scale[3] = packed.y;
                    fp8_e8m0 token_scale;
                    token_scale.storage = packed.x & 0xff;
                    smem.v_token_scale[k_buf_idx][row] = float(token_scale);
                    ptx::fence_view_async_shared();
                    __syncwarp();
                    // Every writer publishes V scales; every reader
                    // releases the index stage it actually consumed.
                    smem.bar_v_scale_full[k_buf_idx].arrive();
                    if (ptx::elect_one_sync()) {
                        smem.bar_K_scale_copy_ready[k_buf_idx].arrive(0u);
                    }
                    smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                    scale_rs.update();
                }
            });
        }
    } else {
        // Scale & Exp threads
        ptx::warpgroup_reg_alloc<176>();

        int local_warp_idx = warp_idx - 12;
        int s_row = idx_in_warpgroup % (H_Q / 2);
        int s_col_base = (local_warp_idx >= 2 ? B_TOPK / 2 : 0);

        ::RingBufferState rs;
        run_outer_loop([&](const OuterloopArgs& args) {
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;
            static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

            NO_UNROLL
            for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                auto [k_buf_idx, k_bar_phase] = rs.template get<NUM_K_BUFS>();
                auto [indices_buf_idx, indices_bar_phase] = rs.template get<NUM_INDEX_BUFS>();
                auto [s_buf_idx, s_bar_phase] = rs.template get<2>();
                smem.bar_valid_coord_scales_full[indices_buf_idx].wait(indices_bar_phase);

                float p[NUM_ELEMS_PER_THREAD];
                float p_max;
                using ValidMask = uint32_t;
                ValidMask valid;

                smem.bar_QK_done[k_buf_idx].wait(k_bar_phase);
                ptx::tcgen05_after_thread_sync();
                {
                    // Same D256 addition as the old path, with both
                    // partials in this thread's TMEM lane. No peer exchange.
                    float p_peer[NUM_ELEMS_PER_THREAD];
                    ptx::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P, p);
                    ptx::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P + B_TOPK / 2, p_peer);
                    ptx::fence_view_async_tmem_load();
                    UNROLL
                    for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 2)
                        *reinterpret_cast<float2*>(p + i) =
                            ptx::float2_add(*reinterpret_cast<float2*>(p + i),
                                            *reinterpret_cast<float2*>(p_peer + i));
                    ptx::fence_view_async_tmem_load();
                    ptx::tcgen05_before_thread_sync();
                    smem.bar_P_empty.arrive(0u);
                    valid = reinterpret_cast<uint32_t*>(
                        smem.is_k_valid[indices_buf_idx])[local_warp_idx / 2];
                    if (valid != static_cast<ValidMask>(-1)) {
                        UNROLL
                        for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                            if (!(valid >> i & 1))
                                p[i] = -CUDART_INF_F;
                        }
                    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1030 && __CUDA_ARCH__ < 1100
                    // Hardware reduction precedes masking, so its result is
                    // usable only when all 32 token positions are valid.
                    p_max = get_max<NUM_ELEMS_PER_THREAD>(p);
#else
                    p_max = get_max<NUM_ELEMS_PER_THREAD>(p);
#endif
                }

                float cur_pi_max = p_max * params.sm_scale_div_log2;

                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                ptx::NamedBarrier::arrive_and_wait(64, barrier_ids::WG2_WARP02_SYNC +
                                                           (local_warp_idx & 1));
                cur_pi_max = max(cur_pi_max, smem.rowwise_max_buf[idx_in_warpgroup ^ 64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                float new_max, scale_for_old;
                if (!should_scale_o) {
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;

                fp8_e4m3 s[NUM_ELEMS_PER_THREAD];
                smem.bar_v_scale_full[k_buf_idx].wait(k_bar_phase);
                const float2 logit_scale2 =
                    make_float2(params.sm_scale_div_log2, params.sm_scale_div_log2);
                const float2 neg_max2 = make_float2(-new_max, -new_max);
                float2 cur_sum2[4] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f),
                                      make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
                float2 s_abs_max2[4] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f),
                                        make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
                UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 2) {
                    const int acc = (i / 2) & 3;
                    float2 logits = ptx::float2_fma(*reinterpret_cast<const float2*>(p + i),
                                                    logit_scale2, neg_max2);
                    float2 s_value = make_float2(exp2f(logits.x), exp2f(logits.y));
                    cur_sum2[acc] = ptx::float2_add(cur_sum2[acc], s_value);
                    const int token = s_col_base + i;
                    float2 scaled_s = ptx::float2_mul(
                        s_value,
                        *reinterpret_cast<const float2*>(&smem.v_token_scale[k_buf_idx][token]));
                    s_abs_max2[acc].x = max(s_abs_max2[acc].x, scaled_s.x);
                    s_abs_max2[acc].y = max(s_abs_max2[acc].y, scaled_s.y);
                    *reinterpret_cast<float2*>(p + i) = scaled_s;
                }
                smem.bar_v_scale_empty[k_buf_idx].arrive();
                cur_sum2[0] = ptx::float2_add(cur_sum2[0], cur_sum2[1]);
                cur_sum2[2] = ptx::float2_add(cur_sum2[2], cur_sum2[3]);
                cur_sum2[0] = ptx::float2_add(cur_sum2[0], cur_sum2[2]);
                UNROLL
                for (int i = 1; i < 4; ++i) {
                    s_abs_max2[0].x = max(s_abs_max2[0].x, s_abs_max2[i].x);
                    s_abs_max2[0].y = max(s_abs_max2[0].y, s_abs_max2[i].y);
                }
                float cur_sum = cur_sum2[0].x + cur_sum2[0].y;
                float s_abs_max = max(s_abs_max2[0].x, s_abs_max2[0].y);
                float raw_s_scale = s_abs_max > 0.0f ? s_abs_max / 448.0f : 1.0f;
                fp8_e8m0 s_scale_exp_e8m0 = fp8_e8m0(raw_s_scale);
                constexpr uint8_t min_s_scale_exp = 61;
                s_scale_exp_e8m0.storage = max(s_scale_exp_e8m0.storage, min_s_scale_exp);
                float s_scale = __uint_as_float(uint32_t(254u - s_scale_exp_e8m0.storage) << 23);
                const float2 scale2 = make_float2(s_scale, s_scale);

                UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 2) {
                    float2 x = *reinterpret_cast<const float2*>(p + i);
                    x = ptx::float2_mul(x, scale2);

                    *reinterpret_cast<__nv_fp8x2_storage_t*>(s + i) =
                        __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
                }
                li = fmaf(li, scale_for_old, cur_sum);

                smem.bar_S_empty[s_buf_idx].wait(s_bar_phase ^ 1);

                smem.s_scale_exp[k_buf_idx][s_row][local_warp_idx >= 2 ? 1 : 0] =
                    s_scale_exp_e8m0.storage;
                ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::S_SCALE_SYNC);
                store_s_scale_to_tmem(tmem_cols::S_scale + s_buf_idx * tmem_cols::S_scale_stride,
                                      local_warp_idx, smem.s_scale_exp[k_buf_idx][s_row][0],
                                      smem.s_scale_exp[k_buf_idx][s_row][1]);
                ptx::fence_view_async_tmem_store();
                UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD / 16; ++i) {
                    fp8_e4m3* sS_base = sm100::k_major_ptr<64, 0>(smem.S[s_buf_idx].data(), s_row,
                                                                s_col_base + i * 16);
                    ptx::st_shared(sS_base, *reinterpret_cast<__int128_t*>(s + i * 16));
                }

                if (k > args.start_block_idx && should_scale_o) {
                    auto [prev_k_buf_idx, prev_k_bar_phase] =
                        rs.offset_by(-1).template get<NUM_K_BUFS>();
                    // Keep the in-place rescale ordered after the previous SV.
                    smem.bar_V_empty[prev_k_buf_idx].wait(prev_k_bar_phase);
                    ptx::tcgen05_after_thread_sync();
                    rescale_O<D_V, 32, tmem_cols::O>(scale_for_old);
                    ptx::tcgen05_before_thread_sync();
                }

                ptx::fence_view_async_shared();
                smem.bar_S_O_full[s_buf_idx].arrive(1u);
                smem.bar_valid_coord_scales_empty[indices_buf_idx].arrive();

                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                // real_mi == -CUDART_INF_F <=> No valid TopK indices
                // We set li to 0 to fit the definition that li := exp(x[i] - mi)
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // Publish both partial sums directly. The 128-writer full barrier
            // replaces the softmax WG's local reduction barrier.
            smem.bar_li_empty.wait(args.outer_loop_phase ^ 1);
            smem.rowwise_li_buf[idx_in_warpgroup] = li;
            if (idx_in_warpgroup < H_Q / 2)
                smem.final_mi[idx_in_warpgroup] = mi;
            smem.bar_li_full.arrive();
        });
    }

    ptx::barrier_cluster_arrive_relaxed();
    ptx::barrier_cluster_wait_acquire();

#else
    asm("trap;");
#endif
}

template <typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
    flash_fwd_splitkv_mla_fp8_sparse_kernel(
        __grid_constant__ const typename Kernel::ArgT params,
        __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template <SparseAttnFwdMode FWD_MODE, int D_QK, int PREFILL_B_TOPK>
void KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>::run(const ArgT& params)
    requires(IS_DECODE)
{
    static_assert(D_QK == 512);

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % 64 == 0);
    KU_ASSERT(params.h_q == TOKEN_H_Q);
    KU_ASSERT(params.s_q > 0);

    KU_ASSERT(params.d_qk == D_QK);

    static_assert(D_Q == 512);
    KU_ASSERT(params.stride_q_h_q == Q_BYTES_PER_HEAD,
              "Q head stride must be 512 e4m3 bytes + an aligned 16-byte scale slot");
    KU_ASSERT(params.stride_kv_row == KV_BYTES_PER_TOKEN,
              "decode KV row stride must include the 16-byte replicated scale slot");
    if (params.extra_topk > 0) {
        KU_ASSERT(params.stride_extra_kv_row == KV_BYTES_PER_TOKEN,
                  "decode extra KV row stride must include the 16-byte replicated scale slot");
        KU_ASSERT(params.stride_extra_kv_block % TMA_K_STRIDE_FOR_DECODING == 0,
                  "decode extra KV page stride must be a multiple of the 512B TMA stride");
    }
    KU_ASSERT(reinterpret_cast<int64_t>(params.q) % 16 == 0);
    KU_ASSERT(reinterpret_cast<int64_t>(params.kv) % 16 == 0);

    CUtensorMap tensor_map_q;
    KU_ASSERT(
        params.stride_q_b % params.stride_q_s_q == 0,
        "In decode mode for MODEL1 sparse fp8 decoding on sm100f, q.stride(0) (on the batch dimension) must be divisible by q.stride(1) (on the sequence dimension).");
    tensor_map_q = ptx::make_tensor_map(
        {128ul, TOKEN_H_Q, 2ul, (D_Q / 128ul) / 2ul,
         (unsigned long)params.b * (params.stride_q_b / params.stride_q_s_q)},
        ptx::make_stride_helper<int>({params.stride_q_h_q, 128, D_Q / 2, params.stride_q_s_q},
                                     sizeof(fp8_e4m3)),
        {128, TOKEN_H_Q, 2, (D_Q / 128) / 2, 1}, params.q, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_extra_kv = {};
    auto get_kv_tensormap = [&](bool is_extra, void* k_ptr, int num_blocks, int64_t stride_kv_block,
                                int64_t stride_kv_row) -> CUtensorMap {
        KU_ASSERT(
            (int64_t)k_ptr % 16 == 0,
            "The base address of %sk_ptr (%p) must be 16B aligned for sparse fp8 attention on sm100f",
            is_extra ? "extra_" : "", k_ptr);
        KU_ASSERT(stride_kv_block % TMA_K_STRIDE_FOR_DECODING == 0,
                  "%sk_cache.stride(0) (%ld) must be a multiple of %d. Padding might be necessary",
                  is_extra ? "extra_" : "", stride_kv_block, TMA_K_STRIDE_FOR_DECODING);
        CUtensorMap tensor_map_kv = ptx::make_tensor_map(
            {D_K, (uint64_t)num_blocks * (stride_kv_block / TMA_K_STRIDE_FOR_DECODING)},
            {TMA_K_STRIDE_FOR_DECODING}, {128, 1}, k_ptr,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
        return tensor_map_kv;
    };
    tensor_map_kv = get_kv_tensormap(false, params.kv, params.num_blocks, params.stride_kv_block,
                                     params.stride_kv_row);
    if (params.extra_topk > 0)
        tensor_map_extra_kv =
            get_kv_tensormap(true, params.extra_kv, params.extra_num_blocks,
                             params.stride_extra_kv_block, params.stride_extra_kv_row);

    CUtensorMap tensor_map_o = ptx::make_tensor_map(
        {64, TOKEN_H_Q, D_V / 128, 2ul, (unsigned long)params.b * params.s_q},
        ptx::make_stride_helper<int>({params.stride_o_h_q, 64, D_V / 2, params.stride_o_s_q},
                                     sizeof(ptx::bf16)),
        {64, TOKEN_H_Q, O_STAGE_D / 128, 2, 1}, params.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

    CUtensorMap tensor_map_o_accum = ptx::make_tensor_map(
        {32, TOKEN_H_Q, D_V / 32, (unsigned long)params.s_q,
         (unsigned long)params.num_sm_parts + params.b},
        ptx::make_stride_helper<int>(
            {params.stride_o_accum_h_q, 32, params.stride_o_accum_s_q, params.stride_o_accum_split},
            sizeof(float)),
        {32, TOKEN_H_Q, B_EPI_SPLITKV / 32, 1, 1}, params.o_accum, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

    TmaParams tma_params = {tensor_map_q, tensor_map_o, tensor_map_o_accum, tensor_map_kv,
                            tensor_map_extra_kv};

    auto kernel =
        &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_shape = dim3(2 * ptx::ceil_div(params.s_q, 2), params.num_sm_parts, 1);

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t config{grid_shape, dim3(NUM_THREADS, 1, 1), smem_size, params.stream, &attr,
                              1};
    KU_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params, tma_params));
}

} // namespace sm100::dual_mxfp8
