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

namespace mxdsa::attention {

template <SparseAttnFwdMode FWD_MODE, int D_QK, int PREFILL_B_TOPK>
__device__ void KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>::sparse_attn_fwd_kernel_devfunc(
    const ArgT& params, const TmaParams& tma_params)
    requires(!IS_DECODE)
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
        for (int slot = 0; slot < 1; ++slot) {
            smem.bar_sQ_full[slot].init(1);

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
        smem.bar_li_full.init(H_Q / 2);
        smem.bar_li_empty.init(128);
        smem.bar_clc_full.init(1);
        smem.bar_clc_empty.init(NUM_WORKER_THREADS);
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
            smem.bar_valid_coord_scales_full[i].init(B_TOPK == 128 ? 2 : B_TOPK / 8);
            smem.bar_valid_coord_scales_empty[i].init(128);
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
    };

    auto run_outer_loop = [&](auto loop_body) -> bool {
        int outer_loop_phase = false;
        ptx::CLCResult next_job = {true, (int)blockIdx.x, 0, 0};
        NO_UNROLL
        while (next_job.is_valid) {
            int s_q_idx = next_job.x / 2; // pair index for this head64 prefill kernel

            int topk_length =
                params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;

            int num_k_blocks = max(mxdsa::ceil_div(topk_length, (int)B_TOPK), 1);
            OuterloopArgs args = {(bool)outer_loop_phase, 0, s_q_idx, 0, num_k_blocks, topk_length};

            loop_body(args);

            smem.bar_clc_full.wait(outer_loop_phase);
            next_job = ptx::get_clc_query_response<true>(smem.clc_response_obj);
            smem.bar_clc_empty.arrive(0u);

            outer_loop_phase ^= 1;
        }
        return outer_loop_phase;
    };

    if (warpgroup_idx == 0) {
        ptx::warpgroup_reg_alloc<B_TOPK == 128 ? 144 : 176>();
        const uint32_t warp_dp_addr = tmem_cols::V_scale + warp_idx * 32 * 65536;
        UNROLL
        for (int col = 0; col < 8; ++col) {
            ptx::SM100_TMEM_STORE_32dp32b1x::copy(warp_idx < 2 ? params.w1[col] : params.w2[col],
                                                  warp_dp_addr + col);
        }
        ptx::fence_view_async_tmem_store();

        auto perform_o_copy_out = [&](const OuterloopArgs& args, bool is_last_o) {
            ptx::bf16* o_storage = smem.O.data();
            ptx::bf16* sO_addrs[B_EPI / 8];
            UNROLL
            for (int i = 0; i < B_EPI / 8; ++i) {

                sO_addrs[i] = mxdsa::attention::k_major_ptr<64, 128>(
                    o_storage, idx_in_warpgroup % 64,
                    (idx_in_warpgroup / 64) * (O_STAGE_D / 2) + i * 8);
            }

            smem.bar_li_full.wait(args.outer_loop_phase);
            float output_scale = smem.rowwise_li_buf[idx_in_warpgroup % 64];
            smem.bar_li_empty.arrive();
            float2 output_scale_float2 = float2{output_scale, output_scale};
            smem.bar_tOut_full.wait(args.outer_loop_phase);

            if (is_last_o && ptx::elect_one_sync()) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            if constexpr (SPLIT_O_STAGING) {
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
                    if (warp_idx == 0 && ptx::elect_one_sync()) {
                        const uint64_t cache_hint =
                            ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_FIRST>();
                        ptx::SM90_TMA_STORE_5D::copy(&tma_params.tensor_map_o, o_storage, 0, 0,
                                                     chunk * 2, 0, 2 * args.s_q_idx + cta_idx,
                                                     cache_hint);
                        ptx::tma_store_arrive();
                    }
                }
            } else {
                UNROLL
                for (int k = 0; k < (D_V / 2) / B_EPI; ++k) {
                    float2 o[B_EPI / 2];
                    ptx::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + k * B_EPI, o);
                    ptx::fence_view_async_tmem_load();
                    if (k == (D_V / 2) / B_EPI - 1) {
                        smem.bar_tOut_empty.arrive(1u);
                    }
                    UNROLL
                    for (int i = 0; i < B_EPI / 8; ++i) {
                        nv_bfloat162 o_bf16[4];
                        UNROLL
                        for (int j = 0; j < 4; ++j) {
                            o[i * 4 + j] = ptx::float2_mul(o[i * 4 + j], output_scale_float2);
                            o_bf16[j] = __float22bfloat162_rn(o[i * 4 + j]);
                        }
                        ptx::bf16* o_do_addr = sO_addrs[i] + k * B_EPI * (H_Q / 2);
                        if (k == 0 && i == 0) {
                            smem.bar_tQ_full[0].wait(args.outer_loop_phase ^ 1 ^ is_last_o);
                        }
                        ptx::st_shared(o_do_addr, *(__int128_t*)o_bf16);
                    }
                }

                ptx::fence_view_async_shared();
                ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                int q_token_idx = 2 * args.s_q_idx + cta_idx;
                bool is_real_q_token = q_token_idx < params.s_q;
                if (warp_idx == 0 && ptx::elect_one_sync() && is_real_q_token) {
                    ptx::SM90_TMA_STORE_5D::copy(&tma_params.tensor_map_o, o_storage, 0, 0, 0,
                                                 q_token_idx, 0);
                    ptx::tma_store_arrive();
                }
            }
        };

        OuterloopArgs last_args;
        last_args.batch_idx = -1;

        run_outer_loop([&](const OuterloopArgs& args) {
            smem.bar_tQ_empty[0].wait(args.outer_loop_phase ^ 1);
            ptx::tcgen05_after_thread_sync();

            if constexpr (SPLIT_O_STAGING) {
                if (warp_idx == 0 && ptx::elect_one_sync()) {
                    ptx::tma_store_wait<0>();
                    ptx::SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
                        &tma_params.tensor_map_q, (uint64_t*)&smem.bar_sQ_full[0],
                        ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_FIRST>(),
                        smem.Q[0].data(), 0, 0, 0, 0, 2 * args.s_q_idx + cta_idx);
                    if (cta_idx == 0) {
                        smem.bar_sQ_full[0].arrive_and_expect_tx(H_Q * D_Q * sizeof(fp8_e4m3));
                    }
                }
            }
            {
                const int head_idx = lane_idx;
                const int q_token_idx = 2 * args.s_q_idx + cta_idx;
                const int q_load_token_idx = min(q_token_idx, params.s_q - 1);
                const uint8_t* q_scale_base = reinterpret_cast<const uint8_t*>(params.q) +
                                              q_load_token_idx * params.stride_q_s_q + D_Q;
                uint4 packed_scale_lo, packed_scale_hi;
                LDG_128(q_scale_base + head_idx * params.stride_q_h_q, &packed_scale_lo, ".nc",
                        "evict_first", "128B");
                LDG_128(q_scale_base + (head_idx + 32) * params.stride_q_h_q, &packed_scale_hi,
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
                ptx::tmem_st_32dp32bNx<8>(tmem_cols::Q_scale + warp_idx * 32 * 65536,
                                          q_scale_words);
                ptx::fence_view_async_tmem_store();
                ptx::tcgen05_before_thread_sync();
                ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                if (warp_idx == 0 && ptx::elect_one_sync()) {
                    if (cta_idx == 0) {
                        smem.bar_Q_scale_ready[0].arrive();
                    } else {
                        smem.bar_Q_scale_ready[0].arrive(uint32_t{0}, 1u);
                    }
                }
            }

            if constexpr (SPLIT_O_STAGING) {
                if (warp_idx == 0 && cta_idx == 0 && ptx::elect_one_sync()) {
                    smem.bar_sQ_full[0].wait(args.outer_loop_phase);
                    ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_tQ_full[0], 1 | 2);
                }
            } else {
                // Copy Q for this round

                if (warp_idx == 0 && ptx::elect_one_sync()) {
                    // Wait for sQ to become empty, and issue G -> S copy for Q
                    ptx::tma_store_wait<0>(); // This thread must be the same one as o copy out
                                              // thread (since `elect_one_sync()` always returns
                                              // the same thread for the same `mask`, according
                                              // to PTX document)

                    int q_token_idx = 2 * args.s_q_idx + cta_idx;
                    // A 2-SM operation cannot drop the second CTA.  Clamp the odd
                    // tail CTA to the final real token so it receives a valid Q
                    // tile and can participate in every transaction and barrier.
                    int q_load_token_idx = min(q_token_idx, params.s_q - 1);
                    ptx::SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
                        &tma_params.tensor_map_q, (uint64_t*)&smem.bar_sQ_full[0],
                        ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_FIRST>(),
                        smem.Q[0].data(), 0, 0, 0, 0, q_load_token_idx);

                    // Both CTAs contribute their native E4M3 Q tile. The SS
                    // operand stays in SMEM until the last QK MMA releases it.
                    if (cta_idx == 0) {
                        smem.bar_sQ_full[0].arrive_and_expect_tx(H_Q * D_Q * sizeof(fp8_e4m3));
                        smem.bar_sQ_full[0].wait(args.outer_loop_phase);
                        // Q remains in SMEM; the SS QK MMA consumes it directly.
                        ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_tQ_full[0], 1 | 2);
                    }
                }
            }

            if (last_args.batch_idx != -1) {

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
        // KV fetching threads.
        ptx::warpgroup_reg_dealloc<B_TOPK == 128 ? 128 : 80>();
        mxdsa::RingBufferState rs;

        const int warp_idx =
            ptx::canonical_warp_idx(); // Using `warp_idx` without `__shfl_sync` is faster
        if (ptx::elect_one_sync()) {
            // KV fetching threads
            // Keep the short-topk loop free of loop-carried prefetch
            // state; merely disabling its loads changed its codegen.
            if constexpr (PREFETCH_INDICES) {
                run_outer_loop([&](const OuterloopArgs& args) {
                    int* gIndices = params.indices + args.s_q_idx * params.stride_indices_s_q;
                    int64_t cache_hint =
                        ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_LAST>();

                    static constexpr int NUM_ROWS_PER_THREAD = B_TOPK / 4;
                    auto load_indices = [&](int k, int (&dest)[NUM_ROWS_PER_THREAD]) {
                        UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD / 8;
                             local_row += 1) {
                            int row = local_row * (4 * 8) + (warp_idx - 4) * 8;
                            // Each vector has eight indices; topk is a multiple of 64.
                            // Never read into the following query's index row.
                            if (local_row >= NUM_ROWS_PER_THREAD / 16 &&
                                k * B_TOPK + 64 >= params.topk) {
                                UNROLL
                                for (int i = 0; i < 8; ++i)
                                    dest[local_row * 8 + i] = -1;
                                continue;
                            }
                            // Let N128 metadata readers reuse these index lines in L1.
                            // Keep low retention priority for the streaming gather path.
                            LDG_256(gIndices + k * B_TOPK + row, dest + local_row * 8, ".nc",
                                    "evict_first", "evict_first", "256B");
                        }
                    };
                    int cur_indices[NUM_ROWS_PER_THREAD];
                    load_indices(args.start_block_idx, cur_indices);

                    NO_UNROLL
                    for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                        auto [k_buf_idx, k_bar_phase] = rs.template get<NUM_K_BUFS>();
                        int next_indices[NUM_ROWS_PER_THREAD];
                        const bool has_next = k + 1 < args.end_block_idx;
                        // Hide the next indices load behind this tile's
                        // slot wait and QK/V gather instructions.
                        if (has_next)
                            load_indices(k + 1, next_indices);
                        // QK completion releases only K; V remains live until
                        // KV-empty. Their separate buffers can be filled separately.
                        smem.bar_QK_done[k_buf_idx].wait(k_bar_phase ^ 1);
                        if (warp_idx == 4 && cta_idx == 0) {
                            smem.bar_K_full[k_buf_idx].arrive_and_expect_tx(B_TOPK * D_K *
                                                                            sizeof(fp8_e4m3));
                        }

                        // Native QK partitions K by tokens: B_TOPK/2 full rows per CTA.
                        UNROLL
                        for (int r = 0; r < NUM_ROWS_PER_THREAD / 2; r += 4) {
                            int row = (warp_idx - 4) * 8 + (r / 8) * 32 + r % 8;
                            int4 indices = cta_idx == 0
                                               ? *reinterpret_cast<int4*>(cur_indices + r)
                                               : *reinterpret_cast<int4*>(
                                                     cur_indices + NUM_ROWS_PER_THREAD / 2 + r);
                            UNROLL
                            for (int col = 0; col < 4; ++col) {
                                ptx::tma_gather4_cta_group_2<true>(
                                    &tma_params.tensor_map_kv, smem.bar_K_full[k_buf_idx],
                                    smem.K[k_buf_idx].data() + row * 128 + col * 128 * (B_TOPK / 2),
                                    col * 128, indices, cache_hint);
                            }
                        }
                        smem.bar_V_empty[k_buf_idx].wait(k_bar_phase ^ 1);
                        if (warp_idx == 4 && cta_idx == 1) {
                            smem.bar_V_full[k_buf_idx].arrive_and_expect_tx(B_TOPK * D_V *
                                                                            sizeof(fp8_e4m3));
                        }
                        UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD / 4;
                             local_row += 1) {
                            int row = (warp_idx - 4) * 8 + (local_row / 2) * (4 * 8) +
                                      (local_row % 2) * 4;
                            int4 indices = *(int4*)(cur_indices + local_row * 4);
                            static_assert(D_K == 512);
                            UNROLL
                            for (int local_col = 0; local_col < (D_K / 128) / 2; ++local_col) {
                                ptx::tma_gather4_cta1(
                                    &tma_params.tensor_map_kv, smem.bar_V_full[k_buf_idx],
                                    smem.V[k_buf_idx].data() + row * 128 + local_col * 128 * B_TOPK,
                                    local_col * 128 + cta_idx * (D_K / 2), indices, cache_hint);
                            }
                        }
                        if (has_next) {
                            UNROLL
                            for (int i = 0; i < NUM_ROWS_PER_THREAD; ++i) {
                                cur_indices[i] = next_indices[i];
                            }
                        }
                        rs.update();
                    }
                });
            } else {
                run_outer_loop([&](const OuterloopArgs& args) {
                    int* gIndices = params.indices + args.s_q_idx * params.stride_indices_s_q;
                    int64_t cache_hint =
                        ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_LAST>();

                    static constexpr int NUM_ROWS_PER_THREAD = B_TOPK / 4;

                    NO_UNROLL
                    for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                        auto [k_buf_idx, k_bar_phase] = rs.template get<NUM_K_BUFS>();

                        int cur_indices[NUM_ROWS_PER_THREAD];
                        UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD / 8;
                             local_row += 1) {
                            int row = local_row * (4 * 8) + (warp_idx - 4) * 8;
                            LDG_256(gIndices + k * B_TOPK + row, cur_indices + local_row * 8, ".nc",
                                    "no_allocate", "evict_first", "256B");
                        }
                        smem.bar_V_empty[k_buf_idx].wait(k_bar_phase ^ 1);

                        // Native QK partitions K by tokens: B_TOPK/2 full rows per CTA.
                        UNROLL
                        for (int r = 0; r < NUM_ROWS_PER_THREAD / 2; r += 4) {
                            int row = (warp_idx - 4) * 8 + (r / 8) * 32 + r % 8;
                            int4 indices = cta_idx == 0
                                               ? *reinterpret_cast<int4*>(cur_indices + r)
                                               : *reinterpret_cast<int4*>(
                                                     cur_indices + NUM_ROWS_PER_THREAD / 2 + r);
                            UNROLL
                            for (int col = 0; col < 4; ++col) {
                                ptx::tma_gather4_cta_group_2<true>(
                                    &tma_params.tensor_map_kv, smem.bar_K_full[k_buf_idx],
                                    smem.K[k_buf_idx].data() + row * 128 + col * 128 * (B_TOPK / 2),
                                    col * 128, indices, cache_hint);
                            }
                        }
                        UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD / 4;
                             local_row += 1) {
                            int row = (warp_idx - 4) * 8 + (local_row / 2) * (4 * 8) +
                                      (local_row % 2) * 4;
                            int4 indices = *(int4*)(cur_indices + local_row * 4);
                            static_assert(D_K == 512);
                            UNROLL
                            for (int local_col = 0; local_col < (D_K / 128) / 2; ++local_col) {
                                ptx::tma_gather4_cta1(
                                    &tma_params.tensor_map_kv, smem.bar_V_full[k_buf_idx],
                                    smem.V[k_buf_idx].data() + row * 128 + local_col * 128 * B_TOPK,
                                    local_col * 128 + cta_idx * (D_K / 2), indices, cache_hint);
                            }
                        }
                        rs.update();
                    }
                });
            }
        }
    } else if (warpgroup_idx == 2) {
        ptx::warpgroup_reg_dealloc<B_TOPK == 128 ? 64 : 80>();

        mxdsa::RingBufferState rs;
        // CTA1 issues SV independently; CTA0 issues QK.
        if (warp_idx == 8 && cta_idx == 1 && ptx::elect_one_sync()) {
            run_outer_loop([&](const OuterloopArgs& args) {
                auto issue_SV = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] =
                        rs.offset_by(rs_offset).template get<NUM_K_BUFS>();
                    auto [s_buf_idx, s_bar_phase] = rs.offset_by(rs_offset).template get<2>();
                    {
                        if constexpr (B_TOPK != 128)
                            smem.bar_V_full[k_buf_idx].arrive_and_expect_tx(B_TOPK * D_V *
                                                                            sizeof(fp8_e4m3));
                        smem.bar_V_full[k_buf_idx].wait(k_bar_phase);
                    }
                    if constexpr (B_TOPK == 128) {
                        smem.bar_S_O_full[s_buf_idx].wait(s_bar_phase);
                        if (k == args.start_block_idx)
                            smem.bar_tOut_empty.wait(args.outer_loop_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();
                        mxdsa::attention::issue_sv<B_TOPK>(
                            smem.S[s_buf_idx].data(), smem.V[k_buf_idx].data(), tmem_cols::O,
                            tmem_cols::S_scale + s_buf_idx * tmem_cols::S_scale_stride,
                            tmem_cols::V_scale, k == args.start_block_idx);
                        ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_S_empty[s_buf_idx],
                                                                 1 | 2);
                    } else {
                        smem.bar_S_O_full[s_buf_idx].wait(s_bar_phase);
                        if (k == args.start_block_idx) {
                            smem.bar_tOut_empty.wait(args.outer_loop_phase ^ 1);
                        }
                        ptx::tcgen05_after_thread_sync();
                        mxdsa::attention::issue_sv<B_TOPK>(
                            smem.S[s_buf_idx].data(), smem.V[k_buf_idx].data(), tmem_cols::O,
                            tmem_cols::S_scale + s_buf_idx * tmem_cols::S_scale_stride,
                            tmem_cols::V_scale, k == args.start_block_idx);
                        ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_S_empty[s_buf_idx],
                                                                 1 | 2);
                    }
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
                smem.bar_tQ_full[0].wait(args.outer_loop_phase);
                smem.bar_Q_scale_ready[0].wait(args.outer_loop_phase);
                ptx::tcgen05_after_thread_sync();

                auto issue_QK = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] =
                        rs.offset_by(rs_offset).template get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).template get<1>();
                    smem.bar_P_empty.wait(bar_phase ^ 1);
                    ptx::transac_bar_t& k_full = smem.bar_K_full[k_buf_idx];
                    if constexpr (B_TOPK != 128)
                        k_full.arrive_and_expect_tx(B_TOPK * D_K * sizeof(fp8_e4m3));
                    k_full.wait(k_bar_phase);
                    ptx::tcgen05_after_thread_sync();

                    smem.bar_K_scale_copy_ready[k_buf_idx].wait(k_bar_phase);
                    copy_k_scale_smem_to_tmem(smem.k_scale_mma[k_buf_idx].data(),
                                              tmem_cols::K_scale);
                    ptx::tcgen05_before_thread_sync();
                    mxdsa::attention::issue_qk<B_TOPK>(
                        smem.Q[0].data(), smem.K[k_buf_idx].data(), tmem_cols::P,
                        tmem_cols::Q_scale + (0), tmem_cols::K_scale);
                    ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_QK_done[k_buf_idx], 1 | 2);
                };

                NO_UNROLL
                for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                    issue_QK(k, 0);
                    rs.update();
                }
                ptx::tcgen05_before_thread_sync();
                ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_tQ_empty[0], 1 | 2);
            });
        } else if (warp_idx == 9) {
            // KV validity loading warp.
            if (!FUSED_VALIDITY && lane_idx < B_TOPK / 8) {
                run_outer_loop([&](const OuterloopArgs& args) {
                    int* gIndices = params.indices + args.s_q_idx * params.stride_indices_s_q;
                    NO_UNROLL
                    for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                        char k_validness_mask = load_indices_and_generate_mask(
                            lane_idx, gIndices + k * B_TOPK, params.s_kv, k * B_TOPK,
                            args.topk_length);

                        auto [indices_buf_idx, indices_bar_phase] =
                            rs.template get<NUM_INDEX_BUFS>();
                        smem.bar_valid_coord_scales_empty[indices_buf_idx].wait(indices_bar_phase ^
                                                                                1);
                        smem.is_k_valid[indices_buf_idx][lane_idx] = k_validness_mask;
                        smem.bar_valid_coord_scales_full[indices_buf_idx].arrive();

                        rs.update();
                    }
                });
            }
        } else if (warp_idx == 10 || warp_idx == 11) {
            // Each scale warp handles B_TOPK/2 tokens, in one or two
            // groups of 32. Scale words are packed for the duplicated
            // DP layout; each UE8M0 byte serves two K=32 MMAs.
            const int scale_warp = warp_idx - 10;
            mxdsa::RingBufferState scale_rs;
            run_outer_loop([&](const OuterloopArgs& args) {
                NO_UNROLL
                for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                    auto [k_buf_idx, k_bar_phase] = scale_rs.template get<NUM_K_BUFS>();
                    int tile_token[2];
                    uint2 tile_scale[2];
                    if constexpr (B_TOPK == 128) {
                        const int* tile_indices =
                            params.indices + args.s_q_idx * params.stride_indices_s_q + k * B_TOPK;
                        const uint8_t* scale_base =
                            reinterpret_cast<const uint8_t*>(params.kv) +
                            static_cast<int64_t>(params.s_kv) * params.h_kv * D_K;
                        UNROLL
                        for (int part = 0; part < 2; ++part) {
                            const int row = scale_warp * 64 + part * 32 + lane_idx;
                            if (k + 1 < args.end_block_idx &&
                                (scale_warp == 0 || (k + 1) * B_TOPK + 64 < params.topk)) {
                                ptx::prefetch_global_l1(tile_indices + B_TOPK + row);
                            }
                            tile_token[part] = scale_warp == 0 || k * B_TOPK + 64 < params.topk
                                                   ? __ldg(tile_indices + row)
                                                   : -1;
                        }
                        UNROLL
                        for (int part = 0; part < 2; ++part) {
                            const int token = tile_token[part];
                            const bool valid = token >= 0 && token < params.s_kv;
                            const uint2* source = reinterpret_cast<const uint2*>(
                                scale_base +
                                static_cast<int64_t>(valid ? token : 0) * K_SCALE_SLOT_BYTES);
                            tile_scale[part] = valid ? __ldg(source) : make_uint2(0, 0);
                        }
                    }
                    uint32_t first_valid_bits = 0;
                    UNROLL
                    for (int row_part = 0; row_part < B_TOPK / 64; ++row_part) {
                        const int row = scale_warp * (B_TOPK / 2) + row_part * 32 + lane_idx;
                        const int* g_indices =
                            params.indices + args.s_q_idx * params.stride_indices_s_q + k * B_TOPK;
                        const int token_idx =
                            B_TOPK == 128 ? tile_token[row_part] : __ldg(g_indices + row);
                        const bool token_valid = token_idx >= 0 && token_idx < params.s_kv;
                        const int safe_token_idx = token_valid ? token_idx : 0;
                        const uint8_t* k_scale_base =
                            reinterpret_cast<const uint8_t*>(params.kv) +
                            static_cast<int64_t>(params.s_kv) * params.h_kv * D_K;
                        const uint32_t* src = reinterpret_cast<const uint32_t*>(
                            k_scale_base +
                            static_cast<int64_t>(safe_token_idx) * K_SCALE_SLOT_BYTES);

                        // Fetch both unique UE8M0 words before waiting for
                        // the ring slot. Native N64 needs token-ordered rows:
                        // UTCCP replicates rows0..31 into DP warps0/1, and
                        // rows32..63 into DP warps2/3, in each CTA.
                        const uint2 packed =
                            B_TOPK == 128
                                ? tile_scale[row_part]
                                : (token_valid ? __ldg(reinterpret_cast<const uint2*>(src))
                                               : make_uint2(0, 0));
                        if constexpr (FUSED_VALIDITY) {
                            // N128 combines two ballots into one 64-bit mask per warp.
                            const uint32_t valid_bits = __ballot_sync(
                                0xffffffffu, token_valid && k * B_TOPK + row < args.topk_length);
                            if (row_part == 0) {
                                first_valid_bits = valid_bits;
                            } else if (lane_idx == 0) {
                                auto [index_buf_idx, index_bar_phase] =
                                    scale_rs.template get<NUM_INDEX_BUFS>();
                                smem.bar_valid_coord_scales_empty[index_buf_idx].wait(
                                    index_bar_phase ^ 1);
                                reinterpret_cast<uint64_t*>(
                                    smem.is_k_valid[index_buf_idx])[scale_warp] =
                                    uint64_t(first_valid_bits) | (uint64_t(valid_bits) << 32);
                                smem.bar_valid_coord_scales_full[index_buf_idx].arrive();
                            }
                        }
                        // Acquiring the whole old stage once covers both
                        // row groups; new consumers wait for the final ready.
                        if (row_part == 0) {
                            // Softmax releases V scales only after acquiring QK-done.
                            smem.bar_v_scale_empty[k_buf_idx].wait(k_bar_phase ^ 1);
                        }
                        if constexpr (B_TOPK == 128) {
                            uint32_t* native_scale =
                                reinterpret_cast<uint32_t*>(smem.k_scale_mma[k_buf_idx].data() +
                                                            (scale_warp * 32 + lane_idx) * 16);
                            if (row_part == 0) {
                                // Both row groups are loaded and the entire old stage is free.
                                const uint4 packed_rows =
                                    make_uint4(tile_scale[0].x, tile_scale[1].x, tile_scale[0].y,
                                               tile_scale[1].y);
                                ptx::st_shared(native_scale,
                                               *reinterpret_cast<const __int128_t*>(&packed_rows));
                            }
                        } else {
                            uint32_t* native_scale = reinterpret_cast<uint32_t*>(
                                smem.k_scale_mma[k_buf_idx].data() + row * 16);
                            native_scale[0] = native_scale[1] = packed.x;
                            native_scale[2] = native_scale[3] = packed.y;
                        }
                        fp8_e8m0 token_scale;
                        token_scale.storage = packed.x & 0xff;
                        smem.v_token_scale[k_buf_idx][row] = token_scale;
                    }
                    ptx::fence_view_async_shared();
                    __syncwarp();
                    smem.bar_v_scale_full[k_buf_idx].arrive();
                    if (ptx::elect_one_sync()) {
                        smem.bar_K_scale_copy_ready[k_buf_idx].arrive(0u);
                    }
                    scale_rs.update();
                }
                // Rejoin the prefetched scale producer before entering
                // the persistent task handoff. This also keeps racecheck
                // from stalling on masked queries with index prefetch.
                if constexpr (PREFETCH_INDICES)
                    __syncwarp();
                // The CLC producer must run inside this same outer-loop
                // body. A second run_outer_loop after scale completion
                // would wait for the very query it is supposed to issue.
                if (warp_idx == 10 && ptx::elect_one_sync()) {
                    if (cta_idx == 0) {
                        smem.bar_clc_empty.wait(args.outer_loop_phase ^ 1);
                        ptx::issue_clc_query_multicast_cluster_all(smem.bar_clc_full,
                                                                   smem.clc_response_obj);
                    }
                    smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));
                }
            });
        }
    } else {
        // Scale & Exp threads
        ptx::warpgroup_reg_alloc<176>();

        int local_warp_idx = warp_idx - 12;
        int s_row = idx_in_warpgroup % (H_Q / 2);
        int s_col_base = (local_warp_idx >= 2 ? B_TOPK / 2 : 0);

        mxdsa::RingBufferState rs;
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
                using ValidMask = std::conditional_t<B_TOPK == 128, uint64_t, uint32_t>;
                ValidMask valid;
                if constexpr (FUSED_VALIDITY) {
                    // Complete the mask read before allowing the producer
                    // to reuse its shared slot, even while QK is pending.
                    valid = ptx::ld_shared_acquire_u64(
                        reinterpret_cast<ValidMask*>(smem.is_k_valid[indices_buf_idx]) +
                        local_warp_idx / 2);
                    smem.bar_valid_coord_scales_empty[indices_buf_idx].arrive();
                }
                smem.bar_QK_done[k_buf_idx].wait(k_bar_phase);
                ptx::tcgen05_after_thread_sync();
                {
                    // Each warp reads its native QK logits without a partial-D
                    // exchange: 32 values for N64 and 64 for N128.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1030 && __CUDA_ARCH__ < 1100
                    if constexpr (B_TOPK == 128) {
                        float max0, max1;
                        ptx::tmem_ld_32dp32b32x_max(tmem_cols::P,
                                                    *reinterpret_cast<float (*)[32]>(p), max0);
                        ptx::tmem_ld_32dp32b32x_max(tmem_cols::P + 32,
                                                    *reinterpret_cast<float (*)[32]>(p + 32), max1);
                        p_max = max(max0, max1);
                    } else {
                        ptx::tmem_ld_32dp32b32x_max(tmem_cols::P, p, p_max);
                    }
#else
                    ptx::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P, p);
#endif
                    ptx::fence_view_async_tmem_load();
                    ptx::tcgen05_before_thread_sync();
                    smem.bar_P_empty.arrive(0u);
                    if constexpr (!FUSED_VALIDITY) {
                        valid = reinterpret_cast<uint32_t*>(
                            smem.is_k_valid[indices_buf_idx])[local_warp_idx / 2];
                    }
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
                    if (valid != static_cast<ValidMask>(-1))
                        p_max = get_max<NUM_ELEMS_PER_THREAD>(p);
#else
                    p_max = get_max<NUM_ELEMS_PER_THREAD>(p);
#endif
                }

                if constexpr (B_TOPK == 128) {
                    const float own_max = p_max * params.sm_scale_div_log2;
                    smem.rowwise_max_buf[idx_in_warpgroup] = own_max;
                    ptx::NamedBarrier::arrive_and_wait(64, barrier_ids::WG2_WARP02_SYNC +
                                                               (local_warp_idx & 1));
                    const float peer_max = smem.rowwise_max_buf[idx_in_warpgroup ^ 64];
                    const float tile_max = max(own_max, peer_max);
                    real_mi = max(real_mi, tile_max);
                    const bool scale_o = __any_sync(0xffffffff, tile_max - mi > 6.0f);
                    const float new_max = scale_o ? max(mi, tile_max) : mi;
                    const float scale_old = scale_o ? exp2f(mi - new_max) : 1.0f;
                    mi = new_max;
                    fp8_e4m3 s[64];
                    uint8_t scale_bits[2];
                    float2 sums[4] = {};
                    smem.bar_v_scale_full[k_buf_idx].wait(k_bar_phase);
                    UNROLL
                    for (int group = 0; group < 2; ++group) {
                        float2 maxima = {};
                        UNROLL
                        for (int i = 0; i < 32; i += 2) {
                            const int col = group * 32 + i;
                            const int acc = (i / 2) & 3;
                            float2 logits = ptx::float2_fma(
                                *reinterpret_cast<const float2*>(p + col),
                                make_float2(params.sm_scale_div_log2, params.sm_scale_div_log2),
                                make_float2(-new_max, -new_max));
                            float2 value = make_float2(exp2f(logits.x), exp2f(logits.y));
                            sums[acc] = ptx::float2_add(sums[acc], value);
                            float2 scaled = ptx::float2_mul(
                                value, *reinterpret_cast<const float2*>(
                                           &smem.v_token_scale[k_buf_idx][s_col_base + col]));
                            maxima.x = max(maxima.x, scaled.x);
                            maxima.y = max(maxima.y, scaled.y);
                            *reinterpret_cast<float2*>(p + col) = scaled;
                        }
                        float abs_max = max(maxima.x, maxima.y);
                        fp8_e8m0 exponent(abs_max > 0.0f ? abs_max / 448.0f : 1.0f);
                        exponent.storage = max(exponent.storage, uint8_t{61});
                        scale_bits[group] = exponent.storage;
                        float scale = __uint_as_float(uint32_t(254u - exponent.storage) << 23);
                        UNROLL
                        for (int i = 0; i < 32; i += 2) {
                            const int col = group * 32 + i;
                            float2 scaled =
                                ptx::float2_mul(*reinterpret_cast<const float2*>(p + col),
                                                make_float2(scale, scale));
                            *reinterpret_cast<__nv_fp8x2_storage_t*>(s + col) =
                                __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
                        }
                    }
                    smem.bar_v_scale_empty[k_buf_idx].arrive();
                    sums[0] = ptx::float2_add(sums[0], sums[1]);
                    sums[2] = ptx::float2_add(sums[2], sums[3]);
                    sums[0] = ptx::float2_add(sums[0], sums[2]);
                    const float sum = sums[0].x + sums[0].y;
                    li = fmaf(li, scale_old, sum);
                    smem.bar_S_empty[s_buf_idx].wait(s_bar_phase ^ 1);
                    // Each half owns one aligned 16-bit field of this four-byte row.
                    const uint16_t own_pair =
                        uint16_t(scale_bits[0]) | (uint16_t(scale_bits[1]) << 8);
                    *reinterpret_cast<uint16_t*>(
                        &smem.s_scale_exp[k_buf_idx][s_row][s_col_base / 32]) = own_pair;
                    ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::S_SCALE_SYNC);
                    const uint32_t all_scales =
                        *reinterpret_cast<const uint32_t*>(smem.s_scale_exp[k_buf_idx][s_row]);
                    const uint32_t dp_addr = local_warp_idx * 32 * 65536;
                    UNROLL
                    for (int row_group = 0; row_group < 4; ++row_group) {
                        ptx::SM100_TMEM_STORE_32dp32b1x::copy(
                            all_scales, tmem_cols::S_scale + s_buf_idx * tmem_cols::S_scale_stride +
                                            dp_addr + row_group);
                    }
                    ptx::fence_view_async_tmem_store();

                    UNROLL
                    for (int i = 0; i < 4; ++i) {
                        ptx::st_shared(mxdsa::attention::k_major_ptr<64, 64>(
                                           smem.S[s_buf_idx].data(), s_row, s_col_base + i * 16),
                                       *reinterpret_cast<__int128_t*>(s + i * 16));
                    }
                    if (k > args.start_block_idx && scale_o) {
                        // A whole-tile S ring frees t-2 here, so explicitly
                        // wait for the immediately preceding PV before O RMW.
                        auto [prev_s_buf, prev_s_phase] = rs.offset_by(-1).template get<2>();
                        smem.bar_S_empty[prev_s_buf].wait(prev_s_phase);
                        ptx::tcgen05_after_thread_sync();
                        rescale_O<D_V, 32, tmem_cols::O>(scale_old);
                        ptx::tcgen05_before_thread_sync();
                    }
                    ptx::fence_view_async_shared();
                    smem.bar_S_O_full[s_buf_idx].arrive(1u);
                } else {
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
                        float2 scaled_s =
                            ptx::float2_mul(s_value, *reinterpret_cast<const float2*>(
                                                         &smem.v_token_scale[k_buf_idx][token]));
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
                    float s_scale =
                        __uint_as_float(uint32_t(254u - s_scale_exp_e8m0.storage) << 23);
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
                    store_s_scale_to_tmem(tmem_cols::S_scale +
                                              s_buf_idx * tmem_cols::S_scale_stride,
                                          local_warp_idx, smem.s_scale_exp[k_buf_idx][s_row][0],
                                          smem.s_scale_exp[k_buf_idx][s_row][1]);
                    ptx::fence_view_async_tmem_store();
                    UNROLL
                    for (int i = 0; i < NUM_ELEMS_PER_THREAD / 16; ++i) {
                        fp8_e4m3* sS_base = mxdsa::attention::k_major_ptr<64, 64>(
                            smem.S[s_buf_idx].data(), s_row, s_col_base + i * 16);
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
                    if constexpr (!FUSED_VALIDITY)
                        smem.bar_valid_coord_scales_empty[indices_buf_idx].arrive();
                }

                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                // real_mi == -CUDART_INF_F <=> No valid TopK indices
                // We set li to 0 to fit the definition that li := exp(x[i] - mi)
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // Reduce li
            smem.bar_li_empty.wait(args.outer_loop_phase ^ 1);
            smem.rowwise_li_buf[idx_in_warpgroup ^ 64] = li;
            ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG2_SYNC);
            li += smem.rowwise_li_buf[idx_in_warpgroup];

            if (idx_in_warpgroup < H_Q / 2) {
                {
                    // Calculate output_scale and save
                    int head_idx = idx_in_warpgroup;
                    float attn_sink = params.attn_sink == nullptr
                                          ? -CUDART_INF_F
                                          : __ldg(params.attn_sink + head_idx);
                    float output_scale =
                        __fdividef(1.0f, li + exp2f(fmaf(attn_sink, CUDART_L2E_F, -mi)));
                    smem.rowwise_li_buf[idx_in_warpgroup] = li == 0.0f ? 0.0f : output_scale;
                    smem.bar_li_full.arrive();
                }
                const int head_idx = idx_in_warpgroup;

                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                int q_token_idx = 2 * args.s_q_idx + cta_idx;
                int global_index = q_token_idx * params.h_q + head_idx;
                params.max_logits[global_index] = real_mi * CUDART_LN2_F;
                params.lse[global_index] = cur_lse;
            }
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
    dual_mxfp8_sparse_attn_fwd_kernel(
        __grid_constant__ const typename Kernel::ArgT params,
        __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template <SparseAttnFwdMode FWD_MODE, int D_QK, int PREFILL_B_TOPK>
void KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>::run(const ArgT& params)
    requires(!IS_DECODE)
{
    static_assert(D_QK == 512);

    ASSERT(params.h_kv == 1);
    ASSERT(params.topk % 64 == 0);
    ASSERT(params.h_q == TOKEN_H_Q);
    ASSERT(params.s_q > 0);
    ASSERT(params.s_q % 2 == 0);
    ASSERT(params.d_qk == D_QK);

    static_assert(D_Q == 512);
    ASSERT(params.stride_q_h_q == Q_BYTES_PER_HEAD,
           "Q head stride must be 512 e4m3 bytes + an aligned 16-byte scale slot");
    ASSERT(params.stride_kv_h_kv == KV_BYTES_PER_TOKEN,
           "KV token envelope must be 512 e4m3 bytes + a replicated 16-byte scale slot");
    ASSERT(params.stride_kv_s_kv == params.h_kv * KV_BYTES_PER_TOKEN,
           "packed KV storage must be contiguous across logical token envelopes");
    ASSERT(reinterpret_cast<int64_t>(params.q) % 16 == 0);
    ASSERT(reinterpret_cast<int64_t>(params.kv) % 16 == 0);

    CUtensorMap tensor_map_q = mxdsa::make_tensor_map(
        {128ul, TOKEN_H_Q, 2ul, (D_Q / 128ul) / 2ul, (unsigned long)params.s_q},
        mxdsa::make_stride_helper<int>({params.stride_q_h_q, 128, D_Q / 2, params.stride_q_s_q},
                                       sizeof(fp8_e4m3)),
        {128, TOKEN_H_Q, 2, (D_Q / 128) / 2, 1}, params.q, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B); // Native QK's logical layout applies the packed-Q
                                             // K32 permutation.

    CUtensorMap tensor_map_kv;
    [[maybe_unused]] CUtensorMap tensor_map_k_scale = {};
    // The storage tensor is exposed as 528-byte logical envelopes, but
    // inside the packed page all 512-byte E4M3 rows are contiguous and
    // the replicated 16-byte scale slots form a separate tail plane.
    tensor_map_kv =
        mxdsa::make_tensor_map({D_QK, (unsigned long)params.s_kv}, {(unsigned long)D_K}, {128, 1},
                               params.kv, CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    // The scale tail contains one replicated 16B UE8M0 slot per K token.

    CUtensorMap tensor_map_o;
    if constexpr (SPLIT_O_STAGING) {
        // Pair output segments separated by D_V/2. Each store covers 64 or
        // 128 dimensions per side, according to the selected staging capacity.
        tensor_map_o = mxdsa::make_tensor_map(
            {64, TOKEN_H_Q, D_V / 128, 2ul, (unsigned long)params.s_q},
            mxdsa::make_stride_helper<int>({D_V, 64, D_V / 2, TOKEN_H_Q * D_V}, sizeof(ptx::bf16)),
            {64, TOKEN_H_Q, O_STAGE_D / 128, 2, 1}, params.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    } else {
        tensor_map_o = mxdsa::make_tensor_map(
            {64, TOKEN_H_Q, D_V / 64, (unsigned long)params.s_q, 1ul},
            mxdsa::make_stride_helper<int>({D_V, 64, TOKEN_H_Q * D_V, TOKEN_H_Q * D_V},
                                           sizeof(ptx::bf16)),
            {64, TOKEN_H_Q, D_V / 64, 1, 1}, params.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    }

    TmaParams tma_params = {tensor_map_q, tensor_map_kv, tensor_map_k_scale, tensor_map_o};

    auto kernel =
        &dual_mxfp8_sparse_attn_fwd_kernel<KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_shape;
    // grid_shape counts CTAs, while each logical query pair is one
    // two-CTA cluster.
    grid_shape = dim3(params.s_q, 1, 1);

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t config{grid_shape, dim3(NUM_THREADS, 1, 1), smem_size, params.stream, &attr,
                              1};
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params, tma_params));
}

} // namespace mxdsa::attention
