#pragma once
#include "attention/phase1.h"

#include <math_constants.h>

#include "common/params.h"
#include "common/utils.h"
#include "attention/common_subroutine.h"

#include "attention/config.h"
#include "common/host.h"
#include "ptx/ptx.cuh"
#include "ptx/ptx_copy.cuh"
#include "ptx/ptx_extra.cuh"
#include "ptx/ptx_tmem.cuh"

namespace sm100::dual_mxfp8::head64 {

using FwdMode = SparseAttnFwdMode;

// Route completion to CTA1, which issues SV in both prefill and decode.
DEVICE void tma_gather4_cta1(const void* desc_ptr, ptx::transac_bar_t& mbar, void* dst, int col,
                             int4 rows, int64_t hint) {
    uint32_t dst_addr = ptx::cast_smem_ptr_to_uint(dst);
    uint32_t bar_addr = (ptx::cast_smem_ptr_to_uint(&mbar) & ptx::Sm100MmaPeerBitMask) |
                        uint32_t(~ptx::Sm100MmaPeerBitMask);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::
            "r"(dst_addr),
        "l"(desc_ptr), "r"(col), "r"(rows.x), "r"(rows.y), "r"(rows.z), "r"(rows.w), "r"(bar_addr),
        "l"(hint)
        : "memory");
}

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

// SM103 can load logits and compute a per-lane max in one instruction.
// SM100 keeps the ordinary load/reduction path below.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1030 && __CUDA_ARCH__ < 1100
DEVICE void load_native_p_and_max(uint32_t tmem_col, float (&p)[32], float& p_max) {
    asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7, "
                 "%8, %9, %10, %11, %12, %13, %14, %15, "
                 "%16, %17, %18, %19, %20, %21, %22, %23, "
                 "%24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                 : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]), "=f"(p[4]), "=f"(p[5]),
                   "=f"(p[6]), "=f"(p[7]), "=f"(p[8]), "=f"(p[9]), "=f"(p[10]), "=f"(p[11]),
                   "=f"(p[12]), "=f"(p[13]), "=f"(p[14]), "=f"(p[15]), "=f"(p[16]), "=f"(p[17]),
                   "=f"(p[18]), "=f"(p[19]), "=f"(p[20]), "=f"(p[21]), "=f"(p[22]), "=f"(p[23]),
                   "=f"(p[24]), "=f"(p[25]), "=f"(p[26]), "=f"(p[27]), "=f"(p[28]), "=f"(p[29]),
                   "=f"(p[30]), "=f"(p[31]), "=f"(p_max)
                 : "r"(tmem_col));
}
#endif

template <FwdMode FWD_MODE, int D_QK, int PREFILL_B_TOPK>
__device__ void KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>::sparse_attn_fwd_kernel_devfunc(
    const ArgT& params, const TmaParams& tma_params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Each 2-CTA cluster handles one adjacent query-token pair, and each CTA
    // owns one native [64, D_Q] tile.  For an odd decode s_q, CTA 1 in the
    // final cluster uses the last real Q as a dummy operand and skips stores.
    // Cluster shape: [2, 1, 1]
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
        for (int slot = 0; slot < (IS_DECODE ? 2 : 1); ++slot) {
            smem.bar_sQ_full[slot].init(1);
            if constexpr (IS_DECODE)
                smem.bar_raw_indices_full[slot].init(1);
            // CTA0's barrier gathers the two warpgroup-0 Q-scale producers before
            // CTA0 issues the single cta_group::2 UTCCP operation.
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
        smem.bar_li_full.init(IS_DECODE ? 128 : H_Q / 2);
        smem.bar_li_empty.init(128);
        if constexpr (FWD_MODE != FwdMode::DecodeWithSplitKV) {
            smem.bar_clc_full.init(1);
            smem.bar_clc_empty.init(NUM_WORKER_THREADS);
        }
        ptx::fence_barrier_init();
    } else if (warp_idx == 3 && ptx::elect_one_sync()) {
        UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
            // TMA completion contributes transaction bytes. Tuned N128 prefill paths
            // use warp4's elected producer (CTA0 for K, CTA1 for V); other
            // paths use the MMA consumer. Direct E4M3 gather does not add the
            // historical decode dequant path's extra arrivals.
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
            smem.bar_valid_coord_scales_full[i].init(!IS_DECODE ? (B_TOPK == 128 ? 2 : B_TOPK / 8)
                                                                : 32);
            // Decode has four elected gather producers, 128 softmax
            // consumers, and 64 scale readers. Count scale reads explicitly
            // before allowing the index producer to reuse this stage.
            smem.bar_valid_coord_scales_empty[i].init(!IS_DECODE ? 128 : 128 + 4 + 64);
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

        int extra_topk_length, num_orig_kv_blocks; // extra-KV related
        bool is_no_split;
        int n_split_idx; // splitkv related
    };

    int decode_q_phase = 0;
    auto run_outer_loop = [&](auto loop_body) -> bool {
        int outer_loop_phase = false;
        int iteration = 0;
        if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
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
                int total_topk_padded =
                    orig_topk_padded + ptx::ceil(extra_topk_length, (int)B_TOPK); // % B_TOPK == 0
                int start_block_idx =
                    batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
                int end_block_idx = batch_idx == sched_meta.end_req_idx
                                        ? sched_meta.end_block_idx
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

                // start_block_idx = 0;
                // end_block_idx = total_topk_padded / B_TOPK;
                // is_split = false;
                // n_split_idx = 0;

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

                if constexpr (IS_DECODE)
                    decode_q_phase = (iteration / 2) & 1;
                loop_body(args);
                if constexpr (IS_DECODE)
                    ++iteration;
                outer_loop_phase ^= 1;
            }
        } else {
            // Prefill mode. Use CLC to allocate different s_q (for decoding, different batches +
            // s_q) to different workers
            ptx::CLCResult next_job = {true, (int)blockIdx.x, !IS_DECODE ? 0 : (int)blockIdx.y, 0};
            NO_UNROLL
            while (next_job.is_valid) {
                int s_q_idx = next_job.x / 2; // pair index for this head64 prefill kernel
                int batch_idx = !IS_DECODE ? 0 : next_job.y;
                int topk_length =
                    params.topk_length != nullptr
                        ? __ldg(params.topk_length + (!IS_DECODE ? s_q_idx : batch_idx))
                        : params.topk;

                if constexpr (!IS_DECODE) {
                    int num_k_blocks =
                        max(ptx::ceil_div(topk_length, (int)B_TOPK), 1); // num_k_blocks always >= 1
                    OuterloopArgs args = {(bool)outer_loop_phase, 0,          s_q_idx, 0,
                                          num_k_blocks,           topk_length};
                    if constexpr (IS_DECODE)
                        decode_q_phase = (iteration / 2) & 1;
                    loop_body(args);
                } else {
                    int orig_topk_padded = max(ptx::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
                    int extra_topk_length = params.extra_topk_length
                                                ? __ldg(params.extra_topk_length + batch_idx)
                                                : params.extra_topk;
                    int total_topk_padded =
                        orig_topk_padded +
                        ptx::ceil(extra_topk_length, (int)B_TOPK); // % B_TOPK == 0

                    OuterloopArgs args = {(bool)outer_loop_phase,
                                          batch_idx,
                                          s_q_idx,
                                          0,
                                          total_topk_padded / B_TOPK,
                                          topk_length,

                                          extra_topk_length,
                                          orig_topk_padded / B_TOPK,
                                          false,
                                          0};
                    if constexpr (IS_DECODE)
                        decode_q_phase = (iteration / 2) & 1;
                    loop_body(args);
                }

                smem.bar_clc_full.wait(outer_loop_phase);
                next_job = ptx::get_clc_query_response<true>(smem.clc_response_obj);
                smem.bar_clc_empty.arrive(0u);

                if constexpr (IS_DECODE)
                    ++iteration;
                outer_loop_phase ^= 1;
            }
        }
        return outer_loop_phase;
    };

    if (warpgroup_idx == 0) {
        // Q fetching and O writing back warpgroup
        ptx::warpgroup_reg_alloc<!IS_DECODE && B_TOPK == 128 ? 144 : 176>();
        const uint32_t warp_dp_addr = tmem_cols::V_scale + warp_idx * 32 * 65536;
        if constexpr (IS_DECODE) {
            uint32_t initial_w_words[8];
            UNROLL
            for (int col = 0; col < 8; ++col)
                initial_w_words[col] = warp_idx < 2 ? params.w1[col] : params.w2[col];
            ptx::tmem_st_32dp32bNx<8>(warp_dp_addr, initial_w_words);
        } else {
            UNROLL
            for (int col = 0; col < 8; ++col) {
                ptx::SM100_TMEM_STORE_32dp32b1x::copy(
                    warp_idx < 2 ? params.w1[col] : params.w2[col], warp_dp_addr + col);
            }
        }
        ptx::fence_view_async_tmem_store();

        auto perform_o_copy_out = [&](const OuterloopArgs& args, bool is_last_o) {
            // This query's last SV completion implies that all its Q reads
            // have completed. The next query uses the other independent slot.
            ptx::bf16* o_storage;
            if constexpr (IS_DECODE)
                o_storage = reinterpret_cast<ptx::bf16*>(smem.Q[args.outer_loop_phase].data());
            else
                o_storage = smem.O.data();
            ptx::bf16* sO_addrs[B_EPI / 8];
            UNROLL
            for (int i = 0; i < B_EPI / 8; ++i) {

                sO_addrs[i] =
                    ptx::k_major_ptr<64, 128>(o_storage, idx_in_warpgroup % 64,
                                              (idx_in_warpgroup / 64) * (O_STAGE_D / 2) + i * 8);
            }

            float* sO_accum_addrs[B_EPI_SPLITKV / 4];
            if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
                // If split-KV is enabled, we need to store back O in float32
                // View the output staging buffer as four FP32 split-KV tiles.

                UNROLL
                for (int i = 0; i < B_EPI_SPLITKV / 4; ++i) {
                    sO_accum_addrs[i] =
                        ptx::k_major_ptr<64, 128>((float*)o_storage, idx_in_warpgroup % 64, i * 4) +
                        (idx_in_warpgroup >= 64 ? (H_Q / 2) * B_EPI_SPLITKV : 0);
                }
            }

            // outer_loop_phase is the loop phase corresponding to s_q_idx

            // Get li (output_scale actually)
            smem.bar_li_full.wait(args.outer_loop_phase);
            float output_scale;
            if constexpr (IS_DECODE) {
                const int head_idx = idx_in_warpgroup % 64;
                // Preserve the original lower-half + upper-half addition order.
                const float final_li =
                    smem.rowwise_li_buf[head_idx] + smem.rowwise_li_buf[head_idx + 64];
                const float final_mi = smem.final_mi[head_idx];
                smem.bar_li_empty.arrive();
                const float attn_sink = params.attn_sink == nullptr
                                            ? -CUDART_INF_F
                                            : __ldg(params.attn_sink + head_idx);
                if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split)
                    output_scale = __fdividef(
                        1.0f, final_li + exp2f(fmaf(attn_sink, CUDART_L2E_F, -final_mi)));
                else
                    output_scale = __fdividef(1.0f, final_li);
                output_scale = final_li == 0.0f ? 0.0f : output_scale;
                if (idx_in_warpgroup < H_Q / 2) {
                    const int head_idx = idx_in_warpgroup;
                    const int q_token_idx = 2 * args.s_q_idx + cta_idx;
                    float cur_lse = fmaf(final_mi, CUDART_LN2_F, logf(final_li));
                    cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                    if (q_token_idx < params.s_q) {
                        if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
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
            } else {
                output_scale = smem.rowwise_li_buf[idx_in_warpgroup % 64];
                smem.bar_li_empty.arrive();
            }
            float2 output_scale_float2 = float2{output_scale, output_scale};

            // Retrieve and store O, and calculate delta := sum(O*dO, dim=-1) if FWD_MODE is
            // Recompute
            smem.bar_tOut_full.wait(args.outer_loop_phase);
            if constexpr (IS_DECODE)
                ptx::tcgen05_after_thread_sync();
            if (is_last_o && ptx::elect_one_sync()) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                if constexpr (SPLIT_O_STAGING) {
                    // Keep the first half in packed BF16 registers while
                    // loading the second half. This releases O TMEM before
                    // waiting to reuse the previous TMA store's SMEM buffer.
                    UNROLL
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        float2 first[B_EPI / 2];
                        nv_bfloat162 first_bf16[B_EPI / 2];
                        ptx::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + chunk * 2 * B_EPI, first);
                        ptx::fence_view_async_tmem_load();
                        UNROLL
                        for (int i = 0; i < B_EPI / 2; ++i) {
                            first_bf16[i] = __float22bfloat162_rn(
                                ptx::float2_mul(first[i], output_scale_float2));
                        }

                        float2 second[B_EPI / 2];
                        ptx::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + (chunk * 2 + 1) * B_EPI,
                                                      second);
                        ptx::fence_view_async_tmem_load();
                        if (chunk == 1) {
                            smem.bar_tOut_empty.arrive(1u);
                            // Only the issuing lane waits for its TMA store;
                            // all writers then observe completion together.
                            if (warp_idx == 0 && ptx::elect_one_sync())
                                ptx::tma_store_wait<0>();
                            ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                        }
                        // Q and O have separate SMEM staging. O stores do
                        // not depend on completion of the next Q-full phase.
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
                            (!IS_DECODE || 2 * args.s_q_idx + cta_idx < params.s_q)) {
                            // O is a streaming write; prefer retaining the
                            // repeatedly gathered KV data in L2 instead.
                            const uint64_t desc =
                                reinterpret_cast<uint64_t>(&tma_params.tensor_map_o);
                            const uint32_t src = ptx::cast_smem_ptr_to_uint(o_storage);
                            const uint64_t cache_hint =
                                ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_FIRST>();
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group.L2::cache_hint "
                                "[%0, {%2, %3, %4, %5, %6}], [%1], %7;" ::"l"(desc),
                                "r"(src), "r"(0), "r"(0), "r"(chunk * 2), "r"(0),
                                "r"((IS_DECODE ? args.batch_idx * params.s_q : 0) +
                                    2 * args.s_q_idx + cta_idx),
                                "l"(cache_hint)
                                : "memory");
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
                                if constexpr (!IS_DECODE)
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
                                                     q_token_idx, IS_DECODE ? args.batch_idx : 0);
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
                        if (k == 0 && i == 0) {
                            if constexpr (!IS_DECODE)
                                smem.bar_tQ_full[0].wait(args.outer_loop_phase ^ 1 ^
                                                         is_last_o); // Wait for sQ's availability
                        }
                        ptx::st_shared(sO_accum_addrs[i] +
                                           cur_buf_idx * ((H_Q / 2) * B_EPI_SPLITKV * 2),
                                       *(__int128_t*)(o + i * 4));
                    }

                    ptx::fence_view_async_shared();
                    ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                    if constexpr (IS_DECODE) { // Otherwise nvcc complains about `tma_params`
                                               // doesn't have `tensor_map_o_accum`
                        float* cur_buf_base =
                            (float*)o_storage + cur_buf_idx * ((H_Q / 2) * B_EPI_SPLITKV * 2);
                        int q_token_idx = 2 * args.s_q_idx + cta_idx;
                        bool is_real_q_token = q_token_idx < params.s_q;
                        if (warp_idx == 0 && ptx::elect_one_sync() && is_real_q_token) {
                            ptx::SM90_TMA_STORE_5D::copy(
                                &tma_params.tensor_map_o_accum, cur_buf_base, 0, 0,
                                k * (B_EPI_SPLITKV / 32), q_token_idx, args.n_split_idx);
                            ptx::tma_store_arrive();
                        } else if (warp_idx == 1 && ptx::elect_one_sync() && is_real_q_token) {
                            ptx::SM90_TMA_STORE_5D::copy(&tma_params.tensor_map_o_accum,
                                                         cur_buf_base + (H_Q / 2) * B_EPI_SPLITKV,
                                                         0, 0,
                                                         k * (B_EPI_SPLITKV / 32) + (D_V / 2) / 32,
                                                         q_token_idx, args.n_split_idx);
                            ptx::tma_store_arrive();
                        }
                    }
                }
            }
        };

        OuterloopArgs last_args;
        last_args.batch_idx = -1;

        bool final_outer_loop_phase = run_outer_loop([&](const OuterloopArgs& args) {
            // Both CTAs must finish reading Q and its TMEM scale words
            // before WG0 publishes the next query. Waiting after the TMA
            // load is too late for an SS operand.
            smem.bar_tQ_empty[IS_DECODE ? args.outer_loop_phase : 0].wait(
                (IS_DECODE ? decode_q_phase : args.outer_loop_phase) ^ 1);
            ptx::tcgen05_after_thread_sync();
            if constexpr (IS_DECODE) {
                // The prior loop wrote the output from this Q slot. Both
                // issuing warps must finish its TMA reads before Q overwrites.
                ptx::tma_store_wait<0>();
                ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
            }
            if constexpr (IS_DECODE) {
                if (warp_idx == 1 && ptx::elect_one_sync()) {
                    const int raw_extra_start = (params.topk + 3) & ~3;
                    const bool raw_staged =
                        raw_extra_start + params.extra_topk <=
                            SharedMemoryPlan::RAW_INDEX_CAPACITY &&
                        (params.topk & 3) == 0 && (params.extra_topk & 3) == 0 &&
                        (params.stride_indices_b & 3) == 0 &&
                        (params.stride_indices_s_q & 3) == 0 &&
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
            }
            if constexpr (SPLIT_O_STAGING || IS_DECODE) {
                if (warp_idx == 0 && ptx::elect_one_sync()) {
                    // Decode Q has independent storage; wait for O immediately
                    // before reusing output staging, after Q has been issued.
                    if constexpr (!IS_DECODE)
                        ptx::tma_store_wait<0>();
                    const int q_batch_base = [&]() {
                        if constexpr (IS_DECODE)
                            return args.batch_idx * (params.stride_q_b / params.stride_q_s_q);
                        else
                            return 0;
                    }();
                    ptx::SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
                        &tma_params.tensor_map_q,
                        (uint64_t*)&smem.bar_sQ_full[IS_DECODE ? args.outer_loop_phase : 0],
                        ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_FIRST>(),
                        smem.Q[IS_DECODE ? args.outer_loop_phase : 0].data(), 0, 0, 0, 0,
                        q_batch_base + (IS_DECODE ? min(2 * args.s_q_idx + cta_idx, params.s_q - 1)
                                                  : 2 * args.s_q_idx + cta_idx));
                    if (cta_idx == 0) {
                        smem.bar_sQ_full[IS_DECODE ? args.outer_loop_phase : 0]
                            .arrive_and_expect_tx(H_Q * D_Q * sizeof(fp8_e4m3));
                    }
                }
            }
            {
                // Each lane loads Q heads r and r+32. The replicated 16B
                // slots provide two packed words per head; arrange them in
                // the eight consecutive Q-scale TMEM columns consumed by
                // the QK MMA. Decode adds the batch stride, while prefill
                // has one batch and uses the sequence stride directly.
                const int head_idx = lane_idx;
                const int q_token_idx = 2 * args.s_q_idx + cta_idx;
                const int q_load_token_idx = min(q_token_idx, params.s_q - 1);
                const int q_batch_offset = [&]() {
                    if constexpr (IS_DECODE) {
                        return args.batch_idx * params.stride_q_b;
                    } else {
                        return 0;
                    }
                }();
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
                ptx::tmem_st_32dp32bNx<8>(
                    (tmem_cols::Q_scale + (IS_DECODE ? args.outer_loop_phase * 8 : 0)) +
                        warp_idx * 32 * 65536,
                    q_scale_words);
                ptx::fence_view_async_tmem_store();
                ptx::tcgen05_before_thread_sync();
                ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                if (warp_idx == 0 && ptx::elect_one_sync()) {
                    // The MMA consumer is elected in CTA0, but both CTAs
                    // contribute their Q-scale TMEM fragment. Publish CTA1's
                    // completion into CTA0's barrier explicitly.
                    if (cta_idx == 0) {
                        smem.bar_Q_scale_ready[IS_DECODE ? args.outer_loop_phase : 0].arrive();
                    } else {
                        smem.bar_Q_scale_ready[IS_DECODE ? args.outer_loop_phase : 0].arrive(
                            uint32_t{0}, 1u);
                    }
                }
            }

            if constexpr (SPLIT_O_STAGING || IS_DECODE) {
                if (warp_idx == 0 && cta_idx == 0 && ptx::elect_one_sync()) {
                    smem.bar_sQ_full[IS_DECODE ? args.outer_loop_phase : 0].wait(
                        (IS_DECODE ? decode_q_phase : args.outer_loop_phase));
                    ptx::umma_arrive_multicast_2x1SM_noelect(
                        smem.bar_tQ_full[IS_DECODE ? args.outer_loop_phase : 0], 1 | 2);
                }
            } else {
                // Copy Q for this round
                if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
                    ptx::tma_store_wait<0>();
                    ptx::NamedBarrier::arrive_and_wait(
                        128, barrier_ids::WG0_SYNC); // Since we use two warps to issue TMA during
                                                     // FwdMode::DecodeWithSplitKV
                }
                if (warp_idx == 0 && ptx::elect_one_sync()) {
                    // Wait for sQ to become empty, and issue G -> S copy for Q
                    if constexpr (FWD_MODE != FwdMode::DecodeWithSplitKV) {
                        ptx::tma_store_wait<0>(); // This thread must be the same one as o copy out
                                                  // thread (since `elect_one_sync()` always returns
                                                  // the same thread for the same `mask`, according
                                                  // to PTX document)
                    }
                    int stride_q_b_div_stride_q_s_q = 0;
                    if constexpr (IS_DECODE) {
                        stride_q_b_div_stride_q_s_q = params.stride_q_b / params.stride_q_s_q;
                    }
                    int q_token_idx = 2 * args.s_q_idx + cta_idx;
                    // A 2-SM operation cannot drop the second CTA.  Clamp the odd
                    // tail CTA to the final real token so it receives a valid Q
                    // tile and can participate in every transaction and barrier.
                    int q_load_token_idx = min(q_token_idx, params.s_q - 1);
                    ptx::SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
                        &tma_params.tensor_map_q,
                        (uint64_t*)&smem.bar_sQ_full[IS_DECODE ? args.outer_loop_phase : 0],
                        ptx::create_simple_cache_policy<ptx::CacheHint::EVICT_FIRST>(),
                        smem.Q[IS_DECODE ? args.outer_loop_phase : 0].data(), 0, 0, 0, 0,
                        (IS_DECODE ? args.batch_idx * stride_q_b_div_stride_q_s_q : 0) +
                            q_load_token_idx);

                    // Both CTAs contribute their native E4M3 Q tile. The SS
                    // operand stays in SMEM until the last QK MMA releases it.
                    if (cta_idx == 0) {
                        smem.bar_sQ_full[IS_DECODE ? args.outer_loop_phase : 0]
                            .arrive_and_expect_tx(H_Q * D_Q * sizeof(fp8_e4m3));
                        smem.bar_sQ_full[IS_DECODE ? args.outer_loop_phase : 0].wait(
                            (IS_DECODE ? decode_q_phase : args.outer_loop_phase));
                        // Q remains in SMEM; the SS QK MMA consumes it directly.
                        ptx::umma_arrive_multicast_2x1SM_noelect(
                            smem.bar_tQ_full[IS_DECODE ? args.outer_loop_phase : 0], 1 | 2);
                    }
                }
            }

            if (last_args.batch_idx != -1) {
                if constexpr (IS_DECODE) {
                    ptx::tma_store_wait<0>();
                    ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                }
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
        // KV fetching threads for prefill, dequant threads for decoding
        ptx::warpgroup_reg_dealloc<!IS_DECODE && B_TOPK == 128 ? 128 : 80>();
        ::RingBufferState rs;

        if constexpr (!IS_DECODE) {
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
                                KU_LDG_256(gIndices + k * B_TOPK + row, dest + local_row * 8, ".nc",
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
                                        smem.K[k_buf_idx].data() + row * 128 +
                                            col * 128 * (B_TOPK / 2),
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
                                    tma_gather4_cta1(
                                        &tma_params.tensor_map_kv, smem.bar_V_full[k_buf_idx],
                                        smem.V[k_buf_idx].data() + row * 128 +
                                            local_col * 128 * B_TOPK,
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
                                KU_LDG_256(gIndices + k * B_TOPK + row, cur_indices + local_row * 8,
                                           ".nc", "no_allocate", "evict_first", "256B");
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
                                        smem.K[k_buf_idx].data() + row * 128 +
                                            col * 128 * (B_TOPK / 2),
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
                                    tma_gather4_cta1(
                                        &tma_params.tensor_map_kv, smem.bar_V_full[k_buf_idx],
                                        smem.V[k_buf_idx].data() + row * 128 +
                                            local_col * 128 * B_TOPK,
                                        local_col * 128 + cta_idx * (D_K / 2), indices, cache_hint);
                                }
                            }
                            rs.update();
                        }
                    });
                }
            }

        } else {
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
                                        ptx::create_simple_cache_policy<
                                            ptx::CacheHint::EVICT_LAST>());
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
                                    tma_gather4_cta1(block_idx >= args.num_orig_kv_blocks
                                                         ? &tma_params.tensor_map_extra_kv
                                                         : &tma_params.tensor_map_kv,
                                                     smem.bar_V_full[k_buf_idx],
                                                     smem.V[k_buf_idx].data() + row * 128 +
                                                         col * 128 * B_TOPK,
                                                     col * 128 + cta_idx * (D_K / 2), coords,
                                                     ptx::create_simple_cache_policy<
                                                         ptx::CacheHint::EVICT_LAST>());
                                }
                            }
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        rs.update();
                    }
                });
            }
        }
    } else if (warpgroup_idx == 2) {
        ptx::warpgroup_reg_dealloc<!IS_DECODE && B_TOPK == 128 ? 64 : 80>();
        // Rotate logical roles only within physical warpgroup 2 for decode.
        // Register allocation, lanes, buffers, and synchronization are unchanged.
        const int wg2_physical_warp = warp_idx;
        const int warp_idx = IS_DECODE ? 8 + ((wg2_physical_warp - 8 + 2) & 3) : wg2_physical_warp;

        ::RingBufferState rs;
        // CTA1 issues SV independently; CTA0 issues QK.
        if (warp_idx == 8 && cta_idx == 1 && ptx::elect_one_sync()) {
            run_outer_loop([&](const OuterloopArgs& args) {
                auto issue_SV = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] =
                        rs.offset_by(rs_offset).template get<NUM_K_BUFS>();
                    auto [s_buf_idx, s_bar_phase] = rs.offset_by(rs_offset).template get<2>();
                    {
                        if constexpr (IS_DECODE || B_TOPK != 128)
                            smem.bar_V_full[k_buf_idx].arrive_and_expect_tx(B_TOPK * D_V *
                                                                            sizeof(fp8_e4m3));
                        smem.bar_V_full[k_buf_idx].wait(k_bar_phase);
                    }
                    if constexpr (!IS_DECODE && B_TOPK == 128) {
                        smem.bar_S_O_full[s_buf_idx].wait(s_bar_phase);
                        if (k == args.start_block_idx)
                            smem.bar_tOut_empty.wait(args.outer_loop_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();
                        ptx::issue_sv<B_TOPK, IS_DECODE>(
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
                        ptx::issue_sv<B_TOPK, IS_DECODE>(
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
                smem.bar_tQ_full[IS_DECODE ? args.outer_loop_phase : 0].wait(
                    (IS_DECODE ? decode_q_phase : args.outer_loop_phase));
                smem.bar_Q_scale_ready[IS_DECODE ? args.outer_loop_phase : 0].wait(
                    (IS_DECODE ? decode_q_phase : args.outer_loop_phase));
                ptx::tcgen05_after_thread_sync();

                auto issue_QK = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] =
                        rs.offset_by(rs_offset).template get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).template get<1>();
                    smem.bar_P_empty.wait(bar_phase ^ 1);
                    ptx::transac_bar_t& k_full = smem.bar_K_full[k_buf_idx];
                    if constexpr (IS_DECODE || B_TOPK != 128)
                        k_full.arrive_and_expect_tx(B_TOPK * D_K * sizeof(fp8_e4m3));
                    k_full.wait(k_bar_phase);
                    ptx::tcgen05_after_thread_sync();

                    smem.bar_K_scale_copy_ready[k_buf_idx].wait(k_bar_phase);
                    copy_k_scale_smem_to_tmem(smem.k_scale_mma[k_buf_idx].data(),
                                              tmem_cols::K_scale);
                    ptx::tcgen05_before_thread_sync();
                    ptx::issue_qk<B_TOPK, IS_DECODE>(
                        smem.Q[IS_DECODE ? args.outer_loop_phase : 0].data(),
                        smem.K[k_buf_idx].data(), tmem_cols::P,
                        tmem_cols::Q_scale + (IS_DECODE ? args.outer_loop_phase * 8 : 0),
                        tmem_cols::K_scale);
                    ptx::umma_arrive_multicast_2x1SM_noelect(smem.bar_QK_done[k_buf_idx], 1 | 2);
                };

                NO_UNROLL
                for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                    issue_QK(k, 0);
                    rs.update();
                }
                ptx::tcgen05_before_thread_sync();
                ptx::umma_arrive_multicast_2x1SM_noelect(
                    smem.bar_tQ_empty[IS_DECODE ? args.outer_loop_phase : 0], 1 | 2);
            });
        } else if (warp_idx == 9) {
            // KV validness loading warp (for prefill), Indices transformation warp (for decode,
            // Responsible for generating: TMA coordinates, scale factors, and valid masks)
            if constexpr (!IS_DECODE) {
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
                            smem.bar_valid_coord_scales_empty[indices_buf_idx].wait(
                                indices_bar_phase ^ 1);
                            smem.is_k_valid[indices_buf_idx][lane_idx] = k_validness_mask;
                            smem.bar_valid_coord_scales_full[indices_buf_idx].arrive();

                            rs.update();
                        }
                    });
                }
            } else {
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
                        raw_extra_start + params.extra_topk <=
                            SharedMemoryPlan::RAW_INDEX_CAPACITY &&
                        (params.topk & 3) == 0 && (params.extra_topk & 3) == 0 &&
                        (params.stride_indices_b & 3) == 0 &&
                        (params.stride_indices_s_q & 3) == 0 &&
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
                                    const uint32_t src_addr =
                                        ptx::cast_smem_ptr_to_uint(raw_src + abs_pos);
                                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                                                 : "=r"(my_indices[0]), "=r"(my_indices[1])
                                                 : "r"(src_addr)
                                                 : "memory");
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
                                    const uint32_t src_addr = ptx::cast_smem_ptr_to_uint(
                                        raw_src + raw_extra_start + abs_pos);
                                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                                                 : "=r"(my_indices[0]), "=r"(my_indices[1])
                                                 : "r"(src_addr)
                                                 : "memory");
                                } else {
                                    my_indices[0] = my_indices[1] = -1;
                                }
                            } else {
                                my_indices[0] =
                                    abs_pos < params.extra_topk
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
                                is_token_valid
                                    ? block_idx * cur_tma_coords_step_per_block +
                                          idx_in_block * tma_coords_step_per_token
                                    : -1; // If the token is invalid because it topk position
                                          // exceeds topk_length, we must manually fill tma_coords
                                          // with -1 to avoid copying-in NaN.
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
                                const auto* ptr =
                                    reinterpret_cast<const uint8_t*>(
                                        IS_EXTRA_BLOCK ? params.extra_kv : params.kv) +
                                    scale_offsets[i];
                                asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr));
                            }
                        }
                        ptx::st_shared(smem.index_storage.k_scale_offset[index_buf_idx] +
                                           lane_idx * 2,
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
                         block_idx < min(args.num_orig_kv_blocks, args.end_block_idx);
                         ++block_idx) {
                        process_one_block(block_idx, IsOrigBlock{});
                    }

                    NO_UNROLL
                    for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks);
                         block_idx < args.end_block_idx; ++block_idx) {
                        process_one_block(block_idx, IsExtraBlock{});
                    }
                });
            }
        } else if (warp_idx == 10 || warp_idx == 11) {
            if constexpr (!IS_DECODE) {
                // Each scale warp handles B_TOPK/2 tokens, in one or two
                // groups of 32. Scale words are packed for the duplicated
                // DP layout; each UE8M0 byte serves two K=32 MMAs.
                const int scale_warp = warp_idx - 10;
                ::RingBufferState scale_rs;
                run_outer_loop([&](const OuterloopArgs& args) {
                    NO_UNROLL
                    for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                        auto [k_buf_idx, k_bar_phase] = scale_rs.template get<NUM_K_BUFS>();
                        int tile_token[2];
                        uint2 tile_scale[2];
                        if constexpr (B_TOPK == 128) {
                            const int* tile_indices = params.indices +
                                                      args.s_q_idx * params.stride_indices_s_q +
                                                      k * B_TOPK;
                            const uint8_t* scale_base =
                                reinterpret_cast<const uint8_t*>(params.kv) +
                                static_cast<int64_t>(params.s_kv) * params.h_kv * D_K;
                            UNROLL
                            for (int part = 0; part < 2; ++part) {
                                const int row = scale_warp * 64 + part * 32 + lane_idx;
                                if (k + 1 < args.end_block_idx &&
                                    (scale_warp == 0 || (k + 1) * B_TOPK + 64 < params.topk)) {
                                    asm volatile("prefetch.global.L1 [%0];" ::"l"(tile_indices +
                                                                                  B_TOPK + row));
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
                            const int* g_indices = params.indices +
                                                   args.s_q_idx * params.stride_indices_s_q +
                                                   k * B_TOPK;
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
                                const uint32_t valid_bits =
                                    __ballot_sync(0xffffffffu, token_valid && k * B_TOPK + row <
                                                                                  args.topk_length);
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
                                        make_uint4(tile_scale[0].x, tile_scale[1].x,
                                                   tile_scale[0].y, tile_scale[1].y);
                                    ptx::st_shared(
                                        native_scale,
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
            } else {
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
                        auto [index_buf_idx, index_bar_phase] =
                            scale_rs.template get<NUM_INDEX_BUFS>();
                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        const int row = scale_warp * (B_TOPK / 2) + lane_idx;
                        const bool token_valid =
                            static_cast<uint8_t>(smem.is_k_valid[index_buf_idx][row / 8]) &
                            (uint8_t(1) << (row % 8));
                        const bool is_extra = block_idx >= args.num_orig_kv_blocks;
                        const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(
                            is_extra ? params.extra_kv : params.kv);
                        const uint32_t* src = reinterpret_cast<const uint32_t*>(
                            scale_base +
                            (token_valid ? smem.index_storage.k_scale_offset[index_buf_idx][row]
                                         : 0));

                        const uint2 packed = token_valid
                                                 ? __ldg(reinterpret_cast<const uint2*>(src))
                                                 : make_uint2(0, 0);
                        // Load into registers before waiting to overwrite the
                        // stage. Softmax releases it only after QK and V reads.
                        smem.bar_v_scale_empty[k_buf_idx].wait(k_bar_phase ^ 1);
                        uint32_t* native_scale = reinterpret_cast<uint32_t*>(
                            smem.k_scale_mma[k_buf_idx].data() + row * 16);
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
                using ValidMask = std::conditional_t<B_TOPK == 128, uint64_t, uint32_t>;
                ValidMask valid;
                if constexpr (FUSED_VALIDITY) {
                    const uint32_t mask_addr = ptx::cast_smem_ptr_to_uint(
                        reinterpret_cast<ValidMask*>(smem.is_k_valid[indices_buf_idx]) +
                        local_warp_idx / 2);
                    // Complete the mask read before allowing the producer
                    // to reuse its shared slot, even while QK is pending.
                    asm volatile("ld.acquire.cta.shared.b64 %0, [%1];"
                                 : "=l"(valid)
                                 : "r"(mask_addr)
                                 : "memory");
                    smem.bar_valid_coord_scales_empty[indices_buf_idx].arrive();
                }
                smem.bar_QK_done[k_buf_idx].wait(k_bar_phase);
                ptx::tcgen05_after_thread_sync();
                {
                    if constexpr (IS_DECODE) {
                        // Same D256 addition as the old path, with both
                        // partials in this thread's TMEM lane. No peer exchange.
                        float p_peer[NUM_ELEMS_PER_THREAD];
                        ptx::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P, p);
                        ptx::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P + B_TOPK / 2,
                                                                     p_peer);
                        ptx::fence_view_async_tmem_load();
                        UNROLL
                        for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 2)
                            *reinterpret_cast<float2*>(p + i) =
                                ptx::float2_add(*reinterpret_cast<float2*>(p + i),
                                                *reinterpret_cast<float2*>(p_peer + i));
                    } else {
                        // Each warp reads its native QK logits without a partial-D
                        // exchange: 32 values for N64 and 64 for N128.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1030 && __CUDA_ARCH__ < 1100
                        if constexpr (B_TOPK == 128) {
                            float max0, max1;
                            load_native_p_and_max(tmem_cols::P, *reinterpret_cast<float (*)[32]>(p),
                                                  max0);
                            load_native_p_and_max(tmem_cols::P + 32,
                                                  *reinterpret_cast<float (*)[32]>(p + 32), max1);
                            p_max = max(max0, max1);
                        } else {
                            load_native_p_and_max(tmem_cols::P, p, p_max);
                        }
#else
                        ptx::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P, p);
#endif
                    }
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
                    if (IS_DECODE || valid != static_cast<ValidMask>(-1))
                        p_max = get_max<NUM_ELEMS_PER_THREAD>(p);
#else
                    p_max = get_max<NUM_ELEMS_PER_THREAD>(p);
#endif
                }

                if constexpr (!IS_DECODE && B_TOPK == 128) {
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
                        ptx::st_shared(ptx::k_major_ptr<64, 64>(smem.S[s_buf_idx].data(), s_row,
                                                                s_col_base + i * 16),
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
                        fp8_e4m3* sS_base = ptx::k_major_ptr<64, IS_DECODE ? 0 : 64>(
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

            if constexpr (IS_DECODE) {
                // Publish both partial sums directly. The 128-writer full barrier
                // replaces the softmax WG's local reduction barrier.
                smem.bar_li_empty.wait(args.outer_loop_phase ^ 1);
                smem.rowwise_li_buf[idx_in_warpgroup] = li;
                if (idx_in_warpgroup < H_Q / 2)
                    smem.final_mi[idx_in_warpgroup] = mi;
                smem.bar_li_full.arrive();
            } else {
                // Reduce li
                smem.bar_li_empty.wait(args.outer_loop_phase ^ 1);
                smem.rowwise_li_buf[idx_in_warpgroup ^ 64] = li;
                ptx::NamedBarrier::arrive_and_wait(128, barrier_ids::WG2_SYNC);
                li += smem.rowwise_li_buf[idx_in_warpgroup];

                if (idx_in_warpgroup < H_Q / 2) {
                    if constexpr (IS_DECODE) {
                        // The existing full/empty protocol protects both values until WG0 reads
                        // them.
                        smem.rowwise_li_buf[idx_in_warpgroup] = li;
                        smem.final_mi[idx_in_warpgroup] = mi;
                        smem.bar_li_full.arrive();
                    } else {
                        // Calculate output_scale and save
                        int head_idx = idx_in_warpgroup;
                        float attn_sink = params.attn_sink == nullptr
                                              ? -CUDART_INF_F
                                              : __ldg(params.attn_sink + head_idx);
                        float output_scale;
                        if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                            output_scale =
                                __fdividef(1.0f, li + exp2f(fmaf(attn_sink, CUDART_L2E_F, -mi)));
                        } else {
                            output_scale = __fdividef(1.0f, li);
                        }
                        smem.rowwise_li_buf[idx_in_warpgroup] = li == 0.0f ? 0.0f : output_scale;
                        smem.bar_li_full.arrive();
                    }
                    const int head_idx = idx_in_warpgroup;

                    if constexpr (!IS_DECODE) {
                        float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                        cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                        if constexpr (!IS_DECODE) {
                            int q_token_idx = 2 * args.s_q_idx + cta_idx;
                            int global_index = q_token_idx * params.h_q + head_idx;
                            params.max_logits[global_index] = real_mi * CUDART_LN2_F;
                            params.lse[global_index] = cur_lse;
                        } else {
                            int q_token_idx = 2 * args.s_q_idx + cta_idx;
                            if (q_token_idx < params.s_q) {
                                if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                                    params.lse[args.batch_idx * params.stride_lse_b +
                                               q_token_idx * params.stride_lse_s_q + head_idx] =
                                        cur_lse;
                                } else {
                                    float cur_lse_2base = log2f(li) + mi;
                                    params.lse_accum[args.n_split_idx *
                                                         params.stride_lse_accum_split +
                                                     q_token_idx * params.stride_lse_accum_s_q +
                                                     head_idx] = cur_lse_2base;
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    ptx::barrier_cluster_arrive_relaxed();
    ptx::barrier_cluster_wait_acquire();

#else
    asm("trap;");
#endif
}

// We have two launchers with different kernel names to distinguish prefill and decode

template <typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
    dual_mxfp8_sparse_attn_fwd_kernel(
        __grid_constant__ const typename Kernel::ArgT params,
        __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template <typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
    flash_fwd_splitkv_mla_fp8_sparse_kernel(
        __grid_constant__ const typename Kernel::ArgT params,
        __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template <FwdMode FWD_MODE, int D_QK, int PREFILL_B_TOPK>
void KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>::run(const ArgT& params) {
    static_assert(D_QK == 576 || D_QK == 512);

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % 64 == 0);
    KU_ASSERT(params.h_q == TOKEN_H_Q);
    KU_ASSERT(params.s_q > 0);
    if constexpr (!IS_DECODE) {
        KU_ASSERT(params.s_q % 2 == 0);
    }
    KU_ASSERT(params.d_qk == D_QK);

    static_assert(D_Q == 512);
    KU_ASSERT(params.stride_q_h_q == Q_BYTES_PER_HEAD,
              "Q head stride must be 512 e4m3 bytes + an aligned 16-byte scale slot");
    if constexpr (!IS_DECODE) {
        KU_ASSERT(params.stride_kv_h_kv == KV_BYTES_PER_TOKEN,
                  "KV token envelope must be 512 e4m3 bytes + a replicated 16-byte scale slot");
        KU_ASSERT(params.stride_kv_s_kv == params.h_kv * KV_BYTES_PER_TOKEN,
                  "packed KV storage must be contiguous across logical token envelopes");
        KU_ASSERT(reinterpret_cast<int64_t>(params.q) % 16 == 0);
        KU_ASSERT(reinterpret_cast<int64_t>(params.kv) % 16 == 0);
    } else {
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
    }

    CUtensorMap tensor_map_q;
    if constexpr (IS_DECODE) {
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
    } else {
        tensor_map_q = ptx::make_tensor_map(
            {128ul, TOKEN_H_Q, 2ul, (D_Q / 128ul) / 2ul, (unsigned long)params.s_q},
            ptx::make_stride_helper<int>({params.stride_q_h_q, 128, D_Q / 2, params.stride_q_s_q},
                                         sizeof(fp8_e4m3)),
            {128, TOKEN_H_Q, 2, (D_Q / 128) / 2, 1}, params.q, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B); // Native QK's logical layout applies the packed-Q
                                                 // K32 permutation.
    }

    CUtensorMap tensor_map_kv;
    [[maybe_unused]] CUtensorMap tensor_map_k_scale = {};
    CUtensorMap tensor_map_extra_kv = {};
    if constexpr (IS_DECODE) {
        auto get_kv_tensormap = [&](bool is_extra, void* k_ptr, int num_blocks,
                                    int64_t stride_kv_block, int64_t stride_kv_row) -> CUtensorMap {
            KU_ASSERT(
                (int64_t)k_ptr % 16 == 0,
                "The base address of %sk_ptr (%p) must be 16B aligned for sparse fp8 attention on sm100f",
                is_extra ? "extra_" : "", k_ptr);
            KU_ASSERT(
                stride_kv_block % TMA_K_STRIDE_FOR_DECODING == 0,
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
        tensor_map_kv = get_kv_tensormap(false, params.kv, params.num_blocks,
                                         params.stride_kv_block, params.stride_kv_row);
        if (params.extra_topk > 0)
            tensor_map_extra_kv =
                get_kv_tensormap(true, params.extra_kv, params.extra_num_blocks,
                                 params.stride_extra_kv_block, params.stride_extra_kv_row);
    } else {
        // The storage tensor is exposed as 528-byte logical envelopes, but
        // inside the packed page all 512-byte E4M3 rows are contiguous and
        // the replicated 16-byte scale slots form a separate tail plane.
        tensor_map_kv =
            ptx::make_tensor_map({D_QK, (unsigned long)params.s_kv}, {(unsigned long)D_K}, {128, 1},
                                 params.kv, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                 CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        // The scale tail contains one replicated 16B UE8M0 slot per K token.
    }

    CUtensorMap tensor_map_o;
    if constexpr (IS_DECODE && SPLIT_O_STAGING) {
        tensor_map_o = ptx::make_tensor_map(
            {64, TOKEN_H_Q, D_V / 128, 2ul, (unsigned long)params.b * params.s_q},
            ptx::make_stride_helper<int>({params.stride_o_h_q, 64, D_V / 2, params.stride_o_s_q},
                                         sizeof(ptx::bf16)),
            {64, TOKEN_H_Q, O_STAGE_D / 128, 2, 1}, params.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    } else if constexpr (IS_DECODE) {
        tensor_map_o = ptx::make_tensor_map(
            {64, TOKEN_H_Q, D_V / 64, (unsigned long)params.s_q, (unsigned long)params.b},
            ptx::make_stride_helper<int>(
                {params.stride_o_h_q, 64, params.stride_o_s_q, params.stride_o_b},
                sizeof(ptx::bf16)),
            {64, TOKEN_H_Q, D_V / 64, 1, 1}, params.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    } else if constexpr (SPLIT_O_STAGING) {
        // Pair output segments separated by D_V/2. Each store covers 64 or
        // 128 dimensions per side, according to the selected staging capacity.
        tensor_map_o = ptx::make_tensor_map(
            {64, TOKEN_H_Q, D_V / 128, 2ul, (unsigned long)params.s_q},
            ptx::make_stride_helper<int>({D_V, 64, D_V / 2, TOKEN_H_Q * D_V}, sizeof(ptx::bf16)),
            {64, TOKEN_H_Q, O_STAGE_D / 128, 2, 1}, params.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    } else {
        tensor_map_o = ptx::make_tensor_map(
            {64, TOKEN_H_Q, D_V / 64, (unsigned long)params.s_q, 1ul},
            ptx::make_stride_helper<int>({D_V, 64, TOKEN_H_Q * D_V, TOKEN_H_Q * D_V},
                                         sizeof(ptx::bf16)),
            {64, TOKEN_H_Q, D_V / 64, 1, 1}, params.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    }

    CUtensorMap tensor_map_o_accum = {};
    if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
        tensor_map_o_accum = ptx::make_tensor_map(
            {32, TOKEN_H_Q, D_V / 32, (unsigned long)params.s_q,
             (unsigned long)params.num_sm_parts + params.b},
            ptx::make_stride_helper<int>({params.stride_o_accum_h_q, 32, params.stride_o_accum_s_q,
                                          params.stride_o_accum_split},
                                         sizeof(float)),
            {32, TOKEN_H_Q, B_EPI_SPLITKV / 32, 1, 1}, params.o_accum,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    }

    TmaParams tma_params;
    if constexpr (IS_DECODE) {
        tma_params = {tensor_map_q, tensor_map_o, tensor_map_o_accum, tensor_map_kv,
                      tensor_map_extra_kv};
    } else {
        tma_params = {tensor_map_q, tensor_map_kv, tensor_map_k_scale, tensor_map_o};
    }

    auto kernel =
        !IS_DECODE
            ? &dual_mxfp8_sparse_attn_fwd_kernel<KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>>
            : &flash_fwd_splitkv_mla_fp8_sparse_kernel<
                  KernelTemplate<FWD_MODE, D_QK, PREFILL_B_TOPK>>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_shape;
    if constexpr (IS_DECODE) {
        grid_shape = dim3(2 * ptx::ceil_div(params.s_q, 2), params.num_sm_parts, 1);
    } else {
        // grid_shape counts CTAs, while each logical query pair is one
        // two-CTA cluster.
        grid_shape = dim3(params.s_q, 1, 1);
    }

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t config{grid_shape, dim3(NUM_THREADS, 1, 1), smem_size, params.stream, &attr,
                              1};
    KU_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params, tma_params));
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

} // namespace sm100::dual_mxfp8::head64
