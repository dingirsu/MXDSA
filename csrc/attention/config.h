#pragma once
#include "attention/phase1.h"
#include "ptx/ptx.cuh"
#include "attention/layout.cuh"

#include <math_constants.h>

#include "common/params.h"

namespace sm100::dual_mxfp8::head64 {

using namespace ptx;

template<SparseAttnFwdMode FWD_MODE, int D_QK, int PREFILL_B_TOPK = 64>
struct KernelTemplate {

static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;
using ArgT = std::conditional_t<
    IS_DECODE,
    SparseAttnDualMxfp8DecodeParams,
    MxFp8SparseAttnFwdParams
>;
using fp8_e4m3 = ptx::fp8_e4m3;
using fp8_e8m0 = ptx::fp8_e8m0;

struct TmaParamsForPrefill {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_k_scale;
    CUtensorMap tensor_map_o;
};

struct TmaParamsForDecode {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_o;
    CUtensorMap tensor_map_o_accum;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_extra_kv;
};

using TmaParams = std::conditional_t<
    IS_DECODE,
    TmaParamsForDecode,
    TmaParamsForPrefill
>;

static_assert(D_QK == 512);

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;

// Avoid -inf - (-inf) when the first tile contains no valid logits.
static constexpr float MAX_INIT_VAL = -1e30;

// The 2-SM MMA still has 128 logical rows, but each CTA contributes all 64
// heads from a different query token in the adjacent-token pair.
static constexpr int TOKEN_H_Q = 64;
static constexpr int H_Q = 2*TOKEN_H_Q;
static constexpr int B_TOPK = !IS_DECODE ? PREFILL_B_TOPK : 64;
static_assert(B_TOPK == 64 || B_TOPK == 128);
static constexpr int NUM_THREADS = 128*4;
// The long path generates validity bytes from the scale indices, so
// warp9 no longer contributes eight CLC consumers in each CTA.
// QK and SV each contribute one elected MMA issuer across the CTA pair.
static constexpr bool FUSED_VALIDITY = !IS_DECODE && B_TOPK == 128;
static constexpr int NUM_WORKER_THREADS = !IS_DECODE
    ? (128 + 4 + (FUSED_VALIDITY ? 0 : B_TOPK/8) + 64 + 128)*2 + 2
    : (128 + 128 + 1 + 32 + 2 + 128)*2;

// Prefill uses two stages for N128 and three for N64; decode uses four.
static constexpr int NUM_K_BUFS = IS_DECODE ? 4 : (B_TOPK == 128 ? 2 : 3);
static constexpr bool PREFETCH_INDICES = !IS_DECODE && B_TOPK == 128;
// N128 uses two 32KiB output stores.
static constexpr bool SPLIT_O_STAGING = IS_DECODE || B_TOPK == 128;
static constexpr int O_STAGE_D = SPLIT_O_STAGING ? D_V/2 : D_V;
static constexpr int NUM_INDEX_BUFS = IS_DECODE ? 6 : 4;  // Keep metadata ahead of four K/V stages.

static constexpr int TMA_K_STRIDE_FOR_DECODING = D_QK;
static constexpr int Q_QUANT_GROUP_SIZE = 64;
static constexpr int Q_SCALE_BYTES = D_Q / Q_QUANT_GROUP_SIZE;
static constexpr int Q_SCALE_SLOT_BYTES = 16;
static constexpr int K_QUANT_GROUP_SIZE = 64;
static constexpr int K_SCALE_BYTES = D_K / K_QUANT_GROUP_SIZE;
static constexpr int K_SCALE_SLOT_BYTES = 16;
static constexpr int Q_BYTES_PER_HEAD = D_Q + Q_SCALE_SLOT_BYTES;
static constexpr int KV_BYTES_PER_TOKEN = D_K + K_SCALE_SLOT_BYTES;
static_assert(Q_SCALE_BYTES == 8);
static_assert(Q_SCALE_SLOT_BYTES == 16);
static_assert(K_SCALE_BYTES == 8);
static_assert(K_SCALE_SLOT_BYTES == 2 * K_SCALE_BYTES);

static constexpr int B_EPI = 64;
static constexpr int B_EPI_SPLITKV = 32;
static constexpr int NUM_EPI_SPLITKV_BUFS = IS_DECODE ? 2 : 4;
static_assert(
    (H_Q/2)*D_Q*sizeof(bf16)
        >= NUM_EPI_SPLITKV_BUFS*(H_Q/2)*(B_EPI_SPLITKV*2)*sizeof(float)
);

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: Output accumulator
    // 256 ~ 320: Native P (N64 prefill: 32 cols; decode: two D256 partials)
    // 384 ~ 392: Double-buffered S scale factors
    // 448 ~ 512: Q/K/V scale factors
    static constexpr int O = 0;
    static constexpr int P = 256;
    static constexpr int S_scale = 384;
    static constexpr int S_scale_stride = 4;
    static constexpr int Q_scale = 448;
    static constexpr int K_scale = 480;
    static constexpr int V_scale = 500;
    static constexpr int V_scale_n_stride = 4;
};

struct LegacyIndexStorage {
    alignas(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    alignas(16) int64_t k_scale_offset[NUM_INDEX_BUFS][B_TOPK];
};

struct SharedMemoryPlanBase {
    // SS QK keeps Q live until the last QK MMA completes. The previous
    // query's epilogue must not overwrite it while the next QK is running.
    array_aligned<fp8_e4m3, (H_Q/2)*D_Q> Q[IS_DECODE ? 2 : 1];
    array_aligned<bf16, IS_DECODE ? 512 : (H_Q/2)*O_STAGE_D> O;  // Decode uses its completed Q slot; retain alignment.
    // SV operand: all B_TOPK tokens, half the value dimensions per CTA.
    // CTA0 holds V[:, 0:256]; CTA1 holds V[:, 256:512].
    array_aligned<fp8_e4m3, B_TOPK*(D_K/2)> V[NUM_K_BUFS];
    // QK operand: half the KV tokens, all D_K dimensions per CTA.
    // This stores K input tiles, not QK scores (which live in TMEM P).
    array_aligned<fp8_e4m3, (B_TOPK/2)*D_K> K[NUM_K_BUFS];
    transac_bar_t bar_K_full[NUM_K_BUFS];
    // Each slot holds one complete B_TOPK tile, alternating across PV steps.
    array_aligned<fp8_e4m3, (H_Q/2)*B_TOPK> S[2];
    alignas(16) float v_token_scale[NUM_K_BUFS][B_TOPK];
    // One S scale per softmax row and K=32 atom. An N128 tile packs
    // four scale bytes into one word and consumes four consecutive atoms.
    alignas(16) uint8_t s_scale_exp[NUM_K_BUFS][H_Q/2][B_TOPK/32];
    // Final 64-row x 16B post-transpose source consumed by 2x64 UTCCP.
    array_aligned<fp8_e8m0, B_TOPK * 16> k_scale_mma[NUM_K_BUFS];
    float reserved_p_exchange[4];  // Preserve existing prefill SMEM offsets.
    float rowwise_max_buf[128], rowwise_li_buf[128];

    alignas(16) char is_k_valid[NUM_INDEX_BUFS][B_TOPK/8];
    LegacyIndexStorage index_storage;

    transac_bar_t bar_sQ_full[IS_DECODE ? 2 : 1];
    transac_bar_t bar_Q_scale_ready[IS_DECODE ? 2 : 1];
    transac_bar_t bar_tQ_empty[IS_DECODE ? 2 : 1], bar_tQ_full[IS_DECODE ? 2 : 1];
    transac_bar_t bar_tOut_full, bar_tOut_empty;
    transac_bar_t bar_V_full[NUM_K_BUFS], bar_V_empty[NUM_K_BUFS];
    transac_bar_t bar_K_scale_copy_ready[NUM_K_BUFS];
    transac_bar_t bar_v_scale_full[NUM_K_BUFS], bar_v_scale_empty[NUM_K_BUFS];
    transac_bar_t bar_P_empty;
    transac_bar_t bar_QK_done[NUM_K_BUFS];
    transac_bar_t bar_S_empty[2], bar_S_O_full[2];
    transac_bar_t bar_li_full, bar_li_empty;

    transac_bar_t bar_clc_full, bar_clc_empty;
    transac_bar_t bar_valid_coord_scales_full[NUM_INDEX_BUFS];
    transac_bar_t bar_valid_coord_scales_empty[NUM_INDEX_BUFS];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

// Raw indices share the double-Q lifetime. Their full barriers are local;
// the existing Q-empty release follows all index consumers for that query.
struct SharedMemoryPlanDecode : SharedMemoryPlanBase {
    static constexpr int RAW_INDEX_CAPACITY = 1024;
    array_aligned<int, RAW_INDEX_CAPACITY, 16> raw_indices[2];
    transac_bar_t bar_raw_indices_full[2];
    alignas(16) float final_mi[H_Q/2];
};
using SharedMemoryPlan = std::conditional_t<IS_DECODE, SharedMemoryPlanDecode, SharedMemoryPlanBase>;

struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG2_SYNC = 1;
    static constexpr int WG2_WARP02_SYNC = 2;
    static constexpr int WG2_WARP13_SYNC = 3;
    static constexpr int S_SCALE_SYNC = 4;
};

static __device__ void
sparse_attn_fwd_kernel_devfunc(const ArgT &params, const TmaParams &tma_params);

static void run(const ArgT& params);

};

}
